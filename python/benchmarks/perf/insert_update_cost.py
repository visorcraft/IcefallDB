#!/usr/bin/env python3
"""Pin the baseline cost of open / single-row INSERT / single-record UPDATE,
**with and without** a unique ``id`` index, across table sizes.

This isolates the secondary index's contribution to both open cost (it is loaded
wholesale on every open) and per-commit cost (it is re-serialized on every
mutation). The ``with_index`` minus ``without_index`` delta at each size is the
index tax that Phase 1 is meant to remove.

Measurements (all on a single-fragment table; medians over N fresh runs):
  * **open** — in-process PyO3 ``IcefallDBConnection`` construction, timed in a
    fresh child process so the engine's global ``MetaCache`` starts cold.
  * **insert** — one-row INSERT via the CLI end-to-end (``icefalldb insert``),
    so it includes process startup + open + commit.
  * **update (in-process)** — point ``UPDATE`` via PyO3 ``mutate()`` with the
    open excluded from the timed region.
  * **update (CLI e2e)** — point ``UPDATE`` via ``icefalldb query`` end-to-end.

Every mutating sample runs against a **fresh clone** of the base table so each
repeat starts from identical on-disk state. Reported as median / min / max.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from benchmarks.mutations.generate import make_table  # noqa: E402
from icefalldb.producer import write_icefalldb_ready_parquet  # noqa: E402

# Sizes for the benchmark. 10M is heavy (large clones) — keep it opt-in
# via --scales; the committed baseline covers 100 and 1M.
DEFAULT_SCALES = [100, 1_000_000]
REPEATS = 9
TABLE_NAME = "bench_data"


def _stats(samples: list[float]) -> dict[str, float]:
    return {"median": median(samples), "min": min(samples), "max": max(samples)}


def _icefalldb_cli() -> Path:
    env_cli = os.environ.get("ICEFALLDB_CLI")
    if env_cli:
        path = Path(env_cli)
        if path.exists():
            return path
        raise RuntimeError(f"ICEFALLDB_CLI set but not found: {env_cli}")
    for candidate in (
        REPO_ROOT / "target" / "release" / "icefalldb",
        REPO_ROOT / "target" / "debug" / "icefalldb",
    ):
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate
    raise RuntimeError(
        "icefalldb CLI not found. Build with: cargo build --release -p icefalldb-cli"
    )


def _run_cli(args: list[str]) -> None:
    cli = _icefalldb_cli()
    try:
        subprocess.run(
            [str(cli)] + args, cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"icefalldb {' '.join(args)} failed: {exc.stderr or ''}"
        ) from exc


def _schema_single_fragment(total_rows: int) -> dict:
    # row_group_target_rows >= total_rows => one fragment in one commit.
    return {
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": True, "field_id": 1},
            {"name": "value", "type": "int64", "nullable": True, "field_id": 2},
            {"name": "category", "type": "utf8", "nullable": True, "field_id": 3},
        ],
        "sort": ["id"],
        "row_group_target_rows": max(1, total_rows),
        "row_group_target_bytes": 1 << 40,
        "dropped_columns": [],
        "max_field_id": 3,
    }


def _write_tsv(path: Path, data: pa.Table) -> None:
    ids = data.column("id").to_pylist()
    values = data.column("value").to_pylist()
    categories = data.column("category").to_pylist()
    with path.open("w", encoding="utf-8") as f:
        f.write("id\tvalue\tcategory\n")
        for i, v, c in zip(ids, values, categories):
            f.write(f"{i}\t{v}\t{c}\n")


def _build_base_table(db: Path, n_rows: int, indexed: bool) -> None:
    """Build a fresh single-fragment table; optionally add a unique id index."""
    if db.exists():
        shutil.rmtree(db)
    db.mkdir(parents=True)

    data = make_table(n_rows)
    schema_path = db / f"{TABLE_NAME}_schema.json"
    schema_path.write_text(json.dumps(_schema_single_fragment(n_rows), indent=2))
    tsv_path = db / f"{TABLE_NAME}.tsv"
    _write_tsv(tsv_path, data)
    try:
        _run_cli(["create", "--schema", str(schema_path), str(db), TABLE_NAME])
        _run_cli(["import", str(db), TABLE_NAME, str(tsv_path)])
        if indexed:
            _run_cli(["create-index", str(db), TABLE_NAME, "id", "--unique"])
    finally:
        schema_path.unlink(missing_ok=True)
        tsv_path.unlink(missing_ok=True)


def _copy_base(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# Cold-process probes. A fresh child => empty MetaCache => honest cold open.
_PROBE = """
import sys, time
sys.path.insert(0, sys.argv[1])
import icefalldb_query_py
mode, db, table = sys.argv[2], sys.argv[3], sys.argv[4]
if mode == "open":
    t0 = time.perf_counter()
    conn = icefalldb_query_py.IcefallDBConnection(db, [table])
    _ = conn.table_count()
    print((time.perf_counter() - t0) * 1000.0)
else:  # update: open is excluded from the timed region
    target_id = sys.argv[5]
    conn = icefalldb_query_py.IcefallDBConnection(db, [table])
    _ = conn.table_count()
    sql = f'UPDATE "{table}" SET value = 424242 WHERE id = {target_id}'
    t0 = time.perf_counter()
    affected = conn.mutate(sql)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert affected == 1, f"expected 1 row affected, got {affected}"
    print(elapsed)
"""


def _run_probe(mode: str, db: Path, target_id: int | None = None) -> float:
    args = [
        sys.executable,
        "-c",
        _PROBE,
        str(REPO_ROOT / "python"),
        mode,
        str(db),
        TABLE_NAME,
    ]
    if target_id is not None:
        args.append(str(target_id))
    result = subprocess.run(
        args, capture_output=True, text=True, check=False, cwd=REPO_ROOT
    )
    if result.returncode != 0:
        raise RuntimeError(f"probe ({mode}) failed: {result.stderr or result.stdout}")
    return float(result.stdout.strip())


def _time_insert_cli(db: Path, next_id: int) -> float:
    """Time one-row INSERT via the CLI end-to-end (startup + open + commit)."""
    row = pa.table(
        {
            "id": pa.array([next_id], type=pa.int64()),
            "value": pa.array([0], type=pa.int64()),
            "category": pa.array(["cat_0"], type=pa.string()),
        }
    )
    frag = db / "_one_row.parquet"
    write_icefalldb_ready_parquet(frag, row, row_group_size=1)
    cli = _icefalldb_cli()
    try:
        t0 = time.perf_counter()
        subprocess.run(
            [str(cli), "insert", str(db), TABLE_NAME, str(frag)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return (time.perf_counter() - t0) * 1000.0
    finally:
        frag.unlink(missing_ok=True)


def _time_update_cli(db: Path, target_id: int) -> float:
    """Time one point UPDATE via `icefalldb query` end-to-end."""
    cli = _icefalldb_cli()
    sql = f'UPDATE "{TABLE_NAME}" SET value = 424242 WHERE id = {target_id}'
    t0 = time.perf_counter()
    subprocess.run(
        [str(cli), "query", str(db / TABLE_NAME), sql],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return (time.perf_counter() - t0) * 1000.0


def _measure(base: Path, tmp: Path, n_rows: int, repeats: int) -> dict:
    target_id = n_rows // 2
    open_s, insert_s, upd_proc_s, upd_cli_s = [], [], [], []
    for i in range(repeats):
        # open: cold child, no clone needed (open does not mutate).
        open_s.append(_run_probe("open", base))

        clone = tmp / f"insert_{i:03d}"
        _copy_base(base, clone)
        insert_s.append(_time_insert_cli(clone, n_rows + i))

        clone = tmp / f"updproc_{i:03d}"
        _copy_base(base, clone)
        upd_proc_s.append(_run_probe("update", clone, target_id))

        clone = tmp / f"updcli_{i:03d}"
        _copy_base(base, clone)
        upd_cli_s.append(_time_update_cli(clone, target_id))
        shutil.rmtree(tmp, ignore_errors=True)
    return {
        "rows": n_rows,
        "open_ms": _stats(open_s),
        "insert_cli_ms": _stats(insert_s),
        "update_inproc_ms": _stats(upd_proc_s),
        "update_cli_ms": _stats(upd_cli_s),
    }


def run_benchmark(out_dir: Path, scales: list[int], repeats: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "_base"
    tmp = out_dir / "_tmp"
    results: dict[str, dict] = {}
    for n_rows in scales:
        results[str(n_rows)] = {}
        for indexed in (True, False):
            key = "with_index" if indexed else "without_index"
            print(f"\n=== rows={n_rows:,} {key} ===")
            _build_base_table(base, n_rows, indexed)
            r = _measure(base, tmp, n_rows, repeats)
            results[str(n_rows)][key] = r
            print(
                f"  open={r['open_ms']['median']:.1f}ms "
                f"insert={r['insert_cli_ms']['median']:.1f}ms "
                f"upd_proc={r['update_inproc_ms']['median']:.1f}ms "
                f"upd_cli={r['update_cli_ms']['median']:.1f}ms"
            )
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return results


def _markdown(results: dict, scales: list[int]) -> str:
    lines = [
        "| rows | index | open (ms) | INSERT 1 e2e (ms) | UPDATE 1 in-proc (ms) | UPDATE 1 e2e (ms) |",
        "|-----:|:------|----------:|------------------:|----------------------:|------------------:|",
    ]

    def cell(s: dict) -> str:
        return f"{s['median']:.1f} ({s['min']:.1f}–{s['max']:.1f})"

    for n_rows in scales:
        for key, label in (("with_index", "unique id"), ("without_index", "none")):
            r = results[str(n_rows)][key]
            lines.append(
                f"| {n_rows:,} | {label} | {cell(r['open_ms'])} | "
                f"{cell(r['insert_cli_ms'])} | {cell(r['update_inproc_ms'])} | "
                f"{cell(r['update_cli_ms'])} |"
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scales",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_SCALES,
        help="Comma-separated row counts (default: 100,1000000; add 10000000 for the full matrix)",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "target" / "tmp" / "perf_insert_update_cost",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = run_benchmark(args.out_dir, args.scales, args.repeats)

    json_path = args.json_out or (args.out_dir / "insert_update_cost.json")
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote JSON: {json_path}")
    print("\nMarkdown:\n")
    print(_markdown(results, args.scales))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
