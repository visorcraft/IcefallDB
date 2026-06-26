#!/usr/bin/env python3
"""Pin the baseline cost of opening a table and of a refresh-dominated mutation.

Produces JSON `{frags: {n: {open_ms_p50, open_ms_p95, mutate_ms_p50, mutate_ms_p95}}}`
and a Markdown row, for fragment counts {10, 100, 1000, 10000}.  The in-process
PyO3 binding is used so the measurement reflects the engine's own registration
and refresh path.

Each timed sample runs in a **fresh child process** so the global `MetaCache`
starts empty. This is essential for measuring the snapshot-checkpoint fast path
honestly: meta paths are table-relative, so re-opening the same (or a copied)
table within one process is served from the warm cache and hides the cold-open
cost the checkpoint is meant to cut. The child prints only the timed duration
(ms); its interpreter startup is excluded from the measurement.

Table generation uses one ``icefalldb import`` call per fragment count: a schema
with ``row_group_target_rows = rows_per_fragment`` is created first, then a TSV
containing all rows is imported.  The writer's row-group planner emits exactly
``n_fragments`` fragments in a single commit, avoiding the subprocess overhead
of per-fragment inserts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median, quantiles

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from benchmarks.mutations.generate import make_table  # noqa: E402

# Fragment counts to measure.
FRAGMENT_COUNTS = [10, 100, 1000, 10000]
# Fixed rows per fragment so total table size grows with fragment count.
ROWS_PER_FRAGMENT = 100
# Repeats per measurement; spec asks for p50/p95.
REPEATS = 20
TABLE_NAME = "bench_data"


def _p50_p95(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    if len(samples) == 2:
        return min(samples), max(samples)
    qs = quantiles(sorted(samples), n=100)
    return qs[49], qs[94]


def _icefalldb_cli() -> Path:
    """Locate the release ``icefalldb`` binary."""
    env_cli = os.environ.get("ICEFALLDB_CLI")
    if env_cli:
        path = Path(env_cli)
        if path.exists():
            return path
        raise RuntimeError(f"ICEFALLDB_CLI set but not found: {env_cli}")

    release_cli = REPO_ROOT / "target" / "release" / "icefalldb"
    if release_cli.exists():
        return release_cli

    debug_cli = REPO_ROOT / "target" / "debug" / "icefalldb"
    if debug_cli.exists():
        return debug_cli

    raise RuntimeError(
        "icefalldb CLI not found. Build with: cargo build --release -p icefalldb-cli"
    )


def _run_cli(args: list[str]) -> None:
    cli = _icefalldb_cli()
    try:
        subprocess.run(
            [str(cli)] + args,
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"icefalldb {' '.join(args)} failed: {exc.stderr or ''}"
        ) from exc


def _schema_for(rows_per_fragment: int) -> dict:
    return {
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": True, "field_id": 1},
            {"name": "value", "type": "int64", "nullable": True, "field_id": 2},
            {"name": "category", "type": "utf8", "nullable": True, "field_id": 3},
        ],
        "sort": ["id"],
        "row_group_target_rows": rows_per_fragment,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": 3,
    }


def _write_tsv(path: Path, data: pa.Table) -> None:
    """Write a TSV file with header id, value, category."""
    ids = data.column("id").to_pylist()
    values = data.column("value").to_pylist()
    categories = data.column("category").to_pylist()

    with path.open("w", encoding="utf-8") as f:
        f.write("id\tvalue\tcategory\n")
        for i, v, c in zip(ids, values, categories):
            f.write(f"{i}\t{v}\t{c}\n")


def _build_base_table(db: Path, n_fragments: int) -> int:
    """Build a fresh IcefallDB table with ``n_fragments`` fragments."""
    total_rows = n_fragments * ROWS_PER_FRAGMENT
    data = make_table(total_rows)

    if db.exists():
        shutil.rmtree(db)
    db.mkdir(parents=True)

    schema_path = db / f"{TABLE_NAME}_schema.json"
    schema_path.write_text(json.dumps(_schema_for(ROWS_PER_FRAGMENT), indent=2))

    tsv_path = db / f"{TABLE_NAME}.tsv"
    _write_tsv(tsv_path, data)

    try:
        _run_cli(["create", "--schema", str(schema_path), str(db), TABLE_NAME])
        _run_cli(["import", str(db), TABLE_NAME, str(tsv_path)])
    finally:
        schema_path.unlink(missing_ok=True)
        tsv_path.unlink(missing_ok=True)

    return total_rows


def _copy_base(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# Runs in a fresh child process so the engine's global MetaCache starts empty
# (cold open). Reads mode + params from argv to avoid quoting/escaping issues.
#   open   mode: times IcefallDBConnection::new + table_count (construction is
#                inside the timed region — that IS the open cost).
#   mutate mode: constructs the connection (untimed), then times mutate() —
#                matching the realistic "open once, then mutate" pattern.
_COLD_PROBE = """
import sys, time
sys.path.insert(0, sys.argv[1])
import icefalldb_query_py
mode, db, table = sys.argv[2], sys.argv[3], sys.argv[4]
if mode == "open":
    t0 = time.perf_counter()
    conn = icefalldb_query_py.IcefallDBConnection(db, [table])
    _ = conn.table_count()
    print((time.perf_counter() - t0) * 1000.0)
else:
    target_id = sys.argv[5]
    conn = icefalldb_query_py.IcefallDBConnection(db, [table])
    _ = conn.table_count()  # finish opening before timing the mutation
    sql = f'DELETE FROM "{table}" WHERE id = {target_id}'
    t0 = time.perf_counter()
    affected = conn.mutate(sql)
    elapsed = (time.perf_counter() - t0) * 1000.0
    assert affected == 1, f"expected 1 row affected, got {affected}"
    print(elapsed)
"""


def _python() -> str:
    return sys.executable


def _run_cold_probe(mode: str, db: Path, target_id: int | None = None) -> float:
    """Run one cold-process measurement and return its duration in ms."""
    args = [
        _python(),
        "-c",
        _COLD_PROBE,
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
        raise RuntimeError(
            f"cold probe ({mode}) failed: {result.stderr or result.stdout}"
        )
    return float(result.stdout.strip())


def _time_open(db: Path) -> float:
    # Fresh child process => empty MetaCache => genuine cold open.
    return _run_cold_probe("open", db)


def _time_mutate(db: Path, target_id: int) -> float:
    # Fresh child process opens, then mutates; only the mutate() call is timed.
    return _run_cold_probe("mutate", db, target_id)


def run_benchmark(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = out_dir / "_snapshot_cost_base"
    tmp_dir = out_dir / "_snapshot_cost_tmp"

    results: dict[str, dict] = {}

    for n_frags in FRAGMENT_COUNTS:
        print(f"\n=== fragment_count={n_frags} ===")
        total_rows = _build_base_table(base_dir, n_frags)
        target_id = total_rows // 2
        print(f"  built {total_rows:,} rows in {n_frags:,} fragments")

        open_samples: list[float] = []
        mutate_samples: list[float] = []

        for i in range(REPEATS):
            # Open cost: fresh copy + fresh (cold-cache) connection.
            # `_time_open`/`_time_mutate` already return milliseconds.
            copy_db = tmp_dir / f"open_{i:03d}"
            _copy_base(base_dir, copy_db)
            open_samples.append(_time_open(copy_db))

            # Mutate cost: fresh copy, fresh connection, timed mutate.
            copy_db = tmp_dir / f"mutate_{i:03d}"
            _copy_base(base_dir, copy_db)
            mutate_samples.append(_time_mutate(copy_db, target_id))

            if (i + 1) % 5 == 0:
                print(
                    f"  {i + 1}/{REPEATS} "
                    f"open_p50={median(open_samples):.2f}ms "
                    f"mutate_p50={median(mutate_samples):.2f}ms"
                )

        open_p50, open_p95 = _p50_p95(open_samples)
        mutate_p50, mutate_p95 = _p50_p95(mutate_samples)

        results[str(n_frags)] = {
            "open_ms_p50": open_p50,
            "open_ms_p95": open_p95,
            "mutate_ms_p50": mutate_p50,
            "mutate_ms_p95": mutate_p95,
            "rows": total_rows,
        }

        print(f"  open    p50={open_p50:.2f}ms p95={open_p95:.2f}ms")
        print(f"  mutate  p50={mutate_p50:.2f}ms p95={mutate_p95:.2f}ms")

    # Clean up scratch directories.
    shutil.rmtree(base_dir, ignore_errors=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def _markdown(results: dict) -> str:
    lines = [
        "| fragments | rows | open_ms_p50 | open_ms_p95 | mutate_ms_p50 | mutate_ms_p95 |",
        "|----------:|-----:|------------:|------------:|--------------:|--------------:|",
    ]
    for n_frags in FRAGMENT_COUNTS:
        r = results[str(n_frags)]
        lines.append(
            f"| {n_frags:,} | {r['rows']:,} | "
            f"{r['open_ms_p50']:.2f} | {r['open_ms_p95']:.2f} | "
            f"{r['mutate_ms_p50']:.2f} | {r['mutate_ms_p95']:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark snapshot open cost and refresh-dominated mutation cost."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "target" / "tmp" / "perf_snapshot_cost",
        help="Directory for scratch datasets and output JSON",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path to write JSON results (default: <out-dir>/snapshot_cost.json)",
    )
    args = parser.parse_args()

    results = run_benchmark(args.out_dir)

    json_path = args.json_out or (args.out_dir / "snapshot_cost.json")
    json_path.write_text(json.dumps({"frags": results}, indent=2) + "\n")
    print(f"\nWrote JSON: {json_path}")

    md = _markdown(results)
    print("\nMarkdown:\n")
    print(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
