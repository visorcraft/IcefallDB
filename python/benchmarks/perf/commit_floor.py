#!/usr/bin/env python3
"""Decompose the per-commit latency floor of a point INSERT/UPDATE/DELETE on a
tiny (100-row) table, and count the durability barriers (fsync/fdatasync) each
commit issues.

The small-table per-op latency is dominated by a fixed floor — locate +
write-fragment-or-DV + manifest write + checkpoint write + the fsyncs that make
them durable — that is independent of table size. This script pins that floor so
WAL-fast-commit and fsync-coalescing have a before/after.

fsync counting uses the `LocalStorage` global barrier counter exposed by the CLI
under `ICEFALLDB_REPORT_FSYNCS=1` (which prints `fsyncs=<n>` to stderr) — `strace`
is not assumed to be available. Each mutating sample runs against a **fresh
clone** of the base table so every repeat starts from identical on-disk state.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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

ROWS = 100
REPEATS = 9
TABLE_NAME = "bench_data"
_FSYNC_RE = re.compile(r"fsyncs=(\d+)")


def _icefalldb_cli() -> Path:
    env_cli = os.environ.get("ICEFALLDB_CLI")
    if env_cli and Path(env_cli).exists():
        return Path(env_cli)
    for c in (
        REPO_ROOT / "target/release/icefalldb",
        REPO_ROOT / "target/debug/icefalldb",
    ):
        if c.is_file() and os.access(str(c), os.X_OK):
            return c
    raise RuntimeError(
        "icefalldb CLI not found; build with: cargo build --release -p icefalldb-cli"
    )


def _run_cli(
    args: list[str], *, report_fsyncs: bool = False, wal: bool | None = None
) -> tuple[float, int | None]:
    """Run the CLI, returning (elapsed_ms, fsync_count or None).

    ``wal`` forces the commit mode via the CLI's ``ICEFALLDB_WAL`` A/B override
    (``1`` = WAL-fast-commit, ``0`` = sync); ``None`` leaves the default.
    """
    env = dict(os.environ)
    if report_fsyncs:
        env["ICEFALLDB_REPORT_FSYNCS"] = "1"
    if wal is not None:
        env["ICEFALLDB_WAL"] = "1" if wal else "0"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [str(_icefalldb_cli()), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    elapsed = (time.perf_counter() - t0) * 1000.0
    fsyncs = None
    if report_fsyncs:
        m = _FSYNC_RE.search(proc.stderr)
        fsyncs = int(m.group(1)) if m else None
    return elapsed, fsyncs


def _schema(total_rows: int) -> dict:
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
    ids, values, cats = (
        data.column("id").to_pylist(),
        data.column("value").to_pylist(),
        data.column("category").to_pylist(),
    )
    with path.open("w", encoding="utf-8") as f:
        f.write("id\tvalue\tcategory\n")
        for i, v, c in zip(ids, values, cats):
            f.write(f"{i}\t{v}\t{c}\n")


def _build_base(db: Path) -> None:
    if db.exists():
        shutil.rmtree(db)
    db.mkdir(parents=True)
    schema_path = db / "s.json"
    schema_path.write_text(json.dumps(_schema(ROWS)))
    tsv_path = db / "d.tsv"
    _write_tsv(tsv_path, make_table(ROWS))
    try:
        _run_cli(["create", "--schema", str(schema_path), str(db), TABLE_NAME])
        _run_cli(["import", str(db), TABLE_NAME, str(tsv_path)])
    finally:
        schema_path.unlink(missing_ok=True)
        tsv_path.unlink(missing_ok=True)


def _one_row_parquet(db: Path, next_id: int) -> Path:
    row = pa.table(
        {
            "id": pa.array([next_id], pa.int64()),
            "value": pa.array([0], pa.int64()),
            "category": pa.array(["cat_0"], pa.string()),
        }
    )
    frag = db / "_one.parquet"
    write_icefalldb_ready_parquet(frag, row, row_group_size=1)
    return frag


def _measure_op(
    base: Path, tmp: Path, op: str, repeats: int, wal: bool | None = None
) -> dict:
    times: list[float] = []
    fsyncs: list[int] = []
    for i in range(repeats):
        clone = tmp / f"{op}_{i:03d}"
        if clone.exists():
            shutil.rmtree(clone)
        shutil.copytree(base, clone)
        if op == "insert":
            frag = _one_row_parquet(clone, ROWS + i)
            args = ["insert", str(clone), TABLE_NAME, str(frag)]
        elif op == "update":
            args = [
                "query",
                str(clone / TABLE_NAME),
                f'UPDATE "{TABLE_NAME}" SET value = 7 WHERE id = {ROWS // 2}',
            ]
        else:  # delete
            args = [
                "query",
                str(clone / TABLE_NAME),
                f'DELETE FROM "{TABLE_NAME}" WHERE id = {ROWS // 2}',
            ]
        ms, fs = _run_cli(args, report_fsyncs=True, wal=wal)
        times.append(ms)
        if fs is not None:
            fsyncs.append(fs)
        shutil.rmtree(clone, ignore_errors=True)
    return {
        "op": op,
        "fsyncs": median(fsyncs) if fsyncs else None,
        "fsyncs_min": min(fsyncs) if fsyncs else None,
        "fsyncs_max": max(fsyncs) if fsyncs else None,
        "ms_median": median(times),
        "ms_min": min(times),
        "ms_max": max(times),
    }


def run(out_dir: Path, repeats: int, wal: bool | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "_base"
    tmp = out_dir / "_tmp"
    _build_base(base)
    results = {
        op: _measure_op(base, tmp, op, repeats, wal)
        for op in ("insert", "update", "delete")
    }
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return results


def _markdown(by_mode: dict) -> str:
    lines = [
        "| op (100-row table, CLI e2e) | mode | fsyncs / commit | total (ms) |",
        "|:------|:-----|----------------:|-----------:|",
    ]
    for op in ("insert", "update", "delete"):
        for mode in by_mode:
            r = by_mode[mode][op]
            fs = f"{r['fsyncs']:.0f}" if r["fsyncs"] is not None else "n/a"
            lines.append(
                f"| {op.upper()} | {mode} | {fs} | "
                f"{r['ms_median']:.1f} ({r['ms_min']:.1f}–{r['ms_max']:.1f}) |"
            )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "target/tmp/perf_commit_floor"
    )
    p.add_argument(
        "--mode",
        choices=["sync", "wal", "both"],
        default="both",
        help="Commit mode to measure (forces the CLI's ICEFALLDB_WAL override)",
    )
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    modes = (
        {"sync": False, "wal": True}
        if args.mode == "both"
        else {args.mode: args.mode == "wal"}
    )
    by_mode = {
        name: run(args.out_dir, args.repeats, wal) for name, wal in modes.items()
    }

    json_path = args.json_out or (args.out_dir / "commit_floor.json")
    json_path.write_text(json.dumps(by_mode, indent=2) + "\n")
    print(f"Wrote JSON: {json_path}\n")
    print(_markdown(by_mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
