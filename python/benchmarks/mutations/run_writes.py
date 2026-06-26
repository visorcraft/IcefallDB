#!/usr/bin/env python3
"""Write-workload benchmark: IcefallDB vs DuckDB native (in-process).

Measures p50 / p95 wall-clock latency for five mutation workloads:
  1. Point DELETE  (by indexed key, single row)
  2. Point UPDATE  (by indexed key, single row)
  3. Bulk  DELETE  (~1% of table by value range)
  4. Bulk  UPDATE  (~1% of table by value range)
  5. CDC MERGE     (80% updates + 20% inserts, batch = 0.5% of table)

Two measurement modes:
  COLD — opens a fresh IcefallDBConnection per timed call on a cloned dataset.
          Each timed sample = connection-open cost + mutation cost.

  WARM — for each run, clones the dataset and opens the connection (excluded from
          timing), then times the mutate() call ONLY.  Each run starts from an
          identical base, so there is no accumulation of deletion-vector fragments.
          This isolates the true per-mutation cost from the one-time connection-open.

Note on ``refresh_table_registration``: after every mutate() the Rust binding
reloads the updated manifest and sidecar files.  In WARM mode this is included
in the timed call (it is inherent to the mutation, not to connection open).  For
the WARM measurements, the base dataset has N_FRAGS fragments each time, so the
refresh cost is stable across runs.

Usage:
    cd <repo-root>/python
    python benchmarks/mutations/run_writes.py [--scale SMOKE|DEFAULT] [--runs N]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from statistics import quantiles
from typing import Callable

import duckdb
import numpy as np
import pyarrow as pa

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from benchmarks.mutations.generate import (  # noqa: E402
    build_duckdb_table,
    build_icefalldb_table,
    make_table,
)
from benchmarks.mutations.params import (  # noqa: E402
    SMOKE_FRAGMENTS,
    SMOKE_RUNS,
    SMOKE_SIZE,
)

# ---------------------------------------------------------------------------
# Default benchmark scale
# ---------------------------------------------------------------------------

DEFAULT_SIZE = SMOKE_SIZE  # 100 000 rows
DEFAULT_FRAGMENTS = SMOKE_FRAGMENTS[0]  # 4 fragments
DEFAULT_RUNS = SMOKE_RUNS  # 3 iterations per workload

# ---------------------------------------------------------------------------
# In-process IcefallDB binding
# ---------------------------------------------------------------------------

try:
    import icefalldb_query_py  # PyO3 binding built via maturin

    _HAS_PY_BINDING = True
except ImportError:
    _HAS_PY_BINDING = False


def _require_binding() -> None:
    if not _HAS_PY_BINDING:
        raise RuntimeError(
            "icefalldb_query_py not available.  Build with:\n"
            "  VIRTUAL_ENV=python/.venv python/.venv/bin/maturin develop --release "
            "-m crates/icefalldb-query-py/Cargo.toml"
        )


def _run_icefalldb_inproc(db_path: Path, table: str, sql: str) -> None:
    """Execute a mutation SQL against a IcefallDB table in-process (COLD path).

    Opens a fresh connection, includes connection-open cost in the call.
    """
    _require_binding()
    conn = icefalldb_query_py.IcefallDBConnection(str(db_path), [table])
    conn.mutate(sql)


def _time_open(db_path: Path, table: str) -> float:
    """Return the wall-clock seconds to open a IcefallDBConnection (no mutation)."""
    t0 = time.perf_counter()
    icefalldb_query_py.IcefallDBConnection(str(db_path), [table])
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# CLI discovery (kept for index creation only)
# ---------------------------------------------------------------------------


def _icefalldb_cli() -> Path:
    env_cli = os.environ.get("ICEFALLDB_CLI")
    if env_cli:
        p = Path(env_cli)
        if p.exists():
            return p
        raise RuntimeError(f"ICEFALLDB_CLI set but not found: {env_cli}")

    for candidate in (
        REPO_ROOT / "target" / "release" / "icefalldb",
        REPO_ROOT / "target" / "debug" / "icefalldb",
    ):
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate

    raise RuntimeError(
        "icefalldb CLI not found.  Build with: cargo build --release -p icefalldb-cli"
    )


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _prepare_dataset(
    tmp_dir: Path, size: int, n_fragments: int, data: pa.Table
) -> tuple[Path, Path]:
    """Build fresh IcefallDB + DuckDB datasets in tmp_dir from pre-generated data."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    icefalldb_db = tmp_dir / "icefalldb_db"
    duckdb_path = tmp_dir / "bench_data.duckdb"

    # Wipe any previous state
    if icefalldb_db.exists():
        shutil.rmtree(icefalldb_db)
    if duckdb_path.exists():
        duckdb_path.unlink()

    icefalldb_db.mkdir()
    build_icefalldb_table(icefalldb_db, "bench_data", data, n_fragments)
    # Create unique index on id (required for MERGE)
    cli = _icefalldb_cli()
    subprocess.run(
        [str(cli), "create-index", "--unique", str(icefalldb_db), "bench_data", "id"],
        check=True,
        capture_output=True,
    )
    build_duckdb_table(duckdb_path, data)
    # Add unique index on id in DuckDB
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS bench_id_idx ON bench_data (id)")
    finally:
        con.close()
    return icefalldb_db, duckdb_path


def _copy_dataset(src_dir: Path, dst_dir: Path) -> tuple[Path, Path]:
    """Clone a prepared dataset directory so each timed run starts identically."""
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)
    return dst_dir / "icefalldb_db", dst_dir / "bench_data.duckdb"


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _timer_seconds(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _p50_p95(samples: list[float]) -> tuple[float, float]:
    if len(samples) == 1:
        return samples[0], samples[0]
    if len(samples) == 2:
        return min(samples), max(samples)
    qs = quantiles(samples, n=100)
    return qs[49], qs[94]


# ---------------------------------------------------------------------------
# COLD workloads (fresh connection per timed run)
# ---------------------------------------------------------------------------


def _workload_point_delete(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
) -> tuple[list[float], list[float]]:
    """COLD: Point DELETE WHERE id = k.  Opens fresh connection per run."""
    ids = data.column("id").to_pylist()
    pick_ids = [ids[int(len(ids) * (i + 0.5) / runs)] for i in range(runs)]

    mdb_times = []
    duck_times = []

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    for i, kid in enumerate(pick_ids):
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"run_{i}")

        sql = f"DELETE FROM bench_data WHERE id = {kid}"
        t = _timer_seconds(lambda: _run_icefalldb_inproc(mdb_db, "bench_data", sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(
                lambda: con.execute(f"DELETE FROM bench_data WHERE id = {kid}")
            )
        finally:
            con.close()
        duck_times.append(t)

    return mdb_times, duck_times


def _workload_point_update(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
) -> tuple[list[float], list[float]]:
    """COLD: Point UPDATE.  Opens fresh connection per run."""
    ids = data.column("id").to_pylist()
    pick_ids = [ids[int(len(ids) * (i + 0.5) / runs)] for i in range(runs)]

    mdb_times = []
    duck_times = []

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    for i, kid in enumerate(pick_ids):
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"run_{i}")

        sql = f"UPDATE bench_data SET value = value + 1 WHERE id = {kid}"
        t = _timer_seconds(lambda: _run_icefalldb_inproc(mdb_db, "bench_data", sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(
                lambda: con.execute(
                    f"UPDATE bench_data SET value = value + 1 WHERE id = {kid}"
                )
            )
        finally:
            con.close()
        duck_times.append(t)

    return mdb_times, duck_times


def _workload_bulk_delete(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
    frac: float = 0.01,
) -> tuple[list[float], list[float]]:
    """COLD: Bulk DELETE affecting ~frac of rows by value range."""
    band_width = int(1_000_000 * frac)
    rng = np.random.default_rng(99)
    starts = rng.integers(0, 1_000_000 - band_width, size=runs).tolist()

    mdb_times = []
    duck_times = []

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    for i, lo in enumerate(starts):
        hi = lo + band_width
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"run_{i}")

        sql = f"DELETE FROM bench_data WHERE value >= {lo} AND value < {hi}"
        t = _timer_seconds(lambda: _run_icefalldb_inproc(mdb_db, "bench_data", sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(lambda: con.execute(sql))
        finally:
            con.close()
        duck_times.append(t)

    return mdb_times, duck_times


def _workload_bulk_update(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
    frac: float = 0.01,
) -> tuple[list[float], list[float]]:
    """COLD: Bulk UPDATE.  Opens fresh connection per run."""
    band_width = int(1_000_000 * frac)
    rng = np.random.default_rng(77)
    starts = rng.integers(0, 1_000_000 - band_width, size=runs).tolist()

    mdb_times = []
    duck_times = []

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    for i, lo in enumerate(starts):
        hi = lo + band_width
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"run_{i}")

        sql = f"UPDATE bench_data SET value = value + 1 WHERE value >= {lo} AND value < {hi}"
        t = _timer_seconds(lambda: _run_icefalldb_inproc(mdb_db, "bench_data", sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(lambda: con.execute(sql))
        finally:
            con.close()
        duck_times.append(t)

    return mdb_times, duck_times


def _workload_cdc_merge(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
    batch_frac: float = 0.005,
) -> tuple[list[float], list[float]]:
    """COLD: CDC MERGE.  Opens fresh connection per run."""
    n = len(data)
    batch_size = max(10, int(n * batch_frac))
    ids = data.column("id").to_pylist()
    max_id = max(ids)

    rng = np.random.default_rng(55)

    mdb_times = []
    duck_times = []

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    for i in range(runs):
        n_upd = int(batch_size * 0.8)
        n_ins = batch_size - n_upd
        upd_ids = rng.choice(ids, size=n_upd, replace=False).tolist()
        ins_ids = [max_id + 1 + j + i * batch_size for j in range(n_ins)]
        upd_vals = rng.integers(0, 1_000_000, size=n_upd).tolist()
        ins_vals = rng.integers(0, 1_000_000, size=n_ins).tolist()

        rows = [(uid, uv, "'cat_0'") for uid, uv in zip(upd_ids, upd_vals)] + [
            (iid, iv, "'cat_0'") for iid, iv in zip(ins_ids, ins_vals)
        ]
        values_clause = ", ".join(f"({rid}, {rv}, {rc})" for rid, rv, rc in rows)

        merge_sql = (
            f"MERGE INTO bench_data USING "
            f"(VALUES {values_clause}) AS src(id, value, category) "
            f"ON bench_data.id = src.id "
            f"WHEN MATCHED THEN UPDATE SET id = src.id, value = src.value, category = src.category "
            f"WHEN NOT MATCHED THEN INSERT (id, value, category) "
            f"VALUES (src.id, src.value, src.category)"
        )

        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"run_{i}")

        t = _timer_seconds(
            lambda: _run_icefalldb_inproc(mdb_db, "bench_data", merge_sql)
        )
        mdb_times.append(t)

        duck_merge_sql = (
            f"MERGE INTO bench_data USING "
            f"(VALUES {values_clause}) AS src(id, value, category) "
            f"ON bench_data.id = src.id "
            f"WHEN MATCHED THEN UPDATE SET value = src.value, category = src.category "
            f"WHEN NOT MATCHED THEN INSERT (id, value, category) "
            f"VALUES (src.id, src.value, src.category)"
        )
        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(lambda: con.execute(duck_merge_sql))
        finally:
            con.close()
        duck_times.append(t)

    return mdb_times, duck_times


# ---------------------------------------------------------------------------
# WARM workloads (connection opened OUTSIDE the timed region per run)
# ---------------------------------------------------------------------------
# Strategy: for each run, CLONE the base dataset (so each run starts from the
# same N-fragment baseline) and open the connection (excluded from timing).
# Only the mutate() call is timed.  This removes the one-time session-init cost
# (Tokio runtime, DataFusion session, initial manifest+index load) from the
# measured value, leaving only the per-mutation work:
#   - DataFusion SQL plan + scan (locate matching rows)
#   - Writer lock + commit (deletion-vector or patch fragment write)
#   - refresh_table_registration (reload updated manifest + N-fragment sidecars)
#
# Because each run starts from the same clean baseline (no accumulated deletion
# vectors), the per-run cost is stable and comparable across runs.
# ---------------------------------------------------------------------------


def _workload_point_delete_warm(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
) -> tuple[list[float], list[float], list[float]]:
    """WARM: Point DELETE.

    Returns (open_times, mdb_mutate_times, duck_mutate_times).
    """
    ids = data.column("id").to_pylist()
    pick_ids = [ids[int(len(ids) * (i + 0.5) / runs)] for i in range(runs)]

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    open_times = []
    mdb_times = []
    duck_times = []

    for i, kid in enumerate(pick_ids):
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"warm_{i}")

        # Open outside timing window.
        t0 = time.perf_counter()
        conn = icefalldb_query_py.IcefallDBConnection(str(mdb_db), ["bench_data"])
        open_times.append(time.perf_counter() - t0)

        # Time only the mutation.
        sql = f"DELETE FROM bench_data WHERE id = {kid}"
        t = _timer_seconds(lambda: conn.mutate(sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(
                lambda: con.execute(f"DELETE FROM bench_data WHERE id = {kid}")
            )
            duck_times.append(t)
        finally:
            con.close()

    return open_times, mdb_times, duck_times


def _workload_point_update_warm(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
) -> tuple[list[float], list[float], list[float]]:
    """WARM: Point UPDATE."""
    ids = data.column("id").to_pylist()
    pick_ids = [ids[int(len(ids) * (i + 0.5) / runs)] for i in range(runs)]

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    open_times = []
    mdb_times = []
    duck_times = []

    for i, kid in enumerate(pick_ids):
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"warm_{i}")

        t0 = time.perf_counter()
        conn = icefalldb_query_py.IcefallDBConnection(str(mdb_db), ["bench_data"])
        open_times.append(time.perf_counter() - t0)

        sql = f"UPDATE bench_data SET value = value + 1 WHERE id = {kid}"
        t = _timer_seconds(lambda: conn.mutate(sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(
                lambda: con.execute(
                    f"UPDATE bench_data SET value = value + 1 WHERE id = {kid}"
                )
            )
            duck_times.append(t)
        finally:
            con.close()

    return open_times, mdb_times, duck_times


def _workload_bulk_delete_warm(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
    frac: float = 0.01,
) -> tuple[list[float], list[float], list[float]]:
    """WARM: Bulk DELETE."""
    band_width = int(1_000_000 * frac)
    rng = np.random.default_rng(99)
    starts = rng.integers(0, 1_000_000 - band_width, size=runs).tolist()

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    open_times = []
    mdb_times = []
    duck_times = []

    for i, lo in enumerate(starts):
        hi = lo + band_width
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"warm_{i}")

        t0 = time.perf_counter()
        conn = icefalldb_query_py.IcefallDBConnection(str(mdb_db), ["bench_data"])
        open_times.append(time.perf_counter() - t0)

        sql = f"DELETE FROM bench_data WHERE value >= {lo} AND value < {hi}"
        t = _timer_seconds(lambda: conn.mutate(sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(lambda: con.execute(sql))
            duck_times.append(t)
        finally:
            con.close()

    return open_times, mdb_times, duck_times


def _workload_bulk_update_warm(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
    frac: float = 0.01,
) -> tuple[list[float], list[float], list[float]]:
    """WARM: Bulk UPDATE."""
    band_width = int(1_000_000 * frac)
    rng = np.random.default_rng(77)
    starts = rng.integers(0, 1_000_000 - band_width, size=runs).tolist()

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    open_times = []
    mdb_times = []
    duck_times = []

    for i, lo in enumerate(starts):
        hi = lo + band_width
        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"warm_{i}")

        t0 = time.perf_counter()
        conn = icefalldb_query_py.IcefallDBConnection(str(mdb_db), ["bench_data"])
        open_times.append(time.perf_counter() - t0)

        sql = f"UPDATE bench_data SET value = value + 1 WHERE value >= {lo} AND value < {hi}"
        t = _timer_seconds(lambda: conn.mutate(sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(lambda: con.execute(sql))
            duck_times.append(t)
        finally:
            con.close()

    return open_times, mdb_times, duck_times


def _workload_cdc_merge_warm(
    data: pa.Table,
    base_dir: Path,
    run_dir: Path,
    runs: int,
    n_fragments: int,
    batch_frac: float = 0.005,
) -> tuple[list[float], list[float], list[float]]:
    """WARM: CDC MERGE."""
    n = len(data)
    batch_size = max(10, int(n * batch_frac))
    ids = data.column("id").to_pylist()
    max_id = max(ids)

    rng = np.random.default_rng(55)

    _prepare_dataset(base_dir, len(data), n_fragments, data)

    open_times = []
    mdb_times = []
    duck_times = []

    for i in range(runs):
        n_upd = int(batch_size * 0.8)
        n_ins = batch_size - n_upd
        upd_ids = rng.choice(ids, size=n_upd, replace=False).tolist()
        ins_ids = [max_id + 1 + j + i * batch_size for j in range(n_ins)]
        upd_vals = rng.integers(0, 1_000_000, size=n_upd).tolist()
        ins_vals = rng.integers(0, 1_000_000, size=n_ins).tolist()

        rows = [(uid, uv, "'cat_0'") for uid, uv in zip(upd_ids, upd_vals)] + [
            (iid, iv, "'cat_0'") for iid, iv in zip(ins_ids, ins_vals)
        ]
        values_clause = ", ".join(f"({rid}, {rv}, {rc})" for rid, rv, rc in rows)

        merge_sql = (
            f"MERGE INTO bench_data USING "
            f"(VALUES {values_clause}) AS src(id, value, category) "
            f"ON bench_data.id = src.id "
            f"WHEN MATCHED THEN UPDATE SET id = src.id, value = src.value, category = src.category "
            f"WHEN NOT MATCHED THEN INSERT (id, value, category) "
            f"VALUES (src.id, src.value, src.category)"
        )
        duck_merge_sql = (
            f"MERGE INTO bench_data USING "
            f"(VALUES {values_clause}) AS src(id, value, category) "
            f"ON bench_data.id = src.id "
            f"WHEN MATCHED THEN UPDATE SET value = src.value, category = src.category "
            f"WHEN NOT MATCHED THEN INSERT (id, value, category) "
            f"VALUES (src.id, src.value, src.category)"
        )

        mdb_db, duck_path = _copy_dataset(base_dir, run_dir / f"warm_{i}")

        t0 = time.perf_counter()
        conn = icefalldb_query_py.IcefallDBConnection(str(mdb_db), ["bench_data"])
        open_times.append(time.perf_counter() - t0)

        t = _timer_seconds(lambda: conn.mutate(merge_sql))
        mdb_times.append(t)

        con = duckdb.connect(str(duck_path))
        try:
            t = _timer_seconds(lambda: con.execute(duck_merge_sql))
            duck_times.append(t)
        finally:
            con.close()

    return open_times, mdb_times, duck_times


# ---------------------------------------------------------------------------
# Measure cold-open cost only (no mutation)
# ---------------------------------------------------------------------------


def _measure_cold_open_cost(
    data: pa.Table,
    base_dir: Path,
    n_fragments: int,
    runs: int,
) -> list[float]:
    """Measure connection-open cost in isolation (no mutation)."""
    _prepare_dataset(base_dir, len(data), n_fragments, data)
    mdb_db = base_dir / "icefalldb_db"
    times = []
    for _ in range(runs):
        t = _time_open(mdb_db, "bench_data")
        times.append(t)
    return times


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

WORKLOADS = [
    ("point_delete", "Point DELETE (1 row, by id)"),
    ("point_update", "Point UPDATE (1 row, by id)"),
    ("bulk_delete", "Bulk DELETE  (~1% rows, value range)"),
    ("bulk_update", "Bulk UPDATE  (~1% rows, value range)"),
    ("cdc_merge", "CDC MERGE    (0.5% batch, 80% upd / 20% ins)"),
]


def run_benchmark(
    size: int = DEFAULT_SIZE,
    n_fragments: int = DEFAULT_FRAGMENTS,
    runs: int = DEFAULT_RUNS,
    out_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run all workloads in COLD and WARM modes; return a results dict."""

    _require_binding()

    if out_dir is None:
        out_dir = REPO_ROOT / "target" / "tmp" / "write_bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("IcefallDB: in-process via icefalldb_query_py binding")
        print(
            f"Scale: {size:,} rows, {n_fragments} fragments, {runs} run(s) per workload"
        )
        print()

    data = make_table(size)

    # ------------------------------------------------------------------
    # Measure cold-open cost in isolation
    # ------------------------------------------------------------------
    if verbose:
        print("Measuring connection cold-open cost ...")
    open_base = out_dir / "base_open"
    open_times = _measure_cold_open_cost(data, open_base, n_fragments, runs)
    open_p50, open_p95 = _p50_p95(open_times)
    if verbose:
        print(f"  Cold open p50={open_p50 * 1000:.1f} ms  p95={open_p95 * 1000:.1f} ms")
        print()

    # ------------------------------------------------------------------
    # Run each workload — COLD and WARM
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    cold_fns = {
        "point_delete": _workload_point_delete,
        "point_update": _workload_point_update,
        "bulk_delete": _workload_bulk_delete,
        "bulk_update": _workload_bulk_update,
        "cdc_merge": _workload_cdc_merge,
    }
    warm_fns = {
        "point_delete": _workload_point_delete_warm,
        "point_update": _workload_point_update_warm,
        "bulk_delete": _workload_bulk_delete_warm,
        "bulk_update": _workload_bulk_update_warm,
        "cdc_merge": _workload_cdc_merge_warm,
    }

    for key, label in WORKLOADS:
        if verbose:
            print(f"Running COLD: {label} ...")
        cold_base = out_dir / f"base_{key}_cold"
        cold_runs_dir = out_dir / f"runs_{key}_cold"
        if cold_runs_dir.exists():
            shutil.rmtree(cold_runs_dir)

        mdb_cold, duck_cold = cold_fns[key](
            data, cold_base, cold_runs_dir, runs, n_fragments
        )

        mdb_cold_p50, mdb_cold_p95 = _p50_p95(mdb_cold)
        duck_cold_p50, duck_cold_p95 = _p50_p95(duck_cold)
        ratio_cold_p50 = (
            mdb_cold_p50 / duck_cold_p50 if duck_cold_p50 > 0 else float("inf")
        )

        if verbose:
            print(
                f"  IcefallDB COLD p50={mdb_cold_p50 * 1000:.1f} ms  p95={mdb_cold_p95 * 1000:.1f} ms"
            )
            print(
                f"  DuckDB    COLD p50={duck_cold_p50 * 1000:.1f} ms  p95={duck_cold_p95 * 1000:.1f} ms"
            )
            print(f"  Ratio(COLD p50) {ratio_cold_p50:.1f}x")

        if verbose:
            print(f"Running WARM: {label} ...")
        warm_base = out_dir / f"base_{key}_warm"
        warm_runs_dir = out_dir / f"runs_{key}_warm"
        if warm_runs_dir.exists():
            shutil.rmtree(warm_runs_dir)

        open_sub_times, mdb_warm, duck_warm = warm_fns[key](
            data, warm_base, warm_runs_dir, runs, n_fragments
        )

        mdb_warm_p50, mdb_warm_p95 = _p50_p95(mdb_warm)
        duck_warm_p50, duck_warm_p95 = _p50_p95(duck_warm)
        ratio_warm_p50 = (
            mdb_warm_p50 / duck_warm_p50 if duck_warm_p50 > 0 else float("inf")
        )
        open_sub_p50, open_sub_p95 = _p50_p95(open_sub_times)

        if verbose:
            print(
                f"  IcefallDB WARM p50={mdb_warm_p50 * 1000:.1f} ms  p95={mdb_warm_p95 * 1000:.1f} ms"
            )
            print(
                f"  DuckDB    WARM p50={duck_warm_p50 * 1000:.1f} ms  p95={duck_warm_p95 * 1000:.1f} ms"
            )
            print(
                f"  Ratio(WARM p50) {ratio_warm_p50:.1f}x  (open outside timing: {open_sub_p50 * 1000:.1f} ms p50)"
            )
            print()

        results[key] = {
            "label": label,
            # COLD: open + mutate
            "mdb_cold_p50_ms": mdb_cold_p50 * 1000,
            "mdb_cold_p95_ms": mdb_cold_p95 * 1000,
            "duck_cold_p50_ms": duck_cold_p50 * 1000,
            "duck_cold_p95_ms": duck_cold_p95 * 1000,
            "ratio_cold_p50": ratio_cold_p50,
            "mdb_cold_raw_ms": [t * 1000 for t in mdb_cold],
            "duck_cold_raw_ms": [t * 1000 for t in duck_cold],
            # WARM: mutate only (connection open excluded from timing)
            "mdb_warm_p50_ms": mdb_warm_p50 * 1000,
            "mdb_warm_p95_ms": mdb_warm_p95 * 1000,
            "duck_warm_p50_ms": duck_warm_p50 * 1000,
            "duck_warm_p95_ms": duck_warm_p95 * 1000,
            "ratio_warm_p50": ratio_warm_p50,
            "open_sub_p50_ms": open_sub_p50 * 1000,
            "mdb_warm_raw_ms": [t * 1000 for t in mdb_warm],
            "duck_warm_raw_ms": [t * 1000 for t in duck_warm],
        }

    return {
        "size": size,
        "n_fragments": n_fragments,
        "runs": runs,
        "open_p50_ms": open_p50 * 1000,
        "open_p95_ms": open_p95 * 1000,
        "open_raw_ms": [t * 1000 for t in open_times],
        "workloads": results,
    }


def write_results_md(bench: dict, results_path: Path) -> None:
    size = bench["size"]
    frags = bench["n_fragments"]
    runs = bench["runs"]
    open_p50 = bench["open_p50_ms"]
    open_p95 = bench["open_p95_ms"]

    lines = [
        "# Mutation Benchmark Results — IcefallDB vs DuckDB Native",
        "",
        "## Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Scale (rows) | {size:,} |",
        f"| IcefallDB fragments | {frags} |",
        f"| Runs per workload | {runs} |",
        "| IcefallDB method | **in-process** via `icefalldb_query_py` PyO3 binding |",
        "| DuckDB method | **in-process** via `duckdb` Python module |",
        "| Release build | yes (maturin --release) |",
        "",
        "## Connection Cold-Open Cost (no mutation)",
        "",
        "Time to open a `IcefallDBConnection` with no mutation — tokio runtime init, "
        "DataFusion session construction, manifest pointer read, manifest JSON parse, "
        f"{frags}× `.meta` sidecar reads, secondary index load.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| p50 | {open_p50:.1f} ms |",
        f"| p95 | {open_p95:.1f} ms |",
        "",
        "The `_rowindex` AddressMap is **not** loaded during connection open — it is "
        "loaded lazily only inside `resolve_live_addresses_storage`, which is not on "
        "any live mutation code path.  The open cost is dominated by the tokio runtime "
        "init + DataFusion session + sidecar reads.",
        "",
        "## WARM Results — per-mutation cost (connection open excluded from timing)",
        "",
        "Each run: clone the base dataset (N fragments, clean baseline), open the "
        "connection (excluded from timing), then time `mutate()`.  Each run starts "
        "from the same fragment count so `refresh_table_registration` cost is stable.",
        "",
        "| Workload | IcefallDB p50 (ms) | IcefallDB p95 (ms) | DuckDB p50 (ms) | DuckDB p95 (ms) | Ratio p50 |",
        "|---|---|---|---|---|---|",
    ]

    for key, _ in WORKLOADS:
        w = bench["workloads"][key]
        lines.append(
            f"| {w['label']} "
            f"| {w['mdb_warm_p50_ms']:.1f} "
            f"| {w['mdb_warm_p95_ms']:.1f} "
            f"| {w['duck_warm_p50_ms']:.1f} "
            f"| {w['duck_warm_p95_ms']:.1f} "
            f"| {w['ratio_warm_p50']:.1f}x |"
        )

    lines += [
        "",
        "Ratio = IcefallDB / DuckDB (lower is better for IcefallDB).",
        "",
        "## COLD Results — open + mutation cost combined",
        "",
        "Each run opens a fresh `IcefallDBConnection` on a cloned dataset; timing "
        "includes both the connection-open overhead and the mutation.",
        "",
        "| Workload | IcefallDB p50 (ms) | IcefallDB p95 (ms) | DuckDB p50 (ms) | DuckDB p95 (ms) | Ratio p50 |",
        "|---|---|---|---|---|---|",
    ]

    for key, _ in WORKLOADS:
        w = bench["workloads"][key]
        lines.append(
            f"| {w['label']} "
            f"| {w['mdb_cold_p50_ms']:.1f} "
            f"| {w['mdb_cold_p95_ms']:.1f} "
            f"| {w['duck_cold_p50_ms']:.1f} "
            f"| {w['duck_cold_p95_ms']:.1f} "
            f"| {w['ratio_cold_p50']:.1f}x |"
        )

    lines += [
        "",
        "Ratio = IcefallDB / DuckDB (lower is better for IcefallDB).",
        "",
        "---",
        "",
        "## Notes",
        "",
        "### What the connection-open cost actually contains",
        "",
        f"The ~{open_p50:.0f} ms cold-open cost is NOT `_rowindex` AddressMap "
        "deserialization — the AddressMap is never touched during connection open.  "
        "The `AddressMap::open` function is only called from "
        "`resolve_live_addresses_storage`, which exists only in unit tests and is not "
        "on any live query or mutation code path.  The actual open cost breaks down as:",
        "",
        "- Tokio runtime creation + DataFusion `SessionContext` construction",
        "- `_manifest.json` pointer read + manifest JSON parse",
        f"- {frags}× `.meta` sidecar JSON reads (one per fragment, includes "
        "  row-group statistics for predicate pruning)",
        "- Secondary B-tree index file reads (one `.json` per indexed column)",
        "",
        "### WARM vs COLD comparison",
        "",
        f"COLD = WARM + ~{open_p50:.0f} ms connection-open overhead.  For the point "
        "workloads, the open overhead is a significant fraction of the total cold time.  "
        "A long-lived connection amortises this across all mutations in its lifetime.",
        "",
        "### WARM mutation breakdown",
        "",
        "Each WARM `mutate()` call includes:",
        "- DataFusion SQL parse + physical plan construction",
        "- `locate_matches`: DataFusion scan over all fragments to find matching rows "
        "  (uses secondary index for point mutations, full scan for bulk/range mutations)",
        "- Writer lock acquire",
        "- Commit: write deletion-vector (DELETE) or patch fragment (UPDATE)",
        "- `refresh_table_registration`: reload manifest + sidecar files so subsequent "
        "  queries and mutations see the updated snapshot",
        "",
        "The `refresh_table_registration` step (manifest + sidecar reload) is inherent "
        "to each mutation call and included in the WARM timings.  It is NOT the same as "
        "the initial connection-open cost (no tokio runtime or DataFusion session creation).",
        "",
        "### MERGE performance note",
        "",
        "The CDC MERGE workload classifies each source key with an individual "
        "`SELECT _rowid, _rowaddr WHERE id = <k>` DataFusion scan.  For a 500-row "
        "batch this means ~500 separate plan+execute cycles — O(N-scans).  This is a "
        "known algorithmic limitation; the correct fix is a batched index probe.  "
        "The MERGE ratio reflects this structural gap, not the commit path.",
        "",
        "### Previous (superseded) results",
        "",
        "The original benchmark timed COLD only (fresh connection per run), reporting "
        "point DELETE at ~150 ms.  This document adds WARM mode, which shows the "
        "per-mutation cost minus the one-time open overhead.",
        "",
        "---",
        "",
        "## Interpretation",
        "",
        f"- **Point WARM vs DuckDB**: IcefallDB point DELETE/UPDATE costs "
        f"~{bench['workloads']['point_delete']['mdb_warm_p50_ms']:.0f} ms "
        f"vs DuckDB ~{bench['workloads']['point_delete']['duck_warm_p50_ms']:.1f} ms.  "
        "The gap is driven by DataFusion scan overhead (plan+execute cycle to locate "
        "the row) vs DuckDB's native B-tree direct lookup.",
        "",
        "- **Bulk WARM**: DataFusion full-table scan + deletion-vector or patch commit.  "
        "IcefallDB avoids full Parquet file rewrites; the scan cost dominates.",
        "",
        "- **Cold overhead**: ~" + f"{open_p50:.0f}" + " ms one-time per connection; "
        "negligible when amortised over many mutations on a long-lived connection.",
        "",
        "- **MERGE**: O(N-scans) algorithmic gap — not a commit-path issue.",
    ]

    results_path.write_text("\n".join(lines) + "\n")


def write_results_json(bench: dict, json_path: Path) -> None:
    """Serialize the already-computed benchmark metrics dict to JSON.

    Shape (mirrors the brief's suggestion):
    {
      "scale": <rows>,
      "fragments": <n>,
      "runs": <n>,
      "workloads": {
        "point_delete":  {"mdb_p50_ms":.., "mdb_p95_ms":.., "duckdb_p95_ms":.., "ratio_vs_duckdb":..},
        "point_update":  {...},
        "bulk_delete_1pct": {...},
        "bulk_update_1pct": {...},
        "merge_upsert":  {...}
      },
      "open_p50_ms": ..,
      "open_p95_ms": ..
    }
    The WARM p95 values are used as the canonical per-mutation cost (connection
    open excluded from timing), matching the bar definitions.
    """
    import json

    key_map = {
        "point_delete": "point_delete",
        "point_update": "point_update",
        "bulk_delete": "bulk_delete_1pct",
        "bulk_update": "bulk_update_1pct",
        "cdc_merge": "merge_upsert",
    }
    workloads_out: dict = {}
    for src_key, dst_key in key_map.items():
        w = bench["workloads"][src_key]
        workloads_out[dst_key] = {
            "mdb_p50_ms": round(w["mdb_warm_p50_ms"], 4),
            "mdb_p95_ms": round(w["mdb_warm_p95_ms"], 4),
            "duckdb_p50_ms": round(w["duck_warm_p50_ms"], 4),
            "duckdb_p95_ms": round(w["duck_warm_p95_ms"], 4),
            "ratio_vs_duckdb": round(w["mdb_warm_p95_ms"] / w["duck_warm_p95_ms"], 4)
            if w["duck_warm_p95_ms"] > 0
            else None,
            # Also record COLD p95 for completeness
            "mdb_cold_p95_ms": round(w["mdb_cold_p95_ms"], 4),
            "duckdb_cold_p95_ms": round(w["duck_cold_p95_ms"], 4),
        }

    payload = {
        "scale": bench["size"],
        "fragments": bench["n_fragments"],
        "runs": bench["runs"],
        "workloads": workloads_out,
        "open_p50_ms": round(bench["open_p50_ms"], 4),
        "open_p95_ms": round(bench["open_p95_ms"], 4),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write-workload benchmark: IcefallDB vs DuckDB native (in-process)"
    )
    parser.add_argument(
        "--scale",
        choices=["SMOKE", "DEFAULT"],
        default="DEFAULT",
        help="Benchmark scale profile (default: DEFAULT = SMOKE_SIZE rows)",
    )
    parser.add_argument("--runs", type=int, default=None, help="Override run count")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "target" / "tmp" / "write_bench",
        help="Scratch directory for datasets",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).parent / "RESULTS.md",
        help="Output Markdown results file",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="If given, also write machine-readable metrics as JSON to this path",
    )
    args = parser.parse_args()

    size = DEFAULT_SIZE
    n_frags = DEFAULT_FRAGMENTS
    runs = args.runs if args.runs is not None else DEFAULT_RUNS

    bench = run_benchmark(size=size, n_fragments=n_frags, runs=runs, out_dir=args.out)

    results_path = args.results
    write_results_md(bench, results_path)
    print(f"\nResults written to: {results_path}")

    if args.json is not None:
        write_results_json(bench, args.json)
        print(f"JSON metrics written to: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
