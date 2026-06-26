#!/usr/bin/env python3
"""Generate mutation-benchmark datasets from a single seed.

Produces THREE representations of the same data:
  (a) a IcefallDB table (via the CLI + insert)
  (b) a DuckDB native table (.duckdb file)
  (c) a single Parquet file (flat copy)

Columns:
  id        int64  — contiguous 0..N-1, unique key
  value     int64  — random in [0, 1_000_000)
  category  utf8   — low-cardinality ("cat_0" … "cat_9")

Usage:
    python generate.py SIZE FRAGMENTS [--out DIR]
    python generate.py 100000 4               # SMOKE scale
    python generate.py 1000000 10 --out /tmp/bench
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from icefalldb.producer import write_icefalldb_ready_parquet  # noqa: E402

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

_CATEGORY_NAMES = [f"cat_{i}" for i in range(10)]
_SEED = 42


def make_table(n: int) -> pa.Table:
    """Return a PyArrow table with ``n`` rows from a fixed seed."""
    rng = np.random.default_rng(_SEED)
    ids = np.arange(n, dtype=np.int64)
    values = rng.integers(0, 1_000_000, size=n, dtype=np.int64)
    categories = rng.choice(_CATEGORY_NAMES, size=n)
    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "value": pa.array(values, type=pa.int64()),
            "category": pa.array(categories, type=pa.string()),
        }
    )


# ---------------------------------------------------------------------------
# IcefallDB helpers — mirrors generate_events.py patterns
# ---------------------------------------------------------------------------


def _icefalldb_cli() -> Path:
    """Locate the ``icefalldb`` binary (release preferred, debug accepted)."""
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
        "icefalldb CLI not found.  "
        "Build it with: cargo build [-p icefalldb-cli] or set ICEFALLDB_CLI."
    )


def _run_cli(args: list[str]) -> None:
    """Run the IcefallDB CLI and raise on failure."""
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
        raise RuntimeError(f"icefalldb {args[0]} failed\n{exc.stderr or ''}") from exc


def _schema_for_bench(rows_per_fragment: int) -> dict:
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


def build_icefalldb_table(
    db: Path,
    table_name: str,
    data: pa.Table,
    n_fragments: int,
) -> None:
    """Write ``data`` into a IcefallDB table split across ``n_fragments`` fragments."""
    n = len(data)
    rows_per_fragment = max(1, n // n_fragments)

    schema = _schema_for_bench(rows_per_fragment)
    schema_path = db / f"{table_name}_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))

    cli = _icefalldb_cli()
    try:
        subprocess.run(
            [str(cli), "create", "--schema", str(schema_path), str(db), table_name],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"icefalldb create failed\n{exc.stderr or ''}") from exc
    finally:
        schema_path.unlink(missing_ok=True)

    # Write fragments one at a time so we hit the requested fragment count.
    tmp_dir = db / f"_tmp_{table_name}"
    tmp_dir.mkdir(exist_ok=True)
    try:
        for frag_idx in range(n_fragments):
            start = frag_idx * rows_per_fragment
            end = start + rows_per_fragment if frag_idx < n_fragments - 1 else n
            if start >= n:
                break
            fragment = data.slice(start, end - start)
            frag_path = tmp_dir / f"fragment_{frag_idx:06d}.parquet"
            write_icefalldb_ready_parquet(
                frag_path, fragment, row_group_size=len(fragment)
            )
            _run_cli(["insert", str(db), table_name, str(frag_path)])
            frag_path.unlink()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# DuckDB helpers
# ---------------------------------------------------------------------------


def build_duckdb_table(
    duckdb_path: Path,
    data: pa.Table,
) -> None:
    """Write ``data`` into a DuckDB native table at ``duckdb_path``."""
    con = duckdb.connect(str(duckdb_path))
    try:
        # Register the Arrow table and insert into a native DuckDB table.
        con.register("_src", data)
        con.execute("CREATE OR REPLACE TABLE bench_data AS SELECT * FROM _src")
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Parquet copy helper
# ---------------------------------------------------------------------------


def build_parquet_copy(parquet_path: Path, data: pa.Table) -> None:
    """Write ``data`` to a single Parquet file."""
    write_icefalldb_ready_parquet(parquet_path, data, row_group_size=len(data))


# ---------------------------------------------------------------------------
# Row-count verification
# ---------------------------------------------------------------------------


def _count_icefalldb_rows(db: Path, table_name: str) -> int:
    """Count live rows in a IcefallDB table via the CLI query command."""
    cli = _icefalldb_cli()
    result = subprocess.run(
        [
            str(cli),
            "query",
            str(db / table_name),
            "SELECT COUNT(*) AS n FROM bench_data",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # Try to parse `[{"n": <count>}]` JSON output.
    # Fall back to counting via the manifest row_counts if query fails.
    if result.returncode == 0 and result.stdout.strip():
        try:
            rows = json.loads(result.stdout.strip())
            if rows:
                return int(rows[0].get("n", 0))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    # Fallback: read the manifest and sum row_counts.
    manifest_ptr = db / table_name / "_manifest.json"
    ptr = json.loads(manifest_ptr.read_text())
    seq = ptr["latest"]
    manifest_path = db / table_name / "_manifests" / f"{seq:09d}.json"
    manifest = json.loads(manifest_path.read_text())
    counts = manifest.get("row_counts", {})
    if isinstance(counts, dict):
        return sum(counts.values())
    return int(counts)


def _count_icefalldb_rows_from_manifest(db: Path, table_name: str) -> int:
    """Read row count from the IcefallDB manifest (no CLI query needed)."""
    manifest_ptr = db / table_name / "_manifest.json"
    ptr = json.loads(manifest_ptr.read_text())
    seq = ptr["latest"]
    manifest_path = db / table_name / "_manifests" / f"{seq:09d}.json"
    manifest = json.loads(manifest_path.read_text())
    # row_counts can be a list [count, ...], a dict {path: count}, or a scalar.
    counts = manifest.get("row_counts", 0)
    if isinstance(counts, list):
        return sum(int(c) for c in counts)
    if isinstance(counts, dict):
        return sum(int(v) for v in counts.values())
    return int(counts)


def _count_duckdb_rows(duckdb_path: Path) -> int:
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        row = con.execute("SELECT COUNT(*) FROM bench_data").fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def _count_parquet_rows(parquet_path: Path) -> int:
    pf = pq.ParquetFile(str(parquet_path))
    return pf.metadata.num_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate(
    size: int,
    n_fragments: int,
    out_dir: Path,
    *,
    verbose: bool = True,
) -> dict[str, int]:
    """Generate all three representations; return {name: row_count} dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sub-paths
    icefalldb_db = out_dir / "icefalldb_db"
    duckdb_path = out_dir / "bench_data.duckdb"
    parquet_path = out_dir / "bench_data.parquet"
    table_name = "bench_data"

    # ----- generate data from one seed -----
    if verbose:
        print(f"Generating {size:,} rows  fragments={n_fragments}  seed={_SEED}")
    data = make_table(size)

    # ----- (a) IcefallDB table -----
    icefalldb_db.mkdir(exist_ok=True)
    if verbose:
        print(f"  Writing IcefallDB table → {icefalldb_db / table_name} ...")
    if (icefalldb_db / table_name).exists():
        shutil.rmtree(icefalldb_db / table_name)
    build_icefalldb_table(icefalldb_db, table_name, data, n_fragments)
    icefalldb_rows = _count_icefalldb_rows_from_manifest(icefalldb_db, table_name)
    if verbose:
        print(f"    IcefallDB rows: {icefalldb_rows:,}")

    # ----- (b) DuckDB native table -----
    if verbose:
        print(f"  Writing DuckDB native table → {duckdb_path} ...")
    if duckdb_path.exists():
        duckdb_path.unlink()
    build_duckdb_table(duckdb_path, data)
    duckdb_rows = _count_duckdb_rows(duckdb_path)
    if verbose:
        print(f"    DuckDB rows:    {duckdb_rows:,}")

    # ----- (c) Parquet copy -----
    if verbose:
        print(f"  Writing Parquet copy → {parquet_path} ...")
    build_parquet_copy(parquet_path, data)
    parquet_rows = _count_parquet_rows(parquet_path)
    if verbose:
        print(f"    Parquet rows:   {parquet_rows:,}")

    # ----- assert counts match -----
    counts = {
        "icefalldb": icefalldb_rows,
        "duckdb": duckdb_rows,
        "parquet": parquet_rows,
    }
    if len(set(counts.values())) != 1:
        raise AssertionError(f"Row count mismatch across representations: {counts}")
    if verbose:
        print(f"\nAll three representations have matching row count: {size:,}")
        print(f"  IcefallDB: {icefalldb_db / table_name}")
        print(f"  DuckDB:    {duckdb_path}")
        print(f"  Parquet:   {parquet_path}")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mutation-benchmark datasets (IcefallDB + DuckDB + Parquet) "
            "from a single seed."
        )
    )
    parser.add_argument("size", type=int, help="Number of rows")
    parser.add_argument("fragments", type=int, help="Number of IcefallDB fragments")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "target" / "tmp" / "mutation_bench",
        help="Output directory (default: target/tmp/mutation_bench)",
    )
    args = parser.parse_args()

    counts = generate(args.size, args.fragments, args.out)
    _ = counts  # already printed by generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
