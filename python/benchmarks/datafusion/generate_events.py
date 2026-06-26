#!/usr/bin/env python3
"""Generate events, events_wide, and categories tables for the DataFusion matrix.

Tables are written sorted by their hot keys and dictionary-encoded where
specified, then imported into a IcefallDB database using the release CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))
from icefalldb.producer import write_icefalldb_ready_parquet  # noqa: E402

CATEGORY_NAMES = [f"cat_{i}" for i in range(10)]
STATUS_NAMES = ["ok", "fail"]


def categories_table() -> pa.Table:
    """Return the 10-row categories dimension table."""
    return pa.table(
        {
            "category_id": pa.array(np.arange(10, dtype=np.int64)),
            "category_name": pa.array(CATEGORY_NAMES),
        }
    )


def _random_timestamps(rng: np.random.Generator, n: int) -> pa.Array:
    """Return ``n`` random timestamps in a 120-day range starting 2024-01-01."""
    start_us = np.datetime64("2024-01-01", "us").astype(np.int64)
    range_us = 120 * 24 * 60 * 60 * 1_000_000
    offsets = rng.integers(0, range_us, size=n, dtype=np.int64)
    # Sort the offsets so the table is already ordered by ``ts``; the producer
    # will re-validate the order when sort_keys are supplied.
    offsets = np.sort(offsets)
    return pa.array(start_us + offsets, type=pa.timestamp("us"))


def events_table(n: int) -> pa.Table:
    """Return the ``events`` table with ``n`` rows."""
    rng = np.random.default_rng(42)
    # Use six-digit decimal values so Parquet footer statistics round-trip
    # stably through JSON serialization used for sidecar checksums.
    values = rng.integers(0, 1_000_000, size=n, dtype=np.int64) / 1_000_000.0
    return pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "ts": _random_timestamps(rng, n),
            "category": pa.array(rng.choice(CATEGORY_NAMES, size=n)),
            "value": pa.array(values, type=pa.float64()),
        }
    )


def _six_digit_floats(rng: np.random.Generator, n: int) -> pa.Array:
    """Return ``n`` float64 values in [0, 1) with at most six decimals."""
    values = rng.integers(0, 1_000_000, size=n, dtype=np.int64) / 1_000_000.0
    return pa.array(values, type=pa.float64())


def events_wide_table(n: int) -> pa.Table:
    """Return the ``events_wide`` table with ``n`` rows."""
    rng = np.random.default_rng(43)
    return pa.table(
        {
            "id": pa.array(np.arange(n, dtype=np.int64)),
            "ts": _random_timestamps(rng, n),
            "int_a": pa.array(rng.integers(0, 1_000_000, size=n, dtype=np.int64)),
            "int_b": pa.array(rng.integers(0, 1_000_000, size=n, dtype=np.int64)),
            "int_c": pa.array(rng.integers(0, 1_000_000, size=n, dtype=np.int64)),
            "int_d": pa.array(rng.integers(0, 1_000_000, size=n, dtype=np.int64)),
            "float_a": _six_digit_floats(rng, n),
            "float_b": _six_digit_floats(rng, n),
            "float_c": _six_digit_floats(rng, n),
            "float_d": _six_digit_floats(rng, n),
            "status": pa.array(rng.choice(STATUS_NAMES, size=n)),
        }
    )


def _icefalldb_cli() -> Path:
    """Locate the ``icefalldb`` binary, building the release CLI if needed."""
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

    # Build release CLI and return the resulting binary.
    subprocess.run(
        ["cargo", "build", "--release", "-p", "icefalldb-cli"],
        cwd=REPO_ROOT,
        check=True,
    )
    return release_cli


def run_cli(args: list[str]) -> None:
    """Run the release IcefallDB CLI and raise on failure."""
    cli = _icefalldb_cli()
    release_cli = REPO_ROOT / "target" / "release" / "icefalldb"
    if cli.resolve() != release_cli.resolve():
        try:
            subprocess.run(
                ["cargo", "build", "--release", "-p", "icefalldb-cli"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"failed to build release icefalldb CLI\n{exc.stderr or ''}"
            ) from exc
        cli = release_cli

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
            f"icefalldb CLI failed: {' '.join(args)}\n{exc.stderr or ''}"
        ) from exc


def copy_table(db: Path, src: str, dst: str) -> None:
    """Create ``dst`` as a deep copy of an existing table ``src``.

    The write lock file is intentionally excluded so the copied table can be
    optimized immediately without carrying a stale lock marker.
    """
    src_dir = db / src
    dst_dir = db / dst
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir, ignore=shutil.ignore_patterns("_write.lock"))


def create_icefalldb_table(db: Path, table_name: str, schema: dict) -> None:
    """Create a IcefallDB table from an inline schema dict."""
    schema_path = db / f"{table_name}_schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    try:
        subprocess.run(
            [
                str(_icefalldb_cli()),
                "create",
                "--schema",
                str(schema_path),
                str(db),
                table_name,
            ],
            check=True,
        )
    finally:
        schema_path.unlink(missing_ok=True)


def insert_parquet(db: Path, table_name: str, parquet_path: Path) -> None:
    """Insert a Parquet file into a IcefallDB table."""
    subprocess.run(
        [str(_icefalldb_cli()), "insert", str(db), table_name, str(parquet_path)],
        check=True,
    )


def _schema_for_events() -> dict:
    return {
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": True, "field_id": 1},
            {"name": "ts", "type": "timestamp[us]", "nullable": True, "field_id": 2},
            {"name": "category", "type": "utf8", "nullable": True, "field_id": 3},
            {"name": "value", "type": "float64", "nullable": True, "field_id": 4},
        ],
        "partition_by": ["category"],
        "sort": ["ts"],
        "row_group_target_rows": 1_000_000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": 4,
    }


def _schema_for_events_wide() -> dict:
    return {
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": True, "field_id": 1},
            {"name": "ts", "type": "timestamp[us]", "nullable": True, "field_id": 2},
            {"name": "int_a", "type": "int64", "nullable": True, "field_id": 3},
            {"name": "int_b", "type": "int64", "nullable": True, "field_id": 4},
            {"name": "int_c", "type": "int64", "nullable": True, "field_id": 5},
            {"name": "int_d", "type": "int64", "nullable": True, "field_id": 6},
            {"name": "float_a", "type": "float64", "nullable": True, "field_id": 7},
            {"name": "float_b", "type": "float64", "nullable": True, "field_id": 8},
            {"name": "float_c", "type": "float64", "nullable": True, "field_id": 9},
            {"name": "float_d", "type": "float64", "nullable": True, "field_id": 10},
            {"name": "status", "type": "utf8", "nullable": True, "field_id": 11},
        ],
        "partition_by": ["status"],
        "sort": ["status", "float_d", "int_d"],
        "row_group_target_rows": 5_000_000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": 11,
    }


def _schema_for_categories() -> dict:
    return {
        "schema_id": 1,
        "columns": [
            {"name": "category_id", "type": "int64", "nullable": True, "field_id": 1},
            {"name": "category_name", "type": "utf8", "nullable": True, "field_id": 2},
        ],
        "sort": ["category_id"],
        "row_group_target_rows": 10,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": 2,
    }


def create_sorted_events(db: Path) -> None:
    """Create ``events_sorted`` as a ts-sorted copy of ``events``."""
    copy_table(db, "events", "events_sorted")
    run_cli(["optimize", str(db), "events_sorted", "--sort", "ts"])


def create_indexed_events(db: Path) -> None:
    """Create ``events_indexed`` as a copy of ``events`` with a B-tree index on category."""
    copy_table(db, "events", "events_indexed")
    run_cli(["create-index", str(db), "events_indexed", "category"])


def create_clustered_events_wide(db: Path) -> None:
    """Create ``events_wide_clustered`` as an int_a-sorted copy of ``events_wide``."""
    copy_table(db, "events_wide", "events_wide_clustered")
    run_cli(["optimize", str(db), "events_wide_clustered", "--sort", "int_a"])


def _write_partitioned_files(
    table: pa.Table,
    partition_col: str,
    prefix: Path,
    sort_keys: list[str] | None,
    dictionary_columns: list[str] | None,
) -> list[Path]:
    """Write one Parquet file per distinct partition value.

    Each output file contains exactly one Parquet row group so IcefallDB's
    zero-copy fast path is used and each IcefallDB row group has valid sidecar
    column offsets.
    """
    unique_values = pc.unique(table.column(partition_col))
    paths: list[Path] = []
    for raw in unique_values.to_pylist():
        mask = pc.equal(table.column(partition_col), raw)
        slice_table = table.filter(mask)
        safe = str(raw).replace("/", "_")
        path = Path(f"{prefix}_{partition_col}={safe}.parquet")
        write_icefalldb_ready_parquet(
            path,
            slice_table,
            row_group_size=slice_table.num_rows,
            sort_keys=sort_keys,
            dictionary_columns=dictionary_columns,
        )
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate DataFusion benchmark datasets for IcefallDB."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=REPO_ROOT / "target" / "tmp" / "datafusion_bench_db",
        help="IcefallDB database directory (default: target/tmp/datafusion_bench_db)",
    )
    parser.add_argument(
        "--scale",
        choices=["1m", "10m"],
        default="1m",
        help="Dataset scale: 1m (events=1M, events_wide=10M) or "
        "10m (events=10M, events_wide=100M)",
    )
    args = parser.parse_args()

    n_events = 1_000_000 if args.scale == "1m" else 10_000_000
    n_wide = 10_000_000 if args.scale == "1m" else 100_000_000

    args.db.mkdir(parents=True, exist_ok=True)

    print(f"Generating DataFusion benchmark dataset at scale={args.scale}")
    print(f"  events:      {n_events:,} rows")
    print(f"  events_wide: {n_wide:,} rows")
    print("  categories:  10 rows")
    print(f"  db:          {args.db}")
    print()

    # Categories.
    print("Building categories...")
    cat = categories_table()
    cat_path = args.db / "categories.parquet"
    write_icefalldb_ready_parquet(
        cat_path,
        cat,
        row_group_size=10,
        sort_keys=["category_id"],
        dictionary_columns=["category_name"],
    )
    print(f"  wrote {cat_path}")

    # Events.
    print("Building events...")
    events = events_table(n_events)
    events_paths = _write_partitioned_files(
        events,
        "category",
        args.db / "events",
        sort_keys=["ts"],
        dictionary_columns=["category"],
    )
    print(f"  wrote {len(events_paths)} parquet file(s)")

    # Events wide.
    print("Building events_wide...")
    events_wide = events_wide_table(n_wide)
    wide_paths = _write_partitioned_files(
        events_wide,
        "status",
        args.db / "events_wide",
        sort_keys=["status", "float_d", "int_d"],
        dictionary_columns=["status"],
    )
    print(f"  wrote {len(wide_paths)} parquet file(s)")

    # Create IcefallDB tables and insert.
    print()
    print("Creating IcefallDB tables...")
    table_defs = [
        ("categories", _schema_for_categories(), [cat_path]),
        ("events", _schema_for_events(), events_paths),
        ("events_wide", _schema_for_events_wide(), wide_paths),
    ]
    for table_name, schema, parquet_paths in table_defs:
        create_icefalldb_table(args.db, table_name, schema)
        for parquet_path in parquet_paths:
            insert_parquet(args.db, table_name, parquet_path)
            parquet_path.unlink()
        print(f"  inserted {table_name}")

    print()
    print("Creating optimized variants...")
    create_sorted_events(args.db)
    create_indexed_events(args.db)
    create_clustered_events_wide(args.db)

    print()
    print("Optimized variants ready:")
    print("  events_sorted")
    print("  events_indexed")
    print("  events_wide_clustered")

    print()
    print(f"Benchmark database ready at {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
