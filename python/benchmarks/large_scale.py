#!/usr/bin/env python3
"""Large-scale benchmarks: 10M and 100M rows.

Compares IcefallDB, raw Parquet, DuckDB-on-Parquet, and SQLite on ingest,
full-table read, and storage size.

Run from the repo root with the Python venv active and the icefalldb CLI on PATH:

    source python/.venv/bin/activate
    python python/benchmarks/large_scale.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import time
import shutil
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
ICEFALLDB_BIN = REPO_ROOT / "target" / "release" / "icefalldb"

sys.path.insert(0, str(REPO_ROOT / "python"))
import icefalldb  # noqa: E402

SCHEMA = {
    "schema_id": 1,
    "columns": [
        {"name": "id", "type": "int64", "nullable": False, "field_id": 1},
        {"name": "value", "type": "float64", "nullable": False, "field_id": 2},
        {"name": "name", "type": "utf8", "nullable": False, "field_id": 3},
    ],
    "partition_by": None,
    "sort": None,
    "row_group_target_rows": 1_000_000,
    "row_group_target_bytes": 512 * 1024 * 1024,
    "dropped_columns": [],
    "max_field_id": 3,
}

TABLE_NAME = "products"
BATCH_SIZE = 1_000_000


def ensure_icefalldb() -> Path:
    if not ICEFALLDB_BIN.exists():
        print("Building icefalldb CLI ...", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "-p", "icefalldb-cli"],
            cwd=REPO_ROOT,
            check=True,
        )
    return ICEFALLDB_BIN


def pa_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("name", pa.utf8(), nullable=False),
        ]
    )


def make_batch(rows: int, offset: int) -> pa.RecordBatch:
    ids = list(range(offset, offset + rows))
    values = [float(i) for i in ids]
    names = [f"name-{i}" for i in ids]
    return pa.RecordBatch.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(values, type=pa.float64()),
            pa.array(names, type=pa.utf8()),
        ],
        schema=pa_schema(),
    )


def write_input_parquet(path: Path, total_rows: int) -> None:
    """Write a Snappy-compressed Parquet file in 1M-row batches."""
    writer = pq.ParquetWriter(
        path,
        pa_schema(),
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )
    for offset in range(0, total_rows, BATCH_SIZE):
        batch = make_batch(min(BATCH_SIZE, total_rows - offset), offset)
        writer.write_batch(batch)
    writer.close()


def dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def create_empty_table(db: Path) -> None:
    bin_ = ensure_icefalldb()
    db.mkdir(parents=True, exist_ok=True)
    schema_path = db.parent / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    subprocess.run(
        [str(bin_), "create", "--schema", str(schema_path), str(db), TABLE_NAME],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def insert_parquet(db: Path, parquet_path: Path) -> None:
    bin_ = ensure_icefalldb()
    subprocess.run(
        [str(bin_), "insert", str(db), TABLE_NAME, str(parquet_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def compact_table(db: Path) -> None:
    bin_ = ensure_icefalldb()
    subprocess.run(
        [str(bin_), "compact", str(db), TABLE_NAME],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def bench_icefalldb(root: Path, parquet_path: Path, rows: int) -> dict:
    db = root / "icefalldb"
    create_empty_table(db)

    t0 = time.perf_counter()
    insert_parquet(db, parquet_path)
    insert_s = time.perf_counter() - t0

    size_before = dir_size(db)

    t0 = time.perf_counter()
    table = icefalldb.read_arrow_table(
        str(db), table=TABLE_NAME, verify_data_checksums=False
    )
    read_s = time.perf_counter() - t0
    assert table.num_rows == rows

    t0 = time.perf_counter()
    compact_table(db)
    compact_s = time.perf_counter() - t0
    size_after = dir_size(db)

    return {
        "write_s": insert_s,
        "write_rows_per_s": rows / insert_s,
        "read_all_s": read_s,
        "compact_s": compact_s,
        "bytes_per_row_before_compact": size_before / rows,
        "bytes_per_row_after_compact": size_after / rows,
    }


def bench_raw_parquet(root: Path, parquet_path: Path, rows: int) -> dict:
    out = root / "raw.parquet"
    schema = pa_schema()
    reader = pq.ParquetFile(parquet_path)
    writer = pq.ParquetWriter(
        out,
        schema,
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
    )
    t0 = time.perf_counter()
    for batch in reader.iter_batches(batch_size=BATCH_SIZE):
        writer.write_batch(batch)
    writer.close()
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    table = pq.read_table(out)
    read_s = time.perf_counter() - t0
    assert table.num_rows == rows

    return {
        "write_s": write_s,
        "write_rows_per_s": rows / write_s,
        "read_all_s": read_s,
        "bytes_per_row": out.stat().st_size / rows,
    }


def bench_duckdb_parquet(root: Path, parquet_path: Path, rows: int) -> dict:
    out = root / "duckdb.parquet"
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL parquet; LOAD parquet;")

    t0 = time.perf_counter()
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{parquet_path}')) TO '{out}' (FORMAT PARQUET)"
    )
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    con.execute(f"SELECT * FROM read_parquet('{out}')").to_arrow_table()
    read_s = time.perf_counter() - t0
    con.close()

    return {
        "write_s": write_s,
        "write_rows_per_s": rows / write_s,
        "read_all_s": read_s,
        "bytes_per_row": out.stat().st_size / rows,
    }


def bench_sqlite(root: Path, parquet_path: Path, rows: int) -> dict:
    db_path = root / "sqlite.db"
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute(f"ATTACH '{db_path}' AS sq (TYPE SQLITE)")

    t0 = time.perf_counter()
    con.execute(
        f"CREATE TABLE sq.products AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    con.execute("SELECT * FROM sq.products").to_arrow_table()
    read_s = time.perf_counter() - t0
    con.close()

    return {
        "write_s": write_s,
        "write_rows_per_s": rows / write_s,
        "read_all_s": read_s,
        "bytes_per_row": db_path.stat().st_size / rows,
    }


def bench_csv(root: Path, parquet_path: Path, rows: int) -> dict:
    out = root / "data.csv"
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL parquet; LOAD parquet;")

    t0 = time.perf_counter()
    con.execute(
        f"COPY (SELECT * FROM read_parquet('{parquet_path}')) TO '{out}' (FORMAT CSV, HEADER TRUE)"
    )
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    with out.open(newline="") as f:
        list(csv.reader(f))
    read_s = time.perf_counter() - t0
    con.close()

    return {
        "write_s": write_s,
        "write_rows_per_s": rows / write_s,
        "read_all_s": read_s,
        "bytes_per_row": out.stat().st_size / rows,
    }


def bench(rows: int) -> dict:
    print(f"\nBenchmarking {rows:,} rows ...", flush=True)
    root = Path(tempfile.mkdtemp(prefix=f"large_scale_{rows}_"))
    try:
        parquet_path = root / "input.parquet"
        t0 = time.perf_counter()
        write_input_parquet(parquet_path, rows)
        parquet_write_s = time.perf_counter() - t0
        print(f"  input parquet written: {parquet_write_s:.2f}s", flush=True)

        print("  IcefallDB ...", flush=True)
        icefalldb_result = bench_icefalldb(root, parquet_path, rows)
        print(
            f"    write {icefalldb_result['write_rows_per_s']:,.0f} rows/s, "
            f"read {icefalldb_result['read_all_s']:.2f}s, "
            f"size {icefalldb_result['bytes_per_row_before_compact']:.2f} bytes/row",
            flush=True,
        )

        print("  raw Parquet ...", flush=True)
        raw_result = bench_raw_parquet(root, parquet_path, rows)
        print(
            f"    write {raw_result['write_rows_per_s']:,.0f} rows/s, "
            f"read {raw_result['read_all_s']:.2f}s, "
            f"size {raw_result['bytes_per_row']:.2f} bytes/row",
            flush=True,
        )

        print("  DuckDB-on-Parquet ...", flush=True)
        duckdb_result = bench_duckdb_parquet(root, parquet_path, rows)
        print(
            f"    write {duckdb_result['write_rows_per_s']:,.0f} rows/s, "
            f"read {duckdb_result['read_all_s']:.2f}s, "
            f"size {duckdb_result['bytes_per_row']:.2f} bytes/row",
            flush=True,
        )

        print("  SQLite ...", flush=True)
        sqlite_result = bench_sqlite(root, parquet_path, rows)
        print(
            f"    write {sqlite_result['write_rows_per_s']:,.0f} rows/s, "
            f"read {sqlite_result['read_all_s']:.2f}s, "
            f"size {sqlite_result['bytes_per_row']:.2f} bytes/row",
            flush=True,
        )

        print("  CSV ...", flush=True)
        csv_result = bench_csv(root, parquet_path, rows)
        print(
            f"    write {csv_result['write_rows_per_s']:,.0f} rows/s, "
            f"read {csv_result['read_all_s']:.2f}s, "
            f"size {csv_result['bytes_per_row']:.2f} bytes/row",
            flush=True,
        )

        return {
            "input_parquet_write_s": parquet_write_s,
            "icefalldb": icefalldb_result,
            "raw_parquet": raw_result,
            "duckdb_parquet": duckdb_result,
            "sqlite": sqlite_result,
            "csv": csv_result,
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    results: dict[int, dict] = {}
    for rows in [10_000_000, 100_000_000]:
        results[rows] = bench(rows)

    print("\n=== Large-scale benchmark results ===")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
