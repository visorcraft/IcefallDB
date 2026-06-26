#!/usr/bin/env python3
"""Value-comparison benchmarks for IcefallDB vs common local-data competitors.

Run from the repo root with the Python venv active and the icefalldb CLI on PATH:

    python/python/.venv/bin/python python/benchmarks/compare.py

The script will build the icefalldb CLI if it is not found in target/release/icefalldb.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
ICEFALLDB_BIN = REPO_ROOT / "target" / "release" / "icefalldb"

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
    "row_group_target_bytes": 128 * 1024 * 1024,
    "dropped_columns": [],
    "max_field_id": 3,
}

TABLE_NAME = "products"


def ensure_icefalldb() -> Path:
    if not ICEFALLDB_BIN.exists():
        print("Building icefalldb CLI ...", flush=True)
        subprocess.run(
            ["cargo", "build", "--release", "-p", "icefalldb-cli"],
            cwd=REPO_ROOT,
            check=True,
        )
    return ICEFALLDB_BIN


def make_table(rows: int, offset: int = 0) -> pa.Table:
    ids = list(range(offset, offset + rows))
    values = [float(i) for i in ids]
    names = [f"name-{i}" for i in ids]
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("value", pa.float64(), nullable=False),
            pa.field("name", pa.utf8(), nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(values, type=pa.float64()),
            pa.array(names, type=pa.utf8()),
        ],
        schema=schema,
    )


def timed(func: Callable[[], None]) -> float:
    start = time.perf_counter()
    func()
    return time.perf_counter() - start


def bench(
    func: Callable[[], None], min_total_seconds: float = 0.5, max_iters: int = 20
) -> float:
    """Run func repeatedly and return the average elapsed time."""
    times: list[float] = []
    total = 0.0
    for _ in range(max_iters):
        elapsed = timed(func)
        times.append(elapsed)
        total += elapsed
        if total >= min_total_seconds:
            break
    return sum(times) / len(times)


def du(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


@dataclass
class Backend:
    name: str
    write: Callable[[Path, pa.Table], Path]
    read_all: Callable[[Path], Callable[[], None]]
    aggregate: Callable[[Path], Callable[[], None]]


def write_icefalldb(root: Path, table: pa.Table) -> Path:
    db, _ = prepare_icefalldb(root, table)
    return db


def prepare_icefalldb(root: Path, table: pa.Table) -> tuple[Path, Path]:
    bin_ = ensure_icefalldb()
    db = root / "icefalldb"
    db.mkdir(parents=True, exist_ok=True)
    schema_path = root / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    parquet_path = root / "input.parquet"
    import icefalldb

    icefalldb.write_icefalldb_ready_parquet(parquet_path, table)

    subprocess.run(
        [str(bin_), "create", "--schema", str(schema_path), str(db), TABLE_NAME],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [str(bin_), "insert", str(db), TABLE_NAME, str(parquet_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return db, parquet_path


def insert_icefalldb(db: Path, parquet_path: Path) -> None:
    bin_ = ensure_icefalldb()
    subprocess.run(
        [str(bin_), "insert", str(db), TABLE_NAME, str(parquet_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def create_empty_icefalldb(root: Path) -> Path:
    bin_ = ensure_icefalldb()
    db = root / "icefalldb"
    db.mkdir(parents=True, exist_ok=True)
    schema_path = root / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    subprocess.run(
        [str(bin_), "create", "--schema", str(schema_path), str(db), TABLE_NAME],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return db


def read_all_icefalldb(db: Path) -> Callable[[], None]:
    import icefalldb

    # Read directly through the native Arrow fast path. This is the IcefallDB
    # equivalent of ``pq.read_table`` for raw Parquet, bypassing DuckDB's
    # SQL tuple-materialization overhead.
    def fn() -> None:
        icefalldb.read_arrow_table(
            str(db), table=TABLE_NAME, verify_data_checksums=False
        )

    return fn


def aggregate_icefalldb(db: Path) -> Callable[[], None]:
    import icefalldb

    # Attach once and reuse the connection across iterations, matching the
    # competitor benchmarks that also keep a single DuckDB connection alive.
    con = icefalldb.attach(str(db), verify_data_checksums=False)

    def fn() -> None:
        con.sql(f"SELECT COUNT(*), AVG(value) FROM {TABLE_NAME}").fetchall()

    return fn


def aggregate_icefalldb_native(db: Path) -> Callable[[], None]:
    import icefalldb

    # Use the native Rust DataFusion query engine instead of DuckDB.
    con = icefalldb.attach(str(db), engine="datafusion", verify_data_checksums=False)

    def fn() -> None:
        con.sql(f"SELECT COUNT(*), AVG(value) FROM {TABLE_NAME}").fetchall()

    return fn


def write_parquet(root: Path, table: pa.Table) -> Path:
    path = root / "data.parquet"
    pq.write_table(table, path)
    return path


def read_all_parquet(path: Path) -> Callable[[], None]:
    def fn() -> None:
        pq.read_table(path)

    return fn


def aggregate_parquet(path: Path) -> Callable[[], None]:
    def fn() -> None:
        table = pq.read_table(path, columns=["value"])
        pc.mean(table.column("value"))

    return fn


def write_duckdb_parquet(root: Path, table: pa.Table) -> Path:
    path = root / "duckdb.parquet"
    con = duckdb.connect()
    con.register("src", table)
    con.execute(f"COPY (SELECT * FROM src) TO '{path}' (FORMAT PARQUET)")
    con.close()
    return path


def read_all_duckdb_parquet(path: Path) -> Callable[[], None]:
    con = duckdb.connect()

    def fn() -> None:
        con.execute(f"SELECT * FROM read_parquet('{path}')").fetchall()

    return fn


def aggregate_duckdb_parquet(path: Path) -> Callable[[], None]:
    con = duckdb.connect()

    def fn() -> None:
        con.execute(
            f"SELECT COUNT(*), AVG(value) FROM read_parquet('{path}')"
        ).fetchall()

    return fn


def write_sqlite(root: Path, table: pa.Table) -> Path:
    path = root / "data.sqlite"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (id INTEGER, value REAL, name TEXT)")
    rows = list(
        zip(
            table["id"].to_pylist(),
            table["value"].to_pylist(),
            table["name"].to_pylist(),
        )
    )
    con.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


def read_all_sqlite(path: Path) -> Callable[[], None]:
    con = sqlite3.connect(path)

    def fn() -> None:
        con.execute("SELECT * FROM t").fetchall()

    return fn


def aggregate_sqlite(path: Path) -> Callable[[], None]:
    con = sqlite3.connect(path)

    def fn() -> None:
        con.execute("SELECT COUNT(*), AVG(value) FROM t").fetchone()

    return fn


def write_csv(root: Path, table: pa.Table) -> Path:
    path = root / "data.csv"
    ids = table["id"].to_pylist()
    values = table["value"].to_pylist()
    names = table["name"].to_pylist()
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "value", "name"])
        for i in range(len(ids)):
            w.writerow([ids[i], values[i], names[i]])
    return path


def read_all_csv(path: Path) -> Callable[[], None]:
    def fn() -> None:
        with path.open(newline="") as f:
            list(csv.reader(f))

    return fn


def aggregate_csv(path: Path) -> Callable[[], None]:
    def fn() -> None:
        total = 0
        count = 0
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += float(row["value"])
                count += 1
        total / count

    return fn


BACKENDS = [
    Backend("IcefallDB", write_icefalldb, read_all_icefalldb, aggregate_icefalldb),
    Backend(
        "IcefallDB (native DataFusion)",
        write_icefalldb,
        read_all_icefalldb,
        aggregate_icefalldb_native,
    ),
    Backend("Parquet", write_parquet, read_all_parquet, aggregate_parquet),
    Backend(
        "DuckDB-on-Parquet",
        write_duckdb_parquet,
        read_all_duckdb_parquet,
        aggregate_duckdb_parquet,
    ),
    Backend("SQLite", write_sqlite, read_all_sqlite, aggregate_sqlite),
    Backend("CSV", write_csv, read_all_csv, aggregate_csv),
]


def run_size(rows: int) -> dict:
    table = make_table(rows)
    results: dict[str, dict] = {}
    for backend in BACKENDS:
        root = Path(tempfile.mkdtemp(prefix=f"bench_{rows}_{backend.name}_"))
        try:
            write_fn = backend.write
            write_path = write_fn(root, table)
            size = du(write_path)
            results[backend.name] = {"size_bytes": size, "size_per_row": size / rows}
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return results


def optimize_icefalldb(db: Path) -> None:
    subprocess.run(
        [
            str(ICEFALLDB_BIN),
            "optimize",
            str(db),
            TABLE_NAME,
            "--retain-snapshots",
            "1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def run_for_rows(rows: int, optimize: bool = False) -> dict:
    table = make_table(rows)
    results: dict[str, dict] = {}
    for backend in BACKENDS:
        root = Path(tempfile.mkdtemp(prefix=f"bench_{rows}_{backend.name}_"))
        try:
            if backend.name.startswith("IcefallDB"):
                # The IcefallDB write path has a separate create step. Create the
                # table once for read/size measurement, and for write throughput
                # create a fresh empty table per iteration (not timed) and time
                # only the insert.
                read_db, parquet_path = prepare_icefalldb(root, table)
                if optimize:
                    optimize_icefalldb(read_db)
                write_path = read_db

                def bench_write() -> None:
                    tmp = Path(tempfile.mkdtemp(prefix=f"benchw_{rows}_IcefallDB_"))
                    try:
                        db = create_empty_icefalldb(tmp)
                        insert_icefalldb(db, parquet_path)
                        if optimize:
                            optimize_icefalldb(db)
                    finally:
                        shutil.rmtree(tmp, ignore_errors=True)

            else:
                write_path = backend.write(root, table)

                def bench_write() -> None:
                    tmp = Path(
                        tempfile.mkdtemp(prefix=f"benchw_{rows}_{backend.name}_")
                    )
                    try:
                        backend.write(tmp, table)
                    finally:
                        shutil.rmtree(tmp, ignore_errors=True)

            write_time = bench(bench_write, min_total_seconds=1.0, max_iters=7)
            read_fn = backend.read_all(write_path)
            read_time = bench(read_fn, min_total_seconds=0.5, max_iters=10)
            agg_fn = backend.aggregate(write_path)
            agg_time = bench(agg_fn, min_total_seconds=0.5, max_iters=10)
            size = du(write_path)
            results[backend.name] = {
                "write_s": write_time,
                "write_rows_per_s": rows / write_time,
                "read_all_s": read_time,
                "read_all_rows_per_s": rows / read_time,
                "aggregate_s": agg_time,
                "size_bytes": size,
                "size_per_row": size / rows,
            }
        finally:
            shutil.rmtree(root, ignore_errors=True)
    return results


def print_table(rows_list: list[int], data: dict[int, dict]) -> None:
    print("\n## Write throughput (rows/s)\n")
    header = f"{'Backend':<22}" + "".join(f"{n:>15}" for n in rows_list)
    print(header)
    print("-" * len(header))
    for backend in [b.name for b in BACKENDS]:
        line = f"{backend:<22}"
        for n in rows_list:
            line += f"{data[n][backend]['write_rows_per_s']:>15,.0f}"
        print(line)

    print("\n## Read-all latency (ms)\n")
    print(header)
    print("-" * len(header))
    for backend in [b.name for b in BACKENDS]:
        line = f"{backend:<22}"
        for n in rows_list:
            line += f"{data[n][backend]['read_all_s'] * 1000:>15,.1f}"
        print(line)

    print("\n## Aggregate latency (ms)\n")
    print(header)
    print("-" * len(header))
    for backend in [b.name for b in BACKENDS]:
        line = f"{backend:<22}"
        for n in rows_list:
            line += f"{data[n][backend]['aggregate_s'] * 1000:>15,.1f}"
        print(line)

    print("\n## Storage size (bytes/row)\n")
    print(header)
    print("-" * len(header))
    for backend in [b.name for b in BACKENDS]:
        line = f"{backend:<22}"
        for n in rows_list:
            line += f"{data[n][backend]['size_per_row']:>15,.1f}"
        print(line)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Value-comparison benchmarks for IcefallDB vs local-data competitors."
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run 'icefalldb optimize' on IcefallDB tables before measuring read/size/aggregate.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "python"))
    sizes = [10_000, 100_000, 1_000_000]
    print("Running value-comparison benchmarks ...", flush=True)
    data: dict[int, dict] = {}
    for rows in sizes:
        print(f"\nBenchmarking {rows:,} rows ...", flush=True)
        data[rows] = run_for_rows(rows, optimize=args.optimize)
    print_table(sizes, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
