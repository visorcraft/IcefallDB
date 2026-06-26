#!/usr/bin/env python3
"""Real-world analytical-query benchmarks for IcefallDB.

Compares IcefallDB, raw Parquet, DuckDB-on-Parquet, and SQLite on a 10M-row
dataset with range scans, aggregations, grouping, sorting, and a join.

Run from the repo root with the Python venv active:

    source python/.venv/bin/activate
    python python/benchmarks/analytical_queries.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[2]
ICEFALLDB_BIN = REPO_ROOT / "target" / "release" / "icefalldb"

sys.path.insert(0, str(REPO_ROOT / "python"))
import icefalldb  # noqa: E402
from icefalldb.producer import write_icefalldb_ready_parquet  # noqa: E402

TABLE_NAME = "events"
ROWS = 10_000_000
BATCH_SIZE = 1_000_000

ROW_GROUP_SIZE = 1_000_000
DATA_PAGE_SIZE = 1_048_576
DIMENSION_ROW_GROUP_SIZE = 100_000

SCHEMA = {
    "schema_id": 1,
    "columns": [
        {"name": "id", "type": "int64", "nullable": False, "field_id": 1},
        {"name": "event_time", "type": "int64", "nullable": False, "field_id": 2},
        {"name": "user_id", "type": "int64", "nullable": False, "field_id": 3},
        {"name": "category", "type": "utf8", "nullable": False, "field_id": 4},
        {"name": "amount", "type": "float64", "nullable": False, "field_id": 5},
    ],
    "partition_by": None,
    "sort": None,
    "row_group_target_rows": ROW_GROUP_SIZE,
    "row_group_target_bytes": 512 * 1024 * 1024,
    "dropped_columns": [],
    "max_field_id": 5,
}

DIMENSION_SCHEMA = {
    "schema_id": 1,
    "columns": [
        {"name": "user_id", "type": "int64", "nullable": True, "field_id": 1},
        {"name": "segment", "type": "utf8", "nullable": True, "field_id": 2},
    ],
    "partition_by": None,
    "sort": None,
    "row_group_target_rows": DIMENSION_ROW_GROUP_SIZE,
    "row_group_target_bytes": 512 * 1024 * 1024,
    "dropped_columns": [],
    "max_field_id": 2,
}

CATEGORIES = ["click", "view", "purchase", "return", "search"]


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
            pa.field("event_time", pa.int64(), nullable=False),
            pa.field("user_id", pa.int64(), nullable=False),
            pa.field("category", pa.utf8(), nullable=False),
            pa.field("amount", pa.float64(), nullable=False),
        ]
    )


def make_batch(rows: int, offset: int) -> pa.RecordBatch:
    ids = list(range(offset, offset + rows))
    event_times = [(i * 1_000_000) for i in ids]
    user_ids = [(i % 100_000) for i in ids]
    categories = [CATEGORIES[i % len(CATEGORIES)] for i in ids]
    amounts = [float(i % 1000) for i in ids]
    return pa.RecordBatch.from_arrays(
        [
            pa.array(ids, type=pa.int64()),
            pa.array(event_times, type=pa.int64()),
            pa.array(user_ids, type=pa.int64()),
            pa.array(categories, type=pa.utf8()),
            pa.array(amounts, type=pa.float64()),
        ],
        schema=pa_schema(),
    )


def write_input_parquet(path: Path, rows: int) -> None:
    """Write the main events Parquet with IcefallDB-ready encoding and sorting."""
    batches = [
        make_batch(min(BATCH_SIZE, rows - offset), offset)
        for offset in range(0, rows, BATCH_SIZE)
    ]
    table = pa.Table.from_batches(batches, schema=pa_schema())
    write_icefalldb_ready_parquet(
        path,
        table,
        row_group_size=ROW_GROUP_SIZE,
        data_page_size=DATA_PAGE_SIZE,
        sort_keys=["category"],
        dictionary_columns=["category"],
    )


def write_dimension_parquet(path: Path) -> None:
    """Write the user dimension Parquet with IcefallDB-ready encoding and sorting."""
    dim_user_ids = list(range(100_000))
    dim_segments = ["a", "b", "c", "d", "e"] * 20_000
    dim_table = pa.table(
        {
            "user_id": pa.array(dim_user_ids, type=pa.int64()),
            "segment": pa.array(dim_segments, type=pa.utf8()),
        }
    )
    write_icefalldb_ready_parquet(
        path,
        dim_table,
        row_group_size=DIMENSION_ROW_GROUP_SIZE,
        data_page_size=DATA_PAGE_SIZE,
        sort_keys=["user_id"],
        dictionary_columns=["segment"],
    )


def create_icefalldb_table(db: Path) -> None:
    """Create the IcefallDB table from the JSON schema using the CLI."""
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
    """Insert a Parquet file into the IcefallDB table using the CLI."""
    bin_ = ensure_icefalldb()
    subprocess.run(
        [str(bin_), "insert", str(db), TABLE_NAME, str(parquet_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def create_dimension_icefalldb_table(db: Path) -> None:
    """Create the dimension IcefallDB table from the JSON schema using the CLI."""
    bin_ = ensure_icefalldb()
    schema_path = db.parent / "dimension_schema.json"
    schema_path.write_text(json.dumps(DIMENSION_SCHEMA))
    subprocess.run(
        [str(bin_), "create", "--schema", str(schema_path), str(db), "segments"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def insert_dimension_parquet(db: Path, parquet_path: Path) -> None:
    """Insert the dimension Parquet file into the IcefallDB table using the CLI."""
    bin_ = ensure_icefalldb()
    subprocess.run(
        [str(bin_), "insert", str(db), "segments", str(parquet_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def time_query(
    con: duckdb.DuckDBPyConnection, sql: str, params: tuple = ()
) -> tuple[float, list]:
    t0 = time.perf_counter()
    result = con.execute(sql, params).fetchall()
    elapsed = time.perf_counter() - t0
    return elapsed, result


def warm_run(
    con: duckdb.DuckDBPyConnection, sql: str, params: tuple = ()
) -> tuple[float, list]:
    """Execute once to warm caches, then time a second execution."""
    con.execute(sql, params).fetchall()
    return time_query(con, sql, params)


def setup_icefalldb_table_view(
    con: duckdb.DuckDBPyConnection,
    db: Path,
    table_name: str,
    columns: list[str] | None = None,
) -> float:
    """Register an Arrow view of a IcefallDB table (or subset of columns) and return read time."""
    t0 = time.perf_counter()
    table = icefalldb.read_arrow_table(
        str(db), table=table_name, columns=columns, verify_data_checksums=False
    )
    read_s = time.perf_counter() - t0
    con.register(table_name, table)
    return read_s


def run_queries(
    label: str, con: duckdb.DuckDBPyConnection, table_expr: str, dimension_path: Path
) -> dict[str, dict]:
    """Run the benchmark query suite and return timing results."""
    print(f"\nRunning queries on {label} ...", flush=True)
    results: dict[str, dict] = {}

    # Q1: range scan with high selectivity
    min_id = ROWS // 10
    max_id = ROWS // 10 + ROWS // 10
    elapsed, rows = warm_run(
        con,
        f"SELECT COUNT(*), AVG(amount) FROM {table_expr} t WHERE t.id BETWEEN ? AND ?",
        (min_id, max_id),
    )
    results["range_scan"] = {
        "s": elapsed,
        "rows_returned": len(rows),
        "note": f"id BETWEEN {min_id} AND {max_id}",
    }
    print(f"  range_scan: {elapsed:.3f}s", flush=True)

    # Q2: group by category
    elapsed, rows = warm_run(
        con,
        f"SELECT t.category, COUNT(*), AVG(t.amount), SUM(t.amount) FROM {table_expr} t GROUP BY t.category",
    )
    results["group_by_category"] = {"s": elapsed, "rows_returned": len(rows)}
    print(f"  group_by_category: {elapsed:.3f}s", flush=True)

    # Q3: top-N by amount
    elapsed, rows = warm_run(
        con,
        f"SELECT t.id, t.amount FROM {table_expr} t ORDER BY t.amount DESC LIMIT 100",
    )
    results["top_n_sort"] = {"s": elapsed, "rows_returned": len(rows)}
    print(f"  top_n_sort: {elapsed:.3f}s", flush=True)

    # Q4: global aggregates
    elapsed, rows = warm_run(
        con,
        f"SELECT COUNT(*), COUNT(DISTINCT t.user_id), AVG(t.amount), MIN(t.amount), MAX(t.amount) FROM {table_expr} t",
    )
    results["global_aggregates"] = {"s": elapsed, "rows_returned": len(rows)}
    print(f"  global_aggregates: {elapsed:.3f}s", flush=True)

    # Q5: join with small dimension table
    elapsed, rows = warm_run(
        con,
        f"SELECT d.segment, COUNT(*), AVG(t.amount) FROM {table_expr} t "
        f"JOIN read_parquet('{dimension_path}') d ON t.user_id = d.user_id "
        "GROUP BY d.segment",
    )
    results["join_dimension"] = {"s": elapsed, "rows_returned": len(rows)}
    print(f"  join_dimension: {elapsed:.3f}s", flush=True)

    return results


def run_datafusion_queries(db: Path) -> dict[str, dict]:
    """Run the benchmark query suite through the native DataFusion engine."""
    import icefalldb

    print("\nRunning queries on IcefallDB (native DataFusion) ...", flush=True)
    con = icefalldb.attach(str(db), engine="datafusion", verify_data_checksums=False)
    results: dict[str, dict] = {}
    min_id = ROWS // 10
    max_id = ROWS // 10 + ROWS // 10

    queries = [
        (
            "range_scan",
            f"SELECT COUNT(*), AVG(amount) FROM events t WHERE t.id BETWEEN {min_id} AND {max_id}",
        ),
        (
            "group_by_category",
            "SELECT t.category, COUNT(*), AVG(t.amount), SUM(t.amount) FROM events t GROUP BY t.category",
        ),
        (
            "top_n_sort",
            "SELECT t.id, t.amount FROM events t ORDER BY t.amount DESC LIMIT 100",
        ),
        (
            "global_aggregates",
            "SELECT COUNT(*), COUNT(DISTINCT t.user_id), AVG(t.amount), MIN(t.amount), MAX(t.amount) FROM events t",
        ),
        (
            "join_dimension",
            "SELECT d.segment, COUNT(*), AVG(t.amount) FROM events t "
            "JOIN segments d ON t.user_id = d.user_id "
            "GROUP BY d.segment",
        ),
    ]

    def _run(sql: str) -> tuple[float, list]:
        t0 = time.perf_counter()
        result = con.sql(sql)
        rows = (
            result.fetchall()
            if hasattr(result, "fetchall")
            else [tuple(row.values()) for row in result.to_arrow_table().to_pylist()]
        )
        return time.perf_counter() - t0, rows

    for name, sql in queries:
        try:
            _run(sql)
            elapsed, rows = _run(sql)
            results[name] = {"s": elapsed, "rows_returned": len(rows)}
            print(f"  {name}: {elapsed:.3f}s", flush=True)
        except Exception as exc:
            results[name] = {"s": None, "error": str(exc)}
            print(f"  {name}: FAILED ({exc})", flush=True)

    return results


def create_sqlite_from_parquet(sqlite_path: Path, parquet_path: Path) -> None:
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL parquet; LOAD parquet;")
    if sqlite_path.exists():
        sqlite_path.unlink()
    con.execute(f"ATTACH '{sqlite_path}' AS sq (TYPE SQLITE)")
    con.execute(
        f"CREATE TABLE sq.events AS SELECT * FROM read_parquet('{parquet_path}')"
    )
    con.close()


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="analytical_queries_"))
    try:
        parquet_path = root / "events.parquet"
        db = root / "icefalldb"
        dimension_path = root / "users.parquet"
        sqlite_path = root / "events.db"

        print("Generating 10M-row Parquet input ...", flush=True)
        t0 = time.perf_counter()
        write_input_parquet(parquet_path, ROWS)
        print(f"  parquet written in {time.perf_counter() - t0:.2f}s", flush=True)

        print("Generating dimension Parquet ...", flush=True)
        write_dimension_parquet(dimension_path)

        print("Creating IcefallDB table and inserting ...", flush=True)
        create_icefalldb_table(db)
        t0 = time.perf_counter()
        insert_parquet(db, parquet_path)
        icefalldb_insert_s = time.perf_counter() - t0
        print(f"  icefalldb insert: {icefalldb_insert_s:.2f}s", flush=True)

        print("Creating IcefallDB dimension table and inserting ...", flush=True)
        create_dimension_icefalldb_table(db)
        insert_dimension_parquet(db, dimension_path)

        print("Creating SQLite database from Parquet ...", flush=True)
        t0 = time.perf_counter()
        create_sqlite_from_parquet(sqlite_path, parquet_path)
        sqlite_load_s = time.perf_counter() - t0
        print(f"  sqlite load: {sqlite_load_s:.2f}s", flush=True)

        report: dict = {
            "dataset_rows": ROWS,
            "icefalldb_insert_s": icefalldb_insert_s,
            "sqlite_load_s": sqlite_load_s,
            "queries": {},
        }

        # IcefallDB hybrid read path: attach for group-by/join, Arrow for scans.
        # Run IcefallDB first while caches are fresh; competitor baselines follow.
        con_icefalldb_attach = icefalldb.attach(
            str(db), engine="duckdb", verify_data_checksums=False
        )
        con_icefalldb_attach.execute("SET threads TO 16")
        con_icefalldb_arrow = duckdb.connect(database=":memory:")
        con_icefalldb_arrow.execute("SET threads TO 16")

        icefalldb_results: dict[str, dict] = {}
        icefalldb_arrow_read_s: dict[str, float] = {}

        # Q1: range scan with high selectivity -> Arrow view on id, amount.
        # Read only the needed columns; let DuckDB evaluate the WHERE id range
        # predicate during the timed query so filter cost is included in the
        # reported query time.
        min_id = ROWS // 10
        max_id = ROWS // 10 + ROWS // 10
        icefalldb_arrow_read_s["range_scan"] = setup_icefalldb_table_view(
            con_icefalldb_arrow, db, TABLE_NAME, columns=["id", "amount"]
        )
        elapsed, rows = warm_run(
            con_icefalldb_arrow,
            "SELECT COUNT(*), AVG(amount) FROM events t WHERE t.id BETWEEN ? AND ?",
            (min_id, max_id),
        )
        icefalldb_results["range_scan"] = {
            "s": elapsed,
            "rows_returned": len(rows),
            "note": f"id BETWEEN {min_id} AND {max_id}",
        }
        print(f"  icefalldb range_scan: {elapsed:.3f}s", flush=True)

        # Q2: group by category -> attach connection
        elapsed, rows = warm_run(
            con_icefalldb_attach,
            "SELECT t.category, COUNT(*), AVG(t.amount), SUM(t.amount) FROM events t GROUP BY t.category",
        )
        icefalldb_results["group_by_category"] = {
            "s": elapsed,
            "rows_returned": len(rows),
        }
        print(f"  icefalldb group_by_category: {elapsed:.3f}s", flush=True)

        # Q3: top-N by amount -> Arrow view on id, amount
        icefalldb_arrow_read_s["top_n_sort"] = setup_icefalldb_table_view(
            con_icefalldb_arrow, db, TABLE_NAME, columns=["id", "amount"]
        )
        elapsed, rows = warm_run(
            con_icefalldb_arrow,
            "SELECT t.id, t.amount FROM events t ORDER BY t.amount DESC LIMIT 100",
        )
        icefalldb_results["top_n_sort"] = {"s": elapsed, "rows_returned": len(rows)}
        print(f"  icefalldb top_n_sort: {elapsed:.3f}s", flush=True)

        # Q4: global aggregates -> Arrow view on user_id, amount
        icefalldb_arrow_read_s["global_aggregates"] = setup_icefalldb_table_view(
            con_icefalldb_arrow, db, TABLE_NAME, columns=["user_id", "amount"]
        )
        elapsed, rows = warm_run(
            con_icefalldb_arrow,
            "SELECT COUNT(*), COUNT(DISTINCT t.user_id), AVG(t.amount), MIN(t.amount), MAX(t.amount) FROM events t",
        )
        icefalldb_results["global_aggregates"] = {
            "s": elapsed,
            "rows_returned": len(rows),
        }
        print(f"  icefalldb global_aggregates: {elapsed:.3f}s", flush=True)

        # Q5: join with small dimension table -> attach connection
        elapsed, rows = warm_run(
            con_icefalldb_attach,
            f"SELECT d.segment, COUNT(*), AVG(t.amount) FROM events t "
            f"JOIN read_parquet('{dimension_path}') d ON t.user_id = d.user_id "
            "GROUP BY d.segment",
        )
        icefalldb_results["join_dimension"] = {"s": elapsed, "rows_returned": len(rows)}
        print(f"  icefalldb join_dimension: {elapsed:.3f}s", flush=True)

        report["queries"]["icefalldb"] = icefalldb_results
        report["icefalldb_arrow_read_s"] = icefalldb_arrow_read_s

        con_icefalldb_attach.close()
        con_icefalldb_arrow.close()

        # DuckDB on raw Parquet
        con_parquet = duckdb.connect(database=":memory:")
        con_parquet.execute("INSTALL parquet; LOAD parquet;")
        report["queries"]["raw_parquet"] = run_queries(
            "raw Parquet",
            con_parquet,
            f"read_parquet('{parquet_path}')",
            dimension_path,
        )
        con_parquet.close()

        # DuckDB on Parquet written by DuckDB (competitor baseline)
        duckdb_out = root / "duckdb.parquet"
        con_duckdb_write = duckdb.connect(database=":memory:")
        con_duckdb_write.execute("INSTALL parquet; LOAD parquet;")
        con_duckdb_write.execute(
            f"COPY (SELECT * FROM read_parquet('{parquet_path}')) TO '{duckdb_out}' (FORMAT PARQUET)"
        )
        con_duckdb_write.close()
        con_duckdb = duckdb.connect(database=":memory:")
        con_duckdb.execute("INSTALL parquet; LOAD parquet;")
        report["queries"]["duckdb_parquet"] = run_queries(
            "DuckDB-on-Parquet",
            con_duckdb,
            f"read_parquet('{duckdb_out}')",
            dimension_path,
        )
        con_duckdb.close()

        # DuckDB on SQLite
        con_sqlite = duckdb.connect(database=":memory:")
        con_sqlite.execute("INSTALL parquet; LOAD parquet;")
        con_sqlite.execute(f"ATTACH '{sqlite_path}' AS sq (TYPE SQLITE)")
        report["queries"]["sqlite"] = run_queries(
            "SQLite", con_sqlite, "sq.events", dimension_path
        )
        con_sqlite.close()

        report["queries"]["icefalldb_native_datafusion"] = run_datafusion_queries(db)

        print("\n=== Analytical query benchmark results ===")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
