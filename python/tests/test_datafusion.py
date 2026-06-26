"""Differential tests for the DataFusion query engine vs DuckDB."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import icefalldb
from icefalldb import IcefallDBError

ICEFALLDB_DEBUG = Path(__file__).resolve().parents[2] / "target" / "debug" / "icefalldb"


def _run_icefalldb(*args):
    if not ICEFALLDB_DEBUG.exists():
        pytest.skip(
            f"icefalldb CLI not found at {ICEFALLDB_DEBUG}; run cargo build first"
        )
    result = subprocess.run(
        [str(ICEFALLDB_DEBUG), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise IcefallDBError(f"icefalldb {' '.join(args)} failed: {result.stderr}")
    return result


def _make_table(db_path: Path, table: str, arrow_table: pa.Table) -> None:
    schema = {
        "schema_id": 1,
        "columns": [
            {
                "name": field.name,
                "type": _arrow_type_to_icefalldb(field.type),
                "nullable": field.nullable,
            }
            for field in arrow_table.schema
        ],
        "row_group_target_rows": 1_000_000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": 0,
    }
    schema_path = db_path / f"{table}_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    _run_icefalldb("create", str(db_path), table, "--schema", str(schema_path))

    parquet_path = db_path / f"{table}.parquet"
    pq.write_table(arrow_table, parquet_path)
    _run_icefalldb("insert", str(db_path), table, str(parquet_path))


def _arrow_type_to_icefalldb(arrow_type: pa.DataType) -> str:
    if pa.types.is_int64(arrow_type):
        return "int64"
    if pa.types.is_float64(arrow_type):
        return "float64"
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "utf8"
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp[us]"
    raise ValueError(f"unsupported Arrow type for IcefallDB test: {arrow_type}")


def test_datafusion_count_min_max_match_duckdb():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        arrow_table = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
            },
            schema=pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("value", pa.float64(), nullable=True),
                ]
            ),
        )
        _make_table(db, "events", arrow_table)

        duck = icefalldb.attach(str(db), tables=["events"], engine="duckdb")
        fusion = icefalldb.attach(str(db), tables=["events"], engine="datafusion")

        queries = [
            "SELECT COUNT(*) AS cnt FROM events",
            "SELECT MIN(id) AS mn, MAX(id) AS mx FROM events",
            "SELECT MIN(value) AS mn, MAX(value) AS mx FROM events",
        ]
        for sql in queries:
            duck_rows = duck.execute(sql).fetchall()
            fusion_rows = fusion.sql(sql).fetchall()
            assert duck_rows == fusion_rows, f"{sql}: {duck_rows} != {fusion_rows}"


def test_datafusion_filtered_query_match_duckdb():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        arrow_table = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["a", "b", "c", "d", "e"],
            },
            schema=pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("name", pa.utf8(), nullable=True),
                ]
            ),
        )
        _make_table(db, "products", arrow_table)

        duck = icefalldb.attach(str(db), tables=["products"], engine="duckdb")
        fusion = icefalldb.attach(str(db), tables=["products"], engine="datafusion")

        queries = [
            "SELECT id FROM products WHERE name = 'c'",
            "SELECT COUNT(*) AS cnt FROM products WHERE id > 2",
            "SELECT id, name FROM products WHERE id BETWEEN 2 AND 4 ORDER BY id",
        ]
        for sql in queries:
            duck_rows = duck.execute(sql).fetchall()
            fusion_rows = fusion.sql(sql).fetchall()
            assert duck_rows == fusion_rows, f"{sql}: {duck_rows} != {fusion_rows}"


def test_default_attach_is_icefalldb_router():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        arrow_table = pa.table(
            {"id": [1, 2, 3]},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        )
        _make_table(db, "numbers", arrow_table)

        con = icefalldb.attach(str(db), tables=["numbers"])
        assert type(con).__name__ == "IcefallDBRouter"
        assert con.execute("SELECT COUNT(*) FROM numbers").fetchall() == [(3,)]


def test_datafusion_multi_table_join_match_duckdb():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        events = pa.table(
            {"event_id": [1, 2, 3], "cat": ["a", "b", "a"]},
            schema=pa.schema(
                [
                    pa.field("event_id", pa.int64(), nullable=False),
                    pa.field("cat", pa.utf8(), nullable=True),
                ]
            ),
        )
        cats = pa.table(
            {"cat": ["a", "b"], "region": ["US", "EU"]},
            schema=pa.schema(
                [
                    pa.field("cat", pa.utf8(), nullable=False),
                    pa.field("region", pa.utf8(), nullable=True),
                ]
            ),
        )
        _make_table(db, "events", events)
        _make_table(db, "cats", cats)

        duck = icefalldb.attach(str(db), tables=["events", "cats"], engine="duckdb")
        fusion = icefalldb.attach(
            str(db), tables=["events", "cats"], engine="datafusion"
        )

        sql = (
            "SELECT e.event_id, c.region FROM events e "
            "JOIN cats c ON e.cat = c.cat ORDER BY e.event_id"
        )
        duck_rows = duck.execute(sql).fetchall()
        fusion_rows = fusion.sql(sql).fetchall()
        assert duck_rows == fusion_rows, f"{sql}: {duck_rows} != {fusion_rows}"


def test_datafusion_correctness_corpus():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        arrow_table = pa.table(
            {
                "id": [1, 2, 3, 4, 5, 6, 7, 8],
                "int_col": [10, 20, 30, 40, 50, 60, 70, 80],
                "float_col": [1.5, 2.5, None, 4.5, 5.5, 6.5, 7.5, 8.5],
                "utf8_col": ["a", "b", "c", "x", "x", "d", None, "e"],
                "nullable_col": [100, None, 200, None, 300, 400, None, 500],
            },
            schema=pa.schema(
                [
                    pa.field("id", pa.int64(), nullable=False),
                    pa.field("int_col", pa.int64(), nullable=False),
                    pa.field("float_col", pa.float64(), nullable=True),
                    pa.field("utf8_col", pa.utf8(), nullable=True),
                    pa.field("nullable_col", pa.int64(), nullable=True),
                ]
            ),
        )
        _make_table(db, "corpus", arrow_table)

        duck = icefalldb.attach(str(db), tables=["corpus"], engine="duckdb")
        fusion = icefalldb.attach(str(db), tables=["corpus"], engine="datafusion")

        queries = [
            "SELECT COUNT(*) AS cnt FROM corpus",
            "SELECT COUNT(nullable_col) AS cnt FROM corpus",
            "SELECT MIN(int_col) AS mn, MAX(int_col) AS mx FROM corpus",
            "SELECT MIN(float_col) AS mn, MAX(float_col) AS mx FROM corpus",
            "SELECT COUNT(*) AS cnt FROM corpus WHERE int_col > 2",
            "SELECT COUNT(*) AS cnt FROM corpus WHERE utf8_col = 'x'",
            "SELECT id, int_col, float_col, utf8_col, nullable_col FROM corpus WHERE nullable_col IS NULL ORDER BY id",
        ]
        for sql in queries:
            duck_rows = duck.execute(sql).fetchall()
            fusion_rows = fusion.sql(sql).fetchall()
            assert duck_rows == fusion_rows, f"{sql}: {duck_rows} != {fusion_rows}"
