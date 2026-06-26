"""Tests for the standalone ResultCacheHandle pyclass."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import icefalldb_query_py as q


def _find_cli() -> Path | None:
    repo = Path(__file__).resolve().parents[2]
    for candidate in (
        repo / "target" / "release" / "icefalldb",
        repo / "target" / "debug" / "icefalldb",
    ):
        if candidate.is_file():
            return candidate
    return None


def _run_icefalldb(*args) -> None:
    cli = _find_cli()
    if cli is None:
        pytest.skip("icefalldb CLI not found; run cargo build first")
    result = subprocess.run(
        [str(cli), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(f"icefalldb {' '.join(args)} failed: {result.stderr}")


def _make_table(db: str, name: str) -> None:
    """Build a minimal IcefallDB table with one int64 column 'a'."""
    db_path = Path(db)
    db_path.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.table({"a": pa.array([1, 2, 3], type=pa.int64())})
    schema = {
        "schema_id": 1,
        "columns": [
            {"name": "a", "type": "int64", "nullable": True},
        ],
        "row_group_target_rows": 1_000_000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": 0,
    }
    schema_path = db_path / f"{name}_schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    _run_icefalldb("create", str(db_path), name, "--schema", str(schema_path))

    parquet_path = db_path / f"{name}.parquet"
    pq.write_table(arrow_table, parquet_path)
    _run_icefalldb("insert", str(db_path), name, str(parquet_path))


def test_handle_round_trip(tmp_path):
    db = str(tmp_path / "db")
    _make_table(db, "t")
    h = q.ResultCacheHandle(db, ["t"], result_cache_mb=16)
    sql = "SELECT * FROM t"
    assert h.get(sql) is None
    tbl = pa.table({"a": [1, 2, 3]})
    h.put(sql, tbl)
    got = h.get(sql)
    assert got is not None and got.num_rows == 3


def test_handle_disabled_when_zero(tmp_path):
    db = str(tmp_path / "db")
    _make_table(db, "t")
    h = q.ResultCacheHandle(db, ["t"], result_cache_mb=0)
    h.put("SELECT * FROM t", pa.table({"a": [1]}))
    assert h.get("SELECT * FROM t") is None  # disabled


def test_handle_clear(tmp_path):
    db = str(tmp_path / "db")
    _make_table(db, "t")
    h = q.ResultCacheHandle(db, ["t"], result_cache_mb=16)
    sql = "SELECT * FROM t"
    h.put(sql, pa.table({"a": [1, 2, 3]}))
    assert h.get(sql) is not None
    h.clear()
    assert h.get(sql) is None


def test_handle_zero_row_preserves_schema(tmp_path):
    """A cached 0-row result must return the original schema."""
    db = str(tmp_path / "db")
    _make_table(db, "t")
    h = q.ResultCacheHandle(db, ["t"], result_cache_mb=16)
    sql = "SELECT a, b FROM t WHERE false"
    empty_tbl = pa.table(
        {
            "a": pa.array([], type=pa.int64()),
            "b": pa.array([], type=pa.string()),
        }
    )
    h.put(sql, empty_tbl)
    got = h.get(sql)
    assert got is not None, "zero-row cache entry must be a hit"
    assert got.column_names == ["a", "b"], (
        f"schema must be preserved on cache hit, got {got.column_names}"
    )
    assert got.num_rows == 0
