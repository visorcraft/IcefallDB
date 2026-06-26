"""Tests for attach(snapshot=N) read-only time-travel and icefalldb.snapshots()."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import icefalldb
from icefalldb import IcefallDBError

REPO = Path(__file__).resolve().parents[2]


def _find_cli() -> Path | None:
    # Prefer the debug build so tests run against the current code that
    # writes committed_at in manifests; fall back to release if debug is absent.
    for candidate in (
        REPO / "target" / "debug" / "icefalldb",
        REPO / "target" / "release" / "icefalldb",
    ):
        if candidate.is_file():
            return candidate
    return None


def _native_available() -> bool:
    try:
        import icefalldb_query_py  # noqa: F401

        return True
    except Exception:
        return False


def _run_icefalldb(*args) -> None:
    cli = _find_cli()
    if cli is None:
        pytest.skip("icefalldb CLI not found; run cargo build first")
    result = subprocess.run(
        [str(cli), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise IcefallDBError(
            f"icefalldb {' '.join(str(a) for a in args)} failed: {result.stderr}"
        )


def _make_table(db_path: Path, table: str, arrow_table: pa.Table) -> None:
    schema = {
        "schema_id": 1,
        "columns": [
            {
                "name": field.name,
                "type": _arrow_type_to_icefalldb(field.type),
                "nullable": field.nullable,
                "field_id": idx + 1,
            }
            for idx, field in enumerate(arrow_table.schema)
        ],
        "row_group_target_rows": 1_000_000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": [],
        "max_field_id": len(arrow_table.schema),
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
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "utf8"
    if pa.types.is_float64(arrow_type):
        return "float64"
    raise ValueError(f"unsupported Arrow type: {arrow_type}")


def _make_db_then_delete(tmp_path: Path) -> Path:
    """Create a DB with table 't', insert 3 rows (seq 1), delete one row (seq 2)."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for time-travel tests")
    db = tmp_path / "db"
    db.mkdir()
    arrow_table = pa.table(
        {"id": [1, 2, 3], "name": ["alpha", "beta", "gamma"]},
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8(), nullable=True),
            ]
        ),
    )
    _make_table(db, "t", arrow_table)
    # Delete one row to advance to snapshot 2.
    import icefalldb_query_py

    conn = icefalldb_query_py.IcefallDBConnection(str(db), ["t"])
    conn.mutate("DELETE FROM t WHERE id = 1")
    return db


def _make_db_two_inserts(tmp_path: Path) -> Path:
    """Create a DB with two separate inserts so there are ≥2 snapshots."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for time-travel tests")
    db = tmp_path / "db"
    db.mkdir()
    arrow_table1 = pa.table(
        {"id": [1, 2], "val": ["a", "b"]},
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("val", pa.utf8(), nullable=True),
            ]
        ),
    )
    _make_table(db, "t", arrow_table1)

    # Second insert: write another parquet and insert.
    parquet_path = tmp_path / "t2.parquet"
    arrow_table2 = pa.table(
        {"id": [3], "val": ["c"]},
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("val", pa.utf8(), nullable=True),
            ]
        ),
    )
    pq.write_table(arrow_table2, parquet_path)
    _run_icefalldb("insert", str(db), "t", str(parquet_path))
    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_attach_as_of_sees_pre_delete(tmp_path):
    """attach(snapshot=1) sees the pre-delete row count; latest sees fewer."""
    db = _make_db_then_delete(tmp_path)
    latest = (
        icefalldb.attach(db, engine="datafusion")
        .sql("SELECT COUNT(*) FROM t")
        .fetchall()[0][0]
    )
    asof = (
        icefalldb.attach(db, snapshot=1).sql("SELECT COUNT(*) FROM t").fetchall()[0][0]
    )
    assert asof == latest + 1


def test_as_of_is_read_only(tmp_path):
    """A connection opened with snapshot= must reject mutations."""
    db = _make_db_then_delete(tmp_path)
    con = icefalldb.attach(db, snapshot=1)
    with pytest.raises(Exception):
        con.mutate("DELETE FROM t WHERE id = 1")


def test_duckdb_plus_snapshot_raises(tmp_path):
    """attach(engine='duckdb', snapshot=N) must raise ValueError."""
    db = _make_db_then_delete(tmp_path)
    with pytest.raises(ValueError):
        icefalldb.attach(db, engine="duckdb", snapshot=1)


def test_snapshots_listing(tmp_path):
    """icefalldb.snapshots() returns ≥2 entries with sequence and committed_at."""
    db = _make_db_two_inserts(tmp_path)
    snaps = icefalldb.snapshots(db, "t")
    assert len(snaps) >= 2
    assert snaps[0]["sequence"] == 1
    assert snaps[1]["committed_at"] is not None
