from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from icefalldb import attach

FIXTURES_DIR = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "icefalldb-core"
    / "tests"
    / "golden"
    / "fixtures"
)


def _table_dir(name: str) -> Path:
    return FIXTURES_DIR / name


@pytest.fixture
def connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def test_golden_simple_int(connection: duckdb.DuckDBPyConnection) -> None:
    con = attach(_table_dir("simple_int"), connection=connection)
    result = con.execute("SELECT id FROM simple_int ORDER BY id").fetchall()
    assert result == [(1,), (2,), (3,)]


def test_golden_multi_commit(connection: duckdb.DuckDBPyConnection) -> None:
    con = attach(_table_dir("multi_commit"), connection=connection)
    result = con.execute("SELECT id FROM multi_commit ORDER BY id").fetchall()
    assert result == [(1,), (2,), (3,), (4,), (5,), (6,)]


def test_golden_multi_row_group(connection: duckdb.DuckDBPyConnection) -> None:
    con = attach(_table_dir("multi_row_group"), connection=connection)
    result = con.execute("SELECT id FROM multi_row_group ORDER BY id").fetchall()
    assert result == [(1,), (2,), (3,), (4,), (5,)]


def test_golden_mixed_types(connection: duckdb.DuckDBPyConnection) -> None:
    con = attach(_table_dir("mixed_types"), connection=connection)
    result = con.execute(
        "SELECT id, value, label, active FROM mixed_types ORDER BY id"
    ).fetchall()
    assert result[0] == (1, 1.5, "a", True)
    assert result[1] == (2, None, "b", False)
    assert result[2] == (3, 3.0, None, True)
