#!/usr/bin/env python3
"""DuckDB oracle helper for the IcefallDB DataFusion benchmark harness."""

from __future__ import annotations

from typing import Any, List

import duckdb

import icefalldb


def duckdb_oracle(db_path: str, tables: List[str], query: str) -> List[Any]:
    """Run ``query`` against ``tables`` in ``db_path`` through DuckDB.

    A fresh in-memory DuckDB connection is created for each call so the oracle
    is independent of any engine-specific connection state.
    """
    con = duckdb.connect()
    icefalldb.attach(db_path, connection=con, tables=tables)
    return con.execute(query).fetchall()


def assert_same(left, right, query_name: str) -> None:
    """Raise ``AssertionError`` with details when two result sets differ."""
    if left == right:
        return

    left_len = len(left) if isinstance(left, list) else "N/A"
    right_len = len(right) if isinstance(right, list) else "N/A"
    raise AssertionError(
        f"{query_name}: result mismatch\n"
        f"  left rows:  {left_len}\n"
        f"  right rows: {right_len}\n"
        f"  left:  {left!r}\n"
        f"  right: {right!r}"
    )
