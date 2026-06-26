"""Sanity checks for the DataFusion benchmark harness query matrix."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "benchmarks" / "datafusion"))

from run_icefalldb_query_bench import QUERIES  # noqa: E402

EXPECTED_QUERY_NAMES = {
    "warm_agg_group",
    "warm_filtered_scan",
    "warm_full_count",
    "join_100m_x_10",
    "wide_agg",
    "wide_filter",
    "sorted_time_window",
    "indexed_equality",
    "clustered_wide_filter",
}


def test_query_names_match_spec():
    names = {q[0] for q in QUERIES}
    assert names == EXPECTED_QUERY_NAMES, f"unexpected query names: {names}"


def test_queries_are_non_empty():
    for name, sql, tables in QUERIES:
        assert sql.strip(), f"{name}: query SQL is empty"
        assert tables, f"{name}: table list is empty"
        for table in tables:
            assert table.strip(), f"{name}: empty table name"


def test_query_sql_has_required_keywords():
    for name, sql, _tables in QUERIES:
        upper = sql.upper()
        assert "SELECT" in upper, f"{name}: missing SELECT"
        assert "FROM" in upper, f"{name}: missing FROM"


if __name__ == "__main__":
    test_query_names_match_spec()
    test_queries_are_non_empty()
    test_query_sql_has_required_keywords()
    print("all sanity checks passed")
