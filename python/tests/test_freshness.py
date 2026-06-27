"""Freshness tests for direct native ``engine="datafusion"`` connections."""

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


def _sch() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("val", pa.utf8(), nullable=True),
        ]
    )


def test_datafusion_auto_refresh_after_external_insert(tmp_path):
    """A live direct native connection sees rows inserted by another process."""
    if not _native_available():
        pytest.skip("native DataFusion extension required")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(db, "t", pa.table({"id": [1, 2], "val": ["a", "b"]}, schema=_sch()))

    con = icefalldb.attach(str(db), tables=["t"], engine="datafusion")
    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(2,)]

    extra = tmp_path / "extra.parquet"
    pq.write_table(pa.table({"id": [3], "val": ["c"]}, schema=_sch()), extra)
    _run_icefalldb("insert", str(db), "t", str(extra))

    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(3,)]


def test_datafusion_auto_refresh_after_external_delete(tmp_path):
    """A live direct native connection sees deletions applied by another process."""
    if not _native_available():
        pytest.skip("native DataFusion extension required")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(
        db, "t", pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]}, schema=_sch())
    )

    con = icefalldb.attach(str(db), tables=["t"], engine="datafusion")
    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(3,)]

    _run_icefalldb("query", str(db) + "/t", "DELETE FROM t WHERE id = 3")

    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(2,)]


def test_datafusion_snapshot_pinned_stays_stable_after_external_write(tmp_path):
    """A snapshot-pinned direct native connection does not auto-refresh."""
    if not _native_available():
        pytest.skip("native DataFusion extension required")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(db, "t", pa.table({"id": [1, 2], "val": ["a", "b"]}, schema=_sch()))

    con = icefalldb.attach(str(db), tables=["t"], engine="datafusion", snapshot=1)
    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(2,)]

    extra = tmp_path / "extra.parquet"
    pq.write_table(pa.table({"id": [3], "val": ["c"]}, schema=_sch()), extra)
    _run_icefalldb("insert", str(db), "t", str(extra))

    # Pinned to snapshot 1, so the external insert is invisible.
    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(2,)]


def test_datafusion_mutate_after_external_write(tmp_path):
    """A live direct native connection refreshes before mutating externally-changed data."""
    if not _native_available():
        pytest.skip("native DataFusion extension required")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(db, "t", pa.table({"id": [1, 2], "val": ["a", "b"]}, schema=_sch()))

    con = icefalldb.attach(str(db), tables=["t"], engine="datafusion")
    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(2,)]

    extra = tmp_path / "extra.parquet"
    pq.write_table(pa.table({"id": [3], "val": ["c"]}, schema=_sch()), extra)
    _run_icefalldb("insert", str(db), "t", str(extra))

    # The local connection must refresh before applying a delta; otherwise the
    # provider's pinned_sequence would not match the delta's previous_sequence.
    affected = con.mutate("DELETE FROM t WHERE id = 3")
    assert affected == 1
    assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(2,)]


def test_datafusion_multi_table_refresh_is_isolated(tmp_path):
    """Per-table freshness: an external write to one table refreshes only that
    table, repeated queries return the fresh (not cache-poisoned) result, and
    one table's write never contaminates another's cached results."""
    if not _native_available():
        pytest.skip("native DataFusion extension required")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(db, "a", pa.table({"id": [1, 2], "val": ["a", "b"]}, schema=_sch()))
    _make_table(db, "b", pa.table({"id": [1], "val": ["x"]}, schema=_sch()))

    con = icefalldb.attach(str(db), tables=["a", "b"], engine="datafusion")

    # Warm the result cache for both tables.
    assert con.sql("SELECT COUNT(*) FROM a").fetchall() == [(2,)]
    assert con.sql("SELECT COUNT(*) FROM b").fetchall() == [(1,)]

    # External write to `a` only.
    extra_a = tmp_path / "a_extra.parquet"
    pq.write_table(pa.table({"id": [3], "val": ["c"]}, schema=_sch()), extra_a)
    _run_icefalldb("insert", str(db), "a", str(extra_a))

    # `a` refreshes; `b` is untouched (no cross-table poisoning).
    assert con.sql("SELECT COUNT(*) FROM a").fetchall() == [(3,)]
    assert con.sql("SELECT COUNT(*) FROM b").fetchall() == [(1,)]
    # Repeat the `a` query: the cache must serve the fresh value, not a stale one.
    assert con.sql("SELECT COUNT(*) FROM a").fetchall() == [(3,)]

    # External write to `b` only.
    extra_b = tmp_path / "b_extra.parquet"
    pq.write_table(pa.table({"id": [2], "val": ["y"]}, schema=_sch()), extra_b)
    _run_icefalldb("insert", str(db), "b", str(extra_b))

    # `b` refreshes; `a` keeps its own latest count, unaffected by `b`'s write.
    assert con.sql("SELECT COUNT(*) FROM b").fetchall() == [(2,)]
    assert con.sql("SELECT COUNT(*) FROM a").fetchall() == [(3,)]


def test_datafusion_repeated_external_writes_never_serve_stale(tmp_path):
    """Each external write to a live connection invalidates the result cache, so
    a repeated identical query is never answered from a poisoned stale entry."""
    if not _native_available():
        pytest.skip("native DataFusion extension required")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(db, "t", pa.table({"id": [1], "val": ["a"]}, schema=_sch()))

    con = icefalldb.attach(str(db), tables=["t"], engine="datafusion")

    for n in range(2, 6):
        # Warm the cache at the current snapshot.
        assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(n - 1,)]
        extra = tmp_path / f"extra_{n}.parquet"
        pq.write_table(pa.table({"id": [n], "val": ["v"]}, schema=_sch()), extra)
        _run_icefalldb("insert", str(db), "t", str(extra))
        # Two reads in a row: the first must refresh, the second must hit the
        # cache for the new snapshot, both fresh.
        assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(n,)]
        assert con.sql("SELECT COUNT(*) FROM t").fetchall() == [(n,)]
