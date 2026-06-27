"""Tests for the IcefallDBRouter (DuckDB SELECTs + native mutations)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import icefalldb
from icefalldb import IcefallDBError
from icefalldb.adapter import (
    IcefallDBRouter,
    _classify_sql,
    _extract_mutation_table,
    _metadata_aggregate_target,
    _table_has_active_deletions,
    _table_is_encrypted,
    _table_referenced_in_sql,
)


def _find_cli() -> Path | None:
    """Locate a built icefalldb CLI (release preferred over debug)."""
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
        raise IcefallDBError(f"icefalldb {' '.join(args)} failed: {result.stderr}")


def _native_available() -> bool:
    try:
        import icefalldb_query_py  # noqa: F401

        return True
    except Exception:
        return False


def _arrow_type_to_icefalldb(arrow_type: pa.DataType) -> str:
    if pa.types.is_int64(arrow_type):
        return "int64"
    if pa.types.is_float64(arrow_type):
        return "float64"
    if pa.types.is_string(arrow_type):
        return "utf8"
    raise ValueError(f"unsupported Arrow type: {arrow_type}")


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


# --------------------------------------------------------------------------- #
# Unit tests: SQL classification (no DB needed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT 1", "select"),
        ("select 1", "select"),
        ("  SELECT 1", "select"),
        ("-- comment\nSELECT 1", "select"),
        ("/* leading */ SELECT 1", "select"),
        ("WITH t AS (SELECT 1) SELECT * FROM t", "select"),
        ("VALUES (1, 2)", "select"),
        ("(SELECT 1)", "select"),
        ("DELETE FROM t WHERE x = 1", "delete"),
        ("delete from t", "delete"),
        ("UPDATE t SET x = 1", "update"),
        (
            "MERGE INTO t USING s ON t.id = s.id WHEN MATCHED THEN UPDATE SET x = 1",
            "merge",
        ),
        (
            "MERGE t USING s ON t.id = s.id WHEN NOT MATCHED THEN INSERT VALUES (1)",
            "merge",
        ),
        ("INSERT INTO t VALUES (1)", "insert"),
        ("CREATE VIEW v AS SELECT 1", "other"),
        ("EXPLAIN SELECT 1", "select"),
    ],
)
def test_classify_sql(sql, expected):
    assert _classify_sql(sql) == expected


def test_classify_sql_rejects_empty():
    with pytest.raises(IcefallDBError):
        _classify_sql("   ")


def test_classify_sql_rejects_multi_statement():
    with pytest.raises(IcefallDBError):
        _classify_sql("SELECT 1; SELECT 2")


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("DELETE FROM events WHERE id = 1", "events"),
        ("UPDATE products SET price = 0", "products"),
        (
            "MERGE INTO target USING src ON target.id = src.id WHEN MATCHED THEN DELETE",
            "target",
        ),
        (
            "MERGE target USING src ON target.id = src.id WHEN NOT MATCHED THEN INSERT VALUES (1)",
            "target",
        ),
        ("SELECT * FROM events", None),
        ('DELETE FROM "quoted"', None),  # quoted identifiers unsupported by the regex
    ],
)
def test_extract_mutation_table(sql, expected):
    assert _extract_mutation_table(sql) == expected


@pytest.mark.parametrize(
    "sql, table, expected",
    [
        ("SELECT * FROM events", "events", True),
        ("SELECT * FROM events WHERE x = 1", "events", True),
        ("SELECT * FROM eventsv2", "events", False),  # word boundary
        ("SELECT my_events.* FROM my_events", "events", False),
        ("SELECT * FROM e JOIN events ON e.id = events.id", "events", True),
        ("SELECT * FROM EVENTS", "events", True),  # case-insensitive
    ],
)
def test_table_referenced_in_sql(sql, table, expected):
    assert _table_referenced_in_sql(sql, table) is expected


# --------------------------------------------------------------------------- #
# Metadata-aggregate detection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "sql, expected",
    [
        ("SELECT COUNT(*) FROM events", "events"),
        ("SELECT COUNT(*) AS cnt FROM events", "events"),
        ("select count(*) from events", "events"),
        ("SELECT COUNT(id) FROM events", "events"),
        ("SELECT MIN(value), MAX(value) FROM events", "events"),
        ("SELECT SUM(amount), AVG(amount) FROM orders", "orders"),
        ("SELECT MIN(ts), MAX(ts), COUNT(*) FROM events", "events"),
        ("SELECT approx_distinct(user_id) FROM events", "events"),
        ('SELECT COUNT(*) AS "cnt" FROM events', "events"),  # quoted alias
        # ineligible: expression inside aggregate (must stay on DuckDB)
        ("SELECT SUM(value * 2) FROM events", None),
        ("SELECT SUM(amount - 1) FROM orders", None),
        # ineligible: has a WHERE
        ("SELECT COUNT(*) FROM events WHERE x = 1", None),
        # ineligible: has GROUP BY
        ("SELECT category, COUNT(*) FROM events GROUP BY category", None),
        # ineligible: JOIN
        ("SELECT COUNT(*) FROM a JOIN b ON a.id = b.id", None),
        # ineligible: bare column (not an aggregate)
        ("SELECT id FROM events", None),
        ("SELECT category, COUNT(*) FROM events", None),
        # ineligible: DISTINCT aggregate
        ("SELECT COUNT(DISTINCT category) FROM events", None),
        # ineligible: non-aggregate function
        ("SELECT upper(name) FROM events", None),
        # ineligible: LIMIT
        ("SELECT COUNT(*) FROM events LIMIT 1", None),
        # ineligible: two tables (comma join)
        ("SELECT COUNT(*) FROM a, b", None),
        # multi-statement is not a metadata target
        ("SELECT COUNT(*) FROM events; SELECT 1", None),
    ],
)
def test_metadata_aggregate_target(sql, expected):
    assert _metadata_aggregate_target(sql) == expected


# --------------------------------------------------------------------------- #
# Integration tests: routing + correctness
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_db(tmp_path):
    db = tmp_path / "db"
    db.mkdir()
    arrow_table = pa.table(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "category": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "value": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("category", pa.utf8(), nullable=True),
                pa.field("value", pa.float64(), nullable=True),
            ]
        ),
    )
    _make_table(db, "events", arrow_table)
    return db


def test_hybrid_is_icefalldb_router(sample_db):
    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    assert isinstance(con, IcefallDBRouter)


def test_hybrid_clean_table_routed_to_duckdb(sample_db):
    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    assert con._native_only == {}, con._native_only
    # A clean, unencrypted table is DuckDB-safe.
    assert con._duckdb_safe("SELECT COUNT(*) FROM events") is True
    # The DuckDB connection holds a view over the table.
    assert con._duck.execute("SELECT COUNT(*) FROM events").fetchall() == [(8,)]


def test_hybrid_select_matches_duckdb(sample_db):
    hybrid = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    duck = icefalldb.attach(str(sample_db), tables=["events"], engine="duckdb")

    queries = [
        "SELECT COUNT(*) AS cnt FROM events",
        "SELECT MIN(id) AS mn, MAX(id) AS mx FROM events",
        "SELECT COUNT(*) AS cnt FROM events WHERE category = 'a'",
        "SELECT SUM(value) AS total FROM events WHERE value > 35",
    ]
    for sql in queries:
        duck_rows = duck.execute(sql).fetchall()
        hybrid_rows = hybrid.sql(sql).fetchall()
        assert hybrid_rows == duck_rows, f"{sql}: {hybrid_rows} != {duck_rows}"


def test_hybrid_encryption_guard_keeps_table_off_duckdb(sample_db):
    """A table with an _encryption.json marker must never be exposed to DuckDB."""
    (sample_db / "events" / "_encryption.json").write_text(
        json.dumps(
            {
                "algorithm": "parquet-modular-encryption-v1",
                "footer_key_id": "events-v1",
                "column_key_ids": {},
                "plaintext_footer": True,
                "aad_prefix": None,
            }
        ),
        encoding="utf-8",
    )

    assert _table_is_encrypted(sample_db, "events") is True

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    assert con._native_only.get("events") == "encrypted"
    # DuckDB must NOT have a view over the encrypted table.
    with pytest.raises(Exception):
        con._duck.execute("SELECT COUNT(*) FROM events").fetchall()
    # The SELECT still routes to the native engine (which reads the plaintext
    # Parquet fine here, since the marker is synthetic).
    if _native_available():
        rows = con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall()
        assert rows == [(8,)]


def test_hybrid_deletion_vector_guard(sample_db):
    """A table with active deletion vectors routes to the native engine."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    # Initially clean.
    assert _table_has_active_deletions(sample_db, "events") is False
    assert con._native_only == {}

    # DELETE half the rows through the hybrid connection.
    result = con.sql("DELETE FROM events WHERE category = 'a'").fetchall()
    affected = result[0][0]
    assert affected == 4

    # The table now has active deletion vectors.
    assert _table_has_active_deletions(sample_db, "events") is True
    assert con._native_only.get("events") == "dirty"

    # The DuckDB view was dropped, so DuckDB can no longer read the table.
    with pytest.raises(Exception):
        con._duck.execute("SELECT COUNT(*) FROM events").fetchall()

    # A subsequent SELECT routes to native and sees the post-delete count (4).
    rows = con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall()
    assert rows == [(4,)]


def _assert_explicit_duckdb_rejects_dirty_table(db: Path, table: str) -> None:
    assert _table_has_active_deletions(db, table) is True

    with pytest.raises(IcefallDBError, match="active deletion vectors"):
        icefalldb.attach(str(db), tables=[table], engine="duckdb")

    with pytest.raises(IcefallDBError, match="active deletion vectors"):
        icefalldb.attach_table(str(db), table, engine="duckdb")


def test_explicit_duckdb_rejects_deleted_rows(sample_db):
    """Explicit DuckDB must fail rather than expose deletion-vector tombstones."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    con.mutate("DELETE FROM events WHERE id = 1")

    _assert_explicit_duckdb_rejects_dirty_table(sample_db, "events")
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(7,)]

    native = icefalldb.attach(str(sample_db), tables=["events"], engine="datafusion")
    assert native.sql("SELECT id FROM events ORDER BY id").fetchall() == [
        (2,),
        (3,),
        (4,),
        (5,),
        (6,),
        (7,),
        (8,),
    ]


def test_explicit_duckdb_rejects_updated_rows(sample_db):
    """Explicit DuckDB must fail rather than expose update pre-images."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    con.mutate("UPDATE events SET value = 999.0 WHERE id = 2")

    _assert_explicit_duckdb_rejects_dirty_table(sample_db, "events")
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(8,)]
    assert con.sql("SELECT value FROM events WHERE id = 2").fetchall() == [(999.0,)]

    native = icefalldb.attach(str(sample_db), tables=["events"], engine="datafusion")
    assert native.sql("SELECT value FROM events WHERE id = 2").fetchall() == [(999.0,)]


def test_explicit_duckdb_rejects_merge_updated_rows(sample_db):
    """Explicit DuckDB must fail rather than expose MERGE update pre-images."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    _run_icefalldb("create-index", "--unique", str(sample_db), "events", "id")
    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    con.mutate(
        "MERGE INTO events USING "
        "(VALUES (2, 'merged', 222.0), (9, 'new', 90.0)) AS src(id, category, value) "
        "ON events.id = src.id "
        "WHEN MATCHED THEN UPDATE SET "
        "id = src.id, category = src.category, value = src.value "
        "WHEN NOT MATCHED THEN INSERT (id, category, value) "
        "VALUES (src.id, src.category, src.value)"
    )

    _assert_explicit_duckdb_rejects_dirty_table(sample_db, "events")
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(9,)]
    assert con.sql("SELECT category, value FROM events WHERE id = 2").fetchall() == [
        ("merged", 222.0)
    ]

    native = icefalldb.attach(str(sample_db), tables=["events"], engine="datafusion")
    assert native.sql("SELECT category, value FROM events WHERE id = 9").fetchall() == [
        ("new", 90.0)
    ]


def test_hybrid_delete_correctness_via_native(sample_db):
    """Rows deleted through hybrid stay deleted; survivors are intact."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    con.sql("DELETE FROM events WHERE id > 5").fetchall()

    survivor_ids = con.sql("SELECT id FROM events ORDER BY id").fetchall()
    assert survivor_ids == [(1,), (2,), (3,), (4,), (5,)]

    total = con.sql("SELECT SUM(value) AS total FROM events").fetchall()
    assert total == [(150.0,)]  # 10+20+30+40+50


def test_hybrid_refresh_after_clean_insert(tmp_path):
    """After an insert-only commit (no deletion vectors), DuckDB sees new rows."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    db = tmp_path / "db"
    db.mkdir()
    _make_table(
        db,
        "events",
        pa.table(
            {"id": pa.array([1, 2], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
    )

    con = icefalldb.attach(str(db), tables=["events"], engine="icefalldb")
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(2,)]

    # Insert more rows through the CLI (insert path, no deletion vectors).
    extra = db / "extra.parquet"
    pq.write_table(
        pa.table(
            {"id": pa.array([3, 4], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
        extra,
    )
    _run_icefalldb("insert", str(db), "events", str(extra))

    # Refresh routing so DuckDB picks up the new snapshot.
    con._refresh_routing()
    assert con._native_only == {}  # still clean
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(4,)]


def test_native_result_cache_not_poisoned_by_stale_connection(tmp_path):
    """A snapshot-isolated connection must key its cached result on the snapshot
    it computed against, not the live manifest pointer.

    Regression: ``read_snapshots`` keyed the native result cache on the current
    on-disk ``_manifest.json`` sequence. A long-lived connection pinned to an
    older snapshot would compute a (correct, isolated) result for *its* snapshot
    but store it under the newer snapshot's key, poisoning the shared
    ``_query_cache`` so a freshly-opened connection returned the stale value.
    """
    if not _native_available():
        pytest.skip("native DataFusion extension required")
    import icefalldb_query_py

    db = tmp_path / "db"
    db.mkdir()
    sch = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    _make_table(
        db, "events", pa.table({"id": pa.array([1, 2], pa.int64())}, schema=sch)
    )

    # Long-lived connection pinned to seq 1 (2 rows).
    stale = icefalldb_query_py.IcefallDBConnection(str(db), ["events"])
    assert stale.sql("SELECT COUNT(*) AS c FROM events").to_pylist() == [{"c": 2}]

    # An external writer advances the table to seq 2 (4 rows).
    extra = db / "extra.parquet"
    pq.write_table(pa.table({"id": pa.array([3, 4], pa.int64())}, schema=sch), extra)
    _run_icefalldb("insert", str(db), "events", str(extra))

    # The stale connection is snapshot-isolated and still computes 2; it must not
    # write that under the live (seq 2) cache key.
    assert stale.sql("SELECT COUNT(*) AS c FROM events").to_pylist() == [{"c": 2}]

    # A freshly-opened connection sees seq 2 and must compute 4 — never the stale
    # connection's poisoned value.
    fresh = icefalldb_query_py.IcefallDBConnection(str(db), ["events"])
    assert fresh.sql("SELECT COUNT(*) AS c FROM events").to_pylist() == [{"c": 4}]


def test_native_mutation_self_invalidates_without_cache_wipe(tmp_path):
    """A native mutation invalidates cached results through the snapshot-keyed
    cache key (each table's pinned_sequence is hashed into the key), not by
    wiping the whole ``_query_cache``. After a mutation the re-query must return
    fresh data, and the superseded entry must remain on disk (proving the
    O(cache-size) per-write wipe is gone — it ages out via LRU instead).
    """
    if not _native_available():
        pytest.skip("native DataFusion extension required")
    import icefalldb_query_py

    db = tmp_path / "db"
    db.mkdir()
    sch = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    _make_table(
        db, "events", pa.table({"id": pa.array([1, 2, 3], pa.int64())}, schema=sch)
    )

    con = icefalldb_query_py.IcefallDBConnection(str(db), ["events"])
    q = "SELECT COUNT(*) AS c FROM events"
    assert con.sql(q).to_pylist() == [{"c": 3}]  # compute + cache at the open seq

    cache_dir = db / "_query_cache"
    before = set(cache_dir.glob("*.arrow"))
    assert before, "the query should have written a cache entry"

    con.mutate("DELETE FROM events WHERE id = 1")

    # The pre-mutation entry must survive (no full-directory wipe); discriminating
    # — this assertion fails if the post-mutation cache.clear() is reintroduced.
    assert before <= set(cache_dir.glob("*.arrow")), (
        "post-mutation wipe deleted the pre-mutation cache entry; the snapshot-"
        "keyed cache should leave it to age out via LRU"
    )

    # The re-query keys on the advanced pinned_sequence → miss → fresh result.
    assert con.sql(q).to_pylist() == [{"c": 2}]


def test_native_cache_reference_aware_cross_connection_sharing(tmp_path):
    """Reference-aware keying: ``SELECT … FROM a`` is keyed on a's snapshot only,
    so two connections with *different* registered table sets share the cached
    result for a query on the common table ``a`` instead of writing duplicates.
    The second connection's identical query reuses the first's entry — no new
    cache file. Discriminating: under all-tables keying the two connections key
    on different table sets ([a, b] vs [a, c]) and never share.
    """
    if not _native_available():
        pytest.skip("native DataFusion extension required")
    import icefalldb_query_py

    db = tmp_path / "db"
    db.mkdir()
    sch = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    for name in ("a", "b", "c"):
        _make_table(
            db, name, pa.table({"id": pa.array([1, 2, 3], pa.int64())}, schema=sch)
        )

    q = "SELECT COUNT(*) AS c FROM a"
    x = icefalldb_query_py.IcefallDBConnection(str(db), ["a", "b"])
    assert x.sql(q).to_pylist() == [{"c": 3}]

    cache_dir = db / "_query_cache"
    snapshot = set(cache_dir.glob("*.arrow"))
    assert len(snapshot) == 1, "the first query should write exactly one entry"

    # A different connection (registers c instead of b) runs the same query on a.
    y = icefalldb_query_py.IcefallDBConnection(str(db), ["a", "c"])
    assert y.sql(q).to_pylist() == [{"c": 3}]

    # Same key (a's snapshot only) → y reuses x's entry; no duplicate is written.
    assert set(cache_dir.glob("*.arrow")) == snapshot, (
        "second connection did not share the cached result — keying is not "
        "reference-aware"
    )


def test_hybrid_mutate_explicit_method(sample_db):
    """The explicit mutate() method returns affected rows and deletes correctly."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    affected = con.mutate("DELETE FROM events WHERE value >= 50")
    assert affected == 4  # values 50,60,70,80
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(4,)]


def test_hybrid_multi_table_routing(sample_db, tmp_path):
    """A dirty table does not force clean tables onto the native engine."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    _make_table(
        sample_db,
        "categories",
        pa.table(
            {"category": pa.array(["a", "b"], pa.utf8())},
            schema=pa.schema([pa.field("category", pa.utf8(), nullable=False)]),
        ),
    )

    con = icefalldb.attach(
        str(sample_db), tables=["events", "categories"], engine="icefalldb"
    )
    # Make 'events' dirty while 'categories' stays clean.
    con.mutate("DELETE FROM events WHERE id = 1")

    assert con._native_only.get("events") == "dirty"
    assert "categories" not in con._native_only

    # A SELECT touching the dirty table is NOT DuckDB-safe.
    assert con._duckdb_safe("SELECT COUNT(*) FROM events") is False
    # A SELECT touching only the clean table IS DuckDB-safe.
    assert con._duckdb_safe("SELECT COUNT(*) FROM categories") is True
    # A join touching both is not DuckDB-safe.
    assert (
        con._duckdb_safe(
            "SELECT COUNT(*) FROM events e JOIN categories c ON e.category = c.category"
        )
        is False
    )


def test_hybrid_metadata_aggregate_routes_to_native(sample_db):
    """Unfiltered aggregates on a clean table route to the native engine."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for metadata routing")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    # COUNT(*) / MIN / MAX are metadata-served shapes.
    assert con._metadata_target("SELECT COUNT(*) FROM events") == "events"
    assert con._metadata_target("SELECT MIN(value) FROM events") == "events"
    # A filtered query is NOT a metadata target.
    assert con._metadata_target("SELECT COUNT(*) FROM events WHERE id > 1") is None
    # A non-aggregate is NOT a metadata target.
    assert con._metadata_target("SELECT id FROM events") is None

    # The metadata path returns correct results matching DuckDB.
    duck = icefalldb.attach(str(sample_db), tables=["events"], engine="duckdb")
    checks = [
        "SELECT COUNT(*) AS cnt FROM events",
        "SELECT MIN(id) AS mn, MAX(id) AS mx FROM events",
        "SELECT MIN(value) AS mn, MAX(value) AS mx FROM events",
        "SELECT COUNT(*) AS cnt, MIN(value) AS mn FROM events",
    ]
    for sql in checks:
        hybrid_rows = con.sql(sql).fetchall()
        duck_rows = duck.execute(sql).fetchall()
        assert hybrid_rows == duck_rows, f"{sql}: {hybrid_rows} != {duck_rows}"


def test_hybrid_metadata_aggregate_falls_back_to_duckdb(sample_db, monkeypatch):
    """If the native engine cannot serve a metadata query, DuckDB answers it.

    Forces the single-table native connection to raise, then confirms the query
    still returns the correct result via DuckDB.
    """
    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    monkeypatch.setattr(
        con,
        "_single_table_native",
        lambda table: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    rows = con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall()
    assert rows == [(8,)]


def test_hybrid_rejects_cross_table_mutation(sample_db):
    """A mutation referencing a second attached table fails with a clear error."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    _make_table(
        sample_db,
        "other",
        pa.table(
            {"id": pa.array([1, 2], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
    )
    con = icefalldb.attach(
        str(sample_db), tables=["events", "other"], engine="icefalldb"
    )
    with pytest.raises(IcefallDBError, match="single target table"):
        con.mutate("DELETE FROM events WHERE id IN (SELECT id FROM other)")


def test_hybrid_freshness_after_external_write(sample_db):
    """A concurrent external write (adding deletion vectors) is detected, so a
    stale DuckDB view never serves ghost rows."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for mutation tests")

    con = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    assert con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall() == [(8,)]
    assert con._native_only == {}  # clean -> DuckDB

    # Simulate an external writer committing a DELETE via a separate connection.
    other = icefalldb.attach(str(sample_db), tables=["events"], engine="icefalldb")
    other.mutate("DELETE FROM events WHERE id > 5")  # removes 3 rows

    # The original connection must see the new snapshot: the table is now dirty
    # (deletion vectors), so it routes to native and reports the post-delete count.
    assert con._native_only.get("events") != "dirty"  # not yet refreshed
    rows = con.sql("SELECT COUNT(*) AS cnt FROM events").fetchall()
    assert rows == [(5,)]
    assert con._native_only.get("events") == "dirty"  # refreshed -> native-only
    # DuckDB no longer holds a view over the now-dirty table.
    with pytest.raises(Exception):
        con._duck.execute("SELECT COUNT(*) FROM events").fetchall()


# --------------------------------------------------------------------------- #
# Migration guard: engine="hybrid" renamed to engine="icefalldb"; IcefallDBRouter exported
# --------------------------------------------------------------------------- #


def test_hybrid_engine_is_removed(tmp_path):
    # The ValueError is raised before any db structure is checked,
    # so we only need an existing directory.
    with pytest.raises(ValueError, match="renamed to 'icefalldb'"):
        icefalldb.attach(str(tmp_path), engine="hybrid")  # renamed to "icefalldb"


def test_icefalldb_router_is_exported():
    assert hasattr(icefalldb, "IcefallDBRouter")
    assert not hasattr(icefalldb, "HybridConnection")  # renamed to IcefallDBRouter


# --------------------------------------------------------------------------- #
# connection= adoption and execute() parity
# --------------------------------------------------------------------------- #


def test_attach_supports_execute_via_icefalldb_engine(tmp_path):
    """IcefallDBRouter.execute() works as an alias for sql() when engine="icefalldb"."""
    db = tmp_path / "db"
    db.mkdir()
    _make_table(
        db,
        "t",
        pa.table(
            {"id": pa.array([1, 2, 3], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
    )
    con = icefalldb.attach(str(db), tables=["t"], engine="icefalldb")
    assert type(con).__name__ == "IcefallDBRouter"
    rows = con.execute("SELECT COUNT(*) FROM t").fetchall()
    assert rows[0][0] == 3


def test_execute_rejects_query_parameters(tmp_path):
    """IcefallDBRouter.execute() raises rather than silently dropping query params."""
    db = tmp_path / "db"
    db.mkdir()
    _make_table(
        db,
        "t",
        pa.table(
            {"id": pa.array([1, 2, 3], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
    )
    con = icefalldb.attach(str(db), tables=["t"], engine="icefalldb")
    # Positional and keyword parameters are both rejected (not ignored).
    with pytest.raises(TypeError):
        con.execute("SELECT * FROM t WHERE id = ?", [1])
    with pytest.raises(TypeError):
        con.execute("SELECT * FROM t WHERE id = $id", id=1)
    # The no-parameter form still works.
    assert con.execute("SELECT COUNT(*) FROM t").fetchall()[0][0] == 3


def test_attach_reuses_supplied_connection(tmp_path):
    """When connection= is passed, IcefallDBRouter adopts it as self._duck."""
    db = tmp_path / "db"
    db.mkdir()
    _make_table(
        db,
        "t",
        pa.table(
            {"id": pa.array([10, 20], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
    )
    raw = duckdb.connect()
    con = icefalldb.attach(str(db), tables=["t"], engine="icefalldb", connection=raw)
    assert con._duck is raw
    rows = con.sql("SELECT COUNT(*) FROM t").fetchall()
    assert rows == [(2,)]


# --------------------------------------------------------------------------- #
# cache-through in IcefallDBRouter SELECT path
# --------------------------------------------------------------------------- #


def _make_cache_db(tmp_path: Path) -> Path:
    """Build a small single-table IcefallDB with a unique `id` index."""
    db = tmp_path / "db"
    db.mkdir()
    arrow_table = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "category": pa.array(["a", "b", "a", "b", "a"], pa.utf8()),
        },
        schema=pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("category", pa.utf8(), nullable=True),
            ]
        ),
    )
    _make_table(db, "t", arrow_table)
    _run_icefalldb("create-index", "--unique", str(db), "t", "id")
    return db


def test_icefalldb_select_repeat_is_cache_hit(tmp_path):
    """Repeating a DuckDB-bound SELECT hits the result cache on the second call."""
    if not _native_available():
        pytest.skip("native DataFusion extension required for result cache")
    import os

    db = _make_cache_db(tmp_path)
    con = icefalldb.attach(
        str(db), tables=["t"], engine="icefalldb", result_cache_mb=16
    )
    sql = "SELECT category, COUNT(*) c FROM t GROUP BY category ORDER BY category"
    first = con.sql(sql).fetchall()
    second = con.sql(sql).fetchall()
    assert first == second
    # Prove the cache file exists for this snapshot key.
    assert any(
        f.endswith(".arrow") for f in os.listdir(os.path.join(str(db), "_query_cache"))
    )


def test_icefalldb_mutation_invalidates_cache(tmp_path):
    """After a mutation, repeating a cached SELECT returns fresh data - never a
    stale cached result. The router does NOT wipe the cache; the mutated table
    becomes "dirty" (a non-empty WAL log) and routes to the native engine, so its
    pre-mutation DuckDB-cached entries are bypassed rather than deleted.
    """
    if not _native_available():
        pytest.skip("native DataFusion extension required for result cache")

    db = _make_cache_db(tmp_path)
    con = icefalldb.attach(
        str(db), tables=["t"], engine="icefalldb", result_cache_mb=16
    )

    sql = "SELECT id FROM t WHERE id > 0 ORDER BY id"
    rows0 = con.sql(sql).fetchall()
    con.sql(sql)  # warm the cache (second call is a hit)

    cache_dir = db / "_query_cache"
    assert any(cache_dir.glob("*.arrow")), "expected a cache entry after warming"

    con.mutate("DELETE FROM t WHERE id = 1")

    # The outcome that matters: the re-query returns fresh, post-mutation data.
    # `t` is now dirty, so this SELECT routes to native and never reads the stale
    # DuckDB cache entry (the wholesale wipe is gone).
    rows1 = con.sql(sql).fetchall()
    assert rows1 == [r for r in rows0 if r[0] != 1], (
        "mutation served a stale cached result"
    )


def test_icefalldb_router_mutation_preserves_other_table_cache(tmp_path):
    """A router mutation on one table must NOT wipe cached SELECTs for another.
    With the wholesale post-mutation wipe gone, u's cached result and its .arrow
    file survive a DELETE on t. Discriminating: the old wipe cleared the entire
    _query_cache, removing u's entry too.
    """
    if not _native_available():
        pytest.skip("native DataFusion extension required for result cache")
    db = tmp_path / "db"
    db.mkdir()
    sch = pa.schema([pa.field("id", pa.int64(), nullable=False)])
    _make_table(db, "t", pa.table({"id": pa.array([1, 2, 3], pa.int64())}, schema=sch))
    _make_table(db, "u", pa.table({"id": pa.array([10, 20], pa.int64())}, schema=sch))
    con = icefalldb.attach(
        str(db), tables=["t", "u"], engine="icefalldb", result_cache_mb=16
    )

    # Cache a (non-aggregate, DuckDB-routed) SELECT on u.
    u_sql = "SELECT id FROM u WHERE id > 0 ORDER BY id"
    assert con.sql(u_sql).fetchall() == [(10,), (20,)]
    con.sql(u_sql)  # warm
    cache_dir = db / "_query_cache"
    u_entry = set(cache_dir.glob("*.arrow"))
    assert len(u_entry) == 1, "u's SELECT should have written one cache entry"

    con.mutate("DELETE FROM t WHERE id = 1")  # mutate an unrelated table

    # u's cached entry survives the mutation (no wholesale wipe), file set unchanged.
    assert set(cache_dir.glob("*.arrow")) == u_entry, (
        "mutation on t wiped u's cached result - the router still wipes wholesale"
    )
    assert con.sql(u_sql).fetchall() == [(10,), (20,)]  # u's result still correct
    # t's SELECT returns fresh data (routes to native after the mutation).
    assert con.sql("SELECT id FROM t ORDER BY id").fetchall() == [(2,), (3,)]


# --------------------------------------------------------------------------- #
# icefalldb is the default engine
# --------------------------------------------------------------------------- #


def test_attach_default_engine_is_icefalldb(tmp_path):
    """attach() with no engine= argument returns a IcefallDBRouter."""
    db = tmp_path / "db"
    db.mkdir()
    _make_table(
        db,
        "t",
        pa.table(
            {"id": pa.array([1], pa.int64())},
            schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
        ),
    )
    assert type(icefalldb.attach(str(db))).__name__ == "IcefallDBRouter"
