from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from icefalldb import (
    attach,
    attach_table,
    create_view,
    import_tsv,
    read_arrow_table,
    refresh_view,
    IcefallDBError,
)
from icefalldb.adapter import _checksum_json

ICEFALLDB_DEBUG = Path(__file__).resolve().parents[2] / "target" / "debug" / "icefalldb"


def icefalldb_cli() -> Path:
    if not ICEFALLDB_DEBUG.exists():
        pytest.skip(
            f"icefalldb CLI not found at {ICEFALLDB_DEBUG}; run cargo build first"
        )
    return ICEFALLDB_DEBUG


def run_icefalldb(*args):
    # The icefalldb CLI may spawn the DuckDB CLI, which is typically installed in
    # the same virtual-environment bin directory as the running Python
    # interpreter. Ensure that directory is on PATH so tests pass even when the
    # venv is not explicitly activated.
    env = os.environ.copy()
    venv_bin = Path(sys.executable).parent
    if (venv_bin / "duckdb").exists():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

    subprocess.run(
        [str(icefalldb_cli()), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def latest_manifest_path(table_dir: Path) -> Path:
    """Return the path to the latest manifest snapshot for a table.

    With the new empty-table init path, the first insert writes sequence 1.
    Tests that read the manifest after an insert should use this helper rather
    than hard-coding sequence 2.
    """
    pointer = json.loads((table_dir / "_manifest.json").read_text(encoding="utf-8"))
    sequence = pointer["latest"]
    return table_dir / "_manifests" / f"{sequence:09d}.json"


def write_schema(path: Path, columns, dropped_columns=None, schema_id: int = 1):
    schema = {
        "schema_id": schema_id,
        "columns": columns,
        "row_group_target_rows": 1_000_000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": dropped_columns or [],
    }
    path.write_text(json.dumps(schema), encoding="utf-8")


def make_parquet(path: Path, table: pa.Table, schema: pa.Schema | None = None):
    if schema is not None:
        table = pa.table(table.columns, schema=schema)
    pq.write_table(table, path)


def test_icefalldb_cli_is_built():
    assert icefalldb_cli().exists(), (
        "cargo build -p icefalldb-cli is required for integration tests"
    )


def test_attach_empty_table():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "empty")

        con = attach(db)
        rows = con.execute('SELECT * FROM "empty"').fetchall()
        assert rows == []
        columns = {c[0] for c in con.execute("DESCRIBE empty").fetchall()}
        assert columns == {"id"}


def test_attach_table_with_rows():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "name", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8(), nullable=True),
            ]
        )
        make_parquet(
            data_path,
            pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        con = attach(db)
        rows = con.execute('SELECT id, name FROM "products" ORDER BY id').fetchall()
        assert rows == [(1, "a"), (2, "b"), (3, "c")]


def test_attach_multiple_row_groups():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(schema_path, [{"name": "id", "type": "int64", "nullable": False}])
        run_icefalldb("create", str(db), "numbers", "--schema", str(schema_path))

        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        for i in range(2):
            path = Path(tmp) / f"batch{i}.parquet"
            make_parquet(
                path, pa.table({"id": [i * 10 + 1, i * 10 + 2]}), schema=schema
            )
            run_icefalldb("insert", str(db), "numbers", str(path))

        con = attach(db)
        rows = con.execute('SELECT id FROM "numbers" ORDER BY id').fetchall()
        assert rows == [(1,), (2,), (11,), (12,)]


def test_attach_drops_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "old_col", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "events", "--schema", str(schema_path))

        data_path = Path(tmp) / "events.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("old_col", pa.utf8(), nullable=True),
            ]
        )
        make_parquet(data_path, pa.table({"id": [1], "old_col": ["x"]}), schema=schema)
        run_icefalldb("insert", str(db), "events", str(data_path))

        # Evolve the schema to drop old_col by writing a new schema file and
        # updating pointers directly; the Python adapter should omit it.
        (db / "events" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "old_col", "type": "utf8", "nullable": True},
                    ],
                    "dropped_columns": ["old_col"],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "events" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        # Manifest must also reference the new schema_id. Reuse the actual
        # row group file written by the CLI so the Parquet data exists.
        manifest = json.loads(
            latest_manifest_path(db / "events").read_text(encoding="utf-8")
        )
        manifest["sequence"] = 2
        manifest["schema_id"] = 2
        manifest["checksum"] = ""
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path = db / "events" / "_manifests" / "000000002.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # The row group meta must also be updated to the new schema_id and its
        # meta checksum recomputed so the adapter accepts the snapshot. The
        # Parquet data checksum is unchanged because the data file is unchanged.
        for rg in manifest["row_groups"]:
            meta_path = db / "events" / rg["meta"]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["schema_id"] = 2
            original_checksum = meta["checksum"]
            meta["checksum"] = ""
            meta["meta_checksum"] = ""
            meta["meta_checksum"] = _checksum_json(meta)
            meta["checksum"] = original_checksum
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

        (db / "events" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        con = attach(db)
        columns = {c[0] for c in con.execute("DESCRIBE events").fetchall()}
        assert columns == {"id"}
        rows = con.execute('SELECT id FROM "events"').fetchall()
        assert rows == [(1,)]


def test_attach_missing_columns_null_filled():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "old_col", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "evolve", "--schema", str(schema_path))

        data_path = Path(tmp) / "evolve.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("old_col", pa.utf8(), nullable=True),
            ]
        )
        make_parquet(
            data_path,
            pa.table({"id": [1, 2], "old_col": ["a", "b"]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "evolve", str(data_path))

        # Evolve the schema to drop old_col and add a new column. Old row groups
        # do not contain the new column. Because a column has been dropped, the
        # dummy relation is retained and fills the missing new column with NULL.
        (db / "evolve" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "old_col", "type": "utf8", "nullable": True},
                        {"name": "name", "type": "utf8", "nullable": True},
                    ],
                    "dropped_columns": ["old_col"],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "evolve" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        # The insert created manifest sequence 1; create sequence 2 with the new schema.
        manifest = json.loads(
            latest_manifest_path(db / "evolve").read_text(encoding="utf-8")
        )
        manifest["sequence"] = 2
        manifest["schema_id"] = 2
        manifest["checksum"] = ""
        manifest["checksum"] = _checksum_json(manifest)
        (db / "evolve" / "_manifests" / "000000002.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        for rg in manifest["row_groups"]:
            meta_path = db / "evolve" / rg["meta"]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["schema_id"] = 2
            original_checksum = meta["checksum"]
            meta["checksum"] = ""
            meta["meta_checksum"] = ""
            meta["meta_checksum"] = _checksum_json(meta)
            meta["checksum"] = original_checksum
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

        (db / "evolve" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        con = attach(db)
        rows = con.execute('SELECT id, name FROM "evolve" ORDER BY id').fetchall()
        assert rows == [(1, None), (2, None)]


def test_attach_empty_table_with_dropped_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "old_col", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "events", "--schema", str(schema_path))

        # Evolve the schema to drop old_col without inserting any rows.
        (db / "events" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "old_col", "type": "utf8", "nullable": True},
                    ],
                    "dropped_columns": ["old_col"],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "events" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        # Point the manifest at the new schema with no row groups.
        manifest = {
            "sequence": 2,
            "schema_id": 2,
            "row_groups": [],
            "checksum": "",
        }
        manifest["checksum"] = _checksum_json(manifest)
        (db / "events" / "_manifests" / "000000002.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (db / "events" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        con = attach(db)
        rows = con.execute('SELECT * FROM "events"').fetchall()
        assert rows == []
        columns = {c[0]: c[1] for c in con.execute("DESCRIBE events").fetchall()}
        assert columns == {"id": "BIGINT"}


def _view_sql(con, view_name: str) -> str:
    """Return the SQL definition for a DuckDB view."""
    rows = con.execute(
        "SELECT sql FROM duckdb_views() WHERE view_name = ?", [view_name]
    ).fetchall()
    assert rows, f"view {view_name!r} not found"
    return rows[0][0]


def test_attach_view_without_dummy_union_when_no_dropped_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "name", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8(), nullable=True),
            ]
        )
        make_parquet(
            data_path,
            pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        con = attach(db, engine="duckdb")
        sql = _view_sql(con, "products")
        assert "UNION ALL" not in sql
        assert "read_parquet" in sql
        rows = con.execute('SELECT id, name FROM "products" ORDER BY id').fetchall()
        assert rows == [(1, "a"), (2, "b"), (3, "c")]


def test_attach_view_includes_dummy_union_when_dropped_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "old_col", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "events", "--schema", str(schema_path))

        data_path = Path(tmp) / "events.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("old_col", pa.utf8(), nullable=True),
            ]
        )
        make_parquet(data_path, pa.table({"id": [1], "old_col": ["x"]}), schema=schema)
        run_icefalldb("insert", str(db), "events", str(data_path))

        # Evolve the schema to drop old_col.
        (db / "events" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "old_col", "type": "utf8", "nullable": True},
                    ],
                    "dropped_columns": ["old_col"],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "events" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        manifest = json.loads(
            latest_manifest_path(db / "events").read_text(encoding="utf-8")
        )
        manifest["sequence"] = 2
        manifest["schema_id"] = 2
        manifest["checksum"] = ""
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path = db / "events" / "_manifests" / "000000002.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        for rg in manifest["row_groups"]:
            meta_path = db / "events" / rg["meta"]
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["schema_id"] = 2
            original_checksum = meta["checksum"]
            meta["checksum"] = ""
            meta["meta_checksum"] = ""
            meta["meta_checksum"] = _checksum_json(meta)
            meta["checksum"] = original_checksum
            meta_path.write_text(json.dumps(meta), encoding="utf-8")

        (db / "events" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        con = attach(db, engine="duckdb")
        sql = _view_sql(con, "events")
        assert "UNION ALL BY NAME" in sql
        rows = con.execute('SELECT id FROM "events"').fetchall()
        assert rows == [(1,)]


def test_manifest_checksum_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "bad")

        # The new init path leaves the table empty (latest == 0). Create a valid
        # empty manifest at sequence 1 so we can corrupt its checksum.
        manifest = {
            "format_version": 1,
            "sequence": 1,
            "schema_id": 1,
            "row_groups": [],
            "checksum": "",
        }
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path = db / "bad" / "_manifests" / "000000001.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (db / "bad" / "_manifest.json").write_text(
            json.dumps({"latest": 1}), encoding="utf-8"
        )

        manifest["checksum"] = (
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(IcefallDBError, match="checksum mismatch"):
            attach(db)


def test_attach_table_only_one_table():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "a")
        run_icefalldb("create", str(db), "b")

        con = duckdb.connect()
        attach_table(db, "a", connection=con)
        with pytest.raises(duckdb.CatalogException):
            con.execute('SELECT * FROM "b"').fetchall()


def test_existing_connection_reused():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "solo")
        con = duckdb.connect()
        returned = attach(db, connection=con, engine="duckdb")
        assert returned is con


def test_checksum_json_escapes_non_ascii_like_rust():
    """Rust's serde_json::to_string escapes non-ASCII; the adapter must match."""
    value = {"partition": "café"}
    expected_payload = '{"partition":"caf\\u00e9"}'.encode("utf-8")
    expected = f"sha256:{hashlib.sha256(expected_payload).hexdigest()}"
    assert _checksum_json(value) == expected


def test_attach_table_rejects_path_traversal_name():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "victims")

        with pytest.raises(IcefallDBError, match="table name"):
            attach_table(db, "../other")


def test_malformed_manifest_pointer_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "bad_pointer")

        (db / "bad_pointer" / "_manifest.json").write_text(
            json.dumps({"latest": "not-an-int"}), encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="latest must be an integer"):
            attach(db)


def test_missing_row_group_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "missing_rg")

        manifest = {
            "format_version": 1,
            "sequence": 1,
            "schema_id": 1,
            "row_groups": [],
            "checksum": "",
        }
        manifest["row_groups"].append(
            {"data": "does_not_exist.parquet", "meta": "does_not_exist.meta"}
        )
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path = db / "missing_rg" / "_manifests" / "000000001.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (db / "missing_rg" / "_manifest.json").write_text(
            json.dumps({"latest": 1}), encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="missing row group file"):
            attach(db)


def test_row_group_path_escape_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "escape_rg")

        # Place a decoy file outside the table directory to make the escape attempt
        # plausible.
        outside = Path(tmp) / "outside.parquet"
        outside.write_bytes(b"not parquet")

        manifest = {
            "format_version": 1,
            "sequence": 1,
            "schema_id": 1,
            "row_groups": [],
            "checksum": "",
        }
        manifest["row_groups"].append(
            {"data": "../outside.parquet", "meta": "../outside.meta"}
        )
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path = db / "escape_rg" / "_manifests" / "000000001.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        (db / "escape_rg" / "_manifest.json").write_text(
            json.dumps({"latest": 1}), encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="escapes table directory"):
            attach(db)


def test_meta_checksum_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "bad_meta")

        data_path = Path(tmp) / "bad_meta.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1]}), schema=schema)
        run_icefalldb("insert", str(db), "bad_meta", str(data_path))

        manifest = json.loads(
            latest_manifest_path(db / "bad_meta").read_text(encoding="utf-8")
        )
        meta_path = db / "bad_meta" / manifest["row_groups"][0]["meta"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["meta_checksum"] = (
            "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        )
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(IcefallDBError, match="meta checksum mismatch"):
            attach(db)


def test_row_group_schema_id_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "schema_mismatch")

        data_path = Path(tmp) / "schema_mismatch.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1]}), schema=schema)
        run_icefalldb("insert", str(db), "schema_mismatch", str(data_path))

        manifest = json.loads(
            latest_manifest_path(db / "schema_mismatch").read_text(encoding="utf-8")
        )
        meta_path = db / "schema_mismatch" / manifest["row_groups"][0]["meta"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["schema_id"] = 999
        meta["meta_checksum"] = ""
        meta["checksum"] = ""
        meta["meta_checksum"] = _checksum_json(meta)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(IcefallDBError, match="schema_id mismatch"):
            attach(db)


def test_create_view_writes_definition():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "products")

        view_file = create_view(db, "top_products", "SELECT * FROM products")
        assert view_file.exists()
        assert view_file.read_text(encoding="utf-8") == "SELECT * FROM products"


def test_create_view_rejects_non_select():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "products")

        with pytest.raises(IcefallDBError, match="single SELECT"):
            create_view(db, "bad", "DROP TABLE products")


def test_create_view_rejects_reserved_name():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "products")

        with pytest.raises(IcefallDBError, match="reserved table name"):
            create_view(db, "views", "SELECT * FROM products")


def test_refresh_view_materializes_query():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "value", "type": "int64", "nullable": False},
            ],
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("value", pa.int64(), nullable=False),
            ]
        )
        make_parquet(
            data_path,
            pa.table({"id": [1, 2, 3, 4], "value": [5, 15, 25, 35]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        create_view(
            db,
            "top_products",
            "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
        )
        refresh_view(db, "top_products")

        con = attach(db)
        rows = con.execute(
            'SELECT id, value FROM "top_products" ORDER BY id'
        ).fetchall()
        assert rows == [(2, 15), (3, 25), (4, 35)]


def test_attach_empty_table_with_latest_zero():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "empty")

        # Simulate the legacy empty-table pointer.
        (db / "empty" / "_manifest.json").write_text(
            json.dumps({"latest": 0}), encoding="utf-8"
        )

        con = attach(db)
        rows = con.execute('SELECT * FROM "empty"').fetchall()
        assert rows == []
        columns = {c[0] for c in con.execute("DESCRIBE empty").fetchall()}
        assert columns == {"id"}


def test_meta_path_traversal_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "meta_escape")

        data_path = Path(tmp) / "meta_escape.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1]}), schema=schema)
        run_icefalldb("insert", str(db), "meta_escape", str(data_path))

        # Create a decoy meta file outside the table directory.
        outside = Path(tmp) / "outside.json"
        outside.write_text(json.dumps({"row_group": "x"}), encoding="utf-8")

        manifest_path = latest_manifest_path(db / "meta_escape")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["row_groups"][0]["meta"] = "../outside.json"
        manifest["checksum"] = ""
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(IcefallDBError, match="escapes table directory"):
            attach(db)


def test_parquet_checksum_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "bad_data")

        data_path = Path(tmp) / "bad_data.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1]}), schema=schema)
        run_icefalldb("insert", str(db), "bad_data", str(data_path))

        manifest = json.loads(
            latest_manifest_path(db / "bad_data").read_text(encoding="utf-8")
        )
        rg_data_path = db / "bad_data" / manifest["row_groups"][0]["data"]
        rg_data_path.write_bytes(rg_data_path.read_bytes() + b"corruption")

        with pytest.raises(IcefallDBError, match="data checksum mismatch"):
            attach(db, verify_data_checksums=True)


def test_dropped_columns_none_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "null_dropped")

        (db / "null_dropped" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [{"name": "id", "type": "int64", "nullable": False}],
                    "dropped_columns": None,
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "null_dropped" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )
        manifest = {
            "sequence": 2,
            "schema_id": 2,
            "row_groups": [],
            "checksum": "",
        }
        manifest["checksum"] = _checksum_json(manifest)
        (db / "null_dropped" / "_manifests" / "000000002.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (db / "null_dropped" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="dropped_columns"):
            attach(db)


def test_dropped_columns_non_string_entry_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "bad_dropped")

        (db / "bad_dropped" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [{"name": "id", "type": "int64", "nullable": False}],
                    "dropped_columns": [123],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "bad_dropped" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )
        manifest = {
            "sequence": 2,
            "schema_id": 2,
            "row_groups": [],
            "checksum": "",
        }
        manifest["checksum"] = _checksum_json(manifest)
        (db / "bad_dropped" / "_manifests" / "000000002.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (db / "bad_dropped" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="dropped column entry"):
            attach(db)


def test_attach_with_bare_string_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "single")

        con = attach(db, tables="single")
        rows = con.execute('SELECT * FROM "single"').fetchall()
        assert rows == []


def test_attach_rejects_non_string_table_name():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "typed")

        with pytest.raises(IcefallDBError, match="table name must be a string"):
            attach(db, tables=[123])


def test_schema_with_quoted_column_name_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "evil_col")

        (db / "evil_col" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [
                        {"name": 'evil"col', "type": "int64", "nullable": False}
                    ],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "evil_col" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )
        manifest = {
            "sequence": 2,
            "schema_id": 2,
            "row_groups": [],
            "checksum": "",
        }
        manifest["checksum"] = _checksum_json(manifest)
        (db / "evil_col" / "_manifests" / "000000002.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (db / "evil_col" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="invalid identifier"):
            attach(db)


def test_absolute_data_path_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "abs_data")

        data_path = Path(tmp) / "abs_data.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1]}), schema=schema)
        run_icefalldb("insert", str(db), "abs_data", str(data_path))

        manifest_path = latest_manifest_path(db / "abs_data")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["row_groups"][0]["data"] = "/etc/passwd"
        manifest["checksum"] = ""
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(IcefallDBError, match="absolute row group data path"):
            attach(db)


def test_absolute_meta_path_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "abs_meta")

        data_path = Path(tmp) / "abs_meta.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1]}), schema=schema)
        run_icefalldb("insert", str(db), "abs_meta", str(data_path))

        manifest_path = latest_manifest_path(db / "abs_meta")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["row_groups"][0]["meta"] = "/etc/passwd"
        manifest["checksum"] = ""
        manifest["checksum"] = _checksum_json(manifest)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(IcefallDBError, match="absolute row group meta path"):
            attach(db)


def test_refresh_view_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "value", "type": "int64", "nullable": False},
            ],
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("value", pa.int64(), nullable=False),
            ]
        )
        make_parquet(
            data_path,
            pa.table({"id": [1, 2, 3, 4], "value": [5, 15, 25, 35]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        create_view(
            db,
            "top_products",
            "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
        )
        refresh_view(db, "top_products")
        refresh_view(db, "top_products")

        con = attach(db)
        rows = con.execute(
            'SELECT id, value FROM "top_products" ORDER BY id'
        ).fetchall()
        assert rows == [(2, 15), (3, 25), (4, 35)]


def test_refresh_view_timestamp_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "ts", "type": "timestamp[us]", "nullable": False},
            ],
        )
        run_icefalldb("create", str(db), "events", "--schema", str(schema_path))

        data_path = Path(tmp) / "events.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("ts", pa.timestamp("us"), nullable=False),
            ]
        )
        make_parquet(
            data_path,
            pa.table(
                {
                    "id": [1, 2],
                    "ts": [
                        datetime.datetime(2024, 1, 1, 12, 0, 0),
                        datetime.datetime(2024, 1, 2, 12, 0, 0),
                    ],
                }
            ),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "events", str(data_path))

        create_view(db, "ts_view", "SELECT id, ts FROM events ORDER BY id")
        refresh_view(db, "ts_view")

        con = attach(db)
        rows = con.execute('SELECT id, ts FROM "ts_view" ORDER BY id').fetchall()
        assert len(rows) == 2
        assert rows[0][0] == 1
        assert rows[1][0] == 2


def test_create_view_accepts_semicolon_in_string_literal():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "products")

        view_file = create_view(db, "semi_view", "SELECT ';' AS semi FROM products")
        assert (
            view_file.read_text(encoding="utf-8") == "SELECT ';' AS semi FROM products"
        )


def test_refresh_view_missing_definition_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "products")

        with pytest.raises(IcefallDBError, match="not found"):
            refresh_view(db, "does_not_exist")


def test_refresh_view_missing_icefalldb_cli_raises(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        run_icefalldb("create", str(db), "products")
        create_view(db, "v", "SELECT * FROM products")

        monkeypatch.setattr("icefalldb.adapter._find_icefalldb_cli", lambda: None)
        with pytest.raises(IcefallDBError, match="icefalldb CLI not found"):
            refresh_view(db, "v")


def test_refresh_view_empty_source_table():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "value", "type": "int64", "nullable": False},
            ],
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        create_view(
            db,
            "empty_view",
            "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
        )
        refresh_view(db, "empty_view")

        con = attach(db)
        rows = con.execute('SELECT * FROM "empty_view"').fetchall()
        assert rows == []


def test_refresh_view_source_schema_change_raises():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(schema_path, [{"name": "id", "type": "int64", "nullable": False}])
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema([pa.field("id", pa.int64(), nullable=False)])
        make_parquet(data_path, pa.table({"id": [1, 2]}), schema=schema)
        run_icefalldb("insert", str(db), "products", str(data_path))

        create_view(db, "narrow", "SELECT id FROM products")
        refresh_view(db, "narrow")

        # Evolve the source table schema by adding a column and commit a new row
        # group that actually contains both columns.
        (db / "products" / "_schemas" / "000002.json").write_text(
            json.dumps(
                {
                    "schema_id": 2,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "value", "type": "int64", "nullable": True},
                    ],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        (db / "products" / "_schema.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        evolved_path = db / "products" / "rg_evolved.parquet"
        evolved_schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("value", pa.int64(), nullable=True),
            ]
        )
        make_parquet(
            evolved_path,
            pa.table({"id": [1, 2], "value": [10, 20]}),
            schema=evolved_schema,
        )
        parquet_bytes = evolved_path.read_bytes()
        data_checksum = f"sha256:{hashlib.sha256(parquet_bytes).hexdigest()}"
        meta = {
            "row_group": "rg_evolved",
            "schema_id": 2,
            "rows": 2,
            "columns": {
                "id": {"min": 1, "max": 2, "nulls": 0},
                "value": {"min": 10, "max": 20, "nulls": 0},
            },
            "checksum": data_checksum,
            "meta_checksum": "",
        }
        meta["meta_checksum"] = _checksum_json(meta)
        (db / "products" / "rg_evolved.meta").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        manifest = {
            "format_version": 1,
            "sequence": 2,
            "schema_id": 2,
            "row_groups": [{"data": "rg_evolved.parquet", "meta": "rg_evolved.meta"}],
            "checksum": "",
        }
        manifest["checksum"] = _checksum_json(manifest)
        (db / "products" / "_manifests" / "000000002.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        (db / "products" / "_manifest.json").write_text(
            json.dumps({"latest": 2}), encoding="utf-8"
        )

        # Update the view definition to select the new column.
        (db / "views" / "narrow.sql").write_text(
            "SELECT id, value FROM products", encoding="utf-8"
        )

        with pytest.raises(IcefallDBError, match="schema"):
            refresh_view(db, "narrow")


def test_attach_table_with_map_column():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "tags", "type": "map<utf8,list<int64>>", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "events", "--schema", str(schema_path))

        # Import via the schema-aware CLI TSV importer, which constructs the
        # correct Arrow field names (e.g. "item" for list elements) that the
        # Rust writer expects.
        tsv_path = Path(tmp) / "events.tsv"
        tsv_path.write_text(
            'id\ttags\n1\t{"a":[1,2],"b":[3]}\n2\t{"c":[4,5,6]}\n',
            encoding="utf-8",
        )
        import_tsv(db, "events", tsv_path)

        con = attach(db)
        rows = con.execute('SELECT id, tags FROM "events" ORDER BY id').fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, {"a": [1, 2], "b": [3]})
        assert rows[1] == (2, {"c": [4, 5, 6]})


def test_read_arrow_table_returns_expected_rows_and_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        schema_path = Path(tmp) / "schema.json"
        write_schema(
            schema_path,
            [
                {"name": "id", "type": "int64", "nullable": False},
                {"name": "name", "type": "utf8", "nullable": True},
            ],
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8(), nullable=True),
            ]
        )
        make_parquet(
            data_path,
            pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        table = read_arrow_table(db, table="products", verify_data_checksums=False)
        assert table.num_rows == 3
        assert table.column_names == ["id", "name"]
        assert table.column("id").to_pylist() == [1, 2, 3]
        assert table.column("name").to_pylist() == ["a", "b", "c"]
