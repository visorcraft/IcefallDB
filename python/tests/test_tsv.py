from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pytest

from icefalldb import attach, export_tsv, import_tsv, read_tsv, write_tsv

ICEFALLDB_DEBUG = Path(__file__).resolve().parents[2] / "target" / "debug" / "icefalldb"


def icefalldb_cli() -> Path:
    if not ICEFALLDB_DEBUG.exists():
        pytest.skip(
            f"icefalldb CLI not found at {ICEFALLDB_DEBUG}; run cargo build first"
        )
    return ICEFALLDB_DEBUG


def run_icefalldb(*args):
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


def test_import_export_round_trip_via_cli():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()

        source = Path(tmp) / "source.tsv"
        source.write_text("id\tname\n1\talice\n2\tbob\n", encoding="utf-8")
        import_tsv(db, "products", source)

        exported = Path(tmp) / "products.tsv"
        export_tsv(db, "products", exported)
        assert exported.exists()
        text = exported.read_text(encoding="utf-8")
        assert "id\tname" in text
        assert "1\talice" in text
        assert "2\tbob" in text

        import_tsv(db, "products_copy", exported)
        con = attach(db)
        rows = con.execute(
            'SELECT id, name FROM "products_copy" ORDER BY id'
        ).fetchall()
        assert rows == [(1, "alice"), (2, "bob")]


def test_read_write_tsv_escapes_special_characters():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "special.tsv"
        table = pa.table(
            {
                "id": [1, 2, 3, 4],
                "text": ["a\tb", "c\nd", "e\\f", "g\r\nh"],
            }
        )
        write_tsv(table, path)

        text = path.read_text(encoding="utf-8")
        assert "a\\tb" in text
        assert "c\\nd" in text
        assert "e\\\\f" in text
        assert "g\\r\\nh" in text

        round_trip = read_tsv(path)
        assert round_trip.column_names == ["id", "text"]
        assert round_trip["text"].to_pylist() == [
            "a\tb",
            "c\nd",
            "e\\f",
            "g\r\nh",
        ]


def test_read_write_tsv_null_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nulls.tsv"
        table = pa.table({"id": [1, 2], "name": ["alice", None]})
        write_tsv(table, path)

        text = path.read_text(encoding="utf-8")
        lines = text.rstrip("\n").split("\n")
        assert lines[1] == "1\talice"
        assert lines[2] == "2\t"

        round_trip = read_tsv(path)
        assert round_trip["name"].to_pylist() == ["alice", None]


def test_write_tsv_map_column_as_json_object():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "maps.tsv"
        table = pa.table(
            {
                "id": [1, 2],
                "tags": pa.array(
                    [[("a", 1), ("b", 2)], [("c", 3)]],
                    type=pa.map_(pa.string(), pa.int64()),
                ),
            }
        )
        write_tsv(table, path)

        text = path.read_text(encoding="utf-8")
        assert '{"a":1,"b":2}' in text
        assert '{"c":3}' in text

        round_trip = read_tsv(path)
        assert round_trip["tags"].to_pylist() == [
            '{"a":1,"b":2}',
            '{"c":3}',
        ]


def test_read_tsv_strict_bool_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bools.tsv"
        path.write_text("id\tactive\n1\ttrue\n2\tfalse\n", encoding="utf-8")
        table = read_tsv(path)
        assert table["active"].to_pylist() == [True, False]


def test_read_tsv_non_lowercase_bool_becomes_string():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mixed_bool.tsv"
        path.write_text("id\tactive\n1\ttrue\n2\tTrue\n3\tFALSE\n", encoding="utf-8")
        table = read_tsv(path)
        # Strict parsing means only "true" and "false" are booleans; mixed
        # casing forces the column back to string inference.
        assert table["active"].to_pylist() == ["true", "True", "FALSE"]


def test_read_tsv_rejects_unknown_escape():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad_escape.tsv"
        path.write_text("id\tname\n1\tfoo\\xbar\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown escape"):
            read_tsv(path)


def test_read_tsv_header_escape_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "header.tsv"
        table = pa.table({"col\tname": ["a"], "col\n2": ["b"]})
        write_tsv(table, path)

        text = path.read_text(encoding="utf-8")
        assert "col\\tname\tcol\\n2" in text

        round_trip = read_tsv(path)
        assert round_trip.column_names == ["col\tname", "col\n2"]


def test_read_tsv_rejects_too_few_fields():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "too_few.tsv"
        path.write_text("id\tname\n1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected 2 fields, found 1"):
            read_tsv(path)


def test_read_tsv_rejects_too_many_fields():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "too_many.tsv"
        path.write_text("id\tname\n1\talice\textra\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected 2 fields, found 3"):
            read_tsv(path)
