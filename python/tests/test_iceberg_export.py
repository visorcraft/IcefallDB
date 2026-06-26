from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

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


def _make_parquet(path: Path, table: pa.Table, schema: pa.Schema | None = None) -> None:
    if schema is not None:
        table = pa.table(table.columns, schema=schema)
    pq.write_table(table, path)


def test_iceberg_export_produces_metadata_and_hint():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        output = Path(tmp) / "iceberg"

        schema_path = Path(tmp) / "schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "schema_id": 1,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "name", "type": "utf8", "nullable": True},
                    ],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8(), nullable=True),
            ]
        )
        _make_parquet(
            data_path,
            pa.table({"id": [1, 2], "name": ["alice", "bob"]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        env = os.environ.copy()
        venv_bin = Path(sys.executable).parent
        if (venv_bin / "duckdb").exists():
            env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

        result = subprocess.run(
            [str(icefalldb_cli()), "iceberg-export", str(db), "products", str(output)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert result.returncode == 0, result.stderr

        metadata_path = Path(result.stdout.strip())
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["format-version"] == 2
        assert metadata["current-snapshot-id"] > 0

        metadata_dir = metadata_path.parent
        assert (metadata_dir / "version-hint.text").exists()
        assert (metadata_dir / "version-hint.text").read_text(
            encoding="utf-8"
        ) == metadata_path.name


def test_iceberg_export_metadata_readable_by_pyiceberg():
    pytest.importorskip("pyiceberg", reason="pyiceberg not installed")
    from pyiceberg.table import StaticTable

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "db"
        db.mkdir()
        output = Path(tmp) / "iceberg"

        schema_path = Path(tmp) / "schema.json"
        schema_path.write_text(
            json.dumps(
                {
                    "schema_id": 1,
                    "columns": [
                        {"name": "id", "type": "int64", "nullable": False},
                        {"name": "name", "type": "utf8", "nullable": True},
                    ],
                    "row_group_target_rows": 1_000_000,
                    "row_group_target_bytes": 134_217_728,
                }
            ),
            encoding="utf-8",
        )
        run_icefalldb("create", str(db), "products", "--schema", str(schema_path))

        data_path = Path(tmp) / "products.parquet"
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("name", pa.utf8(), nullable=True),
            ]
        )
        _make_parquet(
            data_path,
            pa.table({"id": [1, 2], "name": ["alice", "bob"]}),
            schema=schema,
        )
        run_icefalldb("insert", str(db), "products", str(data_path))

        env = os.environ.copy()
        venv_bin = Path(sys.executable).parent
        if (venv_bin / "duckdb").exists():
            env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"

        result = subprocess.run(
            [str(icefalldb_cli()), "iceberg-export", str(db), "products", str(output)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        assert result.returncode == 0, result.stderr

        metadata_path = Path(result.stdout.strip())
        assert metadata_path.exists()

        table = StaticTable.from_metadata(str(metadata_path))
        schema = table.schema()
        field_names = [field.name for field in schema.fields]
        assert "id" in field_names
        assert "name" in field_names
