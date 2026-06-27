"""Encrypted-table create + read through the Python adapter.

`import_tsv(encrypt=...)` shells to the `icefalldb` CLI to create the table;
`attach(...)` reads it back via the native engine, resolving keys from a key
file or `ICEFALLDB_KEY_*` env vars. Skipped unless both the native extension and
an encryption-enabled CLI binary are available.
"""

import json
import os
import pathlib

import pytest

pytest.importorskip("duckdb")

import icefalldb

# 16-byte (AES-128) key. The footer key id for table `orders` at schema_id 1 is
# `orders-v1`, so the env var is `ICEFALLDB_KEY_ORDERS_V1`.
KEY = "000102030405060708090a0b0c0d0e0f"


def _native_available() -> bool:
    try:
        import icefalldb_query_py  # noqa: F401

        return True
    except Exception:
        return False


def _locate_cli():
    env = os.environ.get("ICEFALLDB_CLI")
    if env and pathlib.Path(env).exists():
        return env
    repo = pathlib.Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        cand = repo / "target" / profile / "icefalldb"
        if cand.exists():
            return str(cand)
    return None


_CLI = _locate_cli()

pytestmark = pytest.mark.skipif(
    not (_native_available() and _CLI),
    reason="encryption needs the native extension and an icefalldb CLI binary",
)


def _make_table(tmp_path, monkeypatch, **import_kwargs):
    monkeypatch.setenv("ICEFALLDB_CLI", _CLI)
    monkeypatch.setenv("ICEFALLDB_KEY_ORDERS_V1", KEY)
    db = tmp_path / "db"
    tsv = tmp_path / "orders.tsv"
    tsv.write_text(
        "order_id\tcategory\tamount\n1\tbooks\t9.99\n2\tgames\t39.50\n3\tbooks\t14.00\n"
    )
    icefalldb.import_tsv(str(db), "orders", str(tsv), **import_kwargs)
    return db


def test_create_encrypted_and_read_with_env_key(tmp_path, monkeypatch):
    db = _make_table(tmp_path, monkeypatch, encrypt=True)

    # the table is marked encrypted and the data is encrypted at rest
    assert (db / "orders" / "_encryption.json").is_file()
    rg = next(p for p in (db / "orders").iterdir() if p.suffix == ".parquet")
    assert b"books" not in rg.read_bytes()

    # read it back through the default (router) engine using the env key
    con = icefalldb.attach(str(db))
    rows = dict(
        con.sql("SELECT category, SUM(amount) FROM orders GROUP BY category").fetchall()
    )
    assert round(rows["books"], 2) == 23.99
    assert rows["games"] == 39.5


def test_read_with_key_file(tmp_path, monkeypatch):
    db = _make_table(tmp_path, monkeypatch, encrypt=True)

    # drop the env key and read via a JSON key file instead
    monkeypatch.delenv("ICEFALLDB_KEY_ORDERS_V1", raising=False)
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps({"keys": {"orders-v1": KEY}}))

    con = icefalldb.attach(str(db), engine="datafusion", key_file=str(key_file))
    assert con.sql("SELECT COUNT(*) FROM orders").fetchall()[0][0] == 3


def test_read_without_key_fails(tmp_path, monkeypatch):
    db = _make_table(tmp_path, monkeypatch, encrypt=True)

    monkeypatch.delenv("ICEFALLDB_KEY_ORDERS_V1", raising=False)
    with pytest.raises(Exception):
        con = icefalldb.attach(str(db), engine="datafusion")
        con.sql("SELECT category FROM orders").fetchall()


def test_per_column_encryption(tmp_path, monkeypatch):
    monkeypatch.setenv("ICEFALLDB_CLI", _CLI)
    monkeypatch.setenv("ICEFALLDB_KEY_PEOPLE_V1", KEY)
    monkeypatch.setenv(
        "ICEFALLDB_KEY_PEOPLE_V1_SSN", "0f0e0d0c0b0a09080706050403020100"
    )
    db = tmp_path / "db"
    tsv = tmp_path / "people.tsv"
    tsv.write_text("name\tssn\nalice\t111-22-3333\nbob\t444-55-6666\n")
    icefalldb.import_tsv(str(db), "people", str(tsv), encrypt_columns=["ssn"])

    marker = json.loads((db / "people" / "_encryption.json").read_text())
    assert marker["column_key_ids"] == {"ssn": "people-v1:ssn"}

    con = icefalldb.attach(str(db))
    rows = con.sql("SELECT name, ssn FROM people ORDER BY name").fetchall()
    assert rows == [("alice", "111-22-3333"), ("bob", "444-55-6666")]


def test_duckdb_engine_rejects_encrypted(tmp_path, monkeypatch):
    db = _make_table(tmp_path, monkeypatch, encrypt=True)
    with pytest.raises(Exception):
        icefalldb.attach(str(db), engine="duckdb")


def test_no_plaintext_result_cache_for_encrypted(tmp_path, monkeypatch):
    db = _make_table(tmp_path, monkeypatch, encrypt=True)
    con = icefalldb.attach(str(db), engine="datafusion")
    con.sql("SELECT category, SUM(amount) FROM orders GROUP BY category").fetchall()
    cache = db / "_query_cache"
    leaked = list(cache.glob("*.arrow")) if cache.exists() else []
    assert leaked == [], f"encrypted query wrote plaintext cache files: {leaked}"


def test_per_column_plaintext_query_without_column_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ICEFALLDB_CLI", _CLI)
    monkeypatch.setenv("ICEFALLDB_KEY_PEOPLE_V1", KEY)
    db = tmp_path / "db"
    tsv = tmp_path / "people.tsv"
    tsv.write_text("name\tssn\nalice\t111-22-3333\nbob\t444-55-6666\n")
    # Import with per-column encryption on `ssn`; both keys present for writing.
    monkeypatch.setenv(
        "ICEFALLDB_KEY_PEOPLE_V1_SSN", "0f0e0d0c0b0a09080706050403020100"
    )
    icefalldb.import_tsv(str(db), "people", str(tsv), encrypt_columns=["ssn"])
    monkeypatch.delenv("ICEFALLDB_KEY_PEOPLE_V1_SSN")

    # Reading the plaintext `name` column should succeed with only the footer key.
    con = icefalldb.attach(str(db), engine="datafusion")
    rows = con.sql("SELECT name FROM people ORDER BY name").fetchall()
    assert rows == [("alice",), ("bob",)]

    # Reading the encrypted `ssn` column without its key must fail.
    with pytest.raises(Exception):
        con.sql("SELECT ssn FROM people").fetchall()


def test_encrypted_table_time_travel_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ICEFALLDB_CLI", _CLI)
    monkeypatch.setenv("ICEFALLDB_KEY_ORDERS_V1", KEY)
    db = tmp_path / "db"
    tsv1 = tmp_path / "orders1.tsv"
    tsv2 = tmp_path / "orders2.tsv"
    tsv1.write_text("order_id\tcategory\tamount\n1\tbooks\t9.99\n")
    tsv2.write_text("order_id\tcategory\tamount\n2\tgames\t39.50\n")

    icefalldb.import_tsv(str(db), "orders", str(tsv1), encrypt=True)
    icefalldb.import_tsv(str(db), "orders", str(tsv2), encrypt=True)

    latest = icefalldb.attach(str(db), engine="datafusion")
    asof = icefalldb.attach(str(db), engine="datafusion", snapshot=1)

    latest_count = latest.sql("SELECT COUNT(*) FROM orders").fetchall()[0][0]
    asof_count = asof.sql("SELECT COUNT(*) FROM orders").fetchall()[0][0]
    assert latest_count == 2
    assert asof_count == 1

    with pytest.raises(Exception):
        asof.mutate("DELETE FROM orders WHERE order_id = 1")
