# Encryption

IcefallDB can encrypt a table so that the data on disk is unreadable without a
key. It uses **Parquet Modular Encryption**, a standard part of the Parquet
format (version 2.9 and later), with the widely-trusted AES-GCM cipher. You can
encrypt a whole table or only specific columns - for example, leave most columns
readable but lock down an `ssn` or `email` column. You hold the keys; IcefallDB
never stores them in the table.

You can create and read encrypted tables from the command-line tool and from
Python.

## How keys are supplied

A key is a short piece of random bytes, written in hexadecimal. Use a **16-byte
(AES-128) key**, which is 32 hex characters.

You never name keys directly; IcefallDB derives a key **id** from the table:

- the table's **footer key** id is `<table>-v1`;
- a per-column key id is `<table>-v1:<column>`.

You supply the key *bytes* for those ids in one of two ways:

- **Environment variables** (the default). The variable name is `ICEFALLDB_KEY_`
  followed by the key id upper-cased, with every non-alphanumeric character
  replaced by `_`. So table `orders` reads its footer key from
  `ICEFALLDB_KEY_ORDERS_V1`, and its `ssn` column key from
  `ICEFALLDB_KEY_ORDERS_V1_SSN`.

  ```bash
  export ICEFALLDB_KEY_ORDERS_V1="000102030405060708090a0b0c0d0e0f"
  ```

- **A JSON key file**, passed with `--key-file` (CLI) or `key_file=` (Python):

  ```json
  { "keys": { "orders-v1": "000102030405060708090a0b0c0d0e0f" } }
  ```

Keep keys out of the table directory and out of version control. Anyone with the
files but without the keys sees only ciphertext.

## From the command line

Create an encrypted table and load data in one step with `import`:

```bash
export ICEFALLDB_KEY_ORDERS_V1="000102030405060708090a0b0c0d0e0f"

# encrypt the whole table with the footer key
icefalldb import /tmp/mydb orders orders.tsv --encrypt

# or encrypt only specific columns (the rest stay plaintext)
icefalldb import /tmp/mydb people people.tsv --encrypt-column ssn --encrypt-column email
```

Read it back - just have the key available, in the environment or via a file:

```bash
icefalldb query /tmp/mydb/orders "SELECT category, SUM(amount) FROM orders GROUP BY category"
icefalldb query /tmp/mydb/orders "SELECT * FROM orders" --key-file keys.json
```

Without the key, the read fails rather than returning anything.

## From Python

Create with `import_tsv`, read with `attach`:

```python
import icefalldb

# create + load an encrypted table (keys from ICEFALLDB_KEY_* or key_file=)
icefalldb.import_tsv("/tmp/mydb", "orders", "orders.tsv", encrypt=True)
# per-column:
icefalldb.import_tsv("/tmp/mydb", "people", "people.tsv", encrypt_columns=["ssn"])

# read it back (the default engine routes encrypted tables to the native engine)
con = icefalldb.attach("/tmp/mydb")                        # keys from env vars
con = icefalldb.attach("/tmp/mydb", key_file="keys.json")  # or from a key file
print(con.sql("SELECT COUNT(*) FROM orders").fetchall())
```

Encrypted reads always use the native engine (DuckDB cannot decrypt), and their
results are never written to the on-disk result cache.

## Options

- **Whole table vs. specific columns.** `--encrypt` / `encrypt=True` encrypts
  every column with the footer key. `--encrypt-column C` /
  `encrypt_columns=[...]` encrypts only the named columns, each with its own key,
  and leaves the rest plaintext - useful for protecting just the sensitive
  fields while keeping the rest fast to query.
- **Footer.** By default the Parquet footer is left unencrypted, which keeps
  reads fast (IcefallDB can still use the file's index to skip data) while the
  column values stay encrypted. Pass `--encrypt-footer` / `encrypt_footer=True`
  to encrypt the footer too, at the cost of slower scans.

## What ends up on disk

Alongside the usual `.parquet` data files (with the protected columns
encrypted), IcefallDB writes a small `_encryption.json` marker recording which
key ids and columns are in play. It holds **no key material** and is safe to
commit; its presence is how every part of IcefallDB knows the table is
encrypted.

IcefallDB also suppresses its plaintext **statistics sidecars** for encrypted
columns: the per-fragment min/max/null counts (in `.meta` and checkpoints) and
the aggregate (`.agg`) partials that normally accelerate metadata queries are
omitted for encrypted columns, so they cannot leak the protected values. The
result-cache (`_query_cache`) is likewise disabled for encrypted reads, since it
would store decrypted rows in the clear. One consequence: aggregates over an
encrypted column are computed by scanning rather than from sidecars.

## Embedding in Rust

The same capability is available in the Rust library if you embed IcefallDB
directly: build an `EncryptionWriteConfig` for the writer and an encrypted
session for reads. The runnable examples are
`crates/icefalldb-core/tests/encryption_tests.rs` and
`crates/icefalldb-query/tests/encryption_e2e.rs`. Encryption lives behind a Cargo
`encryption` feature, which is **on by default** for the CLI and the Python
extension (turn it off with `--no-default-features` for a smaller build).

## If the key is wrong or missing

- **Missing key.** Opening an encrypted table without its key fails with a clear
  "key not found" error naming the key id (or the environment variable it looked
  for).
- **Wrong key.** Reading column data with the wrong key fails verification rather
  than returning garbage. With a plaintext footer, non-secret metadata such as
  the row count can still be read - that is the documented trade-off of leaving
  the footer readable.

## Current limits

- **Key rotation** is not implemented: the key ids are fixed when the table is
  created. To change keys, rewrite the table.
- **Appending** to a per-column-encrypted table is not yet supported; create it
  in one `import` step.
- **Mutations** (`DELETE` / `UPDATE` / `MERGE`) on encrypted tables are not yet
  supported through the CLI or Python; encrypted tables are read-only there for
  now.
