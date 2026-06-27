<p align="center">
  <img src="assets/icefalldb-mark.png" alt="IcefallDB logo" width="150" />
</p>

<h1 align="center">IcefallDB</h1>

<p align="center">
  <b>Filesystem-native analytical tables with row-level mutations and a warm-aggregate cache.</b>
  <br />
  Plain Parquet files plus plain JSON metadata, immutable snapshots, and a native DataFusion engine.
  <br />
  Fast Rust core · DELETE / UPDATE / MERGE · zero-copy ingest · offline compaction / GC · Python adapter.
</p>

<p align="center">
  <a href="https://github.com/visorcraft/IcefallDB/releases/latest"><img src="https://img.shields.io/github/v/release/visorcraft/IcefallDB?sort=semver" alt="Latest release" /></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg" alt="License: MIT OR Apache-2.0" /></a>
  <img src="https://img.shields.io/badge/built%20with-Rust-000000?logo=rust&amp;logoColor=white" alt="Built with Rust" />
  <img src="https://img.shields.io/badge/engine-DataFusion%2054-4B8BBE" alt="DataFusion 54" />
  <img src="https://img.shields.io/badge/platform-Linux-333333?logo=linux&amp;logoColor=white" alt="Platform: Linux" />
</p>

---

## What is IcefallDB?

Each IcefallDB table is an ordinary directory of Parquet files plus plain-JSON
schema, immutable numbered manifests, and sidecar statistics. You can inspect it
with `ls`, `cat`, `jq`, or `git`; there is no opaque binary state. On top of that
layout it runs a native **DataFusion 54** SQL engine with two capabilities you
don't usually get from a plain-files table format:

- **Row-level mutations.** `DELETE` / `UPDATE` / `MERGE` (upsert) without ever
  rewriting a Parquet file in place. Mutations use move-stable `u64` row IDs,
  per-fragment deletion vectors, and append-only patch fragments; every commit is
  atomic and crash-safe.
- **An incremental partial-aggregate cache.** Each fragment carries an `.agg`
  sidecar of additive partials, so warm `SUM/COUNT/AVG/VAR/STDDEV/MIN/MAX` and
  declared-key `GROUP BY` are answered by composing cached partials with
  zero/sparse I/O, and stay exact across deletions and compaction.

## Why IcefallDB?

- **Your data stays open.** Tables are just Parquet + JSON on disk, so they are
  inspectable, diffable, and version-control friendly. No proprietary container,
  no vendor lock-in, no opaque write-ahead blob you can't read.
- **Mutations on a plain-file lake.** Atomic row-level `DELETE`/`UPDATE`/`MERGE`
  on Parquet without rewriting whole files, something a plain-Parquet dataset
  normally cannot do at all.
- **Aggregates answered from metadata.** `COUNT(*)`, `MIN`, `MAX`, and warm
  `SUM`/`AVG`/`STDDEV` come back in sub-millisecond time by composing sidecar
  partials instead of scanning, and the answers are exact, not estimates.
- **Auditable history.** Immutable numbered snapshots, hash-chained manifests,
  and snapshot-addressable time-travel reads, all verifiable with `doctor`.
- **Crash-safe and concurrent.** WAL fast-commit makes a `DELETE` cost as little
  as one fsync; a single-writer lock keeps readers on consistent snapshots.
- **Embed it anywhere.** A fast Rust core and CLI, a PyO3 Python adapter, and an
  optional HTTP SQL server, all over the same files.

## Performance

Warm `p50` on a 16-core workstation, the default engine (`engine="icefalldb"`),
`events` = 1M rows and `events_wide` = 10M rows. **First run** is the query
executed fresh; **cached** is the same query served again from the result cache
(or, for unfiltered metadata aggregates, composed from sidecar statistics with no
scan at all). Both are on by default.

| Query shape | first run | cached |
|---|---:|---:|
| `COUNT(*)` over 10M rows (from metadata) | 0.1 ms | **0.1 ms** |
| filtered scan returning rows, 2 predicates (1M) | 7.5 ms | **4.8 ms** |
| `GROUP BY category` + `AVG` (1M rows) | 3.8 ms | **0.13 ms** |
| indexed equality, `COUNT` + `SUM` | 2.4 ms | **0.07 ms** |
| sorted time-window `GROUP BY` | 4.2 ms | **0.10 ms** |
| 100M x 10 join + `GROUP BY` | 4.9 ms | **0.12 ms** |
| clustered wide filter, 4 predicates (10M) | 19 ms | **0.07 ms** |
| wide filter, 4-predicate `COUNT` (10M) | 44 ms | **0.07 ms** |
| wide aggregate, 6 aggregates + 3-predicate filter (10M) | 90 ms | **0.12 ms** |

IcefallDB is built for the analytical queries that keep coming back. The first run
pays for the scan; every repeat comes back from cache in well under a millisecond,
so a 6-aggregate scan over 10M rows drops from ~90 ms to ~0.1 ms. And
`COUNT`/`MIN`/`MAX`/`SUM`/`AVG` over an unfiltered table never scan at all - they
are composed from sidecar statistics, sub-millisecond even on the first run. The
one case the cache cannot speed up is a query that returns many rows (the filtered
scan): there the cost is handing back the result set, not computing it. Aggregate
results are exact - byte-equal for integers, within sketch error for the optional
approximate aggregates.

**Mutations.** A single-row `UPDATE` on a 1M-row table runs in tens of
milliseconds: point predicates locate matched rows through the secondary index,
an in-place `CommitDelta` avoids a full reload, and WAL fast-commit collapses a
`DELETE` to as little as one fsync. `INSERT` appends one fragment at a cost that
is flat in table size, not proportional to it.

## Capabilities

- Plain-file tables with monotonically increasing manifest sequence numbers,
  inspectable and version-control friendly.
- Native DataFusion 54 SQL engine (`icefalldb query`, `engine="datafusion"`):
  sidecar-statistics `COUNT(*)`/`MIN`/`MAX`, file + page-index pruning, hybrid
  native/custom scan, optimizer rules, persistent Arrow-IPC result cache.
- **Mutations:** `DELETE` / `UPDATE` / `MERGE` (MERGE needs a `--unique` key
  index); move-stable row IDs; per-fragment deletion vectors; atomic commits with
  WAL fast-commit by default (one fsync for `DELETE`). Point/`IN` predicates
  locate matched rows through a secondary index (the `_rowid` selection pushdown)
  rather than scanning the fragment.
- **Secondary indexes** (`create-index [--unique]`): canonical JSON B-tree plus a
  mmap'd binary `.idx` for O(1) open and a tiny learned `.model` for affine
  integer keys (constant-size, arithmetic locate); derived and optional. A
  `--unique` index actively enforces uniqueness — creation rejects duplicate live
  keys and INSERT/UPDATE/MERGE reject key collisions under the write lock.
- **Warm-aggregate cache:** `.agg` partials, partial-aggregate pushdown for
  range filters, cross-query reuse, optional approximate `approx_distinct` /
  `approx_percentile_cont` (the `sketches` feature).
- Zero-copy Parquet ingest; content-addressed duplicate detection.
- `icefalldb optimize` / `compact` (ZSTD-1, adaptive encodings, optional sort);
  `gc`; `check` / `doctor` validation and repair.
- TSV import/export (JSON-in-TSV for complex types); one-way Iceberg v2 export.
- **Hash-chained manifests:** each manifest records `parent_hash` (SHA-256 of its
  predecessor's content) and `committed_at` timestamp; `doctor`/`check` verify the
  full chain (genesis and GC-pruned anchors tolerated as `None`).
- **Time-travel reads:** query or attach at any retained snapshot (`icefalldb
  query <table-dir> "<SQL>" --snapshot N`, `attach(db, snapshot=N)`, HTTP `/sql
  {snapshot:N}`); `icefalldb snapshots <db> <table>` lists sequence, timestamp,
  rows, and parent hash for every retained manifest. As-of reads are
  **read-only** and always use the native engine, so a snapshot's deletion
  vectors are correctly applied.
- Read-only S3-compatible object-storage adapter.
- Optional per-table **Parquet Modular Encryption** (Apache Parquet 2.9+).
- HTTP SQL server (`icefalldb-server`) with an optional `/mutate` daemon (CLI
  `--server`, PyO3 `server=`) that pays table-open + engine-startup once across
  many ops; Python adapter and producer profile.

See the [user guide](docs/) for installation, querying, and per-language how-tos.

---

## Setup

**Prerequisites:** Rust >= 1.80; for the Python adapter, Python >= 3.9 with
`pyarrow`.

```sh
git clone https://github.com/visorcraft/IcefallDB.git
cd IcefallDB
cargo build --release -p icefalldb-cli
export PATH="$PWD/target/release:$PATH"
```

## Quick start

```bash
# Create a table from a TSV (schema inferred) and query it
icefalldb import /tmp/mydb orders orders.tsv
icefalldb query /tmp/mydb/orders \
  "SELECT category, SUM(amount), COUNT(*) FROM orders GROUP BY category"

# Mutate it (DELETE / UPDATE / MERGE go through `query`)
icefalldb query /tmp/mydb/orders "DELETE FROM orders WHERE status = 'cancelled'"
icefalldb query /tmp/mydb/orders "UPDATE orders SET amount = amount * 1.1 WHERE category = 'books'"

# Upsert (MERGE needs a unique key index)
icefalldb create-index /tmp/mydb orders order_id --unique
icefalldb query /tmp/mydb/orders \
  "MERGE INTO orders USING (SELECT * FROM (VALUES (1,'books',9.99)) s(order_id,category,amount)) src
   ON orders.order_id = src.order_id
   WHEN MATCHED THEN UPDATE SET amount = src.amount
   WHEN NOT MATCHED THEN INSERT (order_id,category,amount) VALUES (src.order_id,src.category,src.amount)"

# Maintenance
icefalldb optimize /tmp/mydb orders --retain-snapshots 1 --sort order_id
```

From Python (read + mutate over the same files):

```python
import icefalldb

# engine="icefalldb" routes each statement to its fastest path (recommended)
con = icefalldb.attach("/tmp/mydb", engine="icefalldb")
print(con.sql("SELECT COUNT(*), AVG(amount) FROM orders").fetchall())
con.sql("DELETE FROM orders WHERE unpaid")             # routed to the native engine
```

---

## Query engines

A query runs through one of two engines over the same Parquet + JSON files:

- **IcefallDB (`engine="icefalldb"`)** is the recommended default. It routes each
  statement to its fastest path automatically: clean `SELECT`s run on a fast
  vectorized scan path, while `DELETE`/`UPDATE`/`MERGE` and unfiltered metadata
  aggregates (`COUNT(*)`/`MIN`/`MAX`/`SUM`/`AVG`) run on the native engine.
  Encrypted tables and tables with active deletion vectors always run on the
  native engine so reads are always correct.
- **DataFusion (`engine="datafusion"`)** is the native engine directly. It is the
  one that applies deletion vectors and the warm-aggregate cache, and it backs the
  `icefalldb query` CLI and the PyO3 binding.

---

## CLI commands

```text
icefalldb create        <db> <table> [--schema <json>]
icefalldb create-table  <db> <table> [--schema <json>]      # via the catalog
icefalldb drop-table    <db> <table>
icefalldb insert        <db> <table> <file.arrow|file.parquet>
icefalldb import        <db> <table> <file.tsv>
icefalldb export        <db> <table> <file.tsv>
icefalldb create-index  <db> <table> <column> [--unique] [--index-type btree]
icefalldb query         <table-dir | db> "<SQL>" [-t <extra-table>...] [--format json|csv]
icefalldb snapshots     <db> <table>
icefalldb check         <db> <table>
icefalldb doctor        <db> <table> [--repair]
icefalldb compact       <db> <table>
icefalldb optimize      <db> <table> [--retain-snapshots <n>] [--sort <key>]
icefalldb gc            <db> <table> [--retain-snapshots <n>]
icefalldb create-view   <db> <view> <query.sql>
icefalldb refresh-view  <db> <view>
icefalldb iceberg-export <db> <table> <output-dir> [--snapshot <n>]
```

`DELETE` / `UPDATE` / `MERGE` are issued through `icefalldb query` against a
single registered table. HTTP server:

```sh
cargo run -p icefalldb-server -- --host 0.0.0.0 --port 8080 /tmp/mydb
```

---

## Development

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features

# Python gates (venv at python/.venv)
python/.venv/bin/ruff check python
python/.venv/bin/ruff format --check python
python/.venv/bin/python -m pytest python/tests -q
```

`AGENTS.md` is the authoritative source for gates, invariants, and conventions.

Benchmark suites live under `python/benchmarks/`: `datafusion/` (query/throughput
matrix), `mutations/` (write and rewrite cost), and `perf/` (open / commit /
insert-update cost). Agent and contributor conventions and the format invariants
are in [`AGENTS.md`](AGENTS.md).

## Project layout

```text
crates/icefalldb-core/                storage, metadata, writer (mutations, compaction, GC),
                                      reader, rowindex, deletion vectors, agg_cache, encryption
crates/icefalldb-query/               native DataFusion 54 engine + optimizer rules + caches
crates/icefalldb-query-py/            PyO3 extension (sql + mutate)
crates/icefalldb-server/              HTTP SQL server (axum) + optional /mutate daemon
crates/icefalldb-cli/                 `icefalldb` binary
crates/datafusion-encrypted-parquet/  standalone encrypted-Parquet factory for DataFusion
python/                               Python adapter + benchmarks
docs/                                 user guide
```

## Documentation

Full user documentation lives in [`docs/`](docs/):

- [Getting started](docs/getting-started.md)
- [Installation](docs/installation.md)
- [Querying your data](docs/querying.md)
- [The command-line tool](docs/cli.md)
- [Using IcefallDB from Python](docs/python.md)
- [Using IcefallDB from other languages](docs/languages.md)
- [Encryption](docs/encryption.md)
- [Agent conventions](AGENTS.md)

---

## License

IcefallDB is dual-licensed under MIT or Apache-2.0.
