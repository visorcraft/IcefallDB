# Credits & Acknowledgements

IcefallDB stands on a large body of open-source work. This file acknowledges the
principal third-party projects it builds on directly. The complete, versioned
inventory of **every** crate compiled into the shipped binaries — with full
license texts — is in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md),
generated from `Cargo.lock` with `cargo-about`.

## Query engine & columnar data (Rust)

- **Apache DataFusion** (`datafusion` 54) — the SQL engine the native query layer
  is built on, with `datafusion-common`, `datafusion-execution`,
  `datafusion-datasource`, and `datafusion-datasource-parquet`.
- **Apache Arrow** (`arrow` 58, `parquet` 58) — the in-memory columnar format and
  the Parquet reader/writer at the core of every table.
- **object_store** (0.13) — pluggable local / S3 storage backend.
- **sqlparser** (0.62) — SQL parsing in the server.

## Async runtime & concurrency

- **Tokio** (1) — async runtime; with `async-trait` (0.1) and `futures` (0.3).

## Serialization, encoding & hashing

- **Serde** (`serde` / `serde_json` 1) — JSON schema, manifests, and sidecars.
- **rkyv** (0.8) — zero-copy archive for the derived snapshot-checkpoint cache.
- **sha2** (0.11) — content-addressed row-group checksums.
- **roaring** (0.11) — deletion-vector bitmaps; **memmap2** (0.9) — mmap'd binary indexes.
- `base64` (0.22), `hex` (0.4), `bytes` (1).
- **apache-avro** (0.21) — one-way Iceberg v2 export (optional).

## CLI, server & utilities

- **clap** (4) — CLI; **axum** (0.8) — HTTP SQL server; `num_cpus` (1).
- `thiserror` (2), `anyhow` (1), `chrono` (0.4), `uuid` (1), `fs2` (0.4),
  `which` (8), `tracing` / `tracing-subscriber` (0.1 / 0.3), `tempfile` (3).

## Optional features

- **datasketches** (0.3) — approximate aggregates (`sketches` feature).
- **zeroize** (1.9) — key zeroization for Parquet Modular Encryption (`encryption` feature).

## Python adapter

- **DuckDB** (`duckdb` ≥ 1.0) — the default read engine for the Python adapter.
- **Apache Arrow for Python** (`pyarrow` ≥ 14) — zero-copy Arrow interchange.
- **PyO3** (0.28) — Rust ↔ Python bindings for the native extension, built with **maturin**.

## Build & quality tooling

- The **Rust** toolchain and **Cargo** ecosystem; **cargo-about** (this license
  inventory), **cargo-deny**, and **cargo-machete**.
- **ruff** and **pytest** for the Python adapter.

---

IcefallDB itself is dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE). Every third-party dependency is reproduced with its
license in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
