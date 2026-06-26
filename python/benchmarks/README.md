# Value-Comparison Benchmarks

These benchmarks compare IcefallDB against the systems it actually competes with in
practice: raw Parquet, DuckDB-on-Parquet, SQLite, and CSV.

## Running

From the repo root, with the Python venv active:

```bash
python/python/.venv/bin/python python/benchmarks/compare.py
```

The script will build `target/release/icefalldb` if it is missing.

To regenerate the comparison chart:

```bash
python/python/.venv/bin/python python/benchmarks/chart.py
```

This writes `python/benchmarks/comparison_matrix.png`.

## What is measured

- **Write throughput**: rows/second to create a fresh dataset.
- **Read-all latency**: time to fetch every row.
- **Aggregate latency**: time to run `COUNT(*), AVG(value)`.
- **Storage efficiency**: bytes per row on disk.

All backends use the same schema:

- `id` int64
- `value` float64
- `name` utf8

and the same deterministic synthetic data at 10k, 100k, and 1M rows.

## Competitors

- **IcefallDB**: `icefalldb create` + `icefalldb insert` for writes; `icefalldb.attach(...)` +
  DuckDB SQL for reads.
- **Parquet**: PyArrow writer/reader on a single Parquet file.
- **DuckDB-on-Parquet**: DuckDB `COPY ... TO PARQUET` and
  `SELECT * FROM read_parquet(...)`.
- **SQLite**: Python `sqlite3` bulk insert and `SELECT`.
- **CSV**: Python `csv` module.

## Caveats

- These are micro-benchmarks on a single machine; absolute numbers will vary.
- IcefallDB write times include commit, metadata, and fsync overhead (the real
  durability path), not just Parquet encoding.
- IcefallDB read times include DuckDB replacement-scan overhead because that is the
  user-facing query API.
