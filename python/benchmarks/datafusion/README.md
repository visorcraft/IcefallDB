# IcefallDB DataFusion benchmark harness

This directory contains the Python-side benchmark harness for the native
DataFusion query engine in IcefallDB.  It mirrors the Seafowl benchmark matrix but
runs against IcefallDB's own query layer via the existing Python adapter.

## Files

- `generate_events.py` — generate the `events`, `events_wide`, and `categories`
  benchmark tables sorted by hot keys and dictionary-encoded for IcefallDB.
- `oracle.py` — DuckDB oracle helper and result-comparison utility.
- `run_icefalldb_query_bench.py` — warm-cache 9-query matrix runner.
- `bench_cold_cache.py` — cold-cache variant; evicts Parquet files from the OS
  page cache with `posix_fadvise(POSIX_FADV_DONTNEED)` before each timed query.
- `generate_baseline_doc.py` — renders benchmark JSON into a Markdown baseline
  under `python/benchmarks/datafusion/baselines/` by default.
- `README.md` — this file.

## Query matrix

The runner implements six raw benchmark queries from the DataFusion query-engine
spec plus three optimized variants that exercise sorted, indexed, and clustered
tables:

| Query name | Description | Tables |
|---|---|---|
| `warm_agg_group` | Aggregation + filter + group by on `events` | `events` |
| `warm_filtered_scan` | Filtered scan COUNT/SUM on `events` | `events` |
| `warm_full_count` | `SELECT COUNT(*) FROM events` | `events` |
| `join_100m_x_10` | Join `events` with `categories` | `events`, `categories` |
| `wide_agg` | Aggregation over 6 columns on `events_wide` | `events_wide` |
| `wide_filter` | Filter scan on 5 columns on `events_wide` | `events_wide` |
| `sorted_time_window` | Time-window aggregation on sorted `events_sorted` | `events_sorted` |
| `indexed_equality` | Equality-filtered scan on indexed `events_indexed` | `events_indexed` |
| `clustered_wide_filter` | Wide-table filter on clustered `events_wide_clustered` | `events_wide_clustered` |

## Usage

Generate the benchmark datasets with the IcefallDB-native generator, then run the
matrix:

```bash
cd python
source .venv/bin/activate
cd benchmarks/datafusion

# Build the IcefallDB CLI (required for table creation / insert)
cargo build --release -p icefalldb-cli

# Generate datasets (if not already present)
python3 generate_events.py --scale 1m   # events=1M, events_wide=10M
python3 generate_events.py --scale 10m  # events=10M, events_wide=100M

# Run the DataFusion harness
python3 run_icefalldb_query_bench.py --scale 1m --engine duckdb
python3 run_icefalldb_query_bench.py --scale 10m --engine datafusion --iters 5

# Cold-cache variant
python3 bench_cold_cache.py --scale 10m --engine datafusion --iters 5
```

## Engines

- `--engine duckdb` — runs queries through DuckDB using `icefalldb.attach(...)`.
  This is the differential oracle.
- `--engine datafusion` — runs queries through the native DataFusion layer via
  `icefalldb.attach(..., engine='datafusion')`.

## Output

Results are written to JSON files in `target/tmp/`:

- `icefalldb_query_bench_{scale}_{engine}.json`
- `icefalldb_query_bench_cold_{scale}_{engine}.json`

The JSON schema matches the Rust `BenchmarkMatrix` type:

```json
{
  "dataset": "icefalldb_query_bench",
  "scale": "10m",
  "queries": [
    {
      "query_name": "warm_full_count",
      "engine": "duckdb",
      "iterations": [2.1, 2.0, 2.2, 2.1, 2.0],
      "median_ms": 2.1,
      "min_ms": 2.0,
      "max_ms": 2.2
    }
  ]
}
```

A Markdown results table is also printed to stdout.
