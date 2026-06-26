//! Warm-aggregate benchmark: composed from .agg partials vs full Parquet scan.
//!
//! Measures `SELECT SUM(v), AVG(v), STDDEV(v) FROM t` in two scenarios:
//!
//! 1. Single-fragment table (1 row group → 1 scan partition).  The
//!    `MetadataAggregate` physical optimizer rule fires on a Single-mode
//!    `AggregateExec` and answers the query entirely from in-memory `.agg`
//!    sidecar data — zero Parquet I/O.
//!
//! 2. Multi-fragment table (N/100k row groups, one per 100 k rows).
//!    DataFusion generates a two-phase `Final → CoalescePartitions → Partial`
//!    aggregate.  The `MetadataAggregate` rule matches this pattern, composes
//!    the result from the Partial's sidecar data, and replaces the entire
//!    subtree — zero Parquet I/O even for multi-fragment tables.
//!
//! For each scenario the benchmark runs 10 iterations with
//! `metadata_aggregate=true` (fast path) and `metadata_aggregate=false` (full
//! scan baseline) and reports min/median/max.
//!
//! Usage:
//!   cargo run --release -p icefalldb-query --example bench_warm_agg
//!
//! Override row count:
//!   BENCH_ROWS=10000000 cargo run --release -p icefalldb-query --example bench_warm_agg
//!
//! Print physical plans for verification:
//!   EXPLAIN_PLAN=1 cargo run --release -p icefalldb-query --example bench_warm_agg

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::displayable;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
use icefalldb_query::{
    icefalldb_session_config, icefalldb_session_state_from_config, IcefallDBTableProvider,
    ProviderConfig,
};

fn make_provider_config() -> ProviderConfig {
    ProviderConfig {
        batch_size: 8192,
        target_partitions: 1,
        io_coalesce_window: 0,
        io_concurrency: 1,
        // usize::MAX means the native DataFusion Parquet reader is only
        // preferred when >= usize::MAX columns are referenced — effectively
        // always using the custom IcefallDBScanExec, which the
        // MetadataAggregate rule requires.
        native_parquet_threshold: usize::MAX,
        parquet_metadata_cache_capacity: 256,
        tiny_table_cache_threshold_rows: 0,
        tiny_table_cache_threshold_bytes: 0,
        wal_mode: true,
    }
}

/// Write `n_rows` into a table at `storage["bench"]`.
/// `row_group_size` controls the target row-group size; use `n_rows` for a
/// single-fragment table or `100_000` for a ten-fragment table at 1 M rows.
async fn write_table(
    storage: &Arc<dyn Storage>,
    n_rows: usize,
    row_group_size: usize,
    arrow_schema: &Arc<ArrowSchema>,
) {
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(arrow_schema));
    mdb_schema.row_group_target_rows = row_group_size;

    let mut writer = Writer::create(Arc::clone(storage), "bench", mdb_schema)
        .await
        .unwrap();

    let chunk = row_group_size;
    let mut written = 0usize;
    while written < n_rows {
        let this_chunk = chunk.min(n_rows - written);
        let values: Vec<f64> = (0..this_chunk).map(|i| (written + i) as f64).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(arrow_schema),
            vec![Arc::new(Float64Array::from(values))],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        written += this_chunk;
    }
    writer.commit().await.unwrap();
}

/// Run the benchmark query 10 times and return sorted durations.
async fn run_query_times(
    storage: Arc<dyn Storage>,
    metadata_aggregate_enabled: bool,
    iters: usize,
) -> Vec<Duration> {
    let config = if metadata_aggregate_enabled {
        icefalldb_session_config(1, 8192)
    } else {
        icefalldb_session_config(1, 8192).set_str("icefalldb.metadata_aggregate", "false")
    };
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);

    let provider =
        IcefallDBTableProvider::new(Arc::clone(&storage), "bench", make_provider_config())
            .await
            .unwrap();
    ctx.register_table("bench", Arc::new(provider)).unwrap();

    // Print the physical plan once if EXPLAIN_PLAN is set, to verify the fast path fires.
    if std::env::var("EXPLAIN_PLAN").is_ok() {
        let df = ctx
            .sql("SELECT SUM(v), AVG(v), STDDEV(v) FROM bench")
            .await
            .unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let label = if metadata_aggregate_enabled {
            "fast path (metadata_aggregate=true)"
        } else {
            "slow path (metadata_aggregate=false)"
        };
        println!("=== PLAN [{label}] ===");
        println!("{}", displayable(plan.as_ref()).indent(true));
    }

    // Warm-up pass (not counted).
    ctx.sql("SELECT SUM(v), AVG(v), STDDEV(v) FROM bench")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let mut times: Vec<Duration> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let batches = ctx
            .sql("SELECT SUM(v), AVG(v), STDDEV(v) FROM bench")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        times.push(t0.elapsed());
        assert_eq!(batches[0].num_rows(), 1, "result must be a single row");
    }

    times.sort();
    times
}

fn print_times(label: &str, times: &[Duration]) {
    let min = times[0];
    let median = times[times.len() / 2];
    let max = times[times.len() - 1];
    println!(
        "    {label}: min={:.3}ms  median={:.3}ms  max={:.3}ms",
        min.as_secs_f64() * 1000.0,
        median.as_secs_f64() * 1000.0,
        max.as_secs_f64() * 1000.0,
    );
}

#[tokio::main]
async fn main() {
    let n_rows: usize = std::env::var("BENCH_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Float64,
        false,
    )]));

    let iters = 10;

    // =========================================================================
    // Scenario A: single-fragment table — MetadataAggregate rule CAN fire.
    // =========================================================================
    println!("--- Scenario A: single fragment ({n_rows} rows, 1 row group) ---");
    {
        let tmp_dir = std::env::temp_dir().join("icefalldb_bench_warm_agg_a");
        std::fs::remove_dir_all(&tmp_dir).ok();
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(&tmp_dir).unwrap());

        println!("  Writing {n_rows} rows as one row group ...");
        write_table(&storage, n_rows, n_rows, &arrow_schema).await;

        println!("  Running {iters} iterations (metadata_aggregate=true, fast path) ...");
        let fast_times = run_query_times(Arc::clone(&storage), true, iters).await;

        println!("  Running {iters} iterations (metadata_aggregate=false, full scan) ...");
        let slow_times = run_query_times(Arc::clone(&storage), false, iters).await;

        let speedup = slow_times[slow_times.len() / 2].as_secs_f64()
            / fast_times[fast_times.len() / 2].as_secs_f64();

        println!();
        print_times("fast (rule fires)", &fast_times);
        print_times("slow (full scan) ", &slow_times);
        println!("  speedup (slow/fast median): {speedup:.1}x");
        println!("  NOTE: fast path uses CooperativeExec-peeling (Task 6 fix) so the");
        println!("  MetadataAggregate rule fires even through the DataFusion 54 wrapper.");

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    println!();

    // =========================================================================
    // Scenario B: multi-fragment table — DataFusion generates a two-phase
    // Final → CoalescePartitions → Partial aggregate over a multi-partition
    // scan.  The MetadataAggregate rule matches this pattern and replaces the
    // entire subtree with a composed constant row — zero Parquet I/O.
    // =========================================================================
    let n_fragments = n_rows.div_ceil(100_000);
    println!("--- Scenario B: {n_fragments} fragments ({n_rows} rows, 100k rows/fragment) ---");
    {
        let tmp_dir = std::env::temp_dir().join("icefalldb_bench_warm_agg_b");
        std::fs::remove_dir_all(&tmp_dir).ok();
        std::fs::create_dir_all(&tmp_dir).unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(&tmp_dir).unwrap());

        println!("  Writing {n_rows} rows as {n_fragments} fragments of 100k rows ...");
        write_table(&storage, n_rows, 100_000, &arrow_schema).await;

        println!("  Running {iters} iterations (metadata_aggregate=true, fast path) ...");
        let fast_times = run_query_times(Arc::clone(&storage), true, iters).await;

        println!("  Running {iters} iterations (metadata_aggregate=false, full scan) ...");
        let slow_times = run_query_times(Arc::clone(&storage), false, iters).await;

        let speedup = slow_times[slow_times.len() / 2].as_secs_f64()
            / fast_times[fast_times.len() / 2].as_secs_f64();

        println!();
        print_times("fast (rule fires)", &fast_times);
        print_times("slow (full scan) ", &slow_times);
        println!("  speedup (slow/fast median): {speedup:.1}x");
        println!("  NOTE: the MetadataAggregate rule now fires on the two-phase");
        println!("  Final→CoalescePartitions→Partial plan generated for multi-partition");
        println!("  scans, replacing the entire subtree with a composed constant row.");

        std::fs::remove_dir_all(&tmp_dir).ok();
    }

    println!();
    println!("=== bench_warm_agg summary ===");
    println!("rows: {n_rows}");
    println!();
    println!("Scenario A (1 fragment): the MetadataAggregate rule peels through any");
    println!("  CooperativeExec wrappers injected by DataFusion 54's EnsureCooperative");
    println!("  optimizer rule, so the fast path fires in the full SQL pipeline.");
    println!("  Expected speedup: large (microseconds vs milliseconds).");
    println!();
    println!("Scenario B ({n_fragments} fragments): the MetadataAggregate rule matches the");
    println!("  two-phase Final→CoalescePartitions→Partial plan, composes the result");
    println!("  from per-fragment .agg sidecars, and replaces the whole subtree.");
    println!("  Expected speedup: large (sub-ms vs full Parquet scan).");
    println!();
    println!("Use EXPLAIN_PLAN=1 to inspect plans for details.");
    println!();
    println!("--- DuckDB baseline ---");
    println!("python/benchmarks/RESULTS.md records DuckDB numbers for warm_agg_group");
    println!("(GROUP BY + AVG, 1 M rows) but NOT for a plain SUM/AVG/STDDEV without");
    println!("grouping. To obtain a fair comparison, run:");
    println!("  SELECT SUM(v), AVG(v), STDDEV(v) FROM read_parquet('<path>/*.parquet')");
    println!("in DuckDB at the same row count and record the warm median.");
    println!("Comparable DuckDB number (warm, NVMe): not yet measured for this query.");
}
