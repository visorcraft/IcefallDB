//! Prove cross-query per-fragment partial reuse via AggStateCache.
//!
//! The core insight: because each IcefallDB fragment is immutable and
//! content-hash-identified, a DIFFERENT overlapping query can reuse the
//! per-fragment `FragmentAggState` entries that were loaded by an earlier query.
//! This is cross-query reuse that a whole-query result cache (DuckDB-style) cannot
//! provide — an overlapping but distinct query is always a full miss there.
//!
//! The test:
//!   1. Build a 3-fragment table (each fragment has a `.agg` sidecar).
//!   2. Reset the global `AggStateCache` and clear/reset stats.
//!   3. Run query A (`SELECT SUM(v) FROM t`) — misses populate the cache.
//!   4. Record hit count.
//!   5. Run query B (`SELECT AVG(v), STDDEV(v) FROM t`) — DIFFERENT aggregates,
//!      same fragments → should reuse A's per-fragment partials (cache hits).
//!   6. Assert hits_after >= hits_before + n_fragments (every fragment was reused).
//!   7. Assert B's results equal a full-scan ground truth (correctness).

use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::collect;
use datafusion_datasource::source::DataSourceExec;
use icefalldb_core::agg_cache::AggStateCache;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_query::{
    icefalldb_session_config, icefalldb_session_state_from_config, IcefallDBTableProvider,
    ProviderConfig,
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn v_schema_mdb(row_group_target_rows: usize) -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "v".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn v_batch(vals: &[i64]) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals.to_vec()))]).unwrap()
}

async fn make_provider(storage: Arc<dyn Storage>, table: &str) -> Arc<IcefallDBTableProvider> {
    Arc::new(
        IcefallDBTableProvider::new(
            storage,
            table,
            ProviderConfig {
                batch_size: 4096,
                target_partitions: 1,
                io_coalesce_window: 0,
                io_concurrency: 1,
                native_parquet_threshold: usize::MAX,
                parquet_metadata_cache_capacity: 256,
                // disable the tiny-table cache so the planner always goes through
                // the warm-agg rule (tiny table cache bypasses it)
                tiny_table_cache_threshold_rows: 0,
                tiny_table_cache_threshold_bytes: 0,
                wal_mode: true,
            },
        )
        .await
        .unwrap(),
    )
}

fn plan_is_fast_path(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> bool {
    plan.downcast_ref::<DataSourceExec>().is_some()
}

fn float_close(a: f64, b: f64) -> bool {
    let tol = 1e-9 * b.abs().max(1.0);
    (a - b).abs() <= tol
}

/// Extract a single f64 scalar from a one-row RecordBatch column by index.
fn scalar_f64(batches: &[RecordBatch], col_idx: usize) -> f64 {
    batches[0]
        .column(col_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0)
}

/// Extract a single i64 scalar from a one-row RecordBatch column by index.
fn scalar_i64(batches: &[RecordBatch], col_idx: usize) -> i64 {
    batches[0]
        .column(col_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

fn task_ctx() -> Arc<datafusion::execution::TaskContext> {
    Arc::new(datafusion::execution::TaskContext::default())
}

// ── the test ──────────────────────────────────────────────────────────────────

/// Verify that a DIFFERENT aggregate query (B) over the same multi-fragment
/// table reuses the per-fragment `FragmentAggState` entries that were loaded by
/// query A — measurable as AggStateCache hits on query B.
///
/// This is the cross-query reuse DuckDB's whole-query result cache cannot
/// provide: query B is entirely distinct from A (different aggregates), yet it
/// shares the same immutable per-fragment partials.
#[tokio::test]
async fn distinct_query_reuses_per_fragment_partials() {
    // ── set up storage ────────────────────────────────────────────────────────
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3 fragments × 10 rows each; values v = 1..30 distributed across fragments.
    // row_group_target_rows=10 forces each commit into its own fragment.
    const N_FRAGMENTS: usize = 3;
    let mut writer = Writer::create(
        Arc::clone(&storage),
        "xq",
        v_schema_mdb(10), // 10 rows per fragment
    )
    .await
    .unwrap();

    for frag_idx in 0..N_FRAGMENTS {
        let base = (frag_idx * 10 + 1) as i64;
        let vals: Vec<i64> = (base..base + 10).collect(); // [1..10], [11..20], [21..30]
        writer.insert_batch(v_batch(&vals)).await.unwrap();
        writer.commit().await.unwrap();
    }

    // ── global AggStateCache: clear entries + reset hit/miss counters ─────────
    let global_cache = AggStateCache::global();
    global_cache.clear();
    global_cache.reset_stats();

    // ── build the DataFusion session + register the table ─────────────────────
    let provider = make_provider(Arc::clone(&storage), "xq").await;
    let config = icefalldb_session_config(1, 4096);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("xq", Arc::clone(&provider) as _)
        .unwrap();

    // ── Query A: SELECT SUM(v) FROM xq ───────────────────────────────────────
    // This should fire the metadata fast path (warm-aggregate rule) and load
    // each fragment's FragmentAggState into the AggStateCache (cache misses).
    let plan_a = ctx
        .sql("SELECT SUM(v) FROM xq")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    assert!(
        plan_is_fast_path(&plan_a),
        "query A (SELECT SUM(v)) must fire the metadata fast path"
    );

    let batches_a = collect(plan_a, task_ctx()).await.unwrap();
    let sum_a = scalar_i64(&batches_a, 0);

    // Ground truth: sum(1..=30) = 465
    assert_eq!(sum_a, 465, "SUM(v) must equal ground truth 465");

    // After query A the cache should have N_FRAGMENTS entries (one per fragment).
    let (hits_after_a, misses_after_a) = global_cache.stats();
    assert_eq!(
        misses_after_a, N_FRAGMENTS as u64,
        "query A must produce exactly N_FRAGMENTS cache misses (one per .agg load)"
    );
    // There may be additional hits from the dirty-fragment composite-key lookups
    // that also go through agg_cache.get; we only require that misses == N_FRAGMENTS.
    let hits_before_b = hits_after_a;

    // ── Query B: SELECT AVG(v), STDDEV_POP(v) FROM xq — a DIFFERENT query ────
    // This must REUSE A's per-fragment partials (cache hits), not re-read .agg.
    // We need a fresh provider + session so the ScanPlan is rebuilt (scan_internal
    // is called again), going through the AggStateCache::get path.
    let provider_b = make_provider(Arc::clone(&storage), "xq").await;
    let config_b = icefalldb_session_config(1, 4096);
    let state_b = icefalldb_session_state_from_config(config_b);
    let ctx_b = SessionContext::new_with_state(state_b);
    ctx_b
        .register_table("xq", Arc::clone(&provider_b) as _)
        .unwrap();

    let plan_b = ctx_b
        .sql("SELECT AVG(v), STDDEV_POP(v) FROM xq")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();

    assert!(
        plan_is_fast_path(&plan_b),
        "query B (SELECT AVG(v), STDDEV_POP(v)) must fire the metadata fast path"
    );

    let batches_b = collect(plan_b, task_ctx()).await.unwrap();
    let avg_b = scalar_f64(&batches_b, 0);
    let stddev_b = scalar_f64(&batches_b, 1);

    // ── assert cross-query reuse ──────────────────────────────────────────────
    let (hits_after_b, _) = global_cache.stats();

    // Every shared fragment must have been served from cache (hit), not re-read.
    // Without the cache (or if it were cleared between A and B), query B would
    // produce N_FRAGMENTS misses here instead; this assertion is load-bearing.
    assert!(
        hits_after_b >= hits_before_b + N_FRAGMENTS as u64,
        "query B must have generated at least {N_FRAGMENTS} AggStateCache hits \
         (cross-query per-fragment reuse). \
         hits before B = {hits_before_b}, hits after B = {hits_after_b}"
    );

    // ── assert correctness of B's result ─────────────────────────────────────
    // v = 1..=30: mean = 15.5, var_pop = ((30²+29²+...+1²)/30) - 15.5²
    // sum = 465, sumsq = sum(i^2 for i in 1..=30) = 30*31*61/6 = 9455
    // var_pop = 9455/30 - 15.5² = 315.1667 - 240.25 = 74.9167
    // stddev_pop = sqrt(74.9167) ≈ 8.6554
    let n = 30.0_f64;
    let expected_avg = 465.0 / n; // 15.5
    let expected_sumsq = 9455.0_f64; // Σi² for i=1..30
    let expected_var_pop = expected_sumsq / n - expected_avg * expected_avg;
    let expected_stddev_pop = expected_var_pop.sqrt();

    assert!(
        float_close(avg_b, expected_avg),
        "AVG(v) from cache-reuse path = {avg_b}, expected ≈ {expected_avg}"
    );
    assert!(
        float_close(stddev_b, expected_stddev_pop),
        "STDDEV_POP(v) from cache-reuse path = {stddev_b}, expected ≈ {expected_stddev_pop}"
    );
}
