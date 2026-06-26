//! E2E test: the warm-aggregate fast path must survive compaction.
//!
//! 1. Build a multi-fragment table with declared agg_group_keys.
//! 2. Assert SELECT SUM(v) and SELECT k, SUM(v) GROUP BY k fire the fast path.
//! 3. Compact.
//! 4. Assert the SAME queries STILL fire the fast path and are byte-equal to
//!    the pre-compaction results.
//!
//! This test FAILS without the fix: post-compaction fragments have agg: None,
//! so the rule falls back to a full scan.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::{collect, ExecutionPlan};
use datafusion_datasource::source::DataSourceExec;
use icefalldb_core::agg_cache::AggStateCache;
use icefalldb_core::compaction::Compactor;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_query::{
    icefalldb_session_config, icefalldb_session_state_from_config, IcefallDBTableProvider,
    ProviderConfig,
};

fn plan_is_fast_path(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<DataSourceExec>().is_some()
}

fn kv_schema_mdb(row_group_target_rows: usize) -> Schema {
    Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "k".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 1,
            },
            Column {
                name: "v".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 2,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: Some(vec!["k".to_string()]),
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 2,
        dropped_columns: vec![],
    }
}

fn kv_batch(keys: &[i64], vals: &[i64]) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys.to_vec())),
            Arc::new(Int64Array::from(vals.to_vec())),
        ],
    )
    .unwrap()
}

async fn make_provider(storage: Arc<dyn Storage>, table: &str) -> Arc<IcefallDBTableProvider> {
    Arc::new(
        IcefallDBTableProvider::new(
            storage,
            table,
            ProviderConfig {
                batch_size: 1024,
                target_partitions: 1,
                io_coalesce_window: 0,
                io_concurrency: 1,
                native_parquet_threshold: usize::MAX,
                parquet_metadata_cache_capacity: 256,
                tiny_table_cache_threshold_rows: 0,
                tiny_table_cache_threshold_bytes: 0,
                wal_mode: true,
            },
        )
        .await
        .unwrap(),
    )
}

fn sum_from_batches(batches: &[RecordBatch]) -> i64 {
    batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .flatten()
        })
        .sum()
}

/// Fast-path survives compaction for `SELECT SUM(v)` (no GROUP BY).
#[tokio::test]
async fn fast_path_sum_survives_compaction() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3 fragments × 10 rows → compaction produces 1 merged fragment.
    let mut writer = Writer::create(Arc::clone(&storage), "e2e_sum", kv_schema_mdb(10))
        .await
        .unwrap();
    for i in 0..3i64 {
        let keys: Vec<i64> = (0..10).map(|j| j % 3).collect();
        let vals: Vec<i64> = (0..10).map(|j| i * 100 + j).collect();
        writer.insert_batch(kv_batch(&keys, &vals)).await.unwrap();
        writer.commit().await.unwrap();
    }

    // ── Pre-compaction: assert fast path fires ────────────────────────────────
    AggStateCache::global().clear();
    let provider = make_provider(Arc::clone(&storage), "e2e_sum").await;
    let config = icefalldb_session_config(1, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("e2e_sum", Arc::clone(&provider) as _)
        .unwrap();

    let df_pre = ctx.sql("SELECT SUM(v) FROM e2e_sum").await.unwrap();
    let plan_pre = df_pre.create_physical_plan().await.unwrap();
    assert!(
        plan_is_fast_path(&plan_pre),
        "pre-compaction: SUM(v) fast path must fire"
    );
    let pre_batches = collect(
        plan_pre,
        Arc::new(datafusion::execution::TaskContext::default()),
    )
    .await
    .unwrap();
    let pre_sum = sum_from_batches(&pre_batches);

    // ── Compact ───────────────────────────────────────────────────────────────
    Compactor::new(storage.as_ref(), "e2e_sum")
        .compact()
        .await
        .unwrap();

    // ── Post-compaction: assert fast path STILL fires ─────────────────────────
    AggStateCache::global().clear();
    let provider2 = make_provider(Arc::clone(&storage), "e2e_sum").await;
    let config2 = icefalldb_session_config(1, 8192);
    let state2 = icefalldb_session_state_from_config(config2);
    let ctx2 = SessionContext::new_with_state(state2);
    ctx2.register_table("e2e_sum", Arc::clone(&provider2) as _)
        .unwrap();

    let df_post = ctx2.sql("SELECT SUM(v) FROM e2e_sum").await.unwrap();
    let plan_post = df_post.create_physical_plan().await.unwrap();
    assert!(
        plan_is_fast_path(&plan_post),
        "post-compaction: SUM(v) fast path must STILL fire (would fail without .agg fix)"
    );
    let post_batches = collect(
        plan_post,
        Arc::new(datafusion::execution::TaskContext::default()),
    )
    .await
    .unwrap();
    let post_sum = sum_from_batches(&post_batches);

    assert_eq!(
        pre_sum, post_sum,
        "SUM(v) must be byte-equal before and after compaction"
    );
}

/// Collect `(k → SUM(v))` from batches with schema `(k Int64, SUM(v) Int64)`.
fn group_sums(batches: &[RecordBatch]) -> std::collections::BTreeMap<i64, i64> {
    use std::collections::BTreeMap;
    let mut m: BTreeMap<i64, i64> = BTreeMap::new();
    for batch in batches {
        let k = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("k column must be Int64");
        let s = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("SUM(v) column must be Int64");
        for row in 0..batch.num_rows() {
            *m.entry(k.value(row)).or_insert(0) += s.value(row);
        }
    }
    m
}

/// Fast-path survives compaction for `SELECT k, SUM(v) GROUP BY k`.
///
/// Uses a single commit of 3000 rows with `row_group_target_rows = 1000` →
/// 3 fragments in one commit (the proven pattern from `two_phase_grouped_fast_path_fires`).
/// Results are compared as `BTreeMap<k, sum>` so ordering doesn't matter.
#[tokio::test]
async fn fast_path_group_by_survives_compaction() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3000 rows in one commit, 1000 per row-group → 3 fragments.
    let keys: Vec<i64> = (0..3000).map(|i| (i % 3) + 1).collect();
    let vals: Vec<i64> = (0..3000).map(|i| i * 7 + 1).collect();

    let mut writer = Writer::create(Arc::clone(&storage), "e2e_grp", kv_schema_mdb(1000))
        .await
        .unwrap();
    writer.insert_batch(kv_batch(&keys, &vals)).await.unwrap();
    writer.commit().await.unwrap();

    // ── Pre-compaction ────────────────────────────────────────────────────────
    AggStateCache::global().clear();
    let provider = make_provider(Arc::clone(&storage), "e2e_grp").await;
    let config = icefalldb_session_config(1, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("e2e_grp", Arc::clone(&provider) as _)
        .unwrap();

    let df_pre = ctx
        .sql("SELECT k, SUM(v) FROM e2e_grp GROUP BY k")
        .await
        .unwrap();
    let plan_pre = df_pre.create_physical_plan().await.unwrap();
    assert!(
        plan_is_fast_path(&plan_pre),
        "pre-compaction: GROUP BY fast path must fire"
    );
    let pre_batches = collect(
        plan_pre,
        Arc::new(datafusion::execution::TaskContext::default()),
    )
    .await
    .unwrap();
    let pre_sums = group_sums(&pre_batches);

    // ── Compact ───────────────────────────────────────────────────────────────
    Compactor::new(storage.as_ref(), "e2e_grp")
        .compact()
        .await
        .unwrap();

    // ── Post-compaction ───────────────────────────────────────────────────────
    AggStateCache::global().clear();
    let provider2 = make_provider(Arc::clone(&storage), "e2e_grp").await;
    let config2 = icefalldb_session_config(1, 8192);
    let state2 = icefalldb_session_state_from_config(config2);
    let ctx2 = SessionContext::new_with_state(state2);
    ctx2.register_table("e2e_grp", Arc::clone(&provider2) as _)
        .unwrap();

    let df_post = ctx2
        .sql("SELECT k, SUM(v) FROM e2e_grp GROUP BY k")
        .await
        .unwrap();
    let plan_post = df_post.create_physical_plan().await.unwrap();
    assert!(
        plan_is_fast_path(&plan_post),
        "post-compaction: GROUP BY fast path must STILL fire (fails without .agg fix)"
    );
    let post_batches = collect(
        plan_post,
        Arc::new(datafusion::execution::TaskContext::default()),
    )
    .await
    .unwrap();
    let post_sums = group_sums(&post_batches);

    // Results must be equal (order-independent).
    assert_eq!(
        pre_sums, post_sums,
        "GROUP BY k SUM(v) results must be equal before and after compaction"
    );
}
