//! Warm GROUP BY composition end-to-end tests.
//!
//! `SELECT k, SUM(v), AVG(v), COUNT(v) FROM t GROUP BY k` where `k` is a
//! DECLARED `agg_group_key` must be answered from cached per-group partials,
//! byte-equal (sorted) to a full GROUP BY scan, on clean AND mutated
//! multi-fragment tables.  Undeclared keys and multi-column GROUP BY fall back
//! to a real scan.
//!
//! Each test writes a real table via `Writer::insert_batch`, registers a
//! `IcefallDBTableProvider`, and queries through the full DataFusion pipeline so
//! the genuine TWO-PHASE HASH plan (with `RepartitionExec(Hash[k])`) is
//! produced — exactly the shape the rule must peel.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use icefalldb_core::agg_cache::AggStateCache;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
use icefalldb_query::{
    execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    IcefallDBTableProvider, ProviderConfig,
};

/// Returns true when the optimized plan IS the folded fast path produced by the
/// `MetadataAggregate` rule (a `DataSourceExec` over a constant memory source).
fn plan_is_metadata_fast_path(plan: &Arc<dyn ExecutionPlan>) -> bool {
    use datafusion_datasource::source::DataSourceExec;
    plan.downcast_ref::<DataSourceExec>().is_some()
}

fn float_approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * a.abs().max(1.0)
}

/// `(k Int64, v Int64)` schema.
fn kv_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// Build a `(k, v)` batch from parallel slices.
fn kv_batch(keys: &[i64], vals: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        kv_schema(),
        vec![
            Arc::new(Int64Array::from(keys.to_vec())),
            Arc::new(Int64Array::from(vals.to_vec())),
        ],
    )
    .unwrap()
}

/// Collect (k → (sum, count)) ground truth for a set of (k, v) rows.
fn ground_truth(keys: &[i64], vals: &[i64]) -> BTreeMap<i64, (i64, i64)> {
    let mut m: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for (k, v) in keys.iter().zip(vals.iter()) {
        let e = m.entry(*k).or_insert((0, 0));
        e.0 += *v;
        e.1 += 1;
    }
    m
}

/// Read `SELECT k, SUM(v), COUNT(v), AVG(v) FROM t GROUP BY k` from a result
/// batch list into a sorted `(k → (sum, count, avg))` map.
fn read_group_results(batches: &[RecordBatch]) -> BTreeMap<i64, (i64, i64, f64)> {
    let mut out: BTreeMap<i64, (i64, i64, f64)> = BTreeMap::new();
    for batch in batches {
        let k = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("k column must be Int64");
        let sum = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("SUM(v) column must be Int64");
        let cnt = batch
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("COUNT(v) column must be Int64");
        let avg = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("AVG(v) column must be Float64");
        for row in 0..batch.num_rows() {
            out.insert(
                k.value(row),
                (sum.value(row), cnt.value(row), avg.value(row)),
            );
        }
    }
    out
}

async fn collect(plan: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
    let n = plan.output_partitioning().partition_count();
    let mut out = Vec::new();
    for p in 0..n {
        let stream = plan.execute(p, Arc::new(TaskContext::default())).unwrap();
        let mut batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        out.append(&mut batches);
    }
    out
}

/// (1) Single-phase grouped: force one scan partition (target_partitions=1) so
/// DataFusion builds a single-phase grouped aggregate directly on the scan. The
/// fast path must fire and the result must be byte-equal to ground truth.
#[tokio::test]
async fn single_phase_grouped_fast_path_fires() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // Keys in {1,2,3}, deterministic values.
    let keys: Vec<i64> = (0..30).map(|i| (i % 3) + 1).collect();
    let vals: Vec<i64> = (0..30).map(|i| i * 7 + 1).collect();

    let mut mdb_schema = arrow_schema_to_icefalldb(kv_schema());
    mdb_schema.agg_group_keys = Some(vec!["k".to_string()]);
    mdb_schema.row_group_target_rows = 1024; // single fragment

    let mut writer = Writer::create(Arc::clone(&storage), "t_sp", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(kv_batch(&keys, &vals)).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_sp",
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
    .unwrap();

    let config = icefalldb_session_config(1, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t_sp", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT k, SUM(v), COUNT(v), AVG(v) FROM t_sp GROUP BY k")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    assert!(
        plan_is_metadata_fast_path(&plan),
        "single-phase grouped: MetadataAggregate fast path must fire"
    );

    let batches = collect(plan).await;
    let got = read_group_results(&batches);
    let truth = ground_truth(&keys, &vals);

    assert_eq!(got.len(), truth.len(), "group count mismatch");
    for (k, (sum, cnt)) in &truth {
        let (g_sum, g_cnt, g_avg) = got.get(k).unwrap_or_else(|| panic!("missing group {k}"));
        assert_eq!(*g_sum, *sum, "SUM mismatch for group {k}");
        assert_eq!(*g_cnt, *cnt, "COUNT mismatch for group {k}");
        let expect_avg = *sum as f64 / *cnt as f64;
        assert!(
            float_approx_eq(*g_avg, expect_avg),
            "AVG mismatch for group {k}: got {g_avg}, expected {expect_avg}"
        );
    }
}

/// (2) Two-phase grouped on a GENUINE 3-fragment table (default
/// target_partitions): the query MUST fast-path (the rule must peel the
/// `RepartitionExec(Hash[k])`) and be byte-equal to ground truth.
#[tokio::test]
async fn two_phase_grouped_fast_path_fires() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3000 rows, 1000 per fragment → 3 fragments → genuine two-phase HASH plan.
    let keys: Vec<i64> = (0..3000).map(|i| (i % 5) + 1).collect();
    let vals: Vec<i64> = (0..3000).map(|i| (i % 97) as i64).collect();

    let mut mdb_schema = arrow_schema_to_icefalldb(kv_schema());
    mdb_schema.agg_group_keys = Some(vec!["k".to_string()]);
    mdb_schema.row_group_target_rows = 1000;

    let mut writer = Writer::create(Arc::clone(&storage), "t_tp", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(kv_batch(&keys, &vals)).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_tp",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
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
    .unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t_tp", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT k, SUM(v), COUNT(v), AVG(v) FROM t_tp GROUP BY k")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    // Sanity: the table must genuinely be a multi-fragment two-phase HASH plan.
    // With the rule DISABLED the planner emits Final→RepartitionExec(Hash[k])→
    // Partial.  This guards against accidentally testing a single-phase plan.
    {
        let mut cfg = icefalldb_session_config(4, 8192);
        cfg.options_mut()
            .extensions
            .get_mut::<icefalldb_query::IcefallDBConfig>()
            .unwrap()
            .metadata_aggregate = false;
        let raw_state = icefalldb_session_state_from_config(cfg);
        let raw_ctx = SessionContext::new_with_state(raw_state);
        let raw_provider = IcefallDBTableProvider::new(
            Arc::clone(&storage),
            "t_tp",
            ProviderConfig {
                batch_size: 1024,
                target_partitions: 4,
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
        .unwrap();
        raw_ctx
            .register_table("t_tp", Arc::new(raw_provider))
            .unwrap();
        let raw_plan = raw_ctx
            .sql("SELECT k, SUM(v), COUNT(v), AVG(v) FROM t_tp GROUP BY k")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let rendered = format!(
            "{}",
            datafusion::physical_plan::displayable(raw_plan.as_ref()).indent(true)
        );
        assert!(
            rendered.contains("RepartitionExec") && rendered.contains("mode=Partial"),
            "two-phase test must run on a genuine two-phase HASH plan with \
             RepartitionExec; got:\n{rendered}"
        );
    }

    assert!(
        plan_is_metadata_fast_path(&plan),
        "two-phase grouped: MetadataAggregate fast path MUST fire on the \
         RepartitionExec(Hash[k]) two-phase plan — a single-phase-only rule no-ops here"
    );

    let batches = collect(plan).await;
    let got = read_group_results(&batches);
    let truth = ground_truth(&keys, &vals);

    assert_eq!(got.len(), truth.len(), "group count mismatch");
    for (k, (sum, cnt)) in &truth {
        let (g_sum, g_cnt, g_avg) = got.get(k).unwrap_or_else(|| panic!("missing group {k}"));
        assert_eq!(*g_sum, *sum, "two-phase SUM mismatch for group {k}");
        assert_eq!(*g_cnt, *cnt, "two-phase COUNT mismatch for group {k}");
        let expect_avg = *sum as f64 / *cnt as f64;
        assert!(
            float_approx_eq(*g_avg, expect_avg),
            "two-phase AVG mismatch for group {k}: got {g_avg}, expected {expect_avg}"
        );
    }
}

/// (3) Under deletes: DELETE ~5% via the real mutations path, then re-run the
/// grouped query — still byte-equal to a DV-filtered full GROUP BY scan
/// (exercises `retract_grouped` through `scan_internal`).
#[tokio::test]
async fn grouped_under_deletes_byte_equal() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let keys: Vec<i64> = (0..3000).map(|i| (i % 5) + 1).collect();
    let vals: Vec<i64> = (0..3000).map(|i| (i % 89) as i64).collect();

    let mut mdb_schema = arrow_schema_to_icefalldb(kv_schema());
    mdb_schema.agg_group_keys = Some(vec!["k".to_string()]);
    mdb_schema.row_group_target_rows = 1000;

    let mut writer = Writer::create(Arc::clone(&storage), "t_del", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(kv_batch(&keys, &vals)).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);

    let make_provider = |storage: Arc<dyn Storage>| async move {
        IcefallDBTableProvider::new(
            storage,
            "t_del",
            ProviderConfig {
                batch_size: 1024,
                target_partitions: 4,
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
        .unwrap()
    };

    let provider = make_provider(Arc::clone(&storage)).await;
    ctx.register_table("t_del", Arc::new(provider)).unwrap();

    // Delete ~5%: rows where v % 20 == 0.
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "t_del",
        "DELETE FROM t_del WHERE v % 20 = 0",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();
    let provider = make_provider(Arc::clone(&storage)).await;
    ctx.deregister_table("t_del").unwrap();
    ctx.register_table("t_del", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT k, SUM(v), COUNT(v), AVG(v) FROM t_del GROUP BY k")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    assert!(
        plan_is_metadata_fast_path(&plan),
        "grouped under deletes: fast path must fire using live retracted grouped partials"
    );

    let batches = collect(plan).await;
    let got = read_group_results(&batches);

    // Ground truth over survivors (v % 20 != 0).
    let survivor_keys: Vec<i64> = keys
        .iter()
        .zip(vals.iter())
        .filter(|(_, v)| **v % 20 != 0)
        .map(|(k, _)| *k)
        .collect();
    let survivor_vals: Vec<i64> = vals.iter().copied().filter(|v| v % 20 != 0).collect();
    let truth = ground_truth(&survivor_keys, &survivor_vals);

    assert_eq!(got.len(), truth.len(), "group count mismatch after deletes");
    for (k, (sum, cnt)) in &truth {
        let (g_sum, g_cnt, g_avg) = got.get(k).unwrap_or_else(|| panic!("missing group {k}"));
        assert_eq!(*g_sum, *sum, "post-delete SUM mismatch for group {k}");
        assert_eq!(*g_cnt, *cnt, "post-delete COUNT mismatch for group {k}");
        let expect_avg = *sum as f64 / *cnt as f64;
        assert!(
            float_approx_eq(*g_avg, expect_avg),
            "post-delete AVG mismatch for group {k}: got {g_avg}, expected {expect_avg}"
        );
    }
}

/// (4a) Fallback: GROUP BY an UNDECLARED key must NOT fire the rule (the plan
/// stays a real aggregate) and the result is still correct.
#[tokio::test]
async fn grouped_undeclared_key_falls_back() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // Schema (k, v, other). Declare only `k`. GROUP BY `other` must fall back.
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("other", DataType::Int64, false),
    ]));
    let keys: Vec<i64> = (0..600).map(|i| (i % 3) + 1).collect();
    let vals: Vec<i64> = (0..600).map(|i| i * 2).collect();
    let other: Vec<i64> = (0..600).map(|i| (i % 4) + 10).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(keys.clone())),
            Arc::new(Int64Array::from(vals.clone())),
            Arc::new(Int64Array::from(other.clone())),
        ],
    )
    .unwrap();

    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.agg_group_keys = Some(vec!["k".to_string()]);
    mdb_schema.row_group_target_rows = 200; // 3 fragments

    let mut writer = Writer::create(Arc::clone(&storage), "t_un", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_un",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
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
    .unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t_un", Arc::new(provider)).unwrap();

    // GROUP BY undeclared `other` → must NOT fast-path.
    let df = ctx
        .sql("SELECT other, SUM(v) FROM t_un GROUP BY other")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(
        !plan_is_metadata_fast_path(&plan),
        "GROUP BY undeclared key must NOT fire the metadata fast path"
    );

    // Result still correct.
    let batches = collect(plan).await;
    let mut got: BTreeMap<i64, i64> = BTreeMap::new();
    for b in &batches {
        let oc = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let sc = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for r in 0..b.num_rows() {
            got.insert(oc.value(r), sc.value(r));
        }
    }
    let mut truth: BTreeMap<i64, i64> = BTreeMap::new();
    for (o, v) in other.iter().zip(vals.iter()) {
        *truth.entry(*o).or_insert(0) += *v;
    }
    assert_eq!(got, truth, "undeclared GROUP BY result must be correct");
}

/// NULL group key: a nullable key column with NULL values must produce the
/// SQL `GROUP BY` "null" group, byte-equal to a full scan.
#[tokio::test]
async fn grouped_null_key_group() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, true), // nullable key
        Field::new("v", DataType::Int64, false),
    ]));
    // Keys: 1, NULL, 2, NULL, 1, 2, … values deterministic.
    let key_opt: Vec<Option<i64>> = (0..40)
        .map(|i| if i % 3 == 1 { None } else { Some((i % 2) + 1) })
        .collect();
    let vals: Vec<i64> = (0..40).map(|i| i * 3 + 1).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(key_opt.clone())),
            Arc::new(Int64Array::from(vals.clone())),
        ],
    )
    .unwrap();

    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.agg_group_keys = Some(vec!["k".to_string()]);
    mdb_schema.row_group_target_rows = 1024;

    let mut writer = Writer::create(Arc::clone(&storage), "t_null", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_null",
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
    .unwrap();

    let config = icefalldb_session_config(1, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t_null", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT k, SUM(v), COUNT(v) FROM t_null GROUP BY k")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan),
        "NULL-key group: fast path must fire"
    );

    let batches = collect(plan).await;
    // Build got: Option<i64> key → (sum, count).
    let mut got: BTreeMap<Option<i64>, (i64, i64)> = BTreeMap::new();
    for b in &batches {
        let kc = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let sc = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let cc = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        for r in 0..b.num_rows() {
            let key = if kc.is_null(r) {
                None
            } else {
                Some(kc.value(r))
            };
            got.insert(key, (sc.value(r), cc.value(r)));
        }
    }
    // Ground truth.
    let mut truth: BTreeMap<Option<i64>, (i64, i64)> = BTreeMap::new();
    for (k, v) in key_opt.iter().zip(vals.iter()) {
        let e = truth.entry(*k).or_insert((0, 0));
        e.0 += *v;
        e.1 += 1;
    }
    assert_eq!(got, truth, "NULL-key GROUP BY result must match full scan");
}

/// (4b) Fallback: a multi-column GROUP BY (`k, other`) must NOT fire the rule.
#[tokio::test]
async fn grouped_multi_key_falls_back() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
        Field::new("other", DataType::Int64, false),
    ]));
    let keys: Vec<i64> = (0..600).map(|i| (i % 3) + 1).collect();
    let vals: Vec<i64> = (0..600).map(|i| i * 2).collect();
    let other: Vec<i64> = (0..600).map(|i| (i % 4) + 10).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(keys.clone())),
            Arc::new(Int64Array::from(vals.clone())),
            Arc::new(Int64Array::from(other.clone())),
        ],
    )
    .unwrap();

    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.agg_group_keys = Some(vec!["k".to_string()]);
    mdb_schema.row_group_target_rows = 200;

    let mut writer = Writer::create(Arc::clone(&storage), "t_mk", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_mk",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
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
    .unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table("t_mk", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT k, other, SUM(v) FROM t_mk GROUP BY k, other")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(
        !plan_is_metadata_fast_path(&plan),
        "multi-column GROUP BY must NOT fire the metadata fast path"
    );
}
