//! Integration test that runs the six DataFusion benchmark matrix queries using
//! the custom `IcefallDBScanExec` path.
//!
//! `ProviderConfig::native_parquet_threshold` is set to `usize::MAX` so the
//! provider never selects the native DataFusion Parquet reader. The dataset is a
//! small synthetic equivalent of the benchmark DB, including a multi-row-group
//! `events_wide` table that exercises the whole-file fallback for files whose
//! sidecar offsets only describe the first row group.

use std::sync::Arc;

use arrow::array::{
    Float32Array, Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use datafusion::datasource::TableProvider;
use icefalldb_core::arrow_schema_to_icefalldb;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::Writer;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn categories_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("category_id", DataType::Int64, true),
        Field::new("category_name", DataType::Utf8, true),
    ]));
    let id = Int64Array::from(vec![0, 1, 2]);
    let name = StringArray::from(vec!["cat_0", "cat_1", "cat_2"]);
    RecordBatch::try_new(schema, vec![Arc::new(id), Arc::new(name)]).unwrap()
}

fn events_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("category", DataType::Utf8, true),
        Field::new("value", DataType::Float64, true),
    ]));

    // 12 rows spanning February, March and April 2024.
    // 2024-02-15 00:00:00 UTC in microseconds since epoch.
    let base_us = 1707955200000000i64;
    let day_us = 24 * 60 * 60 * 1_000_000i64;
    let ids = Int64Array::from((0..12).collect::<Vec<_>>());
    let ts = TimestampMicrosecondArray::from(
        (0..12)
            .map(|i| Some(base_us + i as i64 * 10 * day_us))
            .collect::<Vec<_>>(),
    );
    let categories = StringArray::from(vec![
        "cat_0", "cat_1", "cat_2", "cat_0", "cat_1", "cat_2", "cat_0", "cat_1", "cat_2", "cat_0",
        "cat_1", "cat_2",
    ]);
    let values = Float64Array::from(vec![
        0.1, 0.6, 0.2, 0.7, 0.3, 0.8, 0.4, 0.9, 0.5, 0.2, 0.7, 0.1,
    ]);
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(ts),
            Arc::new(categories),
            Arc::new(values),
        ],
    )
    .unwrap()
}

fn events_wide_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("int_a", DataType::Int64, true),
        Field::new("int_b", DataType::Int64, true),
        Field::new("int_c", DataType::Int64, true),
        Field::new("int_d", DataType::Int64, true),
        Field::new("float_a", DataType::Float32, true),
        Field::new("float_b", DataType::Float32, true),
        Field::new("float_c", DataType::Float32, true),
        Field::new("float_d", DataType::Float32, true),
        Field::new("status", DataType::Utf8, true),
    ]));

    let n: usize = 20;
    let ids = Int64Array::from((0..n).map(|i| i as i64).collect::<Vec<_>>());
    // 2024-01-01 00:00:00 UTC in microseconds since epoch.
    let base_us = 1704067200000000i64;
    let ts = TimestampMicrosecondArray::from(
        (0..n)
            .map(|i| Some(base_us + i as i64 * 3600 * 1_000_000))
            .collect::<Vec<_>>(),
    );
    let int_a = Int64Array::from((0..n).map(|i| i as i64 * 10).collect::<Vec<_>>());
    let int_b = Int64Array::from((0..n).map(|i| i as i64 * 100).collect::<Vec<_>>());
    let int_c = Int64Array::from((0..n).map(|i| (n - i) as i64 * 10).collect::<Vec<_>>());
    let int_d = Int64Array::from((0..n).map(|i| i as i64 * 5).collect::<Vec<_>>());
    let float_a = Float32Array::from((0..n).map(|i| i as f32 * 0.1).collect::<Vec<_>>());
    let float_b = Float32Array::from((0..n).map(|i| i as f32 * 0.05).collect::<Vec<_>>());
    let float_c = Float32Array::from((0..n).map(|i| i as f32 * 0.01).collect::<Vec<_>>());
    let float_d = Float32Array::from((0..n).map(|i| i as f32 * 0.02).collect::<Vec<_>>());
    let status = StringArray::from(
        (0..n)
            .map(|i| if i % 3 == 0 { "ok" } else { "fail" })
            .collect::<Vec<_>>(),
    );

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(ids),
            Arc::new(ts),
            Arc::new(int_a),
            Arc::new(int_b),
            Arc::new(int_c),
            Arc::new(int_d),
            Arc::new(float_a),
            Arc::new(float_b),
            Arc::new(float_c),
            Arc::new(float_d),
            Arc::new(status),
        ],
    )
    .unwrap()
}

fn make_provider_config() -> ProviderConfig {
    ProviderConfig {
        batch_size: 1024,
        target_partitions: 1,
        io_coalesce_window: 0,
        io_concurrency: 1,
        native_parquet_threshold: usize::MAX,
        parquet_metadata_cache_capacity: 256,
        // Disable the tiny-table cache so these tests target the custom scan path.
        tiny_table_cache_threshold_rows: 0,
        tiny_table_cache_threshold_bytes: 0,
        wal_mode: true,
    }
}

async fn create_test_db(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());

    // categories: tiny single-row-group table.
    let mut cat_schema = arrow_schema_to_icefalldb(categories_batch().schema());
    cat_schema.row_group_target_rows = 10;
    let mut writer = Writer::create(Arc::clone(&storage), "categories", cat_schema)
        .await
        .unwrap();
    writer.insert_batch(categories_batch()).await.unwrap();
    writer.commit().await.unwrap();

    // events: single-row-group table.
    let mut events_schema = arrow_schema_to_icefalldb(events_batch().schema());
    events_schema.row_group_target_rows = 100;
    let mut writer = Writer::create(Arc::clone(&storage), "events", events_schema)
        .await
        .unwrap();
    writer.insert_batch(events_batch()).await.unwrap();
    writer.commit().await.unwrap();

    // events_wide: force four row groups so the multi-row-group whole-file
    // fallback path is exercised.
    let mut wide_schema = arrow_schema_to_icefalldb(events_wide_batch().schema());
    wide_schema.row_group_target_rows = 5;
    let mut writer = Writer::create(Arc::clone(&storage), "events_wide", wide_schema)
        .await
        .unwrap();
    writer.insert_batch(events_wide_batch()).await.unwrap();
    writer.commit().await.unwrap();

    storage
}

fn int64_value(batch: &RecordBatch, col: usize, row: usize) -> i64 {
    batch
        .column(col)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(row)
}

fn float64_value(batch: &RecordBatch, col: usize, row: usize) -> f64 {
    batch
        .column(col)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(row)
}

fn utf8_value(batch: &RecordBatch, col: usize, row: usize) -> String {
    batch
        .column(col)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .value(row)
        .to_string()
}

#[tokio::test]
async fn test_scan_matrix_runs_all_six_queries() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_db(tmp.path()).await;

    let events_provider =
        IcefallDBTableProvider::new(Arc::clone(&storage), "events", make_provider_config())
            .await
            .unwrap();
    let categories_provider =
        IcefallDBTableProvider::new(Arc::clone(&storage), "categories", make_provider_config())
            .await
            .unwrap();
    let wide_provider =
        IcefallDBTableProvider::new(Arc::clone(&storage), "events_wide", make_provider_config())
            .await
            .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("events", Arc::new(events_provider))
        .unwrap();
    ctx.register_table("categories", Arc::new(categories_provider))
        .unwrap();
    ctx.register_table("events_wide", Arc::new(wide_provider))
        .unwrap();

    // 1. warm_agg_group: March 1..April 1 covers ids 2,3,4 (cat_2 value 0.2,
    //    cat_0 value 0.7, cat_1 value 0.3). Ordered by event_count DESC all tie.
    let batches = ctx
        .sql(
            "SELECT category, COUNT(*) AS event_count, AVG(value) AS avg_value
             FROM events
             WHERE ts >= TIMESTAMP '2024-03-01 00:00:00'
               AND ts <  TIMESTAMP '2024-04-01 00:00:00'
             GROUP BY category
             ORDER BY event_count DESC",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    let mut got: Vec<(String, i64, f64)> = (0..3)
        .map(|i| {
            (
                utf8_value(&batches[0], 0, i),
                int64_value(&batches[0], 1, i),
                float64_value(&batches[0], 2, i),
            )
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![
            ("cat_0".into(), 1, 0.7),
            ("cat_1".into(), 1, 0.3),
            ("cat_2".into(), 1, 0.2),
        ]
    );

    // 2. warm_filtered_scan: category='cat_0' and value>0.5 -> only id 3 (0.7).
    let batches = ctx
        .sql(
            "SELECT COUNT(*) AS cnt, SUM(value) AS total
             FROM events
             WHERE category = 'cat_0' AND value > 0.5",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_value(&batches[0], 0, 0), 1);
    assert!((float64_value(&batches[0], 1, 0) - 0.7).abs() < 1e-12);

    // 3. warm_full_count.
    let batches = ctx
        .sql("SELECT COUNT(*) AS cnt FROM events")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(int64_value(&batches[0], 0, 0), 12);

    // 4. join_100m_x_10: each event joins to category_name; counts are 4,4,4.
    let batches = ctx
        .sql(
            "SELECT c.category_name, COUNT(*) AS event_count
             FROM events e
             JOIN categories c ON e.category = c.category_name
             GROUP BY c.category_name
             ORDER BY event_count DESC",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);
    let mut got: Vec<(String, i64)> = (0..3)
        .map(|i| {
            (
                utf8_value(&batches[0], 0, i),
                int64_value(&batches[0], 1, i),
            )
        })
        .collect();
    got.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(
        got,
        vec![
            ("cat_0".into(), 4),
            ("cat_1".into(), 4),
            ("cat_2".into(), 4)
        ]
    );

    // 5. wide_agg: int_d > 10 AND float_d < 0.5 AND status = 'ok'.
    //    int_d = id*5, so int_d > 10 => id > 2.
    //    float_d = id*0.02 < 0.5 => id < 25 (always true for id < 20).
    //    status='ok' => id%3==0.
    //    Combined: id in {3,6,9,12,15,18}.
    let batches = ctx
        .sql(
            "SELECT SUM(int_a) AS s_a, AVG(float_a) AS avg_a,
                    SUM(int_b) AS s_b, AVG(float_b) AS avg_b,
                    AVG(float_c) AS avg_c, SUM(int_c) AS s_c
             FROM events_wide
             WHERE int_d > 10 AND float_d < 0.5 AND status = 'ok'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    // int_a values: 30,60,90,120,150,180 -> sum 630
    assert_eq!(int64_value(&batches[0], 0, 0), 630);
    // float_a values: 0.3,0.6,0.9,1.2,1.5,1.8 -> avg 1.05
    assert!((float64_value(&batches[0], 1, 0) - 1.05).abs() < 1e-6);
    // int_b values: 300,600,900,1200,1500,1800 -> sum 6300
    assert_eq!(int64_value(&batches[0], 2, 0), 6300);
    // float_b values: 0.15,0.30,0.45,0.60,0.75,0.90 -> avg 0.525
    assert!((float64_value(&batches[0], 3, 0) - 0.525).abs() < 1e-6);
    // float_c values: 0.03,0.06,0.09,0.12,0.15,0.18 -> avg 0.105
    assert!((float64_value(&batches[0], 4, 0) - 0.105).abs() < 1e-6);
    // int_c values: 170,140,110,80,50,20 -> sum 570
    assert_eq!(int64_value(&batches[0], 5, 0), 570);

    // 6. wide_filter: same predicate as wide_agg -> 6 rows.
    let batches = ctx
        .sql(
            "SELECT COUNT(*) AS cnt
             FROM events_wide
             WHERE int_d > 10 AND float_d < 0.5 AND status = 'ok'",
        )
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(int64_value(&batches[0], 0, 0), 6);
}

/// Provider config that allows the native DataFusion Parquet reader (threshold
/// 1), unlike [`make_provider_config`] which forces the custom scan.
fn make_native_provider_config() -> ProviderConfig {
    ProviderConfig {
        native_parquet_threshold: 1,
        ..make_provider_config()
    }
}

/// Regression: a native bulk-decode scan (projection ⊆ filter columns) must not
/// drop matching rows when a `LIMIT` is present. The bulk path disables the
/// Parquet row filter, so a limit pushed into the scan would cap *pre-filter*
/// rows; the gate must keep such queries on the pushdown path. Also checks the
/// no-limit bulk path (projection-remap of a filter column) is correct.
#[tokio::test]
async fn test_native_bulk_decode_limit_and_filter() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_db(tmp.path()).await;

    let wide_provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "events_wide",
        make_native_provider_config(),
    )
    .await
    .unwrap();
    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("events_wide", Arc::new(wide_provider))
        .unwrap();

    // int_a = id*10, status='ok' iff id%3==0. `int_a > 150` => id in 16..=19;
    // intersect status='ok' => id 18 only (int_a 180). The first rows scanned
    // (ids 0..) all fail `int_a > 150`, so a pre-filter LIMIT would return 0.
    let collect_int_a = |batches: &[arrow::array::RecordBatch]| -> Vec<i64> {
        let mut out = Vec::new();
        for b in batches {
            for r in 0..b.num_rows() {
                out.push(int64_value(b, 0, r));
            }
        }
        out
    };

    // With LIMIT: must still return the single matching row, not 0.
    let batches = ctx
        .sql("SELECT int_a FROM events_wide WHERE int_a > 150 AND status = 'ok' LIMIT 5")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(collect_int_a(&batches), vec![180]);

    // Without LIMIT the bulk-decode gate fires (projection {int_a} ⊆ filters
    // {int_a, status}); the projection-remap output must match exactly.
    let batches = ctx
        .sql("SELECT int_a FROM events_wide WHERE int_a > 150 AND status = 'ok'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(collect_int_a(&batches), vec![180]);

    // COUNT(*) (empty projection ⊆ filters) over the same predicate -> 1.
    let batches = ctx
        .sql("SELECT COUNT(*) AS cnt FROM events_wide WHERE int_a > 150 AND status = 'ok'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(int64_value(&batches[0], 0, 0), 1);
}

#[tokio::test]
async fn test_scan_matrix_uses_icefalldb_scan_exec() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_db(tmp.path()).await;

    let provider =
        IcefallDBTableProvider::new(Arc::clone(&storage), "events", make_provider_config())
            .await
            .unwrap();

    let ctx = icefalldb_session(1, 1024);
    let filter =
        datafusion::logical_expr::col("category").eq(datafusion::logical_expr::lit("cat_0"));
    let plan = provider
        .scan(&ctx.state(), None, &[filter], None)
        .await
        .unwrap();
    assert!(
        plan.downcast_ref::<icefalldb_query::scan::IcefallDBScanExec>()
            .is_some(),
        "native_parquet_threshold=MAX should force IcefallDBScanExec"
    );
}
