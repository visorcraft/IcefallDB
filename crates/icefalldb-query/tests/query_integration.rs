//! Integration tests for the DataFusion `IcefallDBTableProvider` and the
//! `MetadataAggregate` optimizer rule.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::TableProvider;
use icefalldb_core::arrow_schema_to_icefalldb;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::Writer;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn make_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let a = Int32Array::from(vec![Some(1), Some(2), Some(3), None, Some(5), Some(6)]);
    let b = StringArray::from(vec![
        Some("a"),
        Some("b"),
        Some("c"),
        Some("d"),
        None,
        Some("f"),
    ]);
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
}

async fn create_test_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let mut icefalldb_schema = arrow_schema_to_icefalldb(make_batch().schema());
    // Force multiple row groups so that pruning drops whole files.
    icefalldb_schema.row_group_target_rows = 2;
    let mut writer = Writer::create(Arc::clone(&storage), "t", icefalldb_schema)
        .await
        .unwrap();
    writer.insert_batch(make_batch()).await.unwrap();
    writer.commit().await.unwrap();
    storage
}

fn remove_meta_files(dir: &std::path::Path) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            remove_meta_files(&path);
        } else if path.extension().and_then(|e| e.to_str()) == Some("meta") {
            std::fs::remove_file(&path).unwrap();
        }
    }
}

#[tokio::test]
async fn test_count_star_and_count_column() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_table(tmp.path()).await;

    let provider = IcefallDBTableProvider::new(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let count_star: i64 = ctx
        .sql("SELECT COUNT(*) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count_star, 6);

    let count_a: i64 = ctx
        .sql("SELECT COUNT(a) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count_a, 5);
}

#[tokio::test]
async fn test_min_max() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_table(tmp.path()).await;

    let provider = IcefallDBTableProvider::new(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batches = ctx
        .sql("SELECT MIN(a), MAX(a) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let min = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap()
        .value(0);
    let max = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap()
        .value(0);
    assert_eq!(min, 1);
    assert_eq!(max, 6);
}

#[tokio::test]
async fn test_predicate_prunes_files() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_table(tmp.path()).await;

    let provider = IcefallDBTableProvider::new(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            // Disable the tiny-table cache so this test inspects the custom scan.
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    let filter = datafusion::logical_expr::col("a").gt(datafusion::logical_expr::lit(4i32));
    let support = provider
        .supports_filters_pushdown(&[&filter])
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(
        support,
        datafusion::logical_expr::TableProviderFilterPushDown::Exact
    );

    let plan = provider
        .scan(&ctx.state(), Some(&vec![0]), &[filter], None)
        .await
        .unwrap();
    let scan = plan
        .downcast_ref::<icefalldb_query::scan::IcefallDBScanExec>()
        .expect("scan should return IcefallDBScanExec");
    assert_eq!(
        scan.planned_row_groups().len(),
        1,
        "predicate should prune to one row group"
    );
}

#[tokio::test]
async fn test_fallback_when_meta_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_test_table(tmp.path()).await;

    // Remove all `.meta` sidecar files to force fallback reads.
    remove_meta_files(&tmp.path().join("t"));

    let provider = IcefallDBTableProvider::new(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batches = ctx
        .sql("SELECT COUNT(*), MIN(a), MAX(a) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    let min = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap()
        .value(0);
    let max = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::Int32Array>()
        .unwrap()
        .value(0);

    assert_eq!(count, 6);
    assert_eq!(min, 1);
    assert_eq!(max, 6);
}

fn events_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat_id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    // 20 events across 5 categories; category 10 does not exist in categories.
    let id = Int64Array::from((1..=20).collect::<Vec<_>>());
    let cat_id = Int64Array::from(vec![
        1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 10,
    ]);
    let name = StringArray::from(
        (1..=20)
            .map(|i| Some(format!("event-{i}")))
            .collect::<Vec<_>>(),
    );
    RecordBatch::try_new(schema, vec![Arc::new(id), Arc::new(cat_id), Arc::new(name)]).unwrap()
}

fn categories_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("category_name", DataType::Utf8, true),
    ]));
    let id = Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let category_name = StringArray::from(vec![
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ]);
    RecordBatch::try_new(schema, vec![Arc::new(id), Arc::new(category_name)]).unwrap()
}

async fn create_join_tables(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());

    let mut events_schema = arrow_schema_to_icefalldb(events_batch().schema());
    events_schema.row_group_target_rows = 5;
    let mut writer = Writer::create(Arc::clone(&storage), "events", events_schema)
        .await
        .unwrap();
    writer.insert_batch(events_batch()).await.unwrap();
    writer.commit().await.unwrap();

    let mut categories_schema = arrow_schema_to_icefalldb(categories_batch().schema());
    categories_schema.row_group_target_rows = 10;
    let mut writer = Writer::create(Arc::clone(&storage), "categories", categories_schema)
        .await
        .unwrap();
    writer.insert_batch(categories_batch()).await.unwrap();
    writer.commit().await.unwrap();

    storage
}

/// Count `FilterExec` nodes that sit directly above a `IcefallDBScanExec` in the
/// optimized physical plan. The tiny-build-side join rule pushes dynamic
/// predicates this way so DataFusion handles literal-to-column type coercion.
fn count_pushed_filter_execs(plan: &dyn datafusion::physical_plan::ExecutionPlan) -> usize {
    use datafusion::physical_plan::filter::FilterExec;

    let mut count = 0;
    if let Some(filter) = plan.downcast_ref::<FilterExec>() {
        if filter
            .input()
            .as_ref()
            .downcast_ref::<icefalldb_query::scan::IcefallDBScanExec>()
            .is_some()
        {
            count += 1;
        }
    }
    for child in plan.children() {
        count += count_pushed_filter_execs(child.as_ref());
    }
    count
}

#[tokio::test]
async fn test_tiny_build_join_pushdown() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_join_tables(tmp.path()).await;

    // Disable the tiny-table cache so the probe-side scan remains a
    // `IcefallDBScanExec` that the dynamic-filter pushdown rule can target.
    let config = ProviderConfig {
        batch_size: 1024,
        target_partitions: 1,
        io_coalesce_window: 0,
        io_concurrency: 1,
        native_parquet_threshold: usize::MAX,
        parquet_metadata_cache_capacity: 256,
        tiny_table_cache_threshold_rows: 0,
        tiny_table_cache_threshold_bytes: 0,
        wal_mode: true,
    };

    let events_provider = IcefallDBTableProvider::new(Arc::clone(&storage), "events", config)
        .await
        .unwrap();

    let categories_provider = IcefallDBTableProvider::new(storage, "categories", config)
        .await
        .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("events", Arc::new(events_provider))
        .unwrap();
    ctx.register_table("categories", Arc::new(categories_provider))
        .unwrap();

    let sql = "SELECT e.id, e.cat_id, c.category_name \
               FROM events e JOIN categories c ON e.cat_id = c.id \
               ORDER BY e.id";
    let df = ctx.sql(sql).await.unwrap();

    // The optimized physical plan should contain a pushed filter.
    let physical_plan = df.clone().create_physical_plan().await.unwrap();
    let pushed_filter_count = count_pushed_filter_execs(physical_plan.as_ref());
    assert!(
        pushed_filter_count > 0,
        "optimized plan should push at least one dynamic filter into the probe-side scan"
    );

    let batches = df.collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 20, "all events should match a category");

    let ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let expected: Vec<i64> = (1..=20).collect();
    let actual: Vec<i64> = ids.iter().map(|v| v.unwrap()).collect();
    assert_eq!(actual, expected);
}

fn empty_table_schema() -> arrow::datatypes::Schema {
    arrow::datatypes::Schema::new(vec![
        Field::new("id", DataType::Int64, true),
        Field::new("val", DataType::Float64, true),
    ])
}

async fn create_empty_test_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let icefalldb_schema = arrow_schema_to_icefalldb(Arc::new(empty_table_schema()));
    let _writer = Writer::create(Arc::clone(&storage), "t", icefalldb_schema)
        .await
        .unwrap();
    // No rows inserted; commit would write an empty manifest, so leave the
    // table in its just-created state (the provider sees zero row groups).
    storage
}

#[tokio::test]
async fn test_aggregates_over_empty_table() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_empty_test_table(tmp.path()).await;

    let provider = IcefallDBTableProvider::new(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            // Keep the native reader enabled (the default) so the empty-table
            // guard in `scan()` is exercised; without it the plan would fail
            // with `UnknownPartitioning(0)`.
            native_parquet_threshold: 1,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batches = ctx
        .sql("SELECT COUNT(*) AS c, COUNT(val) AS cv, SUM(val) AS s, AVG(val) AS a, STDDEV(val) AS sd, VAR_POP(val) AS v, MIN(val) AS mn, MAX(val) AS mx FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let c = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(c, 0);

    let cv = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(cv, 0);

    let s = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert!(s.is_null(0));

    let a = batches[0]
        .column(3)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert!(a.is_null(0));

    let sd = batches[0]
        .column(4)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert!(sd.is_null(0));

    let v = batches[0]
        .column(5)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert!(v.is_null(0));

    let mn = batches[0]
        .column(6)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert!(mn.is_null(0));

    let mx = batches[0]
        .column(7)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap();
    assert!(mx.is_null(0));
}
