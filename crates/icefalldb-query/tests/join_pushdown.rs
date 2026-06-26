//! Integration tests for tiny-build-side join dynamic filter pushdown.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::arrow_schema_to_icefalldb;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::Writer;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn events_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat_id", DataType::Int64, true),
        Field::new("name", DataType::Utf8, true),
    ]));
    // 20 events across categories 1-5 and 10; all have matches in categories.
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

    // Disable the tiny-table cache so the probe-side `events` scan remains a
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
