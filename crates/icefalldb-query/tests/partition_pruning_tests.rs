//! Integration tests for partition-column predicate pushdown and pruning.

use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::datasource::TableProvider;
use datafusion::logical_expr::{col, lit, Expr, TableProviderFilterPushDown};
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
    let a = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
    let b = StringArray::from(vec!["east", "west", "east", "west", "east", "west"]);
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
}

async fn create_partitioned_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let mut icefalldb_schema = arrow_schema_to_icefalldb(make_batch().schema());
    icefalldb_schema.row_group_target_rows = 2;
    icefalldb_schema.partition_by = Some(vec!["b".into()]);
    let mut writer = Writer::create(Arc::clone(&storage), "t", icefalldb_schema)
        .await
        .unwrap();
    writer.insert_batch(make_batch()).await.unwrap();
    writer.commit().await.unwrap();
    storage
}

fn provider_config() -> ProviderConfig {
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

#[tokio::test]
async fn test_equality_partition_predicate_is_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_partitioned_table(tmp.path()).await;
    let provider = IcefallDBTableProvider::new(storage, "t", provider_config())
        .await
        .unwrap();

    let filter = col("b").eq(lit("east"));
    let support = provider
        .supports_filters_pushdown(&[&filter])
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(support, TableProviderFilterPushDown::Exact);
}

#[tokio::test]
async fn test_in_list_partition_predicate_is_exact() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_partitioned_table(tmp.path()).await;
    let provider = IcefallDBTableProvider::new(storage, "t", provider_config())
        .await
        .unwrap();

    let filter = Expr::InList(datafusion::logical_expr::expr::InList {
        expr: Box::new(col("b")),
        list: vec![lit("east")],
        negated: false,
    });
    let support = provider
        .supports_filters_pushdown(&[&filter])
        .unwrap()
        .pop()
        .unwrap();
    assert_eq!(support, TableProviderFilterPushDown::Exact);
}

#[tokio::test]
async fn test_partition_equality_prunes_row_groups() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_partitioned_table(tmp.path()).await;
    let provider = IcefallDBTableProvider::new(storage, "t", provider_config())
        .await
        .unwrap();

    let ctx = icefalldb_session(1, 1024);
    let filter = col("b").eq(lit("east"));
    let plan = provider
        .scan(&ctx.state(), Some(&vec![0]), &[filter], None)
        .await
        .unwrap();
    let scan = plan
        .downcast_ref::<icefalldb_query::scan::IcefallDBScanExec>()
        .expect("scan should return IcefallDBScanExec");
    assert_eq!(
        scan.planned_row_groups().len(),
        2,
        "equality on partition column should prune to the 'east' row groups"
    );
}

#[tokio::test]
async fn test_in_list_partition_predicate_prunes_row_groups() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_partitioned_table(tmp.path()).await;
    let provider = IcefallDBTableProvider::new(storage, "t", provider_config())
        .await
        .unwrap();

    let ctx = icefalldb_session(1, 1024);
    let filter = Expr::InList(datafusion::logical_expr::expr::InList {
        expr: Box::new(col("b")),
        list: vec![lit("west")],
        negated: false,
    });
    let plan = provider
        .scan(&ctx.state(), Some(&vec![0]), &[filter], None)
        .await
        .unwrap();
    let scan = plan
        .downcast_ref::<icefalldb_query::scan::IcefallDBScanExec>()
        .expect("scan should return IcefallDBScanExec");
    assert_eq!(
        scan.planned_row_groups().len(),
        2,
        "IN-list on partition column should prune to the 'west' row groups"
    );
}

#[tokio::test]
async fn test_partition_predicate_query_returns_correct_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_partitioned_table(tmp.path()).await;
    let provider = IcefallDBTableProvider::new(storage, "t", provider_config())
        .await
        .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batches = ctx
        .sql("SELECT a FROM t WHERE b = 'east' ORDER BY a")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let values: Vec<i32> = batches
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .flatten()
        })
        .collect();
    assert_eq!(values, vec![1, 3, 5]);
}
