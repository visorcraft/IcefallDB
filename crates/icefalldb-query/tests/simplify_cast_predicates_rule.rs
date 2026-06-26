//! Integration test for the `SimplifyCastPredicates` logical optimizer rule.

use std::sync::Arc;

use arrow::array::{Float32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::arrow_schema_to_icefalldb;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::Writer;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn make_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "f32",
        DataType::Float32,
        true,
    )]));
    let f32 = Float32Array::from(vec![
        Some(0.1),
        Some(0.6),
        Some(1.2),
        None,
        Some(0.4),
        Some(0.9),
    ]);
    RecordBatch::try_new(schema, vec![Arc::new(f32)]).unwrap()
}

async fn create_test_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let mut icefalldb_schema = arrow_schema_to_icefalldb(make_batch().schema());
    icefalldb_schema.row_group_target_rows = 2;
    let mut writer = Writer::create(Arc::clone(&storage), "t", icefalldb_schema)
        .await
        .unwrap();
    writer.insert_batch(make_batch()).await.unwrap();
    writer.commit().await.unwrap();
    storage
}

#[tokio::test]
async fn test_cast_simplification_rule_removes_redundant_cast() {
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
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    // DataFusion type coercion turns the Float32 vs Float64 literal comparison
    // into CAST(f32 AS Float64) > Float64(0.5). The rule should rewrite it back.
    let df = ctx.sql("SELECT * FROM t WHERE f32 > 0.5").await.unwrap();

    // Inspect the optimized logical plan and confirm the cast was removed.
    let optimized = ctx.state().optimize(df.logical_plan()).unwrap();
    let plan_str = format!("{optimized:?}");
    assert!(
        !plan_str.to_ascii_lowercase().contains("cast"),
        "optimized logical plan should not contain a cast after simplification, got:\n{plan_str}"
    );

    // Verify correctness: rows with f32 > 0.5 are 0.6, 1.2, 0.9.
    let batches = df.collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}
