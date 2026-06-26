use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use futures::StreamExt;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::Reader;
use std::path::PathBuf;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("fixtures")
}

async fn read_all(table: &str) -> Vec<RecordBatch> {
    let storage = LocalStorage::new(fixtures_dir()).unwrap();
    let reader = Reader::new(&storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut batches = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            batches.push(batch.unwrap());
        }
    }
    batches
}

fn int_ids(batches: &[RecordBatch]) -> Vec<i64> {
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            ids.push(col.value(i));
        }
    }
    ids
}

#[tokio::test]
async fn test_golden_simple_int() {
    let batches = read_all("simple_int").await;
    assert_eq!(int_ids(&batches), vec![1, 2, 3]);
}

#[tokio::test]
async fn test_golden_multi_commit() {
    let batches = read_all("multi_commit").await;
    assert_eq!(int_ids(&batches), vec![1, 2, 3, 4, 5, 6]);
}

#[tokio::test]
async fn test_golden_multi_row_group() {
    let batches = read_all("multi_row_group").await;
    assert_eq!(int_ids(&batches), vec![1, 2, 3, 4, 5]);
    assert!(
        batches.len() > 1,
        "multi_row_group should span multiple batches/row groups"
    );
}

#[tokio::test]
async fn test_golden_mixed_types() {
    let batches = read_all("mixed_types").await;
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 3);

    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let values = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let labels = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let active = batch
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();

    assert_eq!(ids.value(0), 1);
    assert_eq!(ids.value(1), 2);
    assert_eq!(ids.value(2), 3);

    assert_eq!(values.value(0), 1.5);
    assert!(values.is_null(1));
    assert_eq!(values.value(2), 3.0);

    assert_eq!(labels.value(0), "a");
    assert_eq!(labels.value(1), "b");
    assert!(labels.is_null(2));

    assert!(active.value(0));
    assert!(!active.value(1));
    assert!(active.value(2));
}
