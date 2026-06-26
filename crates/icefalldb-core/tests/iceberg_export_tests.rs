use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::iceberg;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::writer::Writer;
use std::sync::Arc;

fn make_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

fn make_int64_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn make_uint64_batch(ids: Vec<u64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::UInt64, false)]);
    let array = arrow::array::UInt64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

#[tokio::test]
async fn test_iceberg_export_produces_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let mut writer = Writer::new(Arc::new(storage), "products", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_int64_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let output = tempfile::tempdir().unwrap();
    let table_root = tmp.path().join("products");
    let table_root_uri = format!("file://{}", table_root.to_string_lossy().replace('\\', "/"));

    let metadata_path = iceberg::export_table(
        &LocalStorage::new(tmp.path()).unwrap(),
        "products",
        output.path(),
        None,
        &table_root_uri,
    )
    .await
    .unwrap();

    assert!(metadata_path.exists());
    let metadata: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(&metadata_path).await.unwrap()).unwrap();
    assert_eq!(metadata["format-version"], 2);
    assert!(metadata["current-snapshot-id"].as_i64().unwrap() > 0);
    assert!(metadata["last-partition-id"].is_number());

    let metadata_dir = metadata_path.parent().unwrap();
    assert!(metadata_dir.join("version-hint.text").exists());
}

#[tokio::test]
async fn test_iceberg_export_specific_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let mut writer = Writer::new(Arc::new(storage), "products", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_int64_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let output = tempfile::tempdir().unwrap();
    let table_root = tmp.path().join("products");
    let table_root_uri = format!("file://{}", table_root.to_string_lossy().replace('\\', "/"));

    let metadata_path = iceberg::export_table(
        &LocalStorage::new(tmp.path()).unwrap(),
        "products",
        output.path(),
        Some(1),
        &table_root_uri,
    )
    .await
    .unwrap();

    let metadata: serde_json::Value =
        serde_json::from_slice(&tokio::fs::read(&metadata_path).await.unwrap()).unwrap();
    assert_eq!(metadata["format-version"], 2);
    let snapshots = metadata["snapshots"].as_array().unwrap();
    assert_eq!(snapshots.len(), 1);
    assert_eq!(snapshots[0]["sequence-number"], 1);
}

#[tokio::test]
async fn test_iceberg_export_rejects_unsigned_integers() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "uint64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };

    let mut writer = Writer::new(Arc::new(storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_uint64_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let output = tempfile::tempdir().unwrap();
    let table_root = tmp.path().join("products");
    let table_root_uri = format!("file://{}", table_root.to_string_lossy().replace('\\', "/"));

    let err = iceberg::export_table(
        &LocalStorage::new(tmp.path()).unwrap(),
        "products",
        output.path(),
        None,
        &table_root_uri,
    )
    .await
    .unwrap_err();

    let msg = format!("{}", err);
    assert!(
        msg.contains("unsupported") || msg.contains("not supported"),
        "expected unsupported type error, got: {}",
        msg
    );
}
