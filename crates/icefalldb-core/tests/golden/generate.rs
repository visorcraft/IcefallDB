use arrow::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::Writer;
use std::path::PathBuf;
use std::sync::Arc;

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join("fixtures")
}

fn make_int_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".to_string(),
            r#type: "int64".to_string(),
            nullable: false,
            field_id: 0,
        }],
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_mixed_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".to_string(),
                r#type: "int64".to_string(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "value".to_string(),
                r#type: "float64".to_string(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "label".to_string(),
                r#type: "utf8".to_string(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "active".to_string(),
                r#type: "bool".to_string(),
                nullable: false,
                field_id: 0,
            },
        ],
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn int_batch(values: Vec<i64>) -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(values)) as ArrayRef],
    )
    .unwrap()
}

fn mixed_batch(
    ids: Vec<i64>,
    values: Vec<Option<f64>>,
    labels: Vec<Option<&'static str>>,
    active: Vec<bool>,
) -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
        Field::new("label", DataType::Utf8, true),
        Field::new("active", DataType::Boolean, false),
    ]));
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int64Array::from(ids)) as ArrayRef,
            Arc::new(Float64Array::from(values)) as ArrayRef,
            Arc::new(StringArray::from(labels)) as ArrayRef,
            Arc::new(BooleanArray::from(active)) as ArrayRef,
        ],
    )
    .unwrap()
}

async fn clean_output(table: &str) {
    let path = output_dir().join(table);
    if path.exists() {
        tokio::fs::remove_dir_all(&path).await.unwrap();
    }
}

async fn remove_lock_file(table: &str) {
    let lock_path = output_dir().join(table).join("_write.lock");
    if lock_path.exists() {
        tokio::fs::remove_file(&lock_path).await.unwrap();
    }
}

async fn generate_simple_int() {
    clean_output("simple_int").await;
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(output_dir()).unwrap());
    let mut writer = Writer::new(storage, "simple_int", make_int_schema())
        .await
        .unwrap();
    writer.insert_batch(int_batch(vec![1, 2, 3])).await.unwrap();
    writer.commit().await.unwrap();
    remove_lock_file("simple_int").await;
}

async fn generate_multi_commit() {
    clean_output("multi_commit").await;
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(output_dir()).unwrap());
    let values = vec![vec![1, 2], vec![3, 4, 5], vec![6]];
    for chunk in values {
        let mut writer = Writer::new(Arc::clone(&storage), "multi_commit", make_int_schema())
            .await
            .unwrap();
        writer.insert_batch(int_batch(chunk)).await.unwrap();
        writer.commit().await.unwrap();
    }
    remove_lock_file("multi_commit").await;
}

async fn generate_multi_row_group() {
    clean_output("multi_row_group").await;
    let mut schema = make_int_schema();
    schema.row_group_target_rows = 2;
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(output_dir()).unwrap());
    let mut writer = Writer::new(Arc::clone(&storage), "multi_row_group", schema)
        .await
        .unwrap();
    writer
        .insert_batch(int_batch(vec![1, 2, 3, 4, 5]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    remove_lock_file("multi_row_group").await;
}

async fn generate_mixed_types() {
    clean_output("mixed_types").await;
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(output_dir()).unwrap());
    let mut writer = Writer::new(storage, "mixed_types", make_mixed_schema())
        .await
        .unwrap();
    writer
        .insert_batch(mixed_batch(
            vec![1, 2, 3],
            vec![Some(1.5), None, Some(3.0)],
            vec![Some("a"), Some("b"), None],
            vec![true, false, true],
        ))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    remove_lock_file("mixed_types").await;
}

#[tokio::test]
#[ignore = "run manually to regenerate golden fixtures"]
async fn test_generate_golden_fixtures() {
    generate_simple_int().await;
    generate_multi_commit().await;
    generate_multi_row_group().await;
    generate_mixed_types().await;
}
