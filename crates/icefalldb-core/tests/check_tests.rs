use arrow::array::{ArrayRef, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::check::{CheckResult, Severity};
use icefalldb_core::metadata::{Column, Manifest, RowGroupEntry, RowGroupMeta, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::{compute_row_group_meta, Writer};
use icefalldb_core::{Compactor, GarbageCollector};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::path::Path;
use std::sync::Arc;

fn make_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

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
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

async fn setup_committed_table() -> (tempfile::TempDir, LocalStorage) {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    (tmp, storage)
}

fn make_nullable_batch(ids: Vec<Option<i64>>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, true)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn make_nullable_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: true,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

async fn setup_committed_nullable_table() -> (tempfile::TempDir, LocalStorage) {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = make_nullable_schema();
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_nullable_batch(vec![Some(1), None, Some(3)]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    (tmp, storage)
}

async fn mutate_row_group_meta<F>(storage: &LocalStorage, table: &str, f: F)
where
    F: FnOnce(&mut RowGroupMeta),
{
    let data_path = row_group_data_path(storage, table).await;
    let rg_id = Path::new(&data_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let meta_path = format!("{}/{}.meta", table, rg_id);
    let mut meta: RowGroupMeta =
        serde_json::from_slice(&storage.read(&meta_path).await.unwrap()).unwrap();
    f(&mut meta);
    meta.compute_meta_checksum().unwrap();
    storage
        .write(&meta_path, &serde_json::to_vec_pretty(&meta).unwrap())
        .await
        .unwrap();
}

/// Overwrite a row group's Parquet file and metadata with `batch`.
///
/// `meta_schema` is used to compute the row-group metadata checksum; it should
/// match `batch`'s schema.
async fn overwrite_row_group(
    storage: &LocalStorage,
    table: &str,
    data_path: &str,
    batch: RecordBatch,
    meta_schema: &Schema,
) {
    let rg_id = Path::new(data_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();

    let mut parquet_bytes = Vec::new();
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .build();
    let mut writer = ArrowWriter::try_new(&mut parquet_bytes, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let meta = compute_row_group_meta(
        &rg_id,
        meta_schema.schema_id,
        &batch,
        meta_schema,
        &parquet_bytes,
        table,
        &[],
    )
    .unwrap();
    let meta_path = format!("{}/{}.meta", table, rg_id);

    storage.write(data_path, &parquet_bytes).await.unwrap();
    storage
        .write(&meta_path, &serde_json::to_vec_pretty(&meta).unwrap())
        .await
        .unwrap();
}

fn find_issue<'a>(
    results: &'a CheckResult,
    code: &str,
) -> Option<&'a icefalldb_core::check::CheckIssue> {
    results.issues.iter().find(|i| i.code == code)
}

#[tokio::test]
async fn test_check_valid_table_passes() {
    let (_tmp, storage) = setup_committed_table().await;
    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed);
    assert!(result.issues.is_empty());
}

#[tokio::test]
async fn test_check_missing_schema_pointer() {
    let (_tmp, storage) = setup_committed_table().await;
    storage.delete("products/_schema.json").await.unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "MISSING_SCHEMA_POINTER").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    // A missing schema pointer means the table is not initialized; the checker
    // must not proceed to validate manifests or row groups.
    assert!(
        !result.issues.iter().any(|i| i.code == "MISSING_MANIFEST"),
        "should not check manifest files when schema pointer is missing"
    );
    assert!(
        !result
            .issues
            .iter()
            .any(|i| i.code.starts_with("ROW_GROUP_")),
        "should not check row groups when schema pointer is missing"
    );
}

#[tokio::test]
async fn test_check_missing_manifest_pointer() {
    let (_tmp, storage) = setup_committed_table().await;
    storage.delete("products/_manifest.json").await.unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "MISSING_MANIFEST_POINTER").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_corrupt_manifest_checksum() {
    let (_tmp, storage) = setup_committed_table().await;
    let seq: u64 = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        pointer["latest"].as_u64().unwrap()
    };
    let manifest_path = format!("products/{}", Manifest::filename(seq));
    let mut manifest: Manifest =
        serde_json::from_slice(&storage.read(&manifest_path).await.unwrap()).unwrap();
    manifest.checksum =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000".into();
    storage
        .write(
            &manifest_path,
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "CORRUPT_MANIFEST").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_missing_row_group_meta() {
    let (_tmp, storage) = setup_committed_table().await;
    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap()
    };
    let meta_path = format!("products/{}", manifest.row_groups[0].meta);
    storage.delete(&meta_path).await.unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "MISSING_ROW_GROUP_META").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_corrupt_parquet_checksum() {
    let (_tmp, storage) = setup_committed_table().await;
    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap()
    };
    let data_path = format!("products/{}", manifest.row_groups[0].data);
    let mut parquet = storage.read(&data_path).await.unwrap();
    parquet[0] = parquet[0].wrapping_add(1);
    storage.write(&data_path, &parquet).await.unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "ROW_GROUP_CHECKSUM_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_unreferenced_files() {
    let (_tmp, storage) = setup_committed_table().await;
    storage
        .write("products/rg_orphan.parquet", b"orphan-data")
        .await
        .unwrap();
    storage
        .write("products/rg_orphan.meta", b"orphan-meta")
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed);
    let warnings: Vec<_> = result
        .issues
        .iter()
        .filter(|i| i.code == "UNREFERENCED_ROW_GROUP")
        .collect();
    assert_eq!(warnings.len(), 2);
    for warning in &warnings {
        assert_eq!(warning.severity, Severity::Warning);
    }
}

#[tokio::test]
async fn test_check_stale_intent() {
    let (_tmp, storage) = setup_committed_table().await;
    storage
        .write(
            "products/_staging/intents/txn_stale.json",
            b"{\"txn_id\":\"txn_stale\"}",
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed);
    let issue = find_issue(&result, "STALE_INTENT").unwrap();
    assert_eq!(issue.severity, Severity::Warning);
}

async fn row_group_data_path(storage: &LocalStorage, table: &str) -> String {
    let pointer: serde_json::Value = serde_json::from_slice(
        &storage
            .read(&format!("{}/_manifest.json", table))
            .await
            .unwrap(),
    )
    .unwrap();
    let seq = pointer["latest"].as_u64().unwrap();
    let manifest: Manifest = serde_json::from_slice(
        &storage
            .read(&format!("{}/{}", table, Manifest::filename(seq)))
            .await
            .unwrap(),
    )
    .unwrap();
    format!("{}/{}", table, manifest.row_groups[0].data)
}

#[tokio::test]
async fn test_check_schema_column_missing() {
    let (_tmp, storage) = setup_committed_table().await;
    let data_path = row_group_data_path(&storage, "products").await;

    // Replace the row group with a Parquet file that omits the declared `id` column.
    let bad_batch = {
        let schema = ArrowSchema::new(vec![Field::new("x", DataType::Int64, false)]);
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        RecordBatch::try_new(Arc::new(schema), vec![array]).unwrap()
    };
    let meta_schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "x".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    overwrite_row_group(&storage, "products", &data_path, bad_batch, &meta_schema).await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "SCHEMA_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    assert!(
        issue.message.contains("missing column 'id'"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains("expected int64"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains(&data_path),
        "message: {}",
        issue.message
    );
}

#[tokio::test]
async fn test_check_schema_column_type_mismatch() {
    let (_tmp, storage) = setup_committed_table().await;
    let data_path = row_group_data_path(&storage, "products").await;

    // The schema declares `id` as int64, but the file stores it as int32.
    let bad_batch = {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int32, false)]);
        let array: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        RecordBatch::try_new(Arc::new(schema), vec![array]).unwrap()
    };
    let meta_schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int32".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    overwrite_row_group(&storage, "products", &data_path, bad_batch, &meta_schema).await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "SCHEMA_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    assert!(
        issue.message.contains("column 'id'"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains("expected int64"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains("got Int32"),
        "message: {}",
        issue.message
    );
}

#[tokio::test]
async fn test_check_schema_extra_column_error() {
    let (_tmp, storage) = setup_committed_table().await;
    let data_path = row_group_data_path(&storage, "products").await;

    // The file contains an `extra` column that is not declared in the schema.
    let batch = {
        let schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("extra", DataType::Int64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef,
                Arc::new(Int64Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap()
    };
    let meta_schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "extra".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    overwrite_row_group(&storage, "products", &data_path, batch, &meta_schema).await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "SCHEMA_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    assert!(
        issue.message.contains("extra column 'extra'"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains(&data_path),
        "message: {}",
        issue.message
    );
}

#[tokio::test]
async fn test_check_schema_nullability_mismatch() {
    let (_tmp, storage) = setup_committed_table().await;
    let data_path = row_group_data_path(&storage, "products").await;

    // The schema declares `id` as non-nullable, but the file stores it as nullable.
    let bad_batch = {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, true)]);
        let array: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3]));
        RecordBatch::try_new(Arc::new(schema), vec![array]).unwrap()
    };
    let meta_schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: true,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    overwrite_row_group(&storage, "products", &data_path, bad_batch, &meta_schema).await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "SCHEMA_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    assert!(
        issue.message.contains("column 'id'"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains("expected nullable=false"),
        "message: {}",
        issue.message
    );
}

#[tokio::test]
async fn test_check_older_schema_row_group_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let current_schema = Schema {
        schema_id: 2,
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
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", current_schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Add an older compatible schema file.
    let older_schema = Schema {
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
        max_field_id: 0,
        dropped_columns: vec![],
    };
    storage
        .write(
            "products/_schemas/000001.json",
            &serde_json::to_vec_pretty(&older_schema).unwrap(),
        )
        .await
        .unwrap();

    // Mutate the row group meta to reference the older schema.
    let data_path = row_group_data_path(&storage, "products").await;
    let rg_id = Path::new(&data_path)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let meta_path = format!("products/{}.meta", rg_id);
    let mut meta: RowGroupMeta =
        serde_json::from_slice(&storage.read(&meta_path).await.unwrap()).unwrap();
    meta.schema_id = 1;
    meta.compute_meta_checksum().unwrap();
    storage
        .write(&meta_path, &serde_json::to_vec_pretty(&meta).unwrap())
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed, "issues: {:?}", result.issues);
    assert!(!result
        .issues
        .iter()
        .any(|i| i.code == "SCHEMA_ID_MISMATCH_ROW_GROUP"));
}

#[tokio::test]
async fn test_check_row_count_mismatch() {
    let (_tmp, storage) = setup_committed_table().await;
    mutate_row_group_meta(&storage, "products", |meta| {
        meta.rows = 999;
    })
    .await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "ROW_COUNT_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_null_count_mismatch() {
    let (_tmp, storage) = setup_committed_nullable_table().await;
    mutate_row_group_meta(&storage, "products", |meta| {
        meta.columns.get_mut("id").unwrap().nulls = 999;
    })
    .await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "NULL_COUNT_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_min_max_mismatch() {
    let (_tmp, storage) = setup_committed_nullable_table().await;
    mutate_row_group_meta(&storage, "products", |meta| {
        meta.columns.get_mut("id").unwrap().min = Some(42.into());
    })
    .await;

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "MIN_MAX_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
}

#[tokio::test]
async fn test_check_current_schema_reconciliation() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    // Write the initial row group with schema 1: just `id`.
    let older_schema = Schema {
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
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", older_schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Add a newer schema (id 2) with an extra nullable column.
    let current_schema_nullable = Schema {
        schema_id: 2,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "name".into(),
                r#type: "utf8".into(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    storage
        .write(
            "products/_schemas/000002.json",
            &serde_json::to_vec_pretty(&current_schema_nullable).unwrap(),
        )
        .await
        .unwrap();

    // Point the manifest and schema pointers at the new schema.
    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let mut m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        m.schema_id = 2;
        m.checksum = m.compute_checksum().unwrap();
        m
    };
    storage
        .write(
            &format!("products/{}", Manifest::filename(manifest.sequence)),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_schema.json",
            &serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();

    // The older row group lacks the nullable `name` column: warn, but pass.
    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed, "issues: {:?}", result.issues);
    let issue = find_issue(&result, "MISSING_NULLABLE_COLUMN").unwrap();
    assert_eq!(issue.severity, Severity::Warning);
    assert!(
        issue.message.contains("column 'name'"),
        "message: {}",
        issue.message
    );

    // Now make the added column non-nullable: missing it becomes an error.
    let current_schema_required = Schema {
        schema_id: 2,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "name".into(),
                r#type: "utf8".into(),
                nullable: false,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    storage
        .write(
            "products/_schemas/000002.json",
            &serde_json::to_vec_pretty(&current_schema_required).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "SCHEMA_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    assert!(
        issue.message.contains("missing column 'name'"),
        "message: {}",
        issue.message
    );
}

#[tokio::test]
async fn test_check_dropped_column_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let older_schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "old_col".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", older_schema)
        .await
        .unwrap();
    let batch = {
        let schema = ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("old_col", DataType::Int64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(Int64Array::from(vec![4, 5, 6])),
            ],
        )
        .unwrap()
    };
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let current_schema = Schema {
        schema_id: 2,
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
        max_field_id: 0,
        dropped_columns: vec!["old_col".into()],
    };
    storage
        .write(
            "products/_schemas/000002.json",
            &serde_json::to_vec_pretty(&current_schema).unwrap(),
        )
        .await
        .unwrap();

    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let mut m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        m.schema_id = 2;
        m.checksum = m.compute_checksum().unwrap();
        m
    };
    storage
        .write(
            &format!("products/{}", Manifest::filename(manifest.sequence)),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_schema.json",
            &serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed, "issues: {:?}", result.issues);
}

#[tokio::test]
async fn test_check_type_promotion_accepted() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let older_schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int32".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", older_schema)
        .await
        .unwrap();
    let batch = {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int32, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap()
    };
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let current_schema = Schema {
        schema_id: 2,
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
        max_field_id: 0,
        dropped_columns: vec![],
    };
    storage
        .write(
            "products/_schemas/000002.json",
            &serde_json::to_vec_pretty(&current_schema).unwrap(),
        )
        .await
        .unwrap();

    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let mut m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        m.schema_id = 2;
        m.checksum = m.compute_checksum().unwrap();
        m
    };
    storage
        .write(
            &format!("products/{}", Manifest::filename(manifest.sequence)),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_schema.json",
            &serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed, "issues: {:?}", result.issues);
}

#[tokio::test]
async fn test_check_invalid_type_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let older_schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "utf8".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", older_schema)
        .await
        .unwrap();
    let batch = {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Utf8, false)]);
        let array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        RecordBatch::try_new(Arc::new(schema), vec![array]).unwrap()
    };
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let current_schema = Schema {
        schema_id: 2,
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
        max_field_id: 0,
        dropped_columns: vec![],
    };
    storage
        .write(
            "products/_schemas/000002.json",
            &serde_json::to_vec_pretty(&current_schema).unwrap(),
        )
        .await
        .unwrap();

    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let mut m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        m.schema_id = 2;
        m.checksum = m.compute_checksum().unwrap();
        m
    };
    storage
        .write(
            &format!("products/{}", Manifest::filename(manifest.sequence)),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_schema.json",
            &serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(!result.passed);
    let issue = find_issue(&result, "SCHEMA_MISMATCH").unwrap();
    assert_eq!(issue.severity, Severity::Error);
    assert!(
        issue.message.contains("column 'id'"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains("expected int64"),
        "message: {}",
        issue.message
    );
    assert!(
        issue.message.contains("got Utf8"),
        "message: {}",
        issue.message
    );
}

#[tokio::test]
async fn test_check_utf8_to_large_utf8_promotion_accepted() {
    use arrow::array::{ArrayRef, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use icefalldb_core::metadata::{Column, Manifest, Schema};
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::Writer;
    use std::sync::Arc;

    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    let older_schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "name".into(),
            r#type: "utf8".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", older_schema)
        .await
        .unwrap();
    let batch = {
        let schema = ArrowSchema::new(vec![Field::new("name", DataType::Utf8, false)]);
        let array: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        RecordBatch::try_new(Arc::new(schema), vec![array]).unwrap()
    };
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let current_schema = Schema {
        schema_id: 2,
        columns: vec![Column {
            name: "name".into(),
            r#type: "large_utf8".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    storage
        .write(
            "products/_schemas/000002.json",
            &serde_json::to_vec_pretty(&current_schema).unwrap(),
        )
        .await
        .unwrap();

    let manifest: Manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let mut m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        m.schema_id = 2;
        m.checksum = m.compute_checksum().unwrap();
        m
    };
    storage
        .write(
            &format!("products/{}", Manifest::filename(manifest.sequence)),
            &serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_schema.json",
            &serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(result.passed, "issues: {:?}", result.issues);
}

/// Legacy manifests with an empty `checksum` field must still protect their
/// referenced row-group files from orphan warnings. Regression test for M06.
#[tokio::test]
async fn test_check_legacy_manifests_protect_referenced_row_groups() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    // Create a current snapshot with one valid row group.
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage.clone()), "products", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Two additional row-group files referenced only by legacy manifests.
    storage
        .write("products/rg_legacy1.parquet", b"legacy-data-1")
        .await
        .unwrap();
    storage
        .write("products/rg_legacy1.meta", b"legacy-meta-1")
        .await
        .unwrap();
    storage
        .write("products/rg_legacy2.parquet", b"legacy-data-2")
        .await
        .unwrap();
    storage
        .write("products/rg_legacy2.meta", b"legacy-meta-2")
        .await
        .unwrap();

    // Two legacy manifests with empty checksums referencing the legacy row groups.
    for (seq, data, meta) in [
        (2u64, "rg_legacy1.parquet", "rg_legacy1.meta"),
        (3u64, "rg_legacy2.parquet", "rg_legacy2.meta"),
    ] {
        let legacy = Manifest {
            format_version: 1,
            sequence: seq,
            schema_id: 1,
            row_groups: vec![RowGroupEntry {
                data: data.into(),
                meta: meta.into(),
                ..Default::default()
            }],
            checksum: String::new(),
            ..Default::default()
        };
        storage
            .write(
                &format!("products/{}", Manifest::filename(seq)),
                &serde_json::to_vec_pretty(&legacy).unwrap(),
            )
            .await
            .unwrap();
    }

    // Current manifest at sequence 4 with a valid checksum, pointing to the
    // original row group. It does not reference the legacy row groups.
    let current_manifest = {
        let data = storage.read("products/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&data).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let mut m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        m.sequence = 4;
        m.checksum = m.compute_checksum().unwrap();
        m
    };
    storage
        .write(
            &format!("products/{}", Manifest::filename(4)),
            &serde_json::to_vec_pretty(&current_manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_manifest.json",
            &serde_json::to_vec_pretty(&serde_json::json!({ "latest": 4 })).unwrap(),
        )
        .await
        .unwrap();

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(
        result.passed,
        "check must pass, got issues: {:?}",
        result.issues
    );
    let orphan_warnings: Vec<_> = result
        .issues
        .iter()
        .filter(|i| i.code == "UNREFERENCED_ROW_GROUP")
        .collect();
    assert!(
        orphan_warnings.is_empty(),
        "check emitted orphan warnings for files referenced by legacy manifests: {:?}",
        orphan_warnings
    );
}

/// After optimize retains older snapshots, `check` must not warn that files
/// referenced by those older snapshots are unreferenced. Regression test for M08.
#[tokio::test]
async fn test_check_no_orphan_warnings_for_retained_snapshot_files() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();

    // Build three snapshots so GC can retain snapshots 2 and 3 while snapshot 3
    // becomes the latest. Each insert creates a new row group file.
    let schema = make_schema();
    for ids in [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]] {
        let mut writer = Writer::new(Arc::new(storage.clone()), "products", schema.clone())
            .await
            .unwrap();
        writer.insert_batch(make_batch(ids)).await.unwrap();
        writer.commit().await.unwrap();
    }

    // Optimize with retain_snapshots=2: snapshots 3 and 4 are retained.
    // Snapshot 4 (the optimized snapshot) references new compacted row groups,
    // while retained snapshot 3 still references its original row group files.
    // Those original files must not be flagged as orphans.
    let compactor = Compactor::new(&storage, "products");
    compactor.compact().await.unwrap();

    let gc = GarbageCollector::new(&storage, "products", 2);
    let gc_result = gc.run().await.unwrap();
    assert!(
        gc_result.retained_snapshots.contains(&3),
        "snapshot 3 should be retained, got {:?}",
        gc_result.retained_snapshots
    );

    let checker = icefalldb_core::Checker::new(&storage, "products");
    let result = checker.check().await.unwrap();
    assert!(
        result.passed,
        "check must pass, got issues: {:?}",
        result.issues
    );
    let orphan_warnings: Vec<_> = result
        .issues
        .iter()
        .filter(|i| i.code == "UNREFERENCED_ROW_GROUP")
        .collect();
    assert!(
        orphan_warnings.is_empty(),
        "check emitted orphan warnings for files referenced by retained snapshot 3: {:?}",
        orphan_warnings
    );
}
