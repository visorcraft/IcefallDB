#![cfg(feature = "s3")]

use icefalldb_core::storage::s3::S3Storage;
use icefalldb_core::storage::Storage;
use icefalldb_core::IcefallDBError;
use object_store::aws::AmazonS3Builder;
use object_store::memory::InMemory;
use object_store::path::Path;
use object_store::ObjectStoreExt;
use std::sync::Arc;

#[tokio::test]
async fn test_read_write_roundtrip_via_in_memory() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store.clone(), "mydb");

    store
        .put(&Path::from("mydb/products/_manifest.json"), "hello".into())
        .await
        .unwrap();

    let data = storage.read("products/_manifest.json").await.unwrap();
    assert_eq!(data, b"hello");
}

#[tokio::test]
async fn test_list_returns_direct_children_only() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store.clone(), "mydb");

    store
        .put(&Path::from("mydb/products/_manifest.json"), "{}".into())
        .await
        .unwrap();
    store
        .put(
            &Path::from("mydb/products/_manifests/000000001.json"),
            "{}".into(),
        )
        .await
        .unwrap();
    store
        .put(
            &Path::from("mydb/products/_schemas/000001.json"),
            "{}".into(),
        )
        .await
        .unwrap();

    let entries = storage.list("products").await.unwrap();
    assert_eq!(
        entries,
        vec![
            "products/_manifest.json",
            "products/_manifests",
            "products/_schemas",
        ]
    );
}

#[tokio::test]
async fn test_list_empty_prefix_returns_root_children() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store.clone(), "mydb");

    store
        .put(&Path::from("mydb/products/_manifest.json"), "{}".into())
        .await
        .unwrap();
    store
        .put(&Path::from("mydb/_schemas/000001.json"), "{}".into())
        .await
        .unwrap();

    let entries = storage.list("").await.unwrap();
    assert_eq!(entries, vec!["_schemas", "products"]);
}

#[tokio::test]
async fn test_list_missing_prefix_returns_not_found() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store.clone(), "mydb");

    let err = storage.list("missing").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_path_validation_rejects_traversal_and_absolute_paths() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store, "mydb");

    let traversal_err = storage.read("../etc/passwd").await.unwrap_err();
    assert!(
        format!("{traversal_err}").contains("path traversal"),
        "expected traversal error, got {traversal_err}"
    );

    let absolute_err = storage.list("/etc").await.unwrap_err();
    assert!(
        format!("{absolute_err}").contains("absolute paths are not allowed"),
        "expected absolute path error, got {absolute_err}"
    );
}

#[tokio::test]
async fn test_exists() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store.clone(), "mydb");

    assert!(!storage.exists("products/_manifest.json").await.unwrap());

    store
        .put(&Path::from("mydb/products/_manifest.json"), "{}".into())
        .await
        .unwrap();

    assert!(storage.exists("products/_manifest.json").await.unwrap());
}

#[tokio::test]
async fn test_mutating_methods_are_rejected() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store, "mydb");

    storage
        .write("x", b"x")
        .await
        .expect_err("write should be rejected");
    storage
        .delete("x")
        .await
        .expect_err("delete should be rejected");
    storage
        .rename("x", "y")
        .await
        .expect_err("rename should be rejected");
    assert!(
        storage
            .lock_exclusive("x", std::time::Duration::from_secs(1))
            .await
            .is_err(),
        "lock should be rejected"
    );
}

#[tokio::test]
async fn test_read_missing_file_returns_not_found() {
    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store, "mydb");

    let err = storage.read("products/_manifest.json").await.unwrap_err();
    assert!(format!("{err}").contains("not found"));
}

#[tokio::test]
async fn test_catalog_and_reader_work_over_s3_storage() {
    use icefalldb_core::catalog::Catalog;
    use icefalldb_core::metadata::{Column, Manifest, Schema};
    use icefalldb_core::Reader;

    let store = Arc::new(InMemory::new());
    let storage = S3Storage::new(store.clone(), "mydb");

    // Schema
    let schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".to_string(),
            r#type: "int64".to_string(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 134_217_728,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    // Empty manifest
    let mut manifest = Manifest {
        format_version: 1,
        sequence: 1,
        schema_id: 1,
        row_groups: vec![],
        partition_values: None,
        checksum: String::new(),
        ..Default::default()
    };
    manifest.checksum = manifest.compute_checksum().unwrap();

    let schema_json = serde_json::to_vec_pretty(&schema).unwrap();
    let manifest_pointer = serde_json::to_vec_pretty(&serde_json::json!({"latest": 1})).unwrap();
    let manifest_json = serde_json::to_vec_pretty(&manifest).unwrap();

    store
        .put(
            &Path::from("mydb/products/_schema.json"),
            manifest_pointer.clone().into(),
        )
        .await
        .unwrap();
    store
        .put(
            &Path::from("mydb/products/_schemas/000001.json"),
            schema_json.into(),
        )
        .await
        .unwrap();
    store
        .put(
            &Path::from("mydb/products/_manifest.json"),
            manifest_pointer.into(),
        )
        .await
        .unwrap();
    store
        .put(
            &Path::from("mydb/products/_manifests/000000001.json"),
            manifest_json.into(),
        )
        .await
        .unwrap();

    let catalog = Catalog::load(&storage, "products").await.unwrap();
    assert!(catalog.latest_manifest().is_some());

    let reader = Reader::new(&storage, "products").await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.schema.schema_id, 1);
    assert!(plan.row_groups.is_empty());
}

#[tokio::test]
#[ignore = "requires a running MinIO/LocalStack instance"]
async fn test_s3_with_minio() {
    let Ok(endpoint) = std::env::var("ICEFALLDB_S3_ENDPOINT") else {
        eprintln!("SKIP: ICEFALLDB_S3_ENDPOINT not set");
        return;
    };
    let Ok(bucket) = std::env::var("ICEFALLDB_S3_BUCKET") else {
        eprintln!("SKIP: ICEFALLDB_S3_BUCKET not set");
        return;
    };
    let region = std::env::var("ICEFALLDB_S3_REGION").unwrap_or_else(|_| "us-east-1".into());
    let access_key =
        std::env::var("ICEFALLDB_S3_ACCESS_KEY_ID").unwrap_or_else(|_| "minioadmin".into());
    let secret_key =
        std::env::var("ICEFALLDB_S3_SECRET_ACCESS_KEY").unwrap_or_else(|_| "minioadmin".into());

    let allow_http = endpoint.to_ascii_lowercase().starts_with("http://");
    let store = AmazonS3Builder::new()
        .with_bucket_name(bucket)
        .with_region(region)
        .with_endpoint(endpoint)
        .with_allow_http(allow_http)
        .with_access_key_id(access_key)
        .with_secret_access_key(secret_key)
        .build()
        .unwrap();

    let store = Arc::new(store);
    let storage = S3Storage::new(store.clone(), "testdb");

    let manifest_pointer = serde_json::to_vec_pretty(&serde_json::json!({"latest": 1})).unwrap();
    store
        .put(
            &Path::from("testdb/products/_manifest.json"),
            manifest_pointer.into(),
        )
        .await
        .unwrap();

    assert!(storage.exists("products/_manifest.json").await.unwrap());
}
