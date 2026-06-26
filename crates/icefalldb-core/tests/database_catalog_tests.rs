use icefalldb_core::database_catalog::DatabaseCatalog;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use std::sync::Arc;
use std::time::Duration;

fn test_schema() -> Schema {
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
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 134_217_728,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

#[tokio::test]
async fn test_create_table_updates_catalog() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let catalog = DatabaseCatalog::new(storage);

    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    let schema = test_schema();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();

    let tables = catalog.list_tables().await.unwrap();
    assert_eq!(tables, vec!["events"]);
    assert!(catalog
        .storage
        .exists("events/_manifest.json")
        .await
        .unwrap());
}

#[tokio::test]
async fn test_drop_table_removes_entry() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let catalog = DatabaseCatalog::new(storage);

    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    let schema = test_schema();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();
    catalog.drop_table(&guard, "events").await.unwrap();

    let tables = catalog.list_tables().await.unwrap();
    assert!(tables.is_empty());
}

#[tokio::test]
async fn test_reserved_name_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let catalog = DatabaseCatalog::new(storage);

    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    let schema = test_schema();
    let err = catalog
        .create_table(&guard, "_catalog", &schema)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("invalid path"));
}

#[tokio::test]
async fn test_legacy_discovery_when_catalog_missing() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let catalog = DatabaseCatalog::new(storage);

    // Simulate a legacy table created without a central catalog.
    let _guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    let schema = test_schema();
    icefalldb_core::Writer::create(Arc::clone(&catalog.storage), "legacy", schema)
        .await
        .unwrap();

    // No central catalog saved yet, so catalog.list_tables should be empty.
    // The server will fall back to directory discovery.
    assert!(catalog.list_tables().await.unwrap().is_empty());
}
