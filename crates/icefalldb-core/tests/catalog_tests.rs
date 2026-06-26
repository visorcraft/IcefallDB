use icefalldb_core::catalog::Catalog;
use icefalldb_core::metadata::{Column, Manifest, RowGroupEntry, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::IcefallDBError;

#[tokio::test]
async fn test_catalog_loads_empty_table() {
    let storage = MemoryStorage::new();
    let catalog = Catalog::load(&storage, "products").await.unwrap();
    assert!(catalog.latest_manifest().is_none());
}

#[tokio::test]
async fn test_catalog_loads_valid_manifest() {
    let storage = MemoryStorage::new();
    let table = "products";

    let mut manifest = Manifest {
        format_version: 1,
        sequence: 1,
        schema_id: 1,
        row_groups: vec![RowGroupEntry {
            data: "rg0.parquet".into(),
            meta: "rg0.meta.json".into(),
            ..Default::default()
        }],
        partition_values: None,
        checksum: String::new(),
        ..Default::default()
    };
    manifest.checksum = manifest.compute_checksum().unwrap();

    let schema = Schema {
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
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(1)),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&schema).unwrap(),
        )
        .await
        .unwrap();

    let catalog = Catalog::load(&storage, table).await.unwrap();
    assert!(catalog.latest_manifest().is_some());
    assert_eq!(catalog.latest_manifest().unwrap().sequence, 1);
    assert!(catalog.latest_schema().is_some());
    assert_eq!(catalog.latest_schema().unwrap().schema_id, 1);
}

#[tokio::test]
async fn test_catalog_rejects_checksum_mismatch() {
    let storage = MemoryStorage::new();
    let table = "products";

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
    // Tamper with the manifest after computing its checksum.
    manifest.row_groups.push(RowGroupEntry {
        data: "tampered.parquet".into(),
        meta: "tampered.meta.json".into(),
        ..Default::default()
    });

    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(1)),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .await
        .unwrap();

    let result = Catalog::load(&storage, table).await;
    assert!(
        matches!(result, Err(IcefallDBError::ChecksumMismatch { .. })),
        "expected ChecksumMismatch, got {:?}",
        result.as_ref().map(|_| ())
    );
}

#[tokio::test]
async fn test_catalog_rejects_malformed_pointer() {
    let storage = MemoryStorage::new();
    let table = "products";

    storage
        .write(&format!("{}/_manifest.json", table), b"{}")
        .await
        .unwrap();

    let result = Catalog::load(&storage, table).await;
    match result {
        Err(IcefallDBError::InvalidManifestPointer(msg)) => {
            assert!(msg.contains("missing or invalid 'latest'"));
        }
        other => panic!(
            "expected malformed pointer error, got {:?}",
            other.as_ref().map(|_| ())
        ),
    }
}

#[tokio::test]
async fn test_catalog_rejects_schema_id_mismatch() {
    let storage = MemoryStorage::new();
    let table = "products";

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

    let schema = Schema {
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
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(1)),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&schema).unwrap(),
        )
        .await
        .unwrap();

    let result = Catalog::load(&storage, table).await;
    assert!(
        matches!(result, Err(IcefallDBError::SchemaMismatch { .. })),
        "expected SchemaMismatch, got {:?}",
        result.as_ref().map(|_| ())
    );
}

#[tokio::test]
async fn test_catalog_refresh_retains_state_on_missing_pointer() {
    let storage = MemoryStorage::new();
    let table = "products";

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

    let schema = Schema {
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
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(1)),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&schema).unwrap(),
        )
        .await
        .unwrap();

    let mut catalog = Catalog::load(&storage, table).await.unwrap();
    assert!(catalog.latest_manifest().is_some());

    storage
        .delete(&format!("{}/_manifest.json", table))
        .await
        .unwrap();

    catalog.refresh().await.unwrap();
    assert!(catalog.latest_manifest().is_some());
    assert!(catalog.latest_schema().is_some());
}

#[tokio::test]
async fn test_catalog_refresh_retains_state_on_error() {
    let storage = MemoryStorage::new();
    let table = "products";

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

    let schema = Schema {
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
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(1)),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&schema).unwrap(),
        )
        .await
        .unwrap();

    let mut catalog = Catalog::load(&storage, table).await.unwrap();
    assert_eq!(catalog.latest_manifest().unwrap().sequence, 1);
    assert_eq!(catalog.latest_schema().unwrap().schema_id, 1);

    // Corrupt the pointer so that refresh fails validation.
    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":2}"#)
        .await
        .unwrap();

    let result = catalog.refresh().await;
    assert!(
        matches!(result, Err(IcefallDBError::NotFound(_))),
        "expected NotFound for missing manifest 2, got {:?}",
        result.as_ref().map(|_| ())
    );

    // The original valid manifest and schema should still be cached.
    assert_eq!(catalog.latest_manifest().unwrap().sequence, 1);
    assert_eq!(catalog.latest_schema().unwrap().schema_id, 1);
}

#[tokio::test]
async fn test_catalog_rejects_sequence_mismatch() {
    let storage = MemoryStorage::new();
    let table = "products";

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

    let schema = Schema {
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
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    // Pointer references sequence 2, but only sequence 1 manifest exists.
    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":2}"#)
        .await
        .unwrap();
    // Write the sequence-1 manifest to the sequence-2 filename so the pointer
    // resolves a manifest whose internal sequence does not match.
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(2)),
            &serde_json::to_vec(&manifest).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&schema).unwrap(),
        )
        .await
        .unwrap();

    let result = Catalog::load(&storage, table).await;
    match result {
        Err(IcefallDBError::InvalidManifestPointer(msg)) => {
            assert!(msg.contains("sequence mismatch"));
            assert!(msg.contains("pointer expects 2"));
            assert!(msg.contains("manifest has 1"));
        }
        other => panic!(
            "expected InvalidManifestPointer for sequence mismatch, got {:?}",
            other.as_ref().map(|_| ())
        ),
    }
}

#[tokio::test]
async fn test_catalog_accepts_latest_zero_for_empty_table() {
    let storage = MemoryStorage::new();
    let table = "products";

    // Simulate a freshly created table: manifest pointer at 0 with no manifest
    // snapshots.
    storage
        .write(&format!("{}/_manifest.json", table), br#"{"latest":0}"#)
        .await
        .unwrap();

    let catalog = Catalog::load(&storage, table).await.unwrap();
    assert!(catalog.latest_manifest().is_none());
    assert!(catalog.latest_schema().is_none());
}

#[tokio::test]
async fn test_catalog_rejects_missing_manifest_pointer_when_schema_exists() {
    let storage = MemoryStorage::new();
    let table = "products";

    // A schema pointer without a manifest pointer means the table is partially
    // initialized, not empty.
    storage
        .write(&format!("{}/_schema.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&Schema {
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
                row_group_target_rows: 1000,
                row_group_target_bytes: 1024 * 1024,
                max_field_id: 0,
                dropped_columns: vec![],
            })
            .unwrap(),
        )
        .await
        .unwrap();

    let result = Catalog::load(&storage, table).await;
    assert!(
        matches!(result, Err(IcefallDBError::MissingManifestPointer { .. })),
        "expected MissingManifestPointer for partial table, got {:?}",
        result.as_ref().map(|_| ())
    );
}
