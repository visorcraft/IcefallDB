use icefalldb_core::metadata::{
    Column, ColumnStats, Manifest, RowGroupEntry, RowGroupMeta, Schema,
};
use serde_json::json;
use std::collections::HashMap;

#[test]
fn test_checksum_json_is_stable() {
    let value = json!({"b": 2, "a": 1});
    let cs1 = icefalldb_core::metadata::checksum_json(&value);
    let cs2 = icefalldb_core::metadata::checksum_json(&json!({"a": 1, "b": 2}));
    assert_eq!(cs1, cs2);
}

#[test]
fn test_checksum_json_ignores_key_order() {
    let cs1 = icefalldb_core::metadata::checksum_json(&json!({
        "z": "last",
        "a": "first",
        "m": "middle"
    }));
    let cs2 = icefalldb_core::metadata::checksum_json(&json!({
        "a": "first",
        "m": "middle",
        "z": "last"
    }));
    assert_eq!(cs1, cs2);
}

#[test]
fn test_checksum_json_preserves_array_order() {
    let cs1 = icefalldb_core::metadata::checksum_json(&json!({"items": [1, 2, 3]}));
    let cs2 = icefalldb_core::metadata::checksum_json(&json!({"items": [3, 2, 1]}));
    assert_ne!(cs1, cs2);
}

#[test]
fn test_checksum_json_nested_objects() {
    let cs1 = icefalldb_core::metadata::checksum_json(&json!({
        "outer": {
            "b": { "y": 2, "x": 1 },
            "a": ["keep", "order"]
        }
    }));
    let cs2 = icefalldb_core::metadata::checksum_json(&json!({
        "outer": {
            "a": ["keep", "order"],
            "b": { "x": 1, "y": 2 }
        }
    }));
    assert_eq!(cs1, cs2);
}

#[test]
fn test_schema_round_trip() {
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".to_string(),
                r#type: "int64".to_string(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "name".to_string(),
                r#type: "string".to_string(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: Some(vec!["id".to_string()]),
        sort: Some(vec!["id".to_string()]),
        agg_group_keys: None,
        row_group_target_rows: 100_000,
        row_group_target_bytes: 16 * 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec!["old_col".to_string()],
    };

    let serialized = serde_json::to_string(&schema).expect("schema serializes");
    let deserialized: Schema = serde_json::from_str(&serialized).expect("schema deserializes");
    assert_eq!(schema, deserialized);
}

#[test]
fn test_manifest_round_trip() {
    let manifest = Manifest {
        format_version: 1,
        sequence: 42,
        schema_id: 1,
        row_groups: vec![
            RowGroupEntry {
                data: "data/rg1.parquet".to_string(),
                meta: "meta/rg1.meta.json".to_string(),
                ..Default::default()
            },
            RowGroupEntry {
                data: "data/rg2.parquet".to_string(),
                meta: "meta/rg2.meta.json".to_string(),
                ..Default::default()
            },
        ],
        partition_values: Some(
            [(
                "meta/rg1.meta.json".to_string(),
                [("id".to_string(), json!(1))].into_iter().collect(),
            )]
            .into_iter()
            .collect(),
        ),
        checksum: "sha256:abc".to_string(),
        ..Default::default()
    };

    let serialized = serde_json::to_string(&manifest).expect("manifest serializes");
    let deserialized: Manifest = serde_json::from_str(&serialized).expect("manifest deserializes");
    assert_eq!(manifest, deserialized);
}

#[test]
fn test_manifest_self_checksum() {
    let mut manifest = Manifest {
        format_version: 1,
        sequence: 7,
        schema_id: 1,
        row_groups: vec![RowGroupEntry {
            data: "data/rg.parquet".to_string(),
            meta: "meta/rg.meta.json".to_string(),
            ..Default::default()
        }],
        partition_values: None,
        checksum: "".to_string(),
        ..Default::default()
    };

    let checksum = manifest.compute_checksum().unwrap();
    assert!(checksum.starts_with("sha256:"));
    assert_ne!(checksum, "");

    manifest.checksum = checksum.clone();
    assert_eq!(manifest.checksum, checksum);
    assert!(manifest.verify_checksum().unwrap());
}

#[test]
fn test_manifest_verify_checksum_detects_tampering() {
    let mut manifest = Manifest {
        format_version: 1,
        sequence: 7,
        schema_id: 1,
        row_groups: vec![RowGroupEntry {
            data: "data/rg.parquet".to_string(),
            meta: "meta/rg.meta.json".to_string(),
            ..Default::default()
        }],
        partition_values: None,
        checksum: "".to_string(),
        ..Default::default()
    };

    let checksum = manifest.compute_checksum().unwrap();
    manifest.checksum = checksum;
    assert!(manifest.verify_checksum().unwrap());

    manifest.sequence = 8;
    assert!(!manifest.verify_checksum().unwrap());
}

#[test]
fn test_manifest_checksum_distinguishes_none_and_empty_partition_values() {
    let manifest_none = Manifest {
        format_version: 1,
        sequence: 3,
        schema_id: 1,
        row_groups: vec![RowGroupEntry {
            data: "data/rg.parquet".to_string(),
            meta: "meta/rg.meta.json".to_string(),
            ..Default::default()
        }],
        partition_values: None,
        checksum: "".to_string(),
        ..Default::default()
    };

    let manifest_empty = Manifest {
        format_version: 1,
        sequence: 3,
        schema_id: 1,
        row_groups: vec![RowGroupEntry {
            data: "data/rg.parquet".to_string(),
            meta: "meta/rg.meta.json".to_string(),
            ..Default::default()
        }],
        partition_values: Some(HashMap::new()),
        checksum: "".to_string(),
        ..Default::default()
    };

    assert_ne!(
        manifest_none.compute_checksum().unwrap(),
        manifest_empty.compute_checksum().unwrap()
    );
}

#[test]
fn test_row_group_meta_checksum_over_parquet_data() {
    let parquet_bytes = b"fake parquet data";
    let mut meta = RowGroupMeta {
        row_group: "rg_abc123".to_string(),
        schema_id: 1,
        rows: 1000,
        columns: [(
            "id".to_string(),
            ColumnStats {
                min: Some(json!(1)),
                max: Some(json!(1000)),
                nulls: 0,
            },
        )]
        .into_iter()
        .collect(),
        sort: Some(vec!["id".to_string()]),
        checksum: "".to_string(),
        meta_checksum: "".to_string(),
        ..Default::default()
    };

    let checksum = meta.compute_checksum(parquet_bytes).unwrap();
    assert!(checksum.starts_with("sha256:"));
    assert_eq!(meta.checksum, checksum);
    assert!(!meta.meta_checksum.is_empty());
    assert!(meta.verify_against_data(parquet_bytes));
    assert!(meta.verify_meta_checksum().unwrap());

    let other_bytes = b"different data";
    assert!(!meta.verify_against_data(other_bytes));

    // Tampering with a metadata field must invalidate the meta checksum.
    meta.rows = 999;
    assert!(!meta.verify_meta_checksum().unwrap());
}

#[test]
fn test_row_group_meta_checksum_round_trip() {
    let parquet_bytes = b"fake parquet data";
    let mut meta = RowGroupMeta {
        row_group: "rg_abc123".to_string(),
        schema_id: 1,
        rows: 1000,
        columns: [(
            "id".to_string(),
            ColumnStats {
                min: Some(json!(1)),
                max: Some(json!(1000)),
                nulls: 0,
            },
        )]
        .into_iter()
        .collect(),
        sort: Some(vec!["id".to_string()]),
        checksum: "".to_string(),
        meta_checksum: "".to_string(),
        ..Default::default()
    };

    meta.compute_checksum(parquet_bytes).unwrap();
    assert!(meta.verify_meta_checksum().unwrap());

    let serialized = serde_json::to_vec_pretty(&meta).expect("meta serializes");
    let deserialized: RowGroupMeta =
        serde_json::from_slice(&serialized).expect("meta deserializes");
    assert_eq!(meta, deserialized);
    assert!(deserialized.verify_meta_checksum().unwrap());
}

#[test]
fn test_row_group_meta_checksum_detects_meta_tampering() {
    let parquet_bytes = b"fake parquet data";
    let mut meta = RowGroupMeta {
        row_group: "rg_abc123".to_string(),
        schema_id: 1,
        rows: 1000,
        columns: [(
            "id".to_string(),
            ColumnStats {
                min: Some(json!(1)),
                max: Some(json!(1000)),
                nulls: 0,
            },
        )]
        .into_iter()
        .collect(),
        sort: Some(vec!["id".to_string()]),
        checksum: "".to_string(),
        meta_checksum: "".to_string(),
        ..Default::default()
    };

    meta.compute_checksum(parquet_bytes).unwrap();
    assert!(meta.verify_meta_checksum().unwrap());

    meta.schema_id = 2;
    assert!(!meta.verify_meta_checksum().unwrap());

    // Recomputing the checksum makes it valid again.
    meta.compute_checksum(parquet_bytes).unwrap();
    assert!(meta.verify_meta_checksum().unwrap());
}
