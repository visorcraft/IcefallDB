use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::doctor::verify_history;
use icefalldb_core::metadata::{Column, Manifest, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use std::collections::HashMap;
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

async fn setup_table_with_two_inserts() -> (Arc<dyn Storage>, String) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let table = "chain_test".to_string();
    let mut writer = Writer::new(Arc::clone(&storage), &table, schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    (storage, table)
}

/// Corrupt manifest `seq` on disk by flipping a field, without touching the
/// stored `checksum` field.  This makes `verify_checksum()` return false.
async fn tamper_manifest(storage: &Arc<dyn Storage>, table: &str, seq: u64) {
    let path = format!("{}/{}", table, Manifest::filename(seq));
    let bytes = storage.read(&path).await.unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    // Corrupt row_counts if present, otherwise inject a garbage field.
    // Either way the stored `checksum` field no longer matches the content.
    if let Some(arr) = value.get_mut("row_counts").and_then(|v| v.as_array_mut()) {
        if let Some(first) = arr.first_mut() {
            if let Some(n) = first.as_u64() {
                *first = serde_json::Value::Number((n + 999).into());
            }
        }
    } else {
        value["__corrupted__"] = serde_json::Value::Bool(true);
    }
    storage
        .write(&path, &serde_json::to_vec(&value).unwrap())
        .await
        .unwrap();
}

// ── tests ──────────────────────────────────────────────────────────────────

/// A freshly written two-commit table has an intact chain.
#[tokio::test]
async fn intact_chain_verifies() {
    let (storage, table) = setup_table_with_two_inserts().await;
    let report = verify_history(storage.as_ref(), &table).await.unwrap();
    assert!(
        report.intact,
        "fresh chain must be intact: {:?}",
        report
            .breaks
            .iter()
            .map(|b| format!("seq={}: {}", b.sequence, b.reason))
            .collect::<Vec<_>>()
    );
    assert_eq!(report.breaks.len(), 0);
    assert_eq!(report.oldest, 1);
    assert_eq!(report.latest, 2);
}

/// Tampering with manifest 1 on disk (without fixing manifest 2's parent_hash)
/// must be detected: the report must be non-intact and at least one break must
/// name an affected sequence.
#[tokio::test]
async fn tampered_manifest_is_flagged() {
    let (storage, table) = setup_table_with_two_inserts().await;
    tamper_manifest(&storage, &table, 1).await;

    let report = verify_history(storage.as_ref(), &table).await.unwrap();
    assert!(!report.intact, "tampered chain must not be intact");
    assert!(
        report
            .breaks
            .iter()
            .any(|b| b.sequence == 1 || b.sequence == 2),
        "at least one break must name an affected sequence; got: {:?}",
        report
            .breaks
            .iter()
            .map(|b| format!("seq={}: {}", b.sequence, b.reason))
            .collect::<Vec<_>>()
    );
}

/// A manifest with `parent_hash: None` while a predecessor is present on disk
/// must be treated as an anchor (genesis / GC-pruned / legacy), NOT as a break.
///
/// This guards the corrected anchor-vs-break logic against regressions: the
/// original brief draft incorrectly flagged `None` as a break.
#[tokio::test]
async fn legacy_none_parent_hash_is_anchor_not_break() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let table = "legacy_anchor_test".to_string();

    // Write m1 via the normal Writer path (genesis, parent_hash = None).
    let mut writer = Writer::new(Arc::clone(&storage), &table, schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Construct m2 manually with parent_hash = None (simulates a legacy manifest
    // written before hash-chaining was introduced).  The self-checksum is valid.
    let mut m2 = Manifest {
        format_version: 1,
        sequence: 2,
        schema_id: 1,
        row_groups: vec![],
        row_counts: None,
        partition_values: None,
        next_row_id: 3,
        next_fragment_id: 1,
        rowindex_generation: None,
        index_generations: HashMap::new(),
        checkpoint: None,
        parent_hash: None, // legacy: no chain link
        committed_at: None,
        checksum: String::new(),
    };
    m2.checksum = m2.compute_checksum().unwrap();

    // Write m2 to _manifests/ and update the manifest pointer.
    let m2_path = format!("{}/{}", table, Manifest::filename(2));
    storage
        .write(&m2_path, &serde_json::to_vec(&m2).unwrap())
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/_manifest.json", table),
            &serde_json::to_vec(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();

    let report = verify_history(storage.as_ref(), &table).await.unwrap();
    assert!(
        report.intact,
        "None parent_hash on m2 with m1 present must be an anchor (not a break): {:?}",
        report
            .breaks
            .iter()
            .map(|b| format!("seq={}: {}", b.sequence, b.reason))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        report.breaks.len(),
        0,
        "no breaks expected for legacy None parent_hash"
    );
}
