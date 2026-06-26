use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::{build_scan_plan_at, list_snapshots, IcefallDBError, ScanPlan};

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
        // Large target so all rows go into one fragment.
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

/// Live row count: sum of (total rows - deleted rows) across all fragments.
fn total_rows(plan: &ScanPlan) -> u64 {
    plan.row_groups
        .iter()
        .map(|rg| (rg.meta.rows as u64).saturating_sub(rg.deleted_count))
        .sum()
}

/// Insert 3 rows (seq 1), then delete one row (seq 2).
async fn setup_table_then_delete_one_row() -> (Arc<dyn Storage>, String) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "tt_delete_test".to_string();

    // seq 1: insert 3 rows into fragment 0.
    let mut writer = Writer::create(Arc::clone(&storage), &table, make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // seq 2: delete physical offset 0 from fragment 0 (non-WAL mode → real manifest).
    let mut writer2 = Writer::new(Arc::clone(&storage), &table, make_schema())
        .await
        .unwrap();
    writer2
        .commit_deletes(HashMap::from([(0u64, vec![0u32])]))
        .await
        .unwrap();

    (storage, table)
}

/// Two inserts, each producing one committed manifest.
async fn setup_table_with_two_inserts() -> (Arc<dyn Storage>, String) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "tt_two_inserts".to_string();

    let mut writer = Writer::create(Arc::clone(&storage), &table, make_schema())
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

/// The scan plan built from an older snapshot (before a DELETE) must contain
/// the deleted row, so its live count exceeds the newer snapshot's count.
#[tokio::test]
async fn read_as_of_returns_pre_mutation_state() {
    let (storage, table) = setup_table_then_delete_one_row().await;
    let plan_old = build_scan_plan_at(storage.as_ref(), &table, 1)
        .await
        .unwrap();
    let plan_new = build_scan_plan_at(storage.as_ref(), &table, 2)
        .await
        .unwrap();
    let old_rows = total_rows(&plan_old);
    let new_rows = total_rows(&plan_new);
    assert!(
        old_rows > new_rows,
        "old snapshot should have the deleted row present: old={old_rows}, new={new_rows}"
    );
}

/// Requesting a sequence that does not exist must yield SnapshotNotFound.
#[tokio::test]
async fn unknown_snapshot_errors() {
    let (storage, table) = setup_table_with_two_inserts().await;
    let err = build_scan_plan_at(storage.as_ref(), &table, 999)
        .await
        .unwrap_err();
    assert!(
        matches!(err, IcefallDBError::SnapshotNotFound(999)),
        "expected SnapshotNotFound(999), got {err:?}"
    );
}

/// list_snapshots returns an ascending list with correct sequence numbers and
/// timestamps.
#[tokio::test]
async fn list_snapshots_reports_seq_and_time() {
    let (storage, table) = setup_table_with_two_inserts().await;
    let snaps = list_snapshots(storage.as_ref(), &table).await.unwrap();
    assert_eq!(snaps.len(), 2, "expected 2 snapshots, got {}", snaps.len());
    assert_eq!(snaps[0].sequence, 1);
    assert!(
        snaps[1].committed_at.is_some(),
        "second snapshot must have a committed_at timestamp"
    );
}
