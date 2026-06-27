//! End-to-end tests for deferred-commit (mutation WAL) DELETEs at the writer +
//! catalog level: a WAL-mode DELETE defers the manifest swap, a fresh open sees
//! it via replay (crash recovery), and a checkpoint folds it into a normal
//! manifest.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

// These WAL/crash-recovery tests do heavy fsync + manifest-replay I/O against a
// shared process; running them serially (rather than 5-wide) keeps them robust
// under load instead of intermittently failing when the host is contended.
use serial_test::serial;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::catalog::Catalog;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::{mutation_wal, DeletionVector};

fn schema() -> Schema {
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
        row_group_target_rows: 1000, // one fragment for the whole insert
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

fn batch(ids: Vec<i64>) -> RecordBatch {
    let s = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    RecordBatch::try_new(Arc::new(s), vec![Arc::new(Int64Array::from(ids))]).unwrap()
}

async fn pointer_seq(storage: &dyn Storage, table: &str) -> u64 {
    let data = storage
        .read(&format!("{table}/_manifest.json"))
        .await
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&data).unwrap();
    v.get("latest").unwrap().as_u64().unwrap()
}

/// Insert 5 rows into one fragment, then a WAL-mode DELETE of offset 1.
async fn setup_with_deferred_delete(storage: Arc<dyn Storage>, table: &str) {
    let mut w = Writer::create(Arc::clone(&storage), table, schema())
        .await
        .unwrap();
    w.insert_batch(batch(vec![10, 11, 12, 13, 14]))
        .await
        .unwrap();
    w.commit().await.unwrap(); // normal commit → seq 1, fragment 0

    let mut wal = Writer::new(Arc::clone(&storage), table, schema())
        .await
        .unwrap()
        .with_wal_mode(true);
    let mut by_fragment: HashMap<u64, Vec<u32>> = HashMap::new();
    by_fragment.insert(0, vec![1]); // delete physical offset 1 (id = 11)
    wal.commit_deletes(by_fragment).await.unwrap();
}

#[tokio::test]
#[serial]
async fn deferred_delete_does_not_swap_pointer_but_is_visible_via_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    setup_with_deferred_delete(Arc::clone(&storage), "t").await;

    // The pointer still references the pre-delete checkpoint (seq 1)...
    assert_eq!(pointer_seq(storage.as_ref(), "t").await, 1);
    // ...but exactly one WAL record is durably appended.
    let recs = mutation_wal::read_records(storage.as_ref(), "t")
        .await
        .unwrap();
    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].sequence, 2);

    // A FRESH open (simulating a process restart / crash recovery) sees the
    // delete: the catalog replays the WAL onto the checkpoint manifest.
    let catalog = Catalog::load(storage.as_ref(), "t").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    assert_eq!(
        manifest.sequence, 2,
        "live sequence reflects the deferred record"
    );
    assert_eq!(manifest.row_groups[0].deleted_count, 1);
    let del_path = manifest.row_groups[0].deletes.as_deref().unwrap();

    // The deletion vector really marks offset 1.
    let dv_bytes = storage.read(&format!("t/{del_path}")).await.unwrap();
    let dv = DeletionVector::deserialize(&dv_bytes).unwrap();
    assert!(dv.contains(1));
    assert!(!dv.contains(0));
}

#[tokio::test]
#[serial]
async fn checkpoint_folds_deferred_delete_into_a_normal_manifest() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    setup_with_deferred_delete(Arc::clone(&storage), "t").await;

    let did = mutation_wal::checkpoint_if_pending(storage.as_ref(), "t", Duration::from_secs(5))
        .await
        .unwrap();
    assert!(did);

    // Pointer advanced to the live sequence; WAL is gone.
    assert_eq!(pointer_seq(storage.as_ref(), "t").await, 2);
    assert!(mutation_wal::read_records(storage.as_ref(), "t")
        .await
        .unwrap()
        .is_empty());

    // After checkpoint, a plain catalog open (no replay needed) still sees the
    // delete — it is now a normal, checksum-valid manifest.
    let catalog = Catalog::load(storage.as_ref(), "t").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    assert_eq!(manifest.sequence, 2);
    assert_eq!(manifest.row_groups[0].deleted_count, 1);
}

#[tokio::test]
#[serial]
async fn gc_with_pending_wal_checkpoints_first_and_keeps_the_deletion_vector() {
    // The data-loss guard: GC computes its referenced set from the pointer
    // manifest. With a pending WAL it must checkpoint first, otherwise it would
    // collect the deletion-vector file the WAL still references.
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    setup_with_deferred_delete(Arc::clone(&storage), "t").await;
    assert_eq!(pointer_seq(storage.as_ref(), "t").await, 1); // not yet checkpointed

    icefalldb_core::gc::GarbageCollector::new(storage.as_ref(), "t", 1)
        .run()
        .await
        .unwrap();

    // GC checkpointed the WAL (pointer advanced, log cleared) and preserved the
    // deferred delete: the .del file is still referenced and readable.
    assert_eq!(pointer_seq(storage.as_ref(), "t").await, 2);
    assert!(mutation_wal::read_records(storage.as_ref(), "t")
        .await
        .unwrap()
        .is_empty());
    let catalog = Catalog::load(storage.as_ref(), "t").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    assert_eq!(manifest.row_groups[0].deleted_count, 1);
    let del_path = manifest.row_groups[0].deletes.as_deref().unwrap();
    let dv = DeletionVector::deserialize(&storage.read(&format!("t/{del_path}")).await.unwrap())
        .unwrap();
    assert!(dv.contains(1));
}

#[tokio::test]
#[serial]
async fn crash_losing_the_unsynced_del_file_is_recovered_from_the_inlined_record() {
    // In WAL mode the .del file is written without its own fsync; durability is
    // the inlined record. Simulate a crash that lost the un-fsynced .del, then a
    // fresh open: the catalog's replay reconstructs it from the record.
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    setup_with_deferred_delete(Arc::clone(&storage), "t").await;

    // The record carries the inlined deletion-vector bytes.
    let recs = mutation_wal::read_records(storage.as_ref(), "t")
        .await
        .unwrap();
    assert!(!recs[0].staged_artifacts.is_empty(), "DV bytes inlined");
    let del_rel = recs[0].fragment_deletes[0].deletes.clone();

    // Delete the .del file from disk (the un-fsynced copy a crash would lose).
    storage.delete(&format!("t/{del_rel}")).await.unwrap();
    assert!(!storage.exists(&format!("t/{del_rel}")).await.unwrap());

    // A fresh open replays the WAL, which reconstructs the missing .del.
    let catalog = Catalog::load(storage.as_ref(), "t").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    assert_eq!(manifest.row_groups[0].deleted_count, 1);
    let dv =
        DeletionVector::deserialize(&storage.read(&format!("t/{del_rel}")).await.unwrap()).unwrap();
    assert!(dv.contains(1));
}

#[tokio::test]
#[serial]
async fn second_deferred_delete_builds_on_the_first() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    setup_with_deferred_delete(Arc::clone(&storage), "t").await; // deletes offset 1 → seq 2

    // A second WAL-mode delete of offset 3 must see the first delete (replay) and
    // produce a union deletion vector at the next sequence.
    let mut wal = Writer::new(Arc::clone(&storage), "t", schema())
        .await
        .unwrap()
        .with_wal_mode(true);
    let mut by_fragment: HashMap<u64, Vec<u32>> = HashMap::new();
    by_fragment.insert(0, vec![3]);
    wal.commit_deletes(by_fragment).await.unwrap();

    let recs = mutation_wal::read_records(storage.as_ref(), "t")
        .await
        .unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[1].sequence, 3);

    let catalog = Catalog::load(storage.as_ref(), "t").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    assert_eq!(manifest.sequence, 3);
    assert_eq!(
        manifest.row_groups[0].deleted_count, 2,
        "both deletes present"
    );
    let del_path = manifest.row_groups[0].deletes.as_deref().unwrap();
    let dv = DeletionVector::deserialize(&storage.read(&format!("t/{del_path}")).await.unwrap())
        .unwrap();
    assert!(dv.contains(1) && dv.contains(3));
}
