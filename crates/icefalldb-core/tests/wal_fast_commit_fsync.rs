//! A WAL-mode DELETE collapses the synchronous commit to a single
//! durability barrier (deferred manifest), versus the multi-fsync sync path.
//!
//! fsyncs are counted via the process-global `LocalStorage` barrier counter. This
//! test lives in its **own** test binary (its own process) so the counter is not
//! perturbed by other tests running concurrently in a shared process.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::catalog::Catalog;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::{global_fsync_count, LocalStorage};
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;

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
        row_group_target_rows: 1000,
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

/// Build a fresh 5-row table and return its storage.
async fn fresh_table(dir: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir).unwrap());
    let mut w = Writer::create(Arc::clone(&storage), "t", schema())
        .await
        .unwrap();
    w.insert_batch(batch(vec![10, 11, 12, 13, 14]))
        .await
        .unwrap();
    w.commit().await.unwrap();
    storage
}

/// Measure the fsyncs issued by a single `commit_deletes(offset 1)` on `storage`.
async fn delete_fsyncs(storage: &Arc<dyn Storage>, wal_mode: bool) -> u64 {
    let mut w = Writer::new(Arc::clone(storage), "t", schema())
        .await
        .unwrap()
        .with_wal_mode(wal_mode);
    let mut by_fragment: HashMap<u64, Vec<u32>> = HashMap::new();
    by_fragment.insert(0, vec![1]);
    let before = global_fsync_count();
    w.commit_deletes(by_fragment).await.unwrap();
    global_fsync_count() - before
}

#[tokio::test(flavor = "current_thread")]
async fn wal_mode_delete_collapses_the_fsync_floor() {
    let sync_dir = tempfile::tempdir().unwrap();
    let wal_dir = tempfile::tempdir().unwrap();
    let sync_storage = fresh_table(sync_dir.path()).await;
    let wal_storage = fresh_table(wal_dir.path()).await;

    // Sync-mode DELETE: writes + syncs the DV, manifest, checkpoint, pointer.
    let sync_fsyncs = delete_fsyncs(&sync_storage, false).await;
    // WAL-mode DELETE: appends one record (the manifest swap is deferred).
    let wal_fsyncs = delete_fsyncs(&wal_storage, true).await;

    assert!(
        wal_fsyncs <= 1,
        "WAL-fast DELETE must issue at most one fsync, got {wal_fsyncs}"
    );
    assert!(
        wal_fsyncs < sync_fsyncs,
        "WAL DELETE ({wal_fsyncs}) must issue fewer fsyncs than sync ({sync_fsyncs})"
    );

    // Reads-after-write + crash recovery: the pointer is NOT swapped, yet a fresh
    // open replays the WAL so the live manifest reflects the delete (byte-equal
    // deletion state). (Full crash/replay coverage is in mutation_wal_e2e.rs.)
    let ptr: serde_json::Value =
        serde_json::from_slice(&wal_storage.read("t/_manifest.json").await.unwrap()).unwrap();
    assert_eq!(
        ptr["latest"].as_u64().unwrap(),
        1,
        "WAL DELETE defers the pointer swap"
    );
    let catalog = Catalog::load(wal_storage.as_ref(), "t").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    assert_eq!(manifest.sequence, 2, "replay surfaces the deferred delete");
    assert_eq!(manifest.row_groups[0].deleted_count, 1);
}
