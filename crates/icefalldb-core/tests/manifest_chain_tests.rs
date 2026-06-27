use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::metadata::{Column, Manifest, Schema};
use icefalldb_core::mutation_wal;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::Compactor;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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

async fn load_manifest(storage: &Arc<dyn Storage>, table: &str, seq: u64) -> Manifest {
    let path = format!("{}/{}", table, Manifest::filename(seq));
    let bytes = storage.read(&path).await.unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

/// Two sequential inserts must produce a chained, timestamped pair.
#[tokio::test]
async fn two_commits_are_hash_chained() {
    let (storage, table) = setup_table_with_two_inserts().await;
    let m1 = load_manifest(&storage, &table, 1).await;
    let m2 = load_manifest(&storage, &table, 2).await;
    assert!(m1.parent_hash.is_none(), "genesis has no parent");
    assert_eq!(
        m2.parent_hash.as_ref(),
        Some(&m1.checksum),
        "m2 links to m1"
    );
    assert!(
        m1.committed_at.is_some() && m2.committed_at.is_some(),
        "both manifests must have committed_at"
    );
    assert!(m1.verify_checksum().unwrap(), "m1 checksum must verify");
    assert!(m2.verify_checksum().unwrap(), "m2 checksum must verify");
}

fn wal_schema() -> Schema {
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
        // One fragment for the whole insert so every DELETE targets fragment 0.
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

/// A checkpoint that folds N>1 deferred WAL records publishes a single manifest
/// at the highest folded sequence, leaving the intermediate sequence(s) absent
/// on disk. The folded manifest's `parent_hash` must still link to the highest
/// *existing* predecessor (the base) — not `None` — or the chain breaks at every
/// WAL fold and a verifier would false-flag a chain break.
#[tokio::test]
async fn wal_fold_with_sequence_gap_links_to_highest_existing_predecessor() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let table = "wal_gap";

    // seq 1: a normal commit publishes a real manifest on disk.
    let mut w = Writer::create(Arc::clone(&storage), table, wal_schema())
        .await
        .unwrap();
    w.insert_batch(make_batch(vec![10, 11, 12, 13, 14]))
        .await
        .unwrap();
    w.commit().await.unwrap();

    // Two WAL-mode DELETEs defer to the log as records seq 2 and seq 3; neither
    // writes a manifest (the pointer stays at seq 1).
    let mut wal = Writer::new(Arc::clone(&storage), table, wal_schema())
        .await
        .unwrap()
        .with_wal_mode(true);
    let mut d1: HashMap<u64, Vec<u32>> = HashMap::new();
    d1.insert(0, vec![1]);
    wal.commit_deletes(d1).await.unwrap();

    let mut wal = Writer::new(Arc::clone(&storage), table, wal_schema())
        .await
        .unwrap()
        .with_wal_mode(true);
    let mut d2: HashMap<u64, Vec<u32>> = HashMap::new();
    d2.insert(0, vec![3]);
    wal.commit_deletes(d2).await.unwrap();

    let recs = mutation_wal::read_records(storage.as_ref(), table)
        .await
        .unwrap();
    assert_eq!(recs.len(), 2, "two deferred records (seq 2 and 3)");
    assert_eq!(recs[1].sequence, 3);

    // Fold both records into one manifest at seq 3 — seq 2 is never written.
    let did = mutation_wal::checkpoint_if_pending(storage.as_ref(), table, Duration::from_secs(5))
        .await
        .unwrap();
    assert!(did);

    // seq 2 must be absent on disk; seq 1 and seq 3 present (the gap).
    assert!(
        !storage
            .exists(&format!("{}/{}", table, Manifest::filename(2)))
            .await
            .unwrap(),
        "intermediate seq 2 manifest must NOT exist (folded, never written)"
    );

    let m1 = load_manifest(&storage, table, 1).await;
    let m3 = load_manifest(&storage, table, 3).await;

    // The crux: m3 links to the highest EXISTING predecessor (seq 1), not None.
    assert_eq!(
        m3.parent_hash.as_ref(),
        Some(&m1.checksum),
        "folded manifest must chain to the highest surviving predecessor (seq 1)"
    );
    assert!(m1.parent_hash.is_none(), "seq 1 is genesis");
    assert!(m3.committed_at.is_some(), "folded manifest is timestamped");
    assert!(m1.verify_checksum().unwrap());
    assert!(m3.verify_checksum().unwrap());
}

/// `compact`/`optimize` must publish manifests with chain fields populated, just
/// like normal commits. Regression test for M06.
#[tokio::test]
async fn compact_manifest_is_hash_chained_and_timestamped() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let table = "compact_chain";

    // seq 1: a normal insert publishes the genesis manifest.
    let mut w = Writer::create(Arc::clone(&storage), table, make_schema())
        .await
        .unwrap();
    w.insert_batch(make_batch(vec![1, 2, 3, 4, 5]))
        .await
        .unwrap();
    w.commit().await.unwrap();

    let m1 = load_manifest(&storage, table, 1).await;

    // seq 2: optimize rewrites the single row group and must finalize the new
    // manifest with parent_hash and committed_at.
    let compactor = Compactor::with_options(
        storage.as_ref(),
        table,
        icefalldb_core::CompactionOptions {
            force: true,
            ..icefalldb_core::CompactionOptions::default()
        },
    );
    let result = compactor.compact().await.unwrap();
    assert!(
        result.rewrote,
        "compactor should have rewritten the row group"
    );

    let m2 = load_manifest(&storage, table, 2).await;
    assert!(
        m2.committed_at.is_some(),
        "optimized manifest must have committed_at"
    );
    assert_eq!(
        m2.parent_hash.as_ref(),
        Some(&m1.checksum),
        "optimized manifest must link to its predecessor"
    );
    assert!(m2.verify_checksum().unwrap(), "m2 checksum must verify");

    // The whole retained chain must pass verification.
    let history = icefalldb_core::verify_history(storage.as_ref(), table)
        .await
        .unwrap();
    assert!(
        history.intact,
        "chain must be intact after optimize, got breaks: {:?}",
        history.breaks
    );
}
