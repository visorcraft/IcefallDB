//! A steady-state append commit must keep its per-directory fsyncs
//! batched to one-per-directory. A per-instance counting Storage wrapper makes
//! the assertion parallel-safe (the process-global `FSYNC_COUNT` is polluted by
//! concurrent tests).

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use icefalldb_core::database_catalog::DatabaseCatalog;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::writer::Writer;
use icefalldb_core::{MatchLoc, Result};
use std::any::Any;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Counts `sync`/`sync_data` calls and records their paths, delegating
/// everything else (including `local_root`, so the writer keeps its local path).
struct SyncCounter {
    inner: Arc<dyn Storage>,
    dir_syncs: AtomicUsize,
    data_syncs: AtomicUsize,
    paths: Mutex<Vec<String>>,
}

impl SyncCounter {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            dir_syncs: AtomicUsize::new(0),
            data_syncs: AtomicUsize::new(0),
            paths: Mutex::new(Vec::new()),
        }
    }
    fn reset(&self) {
        self.dir_syncs.store(0, Ordering::Relaxed);
        self.data_syncs.store(0, Ordering::Relaxed);
        self.paths.lock().unwrap().clear();
    }
}

#[async_trait]
impl Storage for SyncCounter {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn local_root(&self) -> Option<&Path> {
        self.inner.local_root()
    }
    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        self.inner.read(path).await
    }
    async fn size(&self, path: &str) -> Result<u64> {
        self.inner.size(path).await
    }
    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.inner.read_range(path, offset, len).await
    }
    async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        self.inner.write(path, data).await
    }
    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }
    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        self.inner.rename(from, to).await
    }
    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix).await
    }
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }
    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>> {
        self.inner.lock_exclusive(path, timeout).await
    }
    async fn sync(&self, path: &str) -> Result<()> {
        self.dir_syncs.fetch_add(1, Ordering::Relaxed);
        self.paths.lock().unwrap().push(format!("sync {path}"));
        self.inner.sync(path).await
    }
    async fn sync_data(&self, path: &str) -> Result<()> {
        self.data_syncs.fetch_add(1, Ordering::Relaxed);
        self.paths.lock().unwrap().push(format!("sync_data {path}"));
        self.inner.sync_data(path).await
    }
    async fn sync_root(&self) -> Result<()> {
        self.dir_syncs.fetch_add(1, Ordering::Relaxed);
        self.inner.sync_root().await
    }
}

fn schema() -> Schema {
    Schema {
        schema_id: 1,
        columns: vec![Column::new("id", "int64", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1 << 20,
        dropped_columns: vec![],
        max_field_id: 1,
    }
}

fn batch(ids: Vec<i64>) -> RecordBatch {
    let s = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(s, vec![Arc::new(Int64Array::from(ids))]).unwrap()
}

#[tokio::test]
async fn append_commit_dir_fsyncs_are_batched_per_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let local: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let storage = Arc::new(SyncCounter::new(local));

    // Commit 1 creates the table (schema setup); not what we measure.
    let mut w = Writer::create(Arc::clone(&storage) as Arc<dyn Storage>, "t", schema())
        .await
        .unwrap();
    w.insert_batch(batch(vec![1, 2, 3])).await.unwrap();
    w.commit().await.unwrap();

    // Measure a steady-state append commit (the full intent→data→checkpoint→
    // manifest→pointer ceremony) in isolation.
    storage.reset();
    let mut w = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "t", schema())
        .await
        .unwrap();
    w.insert_batch(batch(vec![4, 5, 6])).await.unwrap();
    w.commit().await.unwrap();

    let dir = storage.dir_syncs.load(Ordering::Relaxed);
    let data = storage.data_syncs.load(Ordering::Relaxed);
    let paths = storage.paths.lock().unwrap().clone();
    eprintln!(
        "append-commit dir_syncs={dir} data_syncs={data}\n{}",
        paths.join("\n")
    );

    // After batching the per-directory fsyncs are coalesced: intent rewrites
    // (filename-update, checkpoint-update) no longer re-fsync `_staging/intents`
    // (in-place, durable entry), and the redundant pre-rename `_manifests` fsync
    // is gone. The directory-fsync budget drops from the 9-fsync baseline to <= 6.
    let dir_only: Vec<&str> = paths
        .iter()
        .filter(|p| p.starts_with("sync "))
        .map(|p| p.as_str())
        .collect();
    let count_of = |dir: &str| dir_only.iter().filter(|p| p.contains(dir)).count();

    // `_manifests` is now fsync'd exactly once (post-rename), like every other
    // commit path. `_staging/intents` is fsync'd at most twice: once when the
    // intent entry is created, once when it is deleted at cleanup — the
    // intermediate rewrites no longer fsync the directory.
    assert!(
        count_of("_manifests") <= 1,
        "_manifests must be fsync'd once, got {}: {dir_only:?}",
        count_of("_manifests")
    );
    assert!(
        count_of("_staging/intents") <= 2,
        "_staging/intents must be fsync'd at most twice (create + delete), got {}: {dir_only:?}",
        count_of("_staging/intents")
    );
    assert!(
        dir <= 6,
        "append commit must issue <= 6 directory fsyncs after batching (baseline 9), got {dir}: {dir_only:?}"
    );
}

fn id_cat(ids: Vec<i64>, cats: Vec<&str>) -> RecordBatch {
    let s = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(cats)),
        ],
    )
    .unwrap()
}

/// An UPDATE of an INDEXED column rewrites the intent to list the new index
/// delta files; that in-place rewrite must NOT re-fsync `_staging/intents`
/// (the directory entry is already durable from the initial intent fsync).
/// Before the batching reached this path the directory was fsync'd three
/// times (create + index-delta rewrite + cleanup); now at most twice.
#[tokio::test]
async fn indexed_update_intent_rewrite_does_not_refsync() {
    let tmp = tempfile::tempdir().unwrap();
    let local: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let storage = Arc::new(SyncCounter::new(local));

    let mut sch = Schema {
        schema_id: 1,
        columns: vec![
            Column::new("id", "int64", false),
            Column::new("cat", "utf8", true),
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1 << 20,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    sch.assign_field_ids(None);

    // Register the table + a btree index on `cat`.
    let dbcat = DatabaseCatalog::new(Arc::clone(&storage) as Arc<dyn Storage>);
    let guard = dbcat.acquire_lock(Duration::from_secs(10)).await.unwrap();
    dbcat.create_table(&guard, "t", &sch).await.unwrap();
    dbcat
        .create_index_definition(&guard, "cat_idx", "t", "cat", "btree")
        .await
        .unwrap();
    drop(guard);

    // Two fragments: ids [1,2] then [3,4] → row_ids [0,1] then [2,3].
    let mut w = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "t", sch.clone())
        .await
        .unwrap();
    w.insert_batch(id_cat(vec![1, 2], vec!["a", "b"]))
        .await
        .unwrap();
    w.commit().await.unwrap();
    w.insert_batch(id_cat(vec![3, 4], vec!["a", "b"]))
        .await
        .unwrap();
    w.commit().await.unwrap();

    let frag1 = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "t")
        .await
        .unwrap()
        .latest_manifest()
        .unwrap()
        .row_groups[1]
        .fragment_id;

    // Measure an indexed-column UPDATE (id=3 → cat "z"): produces an index delta,
    // which fires the intent rewrite that previously re-fsync'd `_staging/intents`.
    storage.reset();
    let mut w3 = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "t", sch)
        .await
        .unwrap();
    w3.commit_update(
        id_cat(vec![3], vec!["z"]),
        vec![MatchLoc {
            fragment_id: frag1,
            offset: 0,
            row_id: 2,
        }],
        &["cat".to_string()],
    )
    .await
    .unwrap();

    let paths = storage.paths.lock().unwrap().clone();
    let intent_dir_fsyncs = paths
        .iter()
        .filter(|p| p.starts_with("sync ") && p.contains("_staging/intents"))
        .count();
    assert!(
        intent_dir_fsyncs <= 1,
        "indexed UPDATE must not re-fsync _staging/intents on the index-delta rewrite \
         (expect 1: the initial create only), got {intent_dir_fsyncs}:\n{}",
        paths.join("\n")
    );
}
