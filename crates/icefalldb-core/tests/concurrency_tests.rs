use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::writer::Writer;
use icefalldb_core::Result;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn make_int_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn make_int_schema(row_group_target_rows: usize) -> icefalldb_core::metadata::Schema {
    use icefalldb_core::metadata::{Column, Schema};
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
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

/// `Storage` wrapper that pauses the first commit's `_manifest.json` pointer
/// swap so a test can observe whether the exclusive writer lock is still held
/// at manifest-publication time.
#[derive(Debug)]
struct ManifestPublishBlockingStorage {
    inner: MemoryStorage,
    publish_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    publish_rx: Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
    manifest_rename_count: AtomicUsize,
}

impl ManifestPublishBlockingStorage {
    fn new() -> (
        Self,
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let (publish_tx, publish_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let storage = Self {
            inner: MemoryStorage::new(),
            publish_tx: Mutex::new(Some(publish_tx)),
            publish_rx: Mutex::new(Some(release_rx)),
            manifest_rename_count: AtomicUsize::new(0),
        };
        (storage, publish_rx, release_tx)
    }
}

#[async_trait]
impl Storage for ManifestPublishBlockingStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
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

    async fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        self.inner.append(path, data).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix).await
    }

    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        // The first rename of `_manifest.json` is table initialization
        // (`{"latest": 0}`). The second rename is the first commit's pointer
        // swap; pause there so the test can inspect lock ownership.
        if to.ends_with("_manifest.json") {
            let count = self.manifest_rename_count.fetch_add(1, Ordering::SeqCst);
            if count == 1 {
                let tx = self.publish_tx.lock().unwrap().take();
                if let Some(tx) = tx {
                    let _ = tx.send(());
                }
                let rx = self.publish_rx.lock().unwrap().take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
            }
        }
        self.inner.rename(from, to).await
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>> {
        self.inner.lock_exclusive(path, timeout).await
    }

    async fn sync(&self, path: &str) -> Result<()> {
        self.inner.sync(path).await
    }

    async fn sync_data(&self, path: &str) -> Result<()> {
        self.inner.sync_data(path).await
    }
}

/// Verifies that the exclusive writer lock remains held until the manifest
/// pointer swap completes.
///
/// A first writer is paused just before publishing `_manifest.json`. While it
/// is paused, a second task attempts to acquire the same writer lock with a
/// short timeout. If the lock were released early (the bug), the second task
/// would acquire it immediately; with the fix it must time out.
#[tokio::test]
async fn test_writer_lock_held_through_manifest_publication() {
    let (storage, publish_rx, release_tx) = ManifestPublishBlockingStorage::new();
    let storage: Arc<dyn Storage> = Arc::new(storage);
    let table = "products";
    let schema = make_int_schema(10);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch(vec![1, 2, 3]))
        .await
        .unwrap();

    let commit = tokio::spawn(async move { writer.commit().await });

    // Wait until the first commit is about to swap the manifest pointer.
    publish_rx.await.unwrap();

    // The lock must still be held: a concurrent lock attempt times out.
    let storage2 = Arc::clone(&storage);
    let lock_attempt = tokio::spawn(async move {
        storage2
            .lock_exclusive(&format!("{}/_write.lock", table), Duration::from_millis(50))
            .await
    });
    let lock_result = tokio::time::timeout(Duration::from_secs(2), lock_attempt)
        .await
        .expect("lock attempt task should finish quickly")
        .unwrap();
    assert!(
        lock_result.is_err(),
        "second writer should not acquire the lock while the first is mid-commit"
    );

    // Allow the first commit to finish and verify success.
    let _ = release_tx.send(());
    let commit_result = tokio::time::timeout(Duration::from_secs(5), commit)
        .await
        .expect("commit should complete after release")
        .unwrap();
    assert!(commit_result.is_ok(), "commit failed: {:?}", commit_result);

    // The table must reflect the committed rows.
    let pointer: serde_json::Value = serde_json::from_slice(
        &storage
            .read(&format!("{}/_manifest.json", table))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(1));
}
