use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use icefalldb_core::metadata::{Column, Manifest, RowGroupEntry, RowGroupMeta, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::writer::{InsertParquetOutcome, Writer};
use icefalldb_core::Result;
use icefalldb_core::{
    build_btree_index, load_index_by_ref, segment_ids, DatabaseCatalog, IndexDefinition,
};
use parquet::arrow::ArrowWriter;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn make_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn make_schema(row_group_target_rows: usize) -> Schema {
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

fn make_partitioned_batch(ids: Vec<i64>, region: Vec<&str>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
    ]);
    let ids = Int64Array::from(ids);
    let region = StringArray::from(region);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids), Arc::new(region)]).unwrap()
}

fn make_partitioned_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "region".into(),
                r#type: "utf8".into(),
                nullable: false,
                field_id: 0,
            },
        ],
        partition_by: Some(vec!["region".into()]),
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

async fn read_latest_sequence(storage: &Arc<dyn Storage>, table: &str) -> u64 {
    let pointer_data = storage
        .read(&format!("{}/_manifest.json", table))
        .await
        .unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    pointer["latest"].as_u64().unwrap()
}

async fn read_latest_manifest(storage: &Arc<dyn Storage>, table: &str) -> Manifest {
    let seq = read_latest_sequence(storage, table).await;
    let manifest_data = storage
        .read(&format!("{}/{}", table, Manifest::filename(seq)))
        .await
        .unwrap();
    serde_json::from_slice(&manifest_data).unwrap()
}

#[tokio::test]
async fn test_writer_inserts_rows() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    assert_eq!(read_latest_sequence(&storage, "products").await, 1);
    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 1);
    assert!(manifest.verify_checksum().unwrap());
}

#[tokio::test]
async fn test_writer_commit_twice_advances_only_with_new_data() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    assert_eq!(read_latest_sequence(&storage, "products").await, 1);

    // Empty commit should be a no-op.
    writer.commit().await.unwrap();
    assert_eq!(read_latest_sequence(&storage, "products").await, 1);

    // New data should advance the sequence.
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    assert_eq!(read_latest_sequence(&storage, "products").await, 2);
}

#[tokio::test]
async fn test_writer_empty_commit_does_not_advance_sequence() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    // First commit on an empty table must not advance the sequence.
    writer.commit().await.unwrap();
    let pointer: serde_json::Value =
        serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap()).unwrap();
    assert_eq!(pointer["latest"].as_u64(), Some(0));

    // Commit some data.
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    assert_eq!(read_latest_sequence(&storage, "products").await, 1);

    // Empty commit after a real commit must not advance the sequence.
    writer.commit().await.unwrap();
    assert_eq!(read_latest_sequence(&storage, "products").await, 1);
}

#[tokio::test]
async fn test_writer_multiple_flushes() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(2);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer.insert_batch(make_batch(vec![1, 2])).await.unwrap();
    writer.insert_batch(make_batch(vec![3, 4])).await.unwrap();
    writer.insert_batch(make_batch(vec![5, 6])).await.unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 3);

    let entries = storage.list("products/").await.unwrap();
    let parquet_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".parquet")).collect();
    assert_eq!(parquet_files.len(), 3);
}

#[tokio::test]
async fn test_row_group_meta_checksum_matches_parquet() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    for entry in &manifest.row_groups {
        let parquet_data = storage
            .read(&format!("products/{}", entry.data))
            .await
            .unwrap();
        let meta_data = storage
            .read(&format!("products/{}", entry.meta))
            .await
            .unwrap();
        let meta: RowGroupMeta = serde_json::from_slice(&meta_data).unwrap();
        assert!(
            meta.verify_against_data(&parquet_data),
            "checksum for {} should match its parquet bytes",
            entry.data
        );
        assert!(
            meta.verify_meta_checksum().unwrap(),
            "meta checksum for {} should be valid",
            entry.meta
        );
    }
}

#[tokio::test]
async fn test_local_storage_writer_commits_files() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let schema = make_schema(2);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer.insert_batch(make_batch(vec![1, 2])).await.unwrap();
    writer
        .insert_batch(make_batch(vec![3, 4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Verify the manifest pointer exists.
    assert!(storage.exists("products/_manifest.json").await.unwrap());

    // Verify manifest and row group files exist on disk.
    let entries = storage.list("products/").await.unwrap();
    let parquet_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".parquet")).collect();
    let meta_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".meta")).collect();
    assert_eq!(parquet_files.len(), 3, "expected 3 row groups");
    assert_eq!(meta_files.len(), 3, "expected 3 meta files");

    // Verify the latest manifest is readable and consistent.
    let pointer_data = storage.read("products/_manifest.json").await.unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    let seq = pointer["latest"].as_u64().unwrap();
    assert_eq!(seq, 1);

    let manifest_data = storage
        .read(&format!("products/{}", Manifest::filename(seq)))
        .await
        .unwrap();
    let manifest: Manifest = serde_json::from_slice(&manifest_data).unwrap();
    assert!(manifest.verify_checksum().unwrap());
    assert_eq!(manifest.row_groups.len(), 3);
}

/// `Storage` wrapper around `LocalStorage` that signals once the first commit
/// has acquired the cross-process `flock()`, then waits on a barrier before
/// returning the guard. Used to verify that the lock excludes other processes.
#[derive(Debug)]
struct SignalingLocalStorage {
    inner: LocalStorage,
    first_lock_signal: Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    first_lock_barrier: Arc<Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
    lock_count: AtomicUsize,
}

impl SignalingLocalStorage {
    fn new(
        inner: LocalStorage,
    ) -> (
        Self,
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
    ) {
        let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let storage = Self {
            inner,
            first_lock_signal: Arc::new(Mutex::new(Some(locked_tx))),
            first_lock_barrier: Arc::new(Mutex::new(Some(release_rx))),
            lock_count: AtomicUsize::new(0),
        };
        (storage, locked_rx, release_tx)
    }
}

#[async_trait]
impl Storage for SignalingLocalStorage {
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
        let count = self.lock_count.fetch_add(1, Ordering::SeqCst) + 1;
        let guard = self.inner.lock_exclusive(path, timeout).await?;
        if count == 2 {
            let signal = self.first_lock_signal.lock().unwrap().take();
            let barrier = self.first_lock_barrier.lock().unwrap().take();
            if let Some(tx) = signal {
                let _ = tx.send(());
                if let Some(rx) = barrier {
                    let _ = rx.await;
                }
            }
        }
        Ok(guard)
    }

    async fn sync(&self, path: &str) -> Result<()> {
        self.inner.sync(path).await
    }
}

#[tokio::test]
async fn test_local_storage_writer_lock_excludes_cross_process_flock() {
    let tmp = tempfile::tempdir().unwrap();
    let inner = LocalStorage::new(tmp.path()).unwrap();
    let (storage, locked_rx, release_tx) = SignalingLocalStorage::new(inner);
    let storage = Arc::new(storage);
    let table = "products";
    let schema = make_schema(10);

    let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, table, schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    let commit = tokio::spawn(async move { writer.commit().await });

    // Wait until the commit has acquired the cross-process lock.
    locked_rx.await.unwrap();

    let lock_file = tmp.path().join(format!("{}/_write.lock", table));

    // While the writer holds the lock, another process must not be able to
    // acquire it with flock(2).
    let lock_file_clone = lock_file.clone();
    let blocked = tokio::task::spawn_blocking(move || {
        std::process::Command::new("flock")
            .arg("-n")
            .arg(&lock_file_clone)
            .arg("-c")
            .arg("echo locked")
            .output()
    })
    .await
    .unwrap()
    .unwrap();
    assert!(
        !blocked.status.success(),
        "flock should fail to acquire the writer lock while it is held"
    );

    // Release the writer and let the commit finish.
    let _ = release_tx.send(());
    commit.await.unwrap().unwrap();

    // Once the writer has released the lock, flock should succeed.
    let acquired = tokio::task::spawn_blocking(move || {
        std::process::Command::new("flock")
            .arg("-n")
            .arg(&lock_file)
            .arg("-c")
            .arg("echo locked")
            .output()
    })
    .await
    .unwrap()
    .unwrap();
    assert!(
        acquired.status.success(),
        "flock should acquire the lock after the writer releases it"
    );
}

#[tokio::test]
async fn test_row_group_meta_row_group_is_id_without_extension() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    let entry = &manifest.row_groups[0];
    let meta: RowGroupMeta = serde_json::from_slice(
        &storage
            .read(&format!("products/{}", entry.meta))
            .await
            .unwrap(),
    )
    .unwrap();

    assert!(
        !meta.row_group.ends_with(".parquet"),
        "row_group field must be the id, not the data filename: {}",
        meta.row_group
    );
    assert!(
        !meta.row_group.ends_with(".meta"),
        "row_group field must be the id, not the meta filename: {}",
        meta.row_group
    );
    assert!(meta.row_group.starts_with("rg_"));
}

#[tokio::test]
async fn test_manifest_partition_values_populated() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_partitioned_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_partitioned_batch(
            vec![1, 2, 3],
            vec!["us-east", "us-east", "us-east"],
        ))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    assert!(
        manifest.partition_values.is_some(),
        "partition_values should be populated when partition_by is declared"
    );
    let partition_values = manifest.partition_values.as_ref().unwrap();
    assert_eq!(
        partition_values.len(),
        1,
        "expected one row group with partition values"
    );

    let entry = &manifest.row_groups[0];
    let rg_partitions = partition_values
        .get(&entry.data)
        .expect("partition values keyed by row-group data filename");
    assert_eq!(
        rg_partitions.get("region"),
        Some(&serde_json::json!("us-east"))
    );
}

#[tokio::test]
async fn test_manifest_partition_values_split_for_mixed_values() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_partitioned_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_partitioned_batch(
            vec![1, 2, 3],
            vec!["us-east", "us-west", "us-east"],
        ))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    assert!(
        manifest.partition_values.is_some(),
        "partition_values should be populated after splitting by partition value"
    );
    let partition_values = manifest.partition_values.as_ref().unwrap();
    assert_eq!(
        partition_values.len(),
        2,
        "expected two homogeneous row groups with partition values"
    );

    let mut seen_regions = Vec::new();
    for entry in &manifest.row_groups {
        let rg_partitions = partition_values
            .get(&entry.data)
            .expect("partition values keyed by row-group data filename");
        seen_regions.push(
            rg_partitions
                .get("region")
                .expect("region partition value")
                .as_str()
                .unwrap()
                .to_string(),
        );
    }
    seen_regions.sort_unstable();
    assert_eq!(seen_regions, vec!["us-east", "us-west"]);
}

#[tokio::test]
async fn test_recovery_preserves_referenced_final_files_in_intent() {
    // Simulate the critical crash window: the manifest pointer has been
    // updated to sequence 2, and the intent for that commit (which records
    // the final table-root filenames) was not deleted before the writer crashed.
    // Recovery must not delete files that are referenced by the current manifest.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_schema(10);

    // Commit sequence 1 normally.
    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Simulate sequence 2 having committed successfully but its intent
    // surviving the crash.
    let committed_data = "rg_committed.parquet";
    let committed_meta = "rg_committed.meta";
    storage
        .write(&format!("{}/{}", table, committed_data), b"committed-data")
        .await
        .unwrap();
    // The checkpoint builder reads existing .meta sidecars, so the fake meta
    // file must be valid JSON even though its contents are otherwise ignored
    // by this recovery test.
    let fake_meta = RowGroupMeta {
        row_group: "rg_committed".into(),
        schema_id: schema.schema_id,
        rows: 0,
        columns: HashMap::new(),
        column_offsets: None,
        sort: None,
        row_ids: vec![],
        checksum: "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
        meta_checksum: String::new(),
    };
    storage
        .write(
            &format!("{}/{}", table, committed_meta),
            serde_json::to_vec(&fake_meta).unwrap().as_slice(),
        )
        .await
        .unwrap();

    let mut manifest_seq2 = Manifest {
        format_version: 1,
        sequence: 2,
        schema_id: schema.schema_id,
        row_groups: vec![RowGroupEntry {
            data: committed_data.into(),
            meta: committed_meta.into(),
            ..Default::default()
        }],
        partition_values: Some(HashMap::new()),
        checksum: String::new(),
        ..Default::default()
    };
    manifest_seq2.checksum = manifest_seq2.compute_checksum().unwrap();
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(2)),
            serde_json::to_vec_pretty(&manifest_seq2)
                .unwrap()
                .as_slice(),
        )
        .await
        .unwrap();
    storage
        .write(&format!("{}/_manifest.json", table), b"{\"latest\": 2}")
        .await
        .unwrap();

    let intent = serde_json::json!({
        "txn_id": "txn_committed",
        "started_at": chrono::Utc::now().to_rfc3339(),
        "schema_id": 1,
        "files": [committed_data, committed_meta],
    });
    storage
        .write(
            &format!("{}/_staging/intents/txn_committed.json", table),
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();

    // A new commit must recover the stale intent and succeed.
    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // The stale intent must be gone.
    assert!(!storage
        .exists(&format!("{}/_staging/intents/txn_committed.json", table))
        .await
        .unwrap());

    // The final files from the (simulated) committed sequence 2 must still
    // exist and be referenced by the latest manifest.
    assert!(storage
        .exists(&format!("{}/{}", table, committed_data))
        .await
        .unwrap());
    assert!(storage
        .exists(&format!("{}/{}", table, committed_meta))
        .await
        .unwrap());

    let pointer: serde_json::Value = serde_json::from_slice(
        &storage
            .read(&format!("{}/_manifest.json", table))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(3));

    let manifest: Manifest = serde_json::from_slice(
        &storage
            .read(&format!("{}/{}", table, Manifest::filename(3)))
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(manifest.verify_checksum().unwrap());
    assert!(manifest
        .row_groups
        .iter()
        .any(|e| e.data == committed_data && e.meta == committed_meta));
}

/// `Storage` wrapper that returns corrupted bytes for the first read of any
/// staged `.meta.part` file, then returns the correct bytes. This exercises the
/// row-group meta-checksum retry path before the files are renamed to their
/// final locations.
#[derive(Debug)]
struct CorruptFirstMetaReadStorage {
    inner: MemoryStorage,
    first_meta_read: std::sync::atomic::AtomicBool,
}

impl CorruptFirstMetaReadStorage {
    fn new() -> Self {
        Self {
            inner: MemoryStorage::new(),
            first_meta_read: std::sync::atomic::AtomicBool::new(true),
        }
    }
}

#[async_trait]
impl Storage for CorruptFirstMetaReadStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        let data = self.inner.read(path).await?;
        if path.ends_with(".meta.part")
            && self.first_meta_read.swap(false, Ordering::SeqCst)
            && !data.is_empty()
        {
            // Corrupt the metadata in a way that still parses but invalidates
            // the meta checksum, so the retry path is exercised.
            let mut value: serde_json::Value = serde_json::from_slice(&data)?;
            if let Some(rows) = value.get_mut("rows").and_then(|v| v.as_u64()) {
                value["rows"] = (rows + 1).into();
            }
            return Ok(serde_json::to_vec_pretty(&value)?);
        }
        Ok(data)
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
        self.inner.sync(path).await
    }
}

#[tokio::test]
async fn test_row_group_meta_checksum_retry_on_corrupt_read() {
    let storage = Arc::new(CorruptFirstMetaReadStorage::new());
    let schema = make_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // The commit should have retried and produced valid files.
    let manifest: Manifest = serde_json::from_slice(
        &storage
            .inner
            .read(&format!("products/{}", Manifest::filename(1)))
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(manifest.verify_checksum().unwrap());
    assert_eq!(manifest.sequence, 1);

    let entry = &manifest.row_groups[0];
    let parquet_bytes = storage
        .inner
        .read(&format!("products/{}", entry.data))
        .await
        .unwrap();
    let meta: RowGroupMeta = serde_json::from_slice(
        &storage
            .inner
            .read(&format!("products/{}", entry.meta))
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(meta.verify_against_data(&parquet_bytes));
    assert!(meta.verify_meta_checksum().unwrap());
}

#[tokio::test]
async fn test_concurrent_writer_new_races_create_once() {
    // Two in-process Writer::new calls racing to create the same non-existent
    // table must serialize through the exclusive writer lock: exactly one
    // creates and the other opens, leaving a valid empty table.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let table = "products";

    let (a, b) = tokio::join!(
        Writer::new(Arc::clone(&storage), table, schema.clone()),
        Writer::new(Arc::clone(&storage), table, schema.clone()),
    );

    assert!(
        a.is_ok(),
        "first writer failed: {:?}",
        a.as_ref().map(|_| ())
    );
    assert!(
        b.is_ok(),
        "second writer failed: {:?}",
        b.as_ref().map(|_| ())
    );

    // Exactly one writer should have created the table; the table must be valid.
    let pointer: serde_json::Value =
        serde_json::from_slice(&storage.read("products/_schema.json").await.unwrap()).unwrap();
    assert_eq!(pointer["latest"].as_u64(), Some(1));
}

#[tokio::test]
async fn test_concurrent_writer_create_and_new_race() {
    // Concurrent Writer::create (fail-if-exists) and Writer::new
    // (open-or-create) on the same non-existent table must serialize so that
    // one succeeds as the creator and the other opens the created table.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema(10);
    let table = "products";

    let (create_result, new_result) = tokio::join!(
        Writer::create(Arc::clone(&storage), table, schema.clone()),
        Writer::new(Arc::clone(&storage), table, schema.clone()),
    );

    assert!(
        create_result.is_ok() && new_result.is_ok(),
        "both writers must succeed: create={:?}, new={:?}",
        create_result.as_ref().map(|_| ()),
        new_result.as_ref().map(|_| ())
    );

    // The table must exist and be valid regardless of which call created it.
    let pointer: serde_json::Value =
        serde_json::from_slice(&storage.read("products/_schema.json").await.unwrap()).unwrap();
    assert_eq!(pointer["latest"].as_u64(), Some(1));
}

/// A [`Storage`] wrapper that delegates to an inner storage while counting
/// calls to `lock_exclusive`.
struct LockCountingStorage {
    inner: Arc<dyn Storage>,
    lock_count: AtomicUsize,
}

impl LockCountingStorage {
    fn new(inner: Arc<dyn Storage>) -> Self {
        Self {
            inner,
            lock_count: AtomicUsize::new(0),
        }
    }

    fn lock_count(&self) -> usize {
        self.lock_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl Storage for LockCountingStorage {
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

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix).await
    }

    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        self.inner.rename(from, to).await
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }

    async fn sync(&self, path: &str) -> Result<()> {
        self.inner.sync(path).await
    }

    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>> {
        self.lock_count.fetch_add(1, Ordering::SeqCst);
        self.inner.lock_exclusive(path, timeout).await
    }
}

#[tokio::test]
async fn test_writer_assume_lock_held_skips_lock_acquisition() {
    let tmp = tempfile::tempdir().unwrap();
    let local = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let counting_inner = Arc::new(LockCountingStorage::new(
        Arc::clone(&local) as Arc<dyn Storage>
    ));
    let counting: Arc<dyn Storage> = Arc::clone(&counting_inner) as Arc<dyn Storage>;

    let schema = make_schema(10);

    // Create the table with normal locking.
    let mut writer = Writer::new(Arc::clone(&counting), "products", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Create acquired at least one lock; commit acquired one more.
    let count_after_normal = counting_inner.lock_count();
    assert!(
        count_after_normal >= 2,
        "normal writer should acquire locks"
    );

    // Now open a writer with assume_lock_held=true. It must not call
    // lock_exclusive, because the caller is responsible for the lock.
    let options = icefalldb_core::WriterOptions {
        lock_timeout: Duration::from_secs(30),
        assume_lock_held: true,
    };
    let mut writer =
        Writer::new_with_options(Arc::clone(&counting), "products", schema.clone(), options)
            .await
            .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    assert_eq!(
        counting_inner.lock_count(),
        count_after_normal,
        "assume_lock_held writer must not acquire additional locks"
    );

    // replace() must also skip lock acquisition.
    let mut writer =
        Writer::new_with_options(Arc::clone(&counting), "products", schema.clone(), options)
            .await
            .unwrap();
    writer
        .insert_batch(make_batch(vec![7, 8, 9]))
        .await
        .unwrap();
    writer.replace().await.unwrap();

    assert_eq!(
        counting_inner.lock_count(),
        count_after_normal,
        "assume_lock_held replace must not acquire additional locks"
    );
}

fn make_parquet_file(path: &std::path::Path, ids: &[i64]) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(Int64Array::from(ids.to_vec()))],
    )
    .unwrap();

    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), arrow_schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test]
async fn test_insert_duplicate_parquet_reuses_data_file() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let schema = make_schema(1_000_000);

    let parquet_path = dir.path().join("input.parquet");
    let ids: Vec<i64> = (0..100).collect();
    make_parquet_file(&parquet_path, &ids);

    let mut writer = Writer::create(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();
    let first = writer
        .insert_parquet(parquet_path.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(first, InsertParquetOutcome::FastPath { rows: 100 });

    let mut writer = Writer::new(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();
    let second = writer
        .insert_parquet(parquet_path.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(second, InsertParquetOutcome::FastPath { rows: 100 });

    // Only one .parquet data file should exist; the two references share it.
    let entries = storage.list("products/").await.unwrap();
    let parquet_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".parquet")).collect();
    assert_eq!(parquet_files.len(), 1);

    // The second insert added a new row-group entry referencing the same data file,
    // so the manifest contains two row groups but only one physical Parquet file.
    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 2);
    assert_eq!(manifest.row_groups[0].data, manifest.row_groups[1].data);
    assert_eq!(manifest.next_row_id, 200);

    // Logical table rows reflect both inserts; physical storage is not doubled.
    let mut total_rows = 0usize;
    let mut all_row_ids = Vec::new();
    for entry in &manifest.row_groups {
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        total_rows += meta.rows;
        all_row_ids.extend(meta.row_ids.iter().flat_map(segment_ids));
    }
    assert_eq!(total_rows, 200);
    assert_eq!(all_row_ids.len(), 200);
    let unique_row_ids: HashSet<_> = all_row_ids.iter().copied().collect();
    assert_eq!(
        unique_row_ids.len(),
        200,
        "dedup append must allocate fresh stable row IDs"
    );
}

#[tokio::test]
async fn test_duplicate_parquet_updates_non_unique_index() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let schema = make_schema(1_000_000);

    let parquet_path = dir.path().join("input.parquet");
    let ids: Vec<i64> = (0..100).collect();
    make_parquet_file(&parquet_path, &ids);

    let mut writer = Writer::create(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();
    writer
        .insert_parquet(parquet_path.to_str().unwrap())
        .await
        .unwrap();

    let catalog = DatabaseCatalog::new(Arc::clone(&storage));
    let lock = catalog.acquire_lock(Duration::from_secs(30)).await.unwrap();
    let manifest = read_latest_manifest(&storage, "products").await;
    let definition = IndexDefinition {
        name: "products_id_idx".into(),
        table: "products".into(),
        column: "id".into(),
        unique: false,
    };
    let index = build_btree_index(storage.as_ref(), &definition, &manifest)
        .await
        .unwrap();
    catalog
        .create_index_definition_with_options(
            &lock,
            "products_id_idx",
            "products",
            "id",
            "btree",
            false,
        )
        .await
        .unwrap();
    index.save(storage.as_ref()).await.unwrap();
    drop(lock);

    let mut writer = Writer::new(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();
    writer
        .insert_parquet(parquet_path.to_str().unwrap())
        .await
        .unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    let index_ref = manifest
        .index_generations
        .get("products_id_idx")
        .expect("dedup commit should record a fresh index generation");
    let index = load_index_by_ref(storage.as_ref(), "products", "products_id_idx", index_ref)
        .await
        .unwrap()
        .expect("index generation should load");
    let hits = index.lookup("42");
    assert_eq!(
        hits.len(),
        2,
        "non-unique index must include both logical copies of a deduped file"
    );
    assert_ne!(hits[0], hits[1]);
}

/// Regression test for M01: content-addressed dedup must not bypass a UNIQUE
/// index. Re-ingesting a Parquet file whose checksum matches an already-committed
/// fragment used to take the reference-append shortcut, silently re-adding the
/// same keys as live rows (with duplicate row_ids no rebuild could detect). With
/// a unique index present, the second ingest must instead run the uniqueness
/// probe and reject the duplicate keys.
#[tokio::test]
async fn test_duplicate_parquet_rejected_under_unique_index() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let schema = make_schema(1_000_000);

    let parquet_path = dir.path().join("input.parquet");
    let ids: Vec<i64> = (0..100).collect();
    make_parquet_file(&parquet_path, &ids);

    let mut writer = Writer::create(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();
    writer
        .insert_parquet(parquet_path.to_str().unwrap())
        .await
        .unwrap();

    // Create a unique index on `id` over the committed fragment.
    let catalog = DatabaseCatalog::new(Arc::clone(&storage));
    let lock = catalog.acquire_lock(Duration::from_secs(30)).await.unwrap();
    let manifest = read_latest_manifest(&storage, "products").await;
    let definition = IndexDefinition {
        name: "products_id_idx".into(),
        table: "products".into(),
        column: "id".into(),
        unique: true,
    };
    let index = build_btree_index(storage.as_ref(), &definition, &manifest)
        .await
        .unwrap();
    catalog
        .create_index_definition_with_options(
            &lock,
            "products_id_idx",
            "products",
            "id",
            "btree",
            true,
        )
        .await
        .unwrap();
    index.save(storage.as_ref()).await.unwrap();
    drop(lock);

    // Re-ingest the identical file. The dedup shortcut must be skipped and the
    // uniqueness probe must reject the duplicate keys.
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();
    let result = writer.insert_parquet(parquet_path.to_str().unwrap()).await;
    assert!(
        matches!(result, Err(icefalldb_core::IcefallDBError::UniqueKeyViolation { .. })),
        "expected UniqueKeyViolation re-ingesting a duplicate into a unique-indexed table, got {result:?}"
    );

    // No second fragment was committed: the table still has exactly one row group
    // and 100 live rows.
    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(
        manifest.row_groups.len(),
        1,
        "duplicate ingest must not add a fragment"
    );
}

/// Integration test: insert a batch, confirm the manifest `RowGroupEntry.agg`
/// is set, the `.agg` file exists, its `content_hash` matches
/// `RowGroupMeta.checksum`, and the partials match a direct Arrow aggregate
/// over the inserted data.
#[tokio::test]
async fn test_agg_sidecar_written_on_insert() {
    use arrow::array::Float64Array;
    use arrow::datatypes::Field;
    use icefalldb_core::agg_cache::{deserialize_agg_state, AggScalar};

    // Build a schema with an Int64 column and a Float64 column.
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "qty".into(),
                r#type: "int64".into(),
                nullable: true,
                field_id: 0,
            },
            Column {
                name: "price".into(),
                r#type: "float64".into(),
                nullable: true,
                field_id: 1,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 1,
        dropped_columns: vec![],
    };

    let arrow_schema = ArrowSchema::new(vec![
        Field::new("qty", DataType::Int64, true),
        Field::new("price", DataType::Float64, true),
    ]);

    // Batch: qty = [10, null, 20, 30], price = [1.5, 2.5, null, 4.0]
    let qty_array = Int64Array::from(vec![Some(10_i64), None, Some(20), Some(30)]);
    let price_array = Float64Array::from(vec![Some(1.5_f64), Some(2.5), None, Some(4.0)]);
    let batch = RecordBatch::try_new(
        Arc::new(arrow_schema),
        vec![Arc::new(qty_array), Arc::new(price_array)],
    )
    .unwrap();

    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "agg_test", schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    // Load the committed manifest and verify the single row group entry.
    let manifest = read_latest_manifest(&storage, "agg_test").await;
    assert_eq!(manifest.row_groups.len(), 1);
    let entry = &manifest.row_groups[0];

    // ── Assert: RowGroupEntry.agg is set ────────────────────────────────────
    let agg_filename = entry
        .agg
        .as_ref()
        .expect("RowGroupEntry.agg must be set after insert");

    // ── Assert: .agg file exists in storage ─────────────────────────────────
    let agg_path = format!("agg_test/{}", agg_filename);
    assert!(
        storage.exists(&agg_path).await.unwrap(),
        ".agg file must exist at {agg_path}"
    );

    // ── Load the .agg file and verify content_hash == RowGroupMeta.checksum ─
    let meta_bytes = storage
        .read(&format!("agg_test/{}", entry.meta))
        .await
        .unwrap();
    let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes).unwrap();

    let agg_bytes = storage.read(&agg_path).await.unwrap();
    let agg_state = deserialize_agg_state(&agg_bytes).unwrap();

    assert_eq!(
        agg_state.content_hash, meta.checksum,
        "agg content_hash must equal RowGroupMeta.checksum"
    );

    // ── Assert partials match direct Arrow aggregates ────────────────────────
    let qty_agg = agg_state.cols.get("qty").expect("qty column in .agg");
    assert_eq!(qty_agg.count_non_null, 3, "qty: three non-null values");
    match &qty_agg.sum {
        AggScalar::Int(s) => assert_eq!(*s, 60, "qty sum must be exact 60"),
        other => panic!("qty sum expected Int, got {other:?}"),
    }
    match &qty_agg.sumsq {
        AggScalar::Int(s) => assert_eq!(*s, 1400, "qty sumsq must be exact 1400"),
        other => panic!("qty sumsq expected Int, got {other:?}"),
    }

    let price_agg = agg_state.cols.get("price").expect("price column in .agg");
    assert_eq!(price_agg.count_non_null, 3, "price: three non-null values");
    let price_sum = match &price_agg.sum {
        AggScalar::Float(v) => *v,
        other => panic!("price sum expected Float, got {other:?}"),
    };
    let price_sumsq = match &price_agg.sumsq {
        AggScalar::Float(v) => *v,
        other => panic!("price sumsq expected Float, got {other:?}"),
    };
    let tol = |expected: f64| 1e-9 * expected.abs().max(1.0);
    assert!(
        (price_sum - 8.0).abs() <= tol(8.0),
        "price sum {price_sum} not within tolerance of 8.0"
    );
    assert!(
        (price_sumsq - 24.5).abs() <= tol(24.5),
        "price sumsq {price_sumsq} not within tolerance of 24.5"
    );
}

/// M15-B: an externally-ingested Parquet file may carry no footer column
/// statistics. Regenerating a `.meta` from the footer alone would record
/// nulls=0/min=max=None, which the checker later flags against the real data.
/// `compute_row_group_meta_from_footer` must read the data for those columns
/// and produce exact statistics.
#[tokio::test]
async fn test_repair_meta_recovers_stats_when_footer_lacks_them() {
    use icefalldb_core::writer::compute_row_group_meta_from_footer;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::{EnabledStatistics, WriterProperties};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("external.parquet");

    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        true,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(Int64Array::from(vec![
            Some(5),
            None,
            Some(2),
            Some(9),
        ]))],
    )
    .unwrap();

    // Disable statistics entirely so the footer carries none.
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::None)
        .build();
    {
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = ArrowWriter::try_new(file, arrow_schema, Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }
    let buf = std::fs::read(&path).unwrap();
    let metadata = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&path).unwrap())
        .unwrap()
        .metadata()
        .as_ref()
        .clone();

    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "v".into(),
            r#type: "int64".into(),
            nullable: true,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);

    let meta = compute_row_group_meta_from_footer(
        "rg_x",
        1,
        &schema,
        &buf,
        &metadata,
        &std::collections::HashSet::new(),
        &[],
    )
    .unwrap();

    let stats = meta.columns.get("v").expect("stats for column v");
    assert_eq!(
        stats.nulls, 1,
        "null count must come from the data, not the footer"
    );
    assert_eq!(stats.min, Some(serde_json::json!(2)));
    assert_eq!(stats.max, Some(serde_json::json!(9)));

    // The regenerated meta must agree with its own data (meta_checksum + data
    // checksum), so a follow-up checker run would not flag it.
    assert!(meta.verify_meta_checksum().unwrap());
    assert!(meta.verify_against_data(&buf));
}
