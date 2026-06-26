use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use icefalldb_core::metadata::{Column, Manifest, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::{IcefallDBError, Result, Writer};

fn make_test_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".to_string(),
            r#type: "int64".to_string(),
            nullable: false,
            field_id: 0,
        }],
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_int_batch(values: Vec<i64>) -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(values)) as ArrayRef],
    )
    .unwrap()
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum FaultOp {
    Write,
    Sync,
    Rename,
    Read,
}

#[derive(Clone, Debug)]
enum PathMatch {
    Contains(&'static str),
    Exact(&'static str),
}

#[derive(Clone, Debug)]
struct FaultRule {
    op: FaultOp,
    path: PathMatch,
    occurrence: usize,
    message: &'static str,
}

#[derive(Clone, Debug)]
struct FaultInjectingStorage {
    inner: Arc<MemoryStorage>,
    rule: Arc<Mutex<Option<FaultRule>>>,
    counter: Arc<AtomicUsize>,
}

impl FaultInjectingStorage {
    fn new() -> Self {
        Self {
            inner: Arc::new(MemoryStorage::new()),
            rule: Arc::new(Mutex::new(None)),
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn inner(&self) -> &MemoryStorage {
        self.inner.as_ref()
    }

    fn set_rule(&self, rule: FaultRule) {
        *self.rule.lock().unwrap() = Some(rule);
        self.counter.store(0, Ordering::SeqCst);
    }

    fn clear_rule(&self) {
        *self.rule.lock().unwrap() = None;
        self.counter.store(0, Ordering::SeqCst);
    }

    fn should_fail(&self, op: &FaultOp, path: &str) -> Option<IcefallDBError> {
        let rule = self.rule.lock().unwrap();
        let rule = rule.as_ref()?;
        if rule.op != *op {
            return None;
        }
        let matches = match &rule.path {
            PathMatch::Contains(s) => path.contains(s),
            PathMatch::Exact(s) => path == *s,
        };
        if !matches {
            return None;
        }
        let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        if count == rule.occurrence {
            Some(IcefallDBError::Other(Box::new(std::io::Error::other(
                rule.message,
            ))))
        } else {
            None
        }
    }
}

#[async_trait]
impl Storage for FaultInjectingStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        if let Some(e) = self.should_fail(&FaultOp::Read, path) {
            return Err(e);
        }
        self.inner.read(path).await
    }

    async fn size(&self, path: &str) -> Result<u64> {
        if let Some(e) = self.should_fail(&FaultOp::Read, path) {
            return Err(e);
        }
        self.inner.size(path).await
    }

    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        self.inner.read_range(path, offset, len).await
    }

    async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        if let Some(e) = self.should_fail(&FaultOp::Write, path) {
            return Err(e);
        }
        self.inner.write(path, data).await
    }

    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }

    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        if let Some(e) = self.should_fail(&FaultOp::Rename, from) {
            return Err(e);
        }
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
        if let Some(e) = self.should_fail(&FaultOp::Sync, path) {
            return Err(e);
        }
        self.inner.sync(path).await
    }
}

async fn latest_sequence(storage: &MemoryStorage, table: &str) -> Option<u64> {
    let pointer_data = storage
        .read(&format!("{}/_manifest.json", table))
        .await
        .ok()?;
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).ok()?;
    let seq = pointer.get("latest").and_then(|v| v.as_u64())?;
    // `latest: 0` indicates an empty table with no committed manifests.
    if seq == 0 {
        None
    } else {
        Some(seq)
    }
}

async fn committed_row_group_count(storage: &MemoryStorage, table: &str) -> usize {
    let Some(seq) = latest_sequence(storage, table).await else {
        return 0;
    };
    let manifest: Manifest = serde_json::from_slice(
        &storage
            .read(&format!("{}/{}", table, Manifest::filename(seq)))
            .await
            .unwrap(),
    )
    .unwrap();
    manifest.row_groups.len()
}

async fn setup_table(storage: &FaultInjectingStorage, table: &str) {
    let _writer = make_writer(storage, table).await;
    // MemoryStorage has no real directories; once the .keep file written by
    // ensure_dir is deleted the directory appears empty again. Leave persistent
    // markers so later writers' ensure_dir calls list successfully and do not
    // rewrite a .keep file while a fault rule is active.
    let _ = storage
        .inner()
        .write(&format!("{}/_staging/intents/.dir", table), b"")
        .await;
    let _ = storage
        .inner()
        .write(&format!("{}/_staging/incoming/.dir", table), b"")
        .await;
    let _ = storage
        .inner()
        .write(&format!("{}/_manifests/.dir", table), b"")
        .await;
}

async fn make_writer(storage: &FaultInjectingStorage, table: &str) -> Writer {
    Writer::new(
        Arc::new(storage.clone()) as Arc<dyn Storage>,
        table,
        make_test_schema(),
    )
    .await
    .unwrap()
}

async fn expect_commit_fails(storage: &FaultInjectingStorage, table: &str, rule: FaultRule) {
    storage.set_rule(rule);
    let mut writer = make_writer(storage, table).await;
    writer
        .insert_batch(make_int_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    assert!(writer.commit().await.is_err());
    assert_eq!(latest_sequence(storage.inner(), table).await, None);
    assert_eq!(committed_row_group_count(storage.inner(), table).await, 0);
}

async fn expect_commit_succeeds(
    storage: &FaultInjectingStorage,
    table: &str,
    expected_seq: u64,
    expected_count: usize,
) {
    let mut writer = make_writer(storage, table).await;
    writer
        .insert_batch(make_int_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    assert_eq!(
        latest_sequence(storage.inner(), table).await,
        Some(expected_seq)
    );
    assert_eq!(
        committed_row_group_count(storage.inner(), table).await,
        expected_count
    );
}

#[tokio::test]
async fn test_fault_during_intent_write_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Write,
            path: PathMatch::Contains("_staging/intents/"),
            occurrence: 1,
            message: "injected intent write fault",
        },
    )
    .await;

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_part_file_write_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Write,
            path: PathMatch::Contains("_staging/incoming/"),
            occurrence: 1,
            message: "injected part file write fault",
        },
    )
    .await;

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_part_rename_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Rename,
            path: PathMatch::Contains("_staging/incoming/"),
            occurrence: 1,
            message: "injected part rename fault",
        },
    )
    .await;

    let root_files = storage.inner().list(table).await.unwrap();
    assert!(
        !root_files.iter().any(|f| f.ends_with(".parquet")),
        "no parquet files should exist at table root: {:?}",
        root_files
    );

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_manifest_tmp_write_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Write,
            path: PathMatch::Contains("_manifests/"),
            occurrence: 1,
            message: "injected manifest tmp write fault",
        },
    )
    .await;

    let manifests = storage
        .inner()
        .list(&format!("{}/_manifests", table))
        .await
        .unwrap_or_default();
    assert!(
        !manifests.iter().any(|f| f.ends_with(".json.tmp")),
        "no manifest tmp orphan should remain: {:?}",
        manifests
    );

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_manifest_rename_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Rename,
            path: PathMatch::Contains("_manifests/"),
            occurrence: 1,
            message: "injected manifest rename fault",
        },
    )
    .await;

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_pointer_tmp_write_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Write,
            path: PathMatch::Contains("_manifest.json.tmp"),
            occurrence: 1,
            message: "injected pointer tmp write fault",
        },
    )
    .await;

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_pointer_rename_rolls_back() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    expect_commit_fails(
        &storage,
        table,
        FaultRule {
            op: FaultOp::Rename,
            path: PathMatch::Contains("_manifest.json.tmp"),
            occurrence: 1,
            message: "injected pointer rename fault",
        },
    )
    .await;

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 1, 1).await;
}

#[tokio::test]
async fn test_fault_during_final_root_sync_commits() {
    let table = "products";
    let storage = FaultInjectingStorage::new();
    setup_table(&storage, table).await;

    storage.set_rule(FaultRule {
        op: FaultOp::Sync,
        path: PathMatch::Exact("products/"),
        occurrence: 2,
        message: "injected final root sync fault",
    });

    let mut writer = make_writer(&storage, table).await;
    writer
        .insert_batch(make_int_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    assert!(writer.commit().await.is_err());

    assert_eq!(latest_sequence(storage.inner(), table).await, Some(1));
    assert_eq!(committed_row_group_count(storage.inner(), table).await, 1);

    storage.clear_rule();
    expect_commit_succeeds(&storage, table, 2, 2).await;
}
