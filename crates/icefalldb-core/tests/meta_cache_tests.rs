//! Regression guard for the process-wide `.meta` sidecar cache.
//!
//! This test verifies that `Reader::scan` reads each `.meta` sidecar exactly
//! once (on first access) and zero times on subsequent scans of the same
//! manifest. Adding a new fragment causes exactly one additional read (the new
//! sidecar only). All three assertions are load-bearing: removing the cache
//! causes the counts to revert to N, N, N+1.

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use icefalldb_core::meta_cache::MetaCache;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::{Reader, Result, Writer};
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Counting storage wrapper
// ---------------------------------------------------------------------------

/// Wraps any `Storage` and counts the number of `read` calls whose path ends
/// with `.meta`.
struct CountingStorage {
    inner: Arc<dyn Storage>,
    meta_reads: Arc<AtomicUsize>,
}

impl CountingStorage {
    fn new(inner: Arc<dyn Storage>) -> (Self, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        let s = Self {
            inner,
            meta_reads: Arc::clone(&counter),
        };
        (s, counter)
    }
}

#[async_trait]
impl Storage for CountingStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        if path.ends_with(".meta") {
            self.meta_reads.fetch_add(1, Ordering::Relaxed);
        }
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
        self.inner.sync(path).await
    }

    async fn sync_data(&self, path: &str) -> Result<()> {
        self.inner.sync_data(path).await
    }

    async fn sync_root(&self) -> Result<()> {
        self.inner.sync_root().await
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_schema(target_rows: usize) -> Schema {
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
        row_group_target_rows: target_rows,
        row_group_target_bytes: 64 * 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn int_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int64Array::from(ids))]).unwrap()
}

// ---------------------------------------------------------------------------
// Regression: counting-storage read counts
// ---------------------------------------------------------------------------

/// Three-assertion regression guard for the `.meta` sidecar cache.
///
/// Assertion 1 (load-bearing): first scan reads exactly N sidecars.
/// Assertion 2 (load-bearing): second scan on the SAME manifest reads 0.
/// Assertion 3 (load-bearing): scan after adding ONE new fragment reads exactly 1.
///
/// Removing the cache makes all three assertions fail (counts become N, N, N+1).
#[tokio::test]
async fn test_meta_cache_read_counts() {
    // Use a unique table name so the UUID-based sidecar paths do not collide
    // with other test runs that share the global cache.
    let table = "meta_cache_count_test";
    const N: usize = 4; // number of initial fragments (≥3 per spec)

    // Build table on a plain MemoryStorage, then wrap it for counting.
    let mem: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    // Write N fragments (each 1 row; target_rows=1 forces one row group per insert).
    let schema = make_schema(1);
    {
        let mut writer = Writer::new(Arc::clone(&mem), table, schema.clone())
            .await
            .unwrap();
        for i in 0..(N as i64) {
            writer.insert_batch(int_batch(vec![i])).await.unwrap();
        }
        writer.commit().await.unwrap();
    }

    // Delete the snapshot checkpoint so this test continues to exercise the
    // per-fragment `.meta` sidecar cache path (the checkpoint path would otherwise
    // read zero `.meta` files).
    {
        let catalog = icefalldb_core::catalog::Catalog::load(mem.as_ref(), table)
            .await
            .unwrap();
        if let Some(cp_rel) = catalog
            .latest_manifest()
            .and_then(|m| m.checkpoint.as_ref())
        {
            mem.delete(&format!("{}/{}", table, cp_rel)).await.unwrap();
        }
    }

    // Wrap the populated storage in the counting shim.
    let (counting, counter) = CountingStorage::new(Arc::clone(&mem));
    let counting: Arc<dyn Storage> = Arc::new(counting);

    // Clear the global cache for deterministic isolation.
    MetaCache::global().clear();

    // -----------------------------------------------------------------------
    // Assertion 1: first scan → exactly N .meta reads.
    // -----------------------------------------------------------------------
    let reader = Reader::new(&*counting, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(
        plan.row_groups.len(),
        N,
        "expected {N} row groups in initial plan"
    );
    let reads_after_first_scan = counter.load(Ordering::Relaxed);
    assert_eq!(
        reads_after_first_scan, N,
        "assertion 1 failed: first scan must read exactly {N} .meta sidecars, got {reads_after_first_scan}"
    );

    // -----------------------------------------------------------------------
    // Assertion 2: second scan on the SAME manifest → 0 .meta reads.
    // -----------------------------------------------------------------------
    counter.store(0, Ordering::Relaxed);
    let plan2 = reader.scan().await.unwrap();
    assert_eq!(plan2.row_groups.len(), N);
    let reads_after_second_scan = counter.load(Ordering::Relaxed);
    assert_eq!(
        reads_after_second_scan, 0,
        "assertion 2 failed: second scan must read 0 .meta sidecars (all cached), got {reads_after_second_scan}"
    );

    // -----------------------------------------------------------------------
    // Assertion 3: append ONE new fragment, re-scan → exactly 1 .meta read.
    // -----------------------------------------------------------------------
    {
        // Write the new fragment directly to the underlying mem storage so
        // the counting wrapper's counter only captures reads, not writes.
        let mut writer = Writer::new(Arc::clone(&mem), table, schema.clone())
            .await
            .unwrap();
        writer.insert_batch(int_batch(vec![99])).await.unwrap();
        writer.commit().await.unwrap();
    }

    // Delete the new snapshot checkpoint so the third scan exercises the
    // `.meta` cache path as well.
    {
        let catalog = icefalldb_core::catalog::Catalog::load(mem.as_ref(), table)
            .await
            .unwrap();
        if let Some(cp_rel) = catalog
            .latest_manifest()
            .and_then(|m| m.checkpoint.as_ref())
        {
            mem.delete(&format!("{}/{}", table, cp_rel)).await.unwrap();
        }
    }

    // Refresh the reader so it picks up the new manifest.
    let reader_refreshed = Reader::new(&*counting, table).await.unwrap();
    // The constructor above already read the new manifest; reset counter before scan.
    counter.store(0, Ordering::Relaxed);

    let plan3 = reader_refreshed.scan().await.unwrap();
    assert_eq!(
        plan3.row_groups.len(),
        N + 1,
        "expected {n1} row groups after appending one fragment",
        n1 = N + 1
    );
    let reads_after_third_scan = counter.load(Ordering::Relaxed);
    assert_eq!(
        reads_after_third_scan, 1,
        "assertion 3 failed: scan after adding 1 new fragment must read exactly 1 .meta sidecar, got {reads_after_third_scan}"
    );
}
