//! Process-wide Parquet metadata cache for the DataFusion query engine.
//!
//! IcefallDB tables are immutable directories of Parquet files. Each row-group
//! file carries a content-addressed `sha256:<hex>` checksum in its sidecar
//! metadata. Because the checksum is stable per file, `(path, checksum)` is a
//! reliable cache key: when a file is replaced by compaction or a new commit,
//! its checksum changes, naturally invalidating stale entries.
//!
//! The cache stores `Arc<ParquetMetaData>` so that multiple scans over the same
//! file share the decoded metadata without copying. It is bounded and evicts
//! entries by access order (LRU). A global singleton with the default capacity
//! is available through [`ParquetMetadataCache::global`].

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock, RwLock};

use parquet::file::metadata::ParquetMetaData;

/// Default maximum number of entries retained in the process-wide cache.
pub(crate) const DEFAULT_CACHE_CAPACITY: usize = 256;

/// A cache key that uniquely identifies a Parquet file's metadata.
#[derive(Clone, Debug, Eq)]
struct CacheKey {
    path: Arc<str>,
    checksum: Arc<str>,
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.checksum == other.checksum
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.path.hash(state);
        self.checksum.hash(state);
    }
}

impl CacheKey {
    fn new(path: &str, checksum: &str) -> Self {
        Self {
            path: Arc::from(path),
            checksum: Arc::from(checksum),
        }
    }
}

/// Cached metadata plus the footer-derived length needed to reconstruct the
/// Parquet file buffer without re-reading the footer tail.
#[derive(Debug, Clone)]
struct CachedEntry {
    metadata: Arc<ParquetMetaData>,
    /// Length of the thrift-encoded Parquet footer metadata (the value stored
    /// in the last 8 bytes of the file just before the magic bytes).
    metadata_len: u64,
}

/// Bounded, thread-safe LRU cache for decoded [`ParquetMetaData`].
///
/// The cache is keyed by `(path, checksum)`. Files are immutable and the
/// checksum is content-addressed, so stale metadata is automatically evicted
/// on file change (the new file has a different checksum).
#[derive(Debug, Clone)]
pub(crate) struct ParquetMetadataCache {
    inner: Arc<RwLock<CacheInner>>,
}

#[derive(Debug)]
struct CacheInner {
    capacity: usize,
    /// Metadata storage, keyed by `(path, checksum)`.
    entries: HashMap<CacheKey, CachedEntry>,
    /// LRU ordering: front is the least-recently used, back is the most
    /// recently used. An entry is moved to the back on every access.
    order: VecDeque<CacheKey>,
}

static GLOBAL_CACHE: OnceLock<ParquetMetadataCache> = OnceLock::new();

impl ParquetMetadataCache {
    /// Create a new cache with the given entry capacity.
    ///
    /// A capacity of `0` disables the cache: all `get` and `put` operations
    /// become no-ops.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                capacity,
                entries: HashMap::with_capacity(capacity.max(1)),
                order: VecDeque::with_capacity(capacity.max(1)),
            })),
        }
    }

    /// Return the process-wide singleton cache with the default capacity.
    ///
    /// The singleton is lazily initialized the first time it is called.
    pub(crate) fn global() -> Self {
        GLOBAL_CACHE
            .get_or_init(|| ParquetMetadataCache::new(DEFAULT_CACHE_CAPACITY))
            .clone()
    }

    /// Return the process-wide singleton cache, initializing it with the given
    /// capacity if it has not been created yet.
    ///
    /// If the singleton already exists, the provided capacity is ignored. This
    /// lets `IcefallDBTableProvider::new` seed the cache size from
    /// `ProviderConfig` without forcing callers to plumb a cache instance
    /// through every scan path.
    pub(crate) fn global_with_capacity(capacity: usize) -> Self {
        GLOBAL_CACHE
            .get_or_init(|| ParquetMetadataCache::new(capacity))
            .clone()
    }

    /// Retrieve metadata from the cache if present, promoting it to the most
    /// recently used position.
    ///
    /// Reads are not serialized with writes: a read lock is used for the lookup
    /// and clone, and a write lock is only taken to update the LRU order.
    pub(crate) fn get(&self, path: &str, checksum: &str) -> Option<Arc<ParquetMetaData>> {
        self.get_entry(path, checksum)
            .map(|e| Arc::clone(&e.metadata))
    }

    /// Retrieve metadata plus the footer metadata length. This lets callers
    /// reconstruct sparse Parquet buffers without re-reading the footer tail.
    pub(crate) fn get_with_footer(
        &self,
        path: &str,
        checksum: &str,
    ) -> Option<(Arc<ParquetMetaData>, u64)> {
        self.get_entry(path, checksum)
            .map(|e| (Arc::clone(&e.metadata), e.metadata_len))
    }

    fn get_entry(&self, path: &str, checksum: &str) -> Option<CachedEntry> {
        let key = CacheKey::new(path, checksum);

        // Fast path: read lock only.
        let value = {
            let guard = self.inner.read().ok()?;
            guard.entries.get(&key).cloned()
        };

        // Promote to most-recently used under a write lock. If the write lock
        // is poisoned, still return the cached metadata.
        if value.is_some() {
            if let Ok(mut guard) = self.inner.write() {
                if guard.entries.contains_key(&key) {
                    guard.order.retain(|k| k != &key);
                    guard.order.push_back(key);
                }
            }
        }

        value
    }

    /// Insert metadata into the cache. If the key already exists, the value is
    /// updated and the entry is promoted. If the cache is over capacity, the
    /// least-recently used entry is evicted.
    ///
    /// When capacity is `0`, this is a no-op.
    pub(crate) fn put(
        &self,
        path: &str,
        checksum: &str,
        metadata: Arc<ParquetMetaData>,
        metadata_len: u64,
    ) {
        let key = CacheKey::new(path, checksum);
        let Ok(mut guard) = self.inner.write() else {
            return;
        };

        if guard.capacity == 0 {
            return;
        }

        let existed = guard.entries.contains_key(&key);
        if existed {
            guard.order.retain(|k| k != &key);
        }
        guard.order.push_back(key.clone());
        guard.entries.insert(
            key,
            CachedEntry {
                metadata,
                metadata_len,
            },
        );

        if !existed && guard.order.len() > guard.capacity {
            if let Some(evicted) = guard.order.pop_front() {
                guard.entries.remove(&evicted);
            }
        }
    }

    /// Remove all entries from the cache.
    #[allow(dead_code)]
    pub(crate) fn clear(&self) {
        let Ok(mut guard) = self.inner.write() else {
            return;
        };
        guard.entries.clear();
        guard.order.clear();
    }

    /// Return the number of entries currently in the cache.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.entries.len())
            .unwrap_or_default()
    }

    /// Return the configured capacity.
    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.capacity)
            .unwrap_or(DEFAULT_CACHE_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parquet::file::metadata::ParquetMetaData;
    use std::sync::Arc;
    use std::thread;

    /// Build a minimal `ParquetMetaData` placeholder for eviction tests.
    ///
    /// The cache only stores and clones `Arc`s, so the metadata content does
    /// not affect the keying or eviction logic. Returns the metadata and the
    /// thrift-encoded metadata length read from the Parquet footer.
    fn dummy_metadata() -> (Arc<ParquetMetaData>, u64) {
        use arrow::array::Int32Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::io::Cursor;

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "x",
            DataType::Int32,
            true,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![Some(1)]))],
        )
        .unwrap();
        let mut buf = Vec::new();
        {
            let mut cursor = Cursor::new(&mut buf);
            let mut writer = ArrowWriter::try_new(&mut cursor, Arc::clone(&schema), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        // Extract the thrift-encoded metadata bytes from the end of the file.
        // The Parquet footer layout is: <metadata><metadata_len (4 LE)><magic (4)>
        const FOOTER_SIZE: usize = 8;
        let metadata_len = u32::from_le_bytes(
            buf[buf.len() - FOOTER_SIZE..buf.len() - 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let metadata_bytes = &buf[buf.len() - FOOTER_SIZE - metadata_len..buf.len() - FOOTER_SIZE];

        let metadata = Arc::new(
            parquet::file::metadata::ParquetMetaDataReader::decode_metadata(metadata_bytes)
                .unwrap(),
        );
        (metadata, metadata_len as u64)
    }

    #[test]
    fn test_put_and_get() {
        let cache = ParquetMetadataCache::new(4);
        let (meta, len) = dummy_metadata();

        assert!(cache.get("path1", "cs1").is_none());
        cache.put("path1", "cs1", Arc::clone(&meta), len);
        let got = cache.get("path1", "cs1").expect("cached metadata");
        assert!(Arc::ptr_eq(&got, &meta));
    }

    #[test]
    fn test_capacity_evicts_lru() {
        let cache = ParquetMetadataCache::new(2);
        let (m1, l1) = dummy_metadata();
        let (m2, l2) = dummy_metadata();
        let (m3, l3) = dummy_metadata();

        cache.put("p1", "cs1", Arc::clone(&m1), l1);
        cache.put("p2", "cs2", Arc::clone(&m2), l2);

        // Access p1 so it becomes most-recently used.
        assert!(cache.get("p1", "cs1").is_some());

        // Insert p3: p2 is the least-recently used and should be evicted.
        cache.put("p3", "cs3", Arc::clone(&m3), l3);

        assert!(cache.get("p1", "cs1").is_some());
        assert!(
            cache.get("p2", "cs2").is_none(),
            "LRU entry should be evicted"
        );
        assert!(cache.get("p3", "cs3").is_some());
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_same_path_different_checksum() {
        let cache = ParquetMetadataCache::new(4);
        let (m1, l1) = dummy_metadata();
        let (m2, l2) = dummy_metadata();

        cache.put("same", "cs_a", Arc::clone(&m1), l1);
        cache.put("same", "cs_b", Arc::clone(&m2), l2);

        assert!(Arc::ptr_eq(&cache.get("same", "cs_a").unwrap(), &m1));
        assert!(Arc::ptr_eq(&cache.get("same", "cs_b").unwrap(), &m2));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_global_singleton() {
        let a = ParquetMetadataCache::global();
        let b = ParquetMetadataCache::global();
        assert_eq!(a.capacity(), DEFAULT_CACHE_CAPACITY);
        assert_eq!(b.capacity(), DEFAULT_CACHE_CAPACITY);

        let (meta, len) = dummy_metadata();
        a.put("global_path", "global_cs", Arc::clone(&meta), len);
        assert!(Arc::ptr_eq(
            &b.get("global_path", "global_cs").unwrap(),
            &meta
        ));

        // Clear the singleton so later tests see a clean global cache.
        a.clear();
    }

    #[test]
    fn test_capacity_zero_disables_cache() {
        let cache = ParquetMetadataCache::new(0);
        let (meta, len) = dummy_metadata();

        assert_eq!(cache.capacity(), 0);
        cache.put("path", "cs", Arc::clone(&meta), len);
        assert!(cache.get("path", "cs").is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_checksum_change_invalidate_entry() {
        // Capacity of 1 means the new checksum entry evicts the old one,
        // demonstrating that different checksums are distinct keys and that
        // a changed file (new checksum) does not return stale metadata for the
        // old checksum.
        let cache = ParquetMetadataCache::new(1);
        let (m1, l1) = dummy_metadata();
        let (m2, l2) = dummy_metadata();

        cache.put("path", "cs_old", Arc::clone(&m1), l1);
        assert!(cache.get("path", "cs_old").is_some());

        // Simulate the file being overwritten: the checksum changes.
        cache.put("path", "cs_new", Arc::clone(&m2), l2);
        assert!(
            cache.get("path", "cs_old").is_none(),
            "old checksum should miss after the file changed"
        );
        assert!(Arc::ptr_eq(&cache.get("path", "cs_new").unwrap(), &m2));
    }

    #[test]
    fn test_concurrent_get_put_consistency() {
        // Capacity must exceed the test's working set so the presence assertions
        // below are deterministic: the 4 odd threads each insert 50 distinct keys
        // (~200) plus the seeded "shared" entry. With a smaller capacity, LRU
        // eviction under concurrency would race the `is_some()` checks (an entry
        // can be legitimately evicted between a put and a get), making the test
        // flaky without indicating any cache defect. Sized to avoid eviction so
        // the test exercises concurrent get/put consistency, not eviction timing.
        let cache = ParquetMetadataCache::new(512);
        let (meta, len) = dummy_metadata();

        // Seed an entry so threads can both read and update it.
        cache.put("shared", "cs", Arc::clone(&meta), len);

        let mut handles = Vec::new();
        for i in 0..8 {
            let cache = cache.clone();
            let meta = Arc::clone(&meta);
            handles.push(thread::spawn(move || {
                for round in 0..100 {
                    // Mix reads and writes across the same and distinct keys.
                    if round % 2 == 0 || i % 2 == 0 {
                        let got = cache.get("shared", "cs");
                        assert!(got.is_some(), "thread {i} round {round}");
                        assert!(Arc::ptr_eq(&got.unwrap(), &meta));
                    } else {
                        let key = format!("key_{i}_{round}");
                        cache.put(&key, "cs", Arc::clone(&meta), len);
                        assert!(cache.get(&key, "cs").is_some());
                    }
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // The shared entry must still be present and consistent.
        assert!(Arc::ptr_eq(&cache.get("shared", "cs").unwrap(), &meta));
    }

    #[test]
    fn test_clear_removes_all_entries() {
        let cache = ParquetMetadataCache::new(4);
        let (meta, len) = dummy_metadata();
        cache.put("p1", "cs", Arc::clone(&meta), len);
        cache.put("p2", "cs", Arc::clone(&meta), len);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get("p1", "cs").is_none());
        assert!(cache.get("p2", "cs").is_none());
    }
}
