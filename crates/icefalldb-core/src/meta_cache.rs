//! Process-wide immutable `.meta` sidecar cache for `Reader::scan_internal`.
//!
//! Fragment files are named `rg_{uuid}` (globally unique, write-once). A `.meta`
//! sidecar path is created exactly once and is never overwritten in place
//! (immutability; patch fragments and compaction allocate new UUIDs). Therefore
//! `(meta_storage_path → RowGroupMeta)` is a permanent bijection — a path-keyed
//! cache never goes stale. Deletion state (`entry.deletes`, `entry.deleted_count`)
//! comes from the manifest entry, which is always re-parsed, NOT from the cached
//! sidecar. The cache covers only the immutable per-fragment stats/offsets/row_ids.
//!
//! The cache is bounded by an LRU policy. A capacity of `0` disables it entirely.
//! The global singleton is initialised lazily via [`MetaCache::global`].

use crate::metadata::RowGroupMeta;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, OnceLock, RwLock};

/// Default maximum number of `.meta` sidecar entries retained process-wide.
///
/// Sidecars are small JSON blobs; 65 536 entries should cover even very large
/// tables without meaningful memory pressure.
pub const DEFAULT_META_CACHE_CAPACITY: usize = 65_536;

/// Bounded, thread-safe LRU cache mapping a storage path to its decoded
/// [`RowGroupMeta`].
///
/// Because sidecar paths are write-once (globally unique UUID-based names), a
/// path-keyed cache is always valid for the lifetime of the process. Only the
/// SUCCESS path (checksum-verified, schema-id-matched) is cached; the
/// `allow_missing_meta` placeholder branch is never inserted.
///
/// Lock discipline mirrors `ParquetMetadataCache`: `get` acquires a read lock
/// for the lookup and a write lock to update LRU order; `put` acquires a write
/// lock. No `await` is ever held while holding either lock.
#[derive(Debug, Clone)]
pub struct MetaCache {
    inner: Arc<RwLock<CacheInner>>,
}

#[derive(Debug)]
struct CacheInner {
    capacity: usize,
    entries: HashMap<String, Arc<RowGroupMeta>>,
    /// LRU order: front = least-recently used, back = most-recently used.
    order: VecDeque<String>,
}

static GLOBAL_META_CACHE: OnceLock<MetaCache> = OnceLock::new();

impl MetaCache {
    /// Create a new cache with the given entry capacity.
    ///
    /// Capacity `0` disables the cache; all `get`/`put` calls become no-ops.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CacheInner {
                capacity,
                entries: HashMap::with_capacity(capacity.max(1)),
                order: VecDeque::with_capacity(capacity.max(1)),
            })),
        }
    }

    /// Return the process-wide singleton cache, lazily initialised with
    /// [`DEFAULT_META_CACHE_CAPACITY`].
    pub fn global() -> Self {
        GLOBAL_META_CACHE
            .get_or_init(|| MetaCache::new(DEFAULT_META_CACHE_CAPACITY))
            .clone()
    }

    /// Return the process-wide singleton, seeding it with `capacity` if it has
    /// not been created yet. If it already exists, the provided capacity is
    /// ignored.
    pub fn global_with_capacity(capacity: usize) -> Self {
        GLOBAL_META_CACHE
            .get_or_init(|| MetaCache::new(capacity))
            .clone()
    }

    /// Look up a sidecar entry by its storage path.
    ///
    /// Returns `Some(Arc<RowGroupMeta>)` on a hit and promotes the entry to the
    /// most-recently-used position. Returns `None` on a miss or when the cache
    /// is disabled (capacity 0).
    pub fn get(&self, path: &str) -> Option<Arc<RowGroupMeta>> {
        // Fast path: read lock for the lookup.
        let value = {
            let guard = self.inner.read().ok()?;
            guard.entries.get(path).cloned()
        };

        // Promote to MRU under a write lock.
        if value.is_some() {
            if let Ok(mut guard) = self.inner.write() {
                if guard.entries.contains_key(path) {
                    guard.order.retain(|k| k != path);
                    guard.order.push_back(path.to_string());
                }
            }
        }

        value
    }

    /// Insert a sidecar entry.
    ///
    /// If the path already exists the value is updated and the entry is promoted
    /// to MRU. When the cache is at capacity, the least-recently-used entry is
    /// evicted before insertion. When capacity is `0`, this is a no-op.
    pub fn put(&self, path: String, meta: Arc<RowGroupMeta>) {
        let Ok(mut guard) = self.inner.write() else {
            return;
        };

        if guard.capacity == 0 {
            return;
        }

        let existed = guard.entries.contains_key(&path);
        if existed {
            guard.order.retain(|k| k != &path);
        }
        guard.order.push_back(path.clone());
        guard.entries.insert(path, meta);

        if !existed && guard.order.len() > guard.capacity {
            if let Some(evicted) = guard.order.pop_front() {
                guard.entries.remove(&evicted);
            }
        }
    }

    /// Remove all entries from the cache.
    ///
    /// Intended for tests that need deterministic isolation; UUID paths make
    /// cross-test collision essentially impossible in production, but `clear()`
    /// makes isolation explicit.
    pub fn clear(&self) {
        let Ok(mut guard) = self.inner.write() else {
            return;
        };
        guard.entries.clear();
        guard.order.clear();
    }

    /// Return the number of entries currently held.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.entries.len())
            .unwrap_or_default()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the configured capacity.
    pub fn capacity(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.capacity)
            .unwrap_or(DEFAULT_META_CACHE_CAPACITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::RowGroupMeta;
    use std::collections::HashMap;

    fn dummy_meta(tag: &str) -> Arc<RowGroupMeta> {
        Arc::new(RowGroupMeta {
            row_group: tag.to_string(),
            schema_id: 1,
            rows: 10,
            columns: HashMap::new(),
            column_offsets: None,
            sort: None,
            row_ids: vec![],
            checksum: "sha256:abc".to_string(),
            meta_checksum: "sha256:def".to_string(),
        })
    }

    #[test]
    fn test_put_and_get() {
        let cache = MetaCache::new(4);
        let m = dummy_meta("rg_a");

        assert!(cache.get("table/rg_a.meta").is_none());
        cache.put("table/rg_a.meta".to_string(), Arc::clone(&m));
        let got = cache.get("table/rg_a.meta").expect("cached meta");
        assert!(Arc::ptr_eq(&got, &m));
    }

    #[test]
    fn test_capacity_zero_disables_cache() {
        let cache = MetaCache::new(0);
        let m = dummy_meta("rg_z");

        assert_eq!(cache.capacity(), 0);
        cache.put("table/rg_z.meta".to_string(), Arc::clone(&m));
        assert!(cache.get("table/rg_z.meta").is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = MetaCache::new(2);
        let m1 = dummy_meta("rg_1");
        let m2 = dummy_meta("rg_2");
        let m3 = dummy_meta("rg_3");

        cache.put("t/rg_1.meta".to_string(), Arc::clone(&m1));
        cache.put("t/rg_2.meta".to_string(), Arc::clone(&m2));

        // Access rg_1 so it becomes MRU.
        assert!(cache.get("t/rg_1.meta").is_some());

        // Insert rg_3: rg_2 is the LRU and should be evicted.
        cache.put("t/rg_3.meta".to_string(), Arc::clone(&m3));

        assert!(cache.get("t/rg_1.meta").is_some(), "rg_1 must survive");
        assert!(
            cache.get("t/rg_2.meta").is_none(),
            "rg_2 must be evicted as LRU"
        );
        assert!(cache.get("t/rg_3.meta").is_some(), "rg_3 must be present");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_clear() {
        let cache = MetaCache::new(4);
        cache.put("t/rg_a.meta".to_string(), dummy_meta("rg_a"));
        cache.put("t/rg_b.meta".to_string(), dummy_meta("rg_b"));
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.get("t/rg_a.meta").is_none());
        assert!(cache.get("t/rg_b.meta").is_none());
    }

    #[test]
    fn test_global_singleton() {
        let a = MetaCache::global();
        let b = MetaCache::global();
        assert_eq!(a.capacity(), DEFAULT_META_CACHE_CAPACITY);
        assert_eq!(b.capacity(), DEFAULT_META_CACHE_CAPACITY);

        let m = dummy_meta("rg_global");
        a.put("t/rg_global.meta".to_string(), Arc::clone(&m));
        assert!(Arc::ptr_eq(&b.get("t/rg_global.meta").unwrap(), &m));

        // Clean up so other tests see a consistent global state.
        a.clear();
    }
}
