//! Persistent query result cache for the IcefallDB DataFusion engine.
//!
//! The cache stores complete query results as Arrow IPC files on disk. Cache
//! entries are keyed by the SQL text, the set of tables referenced, and the
//! snapshot sequence of each table at the time the result was computed. When a
//! table advances to a new snapshot, the key changes naturally and stale
//! results are ignored (and can be garbage-collected later).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use arrow::datatypes::SchemaRef;
use arrow::ipc::reader::FileReader;
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use sha2::{Digest, Sha256};

use crate::{QueryError, Result};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Eviction policy for the on-disk result cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictPolicy {
    Lru,
}

impl EvictPolicy {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "lru" => Ok(EvictPolicy::Lru),
            other => Err(QueryError::Other(format!(
                "unsupported result_cache_evict {other:?} (only \"lru\")"
            ))),
        }
    }
}

/// On-disk cache for query results.
#[derive(Debug, Clone)]
pub struct ResultCache {
    dir: PathBuf,
    max_bytes: u64,
    evict: EvictPolicy,
}

impl ResultCache {
    /// Open or create a cache at `dir` with a byte budget (`0` disables) and policy.
    pub fn new(dir: impl AsRef<Path>, max_bytes: u64, evict: EvictPolicy) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|e| QueryError::Other(format!("cache dir: {e}")))?;
        Ok(Self {
            dir,
            max_bytes,
            evict,
        })
    }

    /// Return the cache directory.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Whether caching is enabled (budget > 0).
    pub fn enabled(&self) -> bool {
        self.max_bytes > 0
    }

    /// Compute the canonical cache key for a query.
    ///
    /// `tables` and `snapshots` must be aligned (same length, table i at
    /// snapshot i). Using a `BTreeMap` internally makes the key independent of
    /// caller ordering.
    pub fn key(sql: &str, tables: &[String], snapshots: &[u64]) -> String {
        let mut map: BTreeMap<&str, u64> = BTreeMap::new();
        for (t, s) in tables.iter().zip(snapshots.iter()) {
            map.insert(t, *s);
        }

        let mut hasher = Sha256::new();
        hasher.update(sql.as_bytes());
        hasher.update(b"\0tables\0");
        for (table, snapshot) in map {
            hasher.update(table.as_bytes());
            hasher.update(snapshot.to_le_bytes());
        }
        let digest = hasher.finalize();
        hex::encode(digest)
    }

    fn path(&self, sql: &str, tables: &[String], snapshots: &[u64]) -> PathBuf {
        self.dir
            .join(format!("{}.arrow", Self::key(sql, tables, snapshots)))
    }

    /// Return cached record batches if a valid entry exists.
    ///
    /// Returns `Ok(None)` on any missing or corrupt file — never `Err`.
    pub fn get(
        &self,
        sql: &str,
        tables: &[String],
        snapshots: &[u64],
    ) -> Result<Option<Vec<RecordBatch>>> {
        if !self.enabled() {
            return Ok(None);
        }
        let path = self.path(sql, tables, snapshots);
        let file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(_) => return Ok(None), // NotFound or raced unlink -> miss
        };
        let reader = match FileReader::try_new(file, None) {
            Ok(r) => r,
            Err(_) => {
                let _ = fs::remove_file(&path); // corrupt -> drop, miss
                return Ok(None);
            }
        };
        let mut batches = Vec::new();
        for batch in reader {
            match batch {
                Ok(b) => batches.push(b),
                Err(_) => {
                    let _ = fs::remove_file(&path);
                    return Ok(None);
                }
            }
        }
        // LRU recency: touch mtime on hit (best effort).
        let _ = filetime_now(&path);
        Ok(Some(batches))
    }

    /// Write record batches to the cache.
    ///
    /// If `batches` is empty and no schema can be inferred, nothing is written.
    /// Use `put_table` to cache zero-row results with a known schema.
    pub fn put(
        &self,
        sql: &str,
        tables: &[String],
        snapshots: &[u64],
        batches: &[RecordBatch],
    ) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let schema = match batches.first() {
            Some(b) => b.schema(),
            None => return Ok(()), // truly unknown schema: nothing to write
        };
        self.put_table(sql, tables, snapshots, &schema, batches)
    }

    /// Write `batches` (possibly empty) under `schema`. Used for zero-row results
    /// where the schema is known independently of the batch list.
    pub fn put_table(
        &self,
        sql: &str,
        tables: &[String],
        snapshots: &[u64],
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        let path = self.path(sql, tables, snapshots);
        let counter = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = self.dir.join(format!(
            "{}.{}.{}.tmp",
            path.file_stem().and_then(|s| s.to_str()).unwrap_or("entry"),
            std::process::id(),
            counter
        ));
        {
            let file = fs::File::create(&tmp)
                .map_err(|e| QueryError::Other(format!("cache create: {e}")))?;
            let mut writer = FileWriter::try_new(file, schema.as_ref())
                .map_err(|e| QueryError::Other(format!("cache writer: {e}")))?;
            if batches.is_empty() {
                let empty = RecordBatch::new_empty(std::sync::Arc::clone(schema));
                writer
                    .write(&empty)
                    .map_err(|e| QueryError::Other(format!("cache write: {e}")))?;
            } else {
                for batch in batches {
                    writer
                        .write(batch)
                        .map_err(|e| QueryError::Other(format!("cache write: {e}")))?;
                }
            }
            writer
                .finish()
                .map_err(|e| QueryError::Other(format!("cache finish: {e}")))?;
        }
        let entry_size = fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        if self.max_bytes > 0 && entry_size > self.max_bytes / 8 {
            let _ = fs::remove_file(&tmp); // too big to cache; drop it
            return Ok(());
        }

        fs::rename(&tmp, &path).map_err(|e| QueryError::Other(format!("cache rename: {e}")))?;
        self.enforce_budget()?;
        Ok(())
    }

    /// Evict oldest-by-mtime `.arrow` entries until total bytes <= 90% of budget.
    fn enforce_budget(&self) -> Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        let EvictPolicy::Lru = self.evict;
        let mut entries: Vec<(std::time::SystemTime, u64, PathBuf)> = Vec::new();
        let mut total: u64 = 0;
        for e in fs::read_dir(&self.dir)
            .map_err(|e| QueryError::Other(format!("cache read_dir: {e}")))?
        {
            let e = match e {
                Ok(e) => e,
                Err(_) => continue,
            };
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("arrow") {
                continue; // ignore *.tmp and others
            }
            let meta = match e.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            total += meta.len();
            entries.push((mtime, meta.len(), p));
        }
        if total <= self.max_bytes {
            return Ok(());
        }
        let low_water = self.max_bytes - self.max_bytes / 10; // 90%
        entries.sort_by_key(|(mtime, _, _)| *mtime); // oldest first
        for (_, size, path) in entries {
            if total <= low_water {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(size);
            }
        }
        Ok(())
    }

    /// Atomically invalidate all entries in the cache.
    pub fn clear(&self) -> Result<()> {
        for entry in fs::read_dir(&self.dir)
            .map_err(|e| QueryError::Other(format!("cache read_dir: {e}")))?
        {
            let entry = entry.map_err(|e| QueryError::Other(format!("cache entry: {e}")))?;
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str());
            if ext == Some("arrow") || ext == Some("tmp") {
                fs::remove_file(&path)
                    .map_err(|e| QueryError::Other(format!("cache remove: {e}")))?;
            }
        }
        Ok(())
    }
}

/// Non-deterministic functions that always appear with a call `(` in SQL.
const NONDETERMINISTIC_FN: &[&str] = &["random", "now", "uuid", "nextval", "gen_random"];

/// Non-deterministic niladic keywords in DuckDB (no parentheses required).
/// These must be matched as bare words to avoid false-positives on identifiers
/// like `current_date_col`.
const NONDETERMINISTIC_BARE: &[&str] = &["current_date", "current_time", "current_timestamp"];

/// Return true if `word` appears in `lower` as a whole token (not part of a
/// longer identifier). Boundaries are positions where neither a letter, digit,
/// nor underscore exists.
fn is_bare_word(lower: &str, word: &str) -> bool {
    let bytes = lower.as_bytes();
    let wlen = word.len();
    let mut start = 0usize;
    while start + wlen <= bytes.len() {
        if let Some(rel) = lower[start..].find(word) {
            let abs = start + rel;
            let before_ok = abs == 0 || {
                let b = bytes[abs - 1];
                !b.is_ascii_alphanumeric() && b != b'_'
            };
            let after_pos = abs + wlen;
            let after_ok = after_pos >= bytes.len() || {
                let b = bytes[after_pos];
                !b.is_ascii_alphanumeric() && b != b'_'
            };
            if before_ok && after_ok {
                return true;
            }
            start = abs + 1;
        } else {
            break;
        }
    }
    false
}

/// Whether a statement's result may be cached: a SELECT that references at least
/// one attached table and uses no non-deterministic function. The same rule is
/// mirrored in the Python adapter so all surfaces agree.
pub fn is_cacheable_select(sql: &str, table_names: &[String]) -> bool {
    let lower = sql.trim_start().to_ascii_lowercase();
    if !lower.starts_with("select") {
        return false;
    }
    // Function-call form: random(...), now(), uuid(), etc.
    if NONDETERMINISTIC_FN
        .iter()
        .any(|f| lower.contains(&format!("{f}(")))
    {
        return false;
    }
    // Niladic bare-keyword form: current_date, current_time, current_timestamp
    if NONDETERMINISTIC_BARE
        .iter()
        .any(|w| is_bare_word(&lower, w))
    {
        return false;
    }
    table_names
        .iter()
        .any(|t| lower.contains(&t.to_ascii_lowercase()))
}

/// Resolve the (lowercased, bare) table names a single `SELECT` actually reads —
/// through CTEs and subqueries — via DataFusion's own reference resolver, so the
/// result-cache key can depend on only those tables instead of every table
/// registered on the connection.
///
/// Returns `None` when the SQL cannot be parsed/resolved or is not a single
/// statement. Callers MUST then key conservatively on all registered tables:
/// under-keying (missing a real dependency) would let a mutation pass unnoticed
/// and serve a stale result. CTE-defined names are excluded (they are not real
/// tables), so `WITH x AS (...) SELECT … FROM x, t` resolves to just `t`.
pub fn referenced_tables(sql: &str) -> Option<Vec<String>> {
    use datafusion::sql::parser::DFParser;
    use datafusion::sql::resolve::resolve_table_references;

    let mut stmts = DFParser::parse_sql(sql).ok()?;
    if stmts.len() != 1 {
        return None;
    }
    let stmt = stmts.pop_front()?;
    let (refs, _ctes) = resolve_table_references(&stmt, true).ok()?;
    Some(
        refs.iter()
            .map(|r| r.table().to_ascii_lowercase())
            .collect(),
    )
}

/// Set a file's mtime to "now" (best-effort LRU recency touch) without an extra
/// dependency: rewrite the file's times via `std::fs::File` + `set_modified`.
fn filetime_now(path: &Path) -> std::io::Result<()> {
    let f = fs::OpenOptions::new().write(true).open(path)?;
    f.set_modified(std::time::SystemTime::now())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::TempDir;

    fn batch_of(n_rows: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(
                (0..n_rows as i32).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    #[test]
    fn test_lru_eviction_keeps_total_under_budget() {
        let dir = TempDir::new().unwrap();
        // Tiny budget so a few entries blow it.
        let cache = ResultCache::new(dir.path(), 4096, EvictPolicy::Lru).unwrap();
        let tables = vec!["t".to_string()];
        for i in 0..50u64 {
            let sql = format!("SELECT a FROM t WHERE a = {i}");
            cache.put(&sql, &tables, &[1], &[batch_of(64)]).unwrap();
        }
        let total: u64 = std::fs::read_dir(cache.dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("arrow"))
            .map(|e| e.metadata().unwrap().len())
            .sum();
        assert!(
            total <= 4096,
            "cache {total} bytes must stay <= budget 4096"
        );
    }

    #[test]
    fn test_per_entry_cap_skips_oversized() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 4096, EvictPolicy::Lru).unwrap(); // cap = 512
        let tables = vec!["t".to_string()];
        cache
            .put("SELECT a FROM t", &tables, &[1], &[batch_of(100_000)])
            .unwrap();
        assert!(
            cache
                .get("SELECT a FROM t", &tables, &[1])
                .unwrap()
                .is_none(),
            "oversized result must not be cached"
        );
    }

    fn make_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_cache_round_trip() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 1 << 20, EvictPolicy::Lru).unwrap();
        let batch = make_batch();
        let sql = "SELECT a, b FROM t";
        let tables = vec!["t".to_string()];
        let snapshots = vec![7u64];

        assert!(cache.get(sql, &tables, &snapshots).unwrap().is_none());
        cache
            .put(sql, &tables, &snapshots, std::slice::from_ref(&batch))
            .unwrap();
        let got = cache.get(sql, &tables, &snapshots).unwrap().unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0], batch);
    }

    #[test]
    fn test_cache_key_includes_snapshot() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 1 << 20, EvictPolicy::Lru).unwrap();
        let batch = make_batch();
        let sql = "SELECT a, b FROM t";
        let tables = vec!["t".to_string()];

        cache
            .put(sql, &tables, &[7], std::slice::from_ref(&batch))
            .unwrap();
        assert!(cache.get(sql, &tables, &[8]).unwrap().is_none());
        assert!(cache.get(sql, &tables, &[7]).unwrap().is_some());
    }

    #[test]
    fn test_cache_key_table_order_independent() {
        let sql = "SELECT * FROM t1, t2";
        let key1 = ResultCache::key(sql, &["t1".into(), "t2".into()], &[1, 2]);
        let key2 = ResultCache::key(sql, &["t2".into(), "t1".into()], &[2, 1]);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_cache_empty_result_round_trips_with_schema() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 1 << 20, EvictPolicy::Lru).unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let empty = RecordBatch::new_empty(schema.clone());
        let sql = "SELECT a FROM t WHERE false";
        let tables = vec!["t".to_string()];
        let snaps = vec![1u64];

        cache
            .put(sql, &tables, &snaps, std::slice::from_ref(&empty))
            .unwrap();
        let got = cache.get(sql, &tables, &snaps).unwrap();
        assert!(got.is_some(), "empty-but-typed result must be a hit");
        let batches = got.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 0);
        assert_eq!(batches[0].schema().field(0).name(), "a");
    }

    #[test]
    fn test_get_missing_file_is_miss_not_error() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 1 << 20, EvictPolicy::Lru).unwrap();
        let sql = "SELECT 1 FROM t";
        let tables = vec!["t".to_string()];
        let snaps = vec![1u64];
        // No put: must be Ok(None), never Err.
        assert!(cache.get(sql, &tables, &snaps).unwrap().is_none());
    }

    #[test]
    fn test_put_table_empty_batches_preserves_schema() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 1 << 20, EvictPolicy::Lru).unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, true),
        ]));
        let sql = "SELECT a, b FROM t WHERE false";
        let tables = vec!["t".to_string()];
        let snaps = vec![1u64];

        // put_table with zero batches
        cache.put_table(sql, &tables, &snaps, &schema, &[]).unwrap();
        let got = cache.get(sql, &tables, &snaps).unwrap();
        assert!(got.is_some(), "zero-batch put_table must be a hit");
        let batches = got.unwrap();
        assert_eq!(batches.len(), 1, "must return exactly one schema-bearer");
        assert_eq!(batches[0].num_rows(), 0, "batch must have 0 rows");
        assert_eq!(batches[0].schema().fields().len(), 2, "must have 2 fields");
        assert_eq!(batches[0].schema().field(0).name(), "a");
        assert_eq!(batches[0].schema().field(1).name(), "b");
    }

    #[test]
    fn test_eligibility() {
        let tables = vec!["events".to_string()];
        assert!(is_cacheable_select("SELECT count(*) FROM events", &tables));
        assert!(is_cacheable_select(
            "  select * from Events where x>1",
            &tables
        ));
        assert!(!is_cacheable_select("SELECT random()", &tables)); // non-deterministic
        assert!(!is_cacheable_select("SELECT now()", &tables)); // non-deterministic
        assert!(!is_cacheable_select("SHOW TABLES", &tables)); // not a select
        assert!(!is_cacheable_select("SELECT 1", &tables)); // no table ref
        assert!(!is_cacheable_select("VALUES (1),(2)", &tables)); // not a select
                                                                  // Bare niladic current_date / current_time / current_timestamp
        assert!(!is_cacheable_select(
            "SELECT current_date, count(*) FROM events GROUP BY current_date",
            &tables
        ));
        assert!(!is_cacheable_select(
            "SELECT current_time FROM events",
            &tables
        ));
        assert!(!is_cacheable_select(
            "SELECT current_timestamp FROM events",
            &tables
        ));
        // A column named current_date_col must NOT be rejected
        assert!(is_cacheable_select(
            "SELECT current_date_col FROM events",
            &tables
        ));
    }

    #[test]
    fn test_referenced_tables_resolves_real_dependencies() {
        let has = |sql: &str, t: &str| referenced_tables(sql).unwrap().contains(&t.to_string());
        assert!(has("SELECT * FROM users WHERE id = 1", "users"));
        // Joins: both sides are dependencies.
        let j = referenced_tables("SELECT * FROM a JOIN b ON a.id = b.id").unwrap();
        assert!(j.contains(&"a".to_string()) && j.contains(&"b".to_string()));
        // A correlated subquery's table is a dependency.
        let s = referenced_tables("SELECT * FROM a WHERE id IN (SELECT id FROM b)").unwrap();
        assert!(s.contains(&"a".to_string()) && s.contains(&"b".to_string()));
        // A CTE's *base* table is a real dependency (the CTE name itself is not).
        assert!(has(
            "WITH c AS (SELECT * FROM base) SELECT * FROM c",
            "base"
        ));
        // Unresolvable inputs return None so the caller keys on all tables.
        assert!(referenced_tables("SELECT * FROM a; SELECT * FROM b").is_none());
        assert!(referenced_tables("this is not valid sql").is_none());
    }

    #[test]
    fn test_get_corrupt_file_is_miss() {
        let dir = TempDir::new().unwrap();
        let cache = ResultCache::new(dir.path(), 1 << 20, EvictPolicy::Lru).unwrap();
        let sql = "SELECT 1 FROM t";
        let tables = vec!["t".to_string()];
        let snaps = vec![1u64];
        let path = cache
            .dir()
            .join(format!("{}.arrow", ResultCache::key(sql, &tables, &snaps)));
        std::fs::write(&path, b"not an arrow file").unwrap();
        assert!(
            cache.get(sql, &tables, &snaps).unwrap().is_none(),
            "corrupt file reads as miss"
        );
    }
}
