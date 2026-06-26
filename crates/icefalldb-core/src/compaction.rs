use crate::agg_cache::{compute_agg_state_with_key, serialize_agg_state};
use crate::catalog::Catalog;
use crate::deletion::DeletionVector;
use crate::metadata::{Manifest, RowGroupEntry, RowGroupMeta, Schema};
use crate::reader::Reader;
use crate::rowid::{segment_ids, RowIdSegment};
use crate::rowindex::writer::rebuild as rebuild_rowindex;
use crate::storage::Storage;
use crate::writer::{
    build_snapshot_checkpoint, cleanup_staging, compute_row_group_meta, write_checkpoint,
};
use crate::{IcefallDBError, Result};
use arrow::array::{
    Array, BooleanArray, Int64Array, LargeStringArray, RecordBatch, StringArray,
    TimestampMicrosecondArray,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, SchemaRef, TimeUnit};
use futures::StreamExt;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::properties::{WriterProperties, WriterVersion};
use parquet::schema::types::ColumnPath;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

fn other<E: std::error::Error + Send + Sync + 'static>(err: E) -> IcefallDBError {
    IcefallDBError::Other(Box::new(err))
}

/// ZSTD level used by the compactor/optimizer (size/speed trade-off).
const COMPACTION_ZSTD_LEVEL: i32 = 1;

/// Maximum number of optimistic compaction attempts before giving up cleanly.
///
/// Each attempt pins a source snapshot, runs the heavy rewrite lock-free, and
/// commits under the writer lock only if the snapshot is still current. A
/// mutation that commits during the heavy phase invalidates the staged work and
/// forces a retry against the fresh snapshot; after this many conflicting
/// rounds the compactor returns without rewriting rather than spin forever.
const MAX_COMMIT_ATTEMPTS: u32 = 3;

/// Outcome of a single optimistic compaction attempt.
enum AttemptOutcome {
    /// The compacted manifest was published.
    Committed(CompactionResult),
    /// The source snapshot advanced during the heavy phase (a mutation
    /// committed); staged work was discarded and the caller should retry.
    Conflict,
    /// Nothing required compaction (empty table, single row group, etc.).
    NothingToDo(CompactionResult),
    /// The heavy phase folded all input rows away (no output); not a conflict.
    Empty(CompactionResult),
}

/// Enable dictionary encoding when distinct values are less than 1/N of total rows.
const DICTIONARY_CARDINALITY_THRESHOLD_DENOMINATOR: usize = 10;

/// Returns true if `values` is non-decreasing (each element <= the next).
fn is_non_decreasing(values: &[i64]) -> bool {
    values.windows(2).all(|w| w[0] <= w[1])
}

/// Extract valid i64 values from an array that is either `Int64Array` or
/// `TimestampMicrosecondArray`.
///
/// This is shared by the `Int64` and `Timestamp(Microsecond, None)` adaptive
/// encoding branches below.
fn collect_valid_i64<T>(array: &arrow::array::PrimitiveArray<T>) -> Vec<i64>
where
    T: ArrowPrimitiveType<Native = i64>,
{
    array.iter().flatten().collect()
}

fn extract_i64_values(array: &dyn Array) -> Option<Vec<i64>> {
    if let Some(a) = array.as_any().downcast_ref::<Int64Array>() {
        return Some(collect_valid_i64(a));
    }
    array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .map(collect_valid_i64)
}

/// Sort `batch` by `sort_keys` ascending, or return a clone if no keys are
/// requested.
///
/// Missing sort keys produce a clear [`IcefallDBError::Other`] error.
fn maybe_sort_batch(batch: &RecordBatch, sort_keys: &[String]) -> Result<RecordBatch> {
    if sort_keys.is_empty() {
        return Ok(batch.clone());
    }

    let schema = batch.schema();
    let field_names: Vec<_> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    let sort_columns: Vec<_> = sort_keys
        .iter()
        .map(|k| {
            let values = batch.column_by_name(k).ok_or_else(|| {
                IcefallDBError::Other(Box::new(std::io::Error::other(format!(
                    "sort key '{}' not found in batch schema {:?}",
                    k, field_names
                ))))
            })?;
            Ok(arrow::compute::SortColumn {
                values: values.clone(),
                options: None,
            })
        })
        .collect::<Result<_>>()?;

    let indices = arrow::compute::lexsort_to_indices(&sort_columns, None).map_err(other)?;
    let columns: Vec<_> = batch
        .columns()
        .iter()
        .map(|c| {
            arrow::compute::take(c.as_ref(), &indices, None)
                .map_err(|e| IcefallDBError::Other(Box::new(e)))
        })
        .collect::<Result<_>>()?;

    RecordBatch::try_new(batch.schema().clone(), columns).map_err(other)
}

/// Build [`WriterProperties`] tuned for the data shape observed in
/// `sample_batches`.
///
/// - ZSTD level 1 compression and Parquet 2.0 writer version are always set.
/// - Low-cardinality UTF8 / LargeUtf8 columns get dictionary encoding enabled;
///   high-cardinality UTF8 / LargeUtf8 columns have dictionary encoding disabled
///   so the writer does not fall back to its default.
/// - Non-decreasing Int64 / microsecond timestamp columns get
///   `DELTA_BINARY_PACKED` encoding.
fn build_writer_properties(
    arrow_schema: &SchemaRef,
    sample_batches: &[RecordBatch],
) -> WriterProperties {
    let total_rows: usize = sample_batches.iter().map(|b| b.num_rows()).sum();

    let mut props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(
            ZstdLevel::try_new(COMPACTION_ZSTD_LEVEL).expect("valid zstd level"),
        ))
        .set_writer_version(WriterVersion::PARQUET_2_0);

    for (idx, field) in arrow_schema.fields().iter().enumerate() {
        match field.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {
                if total_rows == 0 {
                    continue;
                }
                let mut distinct = HashSet::<String>::new();
                for batch in sample_batches {
                    if let Some(a) = batch.column(idx).as_any().downcast_ref::<StringArray>() {
                        for v in a.iter().flatten() {
                            distinct.insert(v.to_owned());
                        }
                    } else if let Some(a) = batch
                        .column(idx)
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                    {
                        for v in a.iter().flatten() {
                            distinct.insert(v.to_owned());
                        }
                    }
                }
                let col_path = ColumnPath::new(vec![field.name().clone()]);
                if distinct.len() * DICTIONARY_CARDINALITY_THRESHOLD_DENOMINATOR < total_rows {
                    props = props.set_column_dictionary_enabled(col_path, true);
                } else {
                    props = props.set_column_dictionary_enabled(col_path, false);
                }
            }
            // Only plain microsecond timestamps (no timezone) are eligible for
            // delta encoding because that is the timestamp variant IcefallDB uses
            // internally. Other timestamp units or tz-aware types are left to
            // the default Parquet encoder.
            DataType::Int64 | DataType::Timestamp(TimeUnit::Microsecond, None) => {
                let values: Vec<i64> = sample_batches
                    .iter()
                    .flat_map(|b| extract_i64_values(b.column(idx)).unwrap_or_default())
                    .collect();
                if !values.is_empty() && is_non_decreasing(&values) {
                    let col_path = ColumnPath::new(vec![field.name().clone()]);
                    props = props
                        .set_column_encoding(col_path.clone(), Encoding::DELTA_BINARY_PACKED)
                        .set_column_dictionary_enabled(col_path, false);
                }
            }
            _ => {}
        }
    }

    props.build()
}

/// Scan `_manifests/` for the highest valid manifest snapshot and atomically
/// repair `_manifest.json` to point to it.
///
/// Returns the sequence and row groups of the repaired snapshot, or `None` if
/// no valid snapshots exist.
async fn repair_manifest_pointer(
    storage: &dyn Storage,
    table: &str,
) -> Result<Option<(u64, Vec<RowGroupEntry>)>> {
    let manifests_dir = format!("{}/_manifests", table);
    let entries = match storage.list(&manifests_dir).await {
        Ok(entries) => entries,
        Err(IcefallDBError::NotFound(_)) => return Ok(None),
        Err(e) => return Err(e),
    };

    let mut sequences: Vec<u64> = entries
        .iter()
        .filter_map(|entry| {
            let filename = std::path::Path::new(entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            filename
                .strip_suffix(".json")
                .and_then(|s| s.parse::<u64>().ok())
        })
        .collect();
    sequences.sort_unstable_by(|a, b| b.cmp(a));

    for seq in sequences {
        let manifest_path = format!("{}/{}", table, Manifest::filename(seq));
        if let Ok(manifest) = read_manifest_validated(storage, &manifest_path, seq).await {
            write_manifest_pointer(storage, table, seq).await?;
            return Ok(Some((seq, manifest.row_groups)));
        }
    }

    Ok(None)
}

/// Read a manifest file, parse it, and verify its checksum.
async fn read_manifest_validated(
    storage: &dyn Storage,
    manifest_path: &str,
    expected_seq: u64,
) -> Result<Manifest> {
    let data = storage.read(manifest_path).await?;
    let manifest: Manifest = serde_json::from_slice(&data)?;
    if manifest.sequence != expected_seq {
        return Err(IcefallDBError::InvalidManifestPointer(format!(
            "filename sequence {} does not match manifest sequence {}",
            expected_seq, manifest.sequence
        )));
    }
    if !manifest.verify_checksum()? {
        return Err(IcefallDBError::ChecksumMismatch {
            path: manifest_path.to_string(),
        });
    }
    Ok(manifest)
}

/// Read the current `_manifest.json` pointer and return the `latest` sequence
/// it references, or `None` if the pointer is missing.
///
/// This is the source-snapshot revalidation primitive for lock-free
/// compaction: the heavy fold pins the sequence returned here, and the brief
/// locked commit re-reads it under the writer lock to detect whether a mutation
/// landed in the meantime.
async fn read_pointer_sequence(storage: &dyn Storage, table: &str) -> Result<Option<u64>> {
    let pointer_path = format!("{}/_manifest.json", table);
    let data = match storage.read(&pointer_path).await {
        Ok(data) => data,
        Err(IcefallDBError::NotFound(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let pointer: serde_json::Value = serde_json::from_slice(&data)?;
    Ok(pointer.get("latest").and_then(|v| v.as_u64()))
}

/// Atomically write `_manifest.json` to point to `seq`.
async fn write_manifest_pointer(storage: &dyn Storage, table: &str, seq: u64) -> Result<()> {
    let pointer_path = format!("{}/_manifest.json", table);
    let tmp_path = format!("{}.tmp", pointer_path);
    let pointer = serde_json::json!({"latest": seq});
    storage
        .write(&tmp_path, serde_json::to_vec(&pointer)?.as_slice())
        .await?;
    storage.sync(&tmp_path).await?;
    storage.rename(&tmp_path, &pointer_path).await?;
    storage.sync(&format!("{}/", table)).await?;
    Ok(())
}

/// Configuration options for a [`Compactor`].
#[derive(Debug, Clone)]
pub struct CompactionOptions {
    /// Target number of rows per output row group.
    pub target_row_group_rows: usize,
    /// Target uncompressed byte size per output row group.
    pub target_row_group_bytes: usize,
    /// Maximum time to wait when acquiring the exclusive writer lock.
    pub lock_timeout: Duration,
    /// Rewrite even a single row group (e.g. to change compression).
    pub force: bool,
    /// Optional keys to sort by before writing output row groups.
    pub sort_keys: Vec<String>,
}

impl Default for CompactionOptions {
    fn default() -> Self {
        Self {
            target_row_group_rows: 1_000_000,
            target_row_group_bytes: 128 * 1024 * 1024,
            lock_timeout: Duration::from_secs(30),
            force: false,
            sort_keys: Vec::new(),
        }
    }
}

/// Statistics about a compaction run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionResult {
    /// Number of row groups in the input snapshot.
    pub input_row_groups: usize,
    /// Number of row groups in the output snapshot.
    pub output_row_groups: usize,
    /// Total rows in the input snapshot.
    pub input_rows: u64,
    /// Total rows in the output snapshot.
    pub output_rows: u64,
    /// Whether the compactor actually rewrote any row group files.
    pub rewrote: bool,
}

/// Accumulates input batches until they reach the target row/byte size for a
/// single output row group, then flushes a concatenated batch. This keeps the
/// compactor's peak memory bounded by one output row group instead of the
/// entire table.
struct StreamingRowGroupBuilder {
    target_rows: usize,
    target_bytes: usize,
    schema: SchemaRef,
    current: Vec<RecordBatch>,
    current_rows: usize,
    current_bytes: usize,
}

impl StreamingRowGroupBuilder {
    fn new(target_rows: usize, target_bytes: usize, schema: SchemaRef) -> Self {
        Self {
            target_rows,
            target_bytes,
            schema,
            current: Vec::new(),
            current_rows: 0,
            current_bytes: 0,
        }
    }

    fn is_full(&self) -> bool {
        self.current_rows >= self.target_rows || self.current_bytes >= self.target_bytes
    }

    /// Slices oversized input batches so every output row group respects the
    /// configured `target_rows` and `target_bytes`. Returns any complete output
    /// batches that should be flushed immediately; the caller is responsible for
    /// checking [`Self::is_full`] afterwards for batches that landed in the
    /// buffer.
    fn add(&mut self, batch: RecordBatch) -> Result<Vec<RecordBatch>> {
        if batch.num_rows() == 0 {
            return Ok(Vec::new());
        }

        let mut flushed = Vec::new();
        let mut remaining = batch;
        let mut remaining_rows = remaining.num_rows();

        // Estimate per-row byte cost from the incoming batch. Sliced arrays
        // share the parent buffer, so get_array_memory_size() stays large even
        // after slicing; using a stable per-row estimate avoids an infinite
        // loop when the byte target is binding.
        let total_bytes = remaining.get_array_memory_size();
        let bytes_per_row = total_bytes.checked_div(remaining_rows).unwrap_or(0);

        // If the current buffer is non-empty, fill it up to the tighter target.
        if !self.current.is_empty()
            && (self.current_rows + remaining_rows > self.target_rows
                || self.current_bytes + remaining_rows * bytes_per_row > self.target_bytes)
        {
            let rows_to_hit_row_target = self.target_rows.saturating_sub(self.current_rows);
            let rows_to_hit_byte_target = self
                .target_bytes
                .saturating_sub(self.current_bytes)
                .checked_div(bytes_per_row)
                .unwrap_or(remaining_rows);
            let take = rows_to_hit_row_target
                .min(rows_to_hit_byte_target)
                .max(1)
                .min(remaining_rows);
            let slice = remaining.slice(0, take);
            self.current_bytes += take * bytes_per_row;
            self.current_rows += take;
            self.current.push(slice);
            flushed.push(self.flush_force()?);
            remaining = remaining.slice(take, remaining_rows - take);
            remaining_rows -= take;
        }

        // Slice off full output row groups while the remainder is oversized.
        let rows_per_group = self
            .target_bytes
            .checked_div(bytes_per_row)
            .map(|rows_for_bytes| self.target_rows.min(rows_for_bytes).max(1))
            .unwrap_or(self.target_rows);
        while remaining_rows > rows_per_group {
            let take = rows_per_group.min(remaining_rows);
            flushed.push(remaining.slice(0, take));
            remaining = remaining.slice(take, remaining_rows - take);
            remaining_rows -= take;
        }

        // Append the tail (guaranteed to fit within targets by itself).
        if remaining_rows > 0 {
            self.current_bytes += remaining_rows * bytes_per_row;
            self.current_rows += remaining_rows;
            self.current.push(remaining);
        }

        Ok(flushed)
    }

    fn flush(&mut self) -> Result<Option<RecordBatch>> {
        if self.current_rows == 0 {
            self.current.clear();
            self.current_bytes = 0;
            return Ok(None);
        }
        Ok(Some(self.flush_force()?))
    }

    fn flush_force(&mut self) -> Result<RecordBatch> {
        let batch = arrow::compute::concat_batches(&self.schema, &self.current).map_err(other)?;
        self.current.clear();
        self.current_rows = 0;
        self.current_bytes = 0;
        Ok(batch)
    }
}

/// Mutable accumulators passed through the streaming compaction loop.
struct StagingState {
    staged_files: Vec<String>,
    intent_files: Vec<String>,
    new_entries: Vec<RowGroupEntry>,
    output_rows: u64,
    output_row_counts: Vec<usize>,
    partition_values: Option<HashMap<String, HashMap<String, serde_json::Value>>>,
    used_rg_ids: HashSet<String>,
    /// Next fragment ID to assign to an output row group.  Initialized from
    /// `manifest.next_fragment_id` so compacted fragments get stable IDs that
    /// do not collide with pre-existing fragments.
    next_fragment_id: u64,
}

/// Immutable context for the commit intent used while staging output row groups.
struct IntentContext<'a> {
    path: &'a str,
    txn_id: &'a str,
    started_at: &'a str,
}

/// An offline compactor that rewrites a table's row groups into a smaller,
/// better-sized set without deleting the old files.
pub struct Compactor<'a> {
    storage: &'a dyn Storage,
    table: String,
    options: CompactionOptions,
}

impl<'a> Compactor<'a> {
    /// Create a compactor for `table` with default options.
    pub fn new(storage: &'a dyn Storage, table: &str) -> Self {
        Self::with_options(storage, table, CompactionOptions::default())
    }

    /// Create a compactor for `table` with the provided options.
    pub fn with_options(storage: &'a dyn Storage, table: &str, options: CompactionOptions) -> Self {
        Self {
            storage,
            table: table.to_string(),
            options,
        }
    }

    /// Validate `sort_keys` against `schema.columns` and, if they differ from
    /// the current sort order, prepare a new schema snapshot in memory.
    ///
    /// This function does not write the schema file or update `_schema.json`;
    /// those steps happen inside the commit block together with the manifest
    /// pointer update so they can be rolled back atomically on failure.
    fn prepare_schema_for_sort(schema: &mut Schema, sort_keys: &[String]) -> Result<Option<u64>> {
        if sort_keys.is_empty() {
            return Ok(None);
        }
        if schema.sort.as_deref() == Some(sort_keys) {
            return Ok(None);
        }

        let column_names: HashSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        for key in sort_keys {
            if !column_names.contains(key.as_str()) {
                return Err(IcefallDBError::Other(Box::new(std::io::Error::other(
                    format!("sort key '{}' not found in schema columns", key),
                ))));
            }
        }

        let new_schema_id = schema.schema_id + 1;
        let mut new_schema = schema.clone();
        new_schema.schema_id = new_schema_id;
        new_schema.sort = Some(sort_keys.to_vec());
        new_schema.assign_field_ids(Some(schema));

        *schema = new_schema;
        Ok(Some(new_schema_id))
    }

    /// Write a schema snapshot file to `_schemas/{id}.json`.
    async fn write_schema_snapshot(
        storage: &dyn Storage,
        table: &str,
        schema: &Schema,
    ) -> Result<()> {
        let schema_path = format!("{}/{}", table, Schema::filename(schema.schema_id));
        let tmp_path = format!("{}.tmp", schema_path);
        let data = serde_json::to_vec(schema)?;
        storage.write(&tmp_path, &data).await?;
        storage.sync(&tmp_path).await?;
        storage.rename(&tmp_path, &schema_path).await?;
        storage.sync(&format!("{}/_schemas", table)).await?;
        Ok(())
    }

    /// Atomically write `_schema.json` to point to `schema_id`.
    async fn write_schema_pointer(
        storage: &dyn Storage,
        table: &str,
        schema_id: u64,
    ) -> Result<()> {
        let pointer_path = format!("{}/_schema.json", table);
        let pointer_tmp = format!("{}.tmp", pointer_path);
        let pointer = serde_json::json!({"latest": schema_id});
        storage
            .write(&pointer_tmp, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        storage.sync(&pointer_tmp).await?;
        storage.rename(&pointer_tmp, &pointer_path).await?;
        storage.sync(&format!("{}/", table)).await?;
        Ok(())
    }

    /// Rewrite the table's row groups according to the configured targets.
    ///
    /// The old row group files are left in place for garbage collection or
    /// repair tools; only the manifest pointer is advanced to the new snapshot.
    ///
    /// # Concurrency (locked-commit / optimistic compaction)
    ///
    /// Compaction is *optimistic*. The expensive rewrite (reading live rows,
    /// applying deletion vectors, building compacted fragments and the
    /// `_rowindex` base) runs **without** holding `_write.lock`, against a
    /// pinned source snapshot sequence `S`. The writer lock is acquired only
    /// for a brief commit that **revalidates** the source snapshot is still
    /// current: it re-reads the manifest pointer sequence `S'` and
    ///
    /// - if `S' == S`, publishes the compacted manifest via the normal atomic
    ///   pointer swap;
    /// - if `S' != S` (a mutation committed during the heavy phase), it
    ///   **aborts** this round — the staged compaction files are cleaned up and
    ///   the intervening mutation is left untouched — then retries the whole
    ///   compaction against the fresh snapshot a bounded number of times.
    ///
    /// The invariant is that a mutation committing during compaction is never
    /// lost or overwritten. After [`MAX_COMMIT_ATTEMPTS`] conflicting rounds the
    /// compactor gives up cleanly and returns `rewrote: false`.
    pub async fn compact(&self) -> Result<CompactionResult> {
        self.compact_with_hook(|_| async {}).await
    }

    /// Like [`Compactor::compact`], but invokes `before_commit` after the
    /// lock-free heavy phase pins source sequence `S` and stages the compacted
    /// output, but *before* the brief locked commit revalidates `S`.
    ///
    /// This is the deterministic seam used to exercise the revalidation path: a
    /// test can commit a real mutation inside `before_commit` (advancing the
    /// pointer to `S+1`) and then assert that the locked commit observes the
    /// conflict and does not clobber it. The pinned sequence `S` is passed to
    /// the hook. In production `compact` passes a no-op hook.
    pub async fn compact_with_hook<F, Fut>(&self, mut before_commit: F) -> Result<CompactionResult>
    where
        F: FnMut(u64) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        for _ in 0..MAX_COMMIT_ATTEMPTS {
            match self.compact_attempt(&mut before_commit).await? {
                AttemptOutcome::Committed(result) => return Ok(result),
                AttemptOutcome::NothingToDo(result) => return Ok(result),
                // The fold produced no output (e.g. every input row was
                // deleted). The snapshot it committed against was valid, so this
                // is a terminal "nothing to rewrite" result, not a conflict.
                AttemptOutcome::Empty(result) => return Ok(result),
                AttemptOutcome::Conflict => {
                    // A mutation landed during the heavy phase; the staged work
                    // has already been cleaned up. Retry against the fresh
                    // snapshot.
                    continue;
                }
            }
        }

        // Exhausted the retry budget: give up cleanly without touching the
        // committed state and report a conservative "nothing rewritten" result.
        Ok(CompactionResult {
            input_row_groups: 0,
            output_row_groups: 0,
            input_rows: 0,
            output_rows: 0,
            rewrote: false,
        })
    }

    /// Perform a single optimistic compaction attempt: pin a source sequence,
    /// run the heavy rewrite lock-free, then acquire the writer lock for a brief
    /// revalidating commit.
    async fn compact_attempt<F, Fut>(&self, before_commit: &mut F) -> Result<AttemptOutcome>
    where
        F: FnMut(u64) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        // ── Brief lock #1: pin the source snapshot and recover stale staging ──
        //
        // We hold the writer lock only long enough to read a consistent source
        // snapshot and run crash recovery (`cleanup_staging`), then release it
        // so concurrent mutations can commit during the heavy phase below.
        let lock_path = format!("{}/_write.lock", self.table);
        let (manifest, mut schema, pinned_seq) = {
            let _lock_guard = self
                .storage
                .lock_exclusive(&lock_path, self.options.lock_timeout)
                .await?;

            // Fold any pending mutation WAL into the pointer before compacting so
            // the rewrite folds the deferred deletions and references a complete
            // manifest. No-op when no `_wal/` log exists (the default).
            crate::mutation_wal::checkpoint_locked(self.storage, &self.table).await?;

            let catalog = self.load_catalog_with_repair().await?;
            let Some(manifest) = catalog.latest_manifest() else {
                // Empty table: nothing to compact.
                return Ok(AttemptOutcome::NothingToDo(CompactionResult {
                    input_row_groups: 0,
                    output_row_groups: 0,
                    input_rows: 0,
                    output_rows: 0,
                    rewrote: false,
                }));
            };
            let manifest = manifest.clone();
            let schema = catalog
                .latest_schema()
                .ok_or_else(|| IcefallDBError::ManifestNotFound(self.table.clone()))?
                .clone();

            // The lock is held, so any leftover intents, staged files, or
            // uncommitted manifest snapshots are from a crashed commit or
            // compaction and can be safely recovered before we pin the source.
            let latest_seq = manifest.sequence;
            let mut referenced_files: HashSet<String> = manifest
                .row_groups
                .iter()
                .flat_map(|e| [e.data.clone(), e.meta.clone()])
                .collect();
            for e in &manifest.row_groups {
                if let Some(agg) = &e.agg {
                    referenced_files.insert(agg.clone());
                }
            }
            cleanup_staging(self.storage, &self.table, latest_seq, &referenced_files).await?;

            let pinned_seq = latest_seq;
            (manifest, schema, pinned_seq)
        };
        // Lock released here: the heavy phase runs lock-free against `pinned_seq`.

        let manifest = &manifest;
        let previous_schema_id = schema.schema_id;

        // Ensure the compaction staging directory exists explicitly.
        ensure_dir(self.storage, &format!("{}/_staging/compact", self.table)).await?;

        let reader = Reader::new(self.storage, &self.table).await?;
        let scan = reader.scan().await?;
        let input_row_groups = scan.row_groups.len();

        if input_row_groups <= 1 && !self.options.force {
            let input_rows: u64 = scan.row_groups.iter().map(|rg| rg.meta.rows as u64).sum();
            return Ok(AttemptOutcome::NothingToDo(CompactionResult {
                input_row_groups,
                output_row_groups: input_row_groups,
                input_rows,
                output_rows: input_rows,
                rewrote: false,
            }));
        }

        // Resolve row-group targets. Explicit options take precedence; when the
        // compactor was created with default options, fall back to the schema's
        // configured targets.
        let default_options = CompactionOptions::default();
        let mut target_row_group_rows = self.options.target_row_group_rows;
        let mut target_row_group_bytes = self.options.target_row_group_bytes;
        if target_row_group_rows == default_options.target_row_group_rows
            && schema.row_group_target_rows > 0
        {
            target_row_group_rows = schema.row_group_target_rows;
        }
        if target_row_group_bytes == default_options.target_row_group_bytes
            && schema.row_group_target_bytes > 0
        {
            target_row_group_bytes = schema.row_group_target_bytes;
        }

        // Validate sort keys and prepare a new schema snapshot in memory if
        // needed. The schema file and pointer are not written yet; they are
        // committed atomically with the manifest pointer inside the commit
        // block below.
        let new_schema_id = Self::prepare_schema_for_sort(&mut schema, &self.options.sort_keys)?;
        let arrow_schema = arrow_schema_from_icefalldb(&schema, &self.table)?;

        // State tracked so a failure before the manifest pointer update can be
        // rolled back without leaking staged files or stale intents.
        let mut intent_path = String::new();
        let mut manifest_sequence = 0u64;
        let mut manifest_pointer_updated = false;
        let mut input_rows: u64 = 0;
        let mut output_row_groups: usize = 0;
        let mut state = StagingState {
            staged_files: Vec::new(),
            intent_files: Vec::new(),
            new_entries: Vec::new(),
            output_rows: 0,
            output_row_counts: Vec::new(),
            partition_values: None,
            used_rg_ids: HashSet::new(),
            next_fragment_id: manifest.next_fragment_id,
        };

        let commit_result: Result<bool> = async {
            ensure_dir(self.storage, &format!("{}/_staging/intents", self.table)).await?;

            // Write the commit intent before staging any files. With streaming
            // output we do not know the final filenames in advance, so the
            // files list is updated incrementally as each row group is staged.
            let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
            intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
            let started_at = chrono::Utc::now().to_rfc3339();
            let intent_ctx = IntentContext {
                path: &intent_path,
                txn_id: &txn_id,
                started_at: &started_at,
            };
            let intent_doc = serde_json::json!({
                "txn_id": txn_id,
                "started_at": started_at,
                "schema_id": schema.schema_id,
                "files": state.intent_files,
            });
            self.storage
                .write(&intent_path, serde_json::to_vec(&intent_doc)?.as_slice())
                .await?;
            self.storage.sync(&intent_path).await?;
            self.storage
                .sync(&format!("{}/_staging/intents", self.table))
                .await?;

            // Group input row groups by their partition values so rows from
            // different partitions are never merged into a single output row
            // group with incorrect partition metadata. Heterogeneous input row
            // groups (no partition values) are compacted together and their
            // output row groups omit partition values.
            let mut groups: std::collections::HashMap<String, Vec<&crate::PlannedRowGroup>> =
                std::collections::HashMap::new();
            for prg in &scan.row_groups {
                let key = partition_values_key(&prg.partition_values);
                groups.entry(key).or_default().push(prg);
            }
            let mut group_keys: Vec<String> = groups.keys().cloned().collect();
            group_keys.sort_unstable();

            for key in group_keys {
                let prgs = groups.remove(&key).expect("key exists");
                let partition_values = prgs[0].partition_values.clone();

                // ── Phase 1: collect all live rows across all source fragments ──
                //
                // We collect into a flat (batch, ids) list so we can optionally
                // sort by row_id before feeding into the streaming builder.  For
                // the non-reclustering path this is required to ensure the output
                // rows are in row-id order (patch fragments may have out-of-order
                // row IDs relative to base fragments).
                let mut all_live_batches: Vec<RecordBatch> = Vec::new();
                let mut all_live_ids: Vec<u64> = Vec::new();

                for prg in prgs {
                    // Load the deletion vector for this fragment (if any).
                    let dv: Option<DeletionVector> = if let Some(ref del_path) = prg.deletes {
                        let bytes = self.storage.read(del_path).await.map_err(|e| {
                            IcefallDBError::Other(Box::new(std::io::Error::other(format!(
                                "compaction: failed to read deletion vector {del_path}: {e}"
                            ))))
                        })?;
                        Some(
                            DeletionVector::deserialize(&bytes)
                                .map_err(|e| IcefallDBError::Other(Box::new(e)))?,
                        )
                    } else {
                        None
                    };

                    // Build a flat ordered list of all row IDs in this fragment.
                    // Physical offset i → row_id from prg.meta.row_ids.
                    let frag_ids: Vec<u64> =
                        prg.meta.row_ids.iter().flat_map(segment_ids).collect();

                    // Track the physical offset within this fragment as we stream
                    // batches; needed to index into frag_ids and the DV.
                    let mut phys_offset: u32 = 0;

                    let mut stream = reader.read_row_group(prg).await?;
                    while let Some(batch) = stream.next().await {
                        let batch = batch?;
                        let batch_len = batch.num_rows() as u32;

                        // Slice out this batch's row IDs.
                        let start = phys_offset as usize;
                        let end = (phys_offset + batch_len) as usize;
                        let batch_ids: Vec<u64> = if frag_ids.is_empty() {
                            // Legacy fragment with no row IDs: emit empty list so
                            // output meta row_ids stays empty (acceptable for
                            // legacy data that was never row-ID annotated).
                            vec![]
                        } else {
                            frag_ids[start..end.min(frag_ids.len())].to_vec()
                        };

                        // Apply the deletion vector to obtain live rows only.
                        let (live_batch, live_ids) =
                            filter_live_rows(&batch, dv.as_ref(), &batch_ids, phys_offset);

                        phys_offset += batch_len;

                        // Count ALL physical rows (including deleted) toward input.
                        input_rows += batch.num_rows() as u64;

                        if live_batch.num_rows() == 0 {
                            continue;
                        }

                        all_live_batches.push(live_batch);
                        all_live_ids.extend(live_ids);
                    }
                }

                if all_live_batches.is_empty() {
                    continue;
                }

                // ── Phase 2: for non-recluster, sort by row_id order ─────────
                //
                // When sort_keys is empty (recluster=false) we must output rows
                // in ascending row_id order so the compacted fragment can be
                // represented as dense Range segments.  Patch fragments may
                // interleave row_ids from the middle of base-fragment ranges,
                // so a simple per-fragment append does not guarantee order.
                let (sorted_batches, sorted_ids): (Vec<RecordBatch>, Vec<u64>) =
                    if self.options.sort_keys.is_empty() && !all_live_ids.is_empty() {
                        // Build a sort permutation by row_id.
                        let mut perm: Vec<usize> = (0..all_live_ids.len()).collect();
                        perm.sort_unstable_by_key(|&i| all_live_ids[i]);

                        // Check whether the ids are already sorted (common case:
                        // no patch fragments or patches are at the tail).
                        let already_sorted =
                            perm.iter().enumerate().all(|(pos, &orig)| pos == orig);

                        if already_sorted {
                            (all_live_batches, all_live_ids)
                        } else {
                            // Concatenate all live batches into one, then apply
                            // the permutation via Arrow's `take` kernel.
                            let concat =
                                arrow::compute::concat_batches(&arrow_schema, &all_live_batches)
                                    .map_err(other)?;

                            let indices: arrow::array::UInt64Array =
                                perm.iter().map(|&i| i as u64).collect();
                            let reordered_cols: Vec<_> = concat
                                .columns()
                                .iter()
                                .map(|c| {
                                    arrow::compute::take(c.as_ref(), &indices, None)
                                        .map_err(|e| IcefallDBError::Other(Box::new(e)))
                                })
                                .collect::<Result<_>>()?;
                            let reordered =
                                RecordBatch::try_new(arrow_schema.clone(), reordered_cols)
                                    .map_err(other)?;

                            let sorted_id_vec: Vec<u64> =
                                perm.iter().map(|&i| all_live_ids[i]).collect();
                            (vec![reordered], sorted_id_vec)
                        }
                    } else {
                        (all_live_batches, all_live_ids)
                    };

                // ── Phase 3: feed sorted live rows through the streaming builder
                //
                // `sorted_ids` is the flat list of row IDs in the order that
                // rows will be written.  `RowIdAccumulator` consumes this list
                // sequentially as the builder flushes chunks of `n` rows.
                let mut builder = StreamingRowGroupBuilder::new(
                    target_row_group_rows,
                    target_row_group_bytes,
                    arrow_schema.clone(),
                );
                let mut id_acc = RowIdAccumulator::new();
                // Pre-load all sorted IDs; take() consumes them in chunk order.
                id_acc.push(sorted_ids);

                for live_batch in sorted_batches {
                    let flushed_batches = builder.add(live_batch)?;
                    for chunk in flushed_batches {
                        let cn = chunk.num_rows();
                        let chunk_ids = id_acc.take(cn);
                        self.stage_chunk(
                            &arrow_schema,
                            &schema,
                            chunk,
                            chunk_ids,
                            &mut state,
                            &intent_ctx,
                            partition_values.as_ref(),
                        )
                        .await?;
                        output_row_groups += 1;
                    }
                    if builder.is_full() {
                        if let Some(batch) = builder.flush()? {
                            let bn = batch.num_rows();
                            let batch_ids = id_acc.take(bn);
                            self.stage_chunk(
                                &arrow_schema,
                                &schema,
                                batch,
                                batch_ids,
                                &mut state,
                                &intent_ctx,
                                partition_values.as_ref(),
                            )
                            .await?;
                            output_row_groups += 1;
                        }
                    }
                }

                // Flush any remaining rows for this partition.
                if let Some(batch) = builder.flush()? {
                    let bn = batch.num_rows();
                    let batch_ids = id_acc.take(bn);
                    self.stage_chunk(
                        &arrow_schema,
                        &schema,
                        batch,
                        batch_ids,
                        &mut state,
                        &intent_ctx,
                        partition_values.as_ref(),
                    )
                    .await?;
                    output_row_groups += 1;
                }
            }

            if input_rows == 0 {
                return Ok(false);
            }

            // ── Deterministic seam: let a test interleave a mutation here ─────
            //
            // Everything above ran lock-free against the pinned source snapshot
            // `pinned_seq`. The hook fires after the heavy rewrite is fully
            // staged but before we acquire the writer lock to commit, so a test
            // can commit a real mutation (advancing the pointer to S+1) and
            // prove the revalidation below refuses to clobber it.
            before_commit(pinned_seq).await;

            // ── Brief lock #2: revalidate the source snapshot, then commit ────
            //
            // Acquire the writer lock only for the brief atomic commit. Re-read
            // the manifest pointer sequence under the lock: if it advanced past
            // `pinned_seq`, a mutation landed during the heavy phase and our
            // staged fragments are built from a stale snapshot. Abort rather
            // than clobber the mutation; the outer loop retries against the
            // fresh snapshot. The lock guard lives until the end of this block,
            // covering the manifest write and the pointer swap.
            let _commit_lock = self
                .storage
                .lock_exclusive(&lock_path, self.options.lock_timeout)
                .await?;
            let current_seq = read_pointer_sequence(self.storage, &self.table).await?;
            if current_seq != Some(pinned_seq) {
                // Conflict: a mutation committed at `current_seq != pinned_seq`.
                // Signal the caller to clean up staged work and retry. The
                // staged files and intent are removed by `rollback_compaction`
                // (manifest pointer was NOT advanced).
                return Err(IcefallDBError::CompactionConflict {
                    pinned: pinned_seq,
                    current: current_seq.unwrap_or(0),
                });
            }

            // The compaction will produce a new manifest, so persist the new
            // schema snapshot now. The pointer is advanced below together with
            // the manifest pointer.
            if let Some(id) = new_schema_id {
                // Defensive: ensure the in-memory schema id matches what we
                // advertised during preparation.
                assert_eq!(schema.schema_id, id);
                Self::write_schema_snapshot(self.storage, &self.table, &schema).await?;
            }

            let next_seq = manifest.sequence + 1;
            let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
            if self.storage.exists(&manifest_path).await? {
                return Err(IcefallDBError::SequenceCollision(next_seq));
            }
            manifest_sequence = next_seq;

            let mut new_manifest = Manifest {
                format_version: 1,
                sequence: next_seq,
                schema_id: schema.schema_id,
                row_groups: std::mem::take(&mut state.new_entries),
                row_counts: Some(std::mem::take(&mut state.output_row_counts)),
                partition_values: state.partition_values.take(),
                // Carry forward the row-id and fragment-id counters so that
                // subsequent writes after compaction never reuse IDs.
                next_row_id: manifest.next_row_id,
                next_fragment_id: state.next_fragment_id,
                // Carry forward secondary index generations (will rebuild
                // those; for now preserve the pre-compaction refs).
                index_generations: manifest.index_generations.clone(),
                checksum: String::new(),
                ..Default::default()
            };

            // ── Rebuild the _rowindex base from the compacted manifest ────────
            // The new fragments have dense Range row_id segments and no deletion
            // vectors, so derive_base produces a minimal, correct address map.
            // We ensure the _rowindex directory exists, write the base file, add
            // it to both the intent journal (so recovery can clean up on crash)
            // and staged_files (so rollback_compaction removes it on error).
            ensure_dir(self.storage, &format!("{}/_rowindex", self.table)).await?;
            let fresh_gen = rebuild_rowindex(self.storage, &self.table, &new_manifest).await?;
            // Record the base path in the intent journal BEFORE touching the
            // manifest so a crash between the base write and the manifest commit
            // leaves the file listed for cleanup.
            if let Some(ref base_rel) = fresh_gen.base {
                state.intent_files.push(base_rel.clone());
                let intent_doc = serde_json::json!({
                    "txn_id": intent_ctx.txn_id,
                    "started_at": intent_ctx.started_at,
                    "schema_id": schema.schema_id,
                    "files": state.intent_files,
                });
                self.storage
                    .write(intent_ctx.path, serde_json::to_vec(&intent_doc)?.as_slice())
                    .await?;
                self.storage.sync(intent_ctx.path).await?;
                self.storage
                    .sync(&format!("{}/_staging/intents", self.table))
                    .await?;
                // Also add to staged_files so rollback_compaction removes it if
                // the commit fails after this point.
                state.staged_files.push(base_rel.clone());
            }
            new_manifest.rowindex_generation = Some(fresh_gen);

            // Emit the snapshot checkpoint inside the atomic commit.
            let checkpoint =
                build_snapshot_checkpoint(self.storage, &self.table, manifest, &new_manifest)
                    .await?;
            let checkpoint_path = write_checkpoint(
                self.storage,
                &self.table,
                &checkpoint,
                next_seq,
                intent_ctx.path,
                &mut state.intent_files,
                &mut state.staged_files,
                intent_ctx.txn_id,
                schema.schema_id,
            )
            .await?;
            new_manifest.checkpoint = Some(checkpoint_path);

            new_manifest.checksum = new_manifest.compute_checksum()?;

            // Write the new manifest snapshot atomically.
            let manifest_data = serde_json::to_vec(&new_manifest)?;
            let manifest_tmp_path = format!("{}.tmp", manifest_path);
            self.storage
                .write(&manifest_tmp_path, &manifest_data)
                .await?;
            self.storage.sync(&manifest_tmp_path).await?;
            self.storage
                .sync(&format!("{}/_manifests", self.table))
                .await?;

            // Read the manifest back and verify its checksum before committing
            // it. Retry once by recomputing and rewriting the checksum if the
            // first verification fails.
            if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
                self.storage
                    .write(&manifest_tmp_path, &manifest_data)
                    .await?;
                self.storage.sync(&manifest_tmp_path).await?;
                if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
                    return Err(IcefallDBError::ChecksumMismatch {
                        path: manifest_tmp_path,
                    });
                }
            }

            self.storage
                .rename(&manifest_tmp_path, &manifest_path)
                .await?;
            self.storage
                .sync(&format!("{}/_manifests", self.table))
                .await?;

            // Advance `_schema.json` first, then `_manifest.json`. If the
            // manifest pointer update fails, rollback will restore the schema
            // pointer. Once the manifest pointer is durable, the compaction is
            // committed and the schema pointer must not be rolled back.
            if let Some(id) = new_schema_id {
                Self::write_schema_pointer(self.storage, &self.table, id).await?;
            }

            // Update the manifest pointer.
            let pointer_path = format!("{}/_manifest.json", self.table);
            let pointer_tmp_path = format!("{}.tmp", pointer_path);
            let pointer = serde_json::json!({"latest": next_seq});
            self.storage
                .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
                .await?;
            self.storage.sync(&pointer_tmp_path).await?;
            self.storage
                .rename(&pointer_tmp_path, &pointer_path)
                .await?;
            manifest_pointer_updated = true;
            self.storage.sync(&format!("{}/", self.table)).await?;

            Ok(true)
        }
        .await;

        let committed = match commit_result {
            Ok(committed) => committed,
            Err(IcefallDBError::CompactionConflict { .. }) => {
                // Optimistic-commit conflict: a mutation landed during the heavy
                // phase. Discard the staged compaction work (the manifest
                // pointer was never advanced) and tell the caller to retry.
                self.rollback_compaction(
                    &intent_path,
                    &state.staged_files,
                    manifest_sequence,
                    manifest_pointer_updated,
                    previous_schema_id,
                    new_schema_id,
                )
                .await;
                return Ok(AttemptOutcome::Conflict);
            }
            Err(e) => {
                self.rollback_compaction(
                    &intent_path,
                    &state.staged_files,
                    manifest_sequence,
                    manifest_pointer_updated,
                    previous_schema_id,
                    new_schema_id,
                )
                .await;
                return Err(e);
            }
        };

        // The compaction is durable; remove the intent best-effort.
        if !intent_path.is_empty() {
            let _ = self.storage.delete(&intent_path).await;
            let _ = self
                .storage
                .sync(&format!("{}/_staging/intents", self.table))
                .await;
        }

        if !committed {
            return Ok(AttemptOutcome::Empty(CompactionResult {
                input_row_groups,
                output_row_groups: input_row_groups,
                input_rows: 0,
                output_rows: 0,
                rewrote: false,
            }));
        }

        Ok(AttemptOutcome::Committed(CompactionResult {
            input_row_groups,
            output_row_groups,
            input_rows,
            output_rows: state.output_rows,
            rewrote: true,
        }))
    }

    /// Load the catalog, self-repairing a missing or corrupt manifest pointer
    /// from valid snapshots in `_manifests/` when possible.
    async fn load_catalog_with_repair(&self) -> Result<Catalog<'_>> {
        match Catalog::load(self.storage, &self.table).await {
            Ok(catalog) if catalog.latest_manifest().is_some() => Ok(catalog),
            Ok(catalog) => {
                // Pointer missing or table is empty. Attempt repair from snapshots.
                if repair_manifest_pointer(self.storage, &self.table)
                    .await?
                    .is_some()
                {
                    Catalog::load(self.storage, &self.table).await
                } else {
                    Ok(catalog)
                }
            }
            Err(_) => {
                // Pointer corrupt or manifest missing/corrupt. Attempt repair.
                if repair_manifest_pointer(self.storage, &self.table)
                    .await?
                    .is_some()
                {
                    Catalog::load(self.storage, &self.table).await
                } else {
                    Err(IcefallDBError::Other(Box::new(std::io::Error::other(
                        "no valid manifest snapshots",
                    ))))
                }
            }
        }
    }

    /// Stage a single flushed output chunk: write the row group, record the
    /// final filenames, update the commit intent, and accumulate partition
    /// values and result metadata.
    ///
    /// When `partition_values` is supplied, it is attached to the output row
    /// group directly. This lets compaction keep row groups from different
    /// partitions separate instead of inferring (and possibly mis-stating)
    /// partition values from a merged batch.
    ///
    /// `chunk_row_ids` carries the stable row IDs (in row order) for every row
    /// in `batch`.  These are threaded through to the `.meta` sidecar so that
    /// compacted fragments remain move-stable.
    #[allow(clippy::too_many_arguments)]
    async fn stage_chunk(
        &self,
        arrow_schema: &SchemaRef,
        schema: &Schema,
        batch: RecordBatch,
        chunk_row_ids: Vec<u64>,
        state: &mut StagingState,
        intent: &IntentContext<'_>,
        partition_values: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        let batch = maybe_sort_batch(&batch, &self.options.sort_keys)?;
        let row_id_segs = ids_to_segments(&chunk_row_ids);
        let rg_id = unique_rg_id(&mut state.used_rg_ids);
        let fragment_id = state.next_fragment_id;
        state.next_fragment_id += 1;
        let chunk = PlannedChunk {
            rg_id: rg_id.clone(),
            data_filename: format!("{}.parquet", rg_id),
            meta_filename: format!("{}.meta", rg_id),
            batch,
            row_ids: row_id_segs,
        };
        let props = build_writer_properties(arrow_schema, std::slice::from_ref(&chunk.batch));

        // Record the planned final filenames in the commit intent *before* the
        // final rename. If the process crashes after the rename, recovery can
        // still find and clean up these files.
        //
        // The intent is rewritten and fsynced once per output row group for
        // crash safety; batching intent updates is a future optimization.
        state.intent_files.push(chunk.data_filename.clone());
        state.intent_files.push(chunk.meta_filename.clone());
        state.intent_files.push(format!("{}.agg", rg_id));
        let intent_doc = serde_json::json!({
            "txn_id": intent.txn_id,
            "started_at": intent.started_at,
            "schema_id": schema.schema_id,
            "files": state.intent_files,
        });
        self.storage
            .write(intent.path, serde_json::to_vec(&intent_doc)?.as_slice())
            .await?;
        self.storage.sync(intent.path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;

        let entry = self
            .write_row_group(
                arrow_schema,
                schema,
                &chunk,
                &mut state.used_rg_ids,
                props,
                fragment_id,
            )
            .await?;
        state.staged_files.push(entry.data.clone());
        state.staged_files.push(entry.meta.clone());
        if let Some(agg) = &entry.agg {
            state.staged_files.push(agg.clone());
        }

        // Collision handling may have chosen different final filenames. Rewrite
        // the intent with the corrected names so recovery matches reality.
        if entry.data != chunk.data_filename || entry.meta != chunk.meta_filename {
            let n = state.intent_files.len();
            state.intent_files[n - 3] = entry.data.clone();
            state.intent_files[n - 2] = entry.meta.clone();
            if let Some(agg) = &entry.agg {
                state.intent_files[n - 1] = agg.clone();
            }
            let intent_doc = serde_json::json!({
                "txn_id": intent.txn_id,
                "started_at": intent.started_at,
                "schema_id": schema.schema_id,
                "files": state.intent_files,
            });
            self.storage
                .write(intent.path, serde_json::to_vec(&intent_doc)?.as_slice())
                .await?;
            self.storage.sync(intent.path).await?;
            self.storage
                .sync(&format!("{}/_staging/intents", self.table))
                .await?;
        }

        state.output_rows += chunk.batch.num_rows() as u64;
        state.output_row_counts.push(chunk.batch.num_rows());
        if let Some(pv) = partition_values {
            state
                .partition_values
                .get_or_insert_with(HashMap::new)
                .insert(entry.data.clone(), pv.clone());
        }
        state.new_entries.push(entry);
        Ok(())
    }

    /// Write a planned chunk to staging, verify its checksums, and rename it to
    /// its final table-root filename.
    async fn write_row_group(
        &self,
        arrow_schema: &SchemaRef,
        schema: &Schema,
        chunk: &PlannedChunk,
        used_rg_ids: &mut HashSet<String>,
        props: WriterProperties,
        fragment_id: u64,
    ) -> Result<RowGroupEntry> {
        let mut rg_id = chunk.rg_id.clone();
        let mut data_filename = chunk.data_filename.clone();
        let mut meta_filename = chunk.meta_filename.clone();

        for attempt in 0..3 {
            let parquet_part = format!("{}/_staging/compact/{}.parquet.part", self.table, rg_id);
            let meta_part = format!("{}/_staging/compact/{}.meta.part", self.table, rg_id);
            let parquet_final = format!("{}/{}", self.table, data_filename);
            let meta_final = format!("{}/{}", self.table, meta_filename);

            self.write_row_group_part(
                &parquet_part,
                &meta_part,
                arrow_schema,
                schema,
                &chunk.batch,
                &rg_id,
                &chunk.row_ids,
                props.clone(),
                fragment_id,
            )
            .await?;
            self.storage
                .sync(&format!("{}/_staging/compact", self.table))
                .await?;

            if !self
                .verify_row_group_checksum(&parquet_part, &meta_part)
                .await?
            {
                // Retry once after a checksum mismatch.
                self.write_row_group_part(
                    &parquet_part,
                    &meta_part,
                    arrow_schema,
                    schema,
                    &chunk.batch,
                    &rg_id,
                    &chunk.row_ids,
                    props.clone(),
                    fragment_id,
                )
                .await?;
                self.storage
                    .sync(&format!("{}/_staging/compact", self.table))
                    .await?;
                if !self
                    .verify_row_group_checksum(&parquet_part, &meta_part)
                    .await?
                {
                    return Err(IcefallDBError::ChecksumMismatch { path: parquet_part });
                }
            }

            if self.storage.exists(&parquet_final).await?
                || self.storage.exists(&meta_final).await?
            {
                if attempt == 2 {
                    return Err(IcefallDBError::Other(Box::new(std::io::Error::other(
                        "failed to find a unique row group id after retries",
                    ))));
                }
                let agg_part_stale = format!("{}/_staging/compact/{}.agg.part", self.table, rg_id);
                let _ = self.storage.delete(&parquet_part).await;
                let _ = self.storage.delete(&meta_part).await;
                let _ = self.storage.delete(&agg_part_stale).await;
                rg_id = unique_rg_id(used_rg_ids);
                data_filename = format!("{}.parquet", rg_id);
                meta_filename = format!("{}.meta", rg_id);
                continue;
            }

            let agg_filename = format!("{}.agg", rg_id);
            let agg_part = format!("{}/_staging/compact/{}.agg.part", self.table, rg_id);
            let agg_final = format!("{}/{}", self.table, agg_filename);
            self.storage.rename(&parquet_part, &parquet_final).await?;
            self.storage.rename(&meta_part, &meta_final).await?;
            self.storage.rename(&agg_part, &agg_final).await?;
            self.storage
                .sync(&format!("{}/_staging/compact", self.table))
                .await?;
            self.storage.sync(&format!("{}/", self.table)).await?;

            return Ok(RowGroupEntry {
                data: data_filename,
                meta: meta_filename,
                fragment_id,
                agg: Some(agg_filename),
                ..Default::default()
            });
        }

        Err(IcefallDBError::Other(Box::new(std::io::Error::other(
            "failed to find a unique row group id after retries",
        ))))
    }

    /// Write Parquet bytes and metadata for a chunk to the staging area.
    #[allow(clippy::too_many_arguments)]
    async fn write_row_group_part(
        &self,
        parquet_part_path: &str,
        meta_part_path: &str,
        arrow_schema: &SchemaRef,
        schema: &Schema,
        batch: &RecordBatch,
        rg_id: &str,
        row_ids: &[RowIdSegment],
        props: WriterProperties,
        fragment_id: u64,
    ) -> Result<()> {
        let mut parquet_bytes = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut parquet_bytes, arrow_schema.clone(), Some(props))
                    .map_err(other)?;
            writer.write(batch).map_err(other)?;
            writer.close().map_err(other)?;
        }
        self.storage
            .write(parquet_part_path, &parquet_bytes)
            .await?;

        let meta = compute_row_group_meta(
            rg_id,
            schema.schema_id,
            batch,
            schema,
            &parquet_bytes,
            &self.table,
            row_ids,
        )?;
        let meta_json = serde_json::to_vec(&meta)?;
        self.storage.write(meta_part_path, &meta_json).await?;

        self.storage.sync(parquet_part_path).await?;
        self.storage.sync(meta_part_path).await?;

        // Compute additive aggregate partials and write `.agg.part`.
        // content_hash is meta.checksum — valid forever because fragments are write-once.
        let key_col = schema
            .agg_group_keys
            .as_deref()
            .and_then(|v| v.first())
            .map(|s| s.as_str());
        let agg_state =
            compute_agg_state_with_key(fragment_id, meta.checksum.clone(), batch, key_col)?;
        let agg_bytes = serialize_agg_state(&agg_state)?;
        let agg_part_path = format!("{}/_staging/compact/{}.agg.part", self.table, rg_id);
        self.storage.write(&agg_part_path, &agg_bytes).await?;
        self.storage.sync(&agg_part_path).await?;

        Ok(())
    }

    /// Verify that staged Parquet bytes match their metadata checksums.
    async fn verify_row_group_checksum(&self, parquet_path: &str, meta_path: &str) -> Result<bool> {
        let parquet_bytes = self.storage.read(parquet_path).await?;
        let meta_bytes = self.storage.read(meta_path).await?;
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
        Ok(meta.verify_against_data(&parquet_bytes) && meta.verify_meta_checksum()?)
    }

    /// Read a manifest `.tmp` file and verify that its stored checksum matches
    /// the recomputed checksum.
    async fn verify_manifest_checksum(&self, manifest_tmp_path: &str) -> Result<bool> {
        let read_data = self.storage.read(manifest_tmp_path).await?;
        let read_manifest: Manifest = serde_json::from_slice(&read_data)?;
        read_manifest.verify_checksum()
    }

    /// Clean up artifacts from a failed compaction. Errors are ignored because
    /// the compaction has already failed and the lock is still held.
    ///
    /// If the manifest pointer has already been updated, the compaction is
    /// durable and rollback must only remove the intent file. Deleting the
    /// manifest snapshot or data files would destroy committed state.
    ///
    /// If the schema was bumped but the manifest pointer was not committed,
    /// `_schema.json` is restored to the previous schema id.
    async fn rollback_compaction(
        &self,
        intent_path: &str,
        staged_files: &[String],
        manifest_sequence: u64,
        manifest_pointer_updated: bool,
        previous_schema_id: u64,
        new_schema_id: Option<u64>,
    ) {
        if manifest_pointer_updated {
            if !intent_path.is_empty() {
                let _ = self.storage.delete(intent_path).await;
            }
            return;
        }

        // Revert an advanced but uncommitted schema pointer before cleaning up
        // the files it references.
        if new_schema_id.is_some() {
            let _ = Self::write_schema_pointer(self.storage, &self.table, previous_schema_id).await;
        }

        // Delete staged final files first so they are cleaned up even if the
        // intent deletion fails. Each deletion is best-effort and independent.
        for file in staged_files {
            let _ = self
                .storage
                .delete(&format!("{}/{}", self.table, file))
                .await;
        }

        // Remove the uncommitted manifest snapshot so a retry does not collide.
        if manifest_sequence > 0 {
            let _ = self
                .storage
                .delete(&format!(
                    "{}/{}",
                    self.table,
                    Manifest::filename(manifest_sequence)
                ))
                .await;
        }

        // Remove the uncommitted schema snapshot. The pointer has already been
        // restored above; deleting the file prevents orphaned snapshots.
        if let Some(id) = new_schema_id {
            let _ = self
                .storage
                .delete(&format!("{}/{}", self.table, Schema::filename(id)))
                .await;
        }

        if !intent_path.is_empty() {
            let _ = self.storage.delete(intent_path).await;
        }
    }
}

/// A planned output chunk held in memory before it is written.
struct PlannedChunk {
    rg_id: String,
    data_filename: String,
    meta_filename: String,
    batch: RecordBatch,
    /// Stable row IDs for each row in `batch`, in row order.
    row_ids: Vec<RowIdSegment>,
}

/// Tracks surviving row IDs for a compaction partition as batches flow through
/// [`StreamingRowGroupBuilder`].
///
/// Instead of mirroring the builder's complex byte/row slicing heuristics, the
/// accumulator keeps a flat ordered list of all live row IDs for the partition.
/// When the builder flushes a chunk of `n` rows, `take(n)` yields the
/// corresponding IDs — the alignment is guaranteed because both the builder and
/// the accumulator process the same (filtered) batch sequence in order.
struct RowIdAccumulator {
    /// All live row IDs for the current partition, in row order.
    all: Vec<u64>,
    /// Cursor into `all`: next ID to yield on the next `take` call.
    cursor: usize,
}

impl RowIdAccumulator {
    fn new() -> Self {
        Self {
            all: Vec::new(),
            cursor: 0,
        }
    }

    /// Append the IDs for the live rows in one input batch.
    fn push(&mut self, ids: Vec<u64>) {
        self.all.extend(ids);
    }

    /// Consume the next `n` IDs (corresponding to a flushed chunk of `n` rows).
    fn take(&mut self, n: usize) -> Vec<u64> {
        let end = (self.cursor + n).min(self.all.len());
        let ids = self.all[self.cursor..end].to_vec();
        self.cursor = end;
        ids
    }
}

/// Convert a flat **sorted** list of row IDs into a sequence of
/// [`RowIdSegment`]s.
///
/// Contiguous sub-sequences are each encoded as a `Range`.  When the IDs are
/// not sorted (e.g. after reclustering reorders rows by a sort key), they
/// cannot be split into `Range`s and fall back to a single `Sorted` segment.
///
/// This guarantees that the non-reclustering path (where IDs are kept in
/// ascending row-id order) produces only `Range` segments — satisfying the
/// gate test assertion `all(|s| matches!(s, RowIdSegment::Range { .. }))`.
fn ids_to_segments(ids: &[u64]) -> Vec<RowIdSegment> {
    if ids.is_empty() {
        return Vec::new();
    }

    // If IDs are not non-decreasing we cannot split into Range segments.
    // Fall back to a single Sorted segment (reclustering path).
    if ids.windows(2).any(|w| w[1] <= w[0]) {
        return vec![RowIdSegment::Sorted { ids: ids.to_vec() }];
    }

    // IDs are strictly increasing: partition into contiguous runs.
    let mut segments = Vec::new();
    let mut run_start = ids[0];
    let mut run_len: u64 = 1;
    for &id in &ids[1..] {
        if id == run_start + run_len {
            run_len += 1;
        } else {
            segments.push(RowIdSegment::Range {
                start: run_start,
                count: run_len,
            });
            run_start = id;
            run_len = 1;
        }
    }
    segments.push(RowIdSegment::Range {
        start: run_start,
        count: run_len,
    });
    segments
}

/// Filter `batch` to only live (non-deleted) rows using `dv`, and return the
/// corresponding surviving row IDs from `all_ids` (one per physical row).
///
/// When `dv` is `None` (no deletion vector), all rows survive.
fn filter_live_rows(
    batch: &RecordBatch,
    dv: Option<&DeletionVector>,
    all_ids: &[u64],
    physical_offset: u32,
) -> (RecordBatch, Vec<u64>) {
    let nrows = batch.num_rows();
    debug_assert_eq!(nrows, all_ids.len(), "batch rows must match id count");

    let Some(dv) = dv else {
        // No deletions: all rows survive.
        return (batch.clone(), all_ids.to_vec());
    };

    // Build a boolean mask: true = live row.
    let mask: BooleanArray = (0..nrows as u32)
        .map(|i| Some(!dv.contains(physical_offset + i)))
        .collect();

    // Filter the batch.
    let filtered_columns: Vec<_> = batch
        .columns()
        .iter()
        .map(|col| {
            arrow::compute::filter(col.as_ref(), &mask)
                .expect("filter is infallible for BooleanArray mask")
        })
        .collect();
    let filtered_batch =
        RecordBatch::try_new(batch.schema(), filtered_columns).expect("same schema after filter");

    // Collect surviving row IDs.
    let surviving_ids: Vec<u64> = (0..nrows)
        .filter(|&i| !dv.contains(physical_offset + i as u32))
        .map(|i| all_ids[i])
        .collect();

    (filtered_batch, surviving_ids)
}

fn arrow_schema_from_icefalldb(schema: &Schema, path: &str) -> Result<SchemaRef> {
    let arrow_schema = schema
        .arrow_schema()
        .ok_or_else(|| IcefallDBError::SchemaMismatch {
            column: "schema".into(),
            expected: "supported types".into(),
            path: path.into(),
        })?;
    Ok(Arc::new(arrow_schema))
}

/// Returns a full UUID with all dashes removed, suitable for filenames.
fn full_uuid() -> String {
    uuid::Uuid::new_v4().to_string().replace('-', "")
}

/// Generate a unique row-group id within this compaction.
fn unique_rg_id(used: &mut HashSet<String>) -> String {
    let rg_id = format!("rg_{}", full_uuid());
    used.insert(rg_id.clone());
    rg_id
}

/// Build a canonical key for a set of partition values so row groups with the
/// same partition tuple are compacted together.
fn partition_values_key(
    partition_values: &Option<std::collections::HashMap<String, serde_json::Value>>,
) -> String {
    match partition_values {
        None => "__none__".to_string(),
        Some(values) => {
            let sorted: std::collections::BTreeMap<_, _> = values.iter().collect();
            serde_json::to_string(&sorted).unwrap_or_default()
        }
    }
}

/// Ensure a directory exists by writing and removing a temporary file. This
/// works for storage backends that create parent directories on write.
async fn ensure_dir(storage: &dyn Storage, path: &str) -> Result<()> {
    match storage.list(path).await {
        Ok(_) => Ok(()),
        Err(IcefallDBError::NotFound(_)) => {
            let tmp = format!("{}/.keep", path);
            storage.write(&tmp, b"").await?;
            let _ = storage.delete(&tmp).await;
            Ok(())
        }
        Err(e) => Err(e),
    }
}
