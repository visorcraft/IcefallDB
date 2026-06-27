use crate::database_catalog::DatabaseCatalog;
use crate::deletion::DeletionVector;
use crate::metadata::manifest::IndexRef;
use crate::metadata::Manifest;
use crate::rowid::segment_ids;
use crate::rowindex::AddressMap;
use crate::storage::Storage;
use crate::{is_not_found, IcefallDBError, Result};
use arrow::array::{Array, AsArray};
use arrow::datatypes::DataType;
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Derived, optional binary postings format (mmap-friendly) written alongside
/// the canonical JSON index. See [`binary`] for the layout and fallback rules.
pub mod binary;

const INDEXES_DIR: &str = "_indexes";

fn other<E: std::error::Error + Send + Sync + 'static>(err: E) -> IcefallDBError {
    IcefallDBError::Other(Box::new(err))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub table: String,
    pub column: String,
    /// Whether this index enforces a uniqueness constraint.  `false` by default
    /// (backward-compatible with existing serialised `IndexDefinition`s).
    #[serde(default)]
    pub unique: bool,
}

/// An immutable B-tree index built for a single manifest snapshot.
///
/// `entries` maps each distinct value (serialised to a string key) to the list
/// of stable row IDs that carry that value in the snapshot the index was built
/// from.  Row IDs are stable across compaction/deletion, so a reader pinned to
/// an older snapshot can use its generation safely even after the data has been
/// rewritten.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeIndex {
    pub definition: IndexDefinition,
    pub snapshot_sequence: u64,
    /// Sorted value-key → list of stable row IDs.
    pub entries: BTreeMap<String, Vec<u64>>,
}

/// A tiny "learned" index for an exactly-affine, unique integer key column
/// When the sorted `(key → row_id)` pairs satisfy
/// `key_i = key_base + i·key_stride` and `row_id_i = rid_base + i·rid_stride`
/// (the auto-increment / contiguous case), the whole index collapses to these
/// few numbers: a point lookup is exact O(1) arithmetic with no postings to
/// store, load, or maintain. Non-affine / non-unique / non-integer columns
/// produce `None` from [`LearnedKeyModel::fit`] and fall back to the binary index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearnedKeyModel {
    pub key_base: i64,
    pub key_stride: i64,
    pub rid_base: u64,
    pub rid_stride: i64,
    pub count: u64,
}

impl LearnedKeyModel {
    /// Fit an exact affine model to an index's entries, or `None` if the column
    /// is not a unique integer key whose sorted keys AND row ids are both affine.
    pub fn fit(index: &BTreeIndex) -> Option<LearnedKeyModel> {
        // Parse to sorted (key:i64, row_id) pairs; every key must be a unique
        // integer mapping to exactly one row id. (BTreeMap iterates key-sorted,
        // but the *string* order is lexicographic, so re-sort numerically.)
        let mut pairs: Vec<(i64, u64)> = Vec::with_capacity(index.entries.len());
        for (k, ids) in &index.entries {
            if ids.len() != 1 {
                return None; // non-unique key
            }
            let key: i64 = k.parse().ok()?; // non-integer key
            pairs.push((key, ids[0]));
        }
        if pairs.len() < 2 {
            return None; // too small to be worth a model; use the index
        }
        pairs.sort_unstable_by_key(|(k, _)| *k);

        // The affine model works in the i64 domain; reject row ids outside it so
        // `lookup`'s i64 reconstruction can never disagree with the index. (All
        // checked arithmetic below treats overflow as "not affine" — `fit` runs
        // inside the commit, so it must never panic on a pathological key set.)
        if pairs.iter().any(|(_, r)| *r > i64::MAX as u64) {
            return None;
        }
        let key_base = pairs[0].0;
        let key_stride = pairs[1].0.checked_sub(pairs[0].0)?;
        let rid_base = pairs[0].1;
        let rid_stride = (pairs[1].1 as i64).checked_sub(pairs[0].1 as i64)?;
        if key_stride == 0 {
            return None; // duplicate keys (not unique/affine)
        }
        for (i, (k, r)) in pairs.iter().enumerate() {
            let i = i as i64;
            if *k != key_base.checked_add(i.checked_mul(key_stride)?)? {
                return None; // keys not affine
            }
            if *r as i64 != (rid_base as i64).checked_add(i.checked_mul(rid_stride)?)? {
                return None; // row ids not affine
            }
        }
        Some(LearnedKeyModel {
            key_base,
            key_stride,
            rid_base,
            rid_stride,
            count: pairs.len() as u64,
        })
    }

    /// Row ids that may contain `value` — exactly one if `value` is an integer on
    /// the model's grid and in range, else empty (matching [`BTreeIndex::lookup`]
    /// for a unique key).
    pub fn lookup(&self, value: &str) -> Vec<u64> {
        let Ok(k) = value.parse::<i64>() else {
            return Vec::new();
        };
        let (Some(off), true) = (k.checked_sub(self.key_base), self.key_stride != 0) else {
            return Vec::new();
        };
        if off % self.key_stride != 0 {
            return Vec::new();
        }
        let i = off / self.key_stride;
        if i < 0 || i as u64 >= self.count {
            return Vec::new();
        }
        let rid = match i
            .checked_mul(self.rid_stride)
            .and_then(|s| (self.rid_base as i64).checked_add(s))
        {
            Some(r) if r >= 0 => r as u64,
            _ => return Vec::new(),
        };
        vec![rid]
    }

    /// Versioned filename for the model sidecar at `seq`.
    pub fn versioned_filename(name: &str, seq: u64) -> String {
        format!("{}/{}/base__v{:09}.model", INDEXES_DIR, name, seq)
    }

    /// Legacy unversioned filename for the model sidecar.
    pub fn legacy_filename(name: &str) -> String {
        format!("{}/{}.model", INDEXES_DIR, name)
    }
}

/// Self-describing `.model` sidecar: the learned model plus the index definition
/// it belongs to (so an opener can match the predicate column without the full
/// index). Written next to the JSON base; a derived, rebuildable cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedIndexFile {
    pub definition: IndexDefinition,
    pub model: LearnedKeyModel,
}

/// Read the learned-model sidecar that sits next to the JSON index base at
/// `json_rel_path` (a table-relative `…base__v<seq>.json` or legacy
/// `<name>.json`). Returns `None` when absent/garbage (caller falls back).
pub async fn load_learned_model(
    storage: &dyn Storage,
    table: &str,
    json_rel_path: &str,
) -> Result<Option<LearnedIndexFile>> {
    let Some(stem) = json_rel_path.strip_suffix(".json") else {
        return Ok(None);
    };
    let path = format!("{}/{}.model", table, stem);
    match storage.read(&path).await {
        Ok(bytes) => Ok(serde_json::from_slice::<LearnedIndexFile>(&bytes).ok()),
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

impl BTreeIndex {
    /// Versioned filename for the index base at sequence `seq`, relative to the
    /// table directory.  Mirrors the `_rowindex/base__v<seq>.idx` convention.
    pub fn versioned_filename(name: &str, seq: u64) -> String {
        format!("{}/{}/base__v{:09}.json", INDEXES_DIR, name, seq)
    }

    /// Legacy single-file path (unversioned).  Kept so `load_index` can still
    /// find old-format files.
    pub fn legacy_filename(name: &str) -> String {
        format!("{}/{}.json", INDEXES_DIR, name)
    }

    /// Row IDs that may contain `value`.
    pub fn lookup(&self, value: &str) -> &[u64] {
        self.entries.get(value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Persist this index at the legacy (unversioned) path for backward
    /// compatibility with callers that use `load_index` to reload by name.
    pub async fn save(&self, storage: &dyn Storage) -> Result<()> {
        let path = format!(
            "{}/{}",
            self.definition.table,
            Self::legacy_filename(&self.definition.name)
        );
        let tmp = format!("{}.tmp", path);
        let data = serde_json::to_vec_pretty(self)?;
        storage.write(&tmp, &data).await?;
        storage.sync_data(&tmp).await?;
        storage.rename(&tmp, &path).await?;

        // Derived sibling caches (Tasks 1.3 / 4.2); see `save_versioned`.
        let legacy = Self::legacy_filename(&self.definition.name);
        if storage.local_root().is_some() {
            let _ = self.save_binary_sibling(storage, &legacy).await;
        }
        let _ = self.save_model_sibling(storage, &legacy).await;

        storage
            .sync(&format!("{}/{}", self.definition.table, INDEXES_DIR))
            .await?;
        Ok(())
    }

    /// Persist this index as a versioned immutable file and return the relative
    /// path written (suitable for storing in `Manifest.index_generations`).
    pub async fn save_versioned(&self, storage: &dyn Storage) -> Result<String> {
        let rel_path = Self::versioned_filename(&self.definition.name, self.snapshot_sequence);
        let storage_path = format!("{}/{}", self.definition.table, rel_path);
        let tmp = format!("{}.tmp", storage_path);
        let data = serde_json::to_vec_pretty(self)?;
        storage.write(&tmp, &data).await?;
        storage.sync_data(&tmp).await?;
        storage.rename(&tmp, &storage_path).await?;

        // Derived binary cache: write the mmap-friendly `.idx` sibling
        // so opens can binary-search without parsing the whole JSON map. Local
        // storage only — non-local readers never mmap it. Best-effort: the JSON
        // is canonical, so a failed/absent `.idx` just means readers fall back.
        if storage.local_root().is_some() {
            let _ = self.save_binary_sibling(storage, &rel_path).await;
        }
        // Learned-model sidecar: when the key is exactly affine, a
        // tiny `.model` lets opens locate by O(1) arithmetic without loading the
        // index. Read via `storage.read`, so it works for any backend.
        let _ = self.save_model_sibling(storage, &rel_path).await;

        storage
            .sync(&format!(
                "{}/{}/{}",
                self.definition.table, INDEXES_DIR, self.definition.name
            ))
            .await?;
        Ok(rel_path)
    }

    /// Write the derived binary index next to the JSON base (`.json` → `.idx`).
    /// The binary is content-equivalent to the JSON and is rebuildable, so this
    /// is intentionally best-effort and never fails the commit.
    async fn save_binary_sibling(&self, storage: &dyn Storage, json_rel_path: &str) -> Result<()> {
        let Some(stem) = json_rel_path.strip_suffix(".json") else {
            return Ok(());
        };
        let bin_storage_path = format!("{}/{}.idx", self.definition.table, stem);
        let tmp = format!("{}.tmp", bin_storage_path);
        let data = binary::serialize(self);
        storage.write(&tmp, &data).await?;
        storage.sync_data(&tmp).await?;
        storage.rename(&tmp, &bin_storage_path).await?;
        Ok(())
    }

    /// Write the learned-model sidecar (`.json` → `.model`) when this index fits
    /// an exact affine model. Best-effort, derived, rebuildable.
    async fn save_model_sibling(&self, storage: &dyn Storage, json_rel_path: &str) -> Result<()> {
        let Some(model) = LearnedKeyModel::fit(self) else {
            return Ok(());
        };
        let Some(stem) = json_rel_path.strip_suffix(".json") else {
            return Ok(());
        };
        let path = format!("{}/{}.model", self.definition.table, stem);
        let tmp = format!("{}.tmp", path);
        let file = LearnedIndexFile {
            definition: self.definition.clone(),
            model,
        };
        let data = serde_json::to_vec(&file)?;
        storage.write(&tmp, &data).await?;
        storage.sync_data(&tmp).await?;
        storage.rename(&tmp, &path).await?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tombstone + add deltas for indexes
// ---------------------------------------------------------------------------

/// An index delta records row IDs that have been logically deleted from an
/// index generation (tombstones) and new `value → row_id` entries to add
/// (adds).
///
/// The delta is stored as a JSON file alongside the base index file and applied
/// at resolution time:
///   1. Remove tombstoned row IDs from every value's list (drop empty entries).
///   2. Insert each add entry: `entries[value_key].push(row_id)`.
///
/// A tombstone-only delta has `adds` empty.  An update
/// delta carries both tombstones (for the updated row_ids)
/// and adds (for the new value → row_id mappings).
///
/// The `tombstoned_row_ids` field name is kept unchanged for backward
/// compatibility with legacy delta files.  The `adds` field defaults
/// to an empty Vec when absent from JSON (backward compatible).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TombstoneDelta {
    /// The row IDs that are no longer live in this index generation.
    pub tombstoned_row_ids: Vec<u64>,
    /// New `(value_key, row_id)` entries to insert into the index.
    ///
    /// Resolution order: tombstones first (removes old mappings), then adds
    /// (inserts new mappings). This ensures `value_new → row_id` is live and
    /// `value_old → row_id` is gone without needing the pre-image old value.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub adds: Vec<(String, u64)>,
}

impl TombstoneDelta {
    /// Relative path (within `_indexes/<name>/`) for a tombstone delta written
    /// at commit sequence `seq`.
    pub fn versioned_filename(name: &str, seq: u64) -> String {
        format!("{}/{}/delta__v{:09}.json", INDEXES_DIR, name, seq)
    }
}

/// Write a tombstone-only delta for `row_ids` into the unique index named
/// `index_name` on `table`, returning an updated `IndexRef` (existing base +
/// existing deltas + the new tombstone delta path).
///
/// The caller is responsible for adding the returned delta path to the commit
/// intent journal so that recovery / GC never treats it as an orphan.
///
/// # Panics / errors
///
/// Returns `Err` if the storage write fails.
pub async fn append_tombstones(
    storage: &dyn Storage,
    table: &str,
    index_name: &str,
    row_ids: &[u64],
    seq: u64,
    current_ref: &IndexRef,
    sync_durable: bool,
) -> Result<IndexRef> {
    let delta = TombstoneDelta {
        tombstoned_row_ids: row_ids.to_vec(),
        adds: vec![],
    };
    let rel_path = TombstoneDelta::versioned_filename(index_name, seq);
    let storage_path = format!("{}/{}", table, rel_path);
    let tmp = format!("{}.tmp", storage_path);

    // Ensure the _indexes/<name>/ directory exists before writing.
    let dir = format!("{}/{}/{}", table, INDEXES_DIR, index_name);
    if let Err(e) = storage.sync(&dir).await {
        if !is_not_found(&e) {
            return Err(e);
        }
    }

    let data = serde_json::to_vec_pretty(&delta)?;
    storage.write(&tmp, &data).await?;
    // In WAL mode (`sync_durable == false`) the delta's bytes are inlined in the
    // mutation record, whose single `fsync` covers durability; the checkpoint
    // `fsync`s the file before the manifest references it.
    if sync_durable {
        storage.sync_data(&tmp).await?;
    }
    storage.rename(&tmp, &storage_path).await?;
    if sync_durable {
        storage.sync(&dir).await?;
    }

    let mut new_ref = current_ref.clone();
    new_ref.deltas.push(rel_path);
    Ok(new_ref)
}

/// Write an update delta for `index_name` on `table` that tombstones the
/// given `row_ids` and inserts new `(value_key, row_id)` add entries.
///
/// Resolution order (base → oldest delta → newest delta):
///   1. Remove tombstoned row IDs from every value's list.
///   2. Insert each add entry: `entries[value_key].push(row_id)`.
///
/// This means the old value→row_id mapping is gone (tombstone drops it from
/// wherever it lived in the base) and the new value→row_id mapping is live
/// (the add inserts it), without ever needing the pre-image old value.
///
/// Returns the updated `IndexRef` (current ref + the new delta path appended).
/// The caller must add the returned delta path to the intent journal.
pub async fn append_index_delta(
    storage: &dyn Storage,
    table: &str,
    index_name: &str,
    row_ids: &[u64],
    adds: Vec<(String, u64)>,
    seq: u64,
    current_ref: &IndexRef,
) -> Result<(IndexRef, String)> {
    let delta = TombstoneDelta {
        tombstoned_row_ids: row_ids.to_vec(),
        adds,
    };
    let rel_path = TombstoneDelta::versioned_filename(index_name, seq);
    let storage_path = format!("{}/{}", table, rel_path);
    let tmp = format!("{}.tmp", storage_path);

    // Ensure the _indexes/<name>/ directory exists before writing.
    let dir = format!("{}/{}/{}", table, INDEXES_DIR, index_name);
    if let Err(e) = storage.sync(&dir).await {
        if !is_not_found(&e) {
            return Err(e);
        }
    }

    let data = serde_json::to_vec_pretty(&delta)?;
    storage.write(&tmp, &data).await?;
    storage.sync_data(&tmp).await?;
    storage.rename(&tmp, &storage_path).await?;
    storage.sync(&dir).await?;

    let mut new_ref = current_ref.clone();
    new_ref.deltas.push(rel_path.clone());
    Ok((new_ref, rel_path))
}

/// Load and apply all delta files in `index_ref` to the given index.
///
/// For each delta (oldest → newest):
///   1. Remove tombstoned row IDs from every value's list; drop empty entries.
///   2. Insert each add entry: `entries[value_key].push(row_id)`.
///
/// Missing delta files are silently skipped (not-found is treated as empty).
async fn apply_deltas(
    storage: &dyn Storage,
    table: &str,
    index_ref: &IndexRef,
    index: &mut BTreeIndex,
) -> Result<()> {
    for delta_path in &index_ref.deltas {
        let storage_path = format!("{}/{}", table, delta_path);
        let delta: TombstoneDelta = match storage.read(&storage_path).await {
            Ok(data) => serde_json::from_slice(&data)?,
            Err(e) if is_not_found(&e) => continue,
            Err(e) => return Err(e),
        };
        // Step 1: remove tombstoned row IDs.
        if !delta.tombstoned_row_ids.is_empty() {
            let tombstoned: std::collections::HashSet<u64> =
                delta.tombstoned_row_ids.iter().copied().collect();
            index.entries.retain(|_, ids| {
                ids.retain(|id| !tombstoned.contains(id));
                !ids.is_empty()
            });
        }
        // Step 2: apply adds.
        for (value_key, row_id) in delta.adds {
            index.entries.entry(value_key).or_default().push(row_id);
        }
    }
    Ok(())
}

/// Load a secondary index by its `IndexRef` (versioned path) or fall back to
/// the legacy unversioned path for backward compatibility.
///
/// If `index_ref.deltas` is non-empty the delta files are applied in order
/// (oldest → newest):
///   1. Tombstones: remove dead row IDs from every value's list.
///   2. Adds: insert new `value_key → row_id` entries.
///
/// This preserves the DELETE tombstone behavior (adds empty) and extends it
/// for UPDATE deltas (tombstones + adds).
pub async fn load_index_by_ref(
    storage: &dyn Storage,
    table: &str,
    name: &str,
    index_ref: &IndexRef,
) -> Result<Option<BTreeIndex>> {
    let mut index_opt = if let Some(ref base_path) = index_ref.base {
        let storage_path = format!("{}/{}", table, base_path);
        match storage.read(&storage_path).await {
            Ok(data) => Some(serde_json::from_slice::<BTreeIndex>(&data)?),
            Err(e) if is_not_found(&e) => None,
            Err(e) => return Err(e),
        }
    } else {
        None
    };

    // Fall back to legacy path if no versioned base was found.
    if index_opt.is_none() {
        index_opt = load_index(storage, table, name).await?;
    }

    // Apply deltas (tombstones then adds) if present.
    if !index_ref.deltas.is_empty() {
        if let Some(ref mut index) = index_opt {
            apply_deltas(storage, table, index_ref, index).await?;
        }
    }

    Ok(index_opt)
}

/// In-memory overlay of an index generation's deltas (tombstones + adds),
/// merged on top of a base (JSON or binary) at lookup time.
///
/// This mirrors [`apply_deltas`] but in a per-key form the binary (mmap) open
/// path can use without materializing the whole `BTreeMap`. Built by replaying
/// deltas oldest → newest via [`load_index_overlay`].
#[derive(Debug, Default, Clone)]
pub struct IndexDeltaOverlay {
    tombstones: std::collections::HashSet<u64>,
    adds: BTreeMap<String, Vec<u64>>,
}

impl IndexDeltaOverlay {
    /// True when there are no deltas to apply (base postings pass through).
    pub fn is_empty(&self) -> bool {
        self.tombstones.is_empty() && self.adds.is_empty()
    }

    /// Merge the overlay into `base` postings for `value`: drop tombstoned row
    /// ids from the base, then append this value's adds (which are kept current
    /// — a later tombstone removes an earlier add during replay, so adds are not
    /// re-filtered here, matching [`apply_deltas`]).
    pub fn merge(&self, value: &str, base: impl IntoIterator<Item = u64>) -> Vec<u64> {
        let mut out: Vec<u64> = base
            .into_iter()
            .filter(|id| !self.tombstones.contains(id))
            .collect();
        if let Some(adds) = self.adds.get(value) {
            out.extend(adds.iter().copied());
        }
        out
    }
}

/// Replay an `IndexRef`'s delta files (oldest → newest) into an
/// [`IndexDeltaOverlay`], WITHOUT loading the (large) base. Used by the binary
/// open path so deltas stay correct over an mmap'd base.
pub async fn load_index_overlay(
    storage: &dyn Storage,
    table: &str,
    index_ref: &IndexRef,
) -> Result<IndexDeltaOverlay> {
    let mut overlay = IndexDeltaOverlay::default();
    for delta_path in &index_ref.deltas {
        let storage_path = format!("{}/{}", table, delta_path);
        let delta: TombstoneDelta = match storage.read(&storage_path).await {
            Ok(data) => serde_json::from_slice(&data)?,
            Err(e) if is_not_found(&e) => continue,
            Err(e) => return Err(e),
        };
        // Tombstones first (matching apply_deltas): a row id that was added by an
        // earlier delta and is tombstoned now must not survive, so drop it from
        // the pending adds too. (O(adds) per tombstone — deltas are small and
        // folded into a new base at compaction. ponytail: revisit if a hot table
        // accumulates many deltas before optimize.)
        for id in &delta.tombstoned_row_ids {
            overlay.tombstones.insert(*id);
            for ids in overlay.adds.values_mut() {
                ids.retain(|x| x != id);
            }
        }
        for (value, id) in delta.adds {
            overlay.adds.entry(value).or_default().push(id);
        }
    }
    Ok(overlay)
}

/// Load a secondary index by name from the legacy unversioned path.
pub async fn load_index(
    storage: &dyn Storage,
    table: &str,
    name: &str,
) -> Result<Option<BTreeIndex>> {
    let file_name = if name.ends_with(".json") {
        name.to_string()
    } else {
        format!("{}.json", name)
    };
    let path = format!("{}/{}/{}", table, INDEXES_DIR, file_name);
    match storage.read(&path).await {
        Ok(data) => match serde_json::from_slice::<BTreeIndex>(&data) {
            Ok(index) => Ok(Some(index)),
            Err(_) if is_legacy_string_index(&data) => {
                // Legacy writers stored `entries` as {value: ["rg_<hash>",
                // ...]} (row-group filenames). The current format keys entries
                // by stable row id (`u64`); a row-group filename is not a row
                // id, so the index cannot be transparently migrated. Surface a
                // clear, actionable error instead of the raw serde message.
                Err(crate::IcefallDBError::LegacyIndex {
                    table: table.to_string(),
                    name: file_name,
                })
            }
            Err(e) => Err(e.into()),
        },
        Err(e) if is_not_found(&e) => Ok(None),
        Err(e) => Err(e),
    }
}

/// Detect the obsolete legacy index format, whose `entries` map stores
/// arrays of strings (row-group filenames) rather than `u64` row ids.
fn is_legacy_string_index(data: &[u8]) -> bool {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return false;
    };
    let Some(entries) = value.get("entries").and_then(|e| e.as_object()) else {
        return false;
    };
    entries
        .values()
        .filter_map(|list| list.as_array())
        .any(|a| a.iter().any(|item| item.is_string()))
}

pub async fn list_index_names(storage: &dyn Storage, table: &str) -> Result<Vec<String>> {
    let prefix = format!("{}/{}", table, INDEXES_DIR);
    match storage.list(&prefix).await {
        Ok(entries) => Ok(entries
            .iter()
            .filter_map(|e| {
                let name = std::path::Path::new(e).file_name()?.to_str()?;
                name.strip_suffix(".json").map(|s| s.to_string())
            })
            .collect()),
        Err(e) if is_not_found(&e) => Ok(Vec::new()),
        Err(e) => Err(e),
    }
}

/// Build a row-ID-keyed B-tree index for `definition` from the rows visible in
/// `manifest`.
///
/// For each fragment, this function:
/// 1. Reads the Parquet file and extracts the indexed column.
/// 2. Skips physical offsets that are present in the fragment's deletion vector.
/// 3. Maps each surviving offset to its stable `row_id` via the fragment's
///    `RowGroupMeta.row_ids` segments.
/// 4. Pushes that `row_id` into `entries[value]`.
///
/// Row IDs are stable across compaction and relocation, so the resulting index
/// is safe to use with any snapshot that references this generation.
///
/// If `definition.unique` is `true`, the build rejects any key that maps to more
/// than one live row id.
pub async fn build_btree_index(
    storage: &dyn Storage,
    definition: &IndexDefinition,
    manifest: &Manifest,
) -> Result<BTreeIndex> {
    let adds = collect_index_adds_for_fragments(storage, definition, &manifest.row_groups).await?;
    let mut entries: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (key, row_id) in adds {
        entries.entry(key).or_default().push(row_id);
    }
    // Sort and deduplicate row-id lists.
    for ids in entries.values_mut() {
        ids.sort_unstable();
        ids.dedup();
    }

    if definition.unique {
        for (key, ids) in &entries {
            if ids.len() > 1 {
                return Err(IcefallDBError::UniqueKeyViolation {
                    table: definition.table.clone(),
                    index: definition.name.clone(),
                    key: key.clone(),
                });
            }
        }
    }

    Ok(BTreeIndex {
        definition: definition.clone(),
        snapshot_sequence: manifest.sequence,
        entries,
    })
}

/// Extract `(value_key, stable_row_id)` for every live, non-null row of the
/// indexed column across `fragments`. Shared by the full builder
/// ([`build_btree_index`]) and the incremental INSERT path
/// ([`IndexMaintainer::maintain_on_insert`]); the latter passes only the
/// fragments added by a commit, so its cost scales with rows inserted, not table
/// size.
async fn collect_index_adds_for_fragments(
    storage: &dyn Storage,
    definition: &IndexDefinition,
    fragments: &[crate::metadata::manifest::RowGroupEntry],
) -> Result<Vec<(String, u64)>> {
    let mut adds: Vec<(String, u64)> = Vec::new();

    for rg in fragments {
        let data_path = format!("{}/{}", definition.table, rg.data);

        // Load the deletion vector for this fragment, if any.
        let deletion_vector = if let Some(ref del_path) = rg.deletes {
            let del_storage_path = format!("{}/{}", definition.table, del_path);
            match storage.read(&del_storage_path).await {
                Ok(bytes) => DeletionVector::deserialize(&bytes)
                    .map_err(|e| IcefallDBError::Other(Box::new(e)))?,
                Err(e) if is_not_found(&e) => DeletionVector::default(),
                Err(e) => return Err(e),
            }
        } else {
            DeletionVector::default()
        };

        // Load the RowGroupMeta to get the row_ids segments.  The meta file path
        // has the `.parquet` extension stripped and `.meta` appended.
        let meta_path = format!("{}/{}", definition.table, rg.meta);
        let meta_bytes = storage.read(&meta_path).await?;
        let meta: crate::metadata::RowGroupMeta = serde_json::from_slice(&meta_bytes)?;

        // Pre-expand row IDs from the segments into a flat Vec so that we can
        // index by physical offset cheaply.  An empty `row_ids` means this is a
        // legacy fragment with no allocated IDs; we skip it (cannot build a
        // row-ID-keyed entry without IDs).
        if meta.row_ids.is_empty() {
            continue;
        }
        let row_id_vec: Vec<u64> = meta.row_ids.iter().flat_map(segment_ids).collect();

        // TODO: use RowGroupMeta column_offsets to read only the indexed column.
        let data = storage.read(&data_path).await?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data)).map_err(other)?;
        let batches = reader.build().map_err(other)?;

        let mut physical_offset: u32 = 0;
        for batch in batches {
            let batch = batch.map_err(other)?;
            let col_idx = batch.schema().index_of(&definition.column).map_err(|_| {
                IcefallDBError::Other(format!("column {} not found", definition.column).into())
            })?;
            let array = batch.column(col_idx);
            for i in 0..array.len() {
                let offset = physical_offset + i as u32;
                // Skip deleted rows.
                if deletion_vector.contains(offset) {
                    continue;
                }
                if array.is_null(i) {
                    continue;
                }
                // Map physical offset → stable row ID.
                let row_id = *row_id_vec.get(offset as usize).ok_or_else(|| {
                    IcefallDBError::Other(
                        format!(
                            "physical offset {} out of range for row_id_vec (len {})",
                            offset,
                            row_id_vec.len()
                        )
                        .into(),
                    )
                })?;
                let key = scalar_to_key(array.as_ref(), i)?;
                adds.push((key, row_id));
            }
            physical_offset += array.len() as u32;
        }
    }

    Ok(adds)
}

/// Resolve a set of row IDs through the `AddressMap` and filter out any whose
/// physical offset is marked deleted in the corresponding fragment's deletion
/// vector.
///
/// `get_deletion_vector(fragment_id)` is called at most once per fragment
/// encountered in the address map; the caller is responsible for loading the
/// vectors (see `resolve_live_addresses_storage` for the storage-backed version).
///
/// Returns `(fragment_id, physical_offset)` pairs for every live row.
pub fn resolve_live_addresses<F>(
    row_ids: &[u64],
    address_map: &AddressMap,
    mut get_deletion_vector: F,
) -> Vec<(u64, u32)>
where
    F: FnMut(u64) -> Option<DeletionVector>,
{
    use std::collections::HashMap;
    let mut dv_cache: HashMap<u64, Option<DeletionVector>> = HashMap::new();

    let mut result = Vec::new();
    for &row_id in row_ids {
        let Some((fragment_id, offset)) = address_map.lookup(row_id) else {
            continue;
        };
        let dv = dv_cache
            .entry(fragment_id)
            .or_insert_with(|| get_deletion_vector(fragment_id));
        if let Some(ref dv) = dv {
            if dv.contains(offset) {
                continue;
            }
        }
        result.push((fragment_id, offset));
    }
    result
}

/// Storage-backed version of `resolve_live_addresses`: opens the `AddressMap`
/// from `manifest.rowindex_generation`, loads deletion vectors as needed, and
/// returns live `(fragment_id, offset)` pairs for `row_ids`.
pub async fn resolve_live_addresses_storage(
    storage: &dyn Storage,
    table: &str,
    manifest: &Manifest,
    row_ids: &[u64],
) -> Result<Vec<(u64, u32)>> {
    // Open the address map from the manifest's rowindex generation.
    let gen = match &manifest.rowindex_generation {
        Some(g) => g,
        None => return Ok(Vec::new()),
    };
    let address_map = AddressMap::open(storage, table, gen).await?;

    // Eagerly load all deletion vectors we might need (only for fragments with
    // non-null `deletes`).
    let mut dv_map: std::collections::HashMap<u64, DeletionVector> =
        std::collections::HashMap::new();
    for rg in &manifest.row_groups {
        if let Some(ref del_path) = rg.deletes {
            let del_storage_path = format!("{}/{}", table, del_path);
            match storage.read(&del_storage_path).await {
                Ok(bytes) => {
                    let dv = DeletionVector::deserialize(&bytes)
                        .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
                    dv_map.insert(rg.fragment_id, dv);
                }
                Err(e) if is_not_found(&e) => {}
                Err(e) => return Err(e),
            }
        }
    }

    // SAFETY of the consuming `remove`: `resolve_live_addresses` memoises the
    // closure's result in an internal `dv_cache` keyed by `fragment_id`, so the
    // closure is invoked at most once per fragment. Removing (rather than
    // cloning) the deletion vector out of `dv_map` is therefore correct: no
    // fragment is ever looked up twice, so a moved-out entry is never needed
    // again.
    let result = resolve_live_addresses(row_ids, &address_map, |frag_id| dv_map.remove(&frag_id));
    Ok(result)
}

fn scalar_to_key(array: &dyn Array, row: usize) -> Result<String> {
    match array.data_type() {
        DataType::Int64 => Ok(array
            .as_primitive::<arrow::datatypes::Int64Type>()
            .value(row)
            .to_string()),
        DataType::Utf8 => Ok(array.as_string::<i32>().value(row).to_string()),
        DataType::LargeUtf8 => Ok(array.as_string::<i64>().value(row).to_string()),
        other => Err(IcefallDBError::TypeNotSupported(other.to_string())),
    }
}

/// Verify that `adds` do not violate the uniqueness invariant of `definition`.
///
/// `existing_index` is the index generation before the change (base plus applied
/// deltas). `tombstoned_row_ids` are row IDs that will be removed before the adds
/// are applied; they are allowed to reappear under the same key (used by UPDATE).
///
/// Returns `UniqueKeyViolation` when:
///   * two adds share the same key but different row ids, or
///   * an add's key already exists in the existing index with a live row id that
///     is not about to be tombstoned.
fn check_unique_adds(
    definition: &IndexDefinition,
    existing_index: &BTreeIndex,
    adds: &[(String, u64)],
    tombstoned_row_ids: &[u64],
) -> Result<()> {
    if !definition.unique {
        return Ok(());
    }

    let tombstoned: std::collections::HashSet<u64> = tombstoned_row_ids.iter().copied().collect();
    let mut seen_in_batch: std::collections::HashMap<&str, u64> =
        std::collections::HashMap::with_capacity(adds.len());

    for (key, row_id) in adds {
        // Duplicate within the incoming batch.
        if let Some(&first_rid) = seen_in_batch.get(key.as_str()) {
            if first_rid != *row_id {
                return Err(IcefallDBError::UniqueKeyViolation {
                    table: definition.table.clone(),
                    index: definition.name.clone(),
                    key: key.clone(),
                });
            }
        } else {
            seen_in_batch.insert(key, *row_id);
        }

        // Collision with an existing live key that is not being tombstoned.
        if let Some(existing_ids) = existing_index.entries.get(key) {
            if existing_ids.iter().any(|id| !tombstoned.contains(id)) {
                return Err(IcefallDBError::UniqueKeyViolation {
                    table: definition.table.clone(),
                    index: definition.name.clone(),
                    key: key.clone(),
                });
            }
        }
    }

    Ok(())
}

pub struct IndexMaintainer;

impl IndexMaintainer {
    /// Rebuild every secondary index defined for `table` from the rows visible
    /// in `manifest`, write each as a versioned immutable base file at
    /// `_indexes/<name>/base__v<seq>.json`, and record the resulting generation
    /// in `manifest.index_generations` (in place).
    ///
    /// Returns the table-relative paths of the index base files written, so the
    /// caller can list them in the commit intent journal (and treat them as
    /// referenced/durable during recovery).
    ///
    /// This is designed to run **inside** the atomic commit, after the new
    /// manifest's `row_groups` and row-id/fragment-id assignments are finalized
    /// but before the manifest is serialized and the pointer is swapped. The
    /// manifest is therefore written exactly once, already containing
    /// `index_generations`; no committed manifest is ever overwritten in place.
    ///
    /// No legacy unversioned `_indexes/<name>.json` file is written: readers
    /// resolve the per-snapshot generation through `manifest.index_generations`,
    /// so a second on-disk copy would only invite stale reads.
    pub async fn maintain(
        storage: Arc<dyn Storage>,
        table: &str,
        manifest: &mut Manifest,
    ) -> Result<Vec<String>> {
        let catalog = DatabaseCatalog::new(storage.clone());
        let data = catalog.load().await?;
        let mut written_paths = Vec::new();
        for (name, entry) in data.indexes {
            if entry.table != table || entry.index_type != "btree" {
                continue;
            }
            let definition = IndexDefinition {
                name: name.clone(),
                table: entry.table,
                column: entry.column,
                unique: entry.unique,
            };
            let index = build_btree_index(storage.as_ref(), &definition, manifest).await?;
            let rel_path = index.save_versioned(storage.as_ref()).await?;
            written_paths.push(rel_path.clone());
            manifest.index_generations.insert(
                name,
                IndexRef {
                    base: Some(rel_path),
                    deltas: vec![],
                },
            );
        }
        Ok(written_paths)
    }

    /// Incrementally maintain secondary indexes after an UPDATE commit.
    ///
    /// Only indexes whose `column` is present in `set_columns` can have
    /// changed; all others are carried forward UNCHANGED (no rebuild, no new
    /// delta). For each changed-column index, writes a delta that:
    ///   - tombstones the updated `row_ids` (removes old value → row_id mapping)
    ///   - adds new `(value_key, row_id)` entries from `updated_rows` (inserts
    ///     new value → row_id mapping)
    ///
    /// Null values in the indexed column are skipped (not indexed).
    ///
    /// Returns the table-relative paths of new delta files written (for the
    /// intent journal). The caller must update `manifest.index_generations` in
    /// place using the returned `IndexRef`s; this function mutates `manifest`
    /// directly.
    ///
    /// # Arguments
    ///
    /// - `storage`: storage backend
    /// - `table`: table name
    /// - `manifest`: new manifest being built (mutated in place)
    /// - `set_columns`: names of columns in the SET clause
    /// - `updated_rows`: the new-value record batch (in same order as `row_ids`)
    /// - `row_ids`: the stable row IDs of the updated rows (in same order as
    ///   `updated_rows`)
    /// - `seq`: commit sequence number (used to name delta files)
    pub async fn maintain_on_update(
        storage: Arc<dyn Storage>,
        table: &str,
        manifest: &mut Manifest,
        set_columns: &[String],
        updated_rows: &arrow::array::RecordBatch,
        row_ids: &[u64],
        seq: u64,
    ) -> Result<Vec<String>> {
        if set_columns.is_empty() {
            // No SET columns → no index can have changed → nothing to do.
            return Ok(Vec::new());
        }

        let catalog = DatabaseCatalog::new(storage.clone());
        let data = catalog.load().await?;
        let set_col_set: std::collections::HashSet<&str> =
            set_columns.iter().map(|s| s.as_str()).collect();

        let mut written_paths = Vec::new();

        for (name, entry) in &data.indexes {
            if entry.table != table || entry.index_type != "btree" {
                continue;
            }
            if !set_col_set.contains(entry.column.as_str()) {
                // This index's column was not updated — carry forward unchanged.
                continue;
            }

            let definition = IndexDefinition {
                name: name.clone(),
                table: entry.table.clone(),
                column: entry.column.clone(),
                unique: entry.unique,
            };

            // Build the adds: for each updated row, extract the new value.
            let schema = updated_rows.schema();
            let col_idx = match schema.index_of(&entry.column) {
                Ok(i) => i,
                Err(_) => continue, // column not in batch — skip
            };
            let array = updated_rows.column(col_idx);

            let mut adds: Vec<(String, u64)> = Vec::new();
            for (k, &row_id) in row_ids.iter().enumerate() {
                if array.is_null(k) {
                    continue; // null values are not indexed
                }
                match scalar_to_key(array.as_ref(), k) {
                    Ok(key) => adds.push((key, row_id)),
                    Err(_) => continue, // unsupported type — skip
                }
            }

            // Retrieve the current IndexRef for this index.
            let current_ref = manifest
                .index_generations
                .get(name)
                .cloned()
                .unwrap_or_default();

            // For unique indexes, verify the new values do not collide with
            // existing live keys (other than the row being updated itself).
            if definition.unique && !adds.is_empty() {
                if let Some(existing) =
                    load_index_by_ref(storage.as_ref(), table, name, &current_ref).await?
                {
                    check_unique_adds(&definition, &existing, &adds, row_ids)?;
                }
            }

            // Write the delta (tombstones = all updated row_ids; adds = new values).
            let (new_ref, delta_path) = append_index_delta(
                storage.as_ref(),
                table,
                name,
                row_ids,
                adds,
                seq,
                &current_ref,
            )
            .await?;

            written_paths.push(delta_path);
            manifest.index_generations.insert(name.clone(), new_ref);
        }

        Ok(written_paths)
    }

    /// Incrementally maintain secondary indexes after a pure-append INSERT.
    ///
    /// Instead of rebuilding each index over the whole table (O(table size)),
    /// this scans only `new_fragments` (the fragments this commit added) and
    /// writes an **adds-only** delta — `(value → new row_id)` entries, no
    /// tombstones — onto the current generation. Cost scales with rows inserted,
    /// not table size, so per-INSERT index cost is flat vs table size.
    ///
    /// The base for each index is carried forward from
    /// `current_index_generations`, or — for an index created via the legacy
    /// (unversioned) path that never recorded a generation — from the legacy
    /// `_indexes/<name>.json` base. If no base exists yet (e.g. an index defined
    /// on a previously-empty table), this falls back to a one-time full build to
    /// establish the base.
    ///
    /// Mutates `manifest.index_generations` in place and returns the
    /// table-relative paths written (for the commit intent journal).
    pub async fn maintain_on_insert(
        storage: Arc<dyn Storage>,
        table: &str,
        manifest: &mut Manifest,
        current_index_generations: &std::collections::HashMap<String, IndexRef>,
        new_fragments: &[crate::metadata::manifest::RowGroupEntry],
        seq: u64,
    ) -> Result<Vec<String>> {
        let catalog = DatabaseCatalog::new(storage.clone());
        let data = catalog.load().await?;
        let mut written_paths = Vec::new();

        for (name, entry) in &data.indexes {
            if entry.table != table || entry.index_type != "btree" {
                continue;
            }
            let definition = IndexDefinition {
                name: name.clone(),
                table: entry.table.clone(),
                column: entry.column.clone(),
                unique: entry.unique,
            };

            // Resolve the generation to extend. Prefer the recorded generation;
            // otherwise adopt the legacy unversioned base if one is on disk.
            let current_ref = match current_index_generations.get(name) {
                Some(r) => Some(r.clone()),
                None => {
                    let legacy_rel = BTreeIndex::legacy_filename(name);
                    let legacy_path = format!("{}/{}", table, legacy_rel);
                    if storage.size(&legacy_path).await.is_ok() {
                        Some(IndexRef {
                            base: Some(legacy_rel),
                            deltas: vec![],
                        })
                    } else {
                        None
                    }
                }
            };

            // No base anywhere → one-time full build to establish it.
            let Some(current_ref) = current_ref.filter(|r| r.base.is_some()) else {
                let index = build_btree_index(storage.as_ref(), &definition, manifest).await?;
                let rel_path = index.save_versioned(storage.as_ref()).await?;
                written_paths.push(rel_path.clone());
                manifest.index_generations.insert(
                    name.clone(),
                    IndexRef {
                        base: Some(rel_path),
                        deltas: vec![],
                    },
                );
                continue;
            };

            // Adds-only delta over just the new fragments.
            let adds =
                collect_index_adds_for_fragments(storage.as_ref(), &definition, new_fragments)
                    .await?;

            // For unique indexes, verify the new keys do not collide with existing
            // live keys and are unique within the incoming batch.
            if definition.unique && !adds.is_empty() {
                if let Some(existing) =
                    load_index_by_ref(storage.as_ref(), table, name, &current_ref).await?
                {
                    check_unique_adds(&definition, &existing, &adds, &[])?;
                }
            }

            let (new_ref, delta_path) =
                append_index_delta(storage.as_ref(), table, name, &[], adds, seq, &current_ref)
                    .await?;
            written_paths.push(delta_path);
            manifest.index_generations.insert(name.clone(), new_ref);
        }

        Ok(written_paths)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::manifest::{RowGroupEntry, RowIndexRef};
    use crate::metadata::RowGroupMeta;
    use crate::rowid::RowIdSegment;
    use crate::rowindex::{encode_idx, AddrSegment};
    use crate::storage::memory::MemoryStorage;
    use std::collections::HashMap;

    #[test]
    fn is_legacy_string_index_detects_pre_p0_12_format() {
        // Legacy format: entries keyed by value -> list of row-group filename
        // strings (the format that caused the opaque serde crash).
        let legacy = br#"{
            "definition": {"name": "idx", "columns": ["category"], "unique": false},
            "snapshot_sequence": 10,
            "entries": {"cat_0": ["rg_807d69587b654297a6944f47b39198d1"]}
        }"#;
        assert!(is_legacy_string_index(legacy));

        // Current format: entries -> list of u64 row ids.
        let current = br#"{
            "definition": {"name": "idx", "columns": ["category"], "unique": false},
            "snapshot_sequence": 10,
            "entries": {"cat_0": [0, 1, 2]}
        }"#;
        assert!(!is_legacy_string_index(current));

        // Garbage / non-index JSON is not mistaken for the legacy format.
        assert!(!is_legacy_string_index(b"not json"));
        assert!(!is_legacy_string_index(br#"{"entries": {}}"#));
    }

    fn index_from(pairs: Vec<(i64, Vec<u64>)>) -> BTreeIndex {
        BTreeIndex {
            definition: IndexDefinition {
                name: "idx".into(),
                table: "t".into(),
                column: "id".into(),
                unique: true,
            },
            snapshot_sequence: 1,
            entries: pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    /// An exactly-affine integer key fits a learned model whose
    /// `lookup` is byte-equal to the full B-tree index (present + missing keys),
    /// while non-affine / non-unique / non-integer columns do not fit (fallback).
    #[test]
    fn learned_model_matches_index_or_falls_back() {
        // Contiguous id 0..50 → row_id == key.
        let contiguous = index_from((0..50).map(|i| (i, vec![i as u64])).collect());
        let model = LearnedKeyModel::fit(&contiguous).expect("contiguous key must fit a model");
        for k in ["0", "25", "49", "-1", "50", "100", "notint"] {
            assert_eq!(
                model.lookup(k),
                contiguous.lookup(k).to_vec(),
                "model lookup({k}) must equal the index"
            );
        }

        // Affine: id 100,110,...,190 → row_id 5..14 (stride 10 keys, stride 1 rids).
        let affine = index_from(
            (0..10)
                .map(|i| (100 + i * 10, vec![5 + i as u64]))
                .collect(),
        );
        let m = LearnedKeyModel::fit(&affine).expect("affine key must fit");
        for k in ["100", "150", "190", "105", "200"] {
            assert_eq!(m.lookup(k), affine.lookup(k).to_vec(), "affine lookup({k})");
        }

        // Non-affine (a gap) → no model.
        assert!(
            LearnedKeyModel::fit(&index_from(vec![(0, vec![0]), (1, vec![1]), (3, vec![2])]))
                .is_none()
        );
        // Non-unique key → no model.
        assert!(LearnedKeyModel::fit(&index_from(vec![(0, vec![0, 9]), (1, vec![1])])).is_none());
        // Non-integer key → no model.
        let non_int = BTreeIndex {
            definition: IndexDefinition {
                name: "idx".into(),
                table: "t".into(),
                column: "c".into(),
                unique: true,
            },
            snapshot_sequence: 1,
            entries: [("a".to_string(), vec![0u64]), ("b".to_string(), vec![1])]
                .into_iter()
                .collect(),
        };
        assert!(LearnedKeyModel::fit(&non_int).is_none());
    }

    /// The binary base + replayed delta overlay must resolve every key
    /// byte-equal to the canonical JSON base + `apply_deltas`, including the
    /// update (tombstone-then-add) and add-then-tombstone orderings.
    #[tokio::test]
    async fn binary_plus_overlay_matches_json_plus_deltas() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "t";
        let def = IndexDefinition {
            name: "idx".into(),
            table: table.into(),
            column: "c".into(),
            unique: false,
        };
        let mut entries: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        entries.insert("a".into(), vec![1u64, 2, 3]);
        entries.insert("b".into(), vec![4u64, 5]);
        entries.insert("c".into(), vec![6u64]);
        let base = BTreeIndex {
            definition: def,
            snapshot_sequence: 1,
            entries,
        };
        let base_rel = base.save_versioned(storage.as_ref()).await.unwrap();
        let mut iref = IndexRef {
            base: Some(base_rel),
            deltas: vec![],
        };

        // Delta: UPDATE row 5 from value "b" to "z" (tombstone 5, add ("z", 5)).
        let (next, _) = append_index_delta(
            storage.as_ref(),
            table,
            "idx",
            &[5],
            vec![("z".into(), 5)],
            2,
            &iref,
        )
        .await
        .unwrap();
        iref = next;
        // Delta: add row 7 under "a" …
        let (next, _) = append_index_delta(
            storage.as_ref(),
            table,
            "idx",
            &[],
            vec![("a".into(), 7)],
            3,
            &iref,
        )
        .await
        .unwrap();
        iref = next;
        // … then a later delta tombstones it (add-then-tombstone edge case).
        let (next, _) = append_index_delta(storage.as_ref(), table, "idx", &[7], vec![], 4, &iref)
            .await
            .unwrap();
        iref = next;

        // Reference: canonical JSON base + apply_deltas.
        let reference = load_index_by_ref(storage.as_ref(), table, "idx", &iref)
            .await
            .unwrap()
            .unwrap();

        // Under test: binary base + replayed overlay.
        let bin_bytes = binary::serialize(&base);
        let bin = binary::BinaryIndexRef::parse(&bin_bytes).unwrap();
        let overlay = load_index_overlay(storage.as_ref(), table, &iref)
            .await
            .unwrap();

        for key in ["a", "b", "c", "z", "missing"] {
            let mut got = overlay.merge(key, bin.lookup(key));
            got.sort_unstable();
            let mut want = reference.lookup(key).to_vec();
            want.sort_unstable();
            assert_eq!(got, want, "postings for key {key:?} must match JSON+deltas");
        }
    }

    // ─── helpers ─────────────────────────────────────────────────────────────

    /// Write a minimal Parquet file with a single Int64 column `col` containing
    /// `values`, return the raw bytes.
    fn make_parquet_bytes(col: &str, values: &[i64]) -> Vec<u8> {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            col,
            DataType::Int64,
            true,
        )]));
        let array = Arc::new(Int64Array::from(values.to_vec()));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![array]).unwrap();

        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        buf
    }

    /// Write a RowGroupMeta for a fragment with a single Range segment starting
    /// at `row_id_start`, covering `row_count` rows.  Returns the serialised
    /// JSON bytes.
    fn make_meta_bytes(rg_id: &str, row_id_start: u64, row_count: u64) -> Vec<u8> {
        let meta = RowGroupMeta {
            row_group: rg_id.to_string(),
            rows: row_count as usize,
            row_ids: vec![RowIdSegment::Range {
                start: row_id_start,
                count: row_count,
            }],
            ..Default::default()
        };
        serde_json::to_vec(&meta).unwrap()
    }

    // ─── index_resolves_value_to_live_rowid ──────────────────────────────────

    /// Build a table with a row-ID-keyed index, confirm the index maps the value
    /// to the correct row_id, and that resolving through the AddressMap yields
    /// the expected live (fragment, offset).
    #[tokio::test]
    async fn index_resolves_value_to_live_rowid() {
        let storage = Arc::new(MemoryStorage::new());

        // Write Parquet file: 3 rows with values [10, 20, 30].
        let parquet = make_parquet_bytes("v", &[10, 20, 30]);
        storage.write("tbl/rg0.parquet", &parquet).await.unwrap();

        // Write RowGroupMeta: row_ids start at 100 (3 rows → IDs 100, 101, 102).
        let meta_bytes = make_meta_bytes("rg0", 100, 3);
        storage.write("tbl/rg0.meta", &meta_bytes).await.unwrap();

        // Build an AddressMap: fragment 7, rows start at physical offset 0.
        let addr_segs = vec![AddrSegment {
            start_row_id: 100,
            fragment_id: 7,
            start_offset: 0,
            len: 3,
        }];
        let idx_bytes = encode_idx(&addr_segs);
        storage
            .write("tbl/_rowindex/base__v1.idx", &idx_bytes)
            .await
            .unwrap();

        // Manifest: one row group, no deletions.
        let manifest = Manifest {
            format_version: 1,
            sequence: 1,
            schema_id: 1,
            row_groups: vec![RowGroupEntry {
                data: "rg0.parquet".into(),
                meta: "rg0.meta".into(),
                fragment_id: 7,
                ..Default::default()
            }],
            rowindex_generation: Some(RowIndexRef {
                base: Some("_rowindex/base__v1.idx".into()),
                deltas: vec![],
            }),
            ..Default::default()
        };

        let definition = IndexDefinition {
            name: "idx_v".to_string(),
            table: "tbl".to_string(),
            column: "v".to_string(),
            unique: false,
        };

        // Build the index.
        let index = build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap();

        // Value 20 should map to row_id 101 (physical offset 1).
        let ids = index.lookup("20");
        assert_eq!(ids, &[101u64], "value 20 must map to row_id 101");

        // Value 10 → row_id 100 (offset 0), value 30 → row_id 102 (offset 2).
        assert_eq!(index.lookup("10"), &[100u64]);
        assert_eq!(index.lookup("30"), &[102u64]);

        // Resolve row_id 101 through the AddressMap.
        let live = resolve_live_addresses_storage(storage.as_ref(), "tbl", &manifest, &[101u64])
            .await
            .unwrap();
        assert_eq!(
            live,
            vec![(7u64, 1u32)],
            "row_id 101 must resolve to (fragment=7, offset=1)"
        );
    }

    // ─── deleted_row_not_indexed_or_not_live ─────────────────────────────────

    /// A value whose row has been deleted must not appear as live after
    /// resolution through the AddressMap + deletion vector.
    #[tokio::test]
    async fn deleted_row_not_indexed_or_not_live() {
        let storage = Arc::new(MemoryStorage::new());

        // Write Parquet: 4 rows with values [1, 2, 3, 4].
        let parquet = make_parquet_bytes("v", &[1, 2, 3, 4]);
        storage.write("tbl/rg0.parquet", &parquet).await.unwrap();

        // Meta: row_ids 0..3 (IDs 0, 1, 2, 3).
        let meta_bytes = make_meta_bytes("rg0", 0, 4);
        storage.write("tbl/rg0.meta", &meta_bytes).await.unwrap();

        // Deletion vector: delete physical offsets 1 (value=2) and 3 (value=4).
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32, 3u32]);
        storage
            .write("tbl/_deletions/rg0.del", &dv.serialize())
            .await
            .unwrap();

        // AddressMap: fragment 5, rows start at offset 0.
        let addr_segs = vec![AddrSegment {
            start_row_id: 0,
            fragment_id: 5,
            start_offset: 0,
            len: 4,
        }];
        storage
            .write("tbl/_rowindex/base__v2.idx", &encode_idx(&addr_segs))
            .await
            .unwrap();

        let manifest = Manifest {
            format_version: 1,
            sequence: 2,
            schema_id: 1,
            row_groups: vec![RowGroupEntry {
                data: "rg0.parquet".into(),
                meta: "rg0.meta".into(),
                fragment_id: 5,
                deletes: Some("_deletions/rg0.del".into()),
                ..Default::default()
            }],
            rowindex_generation: Some(RowIndexRef {
                base: Some("_rowindex/base__v2.idx".into()),
                deltas: vec![],
            }),
            ..Default::default()
        };

        let definition = IndexDefinition {
            name: "idx_v".to_string(),
            table: "tbl".to_string(),
            column: "v".to_string(),
            unique: false,
        };

        // Build index: deleted offsets (1 and 3) must be excluded.
        let index = build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap();

        // Value 2 (offset 1, deleted) must not appear in the index at all.
        assert!(
            index.lookup("2").is_empty(),
            "deleted value '2' must not appear in the index"
        );
        // Value 4 (offset 3, deleted) must not appear in the index at all.
        assert!(
            index.lookup("4").is_empty(),
            "deleted value '4' must not appear in the index"
        );
        // Value 1 (offset 0, live) must be present.
        assert_eq!(index.lookup("1"), &[0u64]);
        // Value 3 (offset 2, live) must be present.
        assert_eq!(index.lookup("3"), &[2u64]);

        // Even if we ask resolve_live_addresses about a deleted row_id, it must
        // not be returned (the DV filter catches it).
        let live =
            resolve_live_addresses_storage(storage.as_ref(), "tbl", &manifest, &[0, 1, 2, 3])
                .await
                .unwrap();
        let live_frags: Vec<u32> = live.iter().map(|(_, off)| *off).collect();
        assert!(
            !live_frags.contains(&1),
            "offset 1 (deleted) must not be live"
        );
        assert!(
            !live_frags.contains(&3),
            "offset 3 (deleted) must not be live"
        );
        assert!(live_frags.contains(&0), "offset 0 (live) must be returned");
        assert!(live_frags.contains(&2), "offset 2 (live) must be returned");
    }

    // ─── index_is_snapshot_scoped ────────────────────────────────────────────

    /// Prove snapshot scoping through the REAL write/read pipeline (not by
    /// hand-loading two `IndexRef`s).
    ///
    /// 1. Create a table + a btree index definition.
    /// 2. Commit one batch — the commit builds generation G_S into manifest S's
    ///    `index_generations`.
    /// 3. Commit a CHANGE (a second batch) — the commit builds a DIFFERENT
    ///    generation G_{S+1} into manifest S+1's `index_generations`.
    /// 4. Load the index through EACH manifest's `index_generations` and assert
    ///    it yields that snapshot's mapping: S must NOT see the newly added
    ///    value; S+1 must.
    ///
    /// This exercises the manifest round-trip (build → versioned write →
    /// generation recorded in the immutable manifest → `load_index_by_ref`),
    /// not just that two files load independently.
    #[tokio::test]
    async fn index_is_snapshot_scoped() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        // 1. Create the table and a btree index on column `v`.
        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat.create_table(&guard, "tbl", &schema).await.unwrap();
        dbcat
            .create_index_definition(&guard, "idx_v", "tbl", "v", "btree")
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        // 2. Commit S: rows [100, 200] → generation G_S.
        let mut writer = Writer::new(storage.clone(), "tbl", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![100, 200]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Capture manifest S and its recorded generation.
        let manifest_s = {
            let cat = Catalog::load(storage.as_ref(), "tbl").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let seq_s = manifest_s.sequence;
        let ref_s = manifest_s
            .index_generations
            .get("idx_v")
            .expect("commit S must record an index generation")
            .clone();
        assert_eq!(
            ref_s.base.as_deref(),
            Some(BTreeIndex::versioned_filename("idx_v", seq_s).as_str()),
            "generation must point at the versioned base file for seq S"
        );

        // 3. Commit S+1: add row [300] → a DIFFERENT generation G_{S+1}.
        let mut writer = Writer::new(storage.clone(), "tbl", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![300]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest_s1 = {
            let cat = Catalog::load(storage.as_ref(), "tbl").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let seq_s1 = manifest_s1.sequence;
        assert!(seq_s1 > seq_s, "second commit must advance the sequence");
        let ref_s1 = manifest_s1
            .index_generations
            .get("idx_v")
            .expect("commit S+1 must record an index generation")
            .clone();
        // Incremental persistence: a pure-append INSERT extends the
        // existing generation with an adds-only delta rather than rewriting a new
        // base. So S and S+1 share the SAME base but the generation still differs
        // — S+1 carries an extra delta.
        assert_ne!(
            ref_s, ref_s1,
            "snapshot S and S+1 must reference DIFFERENT index generations"
        );
        assert_eq!(
            ref_s.base, ref_s1.base,
            "the append must reuse the existing base, not rewrite it"
        );
        assert!(
            ref_s.deltas.is_empty() && ref_s1.deltas.len() == 1,
            "S+1 must add exactly one adds-only delta over the shared base"
        );

        // The manifest for S is immutable and its checksum already covers the
        // populated `index_generations` (written exactly once inside the commit).
        assert!(manifest_s.verify_checksum().unwrap());
        assert!(manifest_s1.verify_checksum().unwrap());

        // 4. Load through each manifest's generation and assert per-snapshot view.
        let loaded_s = load_index_by_ref(storage.as_ref(), "tbl", "idx_v", &ref_s)
            .await
            .unwrap()
            .expect("snapshot S index must be loadable");
        assert_eq!(loaded_s.snapshot_sequence, seq_s);
        assert!(
            loaded_s.lookup("300").is_empty(),
            "snapshot S must NOT see value 300 (added at S+1)"
        );
        assert!(
            !loaded_s.lookup("100").is_empty(),
            "snapshot S must see value 100"
        );

        let loaded_s1 = load_index_by_ref(storage.as_ref(), "tbl", "idx_v", &ref_s1)
            .await
            .unwrap()
            .expect("snapshot S+1 index must be loadable");
        // The reused base was built at seq S, so the loaded index reports the
        // base's sequence (the deltas, not the base, carry the S+1 changes).
        assert_eq!(loaded_s1.snapshot_sequence, seq_s);
        assert!(
            !loaded_s1.lookup("300").is_empty(),
            "snapshot S+1 must see value 300"
        );
        assert!(
            !loaded_s1.lookup("100").is_empty(),
            "snapshot S+1 must still see value 100"
        );
    }

    /// A pure-append INSERT writes a small **adds-only delta** (cost
    /// O(rows inserted)) instead of re-serializing the whole index, and the
    /// resulting base+delta is byte-equal to a full rebuild over the new snapshot.
    #[tokio::test]
    async fn insert_writes_incremental_delta_byte_equal_to_full_rebuild() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 100_000,
            row_group_target_bytes: 1 << 30,
            dropped_columns: vec![],
            max_field_id: 0,
        };
        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat.create_table(&guard, "tbl", &schema).await.unwrap();
        dbcat
            .create_index_definition(&guard, "idx_v", "tbl", "v", "btree")
            .await
            .unwrap();
        drop(guard);
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        // Bulk load establishes the index base (one-time full build).
        const N: i64 = 20_000;
        let mut writer = Writer::new(storage.clone(), "tbl", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from((0..N).collect::<Vec<_>>()))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest0 = Catalog::load(storage.as_ref(), "tbl")
            .await
            .unwrap()
            .latest_manifest()
            .unwrap()
            .clone();
        let ref0 = manifest0.index_generations.get("idx_v").unwrap().clone();
        assert!(ref0.deltas.is_empty(), "bulk load establishes a base");
        let base_size = storage
            .size(&format!("tbl/{}", ref0.base.as_ref().unwrap()))
            .await
            .unwrap();

        // Append one row → adds-only delta, not a base rewrite.
        let mut writer = Writer::new(storage.clone(), "tbl", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![N]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest1 = Catalog::load(storage.as_ref(), "tbl")
            .await
            .unwrap()
            .latest_manifest()
            .unwrap()
            .clone();
        let ref1 = manifest1.index_generations.get("idx_v").unwrap().clone();

        // (b) O(rows inserted): same base, one delta, delta ≪ base.
        assert_eq!(ref1.base, ref0.base, "append must reuse the base");
        assert_eq!(ref1.deltas.len(), 1, "append writes exactly one delta");
        let delta_size = storage
            .size(&format!("tbl/{}", ref1.deltas[0]))
            .await
            .unwrap();
        assert!(
            delta_size * 50 < base_size,
            "delta ({delta_size} B) must be O(1), far below base ({base_size} B)"
        );

        // (a) base+delta is byte-equal to a full rebuild over the new snapshot.
        let loaded = load_index_by_ref(storage.as_ref(), "tbl", "idx_v", &ref1)
            .await
            .unwrap()
            .unwrap();
        let definition = IndexDefinition {
            name: "idx_v".into(),
            table: "tbl".into(),
            column: "v".into(),
            unique: false,
        };
        let rebuilt = build_btree_index(storage.as_ref(), &definition, &manifest1)
            .await
            .unwrap();
        assert_eq!(
            loaded.entries.len(),
            rebuilt.entries.len(),
            "incremental index must have the same key set as a full rebuild"
        );
        for k in [0i64, 1, N / 2, N - 1, N] {
            let key = k.to_string();
            let mut got = loaded.lookup(&key).to_vec();
            got.sort_unstable();
            let mut want = rebuilt.lookup(&key).to_vec();
            want.sort_unstable();
            assert_eq!(
                got, want,
                "postings for key {key} must match a full rebuild"
            );
        }
    }

    /// A `replace()` (drops ALL prior fragments) must NOT take the adds-only
    /// INSERT path — it would carry stale entries from the dropped fragments.
    /// The resulting index must be byte-equal to a full rebuild over the new
    /// snapshot (only the replacement rows, no stale keys).
    #[tokio::test]
    async fn replace_rebuilds_index_dropping_stale_entries() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1 << 30,
            dropped_columns: vec![],
            max_field_id: 0,
        };
        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat.create_table(&guard, "tblr", &schema).await.unwrap();
        dbcat
            .create_index_definition(&guard, "idx_v", "tblr", "v", "btree")
            .await
            .unwrap();
        drop(guard);
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        // Initial rows [10, 20, 30] → establishes the base.
        let mut writer = Writer::new(storage.clone(), "tblr", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![10i64, 20, 30]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Replace the whole table with [40, 50].
        let mut writer = Writer::new(storage.clone(), "tblr", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![40i64, 50]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.replace().await.unwrap();

        let manifest = Catalog::load(storage.as_ref(), "tblr")
            .await
            .unwrap()
            .latest_manifest()
            .unwrap()
            .clone();
        let iref = manifest.index_generations.get("idx_v").unwrap().clone();
        // A replace forces a full rebuild → fresh base, no carried deltas.
        assert!(
            iref.deltas.is_empty(),
            "replace must rebuild the base, not append an adds-only delta"
        );

        let loaded = load_index_by_ref(storage.as_ref(), "tblr", "idx_v", &iref)
            .await
            .unwrap()
            .unwrap();
        let definition = IndexDefinition {
            name: "idx_v".into(),
            table: "tblr".into(),
            column: "v".into(),
            unique: false,
        };
        let rebuilt = build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap();
        assert_eq!(
            loaded.entries, rebuilt.entries,
            "index after replace must equal a full rebuild (no stale entries)"
        );
        // Sanity: dropped values gone, replacement values present.
        for stale in ["10", "20", "30"] {
            assert!(
                loaded.lookup(stale).is_empty(),
                "stale key {stale} from dropped fragments must not survive replace"
            );
        }
        assert!(!loaded.lookup("40").is_empty());
        assert!(!loaded.lookup("50").is_empty());
    }

    // ─── deleted_key_tombstoned_in_unique_index ───────────────────────────────

    /// Build a unique index with one entry (key "a@x.com" → row_id 3), append
    /// a tombstone delta for row_id 3, then resolve: the key must yield no
    /// live row_id because the tombstone delta excludes it.
    ///
    /// Specifically proves that:
    ///  - The base index still contains "a@x.com" → [3] (i.e. the delta is what
    ///    clears it, not a rebuild).
    ///  - After applying the delta, `load_index_by_ref` returns `None` for the
    ///    key (entry list is empty → entry is dropped).
    #[tokio::test]
    async fn deleted_key_tombstoned_in_unique_index() {
        let storage = Arc::new(MemoryStorage::new());
        let table = "tbl";
        let index_name = "email_idx";

        // Write a base index with one entry: "a@x.com" → row_id 3.
        let definition = IndexDefinition {
            name: index_name.to_string(),
            table: table.to_string(),
            column: "email".to_string(),
            unique: true,
        };
        let base = BTreeIndex {
            definition: definition.clone(),
            snapshot_sequence: 1,
            entries: {
                let mut m = BTreeMap::new();
                m.insert("a@x.com".to_string(), vec![3u64]);
                m
            },
        };
        // Save the versioned base file.
        let base_rel_path = base.save_versioned(storage.as_ref()).await.unwrap();

        let base_ref = IndexRef {
            base: Some(base_rel_path.clone()),
            deltas: vec![],
        };

        // Confirm the base index still has the entry before tombstoning.
        let pre_tombstone = load_index_by_ref(storage.as_ref(), table, index_name, &base_ref)
            .await
            .unwrap()
            .expect("base index must load");
        assert_eq!(
            pre_tombstone.lookup("a@x.com"),
            &[3u64],
            "base index must contain the entry before tombstoning"
        );

        // Append a tombstone delta for row_id 3 at sequence 5.
        let updated_ref = append_tombstones(
            storage.as_ref(),
            table,
            index_name,
            &[3u64],
            5,
            &base_ref,
            true,
        )
        .await
        .unwrap();

        // The updated ref must have exactly one delta.
        assert_eq!(
            updated_ref.deltas.len(),
            1,
            "exactly one tombstone delta must be appended"
        );
        // The base must be unchanged.
        assert_eq!(updated_ref.base, base_ref.base);

        // Resolving through the updated ref must return None for "a@x.com"
        // because the tombstone delta removes row_id 3.
        let resolved = load_index_by_ref(storage.as_ref(), table, index_name, &updated_ref)
            .await
            .unwrap();
        let empty = resolved.is_none_or(|idx| idx.lookup("a@x.com").is_empty());
        assert!(
            empty,
            "tombstoned key must yield no live row_id after applying the delta"
        );
    }

    // ─── non_unique_index_not_eagerly_tombstoned ──────────────────────────────

    /// After a DELETE commit, a non-unique index on the deleted row's column
    /// must NOT have a tombstone delta added — its generation's deltas list
    /// remains empty.  This verifies that `commit_deletes` only tombstones
    /// unique indexes.
    #[tokio::test]
    async fn non_unique_index_not_eagerly_tombstoned() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat.create_table(&guard, "tbl2", &schema).await.unwrap();
        // Create a NON-unique index (unique = false, the default).
        dbcat
            .create_index_definition(&guard, "v_idx", "tbl2", "v", "btree")
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        // Commit one batch so the index generation is built.
        let mut writer = Writer::new(storage.clone(), "tbl2", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![42i64, 99]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Find the fragment_id and offset for value 42 (offset 0).
        let manifest_before = {
            let cat = Catalog::load(storage.as_ref(), "tbl2").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let frag_id = manifest_before.row_groups[0].fragment_id;
        let deltas_before = manifest_before
            .index_generations
            .get("v_idx")
            .expect("non-unique index generation must exist")
            .deltas
            .clone();
        assert!(
            deltas_before.is_empty(),
            "generation must have no deltas before delete"
        );

        // DELETE offset 0 (value 42).
        let mut writer2 = Writer::new(storage.clone(), "tbl2", schema.clone())
            .await
            .unwrap();
        writer2
            .commit_deletes(HashMap::from([(frag_id, vec![0u32])]))
            .await
            .unwrap();

        // After the delete commit, the non-unique index generation must still
        // have zero deltas.
        let manifest_after = {
            let cat = Catalog::load(storage.as_ref(), "tbl2").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let deltas_after = manifest_after
            .index_generations
            .get("v_idx")
            .expect("non-unique index generation must be preserved")
            .deltas
            .clone();
        assert!(
            deltas_after.is_empty(),
            "non-unique index must NOT have a tombstone delta after delete; got: {:?}",
            deltas_after
        );
    }

    // ─── tombstone_delta_survives_subsequent_commit ───────────────────────────

    /// After a DELETE commit adds a tombstone delta, a subsequent APPEND commit
    /// must NOT delete the tombstone delta file.  The guard is
    /// `manifest_referenced_files`, which walks `index_generations[name].deltas`
    /// and includes each delta path in the protected set — so `cleanup_staging`
    /// spares the file even when a stale intent names it as a candidate for
    /// deletion.
    ///
    /// This test is a genuine regression guard: it injects a synthetic stale
    /// intent whose `"files"` array names the committed tombstone delta.  The
    /// subsequent commit triggers `cleanup_staging`, which will find that intent
    /// and check each file against `referenced_files`.  Because the tombstone
    /// delta is listed there (via `index_ref.deltas`), it is spared.
    ///
    /// If the `index_ref.deltas` walk were removed from `manifest_referenced_files`,
    /// `referenced_files` would no longer contain the tombstone delta path,
    /// `cleanup_staging` would delete it, and the final `exists` assertion would
    /// fail — making this a true regression guard.
    #[tokio::test]
    async fn tombstone_delta_survives_subsequent_commit() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat.create_table(&guard, "tbl3", &schema).await.unwrap();
        // Create a UNIQUE index.
        dbcat
            .create_index_definition_with_options(&guard, "v_uniq", "tbl3", "v", "btree", true)
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        // Commit S: insert [10, 20].
        let mut writer = Writer::new(storage.clone(), "tbl3", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![10i64, 20]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest_s = {
            let cat = Catalog::load(storage.as_ref(), "tbl3").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let frag_id = manifest_s.row_groups[0].fragment_id;

        // DELETE offset 0 (value 10) → tombstone delta appended to v_uniq.
        // `commit_deletes` deletes its own intent on the success path (best-effort),
        // so by the time the next commit's cleanup_staging runs, there is NO
        // surviving real intent that names the tombstone delta.  We must inject
        // one synthetically to genuinely exercise the referenced_files guard.
        let mut writer2 = Writer::new(storage.clone(), "tbl3", schema.clone())
            .await
            .unwrap();
        writer2
            .commit_deletes(HashMap::from([(frag_id, vec![0u32])]))
            .await
            .unwrap();

        let manifest_d = {
            let cat = Catalog::load(storage.as_ref(), "tbl3").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let delta_path = manifest_d
            .index_generations
            .get("v_uniq")
            .expect("unique index generation must exist after delete")
            .deltas
            .first()
            .expect("tombstone delta must be present")
            .clone();

        // Verify the tombstone delta file actually exists in storage.
        let full_delta_path = format!("tbl3/{}", delta_path);
        assert!(
            storage.exists(&full_delta_path).await.unwrap(),
            "tombstone delta file must exist in storage before next commit"
        );

        // ── Inject a synthetic stale intent that names the committed tombstone
        // delta ──────────────────────────────────────────────────────────────
        //
        // This simulates a leftover intent from a hypothetical prior aborted
        // writer (e.g. one that crashed before the pointer swap after writing
        // the tombstone delta).  `cleanup_staging` will find this intent,
        // iterate its `"files"` array, and check each path against
        // `referenced_files`.  Because `delta_path` is in `referenced_files`
        // (it is walked from `manifest_d.index_generations["v_uniq"].deltas`),
        // cleanup spares the file.
        //
        // This proves referenced_files protects committed tombstone deltas;
        // reverting the `index_ref.deltas` walk in `manifest_referenced_files`
        // makes this fail.
        let stale_intent_path = "tbl3/_staging/intents/stale-aborted-op-qa-verify-idx.json";
        let stale_intent = serde_json::json!({
            "txn_id": "stale-aborted-op-qa-verify-idx",
            "started_at": "2020-01-01T00:00:00Z",
            "schema_id": "qa-verify",
            // The bare relative path — exactly as stored in index_ref.deltas and in the
            // referenced_files HashSet — so the contains() check is path-key consistent.
            "files": [delta_path],
        });
        storage
            .write(
                stale_intent_path,
                serde_json::to_vec(&stale_intent).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // Confirm the stale intent is visible before the next commit.
        assert!(
            storage.exists(stale_intent_path).await.unwrap(),
            "stale intent must be present before the next commit triggers cleanup"
        );

        // Commit S+2: append a new row [30].  The commit calls cleanup_staging
        // with the current manifest's referenced_files.  The stale intent above
        // will be processed: cleanup sees `delta_path` in its files array and
        // checks referenced_files.  Because delta_path IS in referenced_files
        // (manifest_referenced_files walks index_ref.deltas), cleanup skips it.
        // If that walk were removed, cleanup would delete the tombstone delta
        // and the assertion below would fail.
        let mut writer3 = Writer::new(storage.clone(), "tbl3", schema.clone())
            .await
            .unwrap();
        writer3
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![30i64]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer3.commit().await.unwrap();

        // This proves referenced_files protects committed tombstone deltas;
        // reverting the `index_ref.deltas` walk in `manifest_referenced_files`
        // makes this fail.
        assert!(
            storage.exists(&full_delta_path).await.unwrap(),
            "tombstone delta file must survive a subsequent commit's cleanup_staging \
             even when a stale intent names it — referenced_files (via index_ref.deltas) \
             must protect it"
        );
    }

    // ─── acceptance tests ────────────────────────────────────────────────

    /// Helper: build a two-column Parquet batch with an `email` (Utf8) column
    /// and a `v` (Int64) column.
    fn make_two_col_batch(emails: &[&str], vs: &[i64]) -> arrow::record_batch::RecordBatch {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use std::sync::Arc;

        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("email", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let email_arr = Arc::new(StringArray::from(emails.to_vec()));
        let v_arr = Arc::new(Int64Array::from(vs.to_vec()));
        arrow::record_batch::RecordBatch::try_new(schema, vec![email_arr, v_arr]).unwrap()
    }

    /// Acceptance test 1: non-indexed-column update writes no index delta.
    ///
    /// Table has an index on `email`; `commit_update` SETs only `v` (not
    /// indexed).  The index on `email`'s `index_generations` entry must be
    /// UNCHANGED (its `deltas` list must not grow).
    #[tokio::test]
    async fn non_indexed_update_writes_no_index_delta() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::writer::{MatchLoc, Writer};
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        // Two-column schema: email (Utf8) + v (Int64).
        let schema = Schema {
            schema_id: 1,
            columns: vec![
                Column::new("email", "utf8", false),
                Column::new("v", "int64", false),
            ],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_table(&guard, "p23_tbl1", &schema)
            .await
            .unwrap();
        // Create an index on `email` only (not on `v`).
        dbcat
            .create_index_definition(&guard, "email_idx", "p23_tbl1", "email", "btree")
            .await
            .unwrap();
        drop(guard);

        // Commit initial batch: row_id 0 = "old@x.com", v=1.
        let mut writer = Writer::new(storage.clone(), "p23_tbl1", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(make_two_col_batch(&["old@x.com"], &[1]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Capture the index generation BEFORE the update.
        let manifest_before = {
            let cat = Catalog::load(storage.as_ref(), "p23_tbl1").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let ref_before = manifest_before
            .index_generations
            .get("email_idx")
            .expect("email_idx generation must exist after initial commit")
            .clone();

        // Find the location of row_id 0 in the manifest.
        let frag_id = manifest_before.row_groups[0].fragment_id;

        // UPDATE only `v` (not `email`): set_columns = ["v"].
        // The email_idx generation must be carried forward UNCHANGED.
        let mut writer2 = Writer::new(storage.clone(), "p23_tbl1", schema.clone())
            .await
            .unwrap();
        let updated_batch = make_two_col_batch(&["old@x.com"], &[99]);
        writer2
            .commit_update(
                updated_batch,
                vec![MatchLoc {
                    fragment_id: frag_id,
                    offset: 0,
                    row_id: 0,
                }],
                &["v".to_string()],
            )
            .await
            .unwrap();

        // Reload the manifest and verify the email_idx generation is UNCHANGED.
        let manifest_after = {
            let cat = Catalog::load(storage.as_ref(), "p23_tbl1").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let ref_after = manifest_after
            .index_generations
            .get("email_idx")
            .expect("email_idx generation must still exist after update")
            .clone();

        assert_eq!(
            ref_before, ref_after,
            "email_idx generation must be UNCHANGED when only non-indexed column v is updated"
        );
    }

    /// Acceptance test 2: updating an indexed column rewrites entries correctly.
    ///
    /// Index on `email`; row_id 0 starts with `email='old@x.com'`.
    /// After `commit_update` SETs `email='new@x.com'`:
    ///   - resolving `'old@x.com'` yields no live row_id 0
    ///   - resolving `'new@x.com'` yields row_id 0
    ///   WITHOUT a full index rebuild — only a tombstone+add delta is written.
    #[tokio::test]
    async fn indexed_update_rewrites_entries() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::writer::{MatchLoc, Writer};
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![
                Column::new("email", "utf8", false),
                Column::new("v", "int64", false),
            ],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_table(&guard, "p23_tbl2", &schema)
            .await
            .unwrap();
        dbcat
            .create_index_definition(&guard, "email_idx2", "p23_tbl2", "email", "btree")
            .await
            .unwrap();
        drop(guard);

        // Commit initial batch: row_id 0 = "old@x.com", v=1.
        let mut writer = Writer::new(storage.clone(), "p23_tbl2", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(make_two_col_batch(&["old@x.com"], &[1]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest_before = {
            let cat = Catalog::load(storage.as_ref(), "p23_tbl2").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let frag_id = manifest_before.row_groups[0].fragment_id;

        // UPDATE `email` to 'new@x.com': set_columns = ["email"].
        let mut writer2 = Writer::new(storage.clone(), "p23_tbl2", schema.clone())
            .await
            .unwrap();
        let updated_batch = make_two_col_batch(&["new@x.com"], &[1]);
        writer2
            .commit_update(
                updated_batch,
                vec![MatchLoc {
                    fragment_id: frag_id,
                    offset: 0,
                    row_id: 0,
                }],
                &["email".to_string()],
            )
            .await
            .unwrap();

        // Reload the manifest and check the index generation has a new delta.
        let manifest_after = {
            let cat = Catalog::load(storage.as_ref(), "p23_tbl2").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let ref_after = manifest_after
            .index_generations
            .get("email_idx2")
            .expect("email_idx2 generation must exist after update")
            .clone();

        // The base must still be the original (no full rebuild).
        let ref_before = manifest_before
            .index_generations
            .get("email_idx2")
            .expect("email_idx2 generation must exist before update")
            .clone();
        assert_eq!(
            ref_after.base, ref_before.base,
            "base index must not be replaced — only a delta is added"
        );
        assert_eq!(
            ref_after.deltas.len(),
            1,
            "exactly one delta must be appended"
        );

        // Resolve through the updated IndexRef.
        let resolved = load_index_by_ref(storage.as_ref(), "p23_tbl2", "email_idx2", &ref_after)
            .await
            .unwrap()
            .expect("index must be loadable");

        assert!(
            resolved.lookup("old@x.com").is_empty(),
            "old@x.com must yield no live row after the update (tombstone removes row_id 0)"
        );
        assert_eq!(
            resolved.lookup("new@x.com"),
            &[0u64],
            "new@x.com must yield row_id 0 after the update (add inserts it)"
        );
    }

    // ─── unique-index enforcement tests (M01) ───────────────────────────────

    /// Creating a unique index over data that already contains duplicate live
    /// keys must fail with `UniqueKeyViolation`.
    #[tokio::test]
    async fn unique_index_creation_rejects_duplicate_live_keys() {
        use crate::catalog::Catalog;
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1 << 30,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_table(&guard, "uniq_create", &schema)
            .await
            .unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard,
                "v_uniq",
                "uniq_create",
                "v",
                "btree",
                true,
            )
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        // Insert duplicate live keys.
        let mut writer = Writer::new(storage.clone(), "uniq_create", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1i64, 1, 2]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();

        let err = writer.commit().await.unwrap_err();
        assert!(
            matches!(
                err,
                IcefallDBError::UniqueKeyViolation {
                    ref table,
                    ref index,
                    ref key,
                } if table == "uniq_create" && index == "v_uniq" && key == "1"
            ),
            "expected UniqueKeyViolation for duplicate key 1, got {err:?}"
        );

        // The table must remain empty because the commit was rejected.
        let latest = Catalog::load(storage.as_ref(), "uniq_create")
            .await
            .unwrap()
            .latest_manifest()
            .cloned();
        assert!(
            latest.map(|m| m.row_groups.is_empty()).unwrap_or(true),
            "commit must be rejected; table should be empty"
        );
    }

    /// Appending rows whose key already exists as a live row must fail with
    /// `UniqueKeyViolation`.
    #[tokio::test]
    async fn unique_index_append_rejects_duplicate_keys() {
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1 << 30,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_table(&guard, "uniq_append", &schema)
            .await
            .unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard,
                "v_uniq",
                "uniq_append",
                "v",
                "btree",
                true,
            )
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        let mut writer = Writer::new(storage.clone(), "uniq_append", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1i64, 2]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Append a duplicate key.
        let mut writer = Writer::new(storage.clone(), "uniq_append", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1i64]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();

        let err = writer.commit().await.unwrap_err();
        assert!(
            matches!(
                err,
                IcefallDBError::UniqueKeyViolation {
                    ref table,
                    ref index,
                    ref key,
                } if table == "uniq_append" && index == "v_uniq" && key == "1"
            ),
            "expected UniqueKeyViolation for duplicate key 1 on append, got {err:?}"
        );
    }

    /// Re-inserting a key whose only live row has been deleted must succeed.
    #[tokio::test]
    async fn unique_index_allows_reinsert_after_delete() {
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::Writer;
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::collections::HashMap;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1 << 30,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_table(&guard, "uniq_del", &schema)
            .await
            .unwrap();
        dbcat
            .create_index_definition_with_options(&guard, "v_uniq", "uniq_del", "v", "btree", true)
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        let mut writer = Writer::new(storage.clone(), "uniq_del", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1i64, 2]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Delete the row with value 1 (physical offset 0).
        let manifest_before = {
            use crate::catalog::Catalog;
            let cat = Catalog::load(storage.as_ref(), "uniq_del").await.unwrap();
            cat.latest_manifest().unwrap().clone()
        };
        let frag_id = manifest_before.row_groups[0].fragment_id;

        let mut writer = Writer::new(storage.clone(), "uniq_del", schema.clone())
            .await
            .unwrap();
        writer
            .commit_deletes(HashMap::from([(frag_id, vec![0u32])]))
            .await
            .unwrap();

        // Re-insert the deleted key.
        let mut writer = Writer::new(storage.clone(), "uniq_del", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1i64]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        use crate::catalog::Catalog;
        let manifest = Catalog::load(storage.as_ref(), "uniq_del")
            .await
            .unwrap()
            .latest_manifest()
            .unwrap()
            .clone();
        let iref = manifest
            .index_generations
            .get("v_uniq")
            .expect("unique index generation must exist")
            .clone();
        let index = load_index_by_ref(storage.as_ref(), "uniq_del", "v_uniq", &iref)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(index.lookup("1").len(), 1, "exactly one live row for key 1");
        assert_eq!(index.lookup("2").len(), 1, "exactly one live row for key 2");
    }

    /// Updating a unique-indexed column to a value that already belongs to a
    /// different live row must fail with `UniqueKeyViolation`.
    #[tokio::test]
    async fn unique_index_update_rejects_collision() {
        use crate::database_catalog::DatabaseCatalog;
        use crate::metadata::{Column, Schema};
        use crate::writer::{MatchLoc, Writer};
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use arrow::record_batch::RecordBatch;
        use std::time::Duration;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        let schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1 << 30,
            dropped_columns: vec![],
            max_field_id: 0,
        };

        let dbcat = DatabaseCatalog::new(storage.clone());
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_table(&guard, "uniq_upd", &schema)
            .await
            .unwrap();
        dbcat
            .create_index_definition_with_options(&guard, "v_uniq", "uniq_upd", "v", "btree", true)
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));

        let mut writer = Writer::new(storage.clone(), "uniq_upd", schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1i64, 2]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Attempt to UPDATE row_id 0 (value 1) to value 2, colliding with row_id 1.
        let mut writer = Writer::new(storage.clone(), "uniq_upd", schema.clone())
            .await
            .unwrap();
        let updated_batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![2i64]))],
        )
        .unwrap();
        let err = writer
            .commit_update(
                updated_batch,
                vec![MatchLoc {
                    fragment_id: 0,
                    offset: 0,
                    row_id: 0,
                }],
                &["v".to_string()],
            )
            .await
            .unwrap_err();

        assert!(
            matches!(
                err,
                IcefallDBError::UniqueKeyViolation {
                    ref table,
                    ref index,
                    ref key,
                } if table == "uniq_upd" && index == "v_uniq" && key == "2"
            ),
            "expected UniqueKeyViolation for update collision on key 2, got {err:?}"
        );
    }
}
