use crate::metadata::Manifest;
use crate::storage::Storage;
use crate::writer::Writer;
use crate::{is_not_found, IcefallDBError, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Duration;

/// Result of a garbage-collection run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GcResult {
    /// Paths of files deleted by the garbage collector, relative to the storage root.
    pub deleted: Vec<String>,
    /// Sequence numbers of the snapshots that were retained.
    pub retained_snapshots: Vec<u64>,
}

/// Garbage collector for IcefallDB tables.
///
/// Acquires the exclusive writer lock, validates manifest snapshots, and removes
/// unreferenced row-group files, stale staging artifacts, and old manifest
/// snapshots while retaining all valid snapshots with sequence numbers no older
/// than `current_sequence - (retain_snapshots - 1)`.
pub struct GarbageCollector<'a> {
    storage: &'a dyn Storage,
    table: String,
    retain_snapshots: usize,
}

impl<'a> GarbageCollector<'a> {
    /// Create a new garbage collector for `table`.
    ///
    /// `retain_snapshots` is the number of recent snapshots to retain, measured
    /// as a sequence-window from the current catalog sequence. A value of `0`
    /// keeps all valid snapshots.
    pub fn new(storage: &'a dyn Storage, table: &str, retain_snapshots: usize) -> Self {
        Self {
            storage,
            table: table.to_string(),
            retain_snapshots,
        }
    }

    /// Run garbage collection.
    ///
    /// Returns the list of deleted file paths and the sequence numbers of the
    /// retained snapshots.
    pub async fn run(&self) -> Result<GcResult> {
        let lock_path = format!("{}/_write.lock", self.table);
        let _guard = self
            .storage
            .lock_exclusive(&lock_path, Duration::from_secs(30))
            .await?;

        // Fold any pending mutation WAL into the pointer first: GC computes the
        // referenced set from the pointer manifest, so a lagging pointer would
        // make it collect deletion-vector files the WAL still references. No-op
        // when no `_wal/` log exists (the default).
        crate::mutation_wal::checkpoint_locked(self.storage, &self.table).await?;

        let (valid_snapshots, mut sequences) = self.load_valid_snapshots().await?;
        if valid_snapshots.is_empty() {
            return Ok(GcResult {
                deleted: Vec::new(),
                retained_snapshots: Vec::new(),
            });
        }

        sequences.sort_unstable_by(|a, b| b.cmp(a));

        let current_sequence = self
            .resolve_current_sequence(&valid_snapshots, &sequences)
            .await?;

        let cutoff_seq = if self.retain_snapshots == 0 {
            0
        } else {
            current_sequence.saturating_sub(self.retain_snapshots.saturating_sub(1) as u64)
        };

        let mut retained_sequences: Vec<u64> = sequences
            .iter()
            .copied()
            .filter(|seq| *seq >= cutoff_seq)
            .collect();

        // The snapshot referenced by `_manifest.json` must never be collected,
        // even if it falls below the retention cutoff.
        if !retained_sequences.contains(&current_sequence) {
            retained_sequences.push(current_sequence);
            retained_sequences.sort_unstable_by(|a, b| b.cmp(a));
        }

        // Build the full referenced set by unioning manifest_referenced_files
        // across all retained snapshots. This covers data, meta, .agg, .del,
        // row-index (.idx), checkpoint (.json/.rkyv), and secondary-index files
        // — the same set the commit path considers live — so the GC can never
        // delete a file that is still referenced by any retained snapshot.
        //
        // manifest_referenced_files returns BARE relative paths (no table/ prefix).
        // We store them prefixed so later `referenced.contains(&entry)` where
        // `entry` is a full path returned by storage.list works correctly.
        let mut referenced: HashSet<String> = HashSet::new();
        for seq in &retained_sequences {
            if let Some(manifest) = valid_snapshots.get(seq) {
                for bare in Writer::manifest_referenced_files(manifest) {
                    referenced.insert(format!("{}/{}", self.table, bare));
                }
            }
        }

        let mut deleted = Vec::new();

        // Delete unreferenced row-group files in the table root.
        let root_entries = match self.storage.list(&self.table).await {
            Ok(entries) => entries,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        let mut root_deleted = false;
        for entry in root_entries {
            let filename = Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            let is_row_group = filename.starts_with("rg_")
                && (filename.ends_with(".parquet")
                    || filename.ends_with(".meta")
                    || filename.ends_with(".agg"));
            let is_temp = filename.ends_with(".json.tmp");
            if is_temp || (is_row_group && !referenced.contains(&entry)) {
                self.storage.delete(&entry).await?;
                root_deleted = true;
                deleted.push(entry);
            }
        }
        if root_deleted {
            self.storage.sync(&format!("{}/", self.table)).await?;
        }

        // Delete orphan .del files in the _deletions/ subdirectory.
        // .del paths are stored as `_deletions/rg_<id>__v<n>.del` in the manifest;
        // `manifest_referenced_files` includes them so a live .del can never be
        // collected here.
        let deletions_dir = format!("{}/_deletions", self.table);
        let del_entries = match self.storage.list(&deletions_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        let mut dels_deleted = false;
        for entry in del_entries {
            let filename = Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if filename.ends_with(".del") && !referenced.contains(&entry) {
                self.storage.delete(&entry).await?;
                dels_deleted = true;
                deleted.push(entry);
            }
        }
        if dels_deleted {
            self.storage.sync(&deletions_dir).await?;
        }

        // Delete orphan generated metadata artifacts whose paths are represented
        // in `manifest_referenced_files`. These directories were historically
        // protected by the referenced set but not swept, so superseded row-index
        // generations, checkpoints, and versioned secondary-index generations
        // accumulated forever.
        self.delete_unreferenced_direct_files(
            &format!("{}/_rowindex", self.table),
            &referenced,
            &[".idx"],
            &mut deleted,
        )
        .await?;
        self.delete_unreferenced_direct_files(
            &format!("{}/_checkpoints", self.table),
            &referenced,
            &[".json", ".rkyv", ".tmp"],
            &mut deleted,
        )
        .await?;
        self.delete_unreferenced_index_generations(&referenced, &mut deleted)
            .await?;

        // Delete orphan .part files in staging directories.
        for staging_dir in ["_staging/incoming", "_staging/compact"] {
            let dir = format!("{}/{}", self.table, staging_dir);
            let mut parts_deleted = false;
            let entries = match self.storage.list(&dir).await {
                Ok(e) => e,
                Err(e) if is_not_found(&e) => continue,
                Err(e) => return Err(e),
            };
            for entry in entries {
                if entry.ends_with(".part") {
                    self.storage.delete(&entry).await?;
                    parts_deleted = true;
                    deleted.push(entry);
                }
            }
            if parts_deleted {
                self.storage.sync(&dir).await?;
            }
        }

        // Delete stale intent files.
        let intents_dir = format!("{}/_staging/intents", self.table);
        let mut intents_deleted = false;
        let intent_entries = match self.storage.list(&intents_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        for entry in intent_entries {
            if entry.ends_with(".json") {
                self.storage.delete(&entry).await?;
                intents_deleted = true;
                deleted.push(entry);
            }
        }
        if intents_deleted {
            self.storage.sync(&intents_dir).await?;
        }

        // Delete manifest snapshots older than the cutoff, plus any leftover
        // temporary manifest files.
        let manifests_dir = format!("{}/_manifests", self.table);
        let mut manifests_deleted = false;
        let manifest_entries = match self.storage.list(&manifests_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        for entry in manifest_entries {
            let filename = Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if filename.ends_with(".json.tmp") {
                self.storage.delete(&entry).await?;
                manifests_deleted = true;
                deleted.push(entry);
                continue;
            }
            if !filename.ends_with(".json") {
                continue;
            }
            let Some(seq_str) = filename.strip_suffix(".json") else {
                continue;
            };
            let Ok(seq) = seq_str.parse::<u64>() else {
                continue;
            };
            if seq < cutoff_seq && self.storage.exists(&entry).await? {
                self.storage.delete(&entry).await?;
                manifests_deleted = true;
                deleted.push(entry);
            }
        }
        if manifests_deleted {
            self.storage.sync(&manifests_dir).await?;
        }

        Ok(GcResult {
            deleted,
            retained_snapshots: retained_sequences,
        })
    }

    async fn delete_unreferenced_direct_files(
        &self,
        dir: &str,
        referenced: &HashSet<String>,
        suffixes: &[&str],
        deleted: &mut Vec<String>,
    ) -> Result<()> {
        let entries = match self.storage.list(dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok(()),
            Err(e) => return Err(e),
        };
        let mut any_deleted = false;
        for entry in entries {
            let filename = Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if suffixes.iter().any(|suffix| filename.ends_with(suffix))
                && !referenced.contains(&entry)
            {
                self.storage.delete(&entry).await?;
                any_deleted = true;
                deleted.push(entry);
            }
        }
        if any_deleted {
            self.storage.sync(dir).await?;
        }
        Ok(())
    }

    async fn delete_unreferenced_index_generations(
        &self,
        referenced: &HashSet<String>,
        deleted: &mut Vec<String>,
    ) -> Result<()> {
        let indexes_dir = format!("{}/_indexes", self.table);
        let entries = match self.storage.list(&indexes_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok(()),
            Err(e) => return Err(e),
        };

        for entry in entries {
            let children = match self.storage.list(&entry).await {
                Ok(c) => c,
                // Top-level legacy files such as `_indexes/name.json` are live
                // for old manifests but are not recorded in `index_generations`.
                // Do not infer liveness for them here.
                Err(e) if is_not_found(&e) => continue,
                Err(e) => return Err(e),
            };

            let mut any_deleted = false;
            for child in children {
                let filename = Path::new(&child)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                let is_generated = filename.ends_with(".json")
                    || filename.ends_with(".idx")
                    || filename.ends_with(".model")
                    || filename.ends_with(".tmp");
                if is_generated && !referenced.contains(&child) {
                    self.storage.delete(&child).await?;
                    any_deleted = true;
                    deleted.push(child);
                }
            }
            if any_deleted {
                self.storage.sync(&entry).await?;
            }
        }
        Ok(())
    }

    /// Resolve the sequence number that defines the retention window.
    ///
    /// If `_manifest.json` exists and points to a valid snapshot, that sequence
    /// is used. Otherwise the highest valid snapshot is chosen and the pointer
    /// is atomically repaired to point to it.
    async fn resolve_current_sequence(
        &self,
        valid_snapshots: &HashMap<u64, Manifest>,
        sequences: &[u64],
    ) -> Result<u64> {
        match read_pointer(self.storage, &self.table).await {
            Some(seq) if valid_snapshots.contains_key(&seq) => Ok(seq),
            _ => {
                let seq = sequences[0];
                write_manifest_pointer(self.storage, &self.table, seq).await?;
                Ok(seq)
            }
        }
    }

    /// Load and validate all manifest snapshots.
    ///
    /// Returns a map of valid snapshots keyed by sequence and a vector of all
    /// valid sequence numbers. Invalid snapshots are silently ignored.
    async fn load_valid_snapshots(&self) -> Result<(HashMap<u64, Manifest>, Vec<u64>)> {
        let manifests_dir = format!("{}/_manifests", self.table);
        let entries = match self.storage.list(&manifests_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok((HashMap::new(), Vec::new())),
            Err(e) => return Err(e),
        };

        let mut valid = HashMap::new();
        let mut sequences = Vec::new();

        for entry in entries {
            let filename = Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !filename.ends_with(".json") {
                continue;
            }
            let Some(stem) = filename.strip_suffix(".json") else {
                continue;
            };
            let Ok(seq) = stem.parse::<u64>() else {
                continue;
            };

            let data = match self.storage.read(&entry).await {
                Ok(d) => d,
                Err(e) if is_not_found(&e) => continue,
                Err(IcefallDBError::Io(e)) => return Err(IcefallDBError::Io(e)),
                Err(_) => continue,
            };

            let manifest: Manifest = match serde_json::from_slice(&data) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if manifest.sequence != seq {
                continue;
            }
            if manifest.format_version != 1 {
                continue;
            }
            let checksum_ok = match manifest.verify_checksum() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if !checksum_ok {
                continue;
            }

            valid.insert(seq, manifest);
            sequences.push(seq);
        }

        Ok((valid, sequences))
    }
}

/// Read the `latest` value from `_manifest.json`, if it exists and is well-formed.
async fn read_pointer(storage: &dyn Storage, table: &str) -> Option<u64> {
    let pointer_path = format!("{}/_manifest.json", table);
    let data = storage.read(&pointer_path).await.ok()?;
    let value: serde_json::Value = serde_json::from_slice(&data).ok()?;
    value.get("latest")?.as_u64()
}

/// Atomically write `_manifest.json` to point to `seq`.
async fn write_manifest_pointer(storage: &dyn Storage, table: &str, seq: u64) -> Result<()> {
    let pointer_path = format!("{}/_manifest.json", table);
    let tmp_path = format!("{}.tmp", pointer_path);
    let pointer = serde_json::json!({ "latest": seq });
    storage
        .write(&tmp_path, serde_json::to_vec(&pointer)?.as_slice())
        .await?;
    storage.sync(&tmp_path).await?;
    storage.rename(&tmp_path, &pointer_path).await?;
    storage.sync(&format!("{}/", table)).await?;
    Ok(())
}
