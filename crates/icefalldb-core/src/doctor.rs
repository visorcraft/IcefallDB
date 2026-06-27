use crate::metadata::{Manifest, Schema};
use crate::storage::Storage;
use crate::{is_not_found, IcefallDBError, Result};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Duration;

#[cfg(feature = "encryption")]
use crate::encryption::SchemaEncryptionMarker;

/// Kind of repair action performed by [`Doctor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionKind {
    Deleted,
    RolledBack,
    PointerUpdated,
    OrphanRemoved,
    Skipped,
    /// A missing row-group metadata sidecar was regenerated from the Parquet footer.
    Regenerated,
    /// A problem was found that `Doctor` cannot safely repair automatically.
    Unrepairable,
}

impl fmt::Display for ActionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActionKind::Deleted => write!(f, "Deleted"),
            ActionKind::RolledBack => write!(f, "RolledBack"),
            ActionKind::PointerUpdated => write!(f, "PointerUpdated"),
            ActionKind::OrphanRemoved => write!(f, "OrphanRemoved"),
            ActionKind::Skipped => write!(f, "Skipped"),
            ActionKind::Regenerated => write!(f, "Regenerated"),
            ActionKind::Unrepairable => write!(f, "Unrepairable"),
        }
    }
}

fn is_repair_mutation(kind: &ActionKind) -> bool {
    matches!(
        kind,
        ActionKind::RolledBack
            | ActionKind::PointerUpdated
            | ActionKind::OrphanRemoved
            | ActionKind::Regenerated
    )
}

/// A single action taken while repairing a table.
#[derive(Debug, Clone)]
pub struct RepairAction {
    pub kind: ActionKind,
    pub path: String,
    pub detail: String,
}

/// Result of running [`Doctor::repair`] on a table.
#[derive(Debug, Clone)]
pub struct RepairResult {
    pub table: String,
    /// `true` if the repair mutated any state.
    pub repaired: bool,
    /// `false` when unrepairable issues remain; the caller should not treat the
    /// table as fully healthy even if `repaired` is `true`.
    pub healthy: bool,
    pub actions: Vec<RepairAction>,
}

/// Kind of issue found by [`Doctor::diagnose`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagnosisKind {
    StaleIntent,
    InvalidPointer,
    OrphanRowGroup,
    OrphanStagedPart,
    InvalidManifestSnapshot,
    NewerInvalidManifest,
    MissingRowGroupMeta,
    Info,
}

impl fmt::Display for DiagnosisKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosisKind::StaleIntent => write!(f, "StaleIntent"),
            DiagnosisKind::InvalidPointer => write!(f, "InvalidPointer"),
            DiagnosisKind::OrphanRowGroup => write!(f, "OrphanRowGroup"),
            DiagnosisKind::OrphanStagedPart => write!(f, "OrphanStagedPart"),
            DiagnosisKind::InvalidManifestSnapshot => write!(f, "InvalidManifestSnapshot"),
            DiagnosisKind::NewerInvalidManifest => write!(f, "NewerInvalidManifest"),
            DiagnosisKind::MissingRowGroupMeta => write!(f, "MissingRowGroupMeta"),
            DiagnosisKind::Info => write!(f, "Info"),
        }
    }
}

/// A single issue found while diagnosing a table.
#[derive(Debug, Clone)]
pub struct DiagnosisIssue {
    pub kind: DiagnosisKind,
    pub path: String,
    pub detail: String,
}

/// Result of running [`Doctor::diagnose`] on a table.
#[derive(Debug, Clone)]
pub struct DiagnosisResult {
    pub table: String,
    pub healthy: bool,
    pub issues: Vec<DiagnosisIssue>,
}

/// Description of an invalid manifest snapshot encountered while validating.
struct InvalidSnapshot {
    path: String,
    reason: String,
}

/// Repair tool for IcefallDB tables.
///
/// `Doctor` acquires the exclusive writer lock, rolls back stale intents,
/// validates manifest snapshots, repairs the manifest pointer, and removes
/// orphan files. All mutations happen while the lock is held.
pub struct Doctor<'a> {
    storage: &'a dyn Storage,
    table: String,
    lock_timeout: Duration,
}

impl<'a> Doctor<'a> {
    /// Create a new doctor for `table` using `storage`.
    ///
    /// The default writer lock timeout is 30 seconds.
    pub fn new(storage: &'a dyn Storage, table: &str) -> Self {
        Self {
            storage,
            table: table.to_string(),
            lock_timeout: Duration::from_secs(30),
        }
    }

    /// Override the writer lock timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.lock_timeout = timeout;
        self
    }

    /// Check whether `path` exists, treating `NotFound` errors as `false` and
    /// propagating all other errors.
    async fn exists_resolving_not_found(&self, path: &str) -> Result<bool> {
        match self.storage.exists(path).await {
            Ok(exists) => Ok(exists),
            Err(e) if is_not_found(&e) => Ok(false),
            Err(e) => Err(e),
        }
    }

    /// Diagnose the table without modifying any state.
    ///
    /// This method does not acquire the writer lock and performs only read-only
    /// inspection. It reports stale intents, pointer problems, orphan files, and
    /// invalid manifest snapshots.
    pub async fn diagnose(&self) -> Result<DiagnosisResult> {
        crate::reader::require_table_exists(self.storage, &self.table).await?;

        let mut issues = Vec::new();

        self.diagnose_intents(&mut issues).await?;

        let (valid_snapshots, invalid_snapshots) = self.validate_snapshots().await?;
        for invalid in &invalid_snapshots {
            issues.push(DiagnosisIssue {
                kind: DiagnosisKind::InvalidManifestSnapshot,
                path: invalid.path.clone(),
                detail: format!("invalid manifest snapshot: {}", invalid.reason),
            });
        }

        if valid_snapshots.is_empty() {
            let pointer_path = format!("{}/_manifest.json", self.table);
            let schema_pointer_path = format!("{}/_schema.json", self.table);
            if self.exists_resolving_not_found(&pointer_path).await? {
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::InvalidPointer,
                    path: "_manifest.json".into(),
                    detail: "pointer exists but no valid manifest snapshots".into(),
                });
            } else if self
                .exists_resolving_not_found(&schema_pointer_path)
                .await?
            {
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::InvalidPointer,
                    path: "_manifest.json".into(),
                    detail: "manifest pointer missing but schema pointer exists".into(),
                });
            } else if issues.iter().all(|i| i.kind == DiagnosisKind::Info) {
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::Info,
                    path: "_manifest.json".into(),
                    detail: "empty table".into(),
                });
            }

            self.diagnose_orphans(0, &valid_snapshots, &mut issues)
                .await?;

            return Ok(DiagnosisResult {
                table: self.table.clone(),
                healthy: issues.iter().all(|i| i.kind == DiagnosisKind::Info),
                issues,
            });
        }

        let (chosen_seq, current_pointer) = self.choose_sequence(&valid_snapshots).await?;
        if current_pointer != Some(chosen_seq) {
            issues.push(DiagnosisIssue {
                kind: DiagnosisKind::InvalidPointer,
                path: "_manifest.json".into(),
                detail: format!(
                    "pointer {:?} does not point to highest valid sequence {}",
                    current_pointer, chosen_seq
                ),
            });
        }

        self.diagnose_orphans(chosen_seq, &valid_snapshots, &mut issues)
            .await?;

        self.diagnose_row_group_metas(&valid_snapshots, &mut issues)
            .await?;

        self.diagnose_chain(&mut issues).await?;

        Ok(DiagnosisResult {
            table: self.table.clone(),
            healthy: issues.iter().all(|i| i.kind == DiagnosisKind::Info),
            issues,
        })
    }

    /// Repair the table and return a description of all actions taken.
    pub async fn repair(&self) -> Result<RepairResult> {
        crate::reader::require_table_exists(self.storage, &self.table).await?;

        let lock_path = format!("{}/_write.lock", self.table);
        let _guard = self
            .storage
            .lock_exclusive(&lock_path, self.lock_timeout)
            .await?;

        let mut actions = Vec::new();

        let (valid_snapshots, invalid_snapshots) = self.validate_snapshots().await?;
        if valid_snapshots.is_empty() {
            // If the table has no pointer and no manifest snapshots, it is a
            // genuinely empty table: clean up any stray intents/orphans and
            // report a no-op. If a pointer or manifest files exist but are all
            // invalid, that is corruption we cannot safely repair.
            let pointer_path = format!("{}/_manifest.json", self.table);
            let has_pointer = self.exists_resolving_not_found(&pointer_path).await?;
            let has_manifests = !invalid_snapshots.is_empty();
            if !has_pointer && !has_manifests {
                let referenced = HashSet::new();
                self.rollback_intents(&referenced, &mut actions).await?;
                let orphan_actions = self.delete_orphans(0, &valid_snapshots).await?;
                actions.extend(orphan_actions);
                return Ok(RepairResult {
                    table: self.table.clone(),
                    repaired: actions.iter().any(|a| is_repair_mutation(&a.kind)),
                    healthy: !actions.iter().any(|a| a.kind == ActionKind::Unrepairable),
                    actions,
                });
            }
            return Err(IcefallDBError::Other(Box::new(std::io::Error::other(
                "no valid manifest snapshots",
            ))));
        }

        let (chosen_seq, current_pointer) = self.choose_sequence(&valid_snapshots).await?;

        // `_schema.json` is the authoritative marker that a table exists. Do not
        // rebuild a manifest pointer if the schema pointer is missing, because
        // that would leave the table in a partially initialized state.
        let schema_pointer_path = format!("{}/_schema.json", self.table);
        if !self
            .exists_resolving_not_found(&schema_pointer_path)
            .await?
        {
            return Err(IcefallDBError::SchemaNotFound {
                path: schema_pointer_path,
            });
        }

        let referenced = referenced_files(valid_snapshots.values());

        self.rollback_intents(&referenced, &mut actions).await?;

        for invalid in &invalid_snapshots {
            actions.push(RepairAction {
                kind: ActionKind::Skipped,
                path: invalid.path.clone(),
                detail: format!("invalid manifest snapshot: {}", invalid.reason),
            });
        }

        if current_pointer != Some(chosen_seq) {
            self.write_pointer(chosen_seq).await?;
            actions.push(RepairAction {
                kind: ActionKind::PointerUpdated,
                path: "_manifest.json".into(),
                detail: format!("updated pointer to sequence {}", chosen_seq),
            });
        }

        let manifest = valid_snapshots
            .get(&chosen_seq)
            .expect("chosen_seq is a valid snapshot");
        self.repair_row_group_metas(manifest, &mut actions).await?;

        let orphan_actions = self.delete_orphans(chosen_seq, &valid_snapshots).await?;
        actions.extend(orphan_actions);

        Ok(RepairResult {
            table: self.table.clone(),
            repaired: actions.iter().any(|a| is_repair_mutation(&a.kind)),
            healthy: !actions.iter().any(|a| a.kind == ActionKind::Unrepairable),
            actions,
        })
    }

    /// Roll back stale intent files and delete the files they reference.
    ///
    /// Because [`Doctor::repair`] holds the exclusive writer lock, any intent
    /// file present in `_staging/intents/` must be stale. A live writer would
    /// have had to release the lock before its intent could be considered
    /// committed, and the exclusive lock prevents concurrent writers from
    /// creating new intents while repair runs. Per-process lock verification is
    /// therefore not required in v1.
    ///
    /// Files listed in the intent that are also present in `referenced` are
    /// preserved: they belong to a retained valid manifest snapshot and must
    /// not be deleted.
    async fn rollback_intents(
        &self,
        referenced: &HashSet<String>,
        actions: &mut Vec<RepairAction>,
    ) -> Result<()> {
        let intents_dir = format!("{}/_staging/intents", self.table);
        let entries = match self.storage.list(&intents_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok(()),
            Err(e) => return Err(e),
        };

        let mut intents_deleted = false;

        for entry in entries {
            if !entry.ends_with(".json") {
                continue;
            }

            let intent_data = match self.storage.read(&entry).await {
                Ok(d) => d,
                Err(e) if is_not_found(&e) => continue,
                Err(e) => return Err(e),
            };

            let mut deleted_count = 0usize;
            let mut skipped_count = 0usize;
            let mut txn_id: Option<String> = None;

            if let Ok(intent) = serde_json::from_slice::<serde_json::Value>(&intent_data) {
                txn_id = intent
                    .get("txn_id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                if let Some(files) = intent.get("files").and_then(|v| v.as_array()) {
                    for file in files {
                        if let Some(filename) = file.as_str() {
                            let file_path = format!("{}/{}", self.table, filename);
                            let rel = strip_table_prefix(&self.table, &file_path);
                            if referenced.contains(&rel) {
                                actions.push(RepairAction {
                                    kind: ActionKind::Skipped,
                                    path: rel,
                                    detail: "referenced by a retained valid snapshot".into(),
                                });
                                skipped_count += 1;
                                continue;
                            }
                            match self.storage.delete(&file_path).await {
                                Ok(()) => deleted_count += 1,
                                Err(e) if is_not_found(&e) => {}
                                Err(e) => return Err(e),
                            }
                        }
                    }
                }
            }

            match self.storage.delete(&entry).await {
                Ok(()) => intents_deleted = true,
                Err(e) if is_not_found(&e) => {}
                Err(e) => return Err(e),
            }

            let table_relative = strip_table_prefix(&self.table, &entry);
            let detail = match txn_id {
                Some(id) => format!(
                    "rolled back stale intent {}, deleted {} listed files, skipped {} referenced files",
                    id, deleted_count, skipped_count
                ),
                None => format!(
                    "rolled back stale intent, deleted {} listed files, skipped {} referenced files",
                    deleted_count, skipped_count
                ),
            };
            actions.push(RepairAction {
                kind: ActionKind::RolledBack,
                path: table_relative,
                detail,
            });
        }

        if intents_deleted {
            self.storage.sync(&intents_dir).await?;
        }

        Ok(())
    }

    /// Detect stale intent files without modifying state.
    async fn diagnose_intents(&self, issues: &mut Vec<DiagnosisIssue>) -> Result<()> {
        let intents_dir = format!("{}/_staging/intents", self.table);
        let entries = match self.storage.list(&intents_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok(()),
            Err(e) => return Err(e),
        };

        for entry in entries {
            if !entry.ends_with(".json") {
                continue;
            }

            let table_relative = strip_table_prefix(&self.table, &entry);
            issues.push(DiagnosisIssue {
                kind: DiagnosisKind::StaleIntent,
                path: table_relative,
                detail: "stale intent file found".into(),
            });
        }

        Ok(())
    }

    /// Verify the manifest hash chain and append chain-break issues.
    ///
    /// `--repair` does not rewrite history; this is a read-only check.
    async fn diagnose_chain(&self, issues: &mut Vec<DiagnosisIssue>) -> Result<()> {
        let history = verify_history(self.storage, &self.table).await?;
        if history.intact {
            if history.oldest > 0 || history.latest > 0 {
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::Info,
                    path: "_manifests".into(),
                    detail: format!("chain intact [{}..{}]", history.oldest, history.latest),
                });
            }
        } else {
            for b in &history.breaks {
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::InvalidManifestSnapshot,
                    path: format!("_manifests/{:09}.json", b.sequence),
                    detail: format!("chain break: {}", b.reason),
                });
            }
        }
        Ok(())
    }

    /// Detect missing row-group metadata sidecars.
    async fn diagnose_row_group_metas(
        &self,
        valid_snapshots: &HashMap<u64, Manifest>,
        issues: &mut Vec<DiagnosisIssue>,
    ) -> Result<()> {
        let Some(manifest) = valid_snapshots.values().max_by_key(|m| m.sequence) else {
            return Ok(());
        };

        for entry in &manifest.row_groups {
            let meta_path = format!("{}/{}", self.table, entry.meta);
            if self.exists_resolving_not_found(&meta_path).await? {
                continue;
            }
            issues.push(DiagnosisIssue {
                kind: DiagnosisKind::MissingRowGroupMeta,
                path: entry.meta.clone(),
                detail: format!("row group meta file {} is missing", entry.meta),
            });
        }

        Ok(())
    }

    /// Regenerate missing row-group metadata sidecars from the Parquet footer.
    ///
    /// Any sidecar that cannot be regenerated is reported as an explicit
    /// [`ActionKind::Unrepairable`] action so the repair result is not silently
    /// healthy.
    async fn repair_row_group_metas(
        &self,
        manifest: &Manifest,
        actions: &mut Vec<RepairAction>,
    ) -> Result<()> {
        if manifest.row_groups.is_empty() {
            return Ok(());
        }

        let schema = self.load_schema(manifest.schema_id).await?;

        #[cfg(feature = "encryption")]
        let encrypted_columns = match &schema {
            Some(schema) => self.read_encrypted_columns(schema).await?,
            None => HashSet::new(),
        };
        #[cfg(not(feature = "encryption"))]
        let encrypted_columns: HashSet<String> = HashSet::new();

        let mut any_regenerated = false;
        for entry in &manifest.row_groups {
            let meta_path = format!("{}/{}", self.table, entry.meta);
            if self.exists_resolving_not_found(&meta_path).await? {
                continue;
            }

            // Without the schema we cannot interpret the Parquet footer to
            // regenerate a row-group meta sidecar. Report every missing sidecar
            // as unrepairable instead of failing silently.
            let Some(schema) = &schema else {
                actions.push(RepairAction {
                    kind: ActionKind::Unrepairable,
                    path: entry.meta.clone(),
                    detail: format!(
                        "row group meta {} is missing and schema {} is also missing",
                        entry.meta,
                        Schema::filename(manifest.schema_id)
                    ),
                });
                continue;
            };

            let data_path = format!("{}/{}", self.table, entry.data);
            let parquet_bytes = match self.storage.read(&data_path).await {
                Ok(b) => b,
                Err(e) if is_not_found(&e) => {
                    actions.push(RepairAction {
                        kind: ActionKind::Unrepairable,
                        path: entry.meta.clone(),
                        detail: format!(
                            "row group meta {} is missing and data file {} is also missing",
                            entry.meta, entry.data
                        ),
                    });
                    continue;
                }
                Err(e) => return Err(e),
            };

            let rg_id = std::path::Path::new(&entry.data)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&entry.data);

            let metadata = match ParquetRecordBatchReaderBuilder::try_new(Bytes::from(
                parquet_bytes.clone(),
            )) {
                Ok(builder) => builder.metadata().as_ref().clone(),
                Err(e) => {
                    actions.push(RepairAction {
                        kind: ActionKind::Unrepairable,
                        path: entry.meta.clone(),
                        detail: format!(
                            "row group meta {} is missing and the Parquet footer cannot be read: {}",
                            entry.meta, e
                        ),
                    });
                    continue;
                }
            };

            match crate::writer::compute_row_group_meta_from_footer(
                rg_id,
                manifest.schema_id,
                schema,
                &parquet_bytes,
                &metadata,
                &encrypted_columns,
            ) {
                Ok(meta) => {
                    self.storage
                        .write(&meta_path, &serde_json::to_vec(&meta)?)
                        .await?;
                    self.storage.sync(&meta_path).await?;
                    any_regenerated = true;
                    actions.push(RepairAction {
                        kind: ActionKind::Regenerated,
                        path: entry.meta.clone(),
                        detail: format!("regenerated row group meta {}", entry.meta),
                    });
                }
                Err(e) => {
                    actions.push(RepairAction {
                        kind: ActionKind::Unrepairable,
                        path: entry.meta.clone(),
                        detail: format!(
                            "row group meta {} is missing and could not be regenerated: {}",
                            entry.meta, e
                        ),
                    });
                }
            }
        }

        if any_regenerated {
            self.storage.sync(&format!("{}/", self.table)).await?;
        }

        Ok(())
    }

    async fn load_schema(&self, schema_id: u64) -> Result<Option<Schema>> {
        let path = format!("{}/{}", self.table, Schema::filename(schema_id));
        let data = match self.storage.read(&path).await {
            Ok(d) => d,
            Err(e) if is_not_found(&e) => return Ok(None),
            Err(e) => return Err(e),
        };
        Ok(Some(serde_json::from_slice(&data)?))
    }

    #[cfg(feature = "encryption")]
    async fn read_encrypted_columns(&self, schema: &Schema) -> Result<HashSet<String>> {
        let path = format!("{}/_encryption.json", self.table);
        let bytes = match self.storage.read(&path).await {
            Ok(b) => b,
            Err(e) if is_not_found(&e) => return Ok(HashSet::new()),
            Err(e) => return Err(e),
        };
        let marker: SchemaEncryptionMarker = serde_json::from_slice(&bytes).map_err(|e| {
            IcefallDBError::Encryption(format!(
                "failed to parse _encryption.json for table '{}': {}",
                self.table, e
            ))
        })?;
        marker.validate().map_err(|e| {
            IcefallDBError::Encryption(format!(
                "invalid _encryption.json for table '{}': {}",
                self.table, e
            ))
        })?;
        if marker.column_key_ids.is_empty() {
            Ok(schema.columns.iter().map(|c| c.name.clone()).collect())
        } else {
            Ok(marker.column_key_ids.keys().cloned().collect())
        }
    }

    /// Validate all manifest snapshots and return the valid ones keyed by sequence.
    ///
    /// Snapshots that cannot be found between listing and reading are silently
    /// skipped. Other I/O errors are propagated. Invalid snapshots (bad JSON,
    /// sequence mismatch, unsupported format version, or checksum failure) are
    /// returned in the second tuple element so callers can report them.
    async fn validate_snapshots(&self) -> Result<(HashMap<u64, Manifest>, Vec<InvalidSnapshot>)> {
        let manifests_dir = format!("{}/_manifests", self.table);
        let entries = match self.storage.list(&manifests_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok((HashMap::new(), Vec::new())),
            Err(e) => return Err(e),
        };

        let mut valid = HashMap::new();
        let mut invalid = Vec::new();
        for entry in entries {
            let filename = std::path::Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !filename.ends_with(".json") {
                continue;
            }
            let Some(seq_str) = filename.strip_suffix(".json") else {
                continue;
            };
            let Ok(seq) = seq_str.parse::<u64>() else {
                continue;
            };

            let data = match self.storage.read(&entry).await {
                Ok(d) => d,
                Err(e) if is_not_found(&e) => continue,
                Err(IcefallDBError::Io(e)) => return Err(IcefallDBError::Io(e)),
                Err(e) => return Err(e),
            };

            let rel_path = strip_table_prefix(&self.table, &entry);

            let manifest: Manifest = match serde_json::from_slice(&data) {
                Ok(m) => m,
                Err(e) => {
                    invalid.push(InvalidSnapshot {
                        path: rel_path,
                        reason: format!("invalid JSON: {}", e),
                    });
                    continue;
                }
            };
            if manifest.sequence != seq {
                invalid.push(InvalidSnapshot {
                    path: rel_path,
                    reason: format!(
                        "sequence mismatch: filename {} does not match manifest sequence {}",
                        seq, manifest.sequence
                    ),
                });
                continue;
            }
            if manifest.format_version != 1 {
                invalid.push(InvalidSnapshot {
                    path: rel_path,
                    reason: format!("unsupported format version: {}", manifest.format_version),
                });
                continue;
            }
            match manifest.verify_checksum() {
                Ok(true) => {}
                Ok(false) => {
                    invalid.push(InvalidSnapshot {
                        path: rel_path,
                        reason: "checksum mismatch".into(),
                    });
                    continue;
                }
                Err(e) => {
                    invalid.push(InvalidSnapshot {
                        path: rel_path,
                        reason: format!("checksum error: {}", e),
                    });
                    continue;
                }
            }
            valid.insert(seq, manifest);
        }

        Ok((valid, invalid))
    }

    /// Choose the highest valid sequence and return it along with the current pointer value.
    async fn choose_sequence(
        &self,
        valid_snapshots: &HashMap<u64, Manifest>,
    ) -> Result<(u64, Option<u64>)> {
        let pointer_path = format!("{}/_manifest.json", self.table);
        let current_pointer = match self.storage.read(&pointer_path).await {
            Ok(data) => serde_json::from_slice::<serde_json::Value>(&data)
                .ok()
                .and_then(|p| p.get("latest").and_then(|v| v.as_u64())),
            Err(e) if is_not_found(&e) => None,
            Err(e) => return Err(e),
        };

        let chosen = *valid_snapshots
            .keys()
            .max()
            .expect("valid_snapshots is non-empty");

        Ok((chosen, current_pointer))
    }

    /// Atomically write `_manifest.json` to point to `seq`.
    async fn write_pointer(&self, seq: u64) -> Result<()> {
        let pointer_path = format!("{}/_manifest.json", self.table);
        let tmp_path = format!("{}.tmp", pointer_path);
        let pointer = serde_json::json!({ "latest": seq });
        self.storage
            .write(&tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync(&tmp_path).await?;
        self.storage.rename(&tmp_path, &pointer_path).await?;
        self.storage.sync(&format!("{}/", self.table)).await?;
        Ok(())
    }

    /// Delete orphan files: unreferenced row groups, staged `.part` files,
    /// manifest snapshots newer than `chosen_seq`, and leftover `.json.tmp` files.
    async fn delete_orphans(
        &self,
        chosen_seq: u64,
        valid_snapshots: &HashMap<u64, Manifest>,
    ) -> Result<Vec<RepairAction>> {
        let mut actions = Vec::new();
        let referenced = referenced_files(valid_snapshots.values());

        // Unreferenced row group files in the table root.
        let mut row_groups_deleted = false;
        let root_entries = match self.storage.list(&self.table).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        for entry in root_entries {
            let filename = std::path::Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !filename.starts_with("rg_") {
                continue;
            }
            if !filename.ends_with(".parquet") && !filename.ends_with(".meta") {
                continue;
            }
            let rel = strip_table_prefix(&self.table, &entry);
            if !referenced.contains(&rel) {
                self.storage.delete(&entry).await?;
                row_groups_deleted = true;
                actions.push(RepairAction {
                    kind: ActionKind::OrphanRemoved,
                    path: rel.clone(),
                    detail: format!("deleted unreferenced row group file {}", rel),
                });
            }
        }
        if row_groups_deleted {
            self.storage.sync(&self.table).await?;
        }

        // Orphan staged `.part` files.
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
                    let rel = strip_table_prefix(&self.table, &entry);
                    actions.push(RepairAction {
                        kind: ActionKind::OrphanRemoved,
                        path: rel.clone(),
                        detail: format!("deleted orphan staged part file {}", rel),
                    });
                }
            }
            if parts_deleted {
                self.storage.sync(&dir).await?;
            }
        }

        // Manifest snapshots newer than the chosen sequence, plus any leftover
        // `.json.tmp` files.
        let manifests_dir = format!("{}/_manifests", self.table);
        let mut manifests_deleted = false;
        let entries = match self.storage.list(&manifests_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok(actions),
            Err(e) => return Err(e),
        };
        for entry in entries {
            let filename = std::path::Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if filename.ends_with(".json.tmp") {
                self.storage.delete(&entry).await?;
                manifests_deleted = true;
                let rel = strip_table_prefix(&self.table, &entry);
                actions.push(RepairAction {
                    kind: ActionKind::OrphanRemoved,
                    path: rel.clone(),
                    detail: format!("deleted orphan manifest tmp file {}", rel),
                });
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
            if seq > chosen_seq {
                self.storage.delete(&entry).await?;
                manifests_deleted = true;
                let rel = strip_table_prefix(&self.table, &entry);
                actions.push(RepairAction {
                    kind: ActionKind::OrphanRemoved,
                    path: rel.clone(),
                    detail: format!("deleted orphan manifest snapshot {}", rel),
                });
            }
        }
        if manifests_deleted {
            self.storage.sync(&manifests_dir).await?;
        }

        Ok(actions)
    }

    /// Detect orphan files without modifying state.
    async fn diagnose_orphans(
        &self,
        chosen_seq: u64,
        valid_snapshots: &HashMap<u64, Manifest>,
        issues: &mut Vec<DiagnosisIssue>,
    ) -> Result<()> {
        let referenced = referenced_files(valid_snapshots.values());

        // Unreferenced row group files in the table root.
        let root_entries = match self.storage.list(&self.table).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        for entry in root_entries {
            let filename = std::path::Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !filename.starts_with("rg_") {
                continue;
            }
            if !filename.ends_with(".parquet") && !filename.ends_with(".meta") {
                continue;
            }
            let rel = strip_table_prefix(&self.table, &entry);
            if !referenced.contains(&rel) {
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::OrphanRowGroup,
                    path: rel.clone(),
                    detail: format!("unreferenced row group file {}", rel),
                });
            }
        }

        // Orphan staged `.part` files.
        for staging_dir in ["_staging/incoming", "_staging/compact"] {
            let dir = format!("{}/{}", self.table, staging_dir);
            let entries = match self.storage.list(&dir).await {
                Ok(e) => e,
                Err(e) if is_not_found(&e) => continue,
                Err(e) => return Err(e),
            };
            for entry in entries {
                if entry.ends_with(".part") {
                    let rel = strip_table_prefix(&self.table, &entry);
                    issues.push(DiagnosisIssue {
                        kind: DiagnosisKind::OrphanStagedPart,
                        path: rel.clone(),
                        detail: format!("orphan staged part file {}", rel),
                    });
                }
            }
        }

        // Without a chosen sequence there are no "newer" manifests to report;
        // invalid snapshots are already reported by `validate_snapshots`.
        if valid_snapshots.is_empty() {
            return Ok(());
        }

        // Manifest snapshots newer than the chosen sequence are invalid.
        let manifests_dir = format!("{}/_manifests", self.table);
        let entries = match self.storage.list(&manifests_dir).await {
            Ok(e) => e,
            Err(e) if is_not_found(&e) => return Ok(()),
            Err(e) => return Err(e),
        };
        for entry in entries {
            let filename = std::path::Path::new(&entry)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            if !filename.ends_with(".json") {
                continue;
            }
            let Some(seq_str) = filename.strip_suffix(".json") else {
                continue;
            };
            let Ok(seq) = seq_str.parse::<u64>() else {
                continue;
            };
            if seq > chosen_seq {
                let rel = strip_table_prefix(&self.table, &entry);
                issues.push(DiagnosisIssue {
                    kind: DiagnosisKind::NewerInvalidManifest,
                    path: rel.clone(),
                    detail: format!(
                        "manifest snapshot {} is newer than chosen sequence {}",
                        rel, chosen_seq
                    ),
                });
            }
        }

        Ok(())
    }
}

fn strip_table_prefix(table: &str, path: &str) -> String {
    let prefix = format!("{}/", table);
    path.strip_prefix(&prefix).unwrap_or(path).to_string()
}

/// Collect the set of row-group files (data + meta) referenced by any of the
/// supplied manifests.
pub fn referenced_files<'a>(manifests: impl Iterator<Item = &'a Manifest>) -> HashSet<String> {
    let mut referenced = HashSet::new();
    for manifest in manifests {
        for entry in &manifest.row_groups {
            referenced.insert(entry.data.clone());
            referenced.insert(entry.meta.clone());
        }
    }
    referenced
}

/// Load all retained valid manifest snapshots and return them keyed by sequence.
///
/// Snapshots that cannot be found between listing and reading are silently
/// skipped. Other I/O errors are propagated. Invalid snapshots (bad JSON,
/// sequence mismatch, unsupported format version, or checksum failure) are
/// ignored so callers can treat the retained valid set as authoritative.
///
/// **Legacy exception:** manifests with an empty `checksum` field are treated as
/// pre-checksum snapshots and are included. This protects row-group files
/// referenced only by older legacy snapshots from being flagged as orphans.
pub async fn retained_valid_manifests(
    storage: &dyn Storage,
    table: &str,
) -> Result<HashMap<u64, Manifest>> {
    let manifests_dir = format!("{}/_manifests", table);
    let entries = match storage.list(&manifests_dir).await {
        Ok(e) => e,
        Err(e) if is_not_found(&e) => return Ok(HashMap::new()),
        Err(e) => return Err(e),
    };

    let mut valid = HashMap::new();
    for entry in entries {
        let filename = std::path::Path::new(&entry)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if !filename.ends_with(".json") {
            continue;
        }
        let Some(seq_str) = filename.strip_suffix(".json") else {
            continue;
        };
        let Ok(seq) = seq_str.parse::<u64>() else {
            continue;
        };

        let data = match storage.read(&entry).await {
            Ok(d) => d,
            Err(e) if is_not_found(&e) => continue,
            Err(e) => return Err(e),
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
        match manifest.verify_checksum() {
            Ok(true) => {}
            Ok(false) if manifest.checksum.is_empty() => {} // legacy: no checksum
            Ok(false) | Err(_) => continue,
        }
        valid.insert(seq, manifest);
    }

    Ok(valid)
}

// ── Chain verification ────────────────────────────────────────────────────

/// A single break detected in the manifest hash chain.
#[derive(Debug, Clone)]
pub struct ChainBreak {
    /// Sequence number of the manifest where the break was detected.
    pub sequence: u64,
    /// Human-readable description of what failed.
    pub reason: String,
}

/// Result of verifying the retained manifest hash chain.
#[derive(Debug, Clone)]
pub struct HistoryReport {
    /// `true` when no breaks were detected in the retained chain.
    pub intact: bool,
    /// Sequence number of the oldest retained manifest (0 when none exist).
    pub oldest: u64,
    /// Sequence number of the newest retained manifest (0 when none exist).
    pub latest: u64,
    /// All breaks detected during verification; empty when `intact` is `true`.
    pub breaks: Vec<ChainBreak>,
}

/// Verify the retained manifest hash chain for `table`.
///
/// Each retained manifest is verified against its own stored checksum.
/// Adjacent manifests (ascending sequence order) are checked for a valid chain
/// link: `m.parent_hash == Some(prev.checksum)` → linked.
///
/// **Anchor rule (corrected):** `m.parent_hash == None` while a predecessor is
/// present on disk is treated as an *anchor* — valid for genesis manifests,
/// GC-pruned predecessors, and legacy manifests written before hash-chaining
/// was introduced.  It is **never** reported as a break.  Only
/// `Some(other)` where `other != prev.checksum` is a chain break (tampering).
///
/// A self-checksum mismatch at sequence N is always a break.  When that
/// happens `prev` is reset to `None` so subsequent manifests are not compared
/// against a manifest whose checksum cannot be trusted.
pub async fn verify_history(storage: &dyn Storage, table: &str) -> Result<HistoryReport> {
    let manifests_dir = format!("{}/_manifests", table);
    let entries = match storage.list(&manifests_dir).await {
        Ok(e) => e,
        Err(e) if is_not_found(&e) => Vec::new(),
        Err(e) => return Err(e),
    };

    // Parse sequence numbers from filenames, skipping temporaries.
    let mut seqs: Vec<u64> = Vec::new();
    for entry in &entries {
        let filename = std::path::Path::new(entry)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if filename.ends_with(".json.tmp") || !filename.ends_with(".json") {
            continue;
        }
        let Some(seq_str) = filename.strip_suffix(".json") else {
            continue;
        };
        let Ok(seq) = seq_str.parse::<u64>() else {
            continue;
        };
        seqs.push(seq);
    }
    seqs.sort_unstable();

    if seqs.is_empty() {
        return Ok(HistoryReport {
            intact: true,
            oldest: 0,
            latest: 0,
            breaks: Vec::new(),
        });
    }

    let mut breaks: Vec<ChainBreak> = Vec::new();
    let mut prev: Option<Manifest> = None;

    for &seq in &seqs {
        let path = format!("{}/{}", table, Manifest::filename(seq));
        let bytes = match storage.read(&path).await {
            Ok(b) => b,
            Err(e) if is_not_found(&e) => {
                // Listed but vanished between list and read; treat as gap.
                prev = None;
                continue;
            }
            Err(e) => return Err(e),
        };

        let m: Manifest = match serde_json::from_slice(&bytes) {
            Ok(m) => m,
            Err(e) => {
                breaks.push(ChainBreak {
                    sequence: seq,
                    reason: format!("invalid JSON: {}", e),
                });
                prev = None;
                continue;
            }
        };

        // Self-checksum gate: a mismatch means the file was tampered or
        // corrupted.  We cannot trust m.checksum for the chain link, so
        // reset prev and skip the parent check for this entry.
        //
        // An empty `checksum` field marks a pre-checksum (legacy) manifest;
        // treat it like an anchor rather than a break.
        match m.verify_checksum() {
            Ok(true) => {}
            Ok(false) if m.checksum.is_empty() => {
                prev = None;
                continue;
            }
            Ok(false) => {
                breaks.push(ChainBreak {
                    sequence: seq,
                    reason: "self-checksum mismatch".into(),
                });
                prev = None;
                continue;
            }
            Err(e) => {
                breaks.push(ChainBreak {
                    sequence: seq,
                    reason: format!("checksum error: {}", e),
                });
                prev = None;
                continue;
            }
        }

        // Hash chain link check — only when we have a verified predecessor.
        if let Some(p) = &prev {
            match &m.parent_hash {
                // Correctly linked to predecessor.
                Some(ph) if ph == &p.checksum => {}
                // Present but wrong: tampering.
                Some(_) => {
                    breaks.push(ChainBreak {
                        sequence: seq,
                        reason: "parent_hash mismatch: does not match predecessor checksum".into(),
                    });
                }
                // None is an anchor (genesis / GC-pruned / legacy) — not a break.
                None => {}
            }
        }

        prev = Some(m);
    }

    Ok(HistoryReport {
        intact: breaks.is_empty(),
        oldest: seqs.first().copied().unwrap_or(0),
        latest: seqs.last().copied().unwrap_or(0),
        breaks,
    })
}
