use crate::doctor::verify_history;
use crate::metadata::schema::icefalldb_type_to_arrow;
use crate::metadata::{Manifest, RowGroupMeta, Schema};
use crate::storage::Storage;
use crate::Result;
use arrow::array::RecordBatch;
use arrow::datatypes::DataType;
use bytes::Bytes;
#[cfg(feature = "encryption")]
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashSet;

#[cfg(feature = "encryption")]
use {
    crate::encryption::provider::KeyProvider,
    crate::encryption::{
        build_decryption_properties, table_aad_prefix, EncryptionKeySet, KeyIdentifier,
        SchemaEncryptionMarker,
    },
    base64::Engine,
    std::sync::Arc,
};

/// Severity level of a [`CheckIssue`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// A single issue reported by the table checker.
#[derive(Debug, Clone)]
pub struct CheckIssue {
    pub severity: Severity,
    pub code: String,
    pub message: String,
}

/// Result of checking a single table.
#[derive(Debug, Clone)]
pub struct CheckResult {
    pub table: String,
    pub passed: bool,
    pub issues: Vec<CheckIssue>,
}

/// Options controlling how a table check behaves.
#[derive(Clone, Default)]
pub struct CheckOptions {
    /// Optional key provider used to decrypt Parquet Modular Encryption tables.
    /// When `None` (the default), encrypted tables are still validated at the
    /// metadata/checksum level, but data-page validation is skipped.
    #[cfg(feature = "encryption")]
    pub key_provider: Option<Arc<dyn KeyProvider>>,
}

impl std::fmt::Debug for CheckOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #[cfg(feature = "encryption")]
        {
            f.debug_struct("CheckOptions")
                .field(
                    "key_provider",
                    &self.key_provider.as_ref().map(|_| "**REDACTED**"),
                )
                .finish()
        }
        #[cfg(not(feature = "encryption"))]
        {
            f.debug_struct("CheckOptions").finish()
        }
    }
}

impl CheckOptions {
    /// Default options: no key provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// Provide a key provider so the checker can decrypt encrypted row groups.
    #[cfg(feature = "encryption")]
    pub fn with_key_provider(mut self, provider: Arc<dyn KeyProvider>) -> Self {
        self.key_provider = Some(provider);
        self
    }
}

/// How to report a declared column that is absent from the Parquet file.
enum MissingColumnHandling {
    /// Any missing declared column is an error.
    Error,
    /// A missing nullable declared column is a warning; a missing non-nullable
    /// declared column is still an error.
    NullableWarning,
}

/// Read-only validator for a IcefallDB table.
pub struct Checker<'a> {
    storage: &'a dyn Storage,
    table: String,
    #[cfg(feature = "encryption")]
    options: CheckOptions,
}

impl<'a> Checker<'a> {
    /// Create a new checker for `table` using `storage`.
    pub fn new(storage: &'a dyn Storage, table: &str) -> Self {
        Self::new_with_options(storage, table, CheckOptions::default())
    }

    /// Create a new checker with the supplied options.
    pub fn new_with_options(
        storage: &'a dyn Storage,
        table: &str,
        #[cfg_attr(not(feature = "encryption"), allow(unused_variables))] options: CheckOptions,
    ) -> Self {
        Self {
            storage,
            table: table.to_string(),
            #[cfg(feature = "encryption")]
            options,
        }
    }

    /// Run all consistency checks and return the result.
    pub async fn check(&self) -> Result<CheckResult> {
        let mut issues = Vec::new();

        // `_schema.json` is the authoritative marker that a table exists. Check
        // it first so a missing schema pointer short-circuits deeper validation.
        let schema_id_opt = self.check_schema_pointer(&mut issues).await?;
        let latest_opt = self.check_manifest_pointer(&mut issues).await?;

        // If the schema pointer is missing, the table is not initialized; do not
        // proceed to check manifests or row groups.
        let Some(schema_id) = schema_id_opt else {
            self.check_orphans(0, &HashSet::new(), &mut issues).await?;
            return Ok(self.result(issues));
        };

        let Some(latest) = latest_opt else {
            self.check_orphans(0, &HashSet::new(), &mut issues).await?;
            return Ok(self.result(issues));
        };

        // `latest: 0` marks an empty table. The schema pointer must still be
        // valid, but there is no manifest or row groups to validate yet.
        // Verify that the schema file referenced by `_schema.json` exists and is
        // readable before short-circuiting.
        if latest == 0 {
            let schema_path = self.path(&Schema::filename(schema_id));
            match self.storage.read(&schema_path).await {
                Ok(_) => {}
                Err(crate::IcefallDBError::NotFound(_)) => {
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "MISSING_SCHEMA".into(),
                        message: format!("schema file {} is missing", Schema::filename(schema_id)),
                    });
                    return Ok(self.result(issues));
                }
                Err(e) => return Err(e),
            }
            self.check_orphans(0, &HashSet::new(), &mut issues).await?;
            return Ok(self.result(issues));
        }

        // Load every retained valid manifest up-front so orphan detection can
        // protect files referenced by older snapshots (not only the latest).
        let valid_manifests =
            crate::doctor::retained_valid_manifests(self.storage, &self.table).await?;

        let manifest_opt = self.check_manifest(latest, &mut issues).await?;
        let Some(manifest) = manifest_opt else {
            let referenced_files = crate::doctor::referenced_files(valid_manifests.values());
            self.check_orphans(latest, &referenced_files, &mut issues)
                .await?;
            return Ok(self.result(issues));
        };

        if schema_id != manifest.schema_id {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "SCHEMA_POINTER_MISMATCH".into(),
                message: format!(
                    "_schema.json points to schema_id {}, manifest has schema_id {}",
                    schema_id, manifest.schema_id
                ),
            });
        }

        let schema_opt = self.check_schema(&manifest, &mut issues).await?;

        // Validate row groups from the latest manifest, but orphan detection must
        // consider the union of files referenced by *all* retained valid snapshots.
        self.check_row_groups(&manifest, schema_opt.as_ref(), &mut issues)
            .await?;
        let referenced_files = crate::doctor::referenced_files(valid_manifests.values());

        self.check_orphans(latest, &referenced_files, &mut issues)
            .await?;

        self.check_chain(&mut issues).await?;

        Ok(self.result(issues))
    }

    fn result(&self, issues: Vec<CheckIssue>) -> CheckResult {
        let passed = !issues.iter().any(|i| i.severity == Severity::Error);
        CheckResult {
            table: self.table.clone(),
            passed,
            issues,
        }
    }

    /// Verify the manifest hash chain and append any chain-break issues.
    ///
    /// An intact chain produces no issues (no noise on clean tables).
    /// Each break is reported as an Error with code `CHAIN_BREAK`.
    /// `--repair` never rewrites history.
    async fn check_chain(&self, issues: &mut Vec<CheckIssue>) -> Result<()> {
        let history = verify_history(self.storage, &self.table).await?;
        if !history.intact {
            for b in &history.breaks {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "CHAIN_BREAK".into(),
                    message: format!(
                        "chain break at seq {}: {} [chain {}..{}]",
                        b.sequence, b.reason, history.oldest, history.latest
                    ),
                });
            }
        }
        Ok(())
    }

    fn path(&self, rel: &str) -> String {
        if self.table.is_empty() {
            rel.to_string()
        } else {
            format!("{}/{}", self.table, rel)
        }
    }

    #[cfg(feature = "encryption")]
    async fn read_encryption_marker(
        &self,
        schema_id: u64,
    ) -> Result<Option<SchemaEncryptionMarker>> {
        let path = self.path("_encryption.json");
        let bytes = match self.storage.read(&path).await {
            Ok(b) => b,
            Err(crate::IcefallDBError::NotFound(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        let marker: SchemaEncryptionMarker = serde_json::from_slice(&bytes).map_err(|e| {
            crate::IcefallDBError::Encryption(format!(
                "failed to parse _encryption.json for table '{}': {}",
                self.table, e
            ))
        })?;
        marker.validate().map_err(|e| {
            crate::IcefallDBError::Encryption(format!(
                "invalid _encryption.json for table '{}': {}",
                self.table, e
            ))
        })?;
        // A mismatched AAD prefix is a serious integrity problem; report it as
        // an error rather than silently skipping encrypted validation.
        let expected_aad = table_aad_prefix(&self.table, schema_id);
        let stored_aad = marker
            .aad_prefix
            .as_ref()
            .map(|b64| {
                base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        if !stored_aad.is_empty() && stored_aad != expected_aad {
            return Err(crate::IcefallDBError::Encryption(format!(
                "_encryption.json AAD prefix does not match table '{}': expected {:?}, got {:?}",
                self.table, expected_aad, stored_aad
            )));
        }
        Ok(Some(marker))
    }

    #[cfg(feature = "encryption")]
    async fn resolve_decryption_properties(
        &self,
        marker: &SchemaEncryptionMarker,
        provider: &dyn KeyProvider,
        schema_id: u64,
    ) -> Result<Option<std::sync::Arc<parquet::encryption::decrypt::FileDecryptionProperties>>>
    {
        let aad = match &marker.aad_prefix {
            Some(b64) => base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| crate::IcefallDBError::Encryption(format!("invalid AAD: {e}")))?,
            None => table_aad_prefix(&self.table, schema_id),
        };

        let footer_id = KeyIdentifier::new(marker.footer_key_id.clone());
        let footer_key = match provider.get(&footer_id, &aad).await {
            Ok(k) => k,
            Err(crate::IcefallDBError::EncryptionKeyNotFound(_)) => return Ok(None),
            Err(e) => return Err(e),
        };

        let mut col_keys = std::collections::BTreeMap::new();
        for (col, kid) in &marker.column_key_ids {
            let key = match provider.get(&KeyIdentifier::new(kid.clone()), &aad).await {
                Ok(k) => k,
                Err(crate::IcefallDBError::EncryptionKeyNotFound(_)) => return Ok(None),
                Err(e) => return Err(e),
            };
            col_keys.insert(col.clone(), key.as_slice().to_vec());
        }

        let keyset = if col_keys.is_empty() {
            EncryptionKeySet::footer_only(footer_key.as_slice().to_vec(), aad)?
        } else {
            EncryptionKeySet::with_columns(footer_key.as_slice().to_vec(), col_keys, aad)?
        };

        Ok(Some(build_decryption_properties(&keyset)?))
    }

    async fn check_manifest_pointer(&self, issues: &mut Vec<CheckIssue>) -> Result<Option<u64>> {
        let path = self.path("_manifest.json");
        let data = match self.storage.read(&path).await {
            Ok(d) => d,
            Err(crate::IcefallDBError::NotFound(_)) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "MISSING_MANIFEST_POINTER".into(),
                    message: "_manifest.json is missing".into(),
                });
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        if let Ok(seq) = parse_pointer(&data) {
            // `latest: 0` is valid and indicates an empty table with no committed
            // manifests. Any non-negative integer is accepted here.
            return Ok(Some(seq));
        }
        issues.push(CheckIssue {
            severity: Severity::Error,
            code: "INVALID_MANIFEST_POINTER".into(),
            message: "_manifest.json must contain {\"latest\": u64}".into(),
        });
        Ok(None)
    }

    async fn check_schema_pointer(&self, issues: &mut Vec<CheckIssue>) -> Result<Option<u64>> {
        let path = self.path("_schema.json");
        let data = match self.storage.read(&path).await {
            Ok(d) => d,
            Err(crate::IcefallDBError::NotFound(_)) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "MISSING_SCHEMA_POINTER".into(),
                    message: "_schema.json is missing".into(),
                });
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        if let Ok(seq) = parse_pointer(&data) {
            if seq > 0 {
                return Ok(Some(seq));
            }
        }
        issues.push(CheckIssue {
            severity: Severity::Error,
            code: "INVALID_SCHEMA_POINTER".into(),
            message: "_schema.json must contain {\"latest\": u64} with latest > 0".into(),
        });
        Ok(None)
    }

    async fn check_manifest(
        &self,
        latest: u64,
        issues: &mut Vec<CheckIssue>,
    ) -> Result<Option<Manifest>> {
        let path = self.path(&Manifest::filename(latest));
        let data = match self.storage.read(&path).await {
            Ok(d) => d,
            Err(crate::IcefallDBError::NotFound(_)) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "MISSING_MANIFEST".into(),
                    message: format!(
                        "manifest snapshot {} is missing",
                        Manifest::filename(latest)
                    ),
                });
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        let manifest: Manifest = match serde_json::from_slice(&data) {
            Ok(m) => m,
            Err(_) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "CORRUPT_MANIFEST".into(),
                    message: format!(
                        "manifest snapshot {} is not valid JSON",
                        Manifest::filename(latest)
                    ),
                });
                return Ok(None);
            }
        };

        if !manifest.verify_checksum()? {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "CORRUPT_MANIFEST".into(),
                message: format!(
                    "manifest snapshot {} checksum does not match",
                    Manifest::filename(latest)
                ),
            });
            return Ok(None);
        }

        if manifest.sequence != latest {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "MANIFEST_SEQUENCE_MISMATCH".into(),
                message: format!(
                    "manifest sequence {} does not match pointer {}",
                    manifest.sequence, latest
                ),
            });
            return Ok(None);
        }

        if manifest.format_version != 1 {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "MANIFEST_FORMAT_VERSION".into(),
                message: format!(
                    "manifest format_version {} is not supported (expected 1)",
                    manifest.format_version
                ),
            });
            return Ok(None);
        }

        Ok(Some(manifest))
    }

    async fn check_schema(
        &self,
        manifest: &Manifest,
        issues: &mut Vec<CheckIssue>,
    ) -> Result<Option<Schema>> {
        let path = self.path(&Schema::filename(manifest.schema_id));
        let data = match self.storage.read(&path).await {
            Ok(d) => d,
            Err(crate::IcefallDBError::NotFound(_)) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "MISSING_SCHEMA".into(),
                    message: format!(
                        "schema file {} is missing",
                        Schema::filename(manifest.schema_id)
                    ),
                });
                return Ok(None);
            }
            Err(e) => return Err(e),
        };

        let schema: Schema = match serde_json::from_slice(&data) {
            Ok(s) => s,
            Err(_) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "CORRUPT_SCHEMA".into(),
                    message: format!(
                        "schema file {} is not valid JSON",
                        Schema::filename(manifest.schema_id)
                    ),
                });
                return Ok(None);
            }
        };

        if schema.schema_id != manifest.schema_id {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "SCHEMA_ID_MISMATCH".into(),
                message: format!(
                    "schema file {} has schema_id {}, manifest expects {}",
                    Schema::filename(manifest.schema_id),
                    schema.schema_id,
                    manifest.schema_id
                ),
            });
            return Ok(None);
        }

        if schema.columns.is_empty() {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "EMPTY_SCHEMA".into(),
                message: "schema columns must not be empty".into(),
            });
            return Ok(None);
        }

        Ok(Some(schema))
    }

    async fn check_row_groups(
        &self,
        manifest: &Manifest,
        current_schema: Option<&Schema>,
        issues: &mut Vec<CheckIssue>,
    ) -> Result<HashSet<String>> {
        let mut referenced = HashSet::new();

        #[cfg(feature = "encryption")]
        let encryption_marker = self.read_encryption_marker(manifest.schema_id).await?;
        #[cfg(not(feature = "encryption"))]
        let _encryption_marker: Option<()> = None;

        #[cfg(feature = "encryption")]
        let (decryption_props, encryption_keys_unavailable) = if let Some(marker) =
            encryption_marker.as_ref()
        {
            if let Some(provider) = self.options.key_provider.as_ref() {
                match self
                    .resolve_decryption_properties(marker, provider.as_ref(), manifest.schema_id)
                    .await
                {
                    Ok(Some(props)) => (Some(props), false),
                    Ok(None) | Err(_) => (None, true),
                }
            } else {
                (None, true)
            }
        } else {
            (None, false)
        };
        #[cfg(not(feature = "encryption"))]
        let (_decryption_props, encryption_keys_unavailable) = (None::<std::sync::Arc<()>>, false);

        #[cfg(feature = "encryption")]
        if encryption_keys_unavailable {
            issues.push(CheckIssue {
                severity: Severity::Info,
                code: "ENCRYPTION_KEY_UNAVAILABLE".into(),
                message: format!(
                    "encryption keys unavailable for table '{}'; \
                     data-page validation skipped for encrypted row groups",
                    self.table
                ),
            });
        }

        for entry in &manifest.row_groups {
            referenced.insert(entry.data.clone());
            referenced.insert(entry.meta.clone());

            let meta_path = self.path(&entry.meta);
            let meta_bytes = match self.storage.read(&meta_path).await {
                Ok(b) => b,
                Err(crate::IcefallDBError::NotFound(_)) => {
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "MISSING_ROW_GROUP_META".into(),
                        message: format!("row group meta file {} is missing", entry.meta),
                    });
                    continue;
                }
                Err(e) => return Err(e),
            };

            let meta: RowGroupMeta = match serde_json::from_slice(&meta_bytes) {
                Ok(m) => m,
                Err(_) => {
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "CORRUPT_ROW_GROUP_META".into(),
                        message: format!("row group meta file {} is not valid JSON", entry.meta),
                    });
                    continue;
                }
            };

            if !meta.verify_meta_checksum()? {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "CORRUPT_ROW_GROUP_META".into(),
                    message: format!(
                        "row group meta file {} has an invalid meta checksum",
                        entry.meta
                    ),
                });
                continue;
            }

            let rg_schema_path = self.path(&Schema::filename(meta.schema_id));
            let rg_schema: Schema = match self.storage.read(&rg_schema_path).await {
                Ok(b) => match serde_json::from_slice(&b) {
                    Ok(s) => s,
                    Err(_) => {
                        issues.push(CheckIssue {
                            severity: Severity::Error,
                            code: "MISSING_ROW_GROUP_SCHEMA".into(),
                            message: format!(
                                "row group schema file {} is missing or invalid",
                                Schema::filename(meta.schema_id)
                            ),
                        });
                        continue;
                    }
                },
                Err(crate::IcefallDBError::NotFound(_)) => {
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "MISSING_ROW_GROUP_SCHEMA".into(),
                        message: format!(
                            "row group schema file {} is missing",
                            Schema::filename(meta.schema_id)
                        ),
                    });
                    continue;
                }
                Err(e) => return Err(e),
            };

            if rg_schema.schema_id != meta.schema_id {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "MISSING_ROW_GROUP_SCHEMA".into(),
                    message: format!(
                        "schema file {} has schema_id {}, row group {} expects {}",
                        Schema::filename(meta.schema_id),
                        rg_schema.schema_id,
                        meta.row_group,
                        meta.schema_id
                    ),
                });
                continue;
            }

            let data_path = self.path(&entry.data);
            let parquet_bytes = match self.storage.read(&data_path).await {
                Ok(b) => b,
                Err(crate::IcefallDBError::NotFound(_)) => {
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "MISSING_ROW_GROUP_DATA".into(),
                        message: format!("row group parquet file {} is missing", entry.data),
                    });
                    continue;
                }
                Err(e) => return Err(e),
            };

            if !meta.verify_against_data(&parquet_bytes) {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "ROW_GROUP_CHECKSUM_MISMATCH".into(),
                    message: format!(
                        "row group parquet file {} does not match its checksum",
                        entry.data
                    ),
                });
                continue;
            }

            let parquet_bytes = Bytes::from(parquet_bytes);
            let builder_result = {
                #[cfg(feature = "encryption")]
                if let Some(props) = decryption_props.as_ref() {
                    ParquetRecordBatchReaderBuilder::try_new_with_options(
                        parquet_bytes.clone(),
                        ArrowReaderOptions::new().with_file_decryption_properties(props.clone()),
                    )
                } else {
                    ParquetRecordBatchReaderBuilder::try_new(parquet_bytes.clone())
                }
                #[cfg(not(feature = "encryption"))]
                ParquetRecordBatchReaderBuilder::try_new(parquet_bytes.clone())
            };

            match builder_result {
                Ok(builder) => {
                    let file_schema = builder.schema().clone();
                    let issues_before = issues.len();
                    self.check_schema_columns(
                        &rg_schema,
                        &file_schema,
                        &data_path,
                        issues,
                        MissingColumnHandling::Error,
                        false,
                    );

                    // Reconcile the row group against the current manifest schema.
                    if let Some(current) = current_schema {
                        if current.schema_id != rg_schema.schema_id {
                            self.check_schema_columns(
                                current,
                                &file_schema,
                                &data_path,
                                issues,
                                MissingColumnHandling::NullableWarning,
                                true,
                            );
                        }
                    }

                    let schema_errors = issues
                        .iter()
                        .skip(issues_before)
                        .any(|i| i.severity == Severity::Error);

                    if schema_errors {
                        // Do not read batches when the schema is already invalid.
                        if let Err(e) = builder.build() {
                            issues.push(CheckIssue {
                                severity: Severity::Error,
                                code: "PARQUET_OPEN_ERROR".into(),
                                message: format!(
                                    "row group parquet file {} cannot be opened: {}",
                                    entry.data, e
                                ),
                            });
                        }
                        continue;
                    }

                    // When keys are unavailable, validate the schema from the
                    // plaintext footer but skip reading/decrypting data pages.
                    if encryption_keys_unavailable {
                        continue;
                    }

                    match builder.build() {
                        Ok(reader) => {
                            let batches: Vec<RecordBatch> =
                                match reader.collect::<std::result::Result<Vec<_>, _>>() {
                                    Ok(b) => b,
                                    Err(e) => {
                                        issues.push(CheckIssue {
                                            severity: Severity::Error,
                                            code: "PARQUET_OPEN_ERROR".into(),
                                            message: format!(
                                                "row group parquet file {} cannot be read: {}",
                                                entry.data, e
                                            ),
                                        });
                                        continue;
                                    }
                                };
                            self.validate_row_group_stats(&meta, &batches, &data_path, issues);
                        }
                        Err(e) => {
                            issues.push(CheckIssue {
                                severity: Severity::Error,
                                code: "PARQUET_OPEN_ERROR".into(),
                                message: format!(
                                    "row group parquet file {} cannot be opened: {}",
                                    entry.data, e
                                ),
                            });
                        }
                    }
                }
                Err(e) => {
                    if encryption_keys_unavailable {
                        // Encrypted footer (or other key-related failure): the
                        // metadata/checksum checks above already validated what
                        // we can without keys. Do not report a spurious error.
                        continue;
                    }
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "PARQUET_OPEN_ERROR".into(),
                        message: format!(
                            "row group parquet file {} cannot be opened: {}",
                            entry.data, e
                        ),
                    });
                }
            }
        }

        Ok(referenced)
    }

    /// Returns true if `parquet_type` is the same as, or a safely-promotable
    /// narrower type of, `expected_type`.
    fn is_compatible_type(parquet_type: &DataType, expected_type: &DataType) -> bool {
        if parquet_type == expected_type {
            return true;
        }
        matches!(
            (parquet_type, expected_type),
            (
                DataType::Int8,
                DataType::Int16 | DataType::Int32 | DataType::Int64
            ) | (DataType::Int16, DataType::Int32 | DataType::Int64)
                | (DataType::Int32, DataType::Int64)
                | (
                    DataType::UInt8,
                    DataType::UInt16 | DataType::UInt32 | DataType::UInt64
                )
                | (DataType::UInt16, DataType::UInt32 | DataType::UInt64)
                | (DataType::UInt32, DataType::UInt64)
                | (DataType::Float32, DataType::Float64)
                | (DataType::Utf8, DataType::LargeUtf8)
        )
    }

    fn check_schema_columns(
        &self,
        schema: &Schema,
        file_schema: &arrow::datatypes::Schema,
        data_path: &str,
        issues: &mut Vec<CheckIssue>,
        missing_handling: MissingColumnHandling,
        allow_type_promotion: bool,
    ) {
        let declared: HashSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        let dropped: HashSet<&str> = schema.dropped_columns.iter().map(|c| c.as_str()).collect();

        for col in &schema.columns {
            match file_schema.field_with_name(&col.name) {
                Ok(field) => {
                    match icefalldb_type_to_arrow(&col.r#type) {
                        Some(expected) => {
                            let compatible = if allow_type_promotion {
                                Self::is_compatible_type(field.data_type(), &expected)
                            } else {
                                field.data_type() == &expected
                            };
                            if !compatible {
                                issues.push(CheckIssue {
                                    severity: Severity::Error,
                                    code: "SCHEMA_MISMATCH".into(),
                                    message: format!(
                                        "SCHEMA_MISMATCH: column '{}' in {} expected {}, got {}",
                                        col.name,
                                        data_path,
                                        col.r#type,
                                        field.data_type()
                                    ),
                                });
                            }
                        }
                        None => {
                            issues.push(CheckIssue {
                                severity: Severity::Error,
                                code: "UNSUPPORTED_SCHEMA_TYPE".into(),
                                message: format!(
                                    "UNSUPPORTED_SCHEMA_TYPE: column '{}' in {} declares unsupported type '{}'",
                                    col.name, data_path, col.r#type
                                ),
                            });
                        }
                    }

                    if col.nullable && !field.is_nullable() {
                        issues.push(CheckIssue {
                            severity: Severity::Error,
                            code: "SCHEMA_MISMATCH".into(),
                            message: format!(
                                "SCHEMA_MISMATCH: column '{}' in {} expected nullable=true",
                                col.name, data_path
                            ),
                        });
                    }
                    if !col.nullable && field.is_nullable() {
                        issues.push(CheckIssue {
                            severity: Severity::Error,
                            code: "SCHEMA_MISMATCH".into(),
                            message: format!(
                                "SCHEMA_MISMATCH: column '{}' in {} expected nullable=false",
                                col.name, data_path
                            ),
                        });
                    }
                }
                Err(_) => {
                    if matches!(missing_handling, MissingColumnHandling::NullableWarning)
                        && col.nullable
                    {
                        issues.push(CheckIssue {
                            severity: Severity::Warning,
                            code: "MISSING_NULLABLE_COLUMN".into(),
                            message: format!(
                                "MISSING_NULLABLE_COLUMN: nullable column '{}' in {} is missing from older row group",
                                col.name, data_path
                            ),
                        });
                    } else {
                        issues.push(CheckIssue {
                            severity: Severity::Error,
                            code: "SCHEMA_MISMATCH".into(),
                            message: format!(
                                "SCHEMA_MISMATCH: missing column '{}' in {}, expected {}",
                                col.name, data_path, col.r#type
                            ),
                        });
                    }
                }
            }
        }

        for field in file_schema.fields() {
            let name = field.name().as_str();
            if !declared.contains(name) && !dropped.contains(name) {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "SCHEMA_MISMATCH".into(),
                    message: format!(
                        "SCHEMA_MISMATCH: {} contains extra column '{}' ({}) not in schema",
                        data_path,
                        field.name(),
                        field.data_type()
                    ),
                });
            }
        }
    }

    fn validate_row_group_stats(
        &self,
        meta: &RowGroupMeta,
        batches: &[RecordBatch],
        data_path: &str,
        issues: &mut Vec<CheckIssue>,
    ) {
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if total_rows != meta.rows {
            issues.push(CheckIssue {
                severity: Severity::Error,
                code: "ROW_COUNT_MISMATCH".into(),
                message: format!(
                    "row group {} has {} rows, meta declares {}",
                    meta.row_group, total_rows, meta.rows
                ),
            });
        }

        if batches.is_empty() {
            return;
        }

        let combined = match arrow::compute::concat_batches(&batches[0].schema(), batches) {
            Ok(b) => b,
            Err(e) => {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "PARQUET_OPEN_ERROR".into(),
                    message: format!(
                        "row group parquet file {} cannot be concatenated: {}",
                        data_path, e
                    ),
                });
                return;
            }
        };

        for (col_name, stats) in &meta.columns {
            let Some(array) = combined.column_by_name(col_name) else {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "SCHEMA_MISMATCH".into(),
                    message: format!(
                        "SCHEMA_MISMATCH: column '{}' in {} is declared in stats but missing from data",
                        col_name, data_path
                    ),
                });
                continue;
            };

            let nulls = array.null_count();
            if nulls != stats.nulls {
                issues.push(CheckIssue {
                    severity: Severity::Error,
                    code: "NULL_COUNT_MISMATCH".into(),
                    message: format!(
                        "column '{}' in {} has {} nulls, meta declares {}",
                        col_name, data_path, nulls, stats.nulls
                    ),
                });
            }

            match crate::writer::compute_min_max(array, data_path) {
                Ok((actual_min, actual_max)) => {
                    if actual_min != stats.min {
                        issues.push(CheckIssue {
                            severity: Severity::Error,
                            code: "MIN_MAX_MISMATCH".into(),
                            message: format!(
                                "column '{}' in {} has min {:?}, meta declares {:?}",
                                col_name, data_path, actual_min, stats.min
                            ),
                        });
                    }
                    if actual_max != stats.max {
                        issues.push(CheckIssue {
                            severity: Severity::Error,
                            code: "MIN_MAX_MISMATCH".into(),
                            message: format!(
                                "column '{}' in {} has max {:?}, meta declares {:?}",
                                col_name, data_path, actual_max, stats.max
                            ),
                        });
                    }
                }
                Err(e) => {
                    issues.push(CheckIssue {
                        severity: Severity::Error,
                        code: "MIN_MAX_MISMATCH".into(),
                        message: format!(
                            "cannot compute min/max for column '{}' in {}: {}",
                            col_name, data_path, e
                        ),
                    });
                }
            }
        }
    }

    async fn check_orphans(
        &self,
        latest: u64,
        referenced_files: &HashSet<String>,
        issues: &mut Vec<CheckIssue>,
    ) -> Result<()> {
        // Stale intents.
        let intents_dir = self.path("_staging/intents");
        if let Ok(entries) = self.storage.list(&intents_dir).await {
            for entry in entries {
                issues.push(CheckIssue {
                    severity: Severity::Warning,
                    code: "STALE_INTENT".into(),
                    message: format!("stale intent file: {}", entry),
                });
            }
        }

        // Orphan .part files.
        let incoming_dir = self.path("_staging/incoming");
        if let Ok(entries) = self.storage.list(&incoming_dir).await {
            for entry in entries {
                if entry.ends_with(".part") {
                    issues.push(CheckIssue {
                        severity: Severity::Warning,
                        code: "ORPHAN_PART".into(),
                        message: format!("orphan staged part file: {}", entry),
                    });
                }
            }
        }

        // Orphan manifest snapshots.
        let manifests_dir = self.path("_manifests");
        if let Ok(entries) = self.storage.list(&manifests_dir).await {
            for entry in entries {
                let filename = std::path::Path::new(&entry)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                if let Some(seq_str) = filename.strip_suffix(".json") {
                    if let Ok(seq) = seq_str.parse::<u64>() {
                        if seq > latest {
                            issues.push(CheckIssue {
                                severity: Severity::Warning,
                                code: "ORPHAN_MANIFEST".into(),
                                message: format!(
                                    "manifest snapshot {} is newer than latest {}",
                                    entry, latest
                                ),
                            });
                        }
                    }
                }
            }
        }

        // Unreferenced row group files in the table root.
        let root_entries = if self.table.is_empty() {
            self.storage.list("").await.unwrap_or_default()
        } else {
            self.storage.list(&self.table).await.unwrap_or_default()
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
            let rel = if self.table.is_empty() {
                filename.to_string()
            } else {
                entry
                    .strip_prefix(&format!("{}/", self.table))
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| entry.clone())
            };
            if !referenced_files.contains(&rel) {
                issues.push(CheckIssue {
                    severity: Severity::Warning,
                    code: "UNREFERENCED_ROW_GROUP".into(),
                    message: format!("unreferenced row group file: {}", rel),
                });
            }
        }

        Ok(())
    }
}

fn parse_pointer(data: &[u8]) -> std::result::Result<u64, ()> {
    let value: serde_json::Value = serde_json::from_slice(data).map_err(|_| ())?;
    value.get("latest").and_then(|v| v.as_u64()).ok_or(())
}
