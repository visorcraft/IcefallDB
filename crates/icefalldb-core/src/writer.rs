use crate::agg_cache::{compute_agg_state_with_key, serialize_agg_state};
use crate::commit_delta::{CommitDelta, CommitKind};
use crate::index::IndexMaintainer;
use crate::metadata::schema::icefalldb_type_to_arrow;
use crate::metadata::{
    ColumnChunkOffset, ColumnStats, FragmentSummary, Manifest, RowGroupEntry, RowGroupMeta, Schema,
    SnapshotCheckpoint,
};
use crate::rowid::{allocate_range, RowIdSegment};
use crate::storage::Storage;
use crate::{is_not_found, IcefallDBError, Result};
use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::datatypes::TimeUnit;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, SchemaRef};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use parquet::file::statistics::Statistics;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncReadExt;

#[cfg(feature = "encryption")]
use crate::encryption::{
    build_encryption_properties, EncryptionWriteConfig, SchemaEncryptionMarker,
};

fn other<E: std::error::Error + Send + Sync + 'static>(err: E) -> IcefallDBError {
    IcefallDBError::Other(Box::new(err))
}

/// Reserved table names and files that conflict with IcefallDB internals.
const RESERVED_TABLE_NAMES: &[&str] = &[
    "_write",
    "_write.lock",
    "_manifest",
    "_manifest.json",
    "_schema",
    "_schema.json",
    "_manifests",
    "_schemas",
    "_staging",
    "views",
];

pub use crate::schema_util::{DEFAULT_ROW_GROUP_TARGET_BYTES, DEFAULT_ROW_GROUP_TARGET_ROWS};

/// Validate the user-supplied table path.
pub fn validate_table(table: &str) -> Result<()> {
    if table.is_empty()
        || table.starts_with('/')
        || table
            .split('/')
            .any(|component| component == ".." || component == ".")
        || table.split('/').any(|component| component.is_empty())
    {
        return Err(IcefallDBError::InvalidPath(table.to_string()));
    }
    if RESERVED_TABLE_NAMES.contains(&table)
        || table.starts_with("_manifests/")
        || table.starts_with("_schemas/")
        || table.starts_with("_staging/")
    {
        return Err(IcefallDBError::InvalidPath(table.to_string()));
    }
    Ok(())
}

/// A planned row group held in memory while the commit intent is being
/// materialized.
struct PlannedRowGroup {
    /// Shared identifier used for both staged and final filenames.
    rg_id: String,
    /// Final data filename relative to the table root.
    data_filename: String,
    /// Final metadata filename relative to the table root.
    meta_filename: String,
    /// The rows that belong to this row group.
    batch: RecordBatch,
}

/// Tracks the state needed to roll back a commit that failed before or after
/// the manifest pointer was updated.
///
/// `staged_files` contains the final table-root filenames (e.g.
/// `rg_xxx.parquet` and `rg_xxx.meta`) for row groups that have already been
/// renamed from `_staging/incoming/*.part`. These are used to clean up a
/// failed in-progress transaction; they are not the same as the `.part` files
/// recorded in the commit intent.
#[derive(Default)]
struct RollbackInfo {
    intent_path: String,
    staged_files: Vec<String>,
    sequence: u64,
    pointer_updated: bool,
}

/// Configuration options for a [`Writer`].
#[derive(Debug, Clone, Copy)]
pub struct WriterOptions {
    /// Maximum time to wait when acquiring the exclusive writer lock.
    pub lock_timeout: Duration,
    /// If `true`, the caller is responsible for holding the table's exclusive
    /// writer lock for the whole operation. `Writer` will not attempt to
    /// acquire or release the lock in `new_with_options`, `commit`, or
    /// `replace`. This is used by the CLI to hold the view-table lock across
    /// DuckDB execution and the final `replace()`.
    pub assume_lock_held: bool,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            lock_timeout: Duration::from_secs(30),
            assume_lock_held: false,
        }
    }
}

/// Owned, non-`Copy` counterpart to [`WriterOptions`] that supports options
/// which cannot live in a `Copy` struct (notably encryption key material).
///
/// Construct via [`WriterOptionsFull::new`] (defaults) or
/// [`WriterOptionsFull::from_simple`] (when the caller already has a
/// [`WriterOptions`]). Encryption is enabled by calling
/// [`WriterOptionsFull::with_encryption`] and requires the `encryption`
/// feature on `icefalldb-core`.
#[derive(Debug, Clone)]
pub struct WriterOptionsFull {
    /// Maximum time to wait when acquiring the exclusive writer lock.
    pub lock_timeout: Duration,
    /// If `true`, the caller is responsible for holding the table's exclusive
    /// writer lock for the whole operation.
    pub assume_lock_held: bool,
    /// Optional Parquet Modular Encryption configuration. `None` (the default)
    /// means writes are plaintext Parquet — the historical IcefallDB behavior,
    /// with zero crypto overhead.
    #[cfg(feature = "encryption")]
    pub encryption: Option<EncryptionWriteConfig>,
}

impl Default for WriterOptionsFull {
    fn default() -> Self {
        // Match the legacy `WriterOptions::default()` so callers upgrading from
        // the Copy-able options struct see the same behavior (30 s lock
        // timeout, no encryption).
        Self {
            lock_timeout: Duration::from_secs(30),
            assume_lock_held: false,
            #[cfg(feature = "encryption")]
            encryption: None,
        }
    }
}

impl WriterOptionsFull {
    /// Defaults: 30 s lock timeout, caller does not hold the lock, no encryption.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a [`WriterOptionsFull`] from a legacy [`WriterOptions`], with no
    /// encryption configured.
    pub fn from_simple(opts: WriterOptions) -> Self {
        Self {
            lock_timeout: opts.lock_timeout,
            assume_lock_held: opts.assume_lock_held,
            #[cfg(feature = "encryption")]
            encryption: None,
        }
    }

    pub fn with_lock_timeout(mut self, d: Duration) -> Self {
        self.lock_timeout = d;
        self
    }

    pub fn with_assume_lock_held(mut self, v: bool) -> Self {
        self.assume_lock_held = v;
        self
    }

    /// Enable Parquet Modular Encryption for this writer. Requires the
    /// `encryption` feature on `icefalldb-core`. When the feature is off, the
    /// method is absent and callers get a compile-time hint to enable it.
    #[cfg(feature = "encryption")]
    pub fn with_encryption(mut self, cfg: EncryptionWriteConfig) -> Self {
        self.encryption = Some(cfg);
        self
    }
}

impl From<WriterOptions> for WriterOptionsFull {
    fn from(opts: WriterOptions) -> Self {
        Self::from_simple(opts)
    }
}

/// Whether [`Writer::new_with_options`] should create a table if it does not
/// exist, or require that it does not exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CreateMode {
    /// Open an existing table, or create it if it does not exist.
    OpenOrCreate,
    /// Create a new table; fail if it already exists.
    CreateNew,
}

/// Validate a user-supplied schema.
fn validate_schema_for_table(schema: &Schema, table: &str) -> Result<()> {
    if schema.columns.is_empty() {
        return Err(IcefallDBError::InvalidSchema {
            reason: "schema must have at least one column".into(),
            path: table.into(),
        });
    }
    if schema.row_group_target_rows == 0 {
        return Err(IcefallDBError::InvalidSchema {
            reason: "row_group_target_rows must be > 0".into(),
            path: table.into(),
        });
    }
    if schema.row_group_target_bytes == 0 {
        return Err(IcefallDBError::InvalidSchema {
            reason: "row_group_target_bytes must be > 0".into(),
            path: table.into(),
        });
    }

    let mut seen = HashSet::new();
    for col in &schema.columns {
        if col.name.is_empty() {
            return Err(IcefallDBError::InvalidSchema {
                reason: "column name must not be empty".into(),
                path: table.into(),
            });
        }
        if !seen.insert(&col.name) {
            return Err(IcefallDBError::InvalidSchema {
                reason: format!("duplicate column name '{}'", col.name),
                path: table.into(),
            });
        }
    }

    if let Some(partition_by) = &schema.partition_by {
        let column_names: HashSet<&str> = schema.columns.iter().map(|c| c.name.as_str()).collect();
        for col in partition_by {
            if !column_names.contains(col.as_str()) {
                return Err(IcefallDBError::InvalidSchema {
                    reason: format!("partition column '{}' not found in schema", col),
                    path: table.into(),
                });
            }
        }
    }

    Ok(())
}

/// Load the schema file referenced by `_schema.json`.
///
/// Returns `SchemaNotFound` when the schema file is missing so callers do not
/// confuse a missing schema with a missing manifest.
async fn load_existing_schema_file(
    storage: &dyn Storage,
    table: &str,
    schema_id: u64,
) -> Result<Schema> {
    let schema_path = format!("{}/{}", table, Schema::filename(schema_id));
    if !storage.exists(&schema_path).await? {
        return Err(IcefallDBError::SchemaNotFound { path: schema_path });
    }

    let schema: Schema = serde_json::from_slice(&storage.read(&schema_path).await?)?;
    if schema.schema_id != schema_id {
        return Err(IcefallDBError::SchemaMismatch {
            column: "schema_id".into(),
            expected: schema_id.to_string(),
            path: schema_path,
        });
    }
    Ok(schema)
}

/// Initialize a new table while holding the exclusive writer lock.
///
/// Writes `_schemas/000001.json`, `_schema.json`, and `_manifest.json`
/// (containing `{"latest": 0}`) atomically. The supplied schema is normalized:
/// `schema_id` is set to `1` and field IDs are assigned.
async fn initialize_table_locked(
    storage: &dyn Storage,
    table: &str,
    schema: &mut Schema,
) -> Result<()> {
    // New tables always start at schema_id 1 with no dropped columns and no
    // previously assigned field IDs, so IDs are assigned monotonically from 1.
    schema.schema_id = 1;
    schema.dropped_columns.clear();
    schema.max_field_id = 0;
    schema.assign_field_ids(None);

    let schema_path = format!("{}/{}", table, Schema::filename(schema.schema_id));
    let pointer_path = format!("{}/_schema.json", table);
    let manifest_pointer_path = format!("{}/_manifest.json", table);

    // Write the schema file atomically.
    let schema_data = serde_json::to_vec(&schema)?;
    let schema_tmp_path = format!("{}.tmp", schema_path);
    storage.write(&schema_tmp_path, &schema_data).await?;
    storage.sync_data(&schema_tmp_path).await?;
    storage.rename(&schema_tmp_path, &schema_path).await?;
    storage.sync(&format!("{}/_schemas", table)).await?;

    // Write the schema pointer atomically.
    let pointer = serde_json::json!({"latest": schema.schema_id});
    let pointer_data = serde_json::to_vec(&pointer)?;
    let pointer_tmp_path = format!("{}.tmp", pointer_path);
    storage.write(&pointer_tmp_path, &pointer_data).await?;
    storage.sync_data(&pointer_tmp_path).await?;
    storage.rename(&pointer_tmp_path, &pointer_path).await?;

    // Create the manifest directory so external tools and tests can write
    // manifests directly without waiting for the first insert.
    ensure_dir(storage, &format!("{}/_manifests", table)).await?;

    // Write the manifest pointer atomically, indicating an empty table.
    let manifest_pointer = serde_json::json!({"latest": 0});
    let manifest_pointer_data = serde_json::to_vec(&manifest_pointer)?;
    let manifest_pointer_tmp_path = format!("{}.tmp", manifest_pointer_path);
    storage
        .write(&manifest_pointer_tmp_path, &manifest_pointer_data)
        .await?;
    storage.sync_data(&manifest_pointer_tmp_path).await?;
    storage
        .rename(&manifest_pointer_tmp_path, &manifest_pointer_path)
        .await?;
    storage.sync(&format!("{}/", table)).await?;

    Ok(())
}

/// Write the `<table>/_encryption.json` sidecar marker for an encrypted table.
///
/// Contains the algorithm identifier, the footer key identifier (resolved by
/// the reader via a [`crate::encryption::provider::KeyProvider`]), the column
/// key identifiers, the plaintext-footer flag, and the (non-secret) AAD
/// prefix. Contains **no key material**.
#[cfg(feature = "encryption")]
async fn write_encryption_marker_locked(
    storage: &dyn Storage,
    table: &str,
    schema_id: u64,
    cfg: &EncryptionWriteConfig,
) -> Result<()> {
    // The footer key id is by convention `<table>-v<schema_id>` so that a
    // reader can resolve it deterministically without consulting the schema.
    // Production deployments should override this with an explicit key id via
    // a future `EncryptionWriteConfig::footer_key_id` field once the KMS
    // factory lands.
    let footer_kid = format!("{table}-v{schema_id}");
    let marker = SchemaEncryptionMarker::for_write_config(footer_kid, cfg);
    let json = serde_json::to_vec_pretty(&marker)?;
    let path = format!("{table}/_encryption.json");
    let tmp_path = format!("{path}.tmp");
    storage.write(&tmp_path, &json).await?;
    storage.sync_data(&tmp_path).await?;
    storage.rename(&tmp_path, &path).await?;
    storage.sync(&format!("{table}/")).await?;
    Ok(())
}

/// The physical location of a matched row — used by [`Writer::commit_update`]
/// to identify which rows are being replaced.
///
/// `fragment_id` and `offset` describe where the row currently lives (as
/// returned by the query engine that resolved the WHERE clause), and `row_id`
/// is the stable identity that must be preserved in the patch fragment so that
/// the row index can map it to its new location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchLoc {
    /// Fragment (row-group) currently holding the row.
    pub fragment_id: u64,
    /// Physical row offset within that fragment.
    pub offset: u32,
    /// Stable row identifier — carried into the patch fragment.
    pub row_id: u64,
}

/// Result of [`Writer::insert_parquet`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertParquetOutcome {
    /// The file was compatible and committed via the zero-copy fast path.
    FastPath { rows: usize },
    /// The file was not eligible for the fast path; the caller should fall back
    /// to decoding and re-encoding (e.g. [`Writer::insert_batch`]).
    Incompatible,
}

/// Column statistics and optional partition values derived from a Parquet footer.
type FooterStats = (
    HashMap<String, ColumnStats>,
    Option<HashMap<String, serde_json::Value>>,
);

/// An in-process writer for buffered appends to a IcefallDB table.
///
/// Batches supplied to [`Writer::insert_batch`] are held in memory until
/// [`Writer::commit`] is called. Callers are responsible for sizing batches and
/// the commit interval so that the buffered data fits in memory; the writer does
/// not apply backpressure or spill to disk.
pub struct Writer {
    storage: Arc<dyn Storage>,
    table: String,
    schema: Schema,
    arrow_schema: SchemaRef,
    buffer: Vec<RecordBatch>,
    buffered_rows: usize,
    buffered_bytes: usize,
    lock_timeout: Duration,
    assume_lock_held: bool,
    /// When `true`, `commit_deletes` defers the manifest materialization +
    /// pointer-swap ceremony: it appends a compact [`crate::mutation_wal`] record
    /// (durable with the deletion-vector write plus one `fsync`) and checkpoints
    /// only once the log reaches `wal_checkpoint_threshold`. Opt-in; default
    /// `false` (every DELETE swaps the manifest immediately, as before).
    wal_mode: bool,
    /// Number of pending WAL records that triggers a checkpoint inside
    /// `commit_deletes`. Ignored unless `wal_mode`.
    wal_checkpoint_threshold: usize,
    /// Per-table encryption configuration. `None` means writes are plaintext
    /// Parquet (the default). When `Some`, every row-group file produced by
    /// [`Writer::commit`] is encrypted via Parquet Modular Encryption and the
    /// `_schema.json` file carries a [`SchemaEncryptionMarker`] so the reader
    /// knows the table is encrypted.
    #[cfg(feature = "encryption")]
    encryption: Option<EncryptionWriteConfig>,
}

impl Writer {
    pub async fn new(storage: Arc<dyn Storage>, table: &str, schema: Schema) -> Result<Self> {
        Self::new_with_options(storage, table, schema, WriterOptions::default()).await
    }

    /// Create a new table, failing if it already exists.
    ///
    /// This is the programmatic equivalent of `icefalldb create`. Schema validation,
    /// field-ID assignment, and atomic initialization are all performed while
    /// holding the exclusive writer lock.
    pub async fn create(storage: Arc<dyn Storage>, table: &str, schema: Schema) -> Result<Self> {
        Self::create_with_options(storage, table, schema, WriterOptions::default()).await
    }

    /// Create a new table with custom writer options, failing if it already exists.
    pub async fn create_with_options(
        storage: Arc<dyn Storage>,
        table: &str,
        schema: Schema,
        options: WriterOptions,
    ) -> Result<Self> {
        Self::create_with_full(
            storage,
            table,
            schema,
            WriterOptionsFull::from_simple(options),
        )
        .await
    }

    pub async fn new_with_options(
        storage: Arc<dyn Storage>,
        table: &str,
        schema: Schema,
        options: WriterOptions,
    ) -> Result<Self> {
        Self::new_with_full(
            storage,
            table,
            schema,
            WriterOptionsFull::from_simple(options),
        )
        .await
    }

    /// Create a new table with the full options struct (supports encryption),
    /// failing if it already exists.
    pub async fn create_with_full(
        storage: Arc<dyn Storage>,
        table: &str,
        schema: Schema,
        options: WriterOptionsFull,
    ) -> Result<Self> {
        Self::open_or_create_table_full(storage, table, schema, options, CreateMode::CreateNew)
            .await
    }

    /// Open or create a table with the full options struct (supports encryption).
    pub async fn new_with_full(
        storage: Arc<dyn Storage>,
        table: &str,
        schema: Schema,
        options: WriterOptionsFull,
    ) -> Result<Self> {
        Self::open_or_create_table_full(storage, table, schema, options, CreateMode::OpenOrCreate)
            .await
    }

    async fn open_or_create_table_full(
        storage: Arc<dyn Storage>,
        table: &str,
        mut schema: Schema,
        options: WriterOptionsFull,
        mode: CreateMode,
    ) -> Result<Self> {
        validate_table(table)?;
        validate_schema_for_table(&schema, table)?;

        if mode == CreateMode::CreateNew && schema.schema_id != 1 {
            return Err(IcefallDBError::InvalidSchema {
                reason: "new tables must start at schema_id 1".into(),
                path: table.into(),
            });
        }

        // Encrypted partition columns are rejected: partition values are stored
        // in plaintext (in manifests, used for pruning), which would leak the
        // very values the encryption is meant to protect.
        #[cfg(feature = "encryption")]
        if let Some(enc) = &options.encryption {
            if let Some(parts) = &schema.partition_by {
                for p in parts {
                    let encrypted =
                        enc.encrypted_columns.is_empty() || enc.encrypted_columns.contains(p);
                    if encrypted {
                        return Err(IcefallDBError::Encryption(format!(
                            "partition column '{p}' cannot be encrypted: partition values are \
                             stored in plaintext for pruning. Partition by a non-encrypted \
                             column, or do not encrypt '{p}'."
                        )));
                    }
                }
            }
        }

        let arrow_schema = arrow_schema_from_icefalldb(&schema, table)?;
        let lock_path = format!("{}/_write.lock", table);

        // Serialize concurrent table initialization through the exclusive writer
        // lock. The lock is released when `lock_guard` drops at the end of this
        // function. When the caller already holds the lock (e.g. the CLI holding
        // the view-table lock across DuckDB execution), skip re-acquisition.
        let _lock_guard: Option<Box<dyn crate::storage::LockGuard>> = if options.assume_lock_held {
            None
        } else {
            Some(
                storage
                    .lock_exclusive(&lock_path, options.lock_timeout)
                    .await?,
            )
        };

        // `_schema.json` is the authoritative marker that a table exists. Check it
        // under the writer lock so that concurrent initialization races are
        // serialized deterministically.
        let pointer_path = format!("{}/_schema.json", table);
        let pointer_exists = storage.exists(&pointer_path).await?;

        if pointer_exists {
            if mode == CreateMode::CreateNew {
                return Err(IcefallDBError::TableAlreadyExists(table.to_string()));
            }

            let data = storage.read(&pointer_path).await?;
            let pointer: serde_json::Value = serde_json::from_slice(&data).map_err(|_| {
                IcefallDBError::InvalidSchemaPointer {
                    path: pointer_path.clone(),
                }
            })?;
            let pointer_id = pointer
                .get("latest")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| IcefallDBError::InvalidSchemaPointer {
                    path: pointer_path.clone(),
                })?;

            if schema.schema_id != pointer_id {
                return Err(IcefallDBError::SchemaMismatch {
                    column: "schema_id".into(),
                    expected: schema.schema_id.to_string(),
                    path: pointer_path.clone(),
                });
            }

            let mut previous_schema =
                load_existing_schema_file(storage.as_ref(), table, pointer_id).await?;

            // Backward compatibility: if an existing schema was written before field
            // IDs were introduced, repair it in memory and persist the repaired IDs
            // on this write.
            let mut repaired_schema = false;
            if !previous_schema.has_field_ids() {
                previous_schema.repair_field_ids();
                repaired_schema = true;
            }

            schema.assign_field_ids(Some(&previous_schema));

            // Enforce schema immutability: the normalized schema must match the
            // persisted schema exactly.
            if previous_schema != schema {
                return Err(IcefallDBError::SchemaMismatch {
                    column: "schema".into(),
                    expected: "match existing content".into(),
                    path: pointer_path.clone(),
                });
            }

            // Persist repaired field IDs back to the schema file.
            if repaired_schema {
                let schema_path =
                    format!("{}/{}", table, Schema::filename(previous_schema.schema_id));
                let schema_data = serde_json::to_vec(&previous_schema)?;
                let tmp_path = format!("{}.tmp", schema_path);
                storage.write(&tmp_path, &schema_data).await?;
                storage.sync(&tmp_path).await?;
                storage.rename(&tmp_path, &schema_path).await?;
                storage.sync(&format!("{}/_schemas", table)).await?;
            }

            // Enforce encryption-state immutability: a table that has (or has
            // not) an `_encryption.json` marker must be opened with matching
            // encryption options. This prevents silently appending plaintext
            // rows to an encrypted table (or vice versa) via the legacy
            // `Writer::new` constructor that does not pass encryption options,
            // AND prevents reopening an encrypted table with a different key
            // set / AAD prefix / plaintext-footer setting (which would write
            // row groups that the original marker can no longer describe).
            #[cfg(feature = "encryption")]
            {
                let marker_path = format!("{table}/_encryption.json");
                let marker_present = storage.exists(&marker_path).await?;
                let options_encrypted = options.encryption.is_some();
                if marker_present && !options_encrypted {
                    return Err(IcefallDBError::Encryption(format!(
                        "table '{table}' is encrypted (has _encryption.json) but no \
                         encryption config was provided; pass \
                         WriterOptionsFull::with_encryption(...) when opening"
                    )));
                }
                if !marker_present && options_encrypted {
                    return Err(IcefallDBError::Encryption(format!(
                        "table '{table}' is not encrypted but encryption options were \
                         provided; remove the encryption config to open as plaintext"
                    )));
                }
                if marker_present && options_encrypted {
                    // Both sides are encrypted — verify the new options match
                    // the stored marker. The marker is the source of truth for
                    // readers; mismatched writes would corrupt the table.
                    let enc = options.encryption.as_ref().expect("checked above");
                    let footer_kid = format!("{table}-v{}", schema.schema_id);
                    let expected = SchemaEncryptionMarker::for_write_config(&footer_kid, enc);
                    let stored_bytes = storage.read(&marker_path).await?;
                    let stored: SchemaEncryptionMarker = serde_json::from_slice(&stored_bytes)
                        .map_err(|e| {
                            IcefallDBError::Encryption(format!(
                                "failed to parse {marker_path}: {e}"
                            ))
                        })?;
                    if stored.algorithm != expected.algorithm
                        || stored.plaintext_footer != expected.plaintext_footer
                        || stored.aad_prefix != expected.aad_prefix
                        || stored.column_key_ids != expected.column_key_ids
                    {
                        return Err(IcefallDBError::Encryption(format!(
                            "encryption config does not match the marker at {marker_path}; \
                             rotating keys requires an explicit migration tool \
                             (planned, not yet implemented)"
                        )));
                    }
                    // footer_key_id is informational (it identifies which key
                    // to resolve); we don't compare it because future key
                    // rotation may legitimately change it without changing
                    // the algorithm/AAD/encryption mode.
                }
            }
        } else {
            // Staging directories are only needed for tables that will receive
            // writes; create them as part of initialization, not when merely
            // opening an existing table.
            ensure_dir(storage.as_ref(), &format!("{}/_staging/intents", table)).await?;
            ensure_dir(storage.as_ref(), &format!("{}/_staging/incoming", table)).await?;

            initialize_table_locked(storage.as_ref(), table, &mut schema).await?;

            // If encryption is enabled, write a sidecar `<table>/_encryption.json`
            // marker so the reader knows the table is encrypted, the algorithm,
            // and which key identifiers to resolve. The marker contains NO key
            // material — only references.
            #[cfg(feature = "encryption")]
            if let Some(enc) = &options.encryption {
                write_encryption_marker_locked(storage.as_ref(), table, schema.schema_id, enc)
                    .await?;
            }
        }

        Ok(Self {
            storage,
            table: table.to_string(),
            schema,
            arrow_schema,
            buffer: vec![],
            buffered_rows: 0,
            buffered_bytes: 0,
            lock_timeout: options.lock_timeout,
            assume_lock_held: options.assume_lock_held,
            wal_mode: false,
            wal_checkpoint_threshold: 64,
            #[cfg(feature = "encryption")]
            encryption: options.encryption,
        })
    }

    /// Enable deferred-commit (mutation WAL) mode for `commit_deletes`. See the
    /// `wal_mode` field. Opt-in; the default writer swaps the manifest per DELETE.
    pub fn with_wal_mode(mut self, on: bool) -> Self {
        self.wal_mode = on;
        self
    }

    pub async fn insert_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if !schema_equal_ignoring_metadata(batch.schema().as_ref(), &self.arrow_schema) {
            return Err(IcefallDBError::SchemaMismatch {
                column: "schema".into(),
                expected: "match writer schema".into(),
                path: self.table.clone(),
            });
        }
        // Empty batches must not advance the sequence or create empty commits.
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.buffered_rows += batch.num_rows();
        self.buffered_bytes += batch.get_array_memory_size();
        self.buffer.push(batch);
        Ok(())
    }

    /// Insert a local Parquet file using a zero-copy fast path when possible.
    ///
    /// The source file is read once in a streaming pass, its SHA-256 checksum is
    /// computed incrementally, and the bytes are copied directly into the table
    /// without decoding and re-encoding. Row-group statistics are derived from
    /// the Parquet footer statistics.
    ///
    /// If the file extension is not `.parquet`, the source schema is incompatible,
    /// or the footer statistics are missing/unsupported, this method returns
    /// [`InsertParquetOutcome::Incompatible`] and leaves the writer untouched. The
    /// caller can then fall back to the normal decode/re-encode path.
    ///
    /// On success, the commit is durable and the method returns the number of rows
    /// inserted via the fast path.
    pub async fn insert_parquet(&mut self, source_path: &str) -> Result<InsertParquetOutcome> {
        // The fast path is unsafe for encrypted tables: it copies the source
        // bytes verbatim into the table, which would persist plaintext data
        // under an encrypted-table marker. Force the caller down the
        // decode/re-encode path (which `write_row_group_part` runs through the
        // encryption-aware `ArrowWriter`) when encryption is configured.
        #[cfg(feature = "encryption")]
        if self.encryption.is_some() {
            return Ok(InsertParquetOutcome::Incompatible);
        }

        // Reject empty files early so we do not commit empty row groups.
        let metadata = tokio::fs::metadata(source_path).await.map_err(other)?;
        if metadata.len() == 0 {
            return Ok(InsertParquetOutcome::Incompatible);
        }

        // Stream the source file into memory once, hashing incrementally.
        let mut file = tokio::fs::File::open(source_path).await.map_err(other)?;
        let mut hasher = Sha256::new();
        let mut parquet_bytes = Vec::with_capacity(metadata.len() as usize);
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf).await.map_err(other)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            parquet_bytes.extend_from_slice(&buf[..n]);
        }
        let data_checksum = format!("sha256:{}", hex::encode(hasher.finalize()));

        // Parse the Parquet footer from the bytes we just read.
        let bytes = Bytes::from(parquet_bytes);
        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes.clone()).map_err(other)?;
        let source_schema = builder.schema().clone();
        let parquet_metadata = builder.metadata().clone();

        if !self.parquet_schema_compatible(&source_schema) {
            return Ok(InsertParquetOutcome::Incompatible);
        }

        let num_row_groups = parquet_metadata.num_row_groups();
        if num_row_groups != 1 {
            // IcefallDB row groups must map one-to-one with Parquet row groups so
            // that sidecar column_offsets describe the data that is actually read.
            return Ok(InsertParquetOutcome::Incompatible);
        }

        let num_rows = parquet_metadata.file_metadata().num_rows();
        if num_rows == 0 {
            return Ok(InsertParquetOutcome::Incompatible);
        }

        let (columns, partition_values) =
            match self.derive_footer_stats(&parquet_metadata, num_rows as usize)? {
                Some(result) => result,
                None => return Ok(InsertParquetOutcome::Incompatible),
            };

        let mut meta = RowGroupMeta {
            row_group: String::new(), // assigned during staging
            schema_id: self.schema.schema_id,
            rows: num_rows as usize,
            columns,
            column_offsets: compute_column_offsets(&parquet_metadata, &self.schema),
            sort: self.schema.sort.clone(),
            row_ids: vec![],
            checksum: data_checksum.clone(),
            meta_checksum: String::new(),
        };

        // Acquire the writer lock unless the caller already holds it. Keep the
        // guard alive in the outer function scope until the commit returns so
        // the manifest pointer swap is covered by the exclusive lock.
        let _lock: Option<Box<dyn crate::storage::LockGuard>> = if !self.assume_lock_held {
            let lock_path = format!("{}/_write.lock", self.table);
            Some(
                self.storage
                    .lock_exclusive(&lock_path, self.lock_timeout)
                    .await?,
            )
        } else {
            None
        };

        // Content-addressed dedup (referencing an already-committed identical
        // data file instead of copying it) skips row-id allocation and all index
        // maintenance. That is unsafe when a UNIQUE index exists: the
        // re-referenced fragment would re-add the same keys as live rows, with
        // duplicate row_ids and no uniqueness probe, silently violating the
        // invariant in a way later index rebuilds cannot even detect. When the
        // table has a unique index, fall through to the normal copy path, which
        // allocates fresh row_ids and runs `check_unique_adds` (rejecting the
        // duplicate keys).
        // ponytail: gate on unique indexes only; non-unique index staleness in
        // the dedup path is a separate, pre-existing concern outside M01 scope.
        let dedup_allowed = {
            let catalog = crate::database_catalog::DatabaseCatalog::new(self.storage.clone());
            let catalog_data = catalog.load().await?;
            !catalog_data.indexes.values().any(|entry| {
                entry.table == self.table && entry.unique && entry.index_type == "btree"
            })
        };

        // Check whether an identical row group is already committed; if so,
        // reference the existing data file instead of copying it again.
        let (latest_seq, current_manifest) = self.load_current_manifest().await?;
        let existing_row_groups = &current_manifest.row_groups;
        if dedup_allowed {
            for rg in existing_row_groups {
                let meta_path = format!("{}/{}", self.table, rg.meta);
                let meta_bytes = self.storage.read(&meta_path).await?;
                let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
                if meta.checksum == data_checksum {
                    return self
                        .append_reference_row_group(
                            rg,
                            latest_seq,
                            existing_row_groups,
                            partition_values.clone(),
                            &current_manifest,
                        )
                        .await;
                }
            }
        }

        let mut rollback = RollbackInfo::default();
        if let Err(e) = self
            .try_commit_parquet_fast(&bytes, &mut meta, partition_values, &mut rollback)
            .await
        {
            self.rollback_commit(
                &rollback.intent_path,
                &rollback.staged_files,
                rollback.sequence,
                rollback.pointer_updated,
            )
            .await;
            return Err(e);
        }

        // Best-effort cleanup of the intent file now that the commit is durable.
        let _ = self.storage.delete(&rollback.intent_path).await;
        let _ = self
            .storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await;

        Ok(InsertParquetOutcome::FastPath {
            rows: num_rows as usize,
        })
    }

    /// Append a new manifest entry that references an existing data file whose
    /// checksum matches the incoming Parquet file. A fresh `.meta` sidecar is
    /// written with a new row-group id so each snapshot remains independent.
    async fn append_reference_row_group(
        &mut self,
        existing: &RowGroupEntry,
        latest_seq: u64,
        existing_row_groups: &[RowGroupEntry],
        partition_values: Option<HashMap<String, serde_json::Value>>,
        current_manifest: &Manifest,
    ) -> Result<InsertParquetOutcome> {
        let next_seq = latest_seq + 1;

        // Clean up any orphan staging files from prior failed commits before
        // this commit creates new files.
        let referenced_files = Self::manifest_referenced_files(current_manifest);
        cleanup_staging(
            self.storage.as_ref(),
            &self.table,
            latest_seq,
            &referenced_files,
        )
        .await?;

        // Generate a fresh row-group id that does not collide.
        let mut used: HashSet<String> = HashSet::new();
        for rg in existing_row_groups {
            used.insert(
                rg.data
                    .strip_suffix(".parquet")
                    .unwrap_or(&rg.data)
                    .to_string(),
            );
            used.insert(
                rg.meta
                    .strip_suffix(".meta")
                    .unwrap_or(&rg.meta)
                    .to_string(),
            );
        }
        let rg_id = unique_rg_id(&mut used);
        let meta_filename = format!("{}.meta", rg_id);
        let meta_path = format!("{}/{}", self.table, meta_filename);

        // Allocate a fresh fragment ID and bump the high-water mark so the
        // checkpoint reuse map and scan planner see a unique fragment identity.
        let fragment_id = current_manifest.next_fragment_id;
        let next_fragment_id = fragment_id + 1;

        // Read the existing meta, change only the row_group id, and recompute its checksum.
        let existing_meta_path = format!("{}/{}", self.table, existing.meta);
        let existing_meta_bytes = self.storage.read(&existing_meta_path).await?;
        let mut meta: RowGroupMeta = serde_json::from_slice(&existing_meta_bytes)?;
        meta.row_group = rg_id;
        meta.compute_meta_checksum()?;
        let meta_json = serde_json::to_vec(&meta)?;
        self.storage.write(&meta_path, &meta_json).await?;
        self.storage.sync_data(&meta_path).await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // Build the new manifest, reusing the existing data file.
        let mut row_groups = existing_row_groups.to_vec();
        row_groups.push(RowGroupEntry {
            data: existing.data.clone(),
            meta: meta_filename.clone(),
            fragment_id,
            ..Default::default()
        });

        let mut partition_values_map: HashMap<String, HashMap<String, serde_json::Value>> =
            HashMap::new();
        // Copy any partition values keyed on the existing data file.
        if let Some(existing_pv) = current_manifest
            .partition_values
            .as_ref()
            .and_then(|m| m.get(&existing.data))
        {
            partition_values_map.insert(existing.data.clone(), existing_pv.clone());
        }
        if let Some(pv) = partition_values {
            partition_values_map.insert(existing.data.clone(), pv);
        }

        let row_counts = collect_row_counts(
            self.storage.as_ref(),
            &self.table,
            existing_row_groups,
            current_manifest,
            &[meta.rows],
        )
        .await;

        let mut manifest = Manifest {
            format_version: 1,
            sequence: next_seq,
            schema_id: self.schema.schema_id,
            row_groups,
            row_counts,
            partition_values: if partition_values_map.is_empty() {
                None
            } else {
                Some(partition_values_map)
            },
            next_row_id: current_manifest.next_row_id,
            next_fragment_id,
            checksum: String::new(),
            ..Default::default()
        };

        // Emit the snapshot checkpoint inside the atomic commit.
        let checkpoint = build_snapshot_checkpoint(
            self.storage.as_ref(),
            &self.table,
            current_manifest,
            &manifest,
        )
        .await?;

        ensure_dir(
            self.storage.as_ref(),
            &format!("{}/_staging/intents", self.table),
        )
        .await?;
        let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
        let intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
        let mut intent_files = vec![meta_filename.clone()];
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": intent_files,
        });
        self.storage
            .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(&intent_path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;

        let mut _staged_for_rollback = Vec::new();
        let checkpoint_path = write_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &checkpoint,
            next_seq,
            &intent_path,
            &mut intent_files,
            &mut _staged_for_rollback,
            &txn_id,
            self.schema.schema_id,
        )
        .await?;
        manifest.checkpoint = Some(checkpoint_path);
        self.finalize_manifest(&mut manifest, next_seq).await?;

        let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
        let manifest_tmp_path = format!("{}.tmp", manifest_path);
        let manifest_data = serde_json::to_vec(&manifest)?;
        self.storage
            .write(&manifest_tmp_path, &manifest_data)
            .await?;
        self.storage.sync_data(&manifest_tmp_path).await?;
        // Publish: rename .tmp -> final, then one dir fsync to make the rename
        // durable. The content is already durable (sync_data above); a pre-rename
        // dir fsync for the disposable tmp adds nothing for crash recovery
        // (recovery uses the intent journal + pointer, never the tmp name).
        self.storage
            .rename(&manifest_tmp_path, &manifest_path)
            .await?;
        self.storage
            .sync(&format!("{}/_manifests", self.table))
            .await?;

        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        let pointer = serde_json::json!({"latest": next_seq});
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // Best-effort cleanup of the intent file now that the commit is durable.
        let _ = self.storage.delete(&intent_path).await;
        // No fsync of the intents dir: a leftover intent after a crash here is
        // reclaimed by cleanup_staging on the next open, so the deletion need not
        // be durable (matches the MERGE commit path).

        Ok(InsertParquetOutcome::FastPath { rows: meta.rows })
    }

    /// Returns true if the Parquet file's Arrow schema can be copied directly
    /// into this table. Column names, types, and order must match, and source
    /// nullability must be no wider than the table nullability.
    fn parquet_schema_compatible(&self, source: &ArrowSchema) -> bool {
        if source.fields().len() != self.arrow_schema.fields().len() {
            return false;
        }
        source
            .fields()
            .iter()
            .zip(self.arrow_schema.fields().iter())
            .all(|(source_field, table_field)| {
                source_field.name() == table_field.name()
                    && source_field.data_type() == table_field.data_type()
                    && (!source_field.is_nullable() || table_field.is_nullable())
            })
    }

    /// Derive row-group statistics from the Parquet footer.
    ///
    /// Returns `None` if any column is unsupported or has missing statistics.
    /// Whether this writer encrypts at least one column, so the plaintext
    /// `.agg` aggregate sidecar must be suppressed (it would leak SUM/SUMSQ and
    /// group partials of encrypted columns).
    #[cfg(feature = "encryption")]
    fn is_encrypted(&self) -> bool {
        self.encryption.is_some()
    }
    #[cfg(not(feature = "encryption"))]
    fn is_encrypted(&self) -> bool {
        false
    }

    /// Whether column `name`'s data is encrypted on disk. An empty
    /// `encrypted_columns` set means whole-table / footer-key encryption (every
    /// column is encrypted); a non-empty set is column-level encryption.
    #[cfg(feature = "encryption")]
    fn is_column_encrypted(&self, name: &str) -> bool {
        match &self.encryption {
            None => false,
            Some(cfg) => cfg.encrypted_columns.is_empty() || cfg.encrypted_columns.contains(name),
        }
    }
    #[cfg(not(feature = "encryption"))]
    fn is_column_encrypted(&self, _name: &str) -> bool {
        false
    }

    /// Drop plaintext statistics for encrypted columns from a `RowGroupMeta`
    /// and recompute its checksum, so the persisted `.meta` (and any checkpoint
    /// that copies it) cannot leak the protected values. No-op for plaintext
    /// tables. Applied by every path that writes a `RowGroupMeta` from a
    /// plaintext batch (insert, and UPDATE/MERGE patch fragments).
    #[cfg(feature = "encryption")]
    fn redact_encrypted_meta(&self, meta: &mut RowGroupMeta, parquet_bytes: &[u8]) -> Result<()> {
        if self.encryption.is_some() {
            meta.columns
                .retain(|name, _| !self.is_column_encrypted(name));
            meta.compute_checksum(parquet_bytes)?;
        }
        Ok(())
    }
    #[cfg(not(feature = "encryption"))]
    fn redact_encrypted_meta(&self, _meta: &mut RowGroupMeta, _parquet_bytes: &[u8]) -> Result<()> {
        Ok(())
    }

    fn derive_footer_stats(
        &self,
        metadata: &parquet::file::metadata::ParquetMetaData,
        _num_rows: usize,
    ) -> Result<Option<FooterStats>> {
        let schema_descr = metadata.file_metadata().schema_descr();
        let mut columns = HashMap::new();

        for col in &self.schema.columns {
            // Encrypted columns get no plaintext statistics: their min/max/null
            // counts would otherwise leak the protected values into the
            // `.meta` and checkpoint sidecars, defeating the encryption.
            if self.is_column_encrypted(&col.name) {
                continue;
            }
            let arrow_type = match icefalldb_type_to_arrow(&col.r#type) {
                Some(t) => t,
                None => return Ok(None),
            };

            if !is_supported_footer_stat_type(&arrow_type) {
                return Ok(None);
            }

            let leaf_idx = match schema_descr
                .columns()
                .iter()
                .position(|c| c.name() == col.name)
            {
                Some(i) => i,
                None => return Ok(None),
            };

            let mut nulls: usize = 0;
            let mut min: Option<serde_json::Value> = None;
            let mut max: Option<serde_json::Value> = None;

            for rg in metadata.row_groups() {
                let col_meta = rg.column(leaf_idx);
                let stats = match col_meta.statistics() {
                    Some(s) => s,
                    None => return Ok(None),
                };

                nulls += stats.null_count_opt().unwrap_or(0) as usize;

                let (rg_min, rg_max) = parquet_stats_to_json(stats, &arrow_type)?;
                min = min_json(min, rg_min);
                max = max_json(max, rg_max);
            }

            columns.insert(col.name.clone(), ColumnStats { min, max, nulls });
        }

        let partition_values = self.derive_partition_values_from_stats(&columns);

        Ok(Some((columns, partition_values)))
    }

    /// Derive partition values from footer statistics when every partition column
    /// has a single distinct non-null value across the whole file.
    fn derive_partition_values_from_stats(
        &self,
        columns: &HashMap<String, ColumnStats>,
    ) -> Option<HashMap<String, serde_json::Value>> {
        let partition_by = self.schema.partition_by.as_ref()?;
        if partition_by.is_empty() {
            return None;
        }

        let mut values = HashMap::new();
        for col_name in partition_by {
            let stats = columns.get(col_name)?;
            if stats.nulls == 0 && stats.min == stats.max && stats.min.is_some() {
                values.insert(col_name.clone(), stats.min.clone().unwrap());
            } else {
                return None;
            }
        }

        if values.is_empty() {
            None
        } else {
            Some(values)
        }
    }

    /// Executes the body of a fast-path Parquet commit.
    async fn try_commit_parquet_fast(
        &mut self,
        parquet_bytes: &Bytes,
        meta: &mut RowGroupMeta,
        partition_values: Option<HashMap<String, serde_json::Value>>,
        rollback: &mut RollbackInfo,
    ) -> Result<()> {
        let (latest_seq, current_manifest) = self.load_current_manifest().await?;
        let existing_row_groups = current_manifest.row_groups.clone();

        // Re-verify the schema pointer under the lock.
        let schema_pointer_path = format!("{}/_schema.json", self.table);
        let schema_pointer_data = self.storage.read(&schema_pointer_path).await?;
        let schema_pointer: serde_json::Value = serde_json::from_slice(&schema_pointer_data)
            .map_err(|_| IcefallDBError::InvalidSchemaPointer {
                path: schema_pointer_path.clone(),
            })?;
        let pointer_id = schema_pointer
            .get("latest")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| IcefallDBError::InvalidSchemaPointer {
                path: schema_pointer_path.clone(),
            })?;
        if pointer_id != self.schema.schema_id {
            return Err(IcefallDBError::SchemaMismatch {
                column: "schema_id".into(),
                expected: self.schema.schema_id.to_string(),
                path: schema_pointer_path.clone(),
            });
        }

        let referenced_files = Self::manifest_referenced_files(&current_manifest);
        cleanup_staging(
            self.storage.as_ref(),
            &self.table,
            latest_seq,
            &referenced_files,
        )
        .await?;

        let next_seq = latest_seq + 1;
        let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
        if self.storage.exists(&manifest_path).await? {
            return Err(IcefallDBError::SequenceCollision(next_seq));
        }

        // Allocate stable row IDs and a fragment ID for this single fragment,
        // reading high-water marks from the current manifest (0 for a fresh
        // table).  The bumped values are carried into the new manifest so
        // successive ingest paths (commit / insert_parquet / commit) produce
        // disjoint row-id ranges and monotonically increasing fragment IDs.
        let mut next_row_id = current_manifest.next_row_id;
        let mut next_fragment_id = current_manifest.next_fragment_id;
        let row_id_seg = allocate_range(&mut next_row_id, meta.rows as u64);
        let fragment_id = next_fragment_id;
        next_fragment_id += 1;
        meta.row_ids = vec![row_id_seg];

        // Stage the Parquet file with collision retry.
        let mut used_rg_ids: HashSet<String> = HashSet::new();
        let planned_rg_id = unique_rg_id(&mut used_rg_ids);
        let planned_data_filename = format!("{}.parquet", planned_rg_id);
        let planned_meta_filename = format!("{}.meta", planned_rg_id);

        // Write intent before staging data so recovery knows what to clean up if
        // the writer crashes.
        let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
        let intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": [planned_data_filename.clone(), planned_meta_filename.clone()],
        });
        self.storage
            .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(&intent_path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;
        rollback.intent_path = intent_path.clone();

        let (data_filename, meta_filename) = self
            .stage_and_rename_parquet(
                parquet_bytes,
                meta,
                &planned_rg_id,
                &mut used_rg_ids,
                rollback,
            )
            .await?;

        // If collision resolution picked a different id, rewrite the intent so
        // recovery sees the files that were actually staged.
        if data_filename != planned_data_filename {
            let intent = serde_json::json!({
                "txn_id": txn_id,
                "started_at": chrono::Utc::now().to_rfc3339(),
                "schema_id": self.schema.schema_id,
                "files": [data_filename.clone(), meta_filename.clone()],
            });
            // In-place rewrite of the existing intent (durable directory entry
            // from the initial intent fsync above; stable inode), so `sync_data`
            // for the new content is enough — no second `_staging/intents`
            // directory fsync (batching, extended to the parquet-fast path).
            self.storage
                .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
                .await?;
            self.storage.sync_data(&intent_path).await?;
        }

        self.storage
            .sync(&format!("{}/_staging/incoming", self.table))
            .await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // Build the new manifest, appending the copied row group.
        let mut row_groups = existing_row_groups;
        row_groups.push(RowGroupEntry {
            data: data_filename.clone(),
            meta: meta_filename.clone(),
            fragment_id,
            ..Default::default()
        });

        // Carry forward existing partition values and add the new row group.
        let mut partition_values_map = current_manifest
            .partition_values
            .clone()
            .unwrap_or_default();
        if let Some(pv) = partition_values {
            partition_values_map.insert(data_filename.clone(), pv);
        }
        let partition_values_map = if partition_values_map.is_empty() {
            None
        } else {
            Some(partition_values_map)
        };

        let row_counts = collect_row_counts(
            self.storage.as_ref(),
            &self.table,
            &current_manifest.row_groups,
            &current_manifest,
            &[meta.rows],
        )
        .await;

        let mut manifest = Manifest {
            format_version: 1,
            sequence: next_seq,
            schema_id: self.schema.schema_id,
            row_groups,
            row_counts,
            partition_values: partition_values_map,
            // Carry the bumped high-water marks forward so that any ingest path
            // that follows (commit or insert_parquet) produces disjoint row-id
            // ranges and monotonically increasing fragment IDs.
            next_row_id,
            next_fragment_id,
            checksum: String::new(),
            ..Default::default()
        };

        // Rebuild secondary indexes for this snapshot and populate
        // `manifest.index_generations` BEFORE the manifest is serialized and the
        // pointer is swapped, so the manifest is written exactly once already
        // containing the index generations. The index base files are appended to
        // the commit intent so recovery treats them as durable.
        let mut intent_files = self
            .build_indexes_into_manifest(
                &mut manifest,
                &current_manifest,
                &[data_filename.clone(), meta_filename.clone()],
                &intent_path,
                &txn_id,
                true, // buffered insert is a pure append
            )
            .await?;

        // Emit the snapshot checkpoint inside the atomic commit.
        let checkpoint = build_snapshot_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &current_manifest,
            &manifest,
        )
        .await?;
        let checkpoint_path = write_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &checkpoint,
            next_seq,
            &intent_path,
            &mut intent_files,
            &mut rollback.staged_files,
            &txn_id,
            self.schema.schema_id,
        )
        .await?;
        manifest.checkpoint = Some(checkpoint_path);

        self.finalize_manifest(&mut manifest, next_seq).await?;

        // Write the manifest snapshot atomically with checksum retry.
        let manifest_data = serde_json::to_vec(&manifest)?;
        let manifest_tmp_path = format!("{}.tmp", manifest_path);
        self.storage
            .write(&manifest_tmp_path, &manifest_data)
            .await?;
        self.storage.sync_data(&manifest_tmp_path).await?;

        if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
            self.storage
                .write(&manifest_tmp_path, &manifest_data)
                .await?;
            self.storage.sync_data(&manifest_tmp_path).await?;
            if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
                return Err(IcefallDBError::ChecksumMismatch {
                    path: manifest_tmp_path,
                });
            }
        }

        // Publish: rename then one dir fsync. Content is durable (sync_data
        // above); the pre-rename dir fsync for the disposable tmp added nothing.
        self.storage
            .rename(&manifest_tmp_path, &manifest_path)
            .await?;
        self.storage
            .sync(&format!("{}/_manifests", self.table))
            .await?;

        // Update the manifest pointer durably.
        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer = serde_json::json!({"latest": next_seq});
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;

        if let Err(e) = self.storage.sync(&format!("{}/", self.table)).await {
            rollback.pointer_updated = true;
            return Err(e);
        }
        rollback.pointer_updated = true;
        Ok(())
    }

    /// Stage a pre-built Parquet row group, rename it to its final filename, and
    /// return the final filenames. Retries with a new row-group id if the target
    /// filenames already exist.
    async fn stage_and_rename_parquet(
        &self,
        parquet_bytes: &Bytes,
        meta: &mut RowGroupMeta,
        planned_rg_id: &str,
        used_rg_ids: &mut HashSet<String>,
        rollback: &mut RollbackInfo,
    ) -> Result<(String, String)> {
        // The first attempt uses the caller-supplied planned id. Subsequent
        // attempts generate fresh ids.
        let mut rg_id = planned_rg_id.to_string();
        for attempt in 0..3 {
            let data_filename = format!("{}.parquet", rg_id);
            let meta_filename = format!("{}.meta", rg_id);
            let parquet_part = format!("{}/_staging/incoming/{}.parquet.part", self.table, rg_id);
            let meta_part = format!("{}/_staging/incoming/{}.meta.part", self.table, rg_id);
            let parquet_final = format!("{}/{}", self.table, data_filename);
            let meta_final = format!("{}/{}", self.table, meta_filename);

            // On retries the metadata struct still carries the previous row_group
            // id; refresh it and recompute its checksum.
            meta.row_group = rg_id.clone();
            meta.compute_meta_checksum()?;

            self.storage.write(&parquet_part, parquet_bytes).await?;
            let meta_json = serde_json::to_vec(&meta)?;
            self.storage.write(&meta_part, &meta_json).await?;

            if self.storage.exists(&parquet_final).await?
                || self.storage.exists(&meta_final).await?
            {
                if attempt == 2 {
                    return Err(IcefallDBError::Other(Box::new(std::io::Error::other(
                        "failed to find a unique row group id after retries",
                    ))));
                }
                let _ = self.storage.delete(&parquet_part).await;
                let _ = self.storage.delete(&meta_part).await;
                rg_id = unique_rg_id(used_rg_ids);
                continue;
            }

            self.storage.rename(&parquet_part, &parquet_final).await?;
            self.storage.rename(&meta_part, &meta_final).await?;
            rollback.staged_files.push(data_filename.clone());
            rollback.staged_files.push(meta_filename.clone());
            return Ok((data_filename, meta_filename));
        }

        Err(IcefallDBError::Other(Box::new(std::io::Error::other(
            "failed to find a unique row group id after retries",
        ))))
    }

    /// Commit the buffered batches to the table.
    ///
    /// Commits are serialized through an exclusive lock acquired at the very
    /// start of mutation. The lock provides in-process exclusion for all
    /// writers created from the same storage instance, and `LocalStorage` also
    /// uses `flock()` for cross-process exclusion.
    ///
    /// # Important
    ///
    /// A failed `commit` may already have succeeded: once the manifest pointer
    /// has been updated, the writer treats the commit as durable even if a
    /// subsequent fsync returns an error. Callers that need to know whether the
    /// commit is visible must re-read the table (for example via a [`Catalog`]
    /// or manifest scan) rather than simply retrying the failed commit.
    ///
    /// [`Catalog`]: crate::catalog::Catalog
    pub async fn commit(&mut self) -> Result<CommitDelta> {
        // Empty commits are idempotent and must not acquire the lock or advance
        // the sequence. Return a no-op delta so callers can still observe the
        // current snapshot sequence.
        if self.buffer.is_empty() {
            let (_, current_manifest) = self.load_current_manifest().await?;
            return Ok(CommitDelta::new(
                &current_manifest,
                &current_manifest,
                CommitKind::Noop,
            ));
        }

        // Serialize all commits through an exclusive lock acquired at the very
        // start of mutation. Time out using the configured timeout so a stuck
        // lock does not hang the writer forever. If the caller already holds
        // the lock, skip re-acquisition. Keep the guard alive in the outer
        // function scope until the commit returns so the manifest pointer swap
        // is covered by the exclusive lock.
        let _lock: Option<Box<dyn crate::storage::LockGuard>> = if !self.assume_lock_held {
            let lock_path = format!("{}/_write.lock", self.table);
            Some(
                self.storage
                    .lock_exclusive(&lock_path, self.lock_timeout)
                    .await?,
            )
        } else {
            None
        };

        let (_, current_manifest) = self.load_current_manifest().await?;
        let mut rollback = RollbackInfo::default();
        let (new_manifest, added_row_groups) = match self.try_commit(&mut rollback, false).await {
            Ok(m) => m,
            Err(e) => {
                self.rollback_commit(
                    &rollback.intent_path,
                    &rollback.staged_files,
                    rollback.sequence,
                    rollback.pointer_updated,
                )
                .await;
                // If the manifest pointer was already updated, the commit may be
                // durable despite the error. Clear the buffer so a retry does not
                // duplicate the commit and report the error to the caller.
                if rollback.pointer_updated {
                    self.buffer.clear();
                    self.buffered_rows = 0;
                    self.buffered_bytes = 0;
                }
                return Err(e);
            }
        };

        // The commit is durable; clear the buffered batches now.
        self.buffer.clear();
        self.buffered_rows = 0;
        self.buffered_bytes = 0;

        // The pointer has already been advanced, so any remaining cleanup is
        // best-effort. Do not return an error; a failure here must not cause the
        // caller to retry and duplicate the commit.
        //
        // Secondary indexes were already rebuilt and recorded in the committed
        // manifest's `index_generations` inside `try_commit` (the manifest is
        // immutable once written), so there is nothing index-related to do here.
        let _ = self.storage.delete(&rollback.intent_path).await;
        let _ = self
            .storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await;

        Ok(
            CommitDelta::new(&current_manifest, &new_manifest, CommitKind::Append)
                .with_added_row_groups(added_row_groups),
        )
    }

    /// Atomically replace the current table snapshot with the buffered batches.
    ///
    /// Like [`Writer::commit`], this acquires the exclusive writer lock, writes
    /// new row groups, and advances the manifest pointer. Unlike `commit`, the
    /// new manifest contains **only** the newly written row groups; all existing
    /// row groups are dropped from the latest snapshot. Old files become
    /// unreferenced and are cleaned up later by garbage collection.
    ///
    /// An empty replace (no buffered batches) commits a new manifest with no row
    /// groups, making the table empty. This is used by view refreshes that
    /// return zero rows.
    pub async fn replace(&mut self) -> Result<CommitDelta> {
        // If the caller already holds the writer lock, skip re-acquisition.
        // Keep the guard alive in the outer function scope until the replace
        // returns so the manifest pointer swap is covered by the exclusive lock.
        let _lock: Option<Box<dyn crate::storage::LockGuard>> = if !self.assume_lock_held {
            let lock_path = format!("{}/_write.lock", self.table);
            Some(
                self.storage
                    .lock_exclusive(&lock_path, self.lock_timeout)
                    .await?,
            )
        } else {
            None
        };

        let (_, current_manifest) = self.load_current_manifest().await?;
        let mut rollback = RollbackInfo::default();
        let (new_manifest, added_row_groups) = match self.try_commit(&mut rollback, true).await {
            Ok(m) => m,
            Err(e) => {
                self.rollback_commit(
                    &rollback.intent_path,
                    &rollback.staged_files,
                    rollback.sequence,
                    rollback.pointer_updated,
                )
                .await;
                if rollback.pointer_updated {
                    self.buffer.clear();
                    self.buffered_rows = 0;
                    self.buffered_bytes = 0;
                }
                return Err(e);
            }
        };

        self.buffer.clear();
        self.buffered_rows = 0;
        self.buffered_bytes = 0;

        // Index generations for this snapshot were written atomically inside
        // `try_commit`; the committed manifest is immutable, so no post-commit
        // index step is needed here.
        let _ = self.storage.delete(&rollback.intent_path).await;
        let _ = self
            .storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await;

        Ok(
            CommitDelta::new(&current_manifest, &new_manifest, CommitKind::Replace)
                .with_added_row_groups(added_row_groups),
        )
    }

    /// Mark rows as logically deleted and commit a new manifest.
    ///
    /// `by_fragment` maps each fragment id to a list of physical row offsets
    /// within that fragment that should be marked dead.  For each entry:
    ///
    /// 1. The fragment's current deletion vector is loaded (or a fresh empty
    ///    one is created if none exists yet).
    /// 2. The new offsets are unioned in — re-inserting an already-dead offset
    ///    is idempotent.
    /// 3. A new `_deletions/rg_<id>__v<seq>.del` file is written.
    /// 4. The fragment's `RowGroupEntry` is updated: `deletes` ← new path,
    ///    `deleted_count` ← `dv.cardinality()` (NOT an increment — setting to
    ///    cardinality makes the operation idempotent end-to-end).
    ///
    /// The new manifest is published via the same atomic-commit protocol used
    /// by [`Writer::commit`]: intent journal → fsync → manifest write → pointer
    /// swap.  The `.del` file paths are added to BOTH the intent journal and
    /// `manifest_referenced_files` so that subsequent commits' staging cleanup
    /// never treats a committed deletion vector as an orphan.
    ///
    /// Secondary indexes and the row index are preserved unchanged: a DELETE
    /// does not relocate rows, so `_rowindex` remains valid; non-unique indexes
    /// use a query-time liveness mask.
    pub async fn commit_deletes(
        &mut self,
        by_fragment: HashMap<u64, Vec<u32>>,
    ) -> Result<CommitDelta> {
        let lock_path = format!("{}/_write.lock", self.table);
        let _lock = self
            .storage
            .lock_exclusive(&lock_path, self.lock_timeout)
            .await?;

        // ── Load current state ───────────────────────────────────────────────
        let (latest_seq, current_manifest) = self.load_current_manifest().await?;

        // Run the standard staging cleanup so orphaned intents / .part files
        // from crashed prior writers are removed before we read `referenced_files`.
        let referenced_files = Self::manifest_referenced_files(&current_manifest);
        cleanup_staging(
            self.storage.as_ref(),
            &self.table,
            latest_seq,
            &referenced_files,
        )
        .await?;

        let next_seq = latest_seq + 1;
        let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
        if self.storage.exists(&manifest_path).await? {
            return Err(IcefallDBError::SequenceCollision(next_seq));
        }

        // Ensure the _deletions/ directory exists.
        ensure_dir(self.storage.as_ref(), &format!("{}/_deletions", self.table)).await?;

        // ── Build updated row-group entries and write .del files ─────────────
        let mut new_del_paths: Vec<String> = Vec::new();
        // In WAL mode, small artifacts (deletion vectors, index deltas) are
        // written without their own `fsync` and their bytes inlined here, so the
        // single WAL-record `fsync` makes the whole commit durable (~1 `fsync`).
        let mut wal_artifacts: Vec<crate::mutation_wal::StagedArtifact> = Vec::new();
        let mut row_groups = current_manifest.row_groups.clone();

        for entry in &mut row_groups {
            let offsets = match by_fragment.get(&entry.fragment_id) {
                Some(o) if !o.is_empty() => o,
                _ => continue,
            };

            // Load existing deletion vector or start with an empty one.
            let mut dv = if let Some(del_path) = &entry.deletes {
                let bytes = self
                    .storage
                    .read(&format!("{}/{}", self.table, del_path))
                    .await?;
                crate::DeletionVector::deserialize(&bytes)
                    .map_err(|e| IcefallDBError::Other(Box::new(e)))?
            } else {
                crate::DeletionVector::default()
            };

            dv.union_offsets(offsets.iter().copied());

            // Derive the new sequence number for this fragment's deletion file.
            // Parse the version embedded in the current path (if any); the new
            // version is one higher.  For a first-ever delete we start at v1.
            // Return an error rather than silently defaulting to v1 on an
            // unparseable path — a silent collision would overwrite the existing
            // .del file without warning.
            let new_ver = if let Some(cur) = &entry.deletes {
                // filename format: rg_<id>__v<seq>.del
                cur.rsplit("__v")
                    .next()
                    .and_then(|s| s.strip_suffix(".del"))
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|v| v + 1)
                    .ok_or_else(|| {
                        IcefallDBError::Other(Box::new(std::io::Error::other(format!(
                            "commit_deletes: cannot parse version from existing .del path {:?}; \
                             expected format _deletions/rg_<id>__v<N>.del",
                            cur
                        ))))
                    })?
            } else {
                1
            };

            // Use the fragment_id as the stable identifier in the filename so
            // successive versions of the same fragment's deletion vector are
            // easy to trace.
            let rel_del_path =
                format!("_deletions/rg_{:016x}__v{}.del", entry.fragment_id, new_ver);
            let abs_del_path = format!("{}/{}", self.table, rel_del_path);

            let del_bytes = dv.serialize();
            self.storage.write(&abs_del_path, &del_bytes).await?;
            if self.wal_mode {
                wal_artifacts.push(crate::mutation_wal::StagedArtifact {
                    path: rel_del_path.clone(),
                    hex: hex::encode(&del_bytes),
                });
            } else {
                self.storage.sync_data(&abs_del_path).await?;
            }

            entry.deletes = Some(rel_del_path.clone());
            entry.deleted_count = dv.cardinality();
            new_del_paths.push(rel_del_path);
        }

        // Sync the _deletions/ directory so all .del files are durable before
        // the intent is written. (Skipped in WAL mode — durability comes from the
        // inlined record `fsync`; the checkpoint `fsync`s the files.)
        if !self.wal_mode {
            self.storage
                .sync(&format!("{}/_deletions", self.table))
                .await?;
        }

        // ── Eager tombstones for UNIQUE indexes ──────────────────────────────
        // Non-unique indexes rely on the query-time liveness mask; unique
        // indexes get an explicit tombstone delta so the uniqueness invariant
        // (one live row_id per key) is maintained without a full index rebuild.
        let mut updated_index_generations = current_manifest.index_generations.clone();
        let mut new_tombstone_paths: Vec<String> = Vec::new();

        // Collect the deleted row_ids across all affected fragments.
        // offset → row_id mapping comes from each fragment's RowGroupMeta.
        //
        // Legacy fragments may have an empty `row_ids`
        // field.  For non-unique indexes that is harmless (they use a
        // query-time liveness mask), but for UNIQUE indexes it would produce
        // an incomplete tombstone set — the deleted key could still appear
        // live via the index.  Track whether any fragment was skipped so we
        // can reject the operation if unique indexes would be affected.
        let mut all_deleted_row_ids: Vec<u64> = Vec::new();
        let mut fragment_ids_with_empty_row_ids: Vec<u64> = Vec::new();
        for rg in &current_manifest.row_groups {
            let offsets = match by_fragment.get(&rg.fragment_id) {
                Some(o) if !o.is_empty() => o,
                _ => continue,
            };
            let meta_path = format!("{}/{}", self.table, rg.meta);
            let meta_bytes = self.storage.read(&meta_path).await?;
            let meta: crate::metadata::RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
            if meta.row_ids.is_empty() {
                // Cannot map offsets to row_ids for this fragment.
                fragment_ids_with_empty_row_ids.push(rg.fragment_id);
                continue;
            }
            let row_id_vec: Vec<u64> = meta
                .row_ids
                .iter()
                .flat_map(crate::rowid::segment_ids)
                .collect();
            for &offset in offsets {
                if let Some(&row_id) = row_id_vec.get(offset as usize) {
                    all_deleted_row_ids.push(row_id);
                }
            }
        }

        // Load unique index definitions now so we can check whether skipped
        // fragments (empty row_ids) would leave a uniqueness-invariant hole.
        let catalog = crate::database_catalog::DatabaseCatalog::new(self.storage.clone());
        let catalog_data = catalog.load().await?;
        let has_unique_index = catalog_data.indexes.iter().any(|(idx_name, entry)| {
            entry.table == self.table
                && entry.unique
                && entry.index_type == "btree"
                && updated_index_generations.contains_key(idx_name)
        });

        // Guard: a fragment with empty row_ids cannot contribute its deleted
        // row_ids to the tombstone set.  For a UNIQUE index that is a
        // correctness hole — the deleted key remains resolvable via the index.
        // Current-format fragments always carry row_ids; this protects
        // against importing legacy data into a table that has a unique index.
        if has_unique_index && !fragment_ids_with_empty_row_ids.is_empty() {
            return Err(IcefallDBError::Other(Box::new(std::io::Error::other(
                format!(
                    "commit_deletes: fragment(s) {:?} have no row_ids — cannot \
                     produce a complete tombstone set for unique index on table '{}'; \
                     rebuild the index after migrating legacy fragments",
                    fragment_ids_with_empty_row_ids, self.table
                ),
            ))));
        }

        if !all_deleted_row_ids.is_empty() {
            for (idx_name, entry) in &catalog_data.indexes {
                if entry.table != self.table || !entry.unique || entry.index_type != "btree" {
                    continue;
                }
                // Only tombstone an index that has an existing generation in
                // this manifest (no-op if the index was never built).
                let current_ref = match updated_index_generations.get(idx_name) {
                    Some(r) => r.clone(),
                    None => continue,
                };
                let new_ref = crate::index::append_tombstones(
                    self.storage.as_ref(),
                    &self.table,
                    idx_name,
                    &all_deleted_row_ids,
                    next_seq,
                    &current_ref,
                    !self.wal_mode,
                )
                .await?;
                // Record the new delta path for the intent journal.
                if let Some(new_delta) = new_ref.deltas.last() {
                    new_tombstone_paths.push(new_delta.clone());
                    if self.wal_mode {
                        // Inline the (un-`fsync`ed) delta's bytes into the record.
                        let bytes = self
                            .storage
                            .read(&format!("{}/{}", self.table, new_delta))
                            .await?;
                        wal_artifacts.push(crate::mutation_wal::StagedArtifact {
                            path: new_delta.clone(),
                            hex: hex::encode(&bytes),
                        });
                    }
                }
                updated_index_generations.insert(idx_name.clone(), new_ref);
            }
        }

        // ── WAL mode: defer the manifest swap; append one compact record ──────
        // The deletion-vector / tombstone artifacts are already written + synced
        // above. Instead of materializing a full manifest and running the
        // pointer-swap ceremony, append a MutationRecord (one fsync) and let a
        // later checkpoint fold it into a real manifest. Recovery and readers see
        // the deferred delete by replaying the log onto the checkpoint manifest.
        if self.wal_mode {
            let fragment_deletes: Vec<crate::mutation_wal::FragmentDelete> = row_groups
                .iter()
                .filter(|e| {
                    by_fragment
                        .get(&e.fragment_id)
                        .is_some_and(|offs| !offs.is_empty())
                })
                .filter_map(|e| {
                    e.deletes
                        .clone()
                        .map(|deletes| crate::mutation_wal::FragmentDelete {
                            fragment_id: e.fragment_id,
                            deletes,
                            deleted_count: e.deleted_count,
                        })
                })
                .collect();

            let record = crate::mutation_wal::MutationRecord {
                sequence: next_seq,
                base_sequence: latest_seq,
                fragment_deletes,
                fragment_adds: Vec::new(),
                index_generations: updated_index_generations.clone(),
                rowindex_generation: None,
                next_fragment_id: 0,
                next_row_id: current_manifest.next_row_id,
                staged_artifacts: wal_artifacts,
                checksum: String::new(),
            }
            .sealed();
            crate::mutation_wal::append(self.storage.as_ref(), &self.table, &record).await?;

            // In-memory post-commit manifest for the returned delta — identical to
            // what a checkpoint will later materialize for this sequence.
            let new_manifest = Manifest {
                format_version: 1,
                sequence: next_seq,
                schema_id: current_manifest.schema_id,
                row_groups,
                row_counts: current_manifest.row_counts.clone(),
                partition_values: current_manifest.partition_values.clone(),
                next_row_id: current_manifest.next_row_id,
                next_fragment_id: current_manifest.next_fragment_id,
                rowindex_generation: current_manifest.rowindex_generation.clone(),
                index_generations: updated_index_generations,
                checksum: String::new(),
                ..Default::default()
            };

            // Fold the log into a real manifest once it grows past the threshold;
            // we already hold the write lock.
            let pending = crate::mutation_wal::read_records(self.storage.as_ref(), &self.table)
                .await?
                .len();
            if pending >= self.wal_checkpoint_threshold {
                crate::mutation_wal::checkpoint_locked(self.storage.as_ref(), &self.table).await?;
            }

            return Ok(CommitDelta::new(
                &current_manifest,
                &new_manifest,
                CommitKind::Delete,
            ));
        }

        // ── Write intent journal listing all new .del + tombstone delta files ──
        // The intent tells recovery which files belong to this commit.
        // Listing the files here ensures that:
        //   (a) An aborted commit (no pointer swap) → cleanup_staging deletes them.
        //   (b) A crash after pointer swap → the next cleanup_staging sees them in
        //       referenced_files (they are in the new manifest) and preserves them.
        let mut intent_files = new_del_paths.clone();
        intent_files.extend(new_tombstone_paths);
        let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
        let intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": intent_files,
        });
        self.storage
            .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(&intent_path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;

        // ── Build and publish the new manifest ───────────────────────────────
        // rowindex_generation is preserved unchanged: a DELETE does not
        // relocate rows so the _rowindex address map stays valid.
        // index_generations is updated: unique indexes got a tombstone delta
        // appended above; non-unique indexes are unchanged (liveness mask).
        let mut new_manifest = Manifest {
            format_version: 1,
            sequence: next_seq,
            schema_id: current_manifest.schema_id,
            row_groups,
            row_counts: current_manifest.row_counts.clone(),
            partition_values: current_manifest.partition_values.clone(),
            next_row_id: current_manifest.next_row_id,
            next_fragment_id: current_manifest.next_fragment_id,
            rowindex_generation: current_manifest.rowindex_generation.clone(),
            index_generations: updated_index_generations,
            checksum: String::new(),
            ..Default::default()
        };

        // Emit the snapshot checkpoint inside the atomic commit.
        let checkpoint = build_snapshot_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &current_manifest,
            &new_manifest,
        )
        .await?;
        let mut _staged_for_rollback = Vec::new();
        let checkpoint_path = write_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &checkpoint,
            next_seq,
            &intent_path,
            &mut intent_files,
            &mut _staged_for_rollback,
            &txn_id,
            self.schema.schema_id,
        )
        .await?;
        new_manifest.checkpoint = Some(checkpoint_path);

        self.finalize_manifest(&mut new_manifest, next_seq).await?;

        // Write manifest .tmp, verify checksum, then rename to final path.
        let manifest_data = serde_json::to_vec(&new_manifest)?;
        let manifest_tmp_path = format!("{}.tmp", manifest_path);
        self.storage
            .write(&manifest_tmp_path, &manifest_data)
            .await?;
        self.storage.sync_data(&manifest_tmp_path).await?;

        if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
            self.storage
                .write(&manifest_tmp_path, &manifest_data)
                .await?;
            self.storage.sync_data(&manifest_tmp_path).await?;
            if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
                let _ = self.storage.delete(&intent_path).await;
                return Err(IcefallDBError::ChecksumMismatch {
                    path: manifest_tmp_path,
                });
            }
        }

        // Publish: rename then one dir fsync. Content is durable (sync_data
        // above); the pre-rename dir fsync for the disposable tmp added nothing.
        self.storage
            .rename(&manifest_tmp_path, &manifest_path)
            .await?;
        self.storage
            .sync(&format!("{}/_manifests", self.table))
            .await?;

        // Atomic durability point: update the manifest pointer.
        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer = serde_json::json!({"latest": next_seq});
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // Best-effort cleanup of the intent file.
        let _ = self.storage.delete(&intent_path).await;
        // No fsync of the intents dir: a leftover intent after a crash here is
        // reclaimed by cleanup_staging on the next open, so the deletion need not
        // be durable (matches the MERGE commit path).

        Ok(CommitDelta::new(
            &current_manifest,
            &new_manifest,
            CommitKind::Delete,
        ))
    }

    /// Replace a set of rows in-place (move-stable UPDATE).
    ///
    /// `rows` contains the post-SET values for the rows being updated, in the
    /// same order as `locs`.  `locs` describes where each row currently lives
    /// (fragment, physical offset) and what its stable `row_id` is.
    ///
    /// The method:
    /// 1. Sorts `(rows, locs)` by `row_id` so the patch fragment's layout is
    ///    deterministic.
    /// 2. Writes a **new patch fragment** (fresh `fragment_id` from
    ///    `next_fragment_id`) whose `row_ids` are the original, stable IDs —
    ///    `next_row_id` is NOT advanced.
    /// 3. Tombstones the originals: groups `locs` by `fragment_id`, unions
    ///    offsets into deletion vectors, sets `deleted_count = cardinality`.
    /// 4. Writes a `_rowindex/delta` file mapping each `row_id` to its new
    ///    `(patch_fragment_id, new_offset)` and appends it to the manifest's
    ///    `rowindex_generation.deltas`.
    /// 5. Incrementally maintains secondary indexes: only indexes whose column
    ///    appears in `set_columns` are updated (tombstone+add delta); all others
    ///    are carried forward unchanged from the current manifest.
    /// 6. Publishes the new manifest via the atomic-commit protocol.
    ///
    /// `set_columns` should contain the names of columns in the SQL `SET` clause.
    /// Pass an empty slice if no indexed columns are being changed (e.g. updating
    /// only non-indexed columns), which skips all index delta writes.
    pub async fn commit_update(
        &mut self,
        rows: RecordBatch,
        locs: Vec<MatchLoc>,
        set_columns: &[String],
    ) -> Result<CommitDelta> {
        if rows.num_rows() == 0 || locs.is_empty() {
            return Err(IcefallDBError::Other(
                "commit_update: rows and locs must be non-empty".into(),
            ));
        }
        if rows.num_rows() != locs.len() {
            return Err(IcefallDBError::Other(
                format!(
                    "commit_update: rows ({}) and locs ({}) must have equal length",
                    rows.num_rows(),
                    locs.len()
                )
                .into(),
            ));
        }

        // Defensive schema check: the patch batch must match the writer's schema
        // by field names and data types before any files are written. Nullability
        // is intentionally ignored: DataFusion projections of non-null literals
        // may report a non-nullable output field even when the target column is
        // nullable.
        if !schema_equal_names_and_types(rows.schema().as_ref(), &self.arrow_schema) {
            return Err(IcefallDBError::SchemaMismatch {
                column: "schema".into(),
                expected: "match writer schema names and types".into(),
                path: self.table.clone(),
            });
        }

        // ── Step 1: sort rows+locs by row_id ────────────────────────────────
        let mut order: Vec<usize> = (0..locs.len()).collect();
        order.sort_unstable_by_key(|&i| locs[i].row_id);

        let sorted_locs: Vec<MatchLoc> = order.iter().map(|&i| locs[i]).collect();
        // Reorder the batch rows according to the sorted order.
        let sorted_rows = {
            let indices = arrow::array::UInt64Array::from(
                order.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let cols: Vec<arrow::array::ArrayRef> = rows
                .columns()
                .iter()
                .map(|col| arrow::compute::take(col.as_ref(), &indices, None).map_err(other))
                .collect::<Result<_>>()?;
            RecordBatch::try_new(rows.schema(), cols).map_err(other)?
        };

        // Sorted stable row IDs for the patch fragment's row_ids field.
        let sorted_row_ids: Vec<u64> = sorted_locs.iter().map(|l| l.row_id).collect();

        // ── Acquire the exclusive writer lock ────────────────────────────────
        let lock_path = format!("{}/_write.lock", self.table);
        let _lock = self
            .storage
            .lock_exclusive(&lock_path, self.lock_timeout)
            .await?;

        // ── Load current manifest ────────────────────────────────────────────
        let (latest_seq, current_manifest) = self.load_current_manifest().await?;

        let referenced_files = Self::manifest_referenced_files(&current_manifest);
        cleanup_staging(
            self.storage.as_ref(),
            &self.table,
            latest_seq,
            &referenced_files,
        )
        .await?;

        let next_seq = latest_seq + 1;
        let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
        if self.storage.exists(&manifest_path).await? {
            return Err(IcefallDBError::SequenceCollision(next_seq));
        }

        // ── Step 2: allocate a fresh fragment_id (do NOT advance next_row_id) ─
        let patch_fragment_id = current_manifest.next_fragment_id;
        let next_fragment_id = patch_fragment_id + 1;

        // ── Step 3: write the patch .parquet + .meta to staging ─────────────
        ensure_dir(
            self.storage.as_ref(),
            &format!("{}/_staging/incoming", self.table),
        )
        .await?;
        ensure_dir(
            self.storage.as_ref(),
            &format!("{}/_staging/intents", self.table),
        )
        .await?;

        let mut used_rg_ids: HashSet<String> = HashSet::new();
        let patch_rg_id = unique_rg_id(&mut used_rg_ids);

        // Encode the patch batch as Parquet.
        let mut parquet_bytes: Vec<u8> = Vec::new();
        {
            #[allow(unused_mut)]
            let mut props_builder = parquet::file::properties::WriterProperties::builder()
                .set_compression(parquet::basic::Compression::ZSTD(
                    parquet::basic::ZstdLevel::try_new(1).expect("valid zstd level"),
                ));
            #[cfg(feature = "encryption")]
            if let Some(enc) = &self.encryption {
                let fp = crate::encryption::build_encryption_properties(
                    &enc.keys,
                    enc.plaintext_footer,
                    enc.store_aad_prefix,
                    &enc.encrypted_columns,
                )?;
                props_builder = props_builder.with_file_encryption_properties(fp);
            }
            let props = props_builder.build();
            let mut writer =
                ArrowWriter::try_new(&mut parquet_bytes, self.arrow_schema.clone(), Some(props))
                    .map_err(other)?;
            writer.write(&sorted_rows).map_err(other)?;
            writer.close().map_err(other)?;
        }

        // Build the patch row_ids segment (Sorted, since these IDs may not be
        // contiguous — they are the original stable IDs of the updated rows).
        let patch_row_id_seg = crate::rowid::RowIdSegment::Sorted {
            ids: sorted_row_ids.clone(),
        };

        let mut patch_meta = compute_row_group_meta(
            &patch_rg_id,
            self.schema.schema_id,
            &sorted_rows,
            &self.schema,
            &parquet_bytes,
            &self.table,
            std::slice::from_ref(&patch_row_id_seg),
        )?;
        // Redact encrypted-column stats from the patch fragment metadata.
        self.redact_encrypted_meta(&mut patch_meta, &parquet_bytes)?;

        // Write .part files.
        let parquet_part = format!(
            "{}/_staging/incoming/{}.parquet.part",
            self.table, patch_rg_id
        );
        let meta_part = format!("{}/_staging/incoming/{}.meta.part", self.table, patch_rg_id);
        self.storage.write(&parquet_part, &parquet_bytes).await?;
        self.storage
            .write(&meta_part, &serde_json::to_vec(&patch_meta)?)
            .await?;
        self.storage
            .sync(&format!("{}/_staging/incoming", self.table))
            .await?;

        // Build the final filenames and check for collision.
        let patch_data_filename = format!("{}.parquet", patch_rg_id);
        let patch_meta_filename = format!("{}.meta", patch_rg_id);
        let parquet_final = format!("{}/{}", self.table, patch_data_filename);
        let meta_final = format!("{}/{}", self.table, patch_meta_filename);

        if self.storage.exists(&parquet_final).await? || self.storage.exists(&meta_final).await? {
            return Err(IcefallDBError::Other(
                "commit_update: patch rg_id collision; retry".into(),
            ));
        }

        // ── Step 4 (pre-compute): compute .del paths and delta path upfront ──
        // All of these paths are deterministic before any side-effecting write:
        //   • patch data/meta filenames already known above
        //   • .del paths: derived from current manifest fragment_ids and existing
        //     .del version numbers
        //   • delta path: derived from next_seq
        // Compute them all now so we can write the COMPLETE intent ONCE before
        // touching any of the files they name.  A crash after the intent is
        // written but before the pointer swap will leave every written file named
        // in the intent, so cleanup_staging can remove all orphans.

        // Group locs by fragment_id.
        let mut by_fragment: HashMap<u64, Vec<u32>> = HashMap::new();
        for loc in &sorted_locs {
            by_fragment
                .entry(loc.fragment_id)
                .or_default()
                .push(loc.offset);
        }

        // ── Pre-compute: read existing DVs and determine .del paths ───────────
        // We collect (fragment_id, dv, cardinality, rel_del_path) for each
        // affected fragment so we can (a) list paths in the intent, and (b) write
        // the files in a separate pass after the intent is durable.
        struct DelWork {
            fragment_id: u64,
            rel_del_path: String,
            abs_del_path: String,
            del_bytes: Vec<u8>,
            cardinality: u64,
        }
        let mut del_work: Vec<DelWork> = Vec::new();
        let mut new_del_paths: Vec<String> = Vec::new();
        let mut row_groups = current_manifest.row_groups.clone();

        for entry in &row_groups {
            let offsets = match by_fragment.get(&entry.fragment_id) {
                Some(o) if !o.is_empty() => o,
                _ => continue,
            };

            let mut dv = if let Some(del_path) = &entry.deletes {
                let bytes = self
                    .storage
                    .read(&format!("{}/{}", self.table, del_path))
                    .await?;
                crate::DeletionVector::deserialize(&bytes)
                    .map_err(|e| IcefallDBError::Other(Box::new(e)))?
            } else {
                crate::DeletionVector::default()
            };

            dv.union_offsets(offsets.iter().copied());

            let new_ver = if let Some(cur) = &entry.deletes {
                cur.rsplit("__v")
                    .next()
                    .and_then(|s| s.strip_suffix(".del"))
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(|v| v + 1)
                    .ok_or_else(|| {
                        IcefallDBError::Other(Box::new(std::io::Error::other(format!(
                            "commit_update: cannot parse version from .del path {:?}",
                            cur
                        ))))
                    })?
            } else {
                1
            };

            let rel_del_path =
                format!("_deletions/rg_{:016x}__v{}.del", entry.fragment_id, new_ver);
            let abs_del_path = format!("{}/{}", self.table, rel_del_path);
            let cardinality = dv.cardinality();
            let del_bytes = dv.serialize();
            new_del_paths.push(rel_del_path.clone());
            del_work.push(DelWork {
                fragment_id: entry.fragment_id,
                rel_del_path,
                abs_del_path,
                del_bytes,
                cardinality,
            });
        }

        // ── Pre-compute: rowindex delta path ──────────────────────────────────
        let rel_delta_path = format!("_rowindex/delta__v{:09}.idx", next_seq);

        // ── Build complete intent file list and write it ONCE, BEFORE any
        // side-effecting writes.  Every file that will be created by this commit
        // is listed here so recovery can clean up orphans on any crash path.
        let mut all_intent_files = vec![patch_data_filename.clone(), patch_meta_filename.clone()];
        all_intent_files.extend(new_del_paths.iter().cloned());
        all_intent_files.push(rel_delta_path.clone());

        let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
        let intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": all_intent_files,
        });
        self.storage
            .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(&intent_path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;

        // ── Side-effecting writes begin here (intent is already durable) ──────

        // Rename .part → final.
        self.storage.rename(&parquet_part, &parquet_final).await?;
        self.storage.rename(&meta_part, &meta_final).await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // ── Step 4: tombstone original fragments ─────────────────────────────
        ensure_dir(self.storage.as_ref(), &format!("{}/_deletions", self.table)).await?;

        // Apply the pre-computed deletion vectors: write .del files and update
        // the in-memory row_groups entries. In WAL mode the small artifacts
        // (tombstone DVs, rowindex delta) are written without their own fsync and
        // inlined into the record; the patch parquet + index deltas keep theirs.
        let mut wal_artifacts: Vec<crate::mutation_wal::StagedArtifact> = Vec::new();
        for work in &del_work {
            self.storage
                .write(&work.abs_del_path, &work.del_bytes)
                .await?;
            if self.wal_mode {
                wal_artifacts.push(crate::mutation_wal::StagedArtifact {
                    path: work.rel_del_path.clone(),
                    hex: hex::encode(&work.del_bytes),
                });
            } else {
                self.storage.sync_data(&work.abs_del_path).await?;
            }
        }
        // Update the in-memory row_groups entries.
        for entry in &mut row_groups {
            if let Some(work) = del_work.iter().find(|w| w.fragment_id == entry.fragment_id) {
                entry.deletes = Some(work.rel_del_path.clone());
                entry.deleted_count = work.cardinality;
            }
        }

        if !self.wal_mode {
            self.storage
                .sync(&format!("{}/_deletions", self.table))
                .await?;
        }

        // ── Step 5: build rowindex delta ─────────────────────────────────────
        ensure_dir(self.storage.as_ref(), &format!("{}/_rowindex", self.table)).await?;

        // Each sorted row maps to offset 0, 1, 2, … within the patch fragment.
        // Coalesce consecutive rows into AddrSegments.
        let mut delta_segs: Vec<crate::rowindex::AddrSegment> = Vec::new();
        for (patch_offset, &row_id) in sorted_row_ids.iter().enumerate() {
            let patch_offset = patch_offset as u32;
            if let Some(last) = delta_segs.last_mut() {
                if last.fragment_id == patch_fragment_id
                    && last.start_row_id + u64::from(last.len) == row_id
                    && last.start_offset + last.len == patch_offset
                {
                    last.len += 1;
                    continue;
                }
            }
            delta_segs.push(crate::rowindex::AddrSegment {
                start_row_id: row_id,
                fragment_id: patch_fragment_id,
                start_offset: patch_offset,
                len: 1,
            });
        }

        let delta_bytes = crate::rowindex::encode_idx(&delta_segs);
        let abs_delta_path = format!("{}/{}", self.table, rel_delta_path);
        self.storage.write(&abs_delta_path, &delta_bytes).await?;
        if self.wal_mode {
            wal_artifacts.push(crate::mutation_wal::StagedArtifact {
                path: rel_delta_path.clone(),
                hex: hex::encode(&delta_bytes),
            });
        } else {
            self.storage.sync_data(&abs_delta_path).await?;
            self.storage
                .sync(&format!("{}/_rowindex", self.table))
                .await?;
        }

        // ── Step 6: build updated rowindex_generation ────────────────────────
        let mut new_rowindex_gen = current_manifest
            .rowindex_generation
            .clone()
            .unwrap_or_default();
        new_rowindex_gen.deltas.push(rel_delta_path.clone());

        // Append patch fragment to row_groups.
        // Note: .agg is not produced on the commit_update path (only
        // insert_batch produces it); the metadata-aggregate rule falls back for fragments without .agg.
        row_groups.push(RowGroupEntry {
            data: patch_data_filename.clone(),
            meta: patch_meta_filename.clone(),
            fragment_id: patch_fragment_id,
            deletes: None,
            deleted_count: 0,
            agg: None,
        });

        // ── Step 7: build and publish manifest ───────────────────────────────
        let mut new_manifest = Manifest {
            format_version: 1,
            sequence: next_seq,
            schema_id: current_manifest.schema_id,
            row_groups,
            row_counts: None, // will be denormalized by collect_row_counts if needed
            partition_values: current_manifest.partition_values.clone(),
            next_row_id: current_manifest.next_row_id, // NOT advanced — move-stable
            next_fragment_id,
            rowindex_generation: Some(new_rowindex_gen),
            index_generations: current_manifest.index_generations.clone(),
            checksum: String::new(),
            ..Default::default()
        };

        // ── Step 7a: incremental index maintenance ───────────────────────────
        // Only indexes whose column is in `set_columns` get a tombstone+add
        // delta; all others are carried forward UNCHANGED from current_manifest.
        {
            let index_delta_paths = IndexMaintainer::maintain_on_update(
                self.storage.clone(),
                &self.table,
                &mut new_manifest,
                set_columns,
                &sorted_rows,
                &sorted_row_ids,
                next_seq,
            )
            .await?;
            if !index_delta_paths.is_empty() {
                // Extend the intent file list to include the new delta files and
                // rewrite the intent so recovery knows they are staged.
                all_intent_files.extend(index_delta_paths);
                let intent_upd = serde_json::json!({
                    "txn_id": txn_id,
                    "started_at": chrono::Utc::now().to_rfc3339(),
                    "schema_id": self.schema.schema_id,
                    "files": &all_intent_files,
                });
                // In-place rewrite of the existing intent (durable directory entry
                // from the initial intent fsync; stable inode): `sync_data` for the
                // new content suffices — no second `_staging/intents` directory
                // fsync (batching, extended to the UPDATE path).
                self.storage
                    .write(&intent_path, serde_json::to_vec(&intent_upd)?.as_slice())
                    .await?;
                self.storage.sync_data(&intent_path).await?;
            }
        }

        // ── WAL mode: defer the manifest swap; append one record ──────────────
        // The patch fragment, tombstone .del files, rowindex delta and index
        // deltas are already written + synced above; record the manifest diff
        // (the patch fragment add + the tombstone updates) and skip the manifest
        // materialization + pointer-swap ceremony. Readers/recovery replay it.
        if self.wal_mode {
            let current_ids: std::collections::HashSet<u64> = current_manifest
                .row_groups
                .iter()
                .map(|e| e.fragment_id)
                .collect();
            let mut fragment_deletes: Vec<crate::mutation_wal::FragmentDelete> = Vec::new();
            for entry in &new_manifest.row_groups {
                if !current_ids.contains(&entry.fragment_id) {
                    continue; // a newly added fragment, handled below
                }
                let changed = current_manifest
                    .row_groups
                    .iter()
                    .find(|e| e.fragment_id == entry.fragment_id)
                    .map(|old| old.deletes != entry.deletes)
                    .unwrap_or(false);
                if changed {
                    if let Some(deletes) = &entry.deletes {
                        fragment_deletes.push(crate::mutation_wal::FragmentDelete {
                            fragment_id: entry.fragment_id,
                            deletes: deletes.clone(),
                            deleted_count: entry.deleted_count,
                        });
                    }
                }
            }
            let fragment_adds: Vec<crate::metadata::RowGroupEntry> = new_manifest
                .row_groups
                .iter()
                .filter(|e| !current_ids.contains(&e.fragment_id))
                .cloned()
                .collect();

            let record = crate::mutation_wal::MutationRecord {
                sequence: next_seq,
                base_sequence: latest_seq,
                fragment_deletes,
                fragment_adds,
                index_generations: new_manifest.index_generations.clone(),
                rowindex_generation: new_manifest.rowindex_generation.clone(),
                next_fragment_id: new_manifest.next_fragment_id,
                next_row_id: new_manifest.next_row_id,
                staged_artifacts: wal_artifacts,
                checksum: String::new(),
            }
            .sealed();
            crate::mutation_wal::append(self.storage.as_ref(), &self.table, &record).await?;

            // The deferred manifest now references these staged files; the WAL
            // record is the recovery authority, so drop the intent journal.
            let _ = self.storage.delete(&intent_path).await;

            let pending = crate::mutation_wal::read_records(self.storage.as_ref(), &self.table)
                .await?
                .len();
            if pending >= self.wal_checkpoint_threshold {
                crate::mutation_wal::checkpoint_locked(self.storage.as_ref(), &self.table).await?;
            }

            // The returned delta must carry the patch fragment as an added
            // PlannedRowGroup so the in-process provider's apply_committed_delta
            // adds it to the live snapshot (same as the non-WAL path below).
            let patch_partition_values = new_manifest
                .partition_values
                .as_ref()
                .and_then(|m| m.get(&patch_data_filename).cloned());
            let patch_planned = crate::reader::PlannedRowGroup {
                data_path: format!("{}/{}", self.table, patch_data_filename),
                meta_path: format!("{}/{}", self.table, patch_meta_filename),
                meta: patch_meta.clone(),
                partition_values: patch_partition_values,
                snapshot: new_manifest.sequence,
                fallback: false,
                deletes: None,
                deleted_count: 0,
                fragment_id: patch_fragment_id,
                agg_state: None,
            };
            return Ok(
                CommitDelta::new(&current_manifest, &new_manifest, CommitKind::Update)
                    .with_added_row_groups(vec![patch_planned]),
            );
        }

        // Emit the snapshot checkpoint inside the atomic commit.
        let checkpoint = build_snapshot_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &current_manifest,
            &new_manifest,
        )
        .await?;
        let mut _staged_for_rollback = Vec::new();
        let checkpoint_path = write_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &checkpoint,
            next_seq,
            &intent_path,
            &mut all_intent_files,
            &mut _staged_for_rollback,
            &txn_id,
            self.schema.schema_id,
        )
        .await?;
        new_manifest.checkpoint = Some(checkpoint_path);

        self.finalize_manifest(&mut new_manifest, next_seq).await?;

        let manifest_data = serde_json::to_vec(&new_manifest)?;
        let manifest_tmp_path = format!("{}.tmp", manifest_path);
        self.storage
            .write(&manifest_tmp_path, &manifest_data)
            .await?;
        self.storage.sync_data(&manifest_tmp_path).await?;

        if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
            self.storage
                .write(&manifest_tmp_path, &manifest_data)
                .await?;
            self.storage.sync_data(&manifest_tmp_path).await?;
            if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
                return Err(IcefallDBError::ChecksumMismatch {
                    path: manifest_tmp_path,
                });
            }
        }

        // Publish the manifest: rename .tmp -> final, then fsync the manifests
        // directory once so the rename is durable. (The data is already durable
        // via sync_data above; a separate pre-rename dir sync added nothing.)
        self.storage
            .rename(&manifest_tmp_path, &manifest_path)
            .await?;
        self.storage
            .sync(&format!("{}/_manifests", self.table))
            .await?;

        // Atomic durability point: update the manifest pointer.
        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer = serde_json::json!({"latest": next_seq});
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // Best-effort cleanup of the intent file. A leftover intent after a
        // future crash is recovered by `cleanup_staging`, so we do NOT fsync the
        // intents directory here — that dir sync added commit latency for no
        // correctness benefit.
        let _ = self.storage.delete(&intent_path).await;

        let patch_partition_values = new_manifest
            .partition_values
            .as_ref()
            .and_then(|m| m.get(&patch_data_filename).cloned());
        let patch_planned = crate::reader::PlannedRowGroup {
            data_path: format!("{}/{}", self.table, patch_data_filename),
            meta_path: format!("{}/{}", self.table, patch_meta_filename),
            meta: patch_meta,
            partition_values: patch_partition_values,
            snapshot: new_manifest.sequence,
            fallback: false,
            deletes: None,
            deleted_count: 0,
            fragment_id: patch_fragment_id,
            agg_state: None,
        };

        Ok(
            CommitDelta::new(&current_manifest, &new_manifest, CommitKind::Update)
                .with_added_row_groups(vec![patch_planned]),
        )
    }

    /// Stage a pre-encoded MERGE fragment (`.parquet` + `.meta`) to the incoming
    /// staging area and rename it to its final table-root filename.
    ///
    /// The final filename is derived from `data_filename`'s `rg_<uuid>` stem. A
    /// collision with an existing final file is fatal (the rg_ids are freshly
    /// generated UUIDs, so a collision indicates a logic error, not contention).
    async fn stage_merge_fragment(
        &self,
        data_filename: &str,
        meta_filename: &str,
        parquet_bytes: &[u8],
        meta: &RowGroupMeta,
    ) -> Result<()> {
        let meta_json = serde_json::to_vec(meta)?;
        let rg_id = data_filename.trim_end_matches(".parquet");
        let parquet_part = format!("{}/_staging/incoming/{}.parquet.part", self.table, rg_id);
        let meta_part = format!("{}/_staging/incoming/{}.meta.part", self.table, rg_id);
        let parquet_final = format!("{}/{}", self.table, data_filename);
        let meta_final = format!("{}/{}", self.table, meta_filename);
        if self.storage.exists(&parquet_final).await? || self.storage.exists(&meta_final).await? {
            return Err(IcefallDBError::Other(
                "commit_merge: fragment rg_id collision; retry".into(),
            ));
        }
        self.storage.write(&parquet_part, parquet_bytes).await?;
        self.storage.write(&meta_part, &meta_json).await?;
        self.storage
            .sync(&format!("{}/_staging/incoming", self.table))
            .await?;
        self.storage.rename(&parquet_part, &parquet_final).await?;
        self.storage.rename(&meta_part, &meta_final).await?;
        Ok(())
    }

    /// Atomically commit a MERGE: matched updates **and** unmatched inserts in a
    /// SINGLE new manifest and ONE manifest-pointer swap.
    ///
    /// This is the atomic equivalent of running [`Writer::commit_update`] for
    /// the matched rows followed by [`Writer::insert_batch`]+[`Writer::commit`]
    /// for the unmatched rows — but it advances the manifest sequence by exactly
    /// **one**, so a crash can never leave a half-applied MERGE (matched updated
    /// but inserts missing, or vice-versa).
    ///
    /// The single new manifest carries BOTH sides:
    ///
    /// * **Matched updates** (when `matched_locs` is non-empty): identical to
    ///   `commit_update` — sort `(matched_rows, matched_locs)` by `row_id`, write
    ///   a PATCH fragment (fresh `fragment_id`, REUSING the original stable
    ///   `row_id`s so `next_row_id` is NOT advanced for these), tombstone the
    ///   matched originals' offsets in their fragments' deletion vectors
    ///   (`deleted_count = cardinality`), and append a `_rowindex/delta`
    ///   relocating each matched `row_id → (patch_fragment_id, new_offset)`.
    /// * **Unmatched inserts** (when `insert_rows` is non-empty): identical to
    ///   the append path — write an INSERT fragment (fresh `fragment_id`) with a
    ///   FRESHLY ALLOCATED contiguous `row_id` range from `next_row_id`,
    ///   advancing `next_row_id` by the insert count.
    /// * **Indexes**: a FULL index rebuild via [`Self::build_indexes_into_manifest`]
    ///   over the new manifest's live rows. This correctly captures the matched
    ///   rows' new values, the inserted rows' new key→row_id entries, and the
    ///   surviving rows in one pass — no incremental delta juggling.
    ///
    /// Every new file (patch `.parquet`/`.meta`, insert `.parquet`/`.meta`, the
    /// new `.del` files, the `_rowindex/delta` `.idx`, and the rebuilt index base
    /// files) is listed in the intent journal AND in `manifest_referenced_files`
    /// (via the row-group / deletion / rowindex / index-generation entries the
    /// manifest carries), so staging cleanup never treats a committed file as an
    /// orphan.
    ///
    /// At least one of `matched_locs` / `insert_rows` must contribute work; a
    /// MERGE with only matched rows (empty `insert_rows`) or only inserts (empty
    /// `matched_locs`) is supported. `set_columns` is accepted for symmetry with
    /// `commit_update` but does not gate index work here (the rebuild is full).
    pub async fn commit_merge(
        &mut self,
        matched_rows: RecordBatch,
        matched_locs: Vec<MatchLoc>,
        set_columns: &[String],
        insert_rows: RecordBatch,
    ) -> Result<CommitDelta> {
        let _ = set_columns; // full rebuild covers all indexes; kept for API symmetry
        let has_matched = !matched_locs.is_empty();
        let has_inserts = insert_rows.num_rows() > 0;

        if has_matched && matched_rows.num_rows() != matched_locs.len() {
            return Err(IcefallDBError::Other(
                format!(
                    "commit_merge: matched_rows ({}) and matched_locs ({}) must have equal length",
                    matched_rows.num_rows(),
                    matched_locs.len()
                )
                .into(),
            ));
        }
        if !has_matched && !has_inserts {
            return Err(IcefallDBError::Other(
                "commit_merge: nothing to commit (no matched rows and no inserts)".into(),
            ));
        }

        // ── Step 1: sort matched rows+locs by row_id (patch-fragment layout) ─
        let (sorted_matched_rows, sorted_matched_locs, sorted_matched_row_ids) = if has_matched {
            let mut order: Vec<usize> = (0..matched_locs.len()).collect();
            order.sort_unstable_by_key(|&i| matched_locs[i].row_id);

            let sorted_locs: Vec<MatchLoc> = order.iter().map(|&i| matched_locs[i]).collect();
            let indices = arrow::array::UInt64Array::from(
                order.iter().map(|&i| i as u64).collect::<Vec<_>>(),
            );
            let cols: Vec<arrow::array::ArrayRef> = matched_rows
                .columns()
                .iter()
                .map(|col| arrow::compute::take(col.as_ref(), &indices, None).map_err(other))
                .collect::<Result<_>>()?;
            let sorted_rows = RecordBatch::try_new(matched_rows.schema(), cols).map_err(other)?;
            let sorted_row_ids: Vec<u64> = sorted_locs.iter().map(|l| l.row_id).collect();
            (Some(sorted_rows), sorted_locs, sorted_row_ids)
        } else {
            (None, Vec::new(), Vec::new())
        };

        // ── Acquire the exclusive writer lock ────────────────────────────────
        let lock_path = format!("{}/_write.lock", self.table);
        let _lock = self
            .storage
            .lock_exclusive(&lock_path, self.lock_timeout)
            .await?;

        // ── Load current manifest ────────────────────────────────────────────
        let (latest_seq, current_manifest) = self.load_current_manifest().await?;

        let referenced_files = Self::manifest_referenced_files(&current_manifest);
        cleanup_staging(
            self.storage.as_ref(),
            &self.table,
            latest_seq,
            &referenced_files,
        )
        .await?;

        let next_seq = latest_seq + 1;
        let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
        if self.storage.exists(&manifest_path).await? {
            return Err(IcefallDBError::SequenceCollision(next_seq));
        }

        ensure_dir(
            self.storage.as_ref(),
            &format!("{}/_staging/incoming", self.table),
        )
        .await?;
        ensure_dir(
            self.storage.as_ref(),
            &format!("{}/_staging/intents", self.table),
        )
        .await?;

        // ── Step 2: allocate fragment ids and (for inserts) a fresh row range ─
        // Matched updates REUSE stable row_ids → next_row_id NOT advanced for
        // them. Inserts get a fresh contiguous range and DO advance next_row_id.
        let mut next_fragment_id = current_manifest.next_fragment_id;
        let mut next_row_id = current_manifest.next_row_id;

        let patch_fragment_id = if has_matched {
            let id = next_fragment_id;
            next_fragment_id += 1;
            Some(id)
        } else {
            None
        };
        let insert_fragment_id = if has_inserts {
            let id = next_fragment_id;
            next_fragment_id += 1;
            Some(id)
        } else {
            None
        };
        let insert_row_id_seg = if has_inserts {
            Some(allocate_range(
                &mut next_row_id,
                insert_rows.num_rows() as u64,
            ))
        } else {
            None
        };

        let mut used_rg_ids: HashSet<String> = HashSet::new();

        // ── Step 3a: encode the PATCH fragment (matched updates) ─────────────
        let patch_files: Option<(String, String, Vec<u8>, RowGroupMeta)> = if has_matched {
            let sorted_rows = sorted_matched_rows.as_ref().expect("matched rows present");
            let patch_rg_id = unique_rg_id(&mut used_rg_ids);

            let mut parquet_bytes: Vec<u8> = Vec::new();
            {
                #[allow(unused_mut)]
                let mut props_builder = parquet::file::properties::WriterProperties::builder()
                    .set_compression(parquet::basic::Compression::ZSTD(
                        parquet::basic::ZstdLevel::try_new(1).expect("valid zstd level"),
                    ));
                #[cfg(feature = "encryption")]
                if let Some(enc) = &self.encryption {
                    let fp = crate::encryption::build_encryption_properties(
                        &enc.keys,
                        enc.plaintext_footer,
                        enc.store_aad_prefix,
                        &enc.encrypted_columns,
                    )?;
                    props_builder = props_builder.with_file_encryption_properties(fp);
                }
                let props = props_builder.build();
                let mut writer = ArrowWriter::try_new(
                    &mut parquet_bytes,
                    self.arrow_schema.clone(),
                    Some(props),
                )
                .map_err(other)?;
                writer.write(sorted_rows).map_err(other)?;
                writer.close().map_err(other)?;
            }

            // Patch row_ids are the original stable IDs (possibly non-contiguous).
            let patch_row_id_seg = crate::rowid::RowIdSegment::Sorted {
                ids: sorted_matched_row_ids.clone(),
            };
            let mut patch_meta = compute_row_group_meta(
                &patch_rg_id,
                self.schema.schema_id,
                sorted_rows,
                &self.schema,
                &parquet_bytes,
                &self.table,
                std::slice::from_ref(&patch_row_id_seg),
            )?;
            self.redact_encrypted_meta(&mut patch_meta, &parquet_bytes)?;
            Some((
                format!("{}.parquet", patch_rg_id),
                format!("{}.meta", patch_rg_id),
                parquet_bytes,
                patch_meta,
            ))
        } else {
            None
        };

        // ── Step 3b: encode the INSERT fragment (unmatched inserts) ──────────
        let insert_files: Option<(String, String, Vec<u8>, RowGroupMeta)> = if has_inserts {
            if !schema_equal_ignoring_metadata(insert_rows.schema().as_ref(), &self.arrow_schema) {
                return Err(IcefallDBError::SchemaMismatch {
                    column: "schema".into(),
                    expected: "match writer schema".into(),
                    path: self.table.clone(),
                });
            }
            let insert_rg_id = unique_rg_id(&mut used_rg_ids);

            let mut parquet_bytes: Vec<u8> = Vec::new();
            {
                #[allow(unused_mut)]
                let mut props_builder = parquet::file::properties::WriterProperties::builder()
                    .set_compression(parquet::basic::Compression::ZSTD(
                        parquet::basic::ZstdLevel::try_new(1).expect("valid zstd level"),
                    ));
                #[cfg(feature = "encryption")]
                if let Some(enc) = &self.encryption {
                    let fp = crate::encryption::build_encryption_properties(
                        &enc.keys,
                        enc.plaintext_footer,
                        enc.store_aad_prefix,
                        &enc.encrypted_columns,
                    )?;
                    props_builder = props_builder.with_file_encryption_properties(fp);
                }
                let props = props_builder.build();
                let mut writer = ArrowWriter::try_new(
                    &mut parquet_bytes,
                    self.arrow_schema.clone(),
                    Some(props),
                )
                .map_err(other)?;
                writer.write(&insert_rows).map_err(other)?;
                writer.close().map_err(other)?;
            }

            let insert_seg = insert_row_id_seg
                .clone()
                .expect("insert row-id segment present");
            let mut insert_meta = compute_row_group_meta(
                &insert_rg_id,
                self.schema.schema_id,
                &insert_rows,
                &self.schema,
                &parquet_bytes,
                &self.table,
                std::slice::from_ref(&insert_seg),
            )?;
            self.redact_encrypted_meta(&mut insert_meta, &parquet_bytes)?;
            Some((
                format!("{}.parquet", insert_rg_id),
                format!("{}.meta", insert_rg_id),
                parquet_bytes,
                insert_meta,
            ))
        } else {
            None
        };

        // ── Step 4: pre-compute deletion-vector work for matched originals ───
        struct DelWork {
            fragment_id: u64,
            rel_del_path: String,
            abs_del_path: String,
            del_bytes: Vec<u8>,
            cardinality: u64,
        }
        let mut del_work: Vec<DelWork> = Vec::new();
        let mut new_del_paths: Vec<String> = Vec::new();
        let mut row_groups = current_manifest.row_groups.clone();

        if has_matched {
            let mut by_fragment: HashMap<u64, Vec<u32>> = HashMap::new();
            for loc in &sorted_matched_locs {
                by_fragment
                    .entry(loc.fragment_id)
                    .or_default()
                    .push(loc.offset);
            }
            for entry in &row_groups {
                let offsets = match by_fragment.get(&entry.fragment_id) {
                    Some(o) if !o.is_empty() => o,
                    _ => continue,
                };
                let mut dv = if let Some(del_path) = &entry.deletes {
                    let bytes = self
                        .storage
                        .read(&format!("{}/{}", self.table, del_path))
                        .await?;
                    crate::DeletionVector::deserialize(&bytes)
                        .map_err(|e| IcefallDBError::Other(Box::new(e)))?
                } else {
                    crate::DeletionVector::default()
                };
                dv.union_offsets(offsets.iter().copied());

                let new_ver = if let Some(cur) = &entry.deletes {
                    cur.rsplit("__v")
                        .next()
                        .and_then(|s| s.strip_suffix(".del"))
                        .and_then(|s| s.parse::<u64>().ok())
                        .map(|v| v + 1)
                        .ok_or_else(|| {
                            IcefallDBError::Other(Box::new(std::io::Error::other(format!(
                                "commit_merge: cannot parse version from .del path {:?}",
                                cur
                            ))))
                        })?
                } else {
                    1
                };
                let rel_del_path =
                    format!("_deletions/rg_{:016x}__v{}.del", entry.fragment_id, new_ver);
                let abs_del_path = format!("{}/{}", self.table, rel_del_path);
                let cardinality = dv.cardinality();
                let del_bytes = dv.serialize();
                new_del_paths.push(rel_del_path.clone());
                del_work.push(DelWork {
                    fragment_id: entry.fragment_id,
                    rel_del_path,
                    abs_del_path,
                    del_bytes,
                    cardinality,
                });
            }
        }

        // ── Step 5: pre-compute the rowindex delta path (matched relocation) ─
        let rel_delta_path = if has_matched {
            Some(format!("_rowindex/delta__v{:09}.idx", next_seq))
        } else {
            None
        };

        // ── Step 6: write the COMPLETE intent ONCE, before any side-effects ──
        // Every file this commit will create is listed so recovery can clean up
        // orphans on any crash path. Index base files are appended later by
        // build_indexes_into_manifest (which rewrites the intent).
        let mut all_intent_files: Vec<String> = Vec::new();
        if let Some((data, meta, _, _)) = &patch_files {
            all_intent_files.push(data.clone());
            all_intent_files.push(meta.clone());
        }
        if let Some((data, meta, _, _)) = &insert_files {
            all_intent_files.push(data.clone());
            all_intent_files.push(meta.clone());
        }
        all_intent_files.extend(new_del_paths.iter().cloned());
        if let Some(p) = &rel_delta_path {
            all_intent_files.push(p.clone());
        }

        let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
        let intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": all_intent_files,
        });
        self.storage
            .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(&intent_path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;

        // ── Side-effecting writes begin here (intent is durable) ─────────────

        // Stage + rename the fragment files. Collision with an existing final
        // filename is fatal (the rg_ids are freshly generated UUIDs).
        if let Some((data, meta, bytes, rg_meta)) = &patch_files {
            self.stage_merge_fragment(data, meta, bytes, rg_meta)
                .await?;
        }
        if let Some((data, meta, bytes, rg_meta)) = &insert_files {
            self.stage_merge_fragment(data, meta, bytes, rg_meta)
                .await?;
        }
        self.storage.sync(&format!("{}/", self.table)).await?;

        // ── Tombstone matched originals: write .del files, update row_groups ──
        if !del_work.is_empty() {
            ensure_dir(self.storage.as_ref(), &format!("{}/_deletions", self.table)).await?;
            for work in &del_work {
                self.storage
                    .write(&work.abs_del_path, &work.del_bytes)
                    .await?;
                self.storage.sync_data(&work.abs_del_path).await?;
            }
            for entry in &mut row_groups {
                if let Some(work) = del_work.iter().find(|w| w.fragment_id == entry.fragment_id) {
                    entry.deletes = Some(work.rel_del_path.clone());
                    entry.deleted_count = work.cardinality;
                }
            }
            self.storage
                .sync(&format!("{}/_deletions", self.table))
                .await?;
        }

        // ── Build the rowindex relocation delta for matched rows ─────────────
        let mut new_rowindex_gen = current_manifest
            .rowindex_generation
            .clone()
            .unwrap_or_default();
        if let (Some(rel_delta_path), Some(patch_fragment_id)) =
            (&rel_delta_path, patch_fragment_id)
        {
            ensure_dir(self.storage.as_ref(), &format!("{}/_rowindex", self.table)).await?;
            let mut delta_segs: Vec<crate::rowindex::AddrSegment> = Vec::new();
            for (patch_offset, &row_id) in sorted_matched_row_ids.iter().enumerate() {
                let patch_offset = patch_offset as u32;
                if let Some(last) = delta_segs.last_mut() {
                    if last.fragment_id == patch_fragment_id
                        && last.start_row_id + u64::from(last.len) == row_id
                        && last.start_offset + last.len == patch_offset
                    {
                        last.len += 1;
                        continue;
                    }
                }
                delta_segs.push(crate::rowindex::AddrSegment {
                    start_row_id: row_id,
                    fragment_id: patch_fragment_id,
                    start_offset: patch_offset,
                    len: 1,
                });
            }
            let delta_bytes = crate::rowindex::encode_idx(&delta_segs);
            let abs_delta_path = format!("{}/{}", self.table, rel_delta_path);
            self.storage.write(&abs_delta_path, &delta_bytes).await?;
            self.storage.sync_data(&abs_delta_path).await?;
            self.storage
                .sync(&format!("{}/_rowindex", self.table))
                .await?;
            new_rowindex_gen.deltas.push(rel_delta_path.clone());
        }

        // ── Append the patch + insert fragments to row_groups ────────────────
        // Note: .agg is not produced on the commit_merge path (only
        // insert_batch produces it); the metadata-aggregate rule falls back for fragments without .agg.
        if let (Some((data, meta, _, _)), Some(frag)) = (&patch_files, patch_fragment_id) {
            row_groups.push(RowGroupEntry {
                data: data.clone(),
                meta: meta.clone(),
                fragment_id: frag,
                deletes: None,
                deleted_count: 0,
                agg: None,
            });
        }
        if let (Some((data, meta, _, _)), Some(frag)) = (&insert_files, insert_fragment_id) {
            row_groups.push(RowGroupEntry {
                data: data.clone(),
                meta: meta.clone(),
                fragment_id: frag,
                deletes: None,
                deleted_count: 0,
                agg: None,
            });
        }

        // ── Build the single new manifest ────────────────────────────────────
        let rowindex_generation =
            if new_rowindex_gen.base.is_none() && new_rowindex_gen.deltas.is_empty() {
                // No relocation occurred (insert-only MERGE) and there was no prior
                // generation: preserve whatever the current manifest carried.
                current_manifest.rowindex_generation.clone()
            } else {
                Some(new_rowindex_gen)
            };

        let mut new_manifest = Manifest {
            format_version: 1,
            sequence: next_seq,
            schema_id: current_manifest.schema_id,
            row_groups,
            row_counts: None,
            partition_values: current_manifest.partition_values.clone(),
            next_row_id, // advanced only by the inserts (move-stable for matched)
            next_fragment_id,
            rowindex_generation,
            // Index generations are fully rebuilt below from the live-row scan.
            index_generations: current_manifest.index_generations.clone(),
            checksum: String::new(),
            ..Default::default()
        };

        // ── Full index rebuild over the new snapshot ─────────────────────────
        // A full rebuild correctly captures matched rows' new values, inserted
        // rows' fresh key→row_id entries, and surviving rows in one pass. It
        // also appends the rebuilt index base files to the commit intent.
        let mut intent_files = self
            .build_indexes_into_manifest(
                &mut new_manifest,
                &current_manifest,
                &all_intent_files,
                &intent_path,
                &txn_id,
                false, // MERGE changes matched rows' values → full rebuild
            )
            .await?;

        // Build the added-fragment PlannedRowGroups (matched patch + insert) once,
        // for both the WAL delta and the non-WAL return.
        let mut added_row_groups = Vec::new();
        if let (Some((data, meta, _, rg_meta)), Some(frag_id)) = (&patch_files, patch_fragment_id) {
            let partition_values = new_manifest
                .partition_values
                .as_ref()
                .and_then(|m| m.get(data).cloned());
            added_row_groups.push(crate::reader::PlannedRowGroup {
                data_path: format!("{}/{}", self.table, data),
                meta_path: format!("{}/{}", self.table, meta),
                meta: rg_meta.clone(),
                partition_values,
                snapshot: new_manifest.sequence,
                fallback: false,
                deletes: None,
                deleted_count: 0,
                fragment_id: frag_id,
                agg_state: None,
            });
        }
        if let (Some((data, meta, _, rg_meta)), Some(frag_id)) = (&insert_files, insert_fragment_id)
        {
            let partition_values = new_manifest
                .partition_values
                .as_ref()
                .and_then(|m| m.get(data).cloned());
            added_row_groups.push(crate::reader::PlannedRowGroup {
                data_path: format!("{}/{}", self.table, data),
                meta_path: format!("{}/{}", self.table, meta),
                meta: rg_meta.clone(),
                partition_values,
                snapshot: new_manifest.sequence,
                fallback: false,
                deletes: None,
                deleted_count: 0,
                fragment_id: frag_id,
                agg_state: None,
            });
        }

        // ── WAL mode: defer the manifest swap; append one record ──────────────
        // The matched-patch and inserted fragments and their sidecars are already
        // written + synced above; record the manifest diff and skip the swap.
        if self.wal_mode {
            let current_ids: std::collections::HashSet<u64> = current_manifest
                .row_groups
                .iter()
                .map(|e| e.fragment_id)
                .collect();
            let mut fragment_deletes: Vec<crate::mutation_wal::FragmentDelete> = Vec::new();
            for entry in &new_manifest.row_groups {
                if !current_ids.contains(&entry.fragment_id) {
                    continue;
                }
                let changed = current_manifest
                    .row_groups
                    .iter()
                    .find(|e| e.fragment_id == entry.fragment_id)
                    .map(|old| old.deletes != entry.deletes)
                    .unwrap_or(false);
                if changed {
                    if let Some(deletes) = &entry.deletes {
                        fragment_deletes.push(crate::mutation_wal::FragmentDelete {
                            fragment_id: entry.fragment_id,
                            deletes: deletes.clone(),
                            deleted_count: entry.deleted_count,
                        });
                    }
                }
            }
            let fragment_adds: Vec<crate::metadata::RowGroupEntry> = new_manifest
                .row_groups
                .iter()
                .filter(|e| !current_ids.contains(&e.fragment_id))
                .cloned()
                .collect();
            let record = crate::mutation_wal::MutationRecord {
                sequence: next_seq,
                base_sequence: latest_seq,
                fragment_deletes,
                fragment_adds,
                index_generations: new_manifest.index_generations.clone(),
                rowindex_generation: new_manifest.rowindex_generation.clone(),
                next_fragment_id: new_manifest.next_fragment_id,
                next_row_id: new_manifest.next_row_id,
                staged_artifacts: Vec::new(),
                checksum: String::new(),
            }
            .sealed();
            crate::mutation_wal::append(self.storage.as_ref(), &self.table, &record).await?;
            let _ = self.storage.delete(&intent_path).await;
            let pending = crate::mutation_wal::read_records(self.storage.as_ref(), &self.table)
                .await?
                .len();
            if pending >= self.wal_checkpoint_threshold {
                crate::mutation_wal::checkpoint_locked(self.storage.as_ref(), &self.table).await?;
            }
            return Ok(
                CommitDelta::new(&current_manifest, &new_manifest, CommitKind::Merge)
                    .with_added_row_groups(added_row_groups),
            );
        }

        // Emit the snapshot checkpoint inside the atomic commit.
        let checkpoint = build_snapshot_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &current_manifest,
            &new_manifest,
        )
        .await?;
        let mut _staged_for_rollback = Vec::new();
        let checkpoint_path = write_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &checkpoint,
            next_seq,
            &intent_path,
            &mut intent_files,
            &mut _staged_for_rollback,
            &txn_id,
            self.schema.schema_id,
        )
        .await?;
        new_manifest.checkpoint = Some(checkpoint_path);

        self.finalize_manifest(&mut new_manifest, next_seq).await?;

        // ── Single atomic publish: manifest write + pointer swap ─────────────
        let manifest_data = serde_json::to_vec(&new_manifest)?;
        let manifest_tmp_path = format!("{}.tmp", manifest_path);
        self.storage
            .write(&manifest_tmp_path, &manifest_data)
            .await?;
        self.storage.sync_data(&manifest_tmp_path).await?;

        if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
            self.storage
                .write(&manifest_tmp_path, &manifest_data)
                .await?;
            self.storage.sync_data(&manifest_tmp_path).await?;
            if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
                let _ = self.storage.delete(&intent_path).await;
                return Err(IcefallDBError::ChecksumMismatch {
                    path: manifest_tmp_path,
                });
            }
        }

        // Publish: rename then one dir fsync. Content is durable (sync_data
        // above); the pre-rename dir fsync for the disposable tmp added nothing.
        self.storage
            .rename(&manifest_tmp_path, &manifest_path)
            .await?;
        self.storage
            .sync(&format!("{}/_manifests", self.table))
            .await?;

        // Atomic durability point: update the manifest pointer.
        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer = serde_json::json!({"latest": next_seq});
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;
        self.storage.sync(&format!("{}/", self.table)).await?;

        // Best-effort cleanup of the intent file.
        let _ = self.storage.delete(&intent_path).await;
        // No fsync of the intents dir: a leftover intent after a crash here is
        // reclaimed by cleanup_staging on the next open, so the deletion need not
        // be durable (matches the MERGE commit path).

        Ok(
            CommitDelta::new(&current_manifest, &new_manifest, CommitKind::Merge)
                .with_added_row_groups(added_row_groups),
        )
    }

    /// Rebuild secondary indexes for the new snapshot **inside** the commit,
    /// before the manifest is serialized and the pointer is swapped.
    ///
    /// `manifest` must already have its final `row_groups` and row-id/fragment-id
    /// assignments; the row group data/`.meta` files it references must already be
    /// durable at their final table-root paths. Each index is rebuilt from those
    /// fragments, written as a versioned immutable base file, and recorded in
    /// `manifest.index_generations` (in place).
    ///
    /// Because the index files become part of this commit, they are appended to
    /// the commit intent journal alongside `data_meta_files` (the row group
    /// data/meta filenames). Recovery uses the intent + the committed manifest to
    /// decide which staged files are durable; listing the index files there keeps
    /// them from being deleted as orphans if a crash occurs after they are written
    /// but before the pointer swap, and — once the manifest references them via
    /// `index_generations` — keeps a later commit from cleaning them up.
    ///
    /// The manifest itself is **not** written here: the single durability point
    /// remains the one manifest write + pointer swap in the caller, and that
    /// manifest already contains the populated `index_generations`.
    ///
    /// Returns the complete file list (`data_meta_files` plus any rebuilt index
    /// files) that was written to the intent, so callers can append further
    /// commit artifacts such as the snapshot checkpoint.
    async fn build_indexes_into_manifest(
        &self,
        manifest: &mut Manifest,
        current_manifest: &Manifest,
        data_meta_files: &[String],
        intent_path: &str,
        txn_id: &str,
        incremental_append: bool,
    ) -> Result<Vec<String>> {
        let index_paths = if incremental_append {
            // Pure append: scan only the fragments this commit added and write an
            // adds-only index delta, instead of re-serializing each index over the
            // whole table. New fragments are those whose data path is not in the
            // prior snapshot.
            let prior: std::collections::HashSet<&str> = current_manifest
                .row_groups
                .iter()
                .map(|rg| rg.data.as_str())
                .collect();
            let new_fragments: Vec<RowGroupEntry> = manifest
                .row_groups
                .iter()
                .filter(|rg| !prior.contains(rg.data.as_str()))
                .cloned()
                .collect();
            IndexMaintainer::maintain_on_insert(
                self.storage.clone(),
                &self.table,
                manifest,
                &current_manifest.index_generations,
                &new_fragments,
                manifest.sequence,
            )
            .await?
        } else {
            IndexMaintainer::maintain(self.storage.clone(), &self.table, manifest).await?
        };

        // Rewrite the intent so its `files` list covers both the row group
        // data/meta files and the freshly written index base files. The index
        // files are already durable (each `save_versioned` syncs the index dir),
        // so this only needs to make recovery aware of them.
        let mut files: Vec<String> = data_meta_files.to_vec();
        files.extend(index_paths);
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": &files,
        });
        // In-place rewrite of the existing intent (durable directory entry from
        // the initial intent fsync; stable inode), so `sync_data` for the new
        // content is enough — no second `_staging/intents` directory fsync.
        self.storage
            .write(intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(intent_path).await?;
        Ok(files)
    }

    /// Build a [`PlannedRowGroup`] for a row-group entry that has just been
    /// written to its final table-root path. This may read the freshly-written
    /// `.meta` and `.agg` sidecars; that read happens during the commit, not
    /// during the provider's incremental refresh.
    async fn planned_row_group_for_entry(
        &self,
        entry: &RowGroupEntry,
        snapshot: u64,
        partition_values: Option<std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<crate::reader::PlannedRowGroup> {
        use crate::agg_cache::deserialize_agg_state;

        let data_path = format!("{}/{}", self.table, entry.data);
        let meta_path = format!("{}/{}", self.table, entry.meta);

        let meta_bytes = self.storage.read(&meta_path).await?;
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
        if !meta.verify_meta_checksum()? {
            return Err(IcefallDBError::RowGroupChecksumMismatch {
                path: meta_path.clone(),
            });
        }

        let agg_state = if let Some(agg_rel) = &entry.agg {
            let agg_path = format!("{}/{}", self.table, agg_rel);
            let bytes = self.storage.read(&agg_path).await?;
            Some(std::sync::Arc::new(deserialize_agg_state(&bytes)?))
        } else {
            None
        };

        Ok(crate::reader::PlannedRowGroup {
            data_path,
            meta_path,
            meta,
            partition_values,
            snapshot,
            fallback: false,
            deletes: entry
                .deletes
                .as_ref()
                .map(|p| format!("{}/{}", self.table, p)),
            deleted_count: entry.deleted_count,
            fragment_id: entry.fragment_id,
            agg_state,
        })
    }

    /// Collect the set of files referenced by `manifest` that must be preserved
    /// during staging cleanup: every row group's data/meta files plus every
    /// secondary-index base file recorded in `index_generations`. Index base
    /// files are committed durable state, so a leftover intent that happens to
    /// list them must not cause them to be deleted.
    ///
    /// All paths stored here are BARE RELATIVE paths (e.g. `_deletions/rg_…__v1.del`
    /// or `rg_….parquet`); the `table/` prefix is NOT included. `cleanup_staging`
    /// prepends `table/` when deleting, and intent `files` arrays use the same bare
    /// relative form — so the `referenced_files.contains(filename)` check is
    /// path-key consistent.
    pub(crate) fn manifest_referenced_files(manifest: &Manifest) -> HashSet<String> {
        let mut referenced: HashSet<String> = manifest
            .row_groups
            .iter()
            .flat_map(|e| [e.data.clone(), e.meta.clone()])
            .collect();
        // Deletion-vector and aggregate-sidecar files are committed durable
        // state: include them so a subsequent commit's cleanup_staging never
        // treats them as orphans.
        for e in &manifest.row_groups {
            if let Some(del) = &e.deletes {
                referenced.insert(del.clone());
            }
            if let Some(agg) = &e.agg {
                referenced.insert(agg.clone());
            }
        }
        for index_ref in manifest.index_generations.values() {
            if let Some(base) = &index_ref.base {
                referenced.insert(base.clone());
                // The derived binary index sibling (`.idx`) is retained
                // in lockstep with its JSON base so GC keeps the current one and
                // removes superseded ones. Referencing it when it does not exist
                // (non-local storage, or a skipped write) is harmless.
                if let Some(stem) = base.strip_suffix(".json") {
                    referenced.insert(format!("{stem}.idx"));
                    // The derived learned-model sibling (`.model`).
                    referenced.insert(format!("{stem}.model"));
                }
            }
            for delta in &index_ref.deltas {
                referenced.insert(delta.clone());
            }
        }
        // Row-index files (base + deltas) are committed durable state: protect
        // them from cleanup_staging treating them as orphans.
        if let Some(ri) = &manifest.rowindex_generation {
            if let Some(base) = &ri.base {
                referenced.insert(base.clone());
            }
            for delta in &ri.deltas {
                referenced.insert(delta.clone());
            }
        }
        // Snapshot checkpoint files are committed durable state.
        if let Some(checkpoint) = &manifest.checkpoint {
            referenced.insert(checkpoint.clone());
            // The derived zero-copy archive sibling is retained in
            // lockstep with the JSON checkpoint so GC keeps the current one and
            // removes superseded ones. Harmless when it does not exist.
            referenced.insert(crate::metadata::SnapshotCheckpoint::archive_filename(
                manifest.sequence,
            ));
        }
        referenced
    }

    /// Executes the body of a commit up to and including the manifest pointer
    /// update. Populates `rollback` progressively so that `commit()` can clean
    /// up if any step fails before the pointer is durable.
    ///
    /// When `replace` is `true`, the new manifest drops all existing row groups,
    /// committing only the buffered batches (which may be empty).
    ///
    /// Returns the committed manifest together with fully populated
    /// [`PlannedRowGroup`]s for every fragment added by this commit.
    async fn try_commit(
        &mut self,
        rollback: &mut RollbackInfo,
        replace: bool,
    ) -> Result<(Manifest, Vec<crate::reader::PlannedRowGroup>)> {
        // Re-read the current state while holding the lock and verify integrity.
        let (latest_seq, current_manifest) = self.load_current_manifest().await?;
        let existing_row_groups = current_manifest.row_groups.clone();

        // Re-verify the schema pointer. It could have changed between
        // `Writer::new` and `commit` (for example by another process), and
        // committing against the wrong schema would corrupt the table.
        let schema_pointer_path = format!("{}/_schema.json", self.table);
        let schema_pointer_data = self.storage.read(&schema_pointer_path).await?;
        let schema_pointer: serde_json::Value = serde_json::from_slice(&schema_pointer_data)
            .map_err(|_| IcefallDBError::InvalidSchemaPointer {
                path: schema_pointer_path.clone(),
            })?;
        let pointer_id = schema_pointer
            .get("latest")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| IcefallDBError::InvalidSchemaPointer {
                path: schema_pointer_path.clone(),
            })?;
        if pointer_id != self.schema.schema_id {
            return Err(IcefallDBError::SchemaMismatch {
                column: "schema_id".into(),
                expected: self.schema.schema_id.to_string(),
                path: schema_pointer_path.clone(),
            });
        }

        // Because the lock is held, any leftover intents or .part files must be
        // from crashed writers and can be safely recovered. Pass the set of
        // files referenced by the current manifest so recovery can tell whether
        // an intent's files were committed before a post-pointer-update crash.
        // This includes secondary-index base files recorded in the current
        // manifest's `index_generations`, so a stale intent that lists them does
        // not delete committed index state.
        let referenced_files = Self::manifest_referenced_files(&current_manifest);
        cleanup_staging(
            self.storage.as_ref(),
            &self.table,
            latest_seq,
            &referenced_files,
        )
        .await?;

        let next_seq = latest_seq + 1;
        let manifest_path = format!("{}/{}", self.table, Manifest::filename(next_seq));
        if self.storage.exists(&manifest_path).await? {
            return Err(IcefallDBError::SequenceCollision(next_seq));
        }

        // Plan row groups from buffered data and generate final filenames.
        let planned = self.plan_row_groups().await?;
        let added_row_counts: Vec<usize> = planned.iter().map(|p| p.batch.num_rows()).collect();

        // The intent records the final table-root filenames for the row groups
        // that will be committed, including the `.agg` sidecar.  Recovery uses
        // the current manifest to decide whether these files are committed
        // (referenced) or abandoned (unreferenced), so it is safe to list
        // final filenames here.
        // Encrypted tables write no `.agg` sidecar, so it must not be listed in
        // the commit intent (recovery would otherwise reference a file that is
        // never staged).
        let with_agg = !self.is_encrypted();
        let intent_files: Vec<String> = planned
            .iter()
            .flat_map(|p| {
                let mut files = vec![p.data_filename.clone(), p.meta_filename.clone()];
                if with_agg {
                    files.push(format!("{}.agg", p.rg_id));
                }
                files
            })
            .collect();

        rollback.sequence = next_seq;

        // Write intent before staging any data files.
        let txn_id = format!("txn_{}", uuid::Uuid::new_v4());
        let intent_path = format!("{}/_staging/intents/{}.json", self.table, txn_id);
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": self.schema.schema_id,
            "files": intent_files,
        });
        self.storage
            .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
            .await?;
        self.storage.sync_data(&intent_path).await?;
        self.storage
            .sync(&format!("{}/_staging/intents", self.table))
            .await?;

        rollback.intent_path = intent_path.clone();

        // Allocate stable row IDs and fragment IDs for each new fragment in
        // write order, reading the high-water marks from the current manifest
        // (0 for a fresh table).  The bumped values will be written into the
        // new manifest so successive appends produce disjoint id ranges.
        let mut next_row_id = current_manifest.next_row_id;
        let mut next_fragment_id = current_manifest.next_fragment_id;
        let mut fragment_ids: Vec<u64> = Vec::with_capacity(planned.len());
        let mut allocated_row_ids: Vec<RowIdSegment> = Vec::with_capacity(planned.len());
        for planned_rg in &planned {
            let row_count = planned_rg.batch.num_rows() as u64;
            let seg = allocate_range(&mut next_row_id, row_count);
            allocated_row_ids.push(seg);
            let frag_id = next_fragment_id;
            next_fragment_id += 1;
            fragment_ids.push(frag_id);
        }

        // Stage files with proper fsync ordering: write .part files (parquet,
        // meta, and agg sidecar), then sync the incoming directory once.
        for ((planned_rg, row_ids), &frag_id) in planned
            .iter()
            .zip(allocated_row_ids.iter())
            .zip(fragment_ids.iter())
        {
            self.write_row_group_part(
                planned_rg,
                &planned_rg.rg_id,
                std::slice::from_ref(row_ids),
                frag_id,
            )
            .await?;
        }
        self.storage
            .sync(&format!("{}/_staging/incoming", self.table))
            .await?;

        // Verify each row group's checksum while it is still a .part file,
        // retry once on mismatch, and only rename to the final table-root
        // filename after verification succeeds. If the target final filename
        // already exists, generate a new row-group id and retry to avoid
        // silently overwriting existing data.
        let mut used_rg_ids: HashSet<String> = planned.iter().map(|p| p.rg_id.clone()).collect();
        // Renamed (data, meta, agg) filenames — parallel to `planned`. `agg` is
        // `None` when the row group has no `.agg` sidecar (e.g. encrypted tables).
        let mut actual_filenames: Vec<(String, String, Option<String>)> =
            Vec::with_capacity(planned.len());
        for ((planned_rg, row_ids), &frag_id) in planned
            .iter()
            .zip(allocated_row_ids.iter())
            .zip(fragment_ids.iter())
        {
            actual_filenames.push(
                self.rename_row_group_with_collision_retry(
                    planned_rg,
                    std::slice::from_ref(row_ids),
                    frag_id,
                    &mut used_rg_ids,
                    rollback,
                )
                .await?,
            );
        }
        // A single sync of the table root is enough to make the renames durable.
        self.storage.sync(&format!("{}/", self.table)).await?;

        // If any row group had to pick a different final filename due to a
        // collision, rewrite the intent so it accurately lists the files that
        // were staged for this commit — including the `.agg` sidecar.
        let actual_files: Vec<String> = actual_filenames
            .iter()
            .flat_map(|(data, meta, agg)| {
                let mut files = vec![data.clone(), meta.clone()];
                if let Some(a) = agg {
                    files.push(a.clone());
                }
                files
            })
            .collect();
        if actual_files != intent_files {
            let intent = serde_json::json!({
                "txn_id": txn_id,
                "started_at": chrono::Utc::now().to_rfc3339(),
                "schema_id": self.schema.schema_id,
                "files": actual_files,
            });
            // In-place rewrite of the existing intent (durable directory entry
            // from the initial intent fsync; stable inode), so `sync_data` is
            // enough — no second `_staging/intents` directory fsync.
            self.storage
                .write(&intent_path, serde_json::to_vec(&intent)?.as_slice())
                .await?;
            self.storage.sync_data(&intent_path).await?;
        }

        // Build new manifest. For a normal commit the existing row groups are
        // retained and the new ones are appended; for a replace commit the new
        // manifest contains only the newly written row groups.
        let mut row_groups = if replace {
            Vec::new()
        } else {
            existing_row_groups
        };
        row_groups.extend(
            actual_filenames
                .iter()
                .zip(fragment_ids.iter())
                .map(|((data, meta, agg), &frag_id)| RowGroupEntry {
                    data: data.clone(),
                    meta: meta.clone(),
                    fragment_id: frag_id,
                    agg: agg.clone(),
                    ..Default::default()
                })
                .collect::<Vec<_>>(),
        );

        // Compute partition values when the schema declares partition columns.
        // Carry forward any partition values already recorded for existing row
        // groups so incremental commits do not drop metadata needed for pruning.
        let partition_values = if let Some(partition_by) = &self.schema.partition_by {
            if partition_by.is_empty() {
                HashMap::new()
            } else {
                let mut values = current_manifest
                    .partition_values
                    .clone()
                    .unwrap_or_default();
                for (planned_rg, (actual_data, _actual_meta, _actual_agg)) in
                    planned.iter().zip(&actual_filenames)
                {
                    if let Some(pv) =
                        compute_partition_values(&planned_rg.batch, partition_by, &self.table)?
                    {
                        values.insert(actual_data.clone(), pv);
                    }
                }
                values
            }
        } else {
            HashMap::new()
        };
        let partition_values = if partition_values.is_empty() {
            None
        } else {
            Some(partition_values)
        };

        let row_counts = if replace {
            collect_row_counts(
                self.storage.as_ref(),
                &self.table,
                &[],
                &current_manifest,
                &added_row_counts,
            )
            .await
        } else {
            collect_row_counts(
                self.storage.as_ref(),
                &self.table,
                &current_manifest.row_groups,
                &current_manifest,
                &added_row_counts,
            )
            .await
        };

        let mut manifest = Manifest {
            format_version: 1,
            sequence: next_seq,
            schema_id: self.schema.schema_id,
            row_groups,
            row_counts,
            partition_values,
            // Carry the bumped high-water marks forward so successive appends
            // produce disjoint row-id ranges and monotonically increasing
            // fragment IDs.
            next_row_id,
            next_fragment_id,
            checksum: String::new(),
            ..Default::default()
        };

        // Rebuild secondary indexes for this snapshot and populate
        // `manifest.index_generations` BEFORE the manifest is serialized and the
        // pointer is swapped. The manifest is then written exactly once, already
        // containing the index generations; there is no post-commit overwrite of
        // the committed (immutable) manifest. The index base files are appended
        // to the commit intent so recovery treats them as durable.
        let mut intent_files = self
            .build_indexes_into_manifest(
                &mut manifest,
                &current_manifest,
                &actual_files,
                &intent_path,
                &txn_id,
                // A `replace` drops all prior fragments, so the carried-forward
                // base would retain stale entries — force a full rebuild there.
                // A normal commit is a pure append → incremental adds-only delta.
                !replace,
            )
            .await?;

        // Emit the snapshot checkpoint inside the atomic commit.
        let checkpoint = build_snapshot_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &current_manifest,
            &manifest,
        )
        .await?;
        let checkpoint_path = write_checkpoint(
            self.storage.as_ref(),
            &self.table,
            &checkpoint,
            next_seq,
            &intent_path,
            &mut intent_files,
            &mut rollback.staged_files,
            &txn_id,
            self.schema.schema_id,
        )
        .await?;
        manifest.checkpoint = Some(checkpoint_path);

        self.finalize_manifest(&mut manifest, next_seq).await?;

        // Write the manifest snapshot to a .tmp file, verify its checksum while
        // it is still .tmp, retry once on mismatch, and only rename to the
        // final manifest path after verification succeeds.
        let manifest_data = serde_json::to_vec(&manifest)?;
        let manifest_tmp_path = format!("{}.tmp", manifest_path);
        self.storage
            .write(&manifest_tmp_path, &manifest_data)
            .await?;
        // Content is durable via `sync_data`; a pre-rename `_manifests` dir fsync
        // for this disposable tmp adds nothing for crash recovery (never
        // referenced — the pointer only swaps after the post-rename fsync below),
        // matching the other commit paths' publish pattern.
        self.storage.sync_data(&manifest_tmp_path).await?;

        if !self.verify_manifest_checksum(&manifest_tmp_path).await? {
            self.storage
                .write(&manifest_tmp_path, &manifest_data)
                .await?;
            self.storage.sync_data(&manifest_tmp_path).await?;
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

        // Build fully populated PlannedRowGroups for the fragments added by this
        // commit. This reads the just-written `.meta` (and `.agg` if present)
        // sidecars; doing it here keeps the provider's incremental refresh path
        // from needing any sidecar reads. This MUST happen before the manifest
        // pointer is swapped: if these reads fail we are still safely inside the
        // pre-durability window and rollback will clean up the staged files.
        let new_data_files: HashSet<String> =
            actual_filenames.iter().map(|(d, _, _)| d.clone()).collect();
        let mut added_row_groups = Vec::new();
        for entry in manifest
            .row_groups
            .iter()
            .filter(|e| new_data_files.contains(&e.data))
        {
            let partition_values = manifest
                .partition_values
                .as_ref()
                .and_then(|m| m.get(&entry.data).cloned());
            added_row_groups.push(
                self.planned_row_group_for_entry(entry, manifest.sequence, partition_values)
                    .await?,
            );
        }

        // Update the manifest pointer durably.
        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer = serde_json::json!({"latest": next_seq});
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;

        // The pointer rename is the durability point, but the rename itself is
        // not guaranteed to be durable until the table root directory is synced.
        // Require the sync to succeed. If it fails the commit may already be
        // durable, so mark the pointer as updated so rollback does not destroy
        // committed state and report the error to the caller.
        if let Err(e) = self.storage.sync(&format!("{}/", self.table)).await {
            rollback.pointer_updated = true;
            return Err(e);
        }

        rollback.pointer_updated = true;
        Ok((manifest, added_row_groups))
    }

    /// Clean up artifacts from a failed commit. Errors are ignored because the
    /// commit has already failed and the lock is still held.
    ///
    /// If the manifest pointer has already been updated, the commit is durable
    /// and rollback must only remove the intent file. Deleting the manifest
    /// snapshot or data files would destroy committed state.
    async fn rollback_commit(
        &self,
        intent_path: &str,
        staged_files: &[String],
        sequence: u64,
        pointer_updated: bool,
    ) {
        if pointer_updated {
            if !intent_path.is_empty() {
                let _ = self.storage.delete(intent_path).await;
            }
            return;
        }

        // Delete staged final files first so they are cleaned up even if the
        // intent deletion fails. Each deletion is best-effort and independent.
        for file in staged_files {
            let _ = self
                .storage
                .delete(&format!("{}/{}", self.table, file))
                .await;
        }
        if sequence > 0 {
            let _ = self
                .storage
                .delete(&format!("{}/{}", self.table, Manifest::filename(sequence)))
                .await;
        }
        if !intent_path.is_empty() {
            let _ = self.storage.delete(intent_path).await;
        }
    }

    /// Split the buffered batches into planned row groups, assigning final
    /// filenames without yet touching storage.
    ///
    /// Every emitted row group respects both `row_group_target_rows` and
    /// `row_group_target_bytes`. When the schema declares partition columns,
    /// rows are grouped by partition value first so each emitted row group is
    /// partition-homogeneous. The buffer is borrowed, not consumed; it is only
    /// cleared by [`Writer::commit`] once the commit is durable.
    async fn plan_row_groups(&self) -> Result<Vec<PlannedRowGroup>> {
        let batches = &self.buffer;
        let mut used_rg_ids: HashSet<String> = HashSet::new();
        if let Some(partition_by) = &self.schema.partition_by {
            if !partition_by.is_empty() {
                return self
                    .plan_partitioned_row_groups(batches, partition_by, &mut used_rg_ids)
                    .await;
            }
        }
        self.slice_concatenated_batches(batches, &mut used_rg_ids)
            .await
    }

    /// Group the buffered rows by partition value and slice each group into
    /// row groups that respect the configured targets.
    async fn plan_partitioned_row_groups(
        &self,
        batches: &[RecordBatch],
        partition_by: &[String],
        used_rg_ids: &mut HashSet<String>,
    ) -> Result<Vec<PlannedRowGroup>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        let combined =
            arrow::compute::concat_batches(&self.arrow_schema, batches).map_err(other)?;
        let total_rows = combined.num_rows();
        if total_rows == 0 {
            return Ok(Vec::new());
        }

        validate_partition_types(&combined, partition_by, &self.table)?;

        // Group row indices by their partition tuple. A JSON-array key lets
        // NULL partition values form their own group naturally.
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for row in 0..total_rows {
            let key = partition_key_for_row(&combined, partition_by, row)?;
            groups.entry(key).or_default().push(row);
        }

        let mut planned = Vec::new();
        for (_, indices) in groups {
            let taken = take_batch_by_indices(&combined, &indices).map_err(other)?;
            let mut group_planned = self.slice_batch_into_row_groups(&taken, used_rg_ids)?;
            planned.append(&mut group_planned);
        }

        Ok(planned)
    }

    /// Concatenate the provided batches and slice them into row groups that
    /// respect both the row and byte targets.
    async fn slice_concatenated_batches(
        &self,
        batches: &[RecordBatch],
        used_rg_ids: &mut HashSet<String>,
    ) -> Result<Vec<PlannedRowGroup>> {
        if batches.is_empty() {
            return Ok(Vec::new());
        }

        let combined =
            arrow::compute::concat_batches(&self.arrow_schema, batches).map_err(other)?;
        let total_rows = combined.num_rows();
        if total_rows == 0 {
            return Ok(Vec::new());
        }

        self.slice_batch_into_row_groups(&combined, used_rg_ids)
    }

    /// Slice a single batch into row groups that respect both the row and byte
    /// targets.
    fn slice_batch_into_row_groups(
        &self,
        batch: &RecordBatch,
        used_rg_ids: &mut HashSet<String>,
    ) -> Result<Vec<PlannedRowGroup>> {
        let total_rows = batch.num_rows();
        if total_rows == 0 {
            return Ok(Vec::new());
        }
        let total_bytes = batch.get_array_memory_size();

        // Use the full batch size to compute a per-row average.
        // `get_array_memory_size()` on a slice returns the parent buffer size,
        // so estimating from the total gives a correct per-row byte cost.
        let bytes_per_row = total_bytes / total_rows;
        if bytes_per_row == 0 {
            // Degenerate case: rows report zero bytes. Fall back to the row
            // target only so planning can still make progress.
            let target_rows = self.schema.row_group_target_rows.max(1);
            let mut planned = Vec::new();
            let mut start = 0;
            while start < total_rows {
                let len = target_rows.min(total_rows - start);
                let slice = slice_batch(batch, start, len)?;
                let rg_id = unique_rg_id(used_rg_ids);
                planned.push(PlannedRowGroup {
                    rg_id: rg_id.clone(),
                    data_filename: format!("{}.parquet", rg_id),
                    meta_filename: format!("{}.meta", rg_id),
                    batch: slice,
                });
                start += len;
            }
            return Ok(planned);
        }

        let byte_target_rows = (self.schema.row_group_target_bytes / bytes_per_row).max(1);
        let chunk_rows = self
            .schema
            .row_group_target_rows
            .min(byte_target_rows)
            .max(1);
        let schema = batch.schema();

        let mut planned = Vec::new();
        let mut start = 0;
        while start < total_rows {
            let len = chunk_rows.min(total_rows - start);
            let arrays: Vec<ArrayRef> = batch
                .columns()
                .iter()
                .map(|col| col.slice(start, len))
                .collect();
            let slice = RecordBatch::try_new(schema.clone(), arrays).map_err(other)?;
            let rg_id = unique_rg_id(used_rg_ids);
            planned.push(PlannedRowGroup {
                rg_id: rg_id.clone(),
                data_filename: format!("{}.parquet", rg_id),
                meta_filename: format!("{}.meta", rg_id),
                batch: slice,
            });
            start += len;
        }

        Ok(planned)
    }

    /// Write a planned row group's data, metadata, and aggregate-state as
    /// `.part` files in the incoming staging area.
    ///
    /// `row_ids` is the pre-allocated row-id segment for this fragment.  Pass
    /// an empty slice when the caller does not yet assign row IDs (e.g. the
    /// fast-path parquet insert, which has its own commit path).
    ///
    /// `fragment_id` is the stable fragment identifier used inside the `.agg`
    /// sidecar.  Pass `0` on paths that do not compute aggregates.
    async fn write_row_group_part(
        &self,
        planned: &PlannedRowGroup,
        rg_id: &str,
        row_ids: &[RowIdSegment],
        fragment_id: u64,
    ) -> Result<()> {
        let parquet_part_path = format!("{}/_staging/incoming/{}.parquet.part", self.table, rg_id);
        let meta_part_path = format!("{}/_staging/incoming/{}.meta.part", self.table, rg_id);

        let mut parquet_bytes = Vec::new();
        {
            #[allow(unused_mut)]
            let mut props_builder = parquet::file::properties::WriterProperties::builder()
                .set_compression(parquet::basic::Compression::ZSTD(
                    parquet::basic::ZstdLevel::try_new(1).expect("valid zstd level"),
                ));
            #[cfg(feature = "encryption")]
            if let Some(enc) = &self.encryption {
                let fp = build_encryption_properties(
                    &enc.keys,
                    enc.plaintext_footer,
                    enc.store_aad_prefix,
                    &enc.encrypted_columns,
                )?;
                props_builder = props_builder.with_file_encryption_properties(fp);
            }
            let props = props_builder.build();
            let mut writer =
                ArrowWriter::try_new(&mut parquet_bytes, self.arrow_schema.clone(), Some(props))
                    .map_err(other)?;
            writer.write(&planned.batch).map_err(other)?;
            writer.close().map_err(other)?;
        }
        self.storage
            .write(&parquet_part_path, &parquet_bytes)
            .await?;

        #[cfg_attr(not(feature = "encryption"), allow(unused_mut))]
        let mut meta = compute_row_group_meta(
            rg_id,
            self.schema.schema_id,
            &planned.batch,
            &self.schema,
            &parquet_bytes,
            &self.table,
            row_ids,
        )?;
        self.redact_encrypted_meta(&mut meta, &parquet_bytes)?;
        let meta_json = serde_json::to_vec(&meta)?;
        self.storage.write(&meta_part_path, &meta_json).await?;

        // Compute additive aggregate partials from the Arrow batch and write a
        // `.agg.part` staging file.  The content_hash is the data checksum
        // just computed — permanently valid because fragments are write-once.
        // Pass the first declared group key (if any) so grouped partials are
        // computed at write time.
        //
        // Encrypted tables get no `.agg` sidecar: it stores plaintext
        // SUM/SUMSQ/MIN/MAX and group partials that would leak the encrypted
        // column values. Readers fall back to scanning fragments that lack a
        // `.agg` (and encrypted reads scan through the decrypting path anyway).
        if !self.is_encrypted() {
            let key_col = self
                .schema
                .agg_group_keys
                .as_deref()
                .and_then(|v| v.first())
                .map(|s| s.as_str());
            let agg_state =
                compute_agg_state_with_key(fragment_id, meta.checksum, &planned.batch, key_col)?;
            let agg_bytes = serialize_agg_state(&agg_state)?;
            let agg_part_path = format!("{}/_staging/incoming/{}.agg.part", self.table, rg_id);
            self.storage.write(&agg_part_path, &agg_bytes).await?;
        }

        // Per-file fsyncs are batched: the caller syncs the incoming directory
        // once after all row groups have been staged.

        Ok(())
    }

    /// Verify, then rename a staged row group to its final table-root filename.
    ///
    /// If the target final filename already exists, a new row-group id is
    /// generated and the staged files are rewritten with that id (up to three
    /// attempts). This prevents a UUID collision or leftover file from being
    /// silently overwritten.
    ///
    /// `row_ids` is forwarded to [`write_row_group_part`] on retries so the
    /// metadata files rewritten with a new id still carry the correct row-ID
    /// segment.
    ///
    /// Returns `(data_filename, meta_filename, agg_filename)`. `agg_filename` is
    /// `None` when no `.agg` sidecar was staged (e.g. encrypted tables, which
    /// suppress the plaintext aggregate sidecar).
    async fn rename_row_group_with_collision_retry(
        &self,
        planned: &PlannedRowGroup,
        row_ids: &[RowIdSegment],
        fragment_id: u64,
        used_rg_ids: &mut HashSet<String>,
        rollback: &mut RollbackInfo,
    ) -> Result<(String, String, Option<String>)> {
        let mut rg_id = planned.rg_id.clone();
        let mut data_filename = planned.data_filename.clone();
        let mut meta_filename = planned.meta_filename.clone();

        for attempt in 0..3 {
            let parquet_part = format!("{}/_staging/incoming/{}.parquet.part", self.table, rg_id);
            let meta_part = format!("{}/_staging/incoming/{}.meta.part", self.table, rg_id);
            let agg_part = format!("{}/_staging/incoming/{}.agg.part", self.table, rg_id);
            let parquet_final = format!("{}/{}", self.table, data_filename);
            let meta_final = format!("{}/{}", self.table, meta_filename);

            // On a retry the part files were written with a previous id, so
            // rewrite them using the current candidate id.
            if attempt > 0 {
                self.write_row_group_part(planned, &rg_id, row_ids, fragment_id)
                    .await?;
            }

            if !self
                .verify_row_group_checksum(&parquet_part, &meta_part)
                .await?
            {
                self.write_row_group_part(planned, &rg_id, row_ids, fragment_id)
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
                // The current `.part` files were written for an id that cannot
                // be used; delete them before picking a new id so they are not
                // left behind as abandoned staging artifacts.
                let _ = self.storage.delete(&parquet_part).await;
                let _ = self.storage.delete(&meta_part).await;
                let _ = self.storage.delete(&agg_part).await;
                rg_id = unique_rg_id(used_rg_ids);
                data_filename = format!("{}.parquet", rg_id);
                meta_filename = format!("{}.meta", rg_id);
                continue;
            }

            self.storage.rename(&parquet_part, &parquet_final).await?;
            self.storage.rename(&meta_part, &meta_final).await?;
            rollback.staged_files.push(data_filename.clone());
            rollback.staged_files.push(meta_filename.clone());
            // The `.agg` sidecar is optional: encrypted tables suppress it. Only
            // promote it when it was actually staged.
            let agg_filename = if self.storage.exists(&agg_part).await? {
                let agg_filename = format!("{}.agg", rg_id);
                let agg_final = format!("{}/{}", self.table, agg_filename);
                self.storage.rename(&agg_part, &agg_final).await?;
                rollback.staged_files.push(agg_filename.clone());
                Some(agg_filename)
            } else {
                None
            };
            return Ok((data_filename, meta_filename, agg_filename));
        }

        Err(IcefallDBError::Other(Box::new(std::io::Error::other(
            "failed to find a unique row group id after retries",
        ))))
    }

    /// Read a row group (either `.part` staged files or final files) and verify
    /// that its Parquet bytes match the checksum recorded in its metadata file
    /// and that the metadata's own checksum is intact.
    async fn verify_row_group_checksum(&self, parquet_path: &str, meta_path: &str) -> Result<bool> {
        let parquet_bytes = self.storage.read(parquet_path).await?;
        let meta_bytes = self.storage.read(meta_path).await?;
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
        let data_ok = meta.verify_against_data(&parquet_bytes);
        let meta_ok = meta.verify_meta_checksum()?;
        if !data_ok {
            eprintln!(
                "verify_row_group_checksum data mismatch for {}",
                parquet_path
            );
        }
        if !meta_ok {
            eprintln!("verify_row_group_checksum meta mismatch for {}", meta_path);
        }
        Ok(data_ok && meta_ok)
    }

    /// Read a manifest `.tmp` file and verify that its stored checksum matches
    /// the recomputed checksum.
    async fn verify_manifest_checksum(&self, manifest_tmp_path: &str) -> Result<bool> {
        let read_data = self.storage.read(manifest_tmp_path).await?;
        let read_manifest: Manifest = serde_json::from_slice(&read_data)?;
        read_manifest.verify_checksum()
    }

    /// Set `parent_hash` + `committed_at` on `manifest` then recompute its
    /// self-checksum.  Must be called instead of
    /// `manifest.checksum = manifest.compute_checksum()?` at every
    /// manifest-publish site so the snapshot history forms a verified hash chain.
    ///
    /// The parent link is the checksum of the highest on-disk manifest whose
    /// sequence is strictly less than `next_seq`. The common case — the
    /// immediate predecessor `<next_seq-1>.json` exists — is a single read with
    /// no directory scan. Only when that read fails (a WAL fold that skipped
    /// intermediate sequences, or GC pruning) do we list `_manifests/` for the
    /// highest surviving predecessor. `parent_hash = None` (a true chain anchor)
    /// is produced only at genesis or when every predecessor has been pruned.
    async fn finalize_manifest(&self, manifest: &mut Manifest, next_seq: u64) -> Result<()> {
        crate::metadata::finalize_manifest(self.storage.as_ref(), &self.table, manifest, next_seq)
            .await
    }

    /// The writer's view of current table state, WAL-aware.
    ///
    /// Replays the mutation WAL (lock-free, read-only) onto the checkpointed
    /// manifest and returns the live `(sequence, manifest)`, so every commit
    /// (WAL-deferred or not) builds on the deferred mutations. A no-op that
    /// returns the checkpoint unchanged when no `_wal/` log exists (the default).
    async fn load_current_manifest(&self) -> Result<(u64, Manifest)> {
        let (seq, manifest) = self.load_checkpoint_manifest().await?;
        if seq == 0 {
            return Ok((seq, manifest)); // empty table — nothing to replay
        }
        let live = crate::mutation_wal::live_manifest(self.storage.as_ref(), &self.table, manifest)
            .await?;
        Ok((live.sequence, live))
    }

    /// Load the checkpointed manifest the `_manifest.json` pointer references
    /// (WAL not applied — see [`Self::load_current_manifest`] for the live view).
    ///
    /// 1. If `_manifest.json` exists and contains a valid `latest` sequence,
    ///    read the manifest it points to directly.
    ///    - If the manifest file is missing, return `ManifestNotFound`.
    ///    - If the manifest cannot be parsed as JSON, return `Serialization`.
    ///    - If the manifest parses but its checksum is invalid, return
    ///      `ChecksumMismatch`.
    ///      In all three cases the error refers to the manifest referenced by the
    ///      pointer; older manifests are not used as a fallback.
    /// 2. If `_manifest.json` is missing or malformed, recover by scanning
    ///    `{table}/_manifests/` for the highest valid manifest and atomically
    ///    repair the pointer. If no valid manifest exists, an empty table is
    ///    returned.
    /// 3. A `latest` value of `0` is valid and indicates an empty table with no
    ///    committed manifests. If manifests exist while the pointer says `0`, the
    ///    pointer is treated as malformed and recovery is attempted.
    async fn load_checkpoint_manifest(&self) -> Result<(u64, Manifest)> {
        let pointer_path = format!("{}/_manifest.json", self.table);
        let mut pointer_malformed = false;
        let mut latest_zero = false;

        if self.storage.exists(&pointer_path).await? {
            match self.storage.read(&pointer_path).await {
                Ok(data) => {
                    let pointer = serde_json::from_slice::<serde_json::Value>(&data);
                    match pointer {
                        Ok(p) => match p.get("latest").and_then(|v| v.as_u64()) {
                            Some(0) => {
                                // A value of 0 is valid only for a truly empty table.
                                // If manifests exist, treat it as a malformed pointer
                                // and recover below.
                                latest_zero = true;
                            }
                            Some(seq) if seq > 0 => {
                                let manifest_path =
                                    format!("{}/{}", self.table, Manifest::filename(seq));
                                match self.read_manifest_validated(&manifest_path).await {
                                    Ok(manifest) => {
                                        return Ok((seq, manifest));
                                    }
                                    Err(IcefallDBError::NotFound(_)) => {
                                        return Err(IcefallDBError::ManifestNotFound(
                                            manifest_path,
                                        ));
                                    }
                                    Err(e) => return Err(e),
                                }
                            }
                            _ => pointer_malformed = true,
                        },
                        Err(_) => pointer_malformed = true,
                    }
                }
                Err(IcefallDBError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }

        // The pointer is missing or malformed. Recover from `_manifests/`.
        let manifests_dir = format!("{}/_manifests", self.table);
        let entries = match self.storage.list(&manifests_dir).await {
            Ok(entries) => entries,
            Err(e) if is_not_found(&e) => Vec::new(),
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

        let mut highest_valid_seq = 0u64;
        let mut highest_valid_manifest = None;
        for seq in &sequences {
            let manifest_path = format!("{}/{}", self.table, Manifest::filename(*seq));
            if let Ok(manifest) = self.read_manifest_validated(&manifest_path).await {
                highest_valid_seq = *seq;
                highest_valid_manifest = Some(manifest);
                break;
            }
        }

        if highest_valid_seq > 0 {
            self.write_pointer(highest_valid_seq).await?;
            return Ok((highest_valid_seq, highest_valid_manifest.unwrap()));
        }

        // `latest: 0` with no manifest snapshots is a valid empty table. Any
        // other malformed pointer, or `latest: 0` when valid manifest snapshots
        // exist, is an error.
        if latest_zero && !sequences.is_empty() {
            return Err(IcefallDBError::InvalidManifestPointer(pointer_path));
        }
        if pointer_malformed {
            return Err(IcefallDBError::InvalidManifestPointer(pointer_path));
        }

        // No pointer and no manifests: an empty table.
        Ok((
            0,
            Manifest {
                format_version: 1,
                sequence: 0,
                schema_id: 0,
                row_groups: Vec::new(),
                row_counts: None,
                partition_values: None,
                checksum: String::new(),
                ..Default::default()
            },
        ))
    }

    /// Read a manifest file, parse it, and verify its checksum.
    ///
    /// Distinguishes parse errors (`Serialization`) from checksum failures
    /// (`ChecksumMismatch`) and missing files (`NotFound`).
    async fn read_manifest_validated(&self, manifest_path: &str) -> Result<Manifest> {
        let manifest_data = self.storage.read(manifest_path).await?;
        let manifest: Manifest = serde_json::from_slice(&manifest_data)?;
        if !manifest.verify_checksum()? {
            return Err(IcefallDBError::ChecksumMismatch {
                path: manifest_path.to_string(),
            });
        }
        Ok(manifest)
    }

    /// Atomically write `_manifest.json` to point to `seq`.
    async fn write_pointer(&self, seq: u64) -> Result<()> {
        let pointer_path = format!("{}/_manifest.json", self.table);
        let pointer_tmp_path = format!("{}.tmp", pointer_path);
        let pointer = serde_json::json!({"latest": seq});
        self.storage
            .write(&pointer_tmp_path, serde_json::to_vec(&pointer)?.as_slice())
            .await?;
        self.storage.sync_data(&pointer_tmp_path).await?;
        self.storage
            .rename(&pointer_tmp_path, &pointer_path)
            .await?;
        self.storage.sync(&format!("{}/", self.table)).await?;
        Ok(())
    }
}

/// Build a `SnapshotCheckpoint` for `new_manifest`.
///
/// Reuses summaries from the previous checkpoint carried by `current_manifest`
/// for fragments whose `RowGroupEntry` is unchanged, and reads `.meta` sidecars
/// for changed or new fragments.
pub(crate) async fn build_snapshot_checkpoint(
    storage: &dyn Storage,
    table: &str,
    current_manifest: &Manifest,
    new_manifest: &Manifest,
) -> Result<SnapshotCheckpoint> {
    let mut prev_by_fragment: HashMap<u64, FragmentSummary> = HashMap::new();
    if let Some(prev_path) = &current_manifest.checkpoint {
        let abs_path = format!("{}/{}", table, prev_path);
        // The previous checkpoint is an optimization, not required for
        // correctness. If it is missing or unreadable, fall back to reading the
        // per-fragment `.meta` sidecars for all fragments.
        if let Ok(bytes) = storage.read(&abs_path).await {
            if let Ok(prev) = serde_json::from_slice::<SnapshotCheckpoint>(&bytes) {
                for summary in prev.fragments {
                    prev_by_fragment.insert(summary.fragment_id, summary);
                }
            }
        }
    }

    let mut current_entries_by_fragment: HashMap<u64, &RowGroupEntry> = HashMap::new();
    for entry in &current_manifest.row_groups {
        current_entries_by_fragment.insert(entry.fragment_id, entry);
    }

    let cache = crate::meta_cache::MetaCache::global();
    let mut fragments = Vec::with_capacity(new_manifest.row_groups.len());
    for entry in &new_manifest.row_groups {
        let unchanged = current_entries_by_fragment
            .get(&entry.fragment_id)
            .map(|current| {
                current.data == entry.data
                    && current.meta == entry.meta
                    && current.deletes == entry.deletes
                    && current.deleted_count == entry.deleted_count
                    && current.agg == entry.agg
            })
            .unwrap_or(false);

        // Reuse the previous summary only if it carries the checksums required
        // by the aggregate-cache trust gate. Legacy checkpoints with empty
        // checksums are backfilled by reading the `.meta` sidecar once.
        if unchanged {
            if let Some(summary) = prev_by_fragment.get(&entry.fragment_id) {
                if !summary.checksum.is_empty() && !summary.meta_checksum.is_empty() {
                    fragments.push(summary.clone());
                    continue;
                }
            }
        }

        let meta_path = format!("{}/{}", table, entry.meta);
        let meta = if let Some(cached) = cache.get(&meta_path) {
            cached.as_ref().clone()
        } else {
            let meta_bytes = storage.read(&meta_path).await?;
            serde_json::from_slice::<RowGroupMeta>(&meta_bytes)?
        };
        fragments.push(FragmentSummary {
            row_group: meta.row_group.clone(),
            data: entry.data.clone(),
            meta: entry.meta.clone(),
            agg: entry.agg.clone(),
            deletes: entry.deletes.clone(),
            fragment_id: entry.fragment_id,
            rows: meta.rows,
            deleted_count: entry.deleted_count,
            columns: meta.columns.clone(),
            column_offsets: meta.column_offsets.clone(),
            row_ids: meta.row_ids.clone(),
            sort: meta.sort.clone(),
            checksum: meta.checksum.clone(),
            meta_checksum: meta.meta_checksum.clone(),
        });
    }

    let mut checkpoint = SnapshotCheckpoint {
        sequence: new_manifest.sequence,
        schema_id: new_manifest.schema_id,
        fragments,
        checksum: String::new(),
    };
    checkpoint.checksum = checkpoint.compute_checksum()?;
    Ok(checkpoint)
}

/// Write a snapshot checkpoint file atomically and add it to the commit intent.
///
/// The checkpoint is staged in `_staging/incoming/`, renamed to
/// `_checkpoints/{seq:09}.json`, and its relative path is appended to
/// `intent_files` (and rewritten to the intent journal) before the rename.
/// `staged_files` is updated so rollback can remove the file if the pointer swap
/// has not happened.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn write_checkpoint(
    storage: &dyn Storage,
    table: &str,
    checkpoint: &SnapshotCheckpoint,
    seq: u64,
    intent_path: &str,
    intent_files: &mut Vec<String>,
    staged_files: &mut Vec<String>,
    txn_id: &str,
    schema_id: u64,
) -> Result<String> {
    let rel_path = SnapshotCheckpoint::filename(seq);
    let final_path = format!("{}/{}", table, rel_path);
    // Stage the checkpoint in the flat incoming directory so cleanup_staging
    // can list and reclaim it on a failed commit.
    let part_path = format!(
        "{}/_staging/incoming/_checkpoint_{:09}.json.part",
        table, seq
    );

    ensure_dir(storage, &format!("{}/_checkpoints", table)).await?;

    let bytes = serde_json::to_vec(checkpoint)?;
    storage.write(&part_path, &bytes).await?;
    storage.sync_data(&part_path).await?;

    // Append the checkpoint to the intent so a failed/crashed commit can reclaim
    // the orphaned `_checkpoints/{seq}.json`. The intent file already exists with
    // a durable directory entry (the caller's initial intent write fsync'd
    // `_staging/intents`); this is an in-place rewrite of the same path (stable
    // inode), so `sync_data` makes the new content durable and NO second
    // `_staging/intents` directory fsync is needed — that batches the checkpoint
    // writer's barrier into the initial intent's.
    intent_files.push(rel_path.clone());
    let intent = serde_json::json!({
        "txn_id": txn_id,
        "started_at": chrono::Utc::now().to_rfc3339(),
        "schema_id": schema_id,
        "files": intent_files,
    });
    storage
        .write(intent_path, serde_json::to_vec(&intent)?.as_slice())
        .await?;
    storage.sync_data(intent_path).await?;

    storage.rename(&part_path, &final_path).await?;
    staged_files.push(rel_path.clone());

    // Derived zero-copy archive sibling: an `.rkyv` next to the JSON
    // checkpoint so opens can rebuild the scan plan without the O(fragments)
    // serde_json parse. Local storage only; best-effort — the JSON is canonical,
    // so a failed/absent archive just means the reader falls back to it.
    if storage.local_root().is_some() {
        let arch_rel = SnapshotCheckpoint::archive_filename(seq);
        let arch_path = format!("{}/{}", table, arch_rel);
        let arch_tmp = format!("{}.tmp", arch_path);
        let arch_bytes = checkpoint.to_archive_bytes();
        if !arch_bytes.is_empty()
            && storage.write(&arch_tmp, &arch_bytes).await.is_ok()
            && storage.sync_data(&arch_tmp).await.is_ok()
            && storage.rename(&arch_tmp, &arch_path).await.is_ok()
        {
            staged_files.push(arch_rel);
        } else {
            let _ = storage.delete(&arch_tmp).await;
        }
    }

    Ok(rel_path)
}

fn slice_batch(batch: &RecordBatch, start: usize, len: usize) -> Result<RecordBatch> {
    let schema = batch.schema();
    let arrays: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| col.slice(start, len))
        .collect();
    RecordBatch::try_new(schema, arrays).map_err(other)
}

/// Build a JSON-array key representing the partition tuple for `row`.
fn partition_key_for_row(
    batch: &RecordBatch,
    partition_by: &[String],
    row: usize,
) -> Result<String> {
    let mut key_parts = Vec::with_capacity(partition_by.len());
    for col_name in partition_by {
        let array =
            batch
                .column_by_name(col_name)
                .ok_or_else(|| IcefallDBError::SchemaMismatch {
                    column: col_name.clone(),
                    expected: "found in batch".into(),
                    path: "partition".into(),
                })?;
        if !array.is_valid(row) {
            key_parts.push(serde_json::Value::Null);
            continue;
        }
        let value = scalar_to_json_value(array, row)?;
        key_parts.push(value);
    }
    serde_json::to_string(&key_parts).map_err(other)
}

/// Convert the scalar value at `row` in `array` to a JSON value.
///
/// Supported types match those used by [`compute_partition_values`]: booleans,
/// integers, floats, utf8/large_utf8, and plain microsecond timestamps.
fn scalar_to_json_value(array: &ArrayRef, row: usize) -> Result<serde_json::Value> {
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        LargeStringArray, StringArray, TimestampMicrosecondArray, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };

    let value = match array.data_type() {
        DataType::Boolean => {
            let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            arr.value(row).into()
        }
        DataType::Int8 => {
            let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
            (arr.value(row) as i64).into()
        }
        DataType::Int16 => {
            let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
            (arr.value(row) as i64).into()
        }
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            (arr.value(row) as i64).into()
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            arr.value(row).into()
        }
        DataType::UInt8 => {
            let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
            (arr.value(row) as u64).into()
        }
        DataType::UInt16 => {
            let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
            (arr.value(row) as u64).into()
        }
        DataType::UInt32 => {
            let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
            (arr.value(row) as u64).into()
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            arr.value(row).into()
        }
        DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let v = arr.value(row);
            if !v.is_finite() {
                // Non-finite values cannot be represented as partition values,
                // so they are grouped like NULLs.
                return Ok(serde_json::Value::Null);
            }
            (v as f64).into()
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let v = arr.value(row);
            if !v.is_finite() {
                // Non-finite values cannot be represented as partition values,
                // so they are grouped like NULLs.
                return Ok(serde_json::Value::Null);
            }
            v.into()
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            arr.value(row).into()
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            arr.value(row).into()
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            arr.value(row).into()
        }
        other => {
            return Err(IcefallDBError::InvalidSchema {
                reason: format!("partition column has unsupported type {:?}", other),
                path: "partition".into(),
            })
        }
    };
    Ok(value)
}

/// Validate that every partition column has a supported scalar type.
fn validate_partition_types(
    batch: &RecordBatch,
    partition_by: &[String],
    table_path: &str,
) -> Result<()> {
    for col_name in partition_by {
        let array =
            batch
                .column_by_name(col_name)
                .ok_or_else(|| IcefallDBError::SchemaMismatch {
                    column: col_name.clone(),
                    expected: "found in batch".into(),
                    path: table_path.into(),
                })?;
        if !is_supported_partition_type(array.data_type()) {
            return Err(IcefallDBError::InvalidSchema {
                reason: format!(
                    "partition column '{}' has unsupported type {:?}",
                    col_name,
                    array.data_type()
                ),
                path: table_path.into(),
            });
        }
    }
    Ok(())
}

/// Returns true if the Arrow type is supported as a partition column.
fn is_supported_partition_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Timestamp(TimeUnit::Microsecond, None)
    )
}

/// Return a new batch containing only the rows at the given indices.
fn take_batch_by_indices(batch: &RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
    let indices_array =
        arrow::array::Int64Array::from_iter_values(indices.iter().map(|&i| i as i64));
    let columns: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|col| arrow::compute::take(col.as_ref(), &indices_array, None).map_err(other))
        .collect::<Result<_>>()?;
    RecordBatch::try_new(batch.schema().clone(), columns).map_err(other)
}

/// Returns a full UUID with all dashes removed, suitable for filenames.
fn full_uuid() -> String {
    uuid::Uuid::new_v4().to_string().replace('-', "")
}

/// Generate a unique row-group id within this commit.
///
/// The 128-bit UUID space makes collisions negligible, so no retry against
/// existing storage files is necessary.
fn unique_rg_id(used: &mut HashSet<String>) -> String {
    let rg_id = format!("rg_{}", full_uuid());
    used.insert(rg_id.clone());
    rg_id
}

/// Ensure a directory exists by writing and removing a temporary file. This
/// works for storage backends that create parent directories on write.
async fn ensure_dir(storage: &dyn Storage, path: &str) -> Result<()> {
    match storage.list(path).await {
        Ok(_) => Ok(()),
        Err(e) if is_not_found(&e) => {
            let tmp = format!("{}/.keep", path);
            storage.write(&tmp, b"").await?;
            let _ = storage.delete(&tmp).await;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Clean up staging artifacts left behind by a crashed commit or compaction.
///
/// This must only be called while the exclusive writer lock is held and while
/// `referenced_files` contains the files referenced by the latest committed
/// manifest. Recovery relies on the caller already holding the exclusive writer
/// lock; this function does not verify lock ownership or process identity. Any
/// leftover intent is therefore assumed to be from a crashed writer and is
/// treated as stale.
///
/// For each file listed in an intent, recovery checks whether the file is in
/// `referenced_files`. Referenced files survived a post-pointer-update crash and
/// must not be deleted; unreferenced files were abandoned before the pointer
/// update and are removed. The intent file itself is always removed. Any
/// `.part` files in `_staging/incoming/` and `_staging/compact/` are also
/// removed, as are manifest snapshots with sequence numbers greater than
/// `latest_seq` and any `.json.tmp` files in `_manifests/`.
///
/// Final row-group files (`rg_*.parquet` / `rg_*.meta`) that are not referenced
/// by the current manifest are *not* deleted here; garbage collection of
/// uncommitted final row-group files is the responsibility of `icefalldb gc`.
pub(crate) async fn cleanup_staging(
    storage: &dyn Storage,
    table: &str,
    latest_seq: u64,
    referenced_files: &HashSet<String>,
) -> Result<()> {
    let intents_dir = format!("{}/_staging/intents", table);
    let entries: Vec<String> = match storage.list(&intents_dir).await {
        Ok(e) => e,
        Err(e) if is_not_found(&e) => Vec::new(),
        Err(e) => return Err(e),
    };

    for entry in entries {
        if !entry.ends_with(".json") {
            continue;
        }

        let intent_data = match storage.read(&entry).await {
            Ok(d) => d,
            Err(e) if is_not_found(&e) => {
                // Already gone; nothing to clean up.
                continue;
            }
            Err(e) => return Err(e),
        };

        if let Ok(intent) = serde_json::from_slice::<serde_json::Value>(&intent_data) {
            // An intent lists the final table-root filenames for its row groups.
            // If those filenames are referenced by the current manifest, they
            // were committed before a crash and must be preserved. Otherwise
            // they are abandoned staged files and can be deleted.
            if let Some(files) = intent.get("files").and_then(|v| v.as_array()) {
                for file in files {
                    if let Some(filename) = file.as_str() {
                        if referenced_files.contains(filename) {
                            continue;
                        }
                        if let Err(e) = storage.delete(&format!("{}/{}", table, filename)).await {
                            if !is_not_found(&e) {
                                return Err(e);
                            }
                        }
                    }
                }
            }
        }
        // Unparseable intents cannot tell us which files to clean up, but the
        // intent file itself is still stale and must be removed.

        if let Err(e) = storage.delete(&entry).await {
            if !is_not_found(&e) {
                return Err(e);
            }
        }
    }

    // Remove any orphaned staged files left behind by a crashed writer or
    // compactor.
    for staging_dir in ["_staging/incoming", "_staging/compact"] {
        let dir = format!("{}/{}", table, staging_dir);
        let dir_entries: Vec<String> = match storage.list(&dir).await {
            Ok(i) => i,
            Err(e) if is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        for entry in dir_entries {
            if entry.ends_with(".part") {
                if let Err(e) = storage.delete(&entry).await {
                    if !is_not_found(&e) {
                        return Err(e);
                    }
                }
            }
        }
    }

    // Remove any orphaned manifest snapshots left by a crash before the pointer
    // update. These would otherwise cause a SequenceCollision on the next commit.
    let manifests_dir = format!("{}/_manifests", table);
    let manifests: Vec<String> = match storage.list(&manifests_dir).await {
        Ok(m) => m,
        Err(e) if is_not_found(&e) => Vec::new(),
        Err(e) => return Err(e),
    };
    for entry in manifests {
        let filename = std::path::Path::new(&entry)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if filename.ends_with(".json.tmp") {
            if let Err(e) = storage.delete(&entry).await {
                if !is_not_found(&e) {
                    return Err(e);
                }
            }
            continue;
        }
        if let Some(seq_str) = filename.strip_suffix(".json") {
            if let Ok(seq) = seq_str.parse::<u64>() {
                if seq > latest_seq {
                    if let Err(e) = storage.delete(&entry).await {
                        if !is_not_found(&e) {
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn arrow_schema_from_icefalldb(schema: &Schema, path: &str) -> Result<SchemaRef> {
    let fields: Result<Vec<_>> = schema
        .columns
        .iter()
        .map(|c| {
            Ok(Field::new(
                &c.name,
                arrow_type(&c.r#type, &c.name, path)?,
                c.nullable,
            ))
        })
        .collect();
    Ok(Arc::new(ArrowSchema::new(fields?)))
}

/// Returns true if two Arrow schemas have the same number, names, types, and
/// nullability of fields, ignoring any additional metadata that may be attached
/// to the schema or individual fields.
fn schema_equal_ignoring_metadata(a: &ArrowSchema, b: &ArrowSchema) -> bool {
    if a.fields().len() != b.fields().len() {
        return false;
    }
    a.fields().iter().zip(b.fields().iter()).all(|(af, bf)| {
        af.name() == bf.name()
            && af.data_type() == bf.data_type()
            && af.is_nullable() == bf.is_nullable()
    })
}

/// Returns true if two Arrow schemas have the same number, names, and data
/// types of fields, ignoring nullability and any additional metadata.
///
/// This is used for UPDATE patch batches: DataFusion projections of non-null
/// literals produce non-nullable output fields even when the target column is
/// nullable, so nullability must not be part of the defensive check.
fn schema_equal_names_and_types(a: &ArrowSchema, b: &ArrowSchema) -> bool {
    if a.fields().len() != b.fields().len() {
        return false;
    }
    a.fields()
        .iter()
        .zip(b.fields().iter())
        .all(|(af, bf)| af.name() == bf.name() && af.data_type() == bf.data_type())
}

fn arrow_type(type_str: &str, column: &str, path: &str) -> Result<DataType> {
    icefalldb_type_to_arrow(type_str).ok_or_else(|| IcefallDBError::InvalidSchema {
        reason: format!("column '{}' has unsupported type '{}'", column, type_str),
        path: path.into(),
    })
}

/// Compute per-column byte offsets from the Parquet footer.
fn compute_column_offsets(
    parquet_metadata: &parquet::file::metadata::ParquetMetaData,
    schema: &Schema,
) -> Option<HashMap<String, ColumnChunkOffset>> {
    let row_groups = parquet_metadata.row_groups();
    if row_groups.is_empty() {
        return None;
    }
    let rg = &row_groups[0];
    let schema_descr = parquet_metadata.file_metadata().schema_descr();
    let mut offsets = HashMap::new();
    for col in &schema.columns {
        let leaf_idx = schema_descr
            .columns()
            .iter()
            .position(|c| c.name() == col.name)?;
        let col_meta = rg.column(leaf_idx);
        offsets.insert(
            col.name.clone(),
            ColumnChunkOffset {
                offset: col_meta
                    .dictionary_page_offset()
                    .map(|o| o.min(col_meta.data_page_offset().max(0)))
                    .unwrap_or_else(|| col_meta.data_page_offset().max(0))
                    as u64,
                length: col_meta.compressed_size().max(0) as u64,
            },
        );
    }
    Some(offsets)
}

pub fn compute_row_group_meta(
    rg_id: &str,
    schema_id: u64,
    batch: &RecordBatch,
    schema: &Schema,
    parquet_bytes: &[u8],
    path: &str,
    row_ids: &[RowIdSegment],
) -> Result<RowGroupMeta> {
    let mut columns = HashMap::new();
    for col in &schema.columns {
        let array =
            batch
                .column_by_name(&col.name)
                .ok_or_else(|| IcefallDBError::SchemaMismatch {
                    column: col.name.clone(),
                    expected: "found in batch".into(),
                    path: path.into(),
                })?;
        let nulls = array.null_count();
        let (min, max) = compute_min_max(array, path)?;
        columns.insert(col.name.clone(), ColumnStats { min, max, nulls });
    }
    let column_offsets = {
        let bytes = Bytes::copy_from_slice(parquet_bytes);
        ParquetRecordBatchReaderBuilder::try_new(bytes)
            .ok()
            .and_then(|builder| compute_column_offsets(builder.metadata().as_ref(), schema))
    };
    let mut meta = RowGroupMeta {
        row_group: rg_id.to_string(),
        schema_id,
        rows: batch.num_rows(),
        columns,
        column_offsets,
        sort: schema.sort.clone(),
        row_ids: row_ids.to_vec(),
        checksum: String::new(),
        meta_checksum: String::new(),
    };
    meta.compute_checksum(parquet_bytes)?;
    Ok(meta)
}

/// Best-effort regeneration of a `.meta` sidecar from the Parquet footer.
///
/// Used by [`crate::doctor::Doctor`] to recreate a missing row-group metadata
/// file. Statistics are derived from footer column-chunk statistics; encrypted
/// columns (when known) are skipped so the regenerated plaintext sidecar does
/// not leak protected values.
///
/// Row-ID segments cannot be recovered from the Parquet footer, but they *are*
/// retained in the snapshot checkpoint's fragment summary. The caller passes the
/// recovered segments in via `row_ids`; this keeps the regenerated canonical
/// `.meta` consistent with the data so mutations (which locate rows by row id)
/// keep working and a later checkpoint rebuild does not permanently lose them.
/// Pass an empty slice when the segments are genuinely unrecoverable.
pub fn compute_row_group_meta_from_footer(
    rg_id: &str,
    schema_id: u64,
    schema: &Schema,
    parquet_bytes: &[u8],
    parquet_metadata: &parquet::file::metadata::ParquetMetaData,
    encrypted_columns: &std::collections::HashSet<String>,
    row_ids: &[RowIdSegment],
) -> Result<RowGroupMeta> {
    let (mut columns, needs_recompute) =
        derive_footer_stats_for_repair(schema, parquet_metadata, encrypted_columns)?;

    // Externally-ingested Parquet may carry no column statistics in its footer.
    // A footer-only meta would then record nulls=0/min=max=None for those
    // columns, which the checker later flags against the real data. Read just
    // the affected (non-encrypted) columns and compute exact statistics so the
    // regenerated meta matches the data. IcefallDB-written files always carry
    // footer stats, so this fallback never fires for them (no extra I/O).
    if !needs_recompute.is_empty() {
        recompute_repair_stats_from_data(parquet_bytes, &needs_recompute, &mut columns)?;
    }

    let column_offsets = compute_column_offsets(parquet_metadata, schema);
    let mut meta = RowGroupMeta {
        row_group: rg_id.to_string(),
        schema_id,
        rows: parquet_metadata.file_metadata().num_rows() as usize,
        columns,
        column_offsets,
        sort: schema.sort.clone(),
        row_ids: row_ids.to_vec(),
        checksum: String::new(),
        meta_checksum: String::new(),
    };
    meta.compute_checksum(parquet_bytes)?;
    Ok(meta)
}

/// Derive column statistics from the Parquet footer for repair purposes.
///
/// Unlike the fast-path [`Writer::derive_footer_stats`], this function is
/// best-effort: unsupported columns are omitted, and missing min/max are
/// represented as `None`. Encrypted columns are skipped entirely.
///
/// Returns the derived stats plus a list of `(column name, leaf index)` for
/// columns whose footer lacked statistics on at least one row group: those
/// stats are unreliable and the caller must recompute them from the data.
#[allow(clippy::type_complexity)]
fn derive_footer_stats_for_repair(
    schema: &Schema,
    metadata: &parquet::file::metadata::ParquetMetaData,
    encrypted_columns: &std::collections::HashSet<String>,
) -> Result<(HashMap<String, ColumnStats>, Vec<(String, usize)>)> {
    let schema_descr = metadata.file_metadata().schema_descr();
    let mut columns = HashMap::new();
    let mut needs_recompute: Vec<(String, usize)> = Vec::new();

    for col in &schema.columns {
        if encrypted_columns.contains(&col.name) {
            continue;
        }
        let arrow_type = match icefalldb_type_to_arrow(&col.r#type) {
            Some(t) => t,
            None => continue,
        };
        let leaf_idx = match schema_descr
            .columns()
            .iter()
            .position(|c| c.name() == col.name)
        {
            Some(i) => i,
            None => continue,
        };

        let mut nulls: usize = 0;
        let mut min: Option<serde_json::Value> = None;
        let mut max: Option<serde_json::Value> = None;
        let mut stats_present_all = true;

        for rg in metadata.row_groups() {
            let col_meta = rg.column(leaf_idx);
            let Some(stats) = col_meta.statistics() else {
                stats_present_all = false;
                continue;
            };
            nulls += stats.null_count_opt().unwrap_or(0) as usize;
            if is_supported_footer_stat_type(&arrow_type) {
                let (rg_min, rg_max) = parquet_stats_to_json(stats, &arrow_type)?;
                min = min_json(min, rg_min);
                max = max_json(max, rg_max);
            }
        }

        // A row group with rows but no footer statistics yields bogus
        // nulls=0/min=max=None; mark the column for an exact recompute.
        if !stats_present_all {
            needs_recompute.push((col.name.clone(), leaf_idx));
        }
        columns.insert(col.name.clone(), ColumnStats { min, max, nulls });
    }

    Ok((columns, needs_recompute))
}

/// Recompute exact statistics for `needs_recompute` columns by reading the
/// Parquet data. Used by [`compute_row_group_meta_from_footer`] when the footer
/// lacks statistics. Only the listed (non-encrypted, single-leaf) columns are
/// projected, so encrypted column chunks are never touched.
fn recompute_repair_stats_from_data(
    parquet_bytes: &[u8],
    needs_recompute: &[(String, usize)],
    columns: &mut HashMap<String, ColumnStats>,
) -> Result<()> {
    use parquet::arrow::ProjectionMask;

    let bytes = Bytes::copy_from_slice(parquet_bytes);
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes).map_err(other)?;
    let leaves = needs_recompute.iter().map(|(_, idx)| *idx);
    let mask = ProjectionMask::leaves(builder.parquet_schema(), leaves);
    let reader = builder.with_projection(mask).build().map_err(other)?;

    let mut batches: Vec<RecordBatch> = Vec::new();
    for batch in reader {
        batches.push(batch.map_err(other)?);
    }
    if batches.is_empty() {
        return Ok(()); // no rows: nulls=0/min=max=None is already exact
    }

    let combined = arrow::compute::concat_batches(&batches[0].schema(), &batches).map_err(other)?;
    for (name, _) in needs_recompute {
        let Some(array) = combined.column_by_name(name) else {
            continue;
        };
        let nulls = array.null_count();
        let (min, max) = compute_min_max(array, name)?;
        columns.insert(name.clone(), ColumnStats { min, max, nulls });
    }
    Ok(())
}

/// Read a `.meta` sidecar and return its row count if it is valid.
async fn read_meta_rows(storage: &dyn Storage, meta_path: &str) -> Option<usize> {
    match storage.read(meta_path).await {
        Ok(bytes) => serde_json::from_slice::<RowGroupMeta>(&bytes)
            .ok()
            .filter(|m| m.verify_meta_checksum().unwrap_or(false))
            .map(|m| m.rows),
        Err(_) => None,
    }
}

/// Build denormalized row counts for a new manifest.
///
/// `existing_entries` are the row groups carried forward from the current
/// snapshot and `current_manifest` supplies any already-denormalized counts.
/// `added_counts` are the row counts for newly written row groups.
async fn collect_row_counts(
    storage: &dyn Storage,
    table: &str,
    existing_entries: &[RowGroupEntry],
    current_manifest: &Manifest,
    added_counts: &[usize],
) -> Option<Vec<usize>> {
    let mut counts = Vec::with_capacity(existing_entries.len() + added_counts.len());
    if let Some(ref current_counts) = current_manifest.row_counts {
        if current_counts.len() == existing_entries.len() {
            counts.extend_from_slice(current_counts);
        } else {
            return None;
        }
    } else {
        for entry in existing_entries {
            let meta_path = format!("{}/{}", table, entry.meta);
            let rows = read_meta_rows(storage, &meta_path).await?;
            counts.push(rows);
        }
    }
    counts.extend_from_slice(added_counts);
    Some(counts)
}

/// Returns true if the Arrow type is supported for footer-statistics derivation
/// in the Parquet zero-copy fast path.
fn is_supported_footer_stat_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Timestamp(TimeUnit::Microsecond, None)
    )
}

/// Convert Parquet column-chunk statistics to JSON min/max values suitable for
/// [`ColumnStats`]. Returns `Ok((None, None))` when the statistics are absent.
fn parquet_stats_to_json(
    stats: &Statistics,
    arrow_type: &DataType,
) -> Result<(Option<serde_json::Value>, Option<serde_json::Value>)> {
    match (arrow_type, stats) {
        (DataType::Boolean, Statistics::Boolean(s)) => Ok((
            s.min_opt().copied().map(Into::into),
            s.max_opt().copied().map(Into::into),
        )),
        (DataType::Int8, Statistics::Int32(s)) => Ok((
            s.min_opt().map(|&v| (v as i8).into()),
            s.max_opt().map(|&v| (v as i8).into()),
        )),
        (DataType::Int16, Statistics::Int32(s)) => Ok((
            s.min_opt().map(|&v| (v as i16).into()),
            s.max_opt().map(|&v| (v as i16).into()),
        )),
        (DataType::Int32, Statistics::Int32(s)) => Ok((
            s.min_opt().map(|&v| v.into()),
            s.max_opt().map(|&v| v.into()),
        )),
        (DataType::Int64, Statistics::Int64(s)) => Ok((
            s.min_opt().map(|&v| v.into()),
            s.max_opt().map(|&v| v.into()),
        )),
        (DataType::UInt8, Statistics::Int32(s)) => Ok((
            s.min_opt().map(|&v| (v as u8).into()),
            s.max_opt().map(|&v| (v as u8).into()),
        )),
        (DataType::UInt16, Statistics::Int32(s)) => Ok((
            s.min_opt().map(|&v| (v as u16).into()),
            s.max_opt().map(|&v| (v as u16).into()),
        )),
        (DataType::UInt32, Statistics::Int32(s)) => Ok((
            s.min_opt().map(|&v| (v as u32).into()),
            s.max_opt().map(|&v| (v as u32).into()),
        )),
        (DataType::UInt64, Statistics::Int64(s)) => Ok((
            s.min_opt().map(|&v| (v as u64).into()),
            s.max_opt().map(|&v| (v as u64).into()),
        )),
        (DataType::Float32, Statistics::Float(s)) => Ok((
            s.min_opt()
                .filter(|&&v| v.is_finite())
                .map(|&v| (v as f64).into()),
            s.max_opt()
                .filter(|&&v| v.is_finite())
                .map(|&v| (v as f64).into()),
        )),
        (DataType::Float64, Statistics::Double(s)) => Ok((
            s.min_opt().filter(|&&v| v.is_finite()).map(|&v| v.into()),
            s.max_opt().filter(|&&v| v.is_finite()).map(|&v| v.into()),
        )),
        (DataType::Utf8 | DataType::LargeUtf8, Statistics::ByteArray(s)) => Ok((
            s.min_opt()
                .and_then(|ba| std::str::from_utf8(ba.data()).ok().map(Into::into)),
            s.max_opt()
                .and_then(|ba| std::str::from_utf8(ba.data()).ok().map(Into::into)),
        )),
        (DataType::Timestamp(TimeUnit::Microsecond, None), Statistics::Int64(s)) => Ok((
            s.min_opt().map(|&v| v.into()),
            s.max_opt().map(|&v| v.into()),
        )),
        _ => Ok((None, None)),
    }
}

/// JSON-aware minimum. Returns `None` only when both inputs are `None`.
fn json_cmp(a: &serde_json::Value, b: &serde_json::Value) -> Option<std::cmp::Ordering> {
    use serde_json::Value;
    match (a, b) {
        (Value::Bool(a), Value::Bool(b)) => a.partial_cmp(b),
        (Value::Number(a), Value::Number(b)) => {
            if let (Some(ai), Some(bi)) = (a.as_i64(), b.as_i64()) {
                ai.partial_cmp(&bi)
            } else if let (Some(au), Some(bu)) = (a.as_u64(), b.as_u64()) {
                au.partial_cmp(&bu)
            } else if let (Some(af), Some(bf)) = (a.as_f64(), b.as_f64()) {
                af.partial_cmp(&bf)
            } else {
                None
            }
        }
        (Value::String(a), Value::String(b)) => a.partial_cmp(b),
        _ => None,
    }
}

fn min_json(
    a: Option<serde_json::Value>,
    b: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match (a, b) {
        (Some(a), Some(b)) => match json_cmp(&a, &b) {
            Some(std::cmp::Ordering::Greater) => Some(b),
            Some(_) => Some(a),
            None => Some(a),
        },
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

/// JSON-aware maximum. Returns `None` only when both inputs are `None`.
fn max_json(
    a: Option<serde_json::Value>,
    b: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    match (a, b) {
        (Some(a), Some(b)) => match json_cmp(&a, &b) {
            Some(std::cmp::Ordering::Less) => Some(b),
            Some(_) => Some(a),
            None => Some(a),
        },
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

pub(crate) fn compute_min_max(
    array: &ArrayRef,
    path: &str,
) -> Result<(Option<serde_json::Value>, Option<serde_json::Value>)> {
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        LargeStringArray, StringArray, TimestampMicrosecondArray, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };
    use arrow::compute;

    fn downcast_array<'a, T: Array + 'static>(array: &'a ArrayRef, path: &str) -> Result<&'a T> {
        array
            .as_any()
            .downcast_ref::<T>()
            .ok_or_else(|| IcefallDBError::SchemaMismatch {
                column: "array type".into(),
                expected: std::any::type_name::<T>().into(),
                path: path.into(),
            })
    }

    /// Serialize a JSON value and parse it back so that floating-point
    /// numbers use the exact textual representation that serde_json will write
    /// when the metadata file is persisted. This makes checksum verification
    /// stable across JSON round-trips.
    fn canonicalize_json_value(v: serde_json::Value) -> serde_json::Value {
        let s = serde_json::to_string(&v).expect("infallible JSON serialization");
        serde_json::from_str(&s).expect("infallible JSON parsing")
    }

    /// Fallback for float arrays that contain non-finite values. The Arrow
    /// compute kernels return NaN when present, so we scan explicitly to match
    /// the historical behavior of ignoring NaN/Inf when computing statistics.
    fn finite_float_min_max(
        arr: &Float64Array,
    ) -> (Option<serde_json::Value>, Option<serde_json::Value>) {
        let values: Vec<_> = (0..arr.len())
            .filter(|&i| arr.is_valid(i))
            .map(|i| arr.value(i))
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            return (None, None);
        }
        let min = values.iter().min_by(|a, b| a.total_cmp(b));
        let max = values.iter().max_by(|a, b| a.total_cmp(b));
        (
            min.copied().map(Into::into).map(canonicalize_json_value),
            max.copied().map(Into::into).map(canonicalize_json_value),
        )
    }

    macro_rules! primitive_min_max {
        ($arr_ty:ty, $conv:expr) => {{
            let arr = downcast_array::<$arr_ty>(array, path)?;
            let min = compute::min(arr).map($conv);
            let max = compute::max(arr).map($conv);
            Ok((min, max))
        }};
    }

    match array.data_type() {
        DataType::Boolean => {
            let arr = downcast_array::<BooleanArray>(array, path)?;
            Ok((
                compute::min_boolean(arr).map(Into::into),
                compute::max_boolean(arr).map(Into::into),
            ))
        }
        DataType::Int8 => primitive_min_max!(Int8Array, |v: i8| v.into()),
        DataType::Int16 => primitive_min_max!(Int16Array, |v: i16| v.into()),
        DataType::Int32 => primitive_min_max!(Int32Array, |v: i32| v.into()),
        DataType::Int64 => primitive_min_max!(Int64Array, |v: i64| v.into()),
        DataType::UInt8 => primitive_min_max!(UInt8Array, |v: u8| v.into()),
        DataType::UInt16 => primitive_min_max!(UInt16Array, |v: u16| v.into()),
        DataType::UInt32 => primitive_min_max!(UInt32Array, |v: u32| v.into()),
        DataType::UInt64 => primitive_min_max!(UInt64Array, |v: u64| v.into()),
        DataType::Float32 => {
            let arr = downcast_array::<Float32Array>(array, path)?;
            let values: Vec<_> = (0..arr.len())
                .filter(|&i| arr.is_valid(i))
                .map(|i| arr.value(i) as f64)
                .filter(|v| v.is_finite())
                .collect();
            if values.is_empty() {
                return Ok((None, None));
            }
            let min = values.iter().min_by(|a, b| a.total_cmp(b));
            let max = values.iter().max_by(|a, b| a.total_cmp(b));
            Ok((
                min.copied().map(Into::into).map(canonicalize_json_value),
                max.copied().map(Into::into).map(canonicalize_json_value),
            ))
        }
        DataType::Float64 => {
            let arr = downcast_array::<Float64Array>(array, path)?;
            let min = compute::min(arr);
            let max = compute::max(arr);
            if min.is_some_and(|v| !v.is_finite()) || max.is_some_and(|v| !v.is_finite()) {
                Ok(finite_float_min_max(arr))
            } else {
                Ok((
                    min.map(Into::into).map(canonicalize_json_value),
                    max.map(Into::into).map(canonicalize_json_value),
                ))
            }
        }
        DataType::Utf8 => {
            let arr = downcast_array::<StringArray>(array, path)?;
            Ok((
                compute::min_string(arr).map(Into::into),
                compute::max_string(arr).map(Into::into),
            ))
        }
        DataType::LargeUtf8 => {
            let arr = downcast_array::<LargeStringArray>(array, path)?;
            Ok((
                compute::min_string(arr).map(Into::into),
                compute::max_string(arr).map(Into::into),
            ))
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let arr = downcast_array::<TimestampMicrosecondArray>(array, path)?;
            Ok((
                compute::min(arr).map(Into::into),
                compute::max(arr).map(Into::into),
            ))
        }
        // Complex and binary types are supported by IcefallDB but do not have
        // min/max statistics computed in v1.
        _other => Ok((None, None)),
    }
}

/// Computes the partition values for a row group batch.
///
/// For each partition column, the distinct non-null values are collected. If
/// every partition column has exactly one distinct value, a map of
/// column-name -> value is returned. Otherwise `None` is returned, indicating
/// that partition values should be omitted for this row group.
pub(crate) fn compute_partition_values(
    batch: &RecordBatch,
    partition_by: &[String],
    table_path: &str,
) -> Result<Option<HashMap<String, serde_json::Value>>> {
    let mut result = HashMap::with_capacity(partition_by.len());
    for col_name in partition_by {
        let array =
            batch
                .column_by_name(col_name)
                .ok_or_else(|| IcefallDBError::SchemaMismatch {
                    column: col_name.clone(),
                    expected: "found in batch".into(),
                    path: table_path.into(),
                })?;
        let distinct = distinct_non_null_values(array, table_path)?;
        if distinct.len() != 1 {
            return Ok(None);
        }
        result.insert(col_name.clone(), distinct.into_iter().next().unwrap());
    }
    if result.is_empty() {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

/// Returns the distinct non-null scalar values in an array as JSON values.
fn distinct_non_null_values(
    array: &ArrayRef,
    table_path: &str,
) -> Result<HashSet<serde_json::Value>> {
    use arrow::array::{
        BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
        LargeStringArray, StringArray, TimestampMicrosecondArray, UInt16Array, UInt32Array,
        UInt64Array, UInt8Array,
    };

    let mut values = HashSet::new();
    for i in 0..array.len() {
        if !array.is_valid(i) {
            continue;
        }
        let value = match array.data_type() {
            DataType::Boolean => {
                let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                arr.value(i).into()
            }
            DataType::Int8 => {
                let arr = array.as_any().downcast_ref::<Int8Array>().unwrap();
                (arr.value(i) as i64).into()
            }
            DataType::Int16 => {
                let arr = array.as_any().downcast_ref::<Int16Array>().unwrap();
                (arr.value(i) as i64).into()
            }
            DataType::Int32 => {
                let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
                (arr.value(i) as i64).into()
            }
            DataType::Int64 => {
                let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
                arr.value(i).into()
            }
            DataType::UInt8 => {
                let arr = array.as_any().downcast_ref::<UInt8Array>().unwrap();
                arr.value(i).into()
            }
            DataType::UInt16 => {
                let arr = array.as_any().downcast_ref::<UInt16Array>().unwrap();
                arr.value(i).into()
            }
            DataType::UInt32 => {
                let arr = array.as_any().downcast_ref::<UInt32Array>().unwrap();
                arr.value(i).into()
            }
            DataType::UInt64 => {
                let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
                arr.value(i).into()
            }
            DataType::Float32 => {
                let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
                let v = arr.value(i);
                if !v.is_finite() {
                    continue;
                }
                (v as f64).into()
            }
            DataType::Float64 => {
                let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
                let v = arr.value(i);
                if !v.is_finite() {
                    continue;
                }
                v.into()
            }
            DataType::Utf8 => {
                let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                arr.value(i).into()
            }
            DataType::LargeUtf8 => {
                let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                arr.value(i).into()
            }
            DataType::Timestamp(TimeUnit::Microsecond, None) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                arr.value(i).into()
            }
            _ => {
                return Err(IcefallDBError::SchemaMismatch {
                    column: "arrow type".into(),
                    expected: "supported for partition values".into(),
                    path: table_path.into(),
                })
            }
        };
        values.insert(value);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Column;
    use crate::reader::Reader;
    use crate::storage::memory::MemoryStorage;
    use crate::storage::{LockGuard, Storage};
    use arrow::array::{BooleanArray, Int64Array, TimestampMicrosecondArray};
    use async_trait::async_trait;
    use futures::stream::StreamExt;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn test_successive_appends_produce_disjoint_row_ids_and_monotonic_fragment_ids() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "rng_test";
        let schema = make_int_schema(1000, 64 * 1024 * 1024);

        // First append: 5 rows.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3, 4, 5]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Read back first manifest to capture its row-group entry.
        let (seq1, manifest1) = writer.load_current_manifest().await.unwrap();
        assert_eq!(seq1, 1);
        assert_eq!(manifest1.row_groups.len(), 1);
        let entry1 = &manifest1.row_groups[0];
        // fragment_id for first fragment must be 0 (allocated from HWM 0).
        assert_eq!(entry1.fragment_id, 0);
        // HWM must have advanced past the 5 rows.
        assert_eq!(manifest1.next_row_id, 5);
        assert_eq!(manifest1.next_fragment_id, 1);

        // Verify the meta file carries the row_ids segment.
        let meta1_bytes = storage
            .read(&format!("{}/{}", table, entry1.meta))
            .await
            .unwrap();
        let meta1: RowGroupMeta = serde_json::from_slice(&meta1_bytes).unwrap();
        assert_eq!(meta1.row_ids.len(), 1);
        assert_eq!(
            meta1.row_ids[0],
            crate::rowid::RowIdSegment::Range { start: 0, count: 5 }
        );

        // Second append: 3 rows.
        let mut writer2 = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer2
            .insert_batch(make_int_batch(vec![6, 7, 8]))
            .await
            .unwrap();
        writer2.commit().await.unwrap();

        let (seq2, manifest2) = writer2.load_current_manifest().await.unwrap();
        assert_eq!(seq2, 2);
        assert_eq!(manifest2.row_groups.len(), 2);
        let entry2 = &manifest2.row_groups[1];
        // fragment_id for second fragment must be 1 (allocated from HWM 1).
        assert_eq!(entry2.fragment_id, 1);
        // HWM must have advanced past all 8 rows.
        assert_eq!(manifest2.next_row_id, 8);
        assert_eq!(manifest2.next_fragment_id, 2);

        // Verify the second meta file carries the correct disjoint row_ids segment.
        let meta2_bytes = storage
            .read(&format!("{}/{}", table, entry2.meta))
            .await
            .unwrap();
        let meta2: RowGroupMeta = serde_json::from_slice(&meta2_bytes).unwrap();
        assert_eq!(meta2.row_ids.len(), 1);
        assert_eq!(
            meta2.row_ids[0],
            crate::rowid::RowIdSegment::Range { start: 5, count: 3 }
        );
    }

    /// Sequence: commit(rows_a) → insert_parquet(rows_b) → commit(rows_c)
    ///
    /// Verifies that the fast-path insert_parquet correctly allocates row IDs
    /// and carries the high-water marks into the manifest so that all three
    /// ingest operations produce disjoint, contiguous row-id ranges and
    /// monotonically increasing fragment IDs.
    #[tokio::test]
    async fn test_commit_insert_parquet_commit_produces_disjoint_row_ids() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "mixed_ingest_test";

        // Schema shared by all three ingests.
        let schema = make_int_schema(1000, 64 * 1024 * 1024);

        // ── Phase 1: commit(rows_a=4) ──────────────────────────────────────
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![10, 20, 30, 40]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let (seq1, manifest1) = writer.load_current_manifest().await.unwrap();
        assert_eq!(seq1, 1, "first manifest must be sequence 1");
        assert_eq!(manifest1.row_groups.len(), 1);
        let entry_a = &manifest1.row_groups[0];
        assert_eq!(entry_a.fragment_id, 0, "first fragment_id must be 0");
        assert_eq!(manifest1.next_row_id, 4, "HWM after 4 rows");
        assert_eq!(manifest1.next_fragment_id, 1, "HWM after 1 fragment");

        let meta_a_bytes = storage
            .read(&format!("{}/{}", table, entry_a.meta))
            .await
            .unwrap();
        let meta_a: RowGroupMeta = serde_json::from_slice(&meta_a_bytes).unwrap();
        assert_eq!(
            meta_a.row_ids,
            vec![crate::rowid::RowIdSegment::Range { start: 0, count: 4 }],
            "fragment A row_ids"
        );

        // ── Phase 2: insert_parquet(rows_b=3) via fast path ───────────────
        // Build a minimal single-row-group Parquet file on disk.
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("phase2.parquet");
        {
            let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "id",
                DataType::Int64,
                false,
            )]));
            let batch = RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![Arc::new(Int64Array::from(vec![50i64, 60, 70]))],
            )
            .unwrap();
            write_parquet_file(&parquet_path, &batch);
        }

        // Use a fresh Writer instance to prove the HWM is read from disk.
        let mut writer2 = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        let outcome = writer2
            .insert_parquet(parquet_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(
            matches!(outcome, InsertParquetOutcome::FastPath { rows: 3 }),
            "expected FastPath{{rows:3}}, got {:?}",
            outcome
        );

        let (seq2, manifest2) = writer2.load_current_manifest().await.unwrap();
        assert_eq!(seq2, 2, "second manifest must be sequence 2");
        assert_eq!(manifest2.row_groups.len(), 2);
        let entry_b = &manifest2.row_groups[1];
        assert_eq!(entry_b.fragment_id, 1, "second fragment_id must be 1");
        assert_eq!(manifest2.next_row_id, 7, "HWM after 4+3 rows");
        assert_eq!(manifest2.next_fragment_id, 2, "HWM after 2 fragments");

        let meta_b_bytes = storage
            .read(&format!("{}/{}", table, entry_b.meta))
            .await
            .unwrap();
        let meta_b: RowGroupMeta = serde_json::from_slice(&meta_b_bytes).unwrap();
        assert_eq!(
            meta_b.row_ids,
            vec![crate::rowid::RowIdSegment::Range { start: 4, count: 3 }],
            "fragment B row_ids must start where A ended"
        );

        // ── Phase 3: commit(rows_c=5) ──────────────────────────────────────
        let mut writer3 = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer3
            .insert_batch(make_int_batch(vec![80, 90, 100, 110, 120]))
            .await
            .unwrap();
        writer3.commit().await.unwrap();

        let (seq3, manifest3) = writer3.load_current_manifest().await.unwrap();
        assert_eq!(seq3, 3, "third manifest must be sequence 3");
        assert_eq!(manifest3.row_groups.len(), 3);
        let entry_c = &manifest3.row_groups[2];
        assert_eq!(entry_c.fragment_id, 2, "third fragment_id must be 2");
        // (b) fragment_ids are strictly increasing: 0, 1, 2
        assert_eq!(manifest3.row_groups[0].fragment_id, 0);
        assert_eq!(manifest3.row_groups[1].fragment_id, 1);
        assert_eq!(manifest3.row_groups[2].fragment_id, 2);
        // (c) final HWMs: 4+3+5=12 rows, 3 fragments
        assert_eq!(
            manifest3.next_row_id, 12,
            "final next_row_id == rows_a+rows_b+rows_c"
        );
        assert_eq!(manifest3.next_fragment_id, 3, "final next_fragment_id == 3");

        let meta_c_bytes = storage
            .read(&format!("{}/{}", table, entry_c.meta))
            .await
            .unwrap();
        let meta_c: RowGroupMeta = serde_json::from_slice(&meta_c_bytes).unwrap();
        assert_eq!(
            meta_c.row_ids,
            vec![crate::rowid::RowIdSegment::Range { start: 7, count: 5 }],
            "fragment C row_ids must start where B ended"
        );

        // (a) all three ranges are disjoint and contiguous
        // A: [0,4), B: [4,7), C: [7,12)
        let all_ids: Vec<crate::rowid::RowIdSegment> = [&meta_a, &meta_b, &meta_c]
            .iter()
            .flat_map(|m| m.row_ids.iter().cloned())
            .collect();
        let mut prev_end = 0u64;
        for seg in &all_ids {
            if let crate::rowid::RowIdSegment::Range { start, count } = seg {
                assert_eq!(
                    *start, prev_end,
                    "ranges must be contiguous: expected start={prev_end}, got {start}"
                );
                prev_end = start + count;
            }
        }
        assert_eq!(
            prev_end, 12,
            "total covered row IDs must equal rows_a+rows_b+rows_c"
        );
    }

    fn make_int_schema(row_group_target_rows: usize, row_group_target_bytes: usize) -> Schema {
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
            row_group_target_rows,
            row_group_target_bytes,
            max_field_id: 0,
            dropped_columns: vec![],
        };
        schema.assign_field_ids(None);
        schema
    }

    fn make_int_batch(ids: Vec<i64>) -> RecordBatch {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
        let array = Int64Array::from(ids);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
    }

    #[tokio::test]
    async fn test_recovery_deletes_stale_intent_and_unreferenced_final_files() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";

        // Seed a stale intent left by a crashed writer. The intent references
        // final table-root filenames that are not present in the current
        // manifest, so recovery must delete them. Also create an unrelated final
        // file; recovery must leave it alone because garbage collection of
        // uncommitted final row-group files is the responsibility of `icefalldb gc`.
        let stale_data = "rg_stale.parquet";
        let stale_meta = "rg_stale.meta";
        let unrelated_final = "rg_unrelated.parquet";
        storage
            .write(&format!("{}/{}", table, stale_data), b"stale-data")
            .await
            .unwrap();
        storage
            .write(&format!("{}/{}", table, stale_meta), b"stale-meta")
            .await
            .unwrap();
        storage
            .write(&format!("{}/{}", table, unrelated_final), b"unrelated-data")
            .await
            .unwrap();
        let intent = serde_json::json!({
            "txn_id": "txn_stale",
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": 1,
            "files": [stale_data, stale_meta],
        });
        storage
            .write(
                &format!("{}/_staging/intents/txn_stale.json", table),
                serde_json::to_vec(&intent).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // Constructing a writer must NOT recover stale intents.
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, table, schema)
            .await
            .unwrap();
        assert!(storage
            .exists(&format!("{}/{}", table, stale_data))
            .await
            .unwrap());
        assert!(storage
            .exists(&format!("{}/{}", table, stale_meta))
            .await
            .unwrap());
        assert!(storage
            .exists(&format!("{}/{}", table, unrelated_final))
            .await
            .unwrap());
        assert!(storage
            .exists(&format!("{}/_staging/intents/txn_stale.json", table))
            .await
            .unwrap());

        // Recovery runs under the writer lock during commit.
        writer
            .insert_batch(make_int_batch(vec![1, 2]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Unreferenced final files listed in the intent and the intent itself
        // must be gone.
        assert!(!storage
            .exists(&format!("{}/{}", table, stale_data))
            .await
            .unwrap());
        assert!(!storage
            .exists(&format!("{}/{}", table, stale_meta))
            .await
            .unwrap());
        assert!(!storage
            .exists(&format!("{}/_staging/intents/txn_stale.json", table))
            .await
            .unwrap());

        // Unreferenced final table-root files must be preserved during recovery;
        // they are cleaned up by `icefalldb gc`, not by commit-time recovery.
        assert!(storage
            .exists(&format!("{}/{}", table, unrelated_final))
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_recovery_preserves_unreferenced_final_row_group_files() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        // Commit sequence 1 normally so the table has a valid manifest.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Create final row-group files that are not referenced by the current manifest.
        let orphan_data = "rg_orphan.parquet";
        let orphan_meta = "rg_orphan.meta";
        storage
            .write(&format!("{}/{}", table, orphan_data), b"orphan-data")
            .await
            .unwrap();
        storage
            .write(&format!("{}/{}", table, orphan_meta), b"orphan-meta")
            .await
            .unwrap();

        // Recovery runs during the next commit.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Unreferenced final row-group files must be preserved; they are cleaned
        // up by `icefalldb gc`, not by commit-time recovery.
        assert!(storage
            .exists(&format!("{}/{}", table, orphan_data))
            .await
            .unwrap());
        assert!(storage
            .exists(&format!("{}/{}", table, orphan_meta))
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_commit_does_not_leave_part_files() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(2, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        writer
            .insert_batch(make_int_batch(vec![1, 2]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Final files must exist.
        let entries = storage.list("products/").await.unwrap();
        let parquet_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".parquet")).collect();
        assert_eq!(parquet_files.len(), 1);
        let meta_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".meta")).collect();
        assert_eq!(meta_files.len(), 1);

        // No .part files should remain after commit.
        let part_files: Vec<_> = entries.iter().filter(|p| p.ends_with(".part")).collect();
        assert!(
            part_files.is_empty(),
            "part files should be renamed away: {:?}",
            part_files
        );
    }

    #[tokio::test]
    async fn test_commit_fails_on_corrupt_latest_manifest() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        // Commit two sequences.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Corrupt the latest manifest while leaving the pointer intact.
        let manifest_path = format!("products/{}", Manifest::filename(2));
        let mut manifest: Manifest =
            serde_json::from_slice(&storage.read(&manifest_path).await.unwrap()).unwrap();
        manifest.row_groups.push(RowGroupEntry {
            data: "tampered.parquet".into(),
            meta: "tampered.meta".into(),
            ..Default::default()
        });
        storage
            .write(
                &manifest_path,
                serde_json::to_vec(&manifest).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // Create a newer manifest snapshot that is newer than the valid pointer.
        // Because the commit fails before recovery can run, this newer manifest
        // must not be deleted.
        let newer_manifest_path = format!("products/{}", Manifest::filename(3));
        storage.write(&newer_manifest_path, b"{}").await.unwrap();

        // The next commit must fail because the valid pointer references a
        // corrupt manifest. It must not silently downgrade to an older manifest.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            make_int_schema(10, 1024 * 1024),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![7, 8, 9]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::ChecksumMismatch { ref path }) if path == &manifest_path
            ),
            "expected ChecksumMismatch for corrupt latest manifest, got {:?}",
            result
        );

        // The pointer and corrupt manifest must remain unchanged so an operator
        // can diagnose and repair the corruption.
        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));
        assert!(storage.exists(&manifest_path).await.unwrap());

        // Recovery never ran, so the newer manifest snapshot must not have been
        // deleted.
        assert!(
            storage.exists(&newer_manifest_path).await.unwrap(),
            "newer manifest must not be deleted when commit fails on corrupt latest"
        );
    }

    #[tokio::test]
    async fn test_writer_new_and_commit_fail_on_corrupt_latest_manifest() {
        // Writer construction itself does not load the manifest, but the first
        // commit after construction must fail when the valid pointer references
        // a corrupt manifest.
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest_path = format!("products/{}", Manifest::filename(1));
        let mut manifest: Manifest =
            serde_json::from_slice(&storage.read(&manifest_path).await.unwrap()).unwrap();
        manifest.row_groups.push(RowGroupEntry {
            data: "tampered.parquet".into(),
            meta: "tampered.meta".into(),
            ..Default::default()
        });
        storage
            .write(
                &manifest_path,
                serde_json::to_vec(&manifest).unwrap().as_slice(),
            )
            .await
            .unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::ChecksumMismatch { ref path }) if path == &manifest_path
            ),
            "expected ChecksumMismatch for corrupt latest manifest, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_load_current_manifest_distinguishes_checksum_and_parse_errors() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest_path = format!("products/{}", Manifest::filename(1));

        // Tamper with the manifest contents without recomputing the checksum.
        // The file is valid JSON but its checksum no longer matches, so loading
        // it must report `ChecksumMismatch`.
        let mut manifest: Manifest =
            serde_json::from_slice(&storage.read(&manifest_path).await.unwrap()).unwrap();
        manifest.row_groups.push(RowGroupEntry {
            data: "tampered.parquet".into(),
            meta: "tampered.meta".into(),
            ..Default::default()
        });
        storage
            .write(
                &manifest_path,
                serde_json::to_vec(&manifest).unwrap().as_slice(),
            )
            .await
            .unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::ChecksumMismatch { ref path }) if path == &manifest_path
            ),
            "expected ChecksumMismatch for tampered manifest, got {:?}",
            result
        );

        // Now replace the manifest with outright invalid JSON. The error must be
        // a `Serialization` error, not a checksum mismatch.
        storage
            .write(&manifest_path, b"not valid json")
            .await
            .unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![7, 8, 9]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(
            matches!(result, Err(IcefallDBError::Serialization(_))),
            "expected Serialization error for malformed manifest JSON, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_row_group_target_bytes_split() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let batch = make_int_batch(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let total_rows = batch.num_rows();
        let total_bytes = batch.get_array_memory_size();
        // Pick a target smaller than the total batch but larger than a single
        // row so the planner must produce multiple row groups.
        let target_bytes = total_bytes / 3 + 1;
        let schema = make_int_schema(total_rows * 2, target_bytes);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let manifest_data = storage
            .read(&format!("products/{}", Manifest::filename(1)))
            .await
            .unwrap();
        let manifest: Manifest = serde_json::from_slice(&manifest_data).unwrap();
        assert!(
            manifest.row_groups.len() > 1,
            "expected multiple row groups due to byte target, got {}",
            manifest.row_groups.len()
        );

        // Each row group must fit under the byte target using the same per-row
        // average estimate the planner uses.
        for entry in &manifest.row_groups {
            let meta_data = storage
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap();
            let meta: RowGroupMeta = serde_json::from_slice(&meta_data).unwrap();
            let estimated_bytes = meta.rows * total_bytes / total_rows;
            assert!(
                estimated_bytes <= target_bytes,
                "row group {} estimated {} bytes for {} rows, target {}",
                entry.data,
                estimated_bytes,
                meta.rows,
                target_bytes
            );
        }
    }

    #[tokio::test]
    async fn test_insert_batch_rejects_schema_mismatch() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        // Same column type but different field name; valid RecordBatch, but
        // schema does not match the writer's registered schema.
        let bad_schema = ArrowSchema::new(vec![Field::new("different_id", DataType::Int64, false)]);
        let bad_batch = RecordBatch::try_new(
            Arc::new(bad_schema),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let result = writer.insert_batch(bad_batch).await;
        assert!(
            matches!(result, Err(IcefallDBError::SchemaMismatch { .. })),
            "expected SchemaMismatch, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_insert_batch_accepts_extra_arrow_metadata() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        // A batch whose schema carries extra Arrow metadata should be accepted as
        // long as the field names, types, and nullability match the writer schema.
        let mut metadata = HashMap::new();
        metadata.insert("extra".to_string(), "value".to_string());
        let schema_with_metadata = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)])
            .with_metadata(metadata);
        let batch = RecordBatch::try_new(
            Arc::new(schema_with_metadata),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.row_groups.len(), 1);
    }

    #[tokio::test]
    async fn test_writer_rejects_schema_pointer_downgrade() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer.insert_batch(make_int_batch(vec![1])).await.unwrap();
        writer.commit().await.unwrap();

        // Point _schema.json at a different schema id. Opening a writer with a
        // lower schema id must fail rather than silently rewrite the pointer.
        storage
            .write("products/_schema.json", b"{\"latest\": 2}")
            .await
            .unwrap();
        let mut lower_schema = schema.clone();
        lower_schema.schema_id = 1;
        let result = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            lower_schema,
        )
        .await;
        assert!(
            matches!(result, Err(IcefallDBError::SchemaMismatch { .. })),
            "expected SchemaMismatch for pointer downgrade, got {:?}",
            result.as_ref().map(|_| ())
        );

        // _schema.json must not have been rewritten.
        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_schema.json").await.unwrap()).unwrap();
        assert_eq!(pointer["latest"].as_u64(), Some(2));
    }

    #[tokio::test]
    async fn test_writer_rejects_missing_schema_file_when_pointer_exists() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer.insert_batch(make_int_batch(vec![1])).await.unwrap();
        writer.commit().await.unwrap();

        // Remove the schema file but keep the pointer intact. Opening a writer
        // with the same schema id must fail because the schema file is missing.
        storage
            .delete(&format!("products/{}", Schema::filename(schema.schema_id)))
            .await
            .unwrap();

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(result, Err(IcefallDBError::SchemaNotFound { .. })),
            "expected SchemaNotFound when schema file is missing, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_commit_fails_when_schema_pointer_changes() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();

        // Simulate a schema pointer change between Writer::new and commit.
        storage
            .write("products/_schema.json", b"{\"latest\": 2}")
            .await
            .unwrap();

        let result = writer.commit().await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::SchemaMismatch {
                    ref column,
                    ref expected,
                    ref path,
                })
                if column == "schema_id"
                    && expected == "1"
                    && path == "products/_schema.json"
            ),
            "expected SchemaMismatch after schema pointer change, got {:?}",
            result
        );
    }

    #[tokio::test]
    async fn test_insert_batch_ignores_empty_batch() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        // An empty batch must be a no-op and must not advance the sequence.
        let empty = make_int_batch(vec![]);
        writer.insert_batch(empty).await.unwrap();
        writer.commit().await.unwrap();
        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(0));
    }

    #[tokio::test]
    async fn test_malformed_manifest_pointer_is_repaired_from_manifests() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Each malformed pointer should be repaired from the manifest directory,
        // allowing the commit to advance the sequence. `{"latest": 0}` is valid
        // for an empty table and is tested separately.
        for pointer in [r#"{}"#, r#"{"latest": "one"}"#] {
            storage
                .write("products/_manifest.json", pointer.as_bytes())
                .await
                .unwrap();

            let mut writer = Writer::new(
                Arc::clone(&storage) as Arc<dyn Storage>,
                "products",
                schema.clone(),
            )
            .await
            .unwrap();
            writer
                .insert_batch(make_int_batch(vec![4, 5, 6]))
                .await
                .unwrap();
            writer.commit().await.unwrap();
        }

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(3));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(3)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 3);
    }

    #[tokio::test]
    async fn test_missing_manifest_pointer_is_repaired_from_manifests() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Delete the pointer. The next commit must recreate it from the
        // manifest directory.
        storage.delete("products/_manifest.json").await.unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(2)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 2);
    }

    #[tokio::test]
    async fn test_writer_rejects_mismatched_existing_schema() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer.insert_batch(make_int_batch(vec![1])).await.unwrap();
        writer.commit().await.unwrap();

        let mut different_schema = schema.clone();
        different_schema.columns.push(Column {
            name: "extra".into(),
            r#type: "int64".into(),
            nullable: true,
            field_id: 0,
        });

        let result = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            different_schema,
        )
        .await;
        assert!(
            matches!(result, Err(IcefallDBError::SchemaMismatch { .. })),
            "expected SchemaMismatch, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_boolean_min_max() {
        let schema = ArrowSchema::new(vec![Field::new("flag", DataType::Boolean, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(BooleanArray::from(vec![true, false, true]))],
        )
        .unwrap();

        let icefalldb_schema = Schema {
            schema_id: 1,
            columns: vec![Column {
                name: "flag".into(),
                r#type: "bool".into(),
                nullable: false,
                field_id: 0,
            }],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            icefalldb_schema,
        )
        .await
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        let entry = &manifest.row_groups[0];
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        let stats = meta.columns.get("flag").unwrap();
        assert_eq!(stats.min, Some(false.into()));
        assert_eq!(stats.max, Some(true.into()));
    }

    #[tokio::test]
    async fn test_replace_drops_existing_row_groups() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.replace().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        assert_eq!(seq, 2);

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.row_groups.len(), 1);

        let reader = Reader::new(storage.as_ref(), "products").await.unwrap();
        let plan = reader.scan().await.unwrap();
        let mut ids = Vec::new();
        for rg in &plan.row_groups {
            let mut stream = reader.read_row_group(rg).await.unwrap();
            while let Some(batch) = stream.next().await {
                let batch = batch.unwrap();
                let col = batch.column_by_name("id").unwrap();
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..arr.len() {
                    ids.push(arr.value(i));
                }
            }
        }
        assert_eq!(ids, vec![4, 5, 6]);
    }

    #[tokio::test]
    async fn test_replace_with_empty_buffer_creates_empty_snapshot() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer.replace().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer["latest"].as_u64(), Some(2));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(2)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.row_groups.is_empty());
    }

    #[tokio::test]
    async fn test_timestamp_column_round_trip() {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(TimestampMicrosecondArray::from(vec![
                1_000_000, 2_000_000, 3_000_000,
            ]))],
        )
        .unwrap();

        let icefalldb_schema = Schema {
            schema_id: 1,
            columns: vec![Column {
                name: "ts".into(),
                r#type: "timestamp[us]".into(),
                nullable: false,
                field_id: 0,
            }],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "events",
            icefalldb_schema,
        )
        .await
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("events/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        let entry = &manifest.row_groups[0];
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .read(&format!("events/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        let stats = meta.columns.get("ts").unwrap();
        assert_eq!(stats.min, Some(1_000_000i64.into()));
        assert_eq!(stats.max, Some(3_000_000i64.into()));
    }

    #[test]
    fn test_compute_min_max_returns_none_for_unsupported_type() {
        use arrow::array::BinaryArray;
        let array: ArrayRef = Arc::new(BinaryArray::from(vec![b"a".as_slice(), b"b", b"c"]));
        let (min, max) = compute_min_max(&array, "test").unwrap();
        assert!(min.is_none());
        assert!(max.is_none());
    }

    /// Storage wrapper that fails any operation whose path contains a target
    /// substring once failures have been enabled. Used to inject failures at
    /// specific commit stages while allowing an initial setup commit to succeed.
    #[derive(Debug)]
    struct FailingStorage {
        inner: MemoryStorage,
        fail_contains: String,
        enabled: std::sync::atomic::AtomicBool,
    }

    impl FailingStorage {
        fn new(fail_contains: impl Into<String>) -> Self {
            Self {
                inner: MemoryStorage::new(),
                fail_contains: fail_contains.into(),
                enabled: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn enable_failures(&self) {
            self.enabled
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        fn disable_failures(&self) {
            self.enabled
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    fn injected_error(path: &str) -> IcefallDBError {
        IcefallDBError::Other(Box::new(std::io::Error::other(format!(
            "injected failure for {}",
            path
        ))))
    }

    #[async_trait]
    impl Storage for FailingStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
            self.inner.read(path).await
        }

        async fn size(&self, path: &str) -> Result<u64> {
            self.inner.size(path).await
        }

        async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
            self.inner.read_range(path, offset, len).await
        }

        async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
            if self.enabled.load(std::sync::atomic::Ordering::SeqCst)
                && path.contains(&self.fail_contains)
            {
                return Err(injected_error(path));
            }
            self.inner.write(path, data).await
        }

        async fn delete(&self, path: &str) -> Result<()> {
            self.inner.delete(path).await
        }

        async fn rename(&self, from: &str, to: &str) -> Result<()> {
            if self.enabled.load(std::sync::atomic::Ordering::SeqCst)
                && to.contains(&self.fail_contains)
            {
                return Err(injected_error(to));
            }
            self.inner.rename(from, to).await
        }

        async fn list(&self, prefix: &str) -> Result<Vec<String>> {
            self.inner.list(prefix).await
        }

        async fn exists(&self, path: &str) -> Result<bool> {
            self.inner.exists(path).await
        }

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> Result<()> {
            self.inner.sync(path).await
        }
    }

    /// Storage wrapper that fails `delete` operations whose path contains a
    /// target substring once failures have been enabled. Used to verify that
    /// rollback cleans up final files even when intent deletion fails.
    #[derive(Debug)]
    struct DeleteFailingStorage {
        inner: FailingStorage,
        fail_contains: String,
        enabled: std::sync::atomic::AtomicBool,
    }

    impl DeleteFailingStorage {
        fn new(inner: FailingStorage, fail_contains: impl Into<String>) -> Self {
            Self {
                inner,
                fail_contains: fail_contains.into(),
                enabled: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn enable_delete_failures(&self) {
            self.enabled
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        fn enable_write_failures(&self) {
            self.inner.enable_failures();
        }
    }

    #[async_trait]
    impl Storage for DeleteFailingStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
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
            if self.enabled.load(std::sync::atomic::Ordering::SeqCst)
                && path.contains(&self.fail_contains)
            {
                return Err(injected_error(path));
            }
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

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> Result<()> {
            self.inner.sync(path).await
        }
    }

    /// Storage wrapper that fails the table-root `sync` that occurs after the
    /// manifest pointer has been renamed. Simulates a crash where the pointer
    /// rename is visible but the final fsync of the table root returns an error.
    #[derive(Debug)]
    struct PostPointerSyncFailingStorage {
        inner: MemoryStorage,
        table: String,
        enabled: std::sync::atomic::AtomicBool,
        pointer_renames: std::sync::atomic::AtomicUsize,
    }

    impl PostPointerSyncFailingStorage {
        fn new(table: impl Into<String>) -> Self {
            Self {
                inner: MemoryStorage::new(),
                table: table.into(),
                enabled: std::sync::atomic::AtomicBool::new(false),
                pointer_renames: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn enable_failures(&self) {
            self.enabled
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }

        fn is_table_root_sync(&self, path: &str) -> bool {
            path == format!("{}/", self.table)
        }
    }

    #[async_trait]
    impl Storage for PostPointerSyncFailingStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
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
            // Only count pointer renames that happen while failures are enabled,
            // so the first setup commit is not affected by the failure injection.
            if to == format!("{}/_manifest.json", self.table)
                && self.enabled.load(std::sync::atomic::Ordering::SeqCst)
            {
                self.pointer_renames
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            self.inner.rename(from, to).await
        }

        async fn list(&self, prefix: &str) -> Result<Vec<String>> {
            self.inner.list(prefix).await
        }

        async fn exists(&self, path: &str) -> Result<bool> {
            self.inner.exists(path).await
        }

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> Result<()> {
            // Fail only the table-root sync after the first pointer rename that
            // occurs while failures are enabled.
            let renames = self
                .pointer_renames
                .load(std::sync::atomic::Ordering::SeqCst);
            if self.enabled.load(std::sync::atomic::Ordering::SeqCst)
                && renames >= 1
                && self.is_table_root_sync(path)
            {
                return Err(injected_error(path));
            }
            self.inner.sync(path).await
        }
    }

    #[tokio::test]
    async fn test_recovery_deletes_orphan_manifest_snapshot() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        // Commit sequence 1 normally.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Simulate a crash that left an orphan manifest with sequence > latest.
        storage
            .write(&format!("{}/{}", table, Manifest::filename(2)), b"{}")
            .await
            .unwrap();

        // The next commit must recover the orphan and succeed.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Pointer should be at sequence 2 and the manifest should be valid.
        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("{}/{}", table, Manifest::filename(2)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.sequence, 2);
    }

    #[tokio::test]
    async fn test_commit_failure_rolls_back_staged_files() {
        // Fail the second manifest snapshot write so the commit fails after
        // files have been renamed to their final locations.
        let storage = Arc::new(FailingStorage::new("products/_manifests/000000002.json"));
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let before = storage.inner.list("products/").await.unwrap();

        storage.enable_failures();

        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(result.is_err(), "expected commit to fail: {:?}", result);

        // No intent, new final data/meta files, or orphan manifest should remain.
        let after = storage.inner.list("products/").await.unwrap();
        let intents: Vec<_> = after
            .iter()
            .filter(|p| p.contains("_staging/intents/"))
            .collect();
        assert!(
            intents.is_empty(),
            "intent should have been rolled back: {:?}",
            intents
        );
        let new_files: Vec<_> = after.iter().filter(|p| !before.contains(p)).collect();
        assert!(
            new_files.is_empty(),
            "no new files should remain after rollback: {:?}",
            new_files
        );
        assert!(!storage
            .inner
            .exists(&format!("products/{}", Manifest::filename(2)))
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn test_rollback_deletes_final_files_when_intent_delete_fails() {
        // Fail the second manifest snapshot write so the commit fails after
        // files have been renamed to their final locations, and also fail
        // deletion of the intent file. Final files must still be cleaned up.
        let inner = FailingStorage::new("products/_manifests/000000002.json");
        let storage = Arc::new(DeleteFailingStorage::new(inner, "_staging/intents/"));
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let before = storage.inner.inner.list("products/").await.unwrap();

        storage.enable_write_failures();
        storage.enable_delete_failures();

        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(result.is_err(), "expected commit to fail: {:?}", result);

        let after = storage.inner.inner.list("products/").await.unwrap();
        let new_data_or_meta: Vec<_> = after
            .iter()
            .filter(|p| !before.contains(p) && (p.ends_with(".parquet") || p.ends_with(".meta")))
            .collect();
        assert!(
            new_data_or_meta.is_empty(),
            "no new final data/meta files should remain after rollback: {:?}",
            new_data_or_meta
        );
        assert!(!storage
            .inner
            .inner
            .exists(&format!("products/{}", Manifest::filename(2)))
            .await
            .unwrap());

        // The intent deletion failed, so the intent file itself should still
        // be present.
        let intents = storage.list("products/_staging/intents/").await.unwrap();
        assert!(
            !intents.is_empty(),
            "intent file should survive the failed deletion: {:?}",
            intents
        );
    }

    #[tokio::test]
    async fn test_buffered_batches_preserved_after_failed_commit() {
        // Fail the second manifest snapshot write so the commit fails before
        // the pointer update. The buffered batch must remain available for a
        // retry.
        let storage = Arc::new(FailingStorage::new("products/_manifests/000000002.json"));
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        storage.enable_failures();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(result.is_err(), "expected commit to fail: {:?}", result);

        // Disable failures and retry the same buffered batch.
        storage.disable_failures();
        writer.commit().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.inner.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", Manifest::filename(2)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 2);
    }

    #[tokio::test]
    async fn test_writer_rejects_invalid_table_paths() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        for table in [
            "",
            "/absolute",
            "foo/../bar",
            "foo/..",
            ".",
            "./foo",
            "foo/./bar",
            "foo/",
            "/foo",
            "foo//bar",
        ] {
            let result = Writer::new(
                Arc::clone(&storage) as Arc<dyn Storage>,
                table,
                schema.clone(),
            )
            .await;
            assert!(
                matches!(result, Err(IcefallDBError::InvalidPath(_))),
                "table {:?} should be rejected",
                table
            );
        }
    }

    #[tokio::test]
    async fn test_writer_rejects_reserved_prefix_table_paths() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        for table in ["_manifests/foo", "_schemas/foo", "_staging/foo"] {
            let result = Writer::new(
                Arc::clone(&storage) as Arc<dyn Storage>,
                table,
                schema.clone(),
            )
            .await;
            assert!(
                matches!(result, Err(IcefallDBError::InvalidPath(_))),
                "reserved prefix table {:?} should be rejected",
                table
            );
        }
    }

    #[tokio::test]
    async fn test_writer_rejects_reserved_table_names() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);

        for table in [
            "_write",
            "_write.lock",
            "_manifest",
            "_manifest.json",
            "_schema",
            "_schema.json",
            "_manifests",
            "_schemas",
            "_staging",
            "views",
        ] {
            let result = Writer::new(
                Arc::clone(&storage) as Arc<dyn Storage>,
                table,
                schema.clone(),
            )
            .await;
            assert!(
                matches!(result, Err(IcefallDBError::InvalidPath(_))),
                "reserved table {:?} should be rejected",
                table
            );
        }
    }

    #[tokio::test]
    async fn test_writer_rejects_empty_schema_columns() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut schema = make_int_schema(10, 1024 * 1024);
        schema.columns.clear();

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "schema must have at least one column" && path == "products"
            ),
            "empty schema columns should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_rejects_zero_row_group_target_rows() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(0, 1024 * 1024);

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "row_group_target_rows must be > 0" && path == "products"
            ),
            "row_group_target_rows == 0 should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_rejects_zero_row_group_target_bytes() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 0);

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "row_group_target_bytes must be > 0" && path == "products"
            ),
            "row_group_target_bytes == 0 should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_rejects_missing_partition_columns() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut schema = make_int_schema(10, 1024 * 1024);
        schema.partition_by = Some(vec!["missing".into()]);

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "partition column 'missing' not found in schema" && path == "products"
            ),
            "missing partition column should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_rejects_empty_column_name() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Schema {
            schema_id: 1,
            columns: vec![Column {
                name: "".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            }],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 10,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "column name must not be empty" && path == "products"
            ),
            "empty column name should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_rejects_duplicate_column_name() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Schema {
            schema_id: 1,
            columns: vec![
                Column {
                    name: "id".into(),
                    r#type: "int64".into(),
                    nullable: false,
                    field_id: 0,
                },
                Column {
                    name: "id".into(),
                    r#type: "utf8".into(),
                    nullable: true,
                    field_id: 0,
                },
            ],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 10,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "duplicate column name 'id'" && path == "products"
            ),
            "duplicate column name should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_rejects_unsupported_column_type() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Schema {
            schema_id: 1,
            columns: vec![Column {
                name: "value".into(),
                r#type: "date32".into(),
                nullable: true,
                field_id: 0,
            }],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let result =
            Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "column 'value' has unsupported type 'date32'" && path == "products"
            ),
            "unsupported column type should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_writer_create_rejects_schema_id_not_one() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut schema = make_int_schema(10, 1024 * 1024);
        schema.schema_id = 2;

        let result =
            Writer::create(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema).await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::InvalidSchema {
                    ref reason,
                    ref path,
                })
                if reason == "new tables must start at schema_id 1" && path == "products"
            ),
            "create with schema_id != 1 should be rejected, got {:?}",
            result.as_ref().map(|_| ())
        );
    }

    #[tokio::test]
    async fn test_arrow_type_accepts_string_alias() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Schema {
            schema_id: 1,
            columns: vec![Column {
                name: "value".into(),
                r#type: "string".into(),
                nullable: true,
                field_id: 0,
            }],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        // Writer construction should succeed for the documented "string" alias.
        let _writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_post_pointer_sync_failure_returns_err_and_commit_is_durable() {
        // Fail the table-root `sync` that occurs after `_manifest.json` has been
        // renamed. The commit must report the error, but because the pointer was
        // already updated the commit is durable and must not be rolled back.
        let storage = Arc::new(PostPointerSyncFailingStorage::new("products"));
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        storage.enable_failures();

        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        let result = writer.commit().await;
        assert!(
            result.is_err(),
            "commit must report the post-pointer fsync error: {:?}",
            result
        );

        // The pointer rename already happened, so the commit is durable. The
        // pointer and manifest must not have been rolled back.
        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.inner.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", Manifest::filename(2)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 2);

        // Because the error happened after the pointer was updated, the writer
        // must have cleared its buffer so the caller cannot accidentally
        // duplicate the commit.
        assert!(writer.buffer.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_preserves_committed_files_after_pointer_update_crash() {
        // Simulate the critical crash window: the manifest pointer has been
        // updated to sequence 2, but the intent for that commit (which records
        // the staged .part files) was not deleted before the writer crashed.
        //
        // Because the intent references .part files, recovery must not delete
        // the already-committed final files for sequence 2.
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        // Commit sequence 1 normally.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Simulate sequence 2 having committed successfully but its intent
        // surviving the crash. The intent now references the final table-root
        // filenames, so recovery must consult the current manifest before
        // deleting any of them.
        let committed_data = "rg_committed.parquet";
        let committed_meta = "rg_committed.meta";
        storage
            .write(&format!("{}/{}", table, committed_data), b"committed-data")
            .await
            .unwrap();
        // The checkpoint builder reads existing .meta sidecars, so the fake meta
        // file must be valid JSON even though its contents are otherwise ignored
        // by this recovery test.
        let fake_meta = RowGroupMeta {
            row_group: "rg_committed".into(),
            schema_id: schema.schema_id,
            rows: 0,
            columns: std::collections::HashMap::new(),
            column_offsets: None,
            sort: None,
            row_ids: vec![],
            checksum: "sha256:0000000000000000000000000000000000000000000000000000000000000000"
                .into(),
            meta_checksum: String::new(),
        };
        storage
            .write(
                &format!("{}/{}", table, committed_meta),
                serde_json::to_vec(&fake_meta).unwrap().as_slice(),
            )
            .await
            .unwrap();

        let mut manifest_seq2 = Manifest {
            format_version: 1,
            sequence: 2,
            schema_id: schema.schema_id,
            row_groups: vec![RowGroupEntry {
                data: committed_data.into(),
                meta: committed_meta.into(),
                ..Default::default()
            }],
            partition_values: None,
            checksum: String::new(),
            ..Default::default()
        };
        manifest_seq2.checksum = manifest_seq2.compute_checksum().unwrap();
        storage
            .write(
                &format!("{}/{}", table, Manifest::filename(2)),
                serde_json::to_vec(&manifest_seq2).unwrap().as_slice(),
            )
            .await
            .unwrap();
        storage
            .write(&format!("{}/_manifest.json", table), b"{\"latest\": 2}")
            .await
            .unwrap();
        let intent = serde_json::json!({
            "txn_id": "txn_committed",
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": 1,
            "files": [committed_data, committed_meta],
        });
        storage
            .write(
                &format!("{}/_staging/intents/txn_committed.json", table),
                serde_json::to_vec(&intent).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // A new commit must recover the stale intent and succeed.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // The stale intent must be gone.
        assert!(!storage
            .exists(&format!("{}/_staging/intents/txn_committed.json", table))
            .await
            .unwrap());

        // The final files from the (simulated) committed sequence 2 must still
        // exist and be referenced by the latest manifest.
        assert!(storage
            .exists(&format!("{}/{}", table, committed_data))
            .await
            .unwrap());
        assert!(storage
            .exists(&format!("{}/{}", table, committed_meta))
            .await
            .unwrap());

        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(3));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("{}/{}", table, Manifest::filename(3)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert!(manifest
            .row_groups
            .iter()
            .any(|e| e.data == committed_data && e.meta == committed_meta));
    }

    #[tokio::test]
    async fn test_commit_checksums_match() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());

        for entry in &manifest.row_groups {
            let parquet_bytes = storage
                .read(&format!("products/{}", entry.data))
                .await
                .unwrap();
            let meta: RowGroupMeta = serde_json::from_slice(
                &storage
                    .read(&format!("products/{}", entry.meta))
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(
                meta.verify_against_data(&parquet_bytes),
                "parquet bytes must match checksum in {}",
                entry.meta
            );
            assert!(
                meta.verify_meta_checksum().unwrap(),
                "meta checksum must be valid in {}",
                entry.meta
            );
        }
    }

    /// Storage wrapper that returns corrupted bytes for the first read of any
    /// staged `.parquet.part` file, then returns the correct bytes. This
    /// exercises the row-group checksum retry path before the files are renamed
    /// to their final locations.
    #[derive(Debug)]
    struct CorruptFirstParquetReadStorage {
        inner: MemoryStorage,
        first_parquet_read: std::sync::atomic::AtomicBool,
    }

    impl CorruptFirstParquetReadStorage {
        fn new() -> Self {
            Self {
                inner: MemoryStorage::new(),
                first_parquet_read: std::sync::atomic::AtomicBool::new(true),
            }
        }
    }

    #[async_trait]
    impl Storage for CorruptFirstParquetReadStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
            let mut data = self.inner.read(path).await?;
            if path.ends_with(".parquet.part")
                && self.first_parquet_read.swap(false, Ordering::SeqCst)
                && !data.is_empty()
            {
                data[0] ^= 0xFF;
            }
            Ok(data)
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

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> Result<()> {
            self.inner.sync(path).await
        }
    }

    #[tokio::test]
    async fn test_row_group_checksum_retry_on_corrupt_read() {
        let storage = Arc::new(CorruptFirstParquetReadStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // The commit should have retried and produced valid files.
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        let entry = &manifest.row_groups[0];
        let parquet_bytes = storage
            .inner
            .read(&format!("products/{}", entry.data))
            .await
            .unwrap();
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(meta.verify_against_data(&parquet_bytes));
    }

    /// Storage wrapper that returns a tampered manifest for the first read of
    /// any `_manifests/*.json.tmp` file, then returns the correct bytes. This
    /// exercises the manifest checksum retry path before the file is renamed to
    /// its final location.
    #[derive(Debug)]
    struct CorruptFirstManifestReadStorage {
        inner: MemoryStorage,
        first_manifest_read: std::sync::atomic::AtomicBool,
    }

    impl CorruptFirstManifestReadStorage {
        fn new() -> Self {
            Self {
                inner: MemoryStorage::new(),
                first_manifest_read: std::sync::atomic::AtomicBool::new(true),
            }
        }
    }

    #[async_trait]
    impl Storage for CorruptFirstManifestReadStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
            let data = self.inner.read(path).await?;
            if path.contains("_manifests/")
                && path.ends_with(".json.tmp")
                && self.first_manifest_read.swap(false, Ordering::SeqCst)
            {
                let mut value: serde_json::Value = serde_json::from_slice(&data)?;
                if let Some(seq) = value.get_mut("sequence").and_then(|v| v.as_u64()) {
                    value["sequence"] = (seq + 1).into();
                }
                return Ok(serde_json::to_vec(&value)?);
            }
            Ok(data)
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

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> Result<()> {
            self.inner.sync(path).await
        }
    }

    #[tokio::test]
    async fn test_manifest_checksum_retry_on_corrupt_read() {
        let storage = Arc::new(CorruptFirstManifestReadStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.sequence, 1);
    }

    /// `Storage` wrapper that signals when the first commit's `lock_exclusive`
    /// call has acquired the in-process lock, and then waits on a barrier before
    /// returning the guard. A second signal fires once a second writer has
    /// entered `lock_exclusive`, without blocking. Used to verify that a second
    /// writer blocks until the first writer releases the lock.
    #[derive(Debug)]
    struct SignalingStorage {
        inner: MemoryStorage,
        first_lock_signal:
            std::sync::Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        first_lock_barrier:
            std::sync::Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>>,
        second_lock_signal:
            std::sync::Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
        lock_count: std::sync::atomic::AtomicUsize,
    }

    impl SignalingStorage {
        fn new() -> (
            Self,
            tokio::sync::oneshot::Receiver<()>,
            tokio::sync::oneshot::Sender<()>,
            tokio::sync::oneshot::Receiver<()>,
        ) {
            let (locked_tx, locked_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = tokio::sync::oneshot::channel();
            let (second_locked_tx, second_locked_rx) = tokio::sync::oneshot::channel();
            let storage = Self {
                inner: MemoryStorage::new(),
                first_lock_signal: std::sync::Arc::new(std::sync::Mutex::new(Some(locked_tx))),
                first_lock_barrier: std::sync::Arc::new(std::sync::Mutex::new(Some(release_rx))),
                second_lock_signal: std::sync::Arc::new(std::sync::Mutex::new(Some(
                    second_locked_tx,
                ))),
                lock_count: std::sync::atomic::AtomicUsize::new(0),
            };
            (storage, locked_rx, release_tx, second_locked_rx)
        }
    }

    #[async_trait]
    impl Storage for SignalingStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
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

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            let count = self.lock_count.fetch_add(1, Ordering::SeqCst) + 1;
            // Signal once the second writer has entered lock_exclusive, before
            // it attempts to acquire the inner lock, so the test can observe the
            // blocking state without relying on a sleep.
            if count == 3 {
                let signal = self.second_lock_signal.lock().unwrap().take();
                if let Some(tx) = signal {
                    let _ = tx.send(());
                }
            }
            let guard = self.inner.lock_exclusive(path, timeout).await?;
            if count == 2 {
                let signal = self.first_lock_signal.lock().unwrap().take();
                let barrier = self.first_lock_barrier.lock().unwrap().take();
                if let Some(tx) = signal {
                    let _ = tx.send(());
                    if let Some(rx) = barrier {
                        let _ = rx.await;
                    }
                }
            }
            Ok(guard)
        }

        async fn sync(&self, path: &str) -> Result<()> {
            self.inner.sync(path).await
        }
    }

    #[tokio::test]
    async fn test_same_process_concurrent_writers_serialize() {
        let (storage, locked_rx, release_tx, second_locked_rx) = SignalingStorage::new();
        let storage = Arc::new(storage);
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        // Start the first commit in the background. It will acquire the
        // in-process lock and then block on the barrier.
        let storage1: Arc<dyn Storage> = Arc::clone(&storage) as Arc<dyn Storage>;
        let mut writer1 = Writer::new(storage1, table, schema.clone()).await.unwrap();
        writer1
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        let commit1 = tokio::spawn(async move { writer1.commit().await });

        // Wait until the first writer has definitely acquired the lock.
        locked_rx.await.unwrap();

        // Start a second writer while the first still holds the lock. Creation
        // itself now needs the lock, so the whole sequence is spawned.
        let schema2 = schema.clone();
        let storage2: Arc<dyn Storage> = Arc::clone(&storage) as Arc<dyn Storage>;
        let commit2 = tokio::spawn(async move {
            let mut writer2 = Writer::new(storage2, table, schema2).await.unwrap();
            writer2
                .insert_batch(make_int_batch(vec![4, 5, 6]))
                .await
                .unwrap();
            writer2.commit().await
        });

        // Wait until the second writer has actually entered lock_exclusive.
        // It should still be waiting for the lock.
        second_locked_rx.await.unwrap();
        assert!(
            !commit2.is_finished(),
            "second writer should be blocked by the lock"
        );

        // Release the first writer and wait for both commits to finish.
        let _ = release_tx.send(());
        let result1 = commit1.await.unwrap();
        let result2 = commit2.await.unwrap();
        assert!(result1.is_ok(), "first commit failed: {:?}", result1);
        assert!(result2.is_ok(), "second commit failed: {:?}", result2);

        // Both commits must be durable and present in the manifest.
        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("{}/{}", table, Manifest::filename(2)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 2);
    }

    #[tokio::test]
    async fn test_recovery_creates_missing_pointer_from_manifests() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Delete the pointer. The next commit must recreate it from the
        // manifest directory and continue from the latest valid sequence.
        storage
            .delete(&format!("{}/_manifest.json", table))
            .await
            .unwrap();

        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![7, 8, 9]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(3));

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("{}/{}", table, Manifest::filename(3)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 3);
    }

    #[tokio::test]
    async fn test_recovery_errors_when_latest_manifest_missing() {
        // If the manifest referenced by the pointer is missing but an older
        // valid manifest exists, recovery must not fall back to the older
        // manifest. The table is corrupt and the operator must be notified.
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Remove the latest manifest file while leaving the pointer at seq 2.
        storage
            .delete(&format!("{}/{}", table, Manifest::filename(2)))
            .await
            .unwrap();

        let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
            .await
            .unwrap();
        writer
            .insert_batch(make_int_batch(vec![7, 8, 9]))
            .await
            .unwrap();
        let manifest_path = format!("{}/{}", table, Manifest::filename(2));
        let result = writer.commit().await;
        assert!(
            matches!(
                result,
                Err(IcefallDBError::ManifestNotFound(ref path)) if path == &manifest_path
            ),
            "expected ManifestNotFound for missing latest manifest, got {:?}",
            result
        );

        // The pointer must remain unchanged so an operator can diagnose the
        // corruption.
        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));
    }

    #[tokio::test]
    async fn test_in_process_lock_times_out_when_held() {
        let storage = Arc::new(MemoryStorage::new());
        let guard = storage
            .lock_exclusive("products/_write.lock", Duration::from_secs(60))
            .await
            .unwrap();

        let storage2 = Arc::clone(&storage);
        let result = tokio::spawn(async move {
            storage2
                .lock_exclusive("products/_write.lock", Duration::from_millis(10))
                .await
        })
        .await
        .unwrap();
        assert!(
            matches!(result, Err(IcefallDBError::LockTimeout(_))),
            "expected LockTimeout when lock is held"
        );

        drop(guard);
        let _ = storage
            .lock_exclusive("products/_write.lock", Duration::from_secs(1))
            .await
            .unwrap();
    }

    #[test]
    fn test_compute_min_max_nan_only_returns_none() {
        let array: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![
            f64::NAN,
            f64::NAN,
            f64::NEG_INFINITY,
        ]));
        let (min, max) = compute_min_max(&array, "test").unwrap();
        assert_eq!(min, None);
        assert_eq!(max, None);
    }

    #[test]
    fn test_compute_min_max_mixed_finite_and_nan() {
        let array: ArrayRef = Arc::new(arrow::array::Float64Array::from(vec![
            f64::NAN,
            3.0,
            1.0,
            f64::INFINITY,
            2.0,
        ]));
        let (min, max) = compute_min_max(&array, "test").unwrap();
        assert_eq!(min, Some(1.0.into()));
        assert_eq!(max, Some(3.0.into()));
    }

    #[test]
    fn test_compute_min_max_nan_only_float32() {
        let array: ArrayRef = Arc::new(arrow::array::Float32Array::from(vec![
            f32::NAN,
            f32::INFINITY,
        ]));
        let (min, max) = compute_min_max(&array, "test").unwrap();
        assert_eq!(min, None);
        assert_eq!(max, None);
    }

    fn make_partitioned_float_schema() -> Schema {
        let mut schema = Schema {
            schema_id: 1,
            columns: vec![
                Column {
                    name: "id".into(),
                    r#type: "int64".into(),
                    nullable: false,
                    field_id: 0,
                },
                Column {
                    name: "part".into(),
                    r#type: "float64".into(),
                    nullable: true,
                    field_id: 0,
                },
            ],
            partition_by: Some(vec!["part".into()]),
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };
        schema.assign_field_ids(None);
        schema
    }

    #[tokio::test]
    async fn test_partition_values_omitted_for_nan_only_column() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_partitioned_float_schema();
        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("part", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(arrow::array::Float64Array::from(vec![f64::NAN, f64::NAN])),
            ],
        )
        .unwrap();

        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(
            manifest.partition_values, None,
            "partition values should be omitted when the partition column contains only non-finite values"
        );
    }

    #[tokio::test]
    async fn test_recovery_deletes_manifest_json_tmp_orphans() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "products";
        let schema = make_int_schema(10, 1024 * 1024);

        // Commit sequence 1 normally.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Simulate a crash that left a manifest .json.tmp orphan.
        storage
            .write(&format!("{}/_manifests/000000002.json.tmp", table), b"{}")
            .await
            .unwrap();

        // The next commit must remove the orphan and succeed.
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        assert!(
            !storage
                .exists(&format!("{}/_manifests/000000002.json.tmp", table))
                .await
                .unwrap(),
            "manifest .json.tmp orphan should be removed during recovery"
        );

        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(pointer.get("latest").and_then(|v| v.as_u64()), Some(2));
    }

    /// Storage wrapper that creates a leftover final parquet file with the same
    /// row-group id as the first staged `.parquet.part` file. This simulates a
    /// UUID collision or orphaned final file and forces the writer to pick a
    /// different id.
    #[derive(Debug)]
    struct ExistingFinalFileStorage {
        inner: MemoryStorage,
        triggered: std::sync::atomic::AtomicBool,
        table: String,
    }

    impl ExistingFinalFileStorage {
        fn new(table: impl Into<String>) -> Self {
            Self {
                inner: MemoryStorage::new(),
                triggered: std::sync::atomic::AtomicBool::new(false),
                table: table.into(),
            }
        }
    }

    #[async_trait]
    impl Storage for ExistingFinalFileStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        async fn read(&self, path: &str) -> Result<Vec<u8>> {
            self.inner.read(path).await
        }

        async fn size(&self, path: &str) -> Result<u64> {
            self.inner.size(path).await
        }

        async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
            self.inner.read_range(path, offset, len).await
        }

        async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
            if path.contains("_staging/incoming/")
                && path.ends_with(".parquet.part")
                && !self
                    .triggered
                    .swap(true, std::sync::atomic::Ordering::SeqCst)
            {
                if let Some(filename) = std::path::Path::new(path)
                    .file_name()
                    .and_then(|s| s.to_str())
                {
                    if let Some(rg_id) = filename.strip_suffix(".parquet.part") {
                        let _ = self
                            .inner
                            .write(&format!("{}/{}.parquet", self.table, rg_id), b"leftover")
                            .await;
                    }
                }
            }
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

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: Duration,
        ) -> Result<Box<dyn LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> Result<()> {
            self.inner.sync(path).await
        }
    }

    #[tokio::test]
    async fn test_writer_avoids_overwriting_existing_final_parquet() {
        let storage = Arc::new(ExistingFinalFileStorage::new("products"));
        let schema = make_int_schema(10, 1024 * 1024);

        let mut writer = Writer::new(Arc::clone(&storage) as Arc<dyn Storage>, "products", schema)
            .await
            .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // The manifest must reference exactly one valid row group.
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", Manifest::filename(1)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(manifest.verify_checksum().unwrap());
        assert_eq!(manifest.row_groups.len(), 1);

        let entry = &manifest.row_groups[0];

        // Two parquet files exist: the leftover fake and the real row group.
        let parquet_files: Vec<String> = storage
            .inner
            .list("products/")
            .await
            .unwrap()
            .into_iter()
            .filter(|p| p.ends_with(".parquet"))
            .collect();
        assert_eq!(
            parquet_files.len(),
            2,
            "expected leftover + real parquet files"
        );
        assert!(parquet_files.contains(&format!("products/{}", entry.data)));

        // The real parquet file must verify against its metadata.
        let parquet_bytes = storage
            .inner
            .read(&format!("products/{}", entry.data))
            .await
            .unwrap();
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .inner
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(meta.verify_against_data(&parquet_bytes));
    }

    fn write_parquet_file(path: &std::path::Path, batch: &RecordBatch) -> RecordBatch {
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(1).expect("valid zstd level"),
            ))
            .build();
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(path).unwrap(),
            batch.schema().clone(),
            Some(props),
        )
        .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        batch.clone()
    }

    #[tokio::test]
    async fn test_insert_parquet_fast_path_compatible_file() {
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("input.parquet");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(arrow::array::Float64Array::from(vec![
                    Some(1.5),
                    None,
                    Some(3.5),
                ])),
                Arc::new(arrow::array::StringArray::from(vec![
                    Some("a"),
                    Some("b"),
                    Some("c"),
                ])),
            ],
        )
        .unwrap();
        write_parquet_file(&parquet_path, &batch);

        let icefalldb_schema = Schema {
            schema_id: 1,
            columns: vec![
                Column::new("id", "int64", false),
                Column::new("value", "float64", true),
                Column::new("name", "utf8", true),
            ],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            icefalldb_schema,
        )
        .await
        .unwrap();

        let outcome = writer
            .insert_parquet(parquet_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            InsertParquetOutcome::FastPath { rows: 3 }
        ));

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.row_groups.len(), 1);

        let entry = &manifest.row_groups[0];
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(meta.rows, 3);
        assert!(meta.verify_meta_checksum().unwrap());

        let parquet_bytes = storage
            .read(&format!("products/{}", entry.data))
            .await
            .unwrap();
        assert!(meta.verify_against_data(&parquet_bytes));

        let id_stats = meta.columns.get("id").unwrap();
        assert_eq!(id_stats.min, Some(1i64.into()));
        assert_eq!(id_stats.max, Some(3i64.into()));
        assert_eq!(id_stats.nulls, 0);

        let value_stats = meta.columns.get("value").unwrap();
        assert_eq!(value_stats.min, Some(1.5.into()));
        assert_eq!(value_stats.max, Some(3.5.into()));
        assert_eq!(value_stats.nulls, 1);

        let name_stats = meta.columns.get("name").unwrap();
        assert_eq!(name_stats.min, Some("a".into()));
        assert_eq!(name_stats.max, Some("c".into()));
        assert_eq!(name_stats.nulls, 0);
    }

    #[tokio::test]
    async fn test_insert_parquet_rejects_multi_row_group_file() {
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("input.parquet");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        // Write two row groups with disjoint value ranges.
        let props = parquet::file::properties::WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                parquet::basic::ZstdLevel::try_new(1).unwrap(),
            ))
            .set_max_row_group_row_count(Some(2))
            .build();
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(&parquet_path).unwrap(),
            Arc::clone(&arrow_schema),
            Some(props),
        )
        .unwrap();
        let batch1 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![1, 2]))],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![10, 20]))],
        )
        .unwrap();
        writer.write(&batch1).unwrap();
        writer.write(&batch2).unwrap();
        writer.close().unwrap();

        let icefalldb_schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("id", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 2,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            icefalldb_schema,
        )
        .await
        .unwrap();

        let outcome = writer
            .insert_parquet(parquet_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(outcome, InsertParquetOutcome::Incompatible);

        // Simulate the CLI fallback path: decode the Parquet file and insert
        // via batches. The writer should produce one IcefallDB row group per
        // Parquet row group with valid sidecar offsets.
        let batches = read_batches_from_parquet(&parquet_path).await.unwrap();
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        for batch in batches {
            writer.insert_batch(batch).await.unwrap();
        }
        writer.commit().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.row_groups.len(), 2);

        let mut all_rows = 0;
        for entry in &manifest.row_groups {
            let meta: RowGroupMeta = serde_json::from_slice(
                &storage
                    .read(&format!("products/{}", entry.meta))
                    .await
                    .unwrap(),
            )
            .unwrap();
            assert!(
                meta.column_offsets.is_some(),
                "each row group must have valid sidecar column offsets"
            );
            all_rows += meta.rows;
        }
        assert_eq!(all_rows, total_rows);
    }

    async fn read_batches_from_parquet(path: &std::path::Path) -> Result<Vec<RecordBatch>> {
        let bytes = tokio::fs::read(path).await.map_err(other)?;
        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            bytes::Bytes::from(bytes),
        )
        .map_err(other)?;
        let reader = builder.build().map_err(other)?;
        reader
            .into_iter()
            .map(|res| res.map_err(|e| IcefallDBError::ParquetDecode(e.to_string())))
            .collect()
    }

    #[tokio::test]
    async fn test_insert_parquet_fast_path_fallback_on_schema_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("input.parquet");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Utf8,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(arrow::array::StringArray::from(vec!["1", "2"]))],
        )
        .unwrap();
        write_parquet_file(&parquet_path, &batch);

        let icefalldb_schema = Schema {
            schema_id: 1,
            columns: vec![Column::new("id", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            icefalldb_schema,
        )
        .await
        .unwrap();

        let outcome = writer
            .insert_parquet(parquet_path.to_str().unwrap())
            .await
            .unwrap();
        assert_eq!(outcome, InsertParquetOutcome::Incompatible);

        // No commit should have happened.
        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        assert_eq!(pointer["latest"].as_u64(), Some(0));
    }

    #[tokio::test]
    async fn test_insert_parquet_fast_path_partition_values() {
        let dir = tempfile::tempdir().unwrap();
        let parquet_path = dir.path().join("input.parquet");

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("part", DataType::Utf8, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(arrow::array::StringArray::from(vec!["x", "x", "x"])),
            ],
        )
        .unwrap();
        write_parquet_file(&parquet_path, &batch);

        let icefalldb_schema = Schema {
            schema_id: 1,
            columns: vec![
                Column::new("id", "int64", false),
                Column::new("part", "utf8", false),
            ],
            partition_by: Some(vec!["part".into()]),
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            max_field_id: 0,
            dropped_columns: vec![],
        };

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            icefalldb_schema,
        )
        .await
        .unwrap();

        let outcome = writer
            .insert_parquet(parquet_path.to_str().unwrap())
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            InsertParquetOutcome::FastPath { rows: 3 }
        ));

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        let entry = &manifest.row_groups[0];
        let partition_values = manifest.partition_values.as_ref().unwrap();
        let pv = partition_values.get(&entry.data).unwrap();
        assert_eq!(pv.get("part"), Some(&"x".into()));
    }

    #[tokio::test]
    async fn test_commit_populates_row_counts_and_column_offsets() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(10, 1024 * 1024);
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            "products",
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3, 4, 5]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("products/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(manifest.row_counts, Some(vec![5]));

        let entry = &manifest.row_groups[0];
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        let offsets = meta
            .column_offsets
            .as_ref()
            .expect("column_offsets missing");
        assert!(offsets.contains_key("id"));
        let id_offset = &offsets["id"];
        assert!(id_offset.length > 0);
    }

    /// Verify that the intent journal + recovery correctly handles mutation file
    /// types (`.del` deletion vectors, `_rowindex/*.idx` files) in addition to
    /// the existing `.parquet`/`.meta` row-group files.
    ///
    /// Part (a): stage mutation files + write an intent listing them, abort
    /// before the pointer swap, run `cleanup_staging`, and assert the staged
    /// `.del`/`.idx` orphans are removed while the manifest pointer is unchanged.
    ///
    /// Part (b): complete a full commit that references those file types (via the
    /// intent's file list), run `cleanup_staging`, and assert it is a no-op —
    /// every file listed in the intent is in `referenced_files` and survives.
    #[tokio::test]
    async fn mutation_commit_is_atomic() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "mut_atomic";
        let schema = make_int_schema(1000, 64 * 1024 * 1024);

        // ── Bootstrap: commit one normal row group so the table is initialised ──
        let mut writer = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let (seq0, manifest0) = writer.load_current_manifest().await.unwrap();
        assert_eq!(seq0, 1);
        let existing_data = manifest0.row_groups[0].data.clone();
        let existing_meta = manifest0.row_groups[0].meta.clone();

        // ── Part (a): crash-before-pointer-swap ──────────────────────────────
        //
        // Simulate what a mutation commit would do:
        //  1. Stage a .del file and a _rowindex/*.idx file at their final
        //     table-root locations.
        //  2. Write an intent that lists them.
        //  3. Do NOT update the manifest pointer (simulates a crash before swap).
        //  4. Run cleanup_staging with the current manifest's referenced_files.
        //  5. Assert the staged mutation files are deleted (they are orphans).
        //  6. Assert the manifest pointer is unchanged.

        let del_file = "_deletions/rg_aaaa.del";
        let idx_file = "_rowindex/rg_aaaa.idx";

        storage
            .write(&format!("{}/{}", table, del_file), b"del-vector-bytes")
            .await
            .unwrap();
        storage
            .write(&format!("{}/{}", table, idx_file), b"rowindex-bytes")
            .await
            .unwrap();

        // Verify they exist before recovery.
        assert!(
            storage
                .exists(&format!("{}/{}", table, del_file))
                .await
                .unwrap(),
            ".del file should exist before recovery"
        );
        assert!(
            storage
                .exists(&format!("{}/{}", table, idx_file))
                .await
                .unwrap(),
            ".idx file should exist before recovery"
        );

        // Write a stale intent listing both mutation files (plus the pre-existing
        // parquet/meta pair to prove the intent format works for mixed types).
        let txn_id = "txn_mutation_abort";
        let intent = serde_json::json!({
            "txn_id": txn_id,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": schema.schema_id,
            "files": [del_file, idx_file],
        });
        let intent_path = format!("{}/_staging/intents/{}.json", table, txn_id);
        storage
            .write(
                &intent_path,
                serde_json::to_vec(&intent).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // The referenced_files set mirrors what the current committed manifest
        // points to — only the row-group .parquet/.meta files from seq 1.
        let referenced_files: HashSet<String> =
            [existing_data.clone(), existing_meta.clone()].into();

        // Run recovery. The lock is "held" by this test (single-threaded).
        cleanup_staging(storage.as_ref(), table, seq0, &referenced_files)
            .await
            .unwrap();

        // The orphaned mutation files must be gone.
        assert!(
            !storage
                .exists(&format!("{}/{}", table, del_file))
                .await
                .unwrap(),
            ".del orphan should be removed by cleanup_staging"
        );
        assert!(
            !storage
                .exists(&format!("{}/{}", table, idx_file))
                .await
                .unwrap(),
            ".idx orphan should be removed by cleanup_staging"
        );

        // The intent itself must be gone.
        assert!(
            !storage.exists(&intent_path).await.unwrap(),
            "stale intent should be removed by cleanup_staging"
        );

        // The manifest pointer must be unchanged at seq 1.
        let pointer: serde_json::Value = serde_json::from_slice(
            &storage
                .read(&format!("{}/_manifest.json", table))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            pointer.get("latest").and_then(|v| v.as_u64()),
            Some(seq0),
            "manifest pointer must be unchanged after aborted mutation"
        );

        // ── Part (b): crash-after-pointer-swap / durable commit ──────────────
        //
        // Now simulate a commit that DID succeed: the pointer was advanced to
        // seq 2 and the new manifest references the mutation files. Running
        // cleanup_staging must be a no-op — the files must survive.

        let del_file2 = "_deletions/rg_bbbb.del";
        let idx_file2 = "_rowindex/rg_bbbb.idx";

        storage
            .write(&format!("{}/{}", table, del_file2), b"del-vector-committed")
            .await
            .unwrap();
        storage
            .write(&format!("{}/{}", table, idx_file2), b"rowindex-committed")
            .await
            .unwrap();

        // Write an intent that lists the committed mutation files.
        let txn_id2 = "txn_mutation_committed";
        let intent2 = serde_json::json!({
            "txn_id": txn_id2,
            "started_at": chrono::Utc::now().to_rfc3339(),
            "schema_id": schema.schema_id,
            "files": [del_file2, idx_file2],
        });
        let intent_path2 = format!("{}/_staging/intents/{}.json", table, txn_id2);
        storage
            .write(
                &intent_path2,
                serde_json::to_vec(&intent2).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // The referenced_files set now includes the mutation files because the
        // (hypothetical) new manifest at seq 2 references them.
        let mut referenced_files2 = referenced_files.clone();
        referenced_files2.insert(del_file2.to_string());
        referenced_files2.insert(idx_file2.to_string());

        // Simulate the pointer having been advanced to seq 2.
        let seq2 = seq0 + 1;

        cleanup_staging(storage.as_ref(), table, seq2, &referenced_files2)
            .await
            .unwrap();

        // The committed mutation files must still exist (recovery is a no-op).
        assert!(
            storage
                .exists(&format!("{}/{}", table, del_file2))
                .await
                .unwrap(),
            "committed .del file must survive cleanup_staging"
        );
        assert!(
            storage
                .exists(&format!("{}/{}", table, idx_file2))
                .await
                .unwrap(),
            "committed .idx file must survive cleanup_staging"
        );

        // The intent for the committed txn must be removed (intent cleanup is
        // always best-effort after a durable commit).
        assert!(
            !storage.exists(&intent_path2).await.unwrap(),
            "committed intent should be removed by cleanup_staging"
        );
    }

    /// Verify that `commit_deletes` sets `deleted_count = cardinality` and that
    /// re-deleting an already-dead offset does not bump the count (idempotent).
    ///
    /// Critically, this test also proves the `referenced_files` corruption-
    /// prevention property: after `commit_deletes` commits a `.del` file into the
    /// manifest, a SYNTHETIC STALE INTENT (injected directly into storage) that
    /// names that same `.del` path must NOT cause `cleanup_staging` to delete the
    /// committed file.  The stale intent simulates a leftover journal entry from a
    /// hypothetical prior aborted operation.
    ///
    /// This is a TRUE regression guard: if `manifest_referenced_files` did NOT
    /// include committed `.del` paths, `cleanup_staging` would process the stale
    /// intent, find the `.del` not in `referenced_files`, and delete it — causing
    /// silent data corruption.  Reverting the `manifest_referenced_files` extension
    /// (the loop that inserts `e.deletes` into `referenced`) makes this test fail.
    #[tokio::test]
    async fn commit_deletes_sets_cardinality_and_is_idempotent() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "del_idempotent";
        let schema = make_int_schema(1000, 64 * 1024 * 1024);

        // Bootstrap: commit one fragment with 10 rows so the table has a fragment.
        let mut w = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        w.insert_batch(make_int_batch(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
            .await
            .unwrap();
        w.commit().await.unwrap();

        // The first (and only) fragment's id.
        let (_, manifest0) = w.load_current_manifest().await.unwrap();
        assert_eq!(manifest0.row_groups.len(), 1);
        let frag_id = manifest0.row_groups[0].fragment_id;

        // ── First delete: offsets 2 and 5 ──────────────────────────────────
        let m1 = w
            .commit_deletes(HashMap::from([(frag_id, vec![2u32, 5])]))
            .await
            .unwrap()
            .new_manifest;
        let e1 = m1
            .row_groups
            .iter()
            .find(|e| e.fragment_id == frag_id)
            .unwrap();
        assert_eq!(
            e1.deleted_count, 2,
            "deleted_count should be 2 after deleting offsets 2,5"
        );
        assert!(e1.deletes.is_some(), "deletes path should be set");

        // ── Re-delete offset 5 (already dead): must NOT bump deleted_count ──
        let m2 = w
            .commit_deletes(HashMap::from([(frag_id, vec![5u32])]))
            .await
            .unwrap()
            .new_manifest;
        let e2 = m2
            .row_groups
            .iter()
            .find(|e| e.fragment_id == frag_id)
            .unwrap();
        assert_eq!(
            e2.deleted_count, 2,
            "deleted_count must remain 2 after re-deleting an already-dead offset"
        );

        // Capture the bare relative path of the committed .del file (no table/ prefix).
        // This is the path stored in entry.deletes and in manifest_referenced_files.
        let del_path = e2.deletes.clone().unwrap();

        // ── Inject a synthetic stale intent that names the committed .del ────
        //
        // This simulates a leftover intent from a hypothetical prior aborted
        // operation (e.g. a writer that crashed before the pointer swap).
        // `cleanup_staging` will find this intent, iterate its `files` array,
        // and check each filename against `referenced_files`.  If `referenced_files`
        // contains the `.del` path (as it must after the fix), cleanup spares it.
        // If not, cleanup deletes it — silent corruption.
        //
        // Note: commit_deletes already deleted its OWN intent on the success path
        // above, so there is no surviving intent left from the real commit.  This
        // injected intent is the only one that will be seen by the next cleanup pass,
        // and it genuinely names the committed .del path — cleanup WILL try to act on
        // it.  The referenced_files guard is what stops the deletion.
        let stale_intent_path =
            format!("{}/_staging/intents/stale-aborted-op-qa-verify.json", table);
        let stale_intent = serde_json::json!({
            "txn_id": "stale-aborted-op-qa-verify",
            "started_at": "2020-01-01T00:00:00Z",
            "schema_id": "qa-verify",
            // The bare relative path — exactly as stored in entry.deletes and
            // in the referenced_files HashSet, so the contains() check is consistent.
            "files": [del_path],
        });
        storage
            .write(
                &stale_intent_path,
                serde_json::to_vec(&stale_intent).unwrap().as_slice(),
            )
            .await
            .unwrap();

        // Verify the stale intent is visible to storage before triggering cleanup.
        assert!(
            storage.exists(&stale_intent_path).await.unwrap(),
            "stale intent must be present before the next commit triggers cleanup"
        );

        // ── Trigger cleanup_staging via a subsequent append commit ───────────
        // The append commit calls cleanup_staging with the current manifest's
        // referenced_files.  The stale intent (injected above) will be processed:
        // cleanup sees del_path in its files array and checks referenced_files.
        // Because del_path IS in referenced_files (the manifest_referenced_files
        // fix includes e.deletes), cleanup skips it.  Reverting that fix causes
        // cleanup to delete the committed .del file and break the assertion below.
        let mut w2 = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        w2.insert_batch(make_int_batch(vec![100])).await.unwrap();
        w2.commit().await.unwrap();

        // ── Assert: committed .del file STILL EXISTS after cleanup ───────────
        // This proves referenced_files protects committed .del files;
        // reverting the manifest_referenced_files extension makes this fail.
        assert!(
            storage
                .exists(&format!("{}/{}", table, del_path))
                .await
                .unwrap(),
            "committed .del file must survive cleanup_staging even when a stale \
             intent names it — referenced_files must protect it"
        );

        // The manifest after the append must still carry the fragment's
        // deletion metadata with deleted_count == 2.
        let (_, manifest_after) = w2.load_current_manifest().await.unwrap();
        let e_after = manifest_after
            .row_groups
            .iter()
            .find(|e| e.fragment_id == frag_id)
            .unwrap();
        assert_eq!(
            e_after.deleted_count, 2,
            "deleted_count must be preserved across subsequent commits"
        );
        assert_eq!(
            e_after.deletes.as_deref(),
            Some(del_path.as_str()),
            ".del path must be preserved across subsequent commits"
        );
    }

    // ── acceptance test ──────────────────────────────────────────────────

    /// Open a Writer on an in-memory table seeded with one fragment that contains
    /// row_ids 0..=4 (5 rows).  Fragment id 1, offsets 0..=4.
    async fn open_test_writer_p22(table: &str) -> (Writer, Arc<MemoryStorage>) {
        let storage: Arc<MemoryStorage> = Arc::new(MemoryStorage::new());
        let schema = make_int_schema(1000, 64 * 1024 * 1024);
        let mut w = Writer::new(
            Arc::clone(&storage) as Arc<dyn Storage>,
            table,
            schema.clone(),
        )
        .await
        .unwrap();
        // Insert 5 rows so the table has fragment 0 with row_ids [0..5].
        w.insert_batch(make_int_batch(vec![10, 20, 30, 40, 50]))
            .await
            .unwrap();
        w.commit().await.unwrap();
        (w, storage)
    }

    /// Build a trivial RecordBatch carrying `row_ids` as the row ordering and
    /// `values` as the "id" column payload.
    fn rows_for(row_ids: &[i64], values: &[i64]) -> RecordBatch {
        let _ = row_ids; // used only to match position — values carry the actual data
        make_int_batch(values.to_vec())
    }

    /// Build a Vec<MatchLoc> from `(fragment_id, offset, row_id)` triples.
    fn locs_for(triples: &[(u64, u32, u64)]) -> Vec<MatchLoc> {
        triples
            .iter()
            .map(|&(fragment_id, offset, row_id)| MatchLoc {
                fragment_id,
                offset,
                row_id,
            })
            .collect()
    }

    /// Acceptance test: relocation + chained update.
    ///
    /// Verifies that `commit_update`:
    /// - Relocates the updated row to a NEW patch fragment (move-stable).
    /// - Tombstones the original offset.
    /// - A second `commit_update` on the same row_id (now in the patch fragment)
    ///   relocates it again — no stale reference to the first patch.
    #[tokio::test]
    async fn commit_update_relocates_and_chains() {
        let table = "t_commit_update";
        let (mut w, storage) = open_test_writer_p22(table).await;

        // After seeding: fragment 0 with row_ids [0..5].
        // row_id 3 lives at (frag 0, offset 3).
        let (_, m0) = w.load_current_manifest().await.unwrap();
        assert_eq!(m0.row_groups.len(), 1);
        let orig_frag = m0.row_groups[0].fragment_id;
        assert_eq!(orig_frag, 0);

        // First update: change value for row_id 3 (offset 3 in frag 0) to 31.
        // The "id" column is not indexed in this test, so set_columns = &[].
        let m1 = w
            .commit_update(rows_for(&[3], &[31]), locs_for(&[(orig_frag, 3, 3)]), &[])
            .await
            .unwrap()
            .new_manifest;

        // rowindex_generation must be present after the update.
        let gen1 = m1.rowindex_generation.clone().unwrap();
        let am1 = crate::rowindex::AddressMap::open(storage.as_ref(), table, &gen1)
            .await
            .unwrap();

        let (patch_frag, _patch_off) = am1.lookup(3).unwrap();
        assert_ne!(
            patch_frag, orig_frag,
            "row_id 3 must be relocated off the original fragment"
        );

        // Original fragment must have the tombstone.
        let orig_entry = m1
            .row_groups
            .iter()
            .find(|e| e.fragment_id == orig_frag)
            .unwrap();
        assert_eq!(
            orig_entry.deleted_count, 1,
            "original offset must be tombstoned"
        );

        // next_row_id must NOT have advanced (move-stable).
        assert_eq!(
            m1.next_row_id, m0.next_row_id,
            "next_row_id must not advance on update"
        );

        // ── Chained update: update row_id 3 again (now at patch_frag, offset 0) ──
        let m2 = w
            .commit_update(rows_for(&[3], &[32]), locs_for(&[(patch_frag, 0, 3)]), &[])
            .await
            .unwrap()
            .new_manifest;

        let gen2 = m2.rowindex_generation.clone().unwrap();
        let am2 = crate::rowindex::AddressMap::open(storage.as_ref(), table, &gen2)
            .await
            .unwrap();

        let (patch2, _) = am2.lookup(3).unwrap();
        assert_ne!(
            patch2, patch_frag,
            "second update must produce a new patch fragment"
        );

        // The first patch fragment must now be tombstoned.
        let patch1_entry = m2
            .row_groups
            .iter()
            .find(|e| e.fragment_id == patch_frag)
            .unwrap();
        assert_eq!(
            patch1_entry.deleted_count, 1,
            "first patch fragment must be tombstoned after second update"
        );
    }

    /// Acceptance test: every atomic commit emits a snapshot checkpoint.
    ///
    /// Verifies that after inserting two fragments the committed manifest carries
    /// `checkpoint: Some(path)`, the checkpoint file exists, and its fragment
    /// summaries match the per-fragment `.meta` sidecars. Also verifies that an
    /// UPDATE commit advances the checkpoint and preserves the original plus patch
    /// fragment summaries.
    #[tokio::test]
    async fn commit_emits_snapshot_checkpoint() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let table = "t_checkpoint";
        let schema = make_int_schema(10, 64 * 1024 * 1024);

        let mut writer = Writer::new(Arc::clone(&storage), table, schema)
            .await
            .unwrap();
        writer
            .insert_batch(make_int_batch(vec![1, 2, 3]))
            .await
            .unwrap();
        writer.commit().await.unwrap();
        writer
            .insert_batch(make_int_batch(vec![4, 5, 6]))
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let (seq, manifest) = writer.load_current_manifest().await.unwrap();
        assert_eq!(seq, 2);
        assert_eq!(manifest.row_groups.len(), 2);
        let checkpoint_path = manifest
            .checkpoint
            .as_deref()
            .expect("manifest must reference a checkpoint");

        let checkpoint_abs = format!("{}/{}", table, checkpoint_path);
        assert!(
            storage.exists(&checkpoint_abs).await.unwrap(),
            "checkpoint file must exist at {}",
            checkpoint_abs
        );

        let checkpoint_bytes = storage.read(&checkpoint_abs).await.unwrap();
        let checkpoint: SnapshotCheckpoint = serde_json::from_slice(&checkpoint_bytes).unwrap();
        assert_eq!(checkpoint.sequence, seq);
        assert_eq!(checkpoint.schema_id, manifest.schema_id);
        assert_eq!(checkpoint.fragments.len(), 2);

        for (entry, summary) in manifest.row_groups.iter().zip(checkpoint.fragments.iter()) {
            assert_eq!(summary.fragment_id, entry.fragment_id);
            assert_eq!(summary.data, entry.data);
            assert_eq!(summary.meta, entry.meta);

            let meta_bytes = storage
                .read(&format!("{}/{}", table, entry.meta))
                .await
                .unwrap();
            let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes).unwrap();
            assert_eq!(summary.rows, meta.rows);
            assert_eq!(summary.columns, meta.columns);
        }

        // ── UPDATE should also emit a checkpoint ─────────────────────────────
        let updated = writer
            .commit_update(rows_for(&[0], &[32]), locs_for(&[(0, 0, 0)]), &[])
            .await
            .unwrap()
            .new_manifest;
        assert_eq!(updated.sequence, 3);
        let update_checkpoint_path = updated
            .checkpoint
            .as_deref()
            .expect("update manifest must reference a checkpoint");
        let update_checkpoint_abs = format!("{}/{}", table, update_checkpoint_path);
        let update_checkpoint_bytes = storage.read(&update_checkpoint_abs).await.unwrap();
        let update_checkpoint: SnapshotCheckpoint =
            serde_json::from_slice(&update_checkpoint_bytes).unwrap();
        assert_eq!(update_checkpoint.sequence, updated.sequence);
        assert_eq!(update_checkpoint.schema_id, updated.schema_id);
        assert_eq!(update_checkpoint.fragments.len(), updated.row_groups.len());
    }
}
