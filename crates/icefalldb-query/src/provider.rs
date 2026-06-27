//! DataFusion [`TableProvider`] for IcefallDB tables.
//!
//! The provider reads table metadata through [`IcefallDBCatalog`], exposes exact
//! sidecar statistics to DataFusion, and produces [`IcefallDBScanExec`] plans that
//! read only the row groups that survive predicate pruning.

use std::fmt;
#[cfg(test)]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use arrow::datatypes::{Schema as ArrowSchema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::DFSchema;
use datafusion::datasource::MemTable;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::expr::InList;
use datafusion::logical_expr::{
    BinaryExpr, Expr, Operator, TableProviderFilterPushDown, TableType,
};
use datafusion::physical_expr::create_physical_expr;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionExpr;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::{collect as collect_plan, ExecutionPlan, Statistics};
use icefalldb_core::metadata::{Manifest, Schema};
use icefalldb_core::storage::Storage;
use icefalldb_core::{
    build_scan_plan_at, list_index_names, load_index, load_index_by_ref, CommitDelta,
    DeletionVector, FragmentDelta, MatchLoc, PlannedRowGroup, Predicate, RowIdSegment, ScanPlan,
};
use std::collections::{HashMap, HashSet};

use crate::catalog::IcefallDBCatalog;
use crate::index_selector;
use crate::metadata_cache::ParquetMetadataCache;
use crate::parquet_exec::{
    build_native_parquet_exec, native_bulk_decode_projection, should_use_native_parquet,
    NativeParquetConfig,
};
use crate::predicate::expr_to_predicate;
use crate::scan::{IcefallDBScanExec, PSEUDO_COL_ROWADDR, PSEUDO_COL_ROWID};
use crate::session::IcefallDBConfig;
use crate::stats::scan_plan_statistics;
use crate::{QueryError, Result};

/// Configuration for a [`IcefallDBTableProvider`].
#[derive(Debug, Clone, Copy)]
pub struct ProviderConfig {
    /// Record batch size used by the Parquet reader.
    pub batch_size: usize,
    /// Default target partitions hint from the session.
    pub target_partitions: usize,
    /// Maximum gap between sparse range reads before they are coalesced.
    pub io_coalesce_window: u64,
    /// Maximum concurrent range reads per row group.
    pub io_concurrency: usize,
    /// Minimum number of columns referenced by a query before the native
    /// DataFusion Parquet reader is preferred over the custom `IcefallDBScanExec`
    /// for local tables.  Set to `0` to disable the native reader, or `1` to
    /// always use it for local tables.
    pub native_parquet_threshold: usize,
    /// Maximum number of decoded Parquet metadata entries to retain in the
    /// process-wide metadata cache. The cache is keyed by `(path, checksum)`,
    /// so file changes automatically invalidate stale entries. Set to `0` to
    /// disable the cache entirely.
    pub parquet_metadata_cache_capacity: usize,
    /// Row threshold below which a table is loaded into the tiny-table cache.
    pub tiny_table_cache_threshold_rows: usize,
    /// Byte threshold below which a table is loaded into the tiny-table cache.
    pub tiny_table_cache_threshold_bytes: usize,
    /// Deferred-commit (mutation WAL) mode for DELETE/UPDATE/MERGE. When `true`
    /// a mutation appends a compact log record and defers the manifest swap to a
    /// periodic checkpoint, trading the ~7-`fsync` commit ceremony for ~1–2
    /// `fsync`s. Default `true`. The `ICEFALLDB_WAL` env var, when set, overrides
    /// this (`1`/`0`) — intended for A/B benchmarking only.
    pub wal_mode: bool,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        // Tuned on the 1m-row DataFusion matrix benchmark (2026-06-21):
        // batch_size=8192 keeps the narrow-table queries near their best while
        // target_partitions=16 uses all available hardware threads to parallelize
        // the wide-table aggregations and joins. The native DataFusion Parquet
        // reader is used for all local tables: it exploits page-index pruning and
        // morsel-based parallelism that the custom scan cannot match.
        Self {
            batch_size: 8192,
            target_partitions: 16,
            io_coalesce_window: 1_048_576,
            io_concurrency: 4,
            native_parquet_threshold: 1,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        }
    }
}

/// Snapshot-dependent state held by a [`IcefallDBTableProvider`].
///
/// Kept behind a synchronous `RwLock` so synchronous `TableProvider` methods
/// can read it and the asynchronous incremental-refresh path can replace it
/// through `&self`.
#[derive(Clone)]
pub(crate) struct SnapshotState {
    /// Schema containing only the real, Parquet-backed data columns.  Used
    /// internally for predicate translation, statistics, pruning, and filter
    /// pushdown — all logic that must remain unaware of pseudo-columns.
    arrow_schema: SchemaRef,
    /// Schema advertised to DataFusion via `TableProvider::schema()`.  Appends
    /// `_rowid` (UInt64) and `_rowaddr` (UInt64) as trailing fields so the
    /// planner can resolve them when they are explicitly named.
    ///
    /// NOTE — DataFusion 54 has no system-column / hidden-column mechanism.
    /// Consequently `SELECT *` WILL include `_rowid` and `_rowaddr`.  If that
    /// is undesirable the caller can strip them after the fact, or a future
    /// DataFusion version that supports hidden columns can be adopted.  The
    /// pseudo-columns are appended last so that all data-column indices (0..N-1)
    /// remain stable; existing pruning, stats, and index-pushdown logic that
    /// keys off those indices is unaffected.
    full_schema: SchemaRef,
    scan_plan: ScanPlan,
    statistics: Statistics,
    indexes: Arc<Vec<SnapshotIndex>>,
    /// Manifest sequence of the snapshot that both `scan_plan` and `indexes`
    /// were loaded from.  `0` for an empty/uninitialized table (no manifest).
    /// Used to verify that the scan plan and index set are pinned to the same
    /// snapshot (see `pinned_sequence()`).
    pinned_sequence: u64,
}

/// A DataFusion `TableProvider` backed by a IcefallDB table.
pub struct IcefallDBTableProvider {
    storage: Arc<dyn Storage>,
    table: String,
    catalog: IcefallDBCatalog,
    snapshot: std::sync::RwLock<SnapshotState>,
    config: ProviderConfig,
    /// Cached scan plan keyed by the provider's pinned manifest sequence. This
    /// avoids reloading `_manifest.json` and `.meta` sidecars while the provider
    /// remains pinned to the same snapshot; `apply_committed_delta` rekeys it to
    /// the new sequence after an incremental refresh.
    scan_cache: tokio::sync::RwLock<Option<(u64, ScanPlan)>>,
    /// Serializes concurrent calls to `apply_committed_delta` so two mutation
    /// tasks cannot read the same old `SnapshotState` and have the last write win.
    apply_lock: tokio::sync::Mutex<()>,
    /// Cached in-memory copy of tiny tables, keyed by manifest sequence.
    /// Loading the full table once lets DataFusion apply arbitrary projections,
    /// filters, and limits without re-reading Parquet files.
    tiny_table_cache: tokio::sync::RwLock<Option<(u64, Arc<MemTable>)>>,
    /// Test-only counter tracking how many times `apply_committed_delta` has
    /// actually refreshed the provider. Used to assert that batched mutations
    /// perform exactly one refresh.
    #[cfg(test)]
    apply_delta_count: Arc<AtomicU64>,
    /// Optional decryption properties built at construction time from a
    /// `KeyProvider`. When `Some`, the custom scan path uses these to decrypt
    /// Parquet data via `ArrowReaderOptions::with_file_decryption_properties`.
    ///
    /// The native `ParquetSource` scan path is *not* used when these are set;
    /// see `scan()` for the rationale. When the encryption feature is enabled
    /// but the user also wants to use the native scan path for plaintext
    /// tables, they should construct two providers (one plaintext via `new`,
    /// one encrypted via `new_encrypted`).
    #[cfg(feature = "encryption")]
    decryption_properties: Option<Arc<parquet::encryption::decrypt::FileDecryptionProperties>>,
}

/// Build the full schema that is advertised to DataFusion by appending the
/// `_rowid` and `_rowaddr` pseudo-column fields to the data-only schema.
fn make_full_schema(data_schema: &ArrowSchema) -> SchemaRef {
    use arrow::datatypes::DataType;
    let mut fields: Vec<arrow::datatypes::FieldRef> =
        data_schema.fields().iter().cloned().collect();
    fields.push(Arc::new(arrow::datatypes::Field::new(
        PSEUDO_COL_ROWID,
        DataType::UInt64,
        false,
    )));
    fields.push(Arc::new(arrow::datatypes::Field::new(
        PSEUDO_COL_ROWADDR,
        DataType::UInt64,
        false,
    )));
    Arc::new(ArrowSchema::new(fields))
}

/// A secondary index as loaded for a pinned snapshot: either fully parsed from
/// JSON, or an mmap'd binary base with a small in-memory delta
/// overlay. The binary variant lets a large indexed table open without parsing
/// the whole `BTreeMap` — only the looked-up key's postings are decoded.
pub(crate) enum SnapshotIndex {
    Json(icefalldb_core::BTreeIndex),
    Binary {
        index: icefalldb_core::index::binary::MmapBinaryIndex,
        overlay: icefalldb_core::index::IndexDeltaOverlay,
    },
    /// An exactly-affine integer key — the whole index is a tiny model;
    /// `lookup` is O(1) arithmetic and the index postings are never loaded.
    Learned {
        definition: icefalldb_core::index::IndexDefinition,
        model: icefalldb_core::index::LearnedKeyModel,
    },
}

impl SnapshotIndex {
    fn column(&self) -> &str {
        match self {
            SnapshotIndex::Json(i) => &i.definition.column,
            SnapshotIndex::Binary { index, .. } => &index.definition().column,
            SnapshotIndex::Learned { definition, .. } => &definition.column,
        }
    }

    fn lookup(&self, value: &str) -> Option<Vec<u64>> {
        match self {
            SnapshotIndex::Json(i) => Some(i.lookup(value).to_vec()),
            SnapshotIndex::Binary { index, overlay } => {
                Some(overlay.merge(value, index.lookup_checked(value)?))
            }
            SnapshotIndex::Learned { model, .. } => Some(model.lookup(value)),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_binary(&self) -> bool {
        matches!(self, SnapshotIndex::Binary { .. })
    }

    #[cfg(test)]
    pub(crate) fn is_learned(&self) -> bool {
        matches!(self, SnapshotIndex::Learned { .. })
    }
}

impl crate::index_selector::IndexLookup for SnapshotIndex {
    fn column(&self) -> &str {
        SnapshotIndex::column(self)
    }
    fn lookup_ids(&self, value: &str) -> Option<Vec<u64>> {
        SnapshotIndex::lookup(self, value)
    }
}

fn learned_index_file_matches(
    file: &icefalldb_core::index::LearnedIndexFile,
    table: &str,
    name: &str,
) -> bool {
    file.definition.table == table && file.definition.name == name
}

fn binary_index_matches(
    index: &icefalldb_core::index::binary::MmapBinaryIndex,
    table: &str,
    name: &str,
) -> bool {
    index.definition().table == table && index.definition().name == name
}

/// Lightweight per-fragment view used by `try_locate_by_index` — just the
/// fields needed to invert a row id to a physical address and test liveness.
struct FragLocate {
    fragment_id: u64,
    row_ids: Vec<RowIdSegment>,
    deletes: Option<String>,
    deleted_count: u64,
}

/// Inverse of `scan::row_id_at_offset`: the physical offset of `rid` within the
/// fragment's concatenated row-id segments, or `None` if the fragment does not
/// hold it. Segments are concatenated in order, so the offset is the cumulative
/// segment length before the hit plus the position within the hit segment.
fn offset_of_row_id(segs: &[RowIdSegment], rid: u64) -> Option<u32> {
    let mut base: u64 = 0;
    for seg in segs {
        match seg {
            RowIdSegment::Range { start, count } => {
                if rid >= *start && rid < start + count {
                    return Some((base + (rid - start)) as u32);
                }
                base += count;
            }
            RowIdSegment::Sorted { ids } => {
                if let Some(pos) = ids.iter().position(|&x| x == rid) {
                    return Some((base + pos as u64) as u32);
                }
                base += ids.len() as u64;
            }
        }
    }
    None
}

impl IcefallDBTableProvider {
    /// Open a IcefallDB table and pre-compute its exact statistics.
    pub async fn new(
        storage: Arc<dyn Storage>,
        table: impl Into<String>,
        config: ProviderConfig,
    ) -> Result<Self> {
        // Seed the process-wide metadata cache with the configured capacity.
        // The cache is a lazy singleton; this call is idempotent after the
        // first initialization.
        let _ = ParquetMetadataCache::global_with_capacity(config.parquet_metadata_cache_capacity);

        let table = table.into();
        let catalog = IcefallDBCatalog::new(Arc::clone(&storage), &table);
        // O(1)-in-fragments open: read only the schema + the pinned manifest, and
        // defer the per-fragment scan-plan reconstruction to the first query
        // (`current_scan_plan`). Both the (deferred) scan plan and the indexes are
        // resolved from this SAME manifest pointer read (no TOCTOU: the manifest
        // sequence is pinned here and `current_scan_plan` rebuilds at it).
        let (icefalldb_schema, manifest) = catalog.load_schema_and_manifest_allow_empty().await?;
        let arrow_schema = icefalldb_schema
            .arrow_schema()
            .ok_or_else(|| {
                crate::QueryError::Other(format!(
                    "table '{}' has a column with an unsupported Arrow type",
                    table
                ))
            })
            .map(Arc::new)?;
        let full_schema = make_full_schema(&arrow_schema);

        // Exact row count is available cheaply from the manifest's denormalized
        // `row_counts`, so COUNT(*) still folds without building the plan. The
        // per-column min/max stats are deferred (left Absent) until the first
        // scan materializes the plan; pruning correctness is unaffected because
        // `scan()` re-derives stats from the built plan.
        let pinned_sequence = manifest.as_ref().map_or(0, |m| m.sequence);
        let mut statistics = Statistics::new_unknown(&arrow_schema);
        if let Some(n) = manifest.as_ref().and_then(manifest_live_row_count) {
            statistics.num_rows = Precision::Exact(n);
        }
        let placeholder_plan = ScanPlan {
            table: table.clone(),
            schema: icefalldb_schema,
            row_groups: vec![],
        };

        let indexes =
            Self::load_snapshot_indexes(storage.as_ref(), &table, manifest.as_ref(), false).await?;

        Ok(Self::from_snapshot(
            storage,
            table,
            config,
            SnapshotState {
                arrow_schema,
                full_schema,
                scan_plan: placeholder_plan,
                statistics,
                indexes,
                pinned_sequence,
            },
            // Lazy: no pre-seeded plan; built on first `current_scan_plan`.
            None,
        ))
    }

    /// Open a IcefallDB table pinned to a specific historical snapshot.
    ///
    /// Builds the scan plan via [`build_scan_plan_at`] and loads the
    /// secondary indexes from the same historical manifest so the data view and
    /// the index view both correspond to `sequence`. The resulting provider is
    /// read-only for time-travel queries — no `apply_committed_delta` calls are
    /// expected on it.
    ///
    /// Returns [`QueryError::Core`] wrapping
    /// [`icefalldb_core::IcefallDBError::SnapshotNotFound`] when the manifest for
    /// `sequence` is absent or unparseable.
    pub async fn new_at_snapshot(
        storage: Arc<dyn Storage>,
        table: impl Into<String>,
        config: ProviderConfig,
        sequence: u64,
    ) -> Result<Self> {
        let _ = ParquetMetadataCache::global_with_capacity(config.parquet_metadata_cache_capacity);

        let table = table.into();

        // Build the scan plan via the time-travel helper. This reads
        // `{table}/_manifests/{sequence}.json` and the schema-as-of internally.
        let scan_plan = build_scan_plan_at(storage.as_ref(), &table, sequence)
            .await
            .map_err(QueryError::Core)?;

        // Load the historical manifest a second time so index resolution uses
        // `manifest.index_generations` for the pinned snapshot. The manifest is
        // a small JSON file; the extra read is acceptable for a one-time
        // constructor call, and avoids adding a new public API to the reader.
        let manifest_path = format!("{}/{}", table, Manifest::filename(sequence));
        let manifest_bytes = storage
            .read(&manifest_path)
            .await
            .map_err(QueryError::Core)?;
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes).map_err(|e| {
            QueryError::Other(format!(
                "parsing manifest at sequence {sequence} for table '{table}': {e}"
            ))
        })?;

        let arrow_schema = scan_plan
            .schema
            .arrow_schema()
            .ok_or_else(|| {
                QueryError::Other(format!(
                    "table '{}' has a column with an unsupported Arrow type",
                    table
                ))
            })
            .map(Arc::new)?;
        let statistics = scan_plan_statistics(&scan_plan, &arrow_schema);
        let full_schema = make_full_schema(&arrow_schema);

        // Time-travel path: use ONLY the indexes recorded in the historical
        // manifest's `index_generations`.  Legacy flat indexes (`list_index_names`)
        // are excluded — their on-disk state is current (post-snapshot), not
        // historical, so loading them would produce stale as-of results.
        let indexes =
            Self::load_snapshot_indexes(storage.as_ref(), &table, Some(&manifest), true).await?;

        Ok(Self::from_snapshot(
            storage,
            table,
            config,
            SnapshotState {
                arrow_schema,
                full_schema,
                scan_plan: scan_plan.clone(),
                statistics,
                indexes,
                pinned_sequence: sequence,
            },
            // Time travel: pre-seed the cache with the historical plan; the lazy
            // `current_scan_plan` path would otherwise load the LATEST snapshot.
            Some((sequence, scan_plan)),
        ))
    }

    /// Build a provider from an already-loaded [`SnapshotState`].
    ///
    /// This avoids re-reading manifests or sidecars and is used by the batch
    /// mutation path to maintain a write-side snapshot that is refreshed
    /// incrementally after each statement while the public provider is refreshed
    /// exactly once at the end of the batch.
    /// Build a provider from a [`SnapshotState`].
    ///
    /// `scan_cache` pre-seeds the scan-plan cache: pass `Some((seq, plan))` for an
    /// eagerly-built plan (e.g. time travel), or `None` for the lazy live-open
    /// path where `snapshot.scan_plan` is an empty placeholder and the real plan
    /// is built on first `current_scan_plan()`.
    pub(crate) fn from_snapshot(
        storage: Arc<dyn Storage>,
        table: impl Into<String>,
        config: ProviderConfig,
        snapshot: SnapshotState,
        scan_cache: Option<(u64, ScanPlan)>,
    ) -> Self {
        let table = table.into();
        let catalog = IcefallDBCatalog::new(Arc::clone(&storage), &table);
        Self {
            storage,
            table,
            catalog,
            snapshot: std::sync::RwLock::new(snapshot),
            config,
            scan_cache: tokio::sync::RwLock::new(scan_cache),
            apply_lock: tokio::sync::Mutex::new(()),
            tiny_table_cache: tokio::sync::RwLock::new(None),
            #[cfg(test)]
            apply_delta_count: Arc::new(AtomicU64::new(0)),
            #[cfg(feature = "encryption")]
            decryption_properties: None,
        }
    }

    /// Return a clone of the current snapshot state.
    pub(crate) fn snapshot(&self) -> SnapshotState {
        self.snapshot_read().clone()
    }

    /// Open an encrypted IcefallDB table, resolving the table's keys via the
    /// given `KeyProvider`.
    ///
    /// The caller must know the key identifiers (e.g. from `_encryption.json`)
    /// and pass them in. The provider is queried once at construction; the
    /// resolved key set is held for the lifetime of the provider.
    ///
    /// The AAD prefix is loaded from the stored `_encryption.json` marker
    /// (rather than recomputed from a hardcoded schema id) so this constructor
    /// works correctly for tables whose schema has evolved or whose marker
    /// was written with a custom AAD. Falls back to deriving one from the
    /// table name + schema id 1 only if the marker is absent or malformed.
    #[cfg(feature = "encryption")]
    pub async fn new_encrypted(
        storage: Arc<dyn Storage>,
        table: impl Into<String>,
        config: ProviderConfig,
        provider: std::sync::Arc<dyn icefalldb_core::encryption::provider::KeyProvider>,
        footer_key_id: impl Into<icefalldb_core::encryption::KeyIdentifier> + Clone,
        column_key_ids: std::collections::BTreeMap<
            String,
            icefalldb_core::encryption::KeyIdentifier,
        >,
    ) -> Result<Self> {
        let table_str: String = table.into();
        let aad = read_table_aad_prefix(&storage, &table_str).await?;
        let keys = crate::encryption::load_table_keys(
            provider.as_ref(),
            &footer_key_id.into(),
            &column_key_ids,
            &aad,
        )
        .await?;
        let dec = crate::encryption::build_decryption_properties_for_table(&keys)?;

        // We do NOT register the encryption factory on the session here: the
        // caller does that separately via `icefalldb_encrypted_session(...)`.
        // The provider just holds the pre-resolved decryption properties, which
        // the custom scan path uses directly. This avoids the per-scan key
        // lookup that the factory mechanism would require.
        let mut provider = Self::new(Arc::clone(&storage), table_str, config).await?;
        provider.decryption_properties = Some(dec);
        Ok(provider)
    }

    /// Load the secondary indexes that belong to the snapshot this provider is
    /// pinned to.
    ///
    /// `manifest` is the manifest that was already read while building the scan
    /// plan.  Passing it here ensures both the scan plan and the index set come
    /// from the **same** manifest-pointer read (one pointer read total in
    /// `provider::new`), eliminating the TOCTOU window where a concurrent commit
    /// could advance the pointer between the scan-plan load and the index load.
    ///
    /// Each index is loaded from the generation recorded in the **pinned
    /// manifest's** `index_generations` map (via `load_index_by_ref`), so a
    /// reader pinned to snapshot S sees the index view that was committed for S
    /// rather than the latest index file on disk. For a manifest that has no
    /// `index_generations` entry for an index — i.e. a legacy table — the
    /// legacy unversioned `_indexes/<name>.json` file is loaded instead so older
    /// tables keep working.
    ///
    /// When `manifest` is `None` (empty/uninitialized table), no indexes can
    /// exist and an empty `Vec` is returned immediately.
    /// Load the secondary indexes for the pinned snapshot.
    ///
    /// `versioned_only` controls whether legacy unversioned flat indexes are
    /// included:
    ///
    /// * `false` (the live-path default): index names are gathered from both
    ///   `manifest.index_generations` AND the flat `_indexes/<name>.json` scan
    ///   via `list_index_names`.  This preserves back-compat for legacy
    ///   tables that have no `index_generations` entry.
    ///
    /// * `true` (the time-travel path): ONLY names recorded in the historical
    ///   manifest's `index_generations` map are loaded.  Legacy flat indexes are
    ///   intentionally excluded: their on-disk state reflects the *current*
    ///   (latest) write, which may be arbitrarily newer than the pinned snapshot.
    ///   Loading them for an as-of read would produce stale results — e.g. a key
    ///   deleted after the pinned snapshot would appear missing even though it
    ///   existed then.  When no versioned generation exists for an index at the
    ///   requested snapshot, the caller falls back to the stats-pruned scan,
    ///   which is always correct.
    async fn load_snapshot_indexes(
        storage: &dyn Storage,
        table: &str,
        manifest: Option<&Manifest>,
        versioned_only: bool,
    ) -> Result<Arc<Vec<SnapshotIndex>>> {
        // An empty/uninitialized table has no committed manifest and therefore
        // no indexes.
        let manifest = match manifest {
            Some(m) => m,
            None => return Ok(Arc::new(Vec::new())),
        };

        // Collect index names.
        //
        // For the live path (`versioned_only=false`) we gather from both:
        //   * the pinned manifest's `index_generations` keys — the authoritative
        //     source for newer tables, whose versioned base files live at
        //     `_indexes/<name>/base__v<seq>.json` and are NOT discoverable via
        //     the flat `list_index_names` scan; and
        //   * `list_index_names`, which finds legacy unversioned
        //     `_indexes/<name>.json` files from legacy tables.
        //
        // For the time-travel path (`versioned_only=true`) we use ONLY the
        // historical manifest's `index_generations` keys.  Legacy flat indexes
        // are skipped because they reflect the current on-disk state, not the
        // requested snapshot.
        let mut names: HashSet<String> = manifest.index_generations.keys().cloned().collect();
        if !versioned_only {
            for name in list_index_names(storage, table)
                .await
                .map_err(QueryError::Core)?
            {
                names.insert(name);
            }
        }

        let local_root = storage.local_root();
        let mut indexes = Vec::new();
        for name in names {
            if let Some(index_ref) = manifest.index_generations.get(&name) {
                // Learned-model fast path: if the base is an exactly
                // affine key, locate by O(1) arithmetic without loading any
                // postings. Only valid when no deltas have been folded on top
                // (a mutation may have changed the key distribution).
                if index_ref.deltas.is_empty() {
                    if let Some(base_rel) = index_ref.base.as_deref() {
                        if let Some(lf) =
                            icefalldb_core::index::load_learned_model(storage, table, base_rel)
                                .await
                                .map_err(QueryError::Core)?
                        {
                            if learned_index_file_matches(&lf, table, &name) {
                                indexes.push(SnapshotIndex::Learned {
                                    definition: lf.definition,
                                    model: lf.model,
                                });
                                continue;
                            }
                        }
                    }
                }
                // Fast path: mmap the derived binary base instead of parsing the
                // whole JSON map. Only the directory + looked-up postings are
                // paged in, so open cost is independent of index size. The small
                // delta files are replayed into an overlay applied at lookup.
                if let (Some(root), Some(base_rel)) = (local_root, index_ref.base.as_deref()) {
                    if let Some(index) =
                        icefalldb_core::index::binary::open_mmap_binary_index(root, table, base_rel)
                    {
                        if binary_index_matches(&index, table, &name) {
                            let overlay = icefalldb_core::index::load_index_overlay(
                                storage, table, index_ref,
                            )
                            .await
                            .map_err(QueryError::Core)?;
                            indexes.push(SnapshotIndex::Binary { index, overlay });
                            continue;
                        }
                    }
                }
                // Fallback: parse the JSON base (+ apply deltas) for this exact
                // generation. Used for non-local storage or a missing `.idx`.
                if let Some(index) = load_index_by_ref(storage, table, &name, index_ref)
                    .await
                    .map_err(QueryError::Core)?
                {
                    indexes.push(SnapshotIndex::Json(index));
                }
            } else {
                // Back-compat: legacy manifest (or a freshly `create-index`'d
                // table) with no generation entry — the index lives at the legacy
                // unversioned path. Prefer its binary sibling (no deltas exist on
                // this path), else parse the legacy JSON.
                let legacy_rel = icefalldb_core::BTreeIndex::legacy_filename(&name);
                // Learned-model fast path: a never-mutated legacy index (no
                // generation entry → no deltas) with an affine key.
                if let Some(lf) =
                    icefalldb_core::index::load_learned_model(storage, table, &legacy_rel)
                        .await
                        .map_err(QueryError::Core)?
                {
                    if learned_index_file_matches(&lf, table, &name) {
                        indexes.push(SnapshotIndex::Learned {
                            definition: lf.definition,
                            model: lf.model,
                        });
                        continue;
                    }
                }
                if let Some(root) = local_root {
                    if let Some(index) = icefalldb_core::index::binary::open_mmap_binary_index(
                        root,
                        table,
                        &legacy_rel,
                    ) {
                        if binary_index_matches(&index, table, &name) {
                            indexes.push(SnapshotIndex::Binary {
                                index,
                                overlay: icefalldb_core::index::IndexDeltaOverlay::default(),
                            });
                            continue;
                        }
                    }
                }
                if let Some(index) = load_index(storage, table, &name)
                    .await
                    .map_err(QueryError::Core)?
                {
                    indexes.push(SnapshotIndex::Json(index));
                }
            }
        }
        Ok(Arc::new(indexes))
    }

    /// Fast-path row location for a DELETE/UPDATE predicate using the in-memory
    /// secondary index + row-id segments, bypassing a DataFusion scan entirely.
    ///
    /// Returns `Ok(Some(locs))` only when `predicate` is a single indexed
    /// equality / `IN` this path can answer *exactly* — `index_equality_lookup`
    /// matches a bare `col = lit` / `col IN (..)`, so the index fully covers the
    /// predicate with no residual filter. Returns `Ok(None)` (caller falls back
    /// to the scan path) when the predicate is not index-answerable, when a
    /// candidate row id is unresolvable, or when it resolves to more than one
    /// live location (defensive: never guess on an inconsistency).
    ///
    /// Liveness: a candidate is emitted only if exactly one of its physical
    /// locations is *not* covered by that fragment's deletion vector. This
    /// correctly handles UPDATE patch fragments — the pre-image offset is
    /// tombstoned in the old fragment's DV while the post-image carries the same
    /// row id live in the patch fragment.
    pub(crate) async fn try_locate_by_index(
        &self,
        predicate: &Expr,
    ) -> Result<Option<Vec<MatchLoc>>> {
        // Materialize/refresh the scan plan FIRST so the indexes read below are
        // re-pinned consistently with it (same ordering as `scan()`): on an
        // external manifest advance, reading indexes before this would match
        // stale open-time row-ids against the freshly-loaded fragment list and
        // silently under-locate the externally-added rows.
        let plan = self.current_scan_plan().await?;

        // Resolve candidate row ids from the (now re-pinned) indexes under the
        // snapshot read lock.
        let row_ids = {
            let snap = self.snapshot_read();
            let mut ids: Option<HashSet<u64>> = None;
            for index in snap.indexes.iter() {
                if let Some(found) = index_selector::index_equality_lookup(index, predicate) {
                    ids = Some(found);
                    break;
                }
            }
            ids
        };
        let Some(row_ids) = row_ids else {
            return Ok(None);
        };
        // Clone only the row-id segments (not full meta) from the pinned plan.
        let frags: Vec<FragLocate> = plan
            .row_groups
            .iter()
            .map(|rg| FragLocate {
                fragment_id: rg.fragment_id,
                row_ids: rg.meta.row_ids.clone(),
                deletes: rg.deletes.clone(),
                deleted_count: rg.deleted_count,
            })
            .collect();

        // Each fragment's deletion vector is read at most once.
        let mut dv_cache: HashMap<u64, Option<DeletionVector>> = HashMap::new();
        let mut out = Vec::with_capacity(row_ids.len());
        for rid in row_ids {
            let mut live: Option<MatchLoc> = None;
            let mut live_count = 0u32;
            for frag in &frags {
                let Some(offset) = offset_of_row_id(&frag.row_ids, rid) else {
                    continue;
                };
                if let Some(path) = &frag.deletes {
                    if frag.deleted_count > 0 && !dv_cache.contains_key(&frag.fragment_id) {
                        let bytes = self.storage.read(path).await.map_err(QueryError::Core)?;
                        let dv = DeletionVector::deserialize(&bytes).map_err(|e| {
                            QueryError::Other(format!("deletion vector decode: {e}"))
                        })?;
                        dv_cache.insert(frag.fragment_id, Some(dv));
                    }
                }
                let deleted = dv_cache
                    .get(&frag.fragment_id)
                    .and_then(|d| d.as_ref())
                    .map(|d| d.contains(offset))
                    .unwrap_or(false);
                if !deleted {
                    live_count += 1;
                    live = Some(MatchLoc {
                        fragment_id: frag.fragment_id,
                        offset,
                        row_id: rid,
                    });
                }
            }
            match live_count {
                0 => {} // deleted or absent: not a live match
                1 => out.push(live.expect("live set when count==1")),
                _ => return Ok(None), // ambiguous: fall back to scan
            }
        }
        Ok(Some(out))
    }

    /// Return the table name.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Return the materialized scan plan for the pinned snapshot, building it on
    /// first access (open is lazy, so the in-memory `SnapshotState.scan_plan` may
    /// be an empty placeholder until the first query).
    pub async fn scan_plan(&self) -> Result<ScanPlan> {
        self.current_scan_plan().await
    }

    /// Return the manifest sequence that both the scan plan and the secondary
    /// indexes were loaded from.  `0` for empty/uninitialized tables.
    ///
    /// Because `new` reads the manifest pointer exactly once and passes the
    /// resulting `Manifest` to both the scan-plan builder and the index loader,
    /// this value is guaranteed to equal the sequence embedded in every
    /// `PlannedRowGroup::snapshot` field (for non-empty tables), confirming
    /// there is no TOCTOU split between the two data sets.
    pub fn pinned_sequence(&self) -> u64 {
        self.snapshot_read().pinned_sequence
    }

    /// Return the number of times `apply_committed_delta` has refreshed this
    /// provider. Available only in test builds so production code cannot depend
    /// on this counter.
    #[cfg(test)]
    pub fn apply_delta_count(&self) -> u64 {
        self.apply_delta_count.load(Ordering::SeqCst)
    }

    /// Return a reference to the provider configuration.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// Acquire a read lock on `snapshot`, recovering from poison so a prior
    /// panic while holding the lock does not make later calls panic.
    fn snapshot_read(&self) -> std::sync::RwLockReadGuard<'_, SnapshotState> {
        self.snapshot.read().unwrap_or_else(|p| p.into_inner())
    }

    /// Test-only: which of the pinned snapshot's indexes are mmap-backed binary
    /// (vs JSON-parsed). Used to assert the mmap'd binary open path is taken.
    #[cfg(test)]
    pub(crate) fn test_indexes_are_binary(&self) -> Vec<bool> {
        self.snapshot_read()
            .indexes
            .iter()
            .map(|i| i.is_binary())
            .collect()
    }

    /// Test-only: which pinned indexes are learned-model.
    #[cfg(test)]
    pub(crate) fn test_indexes_are_learned(&self) -> Vec<bool> {
        self.snapshot_read()
            .indexes
            .iter()
            .map(|i| i.is_learned())
            .collect()
    }

    /// Test-only: how many secondary indexes are loaded for the pinned snapshot.
    ///
    /// Used to assert that `new_at_snapshot` loads only the indexes recorded in
    /// the historical manifest's `index_generations` and does not surface
    /// additional legacy flat indexes discovered via `list_index_names`.
    #[cfg(test)]
    pub(crate) fn test_loaded_index_count(&self) -> usize {
        self.snapshot_read().indexes.len()
    }

    /// Acquire a write lock on `snapshot`, recovering from poison.
    fn snapshot_write(&self) -> std::sync::RwLockWriteGuard<'_, SnapshotState> {
        self.snapshot.write().unwrap_or_else(|p| p.into_inner())
    }

    /// Load the scan plan for the snapshot this provider is pinned to.
    ///
    /// A cached copy is returned when its sequence matches the provider's
    /// `pinned_sequence`, avoiding any `_manifest.json` or `.meta` read. On a
    /// cache miss the manifest pointer is read and the snapshot is reloaded from
    /// the catalog.
    async fn current_scan_plan(&self) -> Result<ScanPlan> {
        // Bounded retry: a concurrent query/apply can advance the snapshot past
        // the sequence we just loaded, in which case `repin_snapshot` skips its
        // (rollback-guarded) install and we must NOT cache/return a plan that
        // `scan()`/`try_locate` would pair with the newer indexes/stats. Concurrent
        // advances are finite, so this converges in one or two iterations.
        for _ in 0..16 {
            let pinned = self.pinned_sequence();

            {
                let cache = self.scan_cache.read().await;
                if let Some((seq, plan)) = cache.as_ref() {
                    if *seq == pinned {
                        return Ok(plan.clone());
                    }
                }
            }

            // Miss: build the plan. Use the LOADED manifest sequence (WAL-folded,
            // same basis as `pinned_sequence`) as the cache key, not the on-disk
            // pointer which can lag in WAL mode.
            let (plan, schema, manifest) = self
                .catalog
                .load_snapshot_allow_empty_with_manifest()
                .await?;
            let latest = manifest.as_ref().map_or(0, |m| m.sequence);

            // If the manifest advanced since this provider was pinned (e.g. a
            // different process holding the write lock committed), re-pin the
            // indexes, statistics, and sequence together so the served snapshot
            // stays internally consistent: a fresh plan must not be paired with
            // the open-time indexes/stats (that would drop rows from an indexed
            // lookup and disagree with COUNT(*)). The common case
            // (`latest == pinned`) keeps the existing indexes/stats.
            if latest != pinned {
                self.repin_snapshot(&schema, manifest.as_ref(), latest)
                    .await?;
            }

            // Only cache/return once the snapshot is actually at `latest`. If a
            // concurrent operation advanced it past `latest` during the load/repin
            // (so `repin_snapshot` skipped its install), retry rather than cache a
            // plan inconsistent with the now-newer indexes/stats.
            if self.pinned_sequence() != latest {
                continue;
            }
            let mut cache = self.scan_cache.write().await;
            *cache = Some((latest, plan.clone()));
            return Ok(plan);
        }

        // Pathological persistent contention: fall back to a direct load so the
        // call still makes progress; the next call settles into the cache.
        let (plan, _schema, _manifest) = self
            .catalog
            .load_snapshot_allow_empty_with_manifest()
            .await?;
        Ok(plan)
    }

    /// Refresh `SnapshotState` (schema, statistics, indexes, sequence) to `latest`
    /// so the provider serves a consistent snapshot after the manifest advanced
    /// out from under it (a different process committed; this provider's own
    /// mutations advance the sequence through `apply_committed_delta` instead).
    /// Per-column statistics stay deferred (Absent) like the open path; only the
    /// exact row count is restored from the manifest. The scan plan stays a
    /// placeholder — the freshly-built plan is cached by the caller. Guarded so a
    /// concurrent newer `apply` is never clobbered.
    async fn repin_snapshot(
        &self,
        schema: &Schema,
        manifest: Option<&Manifest>,
        latest: u64,
    ) -> Result<()> {
        let arrow_schema = Arc::new(schema.arrow_schema().ok_or_else(|| {
            QueryError::Other(format!(
                "table '{}' has a column with an unsupported Arrow type",
                self.table
            ))
        })?);
        let full_schema = make_full_schema(&arrow_schema);
        let indexes =
            Self::load_snapshot_indexes(self.storage.as_ref(), &self.table, manifest, false)
                .await?;
        let mut statistics = Statistics::new_unknown(&arrow_schema);
        if let Some(n) = manifest.and_then(manifest_live_row_count) {
            statistics.num_rows = Precision::Exact(n);
        }

        let mut guard = self.snapshot_write();
        // Only advance; never roll back over a concurrently-applied newer state.
        if latest >= guard.pinned_sequence {
            guard.arrow_schema = arrow_schema;
            guard.full_schema = full_schema;
            guard.scan_plan = ScanPlan {
                table: self.table.clone(),
                schema: schema.clone(),
                row_groups: vec![],
            };
            guard.statistics = statistics;
            guard.indexes = indexes;
            guard.pinned_sequence = latest;
        }
        Ok(())
    }

    /// Incrementally update the provider's snapshot state from a committed
    /// [`CommitDelta`] without re-reading the manifest or `.meta` sidecars.
    ///
    /// This is the hot-path refresh used by the SQL mutation path
    /// (`DELETE`/`UPDATE`/`MERGE`). It mutates snapshot state through `&self`
    /// using interior mutability.
    pub async fn apply_committed_delta(&self, delta: &CommitDelta) -> Result<()> {
        if delta.is_noop() {
            return Ok(());
        }

        // Serialize concurrent incremental refreshes so two mutation tasks cannot
        // both read the old `SnapshotState` and race to install inconsistent state.
        let _apply_guard = self.apply_lock.lock().await;

        // Defensive check: a stale delta (e.g., from a delayed task or a reused
        // delta object) must not be applied on top of a newer snapshot.
        let pinned = self.pinned_sequence();
        if pinned != delta.previous_sequence {
            return Err(QueryError::Other(format!(
                "provider pinned to sequence {pinned} but delta starts from {}; refresh stale",
                delta.previous_sequence
            )));
        }

        // The incremental path needs the pre-delta base plan in hand. A warm
        // provider has it cached at `pinned` (seeded at open/time-travel, or by a
        // prior scan/apply). A lazily-opened provider that has never been scanned
        // (e.g. a daemon's public provider refreshed straight after a /tx/commit)
        // has no cached base, and after the commit the manifest+WAL already
        // reflect the delta — so there is no pre-delta state to read. For that
        // cold case, invalidate the caches and bump the pinned sequence; the next
        // query rebuilds from the post-commit snapshot via `current_scan_plan`.
        let warm = {
            let cache = self.scan_cache.read().await;
            matches!(cache.as_ref(), Some((seq, _)) if *seq == pinned)
        };
        if !warm {
            return self.invalidate_to_committed_delta(delta).await;
        }

        // Warm path: the cached base is the pre-delta plan at `pinned` (cache hit,
        // no manifest/.meta read), keeping the incremental refresh invariant.
        let base_plan = self.current_scan_plan().await?;

        // Read the current schemas and build the updated row-group list. Keep the
        // read lock inside a tight scope so it is never held across an await point.
        let (
            current_arrow_schema,
            current_full_schema,
            current_scan_plan_schema,
            schema_changed,
            new_row_groups,
        ) = {
            let current = self.snapshot_read();
            let removed: std::collections::HashSet<u64> =
                delta.removed_fragment_ids().into_iter().collect();
            let mut new_row_groups: Vec<PlannedRowGroup> = base_plan
                .row_groups
                .iter()
                .filter(|rg| !removed.contains(&rg.fragment_id))
                .cloned()
                .collect();

            let updated: std::collections::HashMap<u64, FragmentDelta> = delta
                .updated_fragments()
                .into_iter()
                .map(|fd| (fd.fragment_id, fd))
                .collect();
            for rg in &mut new_row_groups {
                if let Some(fd) = updated.get(&rg.fragment_id) {
                    rg.deletes = fd
                        .new_deletes
                        .as_ref()
                        .map(|p| format!("{}/{}", self.table, p));
                    rg.deleted_count = fd.new_deleted_count;
                }
            }

            new_row_groups.extend(delta.added_row_groups.iter().cloned());

            (
                Arc::clone(&current.arrow_schema),
                Arc::clone(&current.full_schema),
                base_plan.schema.clone(),
                delta.schema_changed(),
                new_row_groups,
            )
        };

        let (arrow_schema, full_schema, scan_plan_schema) = if schema_changed {
            let schema_path = format!(
                "{}/{}",
                self.table,
                Schema::filename(delta.new_manifest.schema_id)
            );
            let schema_bytes = self
                .storage
                .read(&schema_path)
                .await
                .map_err(QueryError::Core)?;
            let icefalldb_schema: Schema = serde_json::from_slice(&schema_bytes)
                .map_err(|e| QueryError::Other(format!("schema parse error: {e}")))?;
            let arrow_schema = Arc::new(icefalldb_schema.arrow_schema().ok_or_else(|| {
                QueryError::Other(format!(
                    "table '{}' has a column with an unsupported Arrow type",
                    self.table
                ))
            })?);
            let full_schema = make_full_schema(&arrow_schema);
            (Arc::clone(&arrow_schema), full_schema, icefalldb_schema)
        } else {
            (
                current_arrow_schema,
                current_full_schema,
                current_scan_plan_schema,
            )
        };

        let new_scan_plan = ScanPlan {
            table: self.table.clone(),
            schema: scan_plan_schema,
            row_groups: new_row_groups,
        };
        let new_statistics = scan_plan_statistics(&new_scan_plan, &arrow_schema);

        let new_indexes = Self::load_snapshot_indexes(
            self.storage.as_ref(),
            &self.table,
            Some(&delta.new_manifest),
            false, // live path: include legacy flat indexes for back-compat
        )
        .await?;

        // Cache the incrementally-built scan plan at the new manifest sequence
        // so the next query's `current_scan_plan()` returns it without reading
        // `_manifest.json` or `.meta` sidecars.
        let scan_cache_entry = (delta.new_sequence, new_scan_plan.clone());

        let new_state = SnapshotState {
            arrow_schema,
            full_schema,
            scan_plan: new_scan_plan,
            statistics: new_statistics,
            indexes: new_indexes,
            pinned_sequence: delta.new_sequence,
        };

        // Replace the snapshot state inside a tight scope so the write lock is
        // never held across an await point.
        {
            let mut guard = self.snapshot_write();
            *guard = new_state;
        }

        // Seed the scan cache with the incremental plan; clear the tiny-table
        // cache because it holds materialized rows from the previous snapshot.
        {
            let mut cache = self.scan_cache.write().await;
            *cache = Some(scan_cache_entry);
        }
        {
            let mut cache = self.tiny_table_cache.write().await;
            *cache = None;
        }

        #[cfg(test)]
        self.apply_delta_count.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Cold-provider refresh for `apply_committed_delta`: the provider was never
    /// scanned, so there is no in-memory pre-delta base to update incrementally.
    /// Reset the snapshot to an empty placeholder at the new sequence (loading the
    /// new schema only if the delta changed it) and drop the caches; the next
    /// query rebuilds from the post-commit snapshot via `current_scan_plan`. This
    /// stays O(1) in fragment count (no scan-plan build here).
    async fn invalidate_to_committed_delta(&self, delta: &CommitDelta) -> Result<()> {
        let (arrow_schema, full_schema, scan_plan_schema) = if delta.schema_changed() {
            let schema_path = format!(
                "{}/{}",
                self.table,
                Schema::filename(delta.new_manifest.schema_id)
            );
            let schema_bytes = self
                .storage
                .read(&schema_path)
                .await
                .map_err(QueryError::Core)?;
            let icefalldb_schema: Schema = serde_json::from_slice(&schema_bytes)
                .map_err(|e| QueryError::Other(format!("schema parse error: {e}")))?;
            let arrow_schema = Arc::new(icefalldb_schema.arrow_schema().ok_or_else(|| {
                QueryError::Other(format!(
                    "table '{}' has a column with an unsupported Arrow type",
                    self.table
                ))
            })?);
            let full_schema = make_full_schema(&arrow_schema);
            (arrow_schema, full_schema, icefalldb_schema)
        } else {
            let current = self.snapshot_read();
            (
                Arc::clone(&current.arrow_schema),
                Arc::clone(&current.full_schema),
                current.scan_plan.schema.clone(),
            )
        };

        let new_indexes = Self::load_snapshot_indexes(
            self.storage.as_ref(),
            &self.table,
            Some(&delta.new_manifest),
            false,
        )
        .await?;

        let statistics = Statistics::new_unknown(&arrow_schema);
        let placeholder = ScanPlan {
            table: self.table.clone(),
            schema: scan_plan_schema,
            row_groups: vec![],
        };
        {
            let mut guard = self.snapshot_write();
            *guard = SnapshotState {
                arrow_schema,
                full_schema,
                scan_plan: placeholder,
                statistics,
                indexes: new_indexes,
                pinned_sequence: delta.new_sequence,
            };
        }
        {
            let mut cache = self.scan_cache.write().await;
            *cache = None;
        }
        {
            let mut cache = self.tiny_table_cache.write().await;
            *cache = None;
        }

        #[cfg(test)]
        self.apply_delta_count.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Replace this provider's snapshot state with the final state of `source`.
    ///
    /// Used at the end of a batched mutation: the write-side context's provider
    /// is refreshed incrementally after each statement, and this method copies
    /// that final snapshot to the public provider exactly once.  This avoids the
    /// complexity of merging a sequence of deltas while still performing only a
    /// single public refresh and zero extra manifest/sidecar reads.
    pub(crate) async fn replace_snapshot_from(
        &self,
        source: &IcefallDBTableProvider,
    ) -> Result<()> {
        let _apply_guard = self.apply_lock.lock().await;

        // Materialize the source's plan rather than reading `source.snapshot()
        // .scan_plan` directly: with lazy open the latter is an empty placeholder
        // until the source is scanned/applied. Every current caller has already
        // applied a delta (so this is a cache hit / no-op), but materializing here
        // keeps the public provider correct even for a future caller whose source
        // was never warmed — otherwise it would install a zero-row plan.
        let new_plan = source.scan_plan().await?;
        let mut new_state = source.snapshot();
        let new_seq = new_state.pinned_sequence;
        // Keep the installed SnapshotState's plan consistent with the seeded cache.
        new_state.scan_plan = new_plan.clone();

        {
            let mut guard = self.snapshot_write();
            *guard = new_state;
        }
        {
            let mut cache = self.scan_cache.write().await;
            *cache = Some((new_seq, new_plan));
        }
        {
            let mut cache = self.tiny_table_cache.write().await;
            *cache = None;
        }

        #[cfg(test)]
        self.apply_delta_count.fetch_add(1, Ordering::SeqCst);

        Ok(())
    }

    /// Return the declared low-cardinality GROUP BY keys for this table
    /// (`Schema.agg_group_keys`), or an empty vector when none are declared.
    /// Threaded into `IcefallDBScanExec` so the `MetadataAggregate` rule can
    /// compose warm GROUP BY results from cached per-group partials.
    fn declared_agg_group_keys(&self) -> Vec<String> {
        self.snapshot_read()
            .scan_plan
            .schema
            .agg_group_keys
            .clone()
            .unwrap_or_default()
    }

    /// Estimate the total rows and bytes of `plan`. Returns `None` if any
    /// row-group data file size cannot be determined, which prevents a size
    /// failure from making a large table look tiny.
    ///
    /// If the row count alone already exceeds the row threshold, file sizes are
    /// not queried so that large tables avoid unnecessary I/O on every scan.
    async fn estimate_tiny_table_size(&self, plan: &ScanPlan) -> Option<(usize, u64)> {
        let mut total_rows: usize = 0;
        for rg in &plan.row_groups {
            total_rows = total_rows.saturating_add(rg.meta.rows);
            if total_rows > self.config.tiny_table_cache_threshold_rows {
                return Some((total_rows, u64::MAX));
            }
        }
        let mut total_bytes: u64 = 0;
        for rg in &plan.row_groups {
            let size = self.storage.size(&rg.data_path).await.ok()?;
            total_bytes = total_bytes.saturating_add(size);
        }
        Some((total_rows, total_bytes))
    }

    /// Load the full table described by `row_groups` into a DataFusion
    /// `MemTable`. Returns `Ok(None)` for empty tables; returns `Err` if the
    /// scan or MemTable construction fails.
    async fn build_tiny_mem_table(
        &self,
        row_groups: &[PlannedRowGroup],
    ) -> Result<Option<MemTable>> {
        let arrow_schema = Arc::clone(&self.snapshot_read().arrow_schema);
        let total_rows: usize = row_groups.iter().map(|rg| rg.meta.rows).sum();
        if total_rows == 0 {
            return Ok(None);
        }

        let exec = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&self.storage),
            Arc::clone(&arrow_schema),
            row_groups.to_vec(),
            None,   // load all columns
            vec![], // no pushed-down filters; the MemTable will apply them
            None,   // no limit; the MemTable will apply it
            self.config.batch_size,
            self.config.io_coalesce_window,
            self.config.io_concurrency,
            self.config.target_partitions,
        )?;

        let batches = collect_plan(Arc::new(exec), Arc::new(TaskContext::default())).await?;

        let mem_table = MemTable::try_new(Arc::clone(&arrow_schema), vec![batches])?;

        Ok(Some(mem_table))
    }

    /// Build a physical plan over a cached `MemTable` that applies the same
    /// projection, filters, and limit that DataFusion pushed down. Filters are
    /// evaluated with a `FilterExec` because `MemTable` ignores pushed filters;
    /// projection is applied with a `ProjectionExec` so that filter columns not
    /// in the output can still be referenced.
    async fn build_tiny_table_plan(
        &self,
        mem_table: &MemTable,
        session: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let arrow_schema = Arc::clone(&self.snapshot_read().arrow_schema);
        // Load all columns from the MemTable so filters can reference any column.
        let mut plan = mem_table.scan(session, None, &[], limit).await?;

        if !filters.is_empty() {
            let df_schema = DFSchema::try_from(arrow_schema.as_ref().clone())?;
            let props = ExecutionProps::new();
            let mut physical_filters = Vec::with_capacity(filters.len());
            for filter in filters {
                physical_filters.push(create_physical_expr(filter, &df_schema, &props)?);
            }
            let predicate = combine_physical_filters(physical_filters, arrow_schema.as_ref())?;
            plan = Arc::new(FilterExec::try_new(predicate, plan)?);
        }

        if let Some(indices) = projection {
            let mut exprs = Vec::with_capacity(indices.len());
            for &idx in indices {
                let field = arrow_schema.field(idx);
                exprs.push(ProjectionExpr::new(
                    Arc::new(Column::new(field.name(), idx)),
                    field.name(),
                ));
            }
            plan = Arc::new(ProjectionExec::try_new(exprs, plan)?);
        }

        Ok(plan)
    }
}

impl fmt::Debug for IcefallDBTableProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcefallDBTableProvider")
            .field("table", &self.table)
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for IcefallDBTableProvider {
    fn schema(&self) -> SchemaRef {
        // Return the full schema including the _rowid and _rowaddr pseudo-column
        // fields so DataFusion can resolve them when they are explicitly named.
        // Data-column indices (0..N-1) are identical between `full_schema` and
        // `arrow_schema`, keeping all pruning/stats logic unaffected.
        Arc::clone(&self.snapshot_read().full_schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        let snapshot = self.snapshot_read();
        let partition_by = snapshot.scan_plan.schema.partition_by.as_deref();
        Ok(filters
            .iter()
            .map(|filter| classify_filter(filter, &snapshot.arrow_schema, partition_by))
            .collect())
    }

    fn statistics(&self) -> Option<Statistics> {
        Some(self.snapshot_read().statistics.clone())
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        // Materialize/refresh the scan plan FIRST. On a lazily-opened provider —
        // or one whose manifest advanced externally — `current_scan_plan` re-pins
        // the snapshot's schema/indexes/statistics to match the loaded plan, so
        // the fields read below are consistent with it (reading indexes BEFORE
        // this would pair a fresh plan with the stale open-time indexes and drop
        // rows from an indexed lookup).
        let scan_plan = self
            .current_scan_plan()
            .await
            .map_err(|e: crate::QueryError| DataFusionError::External(Box::new(e)))?;

        // Read the snapshot-dependent fields we need (after any re-pin) and
        // release the lock before any further await point.
        let (arrow_schema, full_schema, indexes) = {
            let snapshot = self.snapshot_read();
            (
                Arc::clone(&snapshot.arrow_schema),
                Arc::clone(&snapshot.full_schema),
                Arc::clone(&snapshot.indexes),
            )
        };

        // Detect whether the projection requests any pseudo-columns.  The full
        // schema returned by `schema()` places `_rowid` at index N and `_rowaddr`
        // at index N+1 (where N = data column count).  Indices below N are pure
        // data-column references and are valid in both `full_schema` and
        // `arrow_schema`.  Any index >= N means a pseudo-column is requested and
        // the tiny-table and native-Parquet paths must be bypassed (they only
        // understand real Parquet columns).
        let n_data = arrow_schema.fields().len();
        let has_pseudo_in_projection = projection
            .map(|p| p.iter().any(|&idx| idx >= n_data))
            .unwrap_or(false);

        // Capture the full set of row groups before pruning so that the
        // tiny-table cache can store the whole table and remain reusable across
        // queries with different filters/projections.
        let full_row_groups = scan_plan.row_groups.clone();

        // Translate every filter we understand; prune whole row groups with the
        // predicates that sidecar stats can decide exactly.  Always use
        // `arrow_schema` (data-only) for predicate translation — pseudo-columns
        // are not present in the Parquet files and cannot be predicate-pruned.
        let mut predicates: Vec<Predicate> = Vec::with_capacity(filters.len());
        for filter in filters {
            if let Some((_, pred)) = expr_to_predicate(filter, &arrow_schema) {
                predicates.push(pred);
            }
        }
        let mut pruned = scan_plan.prune(&predicates).map_err(QueryError::Core)?;

        // Apply index pruning AFTER stats/partition pruning, and only when more
        // than one row group survives to refine. This skips redundant index
        // work when partition/stats pruning already isolated the matches (e.g.
        // an equality on a partition column). The index prunes whole row groups
        // via `retain_row_groups_by_row_ids`; combined with the O(log N)
        // segment-overlap test there, indexed equality no longer expands a
        // `Range { count }` segment into per-row hash probes.
        if pruned.row_groups.len() > 1 {
            let mut candidate_row_ids: Option<HashSet<u64>> = None;
            for index in indexes.iter() {
                for filter in filters {
                    if let Some(ids) = index_selector::index_equality_lookup(index, filter) {
                        candidate_row_ids = Some(match candidate_row_ids {
                            Some(existing) => existing.intersection(&ids).copied().collect(),
                            None => ids,
                        });
                    }
                }
            }
            if let Some(ids) = candidate_row_ids {
                pruned.retain_row_groups_by_row_ids(&ids);
            }
        }

        // Encrypted tables bypass the native reader and the tiny-table cache.
        // The custom scan path threads `FileDecryptionProperties` into the Parquet
        // arrow reader directly, while the native `ParquetSource` path requires
        // per-scan factory configuration that our session does not currently do.
        // We also avoid holding decrypted data in memory longer than necessary.
        #[cfg(feature = "encryption")]
        let encryption_active = self.decryption_properties.is_some();
        #[cfg(not(feature = "encryption"))]
        let encryption_active = false;

        // Tiny tables are loaded once into a DataFusion MemTable and then served
        // from memory. This avoids repeated Parquet decoding for small tables
        // while still letting DataFusion apply projections, filters, and limits.
        //
        // DELETION-VECTOR INVARIANT: DV correctness for cached tiny tables depends
        // on `build_tiny_mem_table` materializing through `IcefallDBScanExec`, which
        // applies each fragment's `.del` deletion vector as a `RowSelection` during
        // Parquet decoding. Any future change to that materialization path (e.g. a
        // direct-Parquet shortcut) MUST preserve deletion-vector application or
        // deleted rows will silently appear in the cached MemTable and be served to
        // all subsequent queries. See `tiny_table_cache_skips_deleted_rows` in
        // provider.rs tests.
        //
        // The tiny-table cache holds only data columns (built from `arrow_schema`).
        // When pseudo-columns are projected we must bypass the cache entirely and
        // let the custom scan synthesize them.
        if !encryption_active && !has_pseudo_in_projection && !full_row_groups.is_empty() {
            let sequence = pruned
                .row_groups
                .first()
                .map(|rg| rg.snapshot)
                .unwrap_or_else(|| full_row_groups.first().map(|rg| rg.snapshot).unwrap_or(0));

            // Check for a cached MemTable at the current manifest sequence.
            {
                let cache = self.tiny_table_cache.read().await;
                if let Some((seq, mem_table)) = cache.as_ref() {
                    if *seq == sequence {
                        if let Ok(plan) = self
                            .build_tiny_table_plan(mem_table, state, projection, filters, limit)
                            .await
                        {
                            return Ok(plan);
                        }
                        // Plan building failed; fall through and reload.
                    }
                }
            }

            // If the pruned result set is tiny, load the full table into memory.
            if let Some((total_rows, total_bytes)) = self.estimate_tiny_table_size(&pruned).await {
                if total_rows <= self.config.tiny_table_cache_threshold_rows
                    || total_bytes <= self.config.tiny_table_cache_threshold_bytes as u64
                {
                    match self.build_tiny_mem_table(&full_row_groups).await {
                        Ok(Some(mem_table)) => {
                            if let Ok(plan) = self
                                .build_tiny_table_plan(
                                    &mem_table, state, projection, filters, limit,
                                )
                                .await
                            {
                                let cached = Arc::new(mem_table);
                                let mut cache = self.tiny_table_cache.write().await;
                                *cache = Some((sequence, cached));
                                return Ok(plan);
                            }
                            // Fall through to the normal scan path if plan
                            // construction fails. Do not poison the cache.
                        }
                        Ok(None) | Err(_) => {
                            // Fall through to the normal scan path on any load
                            // failure or empty table. Do not poison the cache.
                        }
                    }
                }
            }
        }

        // For wide-schema selective scans, DataFusion's native Parquet reader is
        // faster than the custom IcefallDBScanExec because it can exploit page-index
        // pruning and morsel-based parallelism across files. The native path is
        // skipped for encrypted tables (see above) and for index-pruned scans, where
        // the custom path avoids the scheduling overhead of DataFusion's native
        // reader for the small set of surviving row groups.

        // The native ParquetSource path has no per-file RowSelection injection
        // capability, so it cannot honour deletion vectors. Skip it when any
        // surviving row group carries a deletion vector.
        //
        // The native path also does not synthesize pseudo-columns; skip it
        // whenever any pseudo-column index appears in the projection.
        let has_deletions = pruned.row_groups.iter().any(|rg| rg.deletes.is_some());

        // For the native-parquet path, supply only data-column projection indices.
        // Pseudo-column indices (>= n_data) are silently ignored here because
        // `has_pseudo_in_projection` causes us to skip this path entirely.
        let native_config = NativeParquetConfig {
            min_filter_columns: self.config.native_parquet_threshold,
        };
        if !encryption_active
            && !has_deletions
            && !has_pseudo_in_projection
            && should_use_native_parquet(
                &self.storage,
                &pruned,
                projection,
                filters,
                &arrow_schema,
                self.config.target_partitions,
                native_config,
            )
        {
            // When the projection is a subset of the filter columns the Parquet
            // row filter can late-materialize nothing, so bulk-decode + a
            // `FilterExec` beats the selective-decode path. Statistics/page-index
            // pruning is preserved (the predicate is still supplied to the scan).
            //
            // Skip the bulk path when a `limit` is present: DataFusion pushes the
            // limit into the scan (the filter is `Exact`), which is only valid
            // when the scan also applies the filter. With the row filter off the
            // limit would cap *pre-filter* rows, returning fewer than `limit`
            // matches. The pushdown path keeps the limit correct and can
            // early-terminate, which is also faster for small limits.
            let native_bulk_decode = limit.is_none()
                && state
                    .config()
                    .options()
                    .extensions
                    .get::<IcefallDBConfig>()
                    .map(|c| c.native_bulk_decode)
                    .unwrap_or(true);
            if native_bulk_decode {
                if let Some(scan_proj) =
                    native_bulk_decode_projection(projection, filters, &arrow_schema)
                {
                    return build_native_bulk_decode_plan(
                        &self.storage,
                        &pruned,
                        &arrow_schema,
                        projection,
                        &scan_proj,
                        filters,
                        self.config.batch_size,
                        self.config.target_partitions,
                    );
                }
            }
            return build_native_parquet_exec(
                &self.storage,
                &pruned,
                &arrow_schema,
                projection,
                filters,
                limit,
                self.config.batch_size,
                self.config.target_partitions,
                true,
            )
            .map_err(|e: crate::QueryError| DataFusionError::External(Box::new(e)));
        }

        // Convert all pushed-down filters into physical expressions for the scan
        // to evaluate on the actual row data.  Use `arrow_schema` (data-only) so
        // that column index references in the physical expressions match the
        // Parquet schema, which does not contain pseudo-columns.
        let df_schema = DFSchema::try_from(arrow_schema.as_ref().clone())?;
        let props = ExecutionProps::new();
        let mut physical_filters = Vec::with_capacity(filters.len());
        for filter in filters {
            match create_physical_expr(filter, &df_schema, &props) {
                Ok(expr) => physical_filters.push(expr),
                Err(e) => return Err(e),
            }
        }

        // Pass `full_schema` to `IcefallDBScanExec` so that projection indices
        // pointing at pseudo-column slots (>= n_data) are correctly resolved to
        // `_rowid`/`_rowaddr` and the scan synthesizes those columns.  The
        // projection vector from DataFusion already uses `full_schema` indices.
        #[cfg(feature = "encryption")]
        if let Some(dec) = &self.decryption_properties {
            let exec = IcefallDBScanExec::new_encrypted(
                Arc::clone(&self.storage),
                Arc::clone(&full_schema),
                pruned.row_groups,
                projection.cloned(),
                physical_filters,
                limit,
                self.config.batch_size,
                self.config.io_coalesce_window,
                self.config.io_concurrency,
                self.config.target_partitions,
                Arc::clone(dec),
            )
            .map_err(|e: crate::QueryError| DataFusionError::External(Box::new(e)))?;
            let exec = exec.with_agg_group_keys(self.declared_agg_group_keys());
            return Ok(Arc::new(exec));
        }

        let exec = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&self.storage),
            Arc::clone(&full_schema),
            pruned.row_groups,
            projection.cloned(),
            physical_filters,
            limit,
            self.config.batch_size,
            self.config.io_coalesce_window,
            self.config.io_concurrency,
            self.config.target_partitions,
        )
        .map_err(|e: crate::QueryError| DataFusionError::External(Box::new(e)))?
        .with_agg_group_keys(self.declared_agg_group_keys());

        Ok(Arc::new(exec))
    }
}

/// Extract the `_rowid` value set from a fully-`_rowid`-equality filter (handles
/// `= lit`, `IN (lits)`, and the `OR`-chain rewrite), for the row-selection
/// pushdown. Returns `None` when no such filter is present.
/// Classify a logical filter as `Exact`, `Inexact`, or `Unsupported`.
///
/// Any predicate that can be translated into a IcefallDB predicate is classified as
/// `Exact`, because `IcefallDBScanExec` will evaluate it via Parquet `RowFilter`
/// and therefore only emit rows that satisfy it.  Sidecar statistics are still
/// used inside `scan()` to drop whole row groups when provable, but that is an
/// optimization independent of the pushdown guarantee we give to DataFusion.
/// Missing sidecars fall back to reading the Parquet data, still correctly,
/// so they remain `Exact`.
///
/// Simple equality, range, and IN-list predicates on partition columns are
/// always declared `Exact` so DataFusion pushes them down and `scan()` can use
/// them for file-level pruning, even if the literal values are not otherwise
/// translatable into a IcefallDB predicate.
fn classify_filter(
    filter: &Expr,
    arrow_schema: &ArrowSchema,
    partition_by: Option<&[String]>,
) -> TableProviderFilterPushDown {
    if is_partition_pushdown_predicate(filter, partition_by) {
        return TableProviderFilterPushDown::Exact;
    }
    // NOTE: `_rowid` (pseudo-column) filters are deliberately NOT pushed through
    // the TableProvider — DataFusion 54 fails to resolve `_rowid` against the
    // table source schema during filter pushdown. The `_rowid` row-selection
    // pushdown is applied instead by the `RowIdSelectionPushdown` physical rule,
    // which rewrites `FilterExec(_rowid ..) -> IcefallDBScanExec` after the plan
    // is fully resolved.
    if expr_to_predicate(filter, arrow_schema).is_some() {
        TableProviderFilterPushDown::Exact
    } else {
        TableProviderFilterPushDown::Unsupported
    }
}

/// Returns true when `expr` is a simple equality, range, or non-negated
/// IN-list predicate on a partition column.
fn is_partition_pushdown_predicate(expr: &Expr, partition_by: Option<&[String]>) -> bool {
    let Some(partition_by) = partition_by else {
        return false;
    };
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if !matches!(
                op,
                Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
            ) {
                return false;
            }
            (is_partition_column(left, partition_by) && is_literal(right))
                || (is_literal(left) && is_partition_column(right, partition_by))
        }
        Expr::InList(InList {
            expr,
            list,
            negated,
        }) if !negated => is_partition_column(expr, partition_by) && list.iter().all(is_literal),
        _ => false,
    }
}

/// Returns true when `expr` references a top-level column in `partition_by`.
fn is_partition_column(expr: &Expr, partition_by: &[String]) -> bool {
    match expr {
        Expr::Column(c) => partition_by.iter().any(|name| name == c.name()),
        _ => false,
    }
}

/// Returns true when `expr` is a literal value.
fn is_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(_, _))
}

/// Combine a non-empty list of physical filter expressions with logical AND.
fn combine_physical_filters(
    filters: Vec<Arc<dyn datafusion::physical_expr::PhysicalExpr>>,
    schema: &ArrowSchema,
) -> Result<Arc<dyn datafusion::physical_expr::PhysicalExpr>> {
    use datafusion::physical_expr::expressions::binary;

    if filters.is_empty() {
        return Err(QueryError::Other(
            "combine_physical_filters called with no filters".into(),
        ));
    }
    let mut combined = Arc::clone(&filters[0]);
    for f in filters.iter().skip(1) {
        combined = binary(Arc::clone(&combined), Operator::And, Arc::clone(f), schema)?;
    }
    Ok(combined)
}

/// Sum of live (non-deleted) rows across a manifest's fragments, computed from
/// the denormalized `row_counts` without reading any `.meta` sidecar or
/// checkpoint. Returns `None` when `row_counts` is absent (legacy manifest) or
/// does not line up with `row_groups`, in which case the caller leaves
/// `num_rows` unknown until the first scan materializes the plan.
fn manifest_live_row_count(manifest: &Manifest) -> Option<usize> {
    let counts = manifest.row_counts.as_ref()?;
    if counts.len() != manifest.row_groups.len() {
        return None;
    }
    Some(
        counts
            .iter()
            .zip(manifest.row_groups.iter())
            .map(|(rows, rg)| rows.saturating_sub(rg.deleted_count as usize))
            .sum(),
    )
}

/// Build a native Parquet scan with the row filter disabled, re-applying the
/// predicate as a `FilterExec` over the bulk-decoded output and projecting back
/// to the requested columns.
///
/// `scan_proj` is the ascending set of columns the scan reads
/// (projection ∪ filter columns); because the projection is a subset of the
/// filter columns it equals the filter columns. The selective-decode path is
/// skipped (faster when nothing can be late-materialized) while statistics and
/// page-index pruning are preserved by `build_native_parquet_exec`.
///
/// No `limit` is passed to the scan: the caller only takes this path when the
/// query has no limit. A pushed limit would cap pre-filter rows here (the row
/// filter is off), so it must stay on the pushdown path.
#[allow(clippy::too_many_arguments)]
fn build_native_bulk_decode_plan(
    storage: &Arc<dyn Storage>,
    pruned: &ScanPlan,
    arrow_schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
    scan_proj: &[usize],
    filters: &[Expr],
    batch_size: usize,
    target_partitions: usize,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let scan_proj_vec = scan_proj.to_vec();
    let mut plan = build_native_parquet_exec(
        storage,
        pruned,
        arrow_schema,
        Some(&scan_proj_vec),
        filters,
        None,
        batch_size,
        target_partitions,
        false,
    )
    .map_err(|e: QueryError| DataFusionError::External(Box::new(e)))?;

    // Re-apply the predicate over the bulk-decoded (filter-column) output.
    let scan_schema = arrow_schema.project(scan_proj)?;
    let df_schema = DFSchema::try_from(scan_schema.clone())?;
    let props = ExecutionProps::new();
    let mut physical = Vec::with_capacity(filters.len());
    for f in filters {
        physical.push(create_physical_expr(f, &df_schema, &props)?);
    }
    let predicate = combine_physical_filters(physical, &scan_schema)
        .map_err(|e: QueryError| DataFusionError::External(Box::new(e)))?;
    plan = Arc::new(FilterExec::try_new(predicate, plan)?);

    // Project back to the requested output columns (no-op when they already
    // equal `scan_proj`, e.g. a COUNT(*) whose projection is empty).
    if projection.map(Vec::as_slice) != Some(scan_proj) {
        let out: &[usize] = projection.map(Vec::as_slice).unwrap_or(&[]);
        let mut exprs = Vec::with_capacity(out.len());
        for &orig in out {
            let pos = scan_proj
                .iter()
                .position(|&x| x == orig)
                .expect("projection is a subset of scan_proj");
            let name = arrow_schema.field(orig).name();
            exprs.push(ProjectionExpr::new(Arc::new(Column::new(name, pos)), name));
        }
        plan = Arc::new(ProjectionExec::try_new(exprs, plan)?);
    }
    Ok(plan)
}

/// Read the AAD prefix for an encrypted table from its `_encryption.json`
/// marker. Returns an empty `Vec` if the marker is absent or unparseable, in
/// which case the caller (or the file itself, when `with_aad_prefix_storage(true)`
/// was used at write time) provides the AAD. A marker that parses cleanly but
/// advertises an unknown algorithm is rejected (propagated as an error) rather
/// than silently treated as decryptable.
#[cfg(feature = "encryption")]
async fn read_table_aad_prefix(storage: &Arc<dyn Storage>, table: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    use icefalldb_core::encryption::SchemaEncryptionMarker;

    let marker_path = format!("{table}/_encryption.json");
    let bytes = match storage.read(&marker_path).await {
        Ok(b) => b,
        Err(_) => return Ok(Vec::new()),
    };
    let marker: SchemaEncryptionMarker = match serde_json::from_slice(&bytes) {
        Ok(m) => m,
        Err(_) => return Ok(Vec::new()),
    };
    // A well-formed marker must advertise a scheme we know how to read.
    marker.validate()?;
    Ok(marker
        .aad_prefix
        .and_then(|b64| base64::engine::general_purpose::STANDARD.decode(b64).ok())
        .unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── try_locate_by_index inversion: offset_of_row_id ───────────────────────
    // Inverse of scan::row_id_at_offset; getting this wrong deletes the WRONG row.
    #[test]
    fn offset_of_row_id_inverts_segments() {
        // Two concatenated segments: Range[100..105) then Sorted[7, 9, 42].
        let segs = vec![
            RowIdSegment::Range {
                start: 100,
                count: 5,
            },
            RowIdSegment::Sorted {
                ids: vec![7, 9, 42],
            },
        ];
        // Range covers physical offsets 0..5.
        assert_eq!(offset_of_row_id(&segs, 100), Some(0));
        assert_eq!(offset_of_row_id(&segs, 104), Some(4));
        // Sorted segment is appended after the 5 range rows.
        assert_eq!(offset_of_row_id(&segs, 7), Some(5));
        assert_eq!(offset_of_row_id(&segs, 9), Some(6));
        assert_eq!(offset_of_row_id(&segs, 42), Some(7));
        // Absent ids resolve to None (caller falls back / skips).
        assert_eq!(offset_of_row_id(&segs, 105), None);
        assert_eq!(offset_of_row_id(&segs, 8), None);
        // Empty (legacy fragment) → None.
        assert_eq!(offset_of_row_id(&[], 100), None);
    }

    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use async_trait::async_trait;
    use datafusion::common::stats::Precision;
    use datafusion::logical_expr::{col, lit};
    use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
    use icefalldb_core::storage::Storage;
    use icefalldb_core::PlannedRowGroup;
    use serde_json::Value;

    fn test_arrow_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ])
    }

    fn make_scan_plan(row_groups: Vec<(usize, Vec<(String, ColumnStats)>)>) -> ScanPlan {
        let mut planned = Vec::new();
        for (rows, columns) in row_groups {
            let meta = RowGroupMeta {
                rows,
                columns: columns.into_iter().collect(),
                ..Default::default()
            };
            planned.push(PlannedRowGroup {
                meta,
                ..Default::default()
            });
        }
        ScanPlan {
            table: "t".into(),
            schema: icefalldb_core::metadata::Schema {
                schema_id: 1,
                columns: vec![],
                partition_by: None,
                sort: None,
                agg_group_keys: None,
                row_group_target_rows: 1000,
                row_group_target_bytes: 1024,
                dropped_columns: vec![],
                max_field_id: 0,
            },
            row_groups: planned,
        }
    }

    #[test]
    fn test_exact_translatable() {
        let schema = test_arrow_schema();
        let expr = col("a").gt(lit(20i32));
        assert_eq!(
            classify_filter(&expr, &schema, None),
            TableProviderFilterPushDown::Exact
        );
    }

    #[test]
    fn test_unsupported_untranslatable() {
        let schema = test_arrow_schema();
        // LIKE has no IcefallDB predicate translator.
        let expr = col("b").like(lit("%x"));
        assert_eq!(
            classify_filter(&expr, &schema, None),
            TableProviderFilterPushDown::Unsupported
        );
    }

    #[test]
    fn test_partition_predicate_exact() {
        let schema = test_arrow_schema();
        let partition_by = &["b".to_string()];
        let expr = col("b").eq(lit("cat_0"));
        assert_eq!(
            classify_filter(&expr, &schema, Some(partition_by)),
            TableProviderFilterPushDown::Exact
        );
    }

    #[test]
    fn test_partition_range_predicate_exact() {
        let schema = test_arrow_schema();
        let partition_by = &["a".to_string()];
        let expr = col("a").gt(lit(20i32));
        assert_eq!(
            classify_filter(&expr, &schema, Some(partition_by)),
            TableProviderFilterPushDown::Exact
        );
    }

    #[test]
    fn test_non_partition_predicate_still_unsupported() {
        let schema = test_arrow_schema();
        let partition_by = &["a".to_string()];
        // LIKE on a non-partition column remains unsupported.
        let expr = col("b").like(lit("%x"));
        assert_eq!(
            classify_filter(&expr, &schema, Some(partition_by)),
            TableProviderFilterPushDown::Unsupported
        );
    }

    #[test]
    fn test_count_star_statistics() {
        let arrow = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let scan = make_scan_plan(vec![
            (
                10,
                vec![(
                    "a".into(),
                    ColumnStats {
                        min: Some(Value::from(1)),
                        max: Some(Value::from(10)),
                        nulls: 2,
                    },
                )],
            ),
            (
                20,
                vec![(
                    "a".into(),
                    ColumnStats {
                        min: Some(Value::from(11)),
                        max: Some(Value::from(30)),
                        nulls: 3,
                    },
                )],
            ),
        ]);
        let stats = scan_plan_statistics(&scan, &arrow);
        assert_eq!(stats.num_rows, Precision::Exact(30));
        assert_eq!(stats.column_statistics[0].null_count, Precision::Exact(5));
        assert_eq!(
            stats.column_statistics[0].min_value,
            Precision::Exact(datafusion::common::ScalarValue::Int32(Some(1)))
        );
        assert_eq!(
            stats.column_statistics[0].max_value,
            Precision::Exact(datafusion::common::ScalarValue::Int32(Some(30)))
        );
    }

    #[tokio::test]
    async fn test_tiny_table_cache_populates_and_reuses() {
        use arrow::array::{Int32Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::Writer;

        fn make_batch(n: usize) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ]));
            let a: Vec<Option<i32>> = (0..n).map(|i| Some(i as i32)).collect();
            let b: Vec<Option<&str>> = (0..n).map(|_i| Some("x")).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(a)),
                    Arc::new(StringArray::from(b)),
                ],
            )
            .unwrap()
        }

        async fn create_table(root: &std::path::Path, rows: usize) -> Arc<dyn Storage> {
            let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
            let mut schema = arrow_schema_to_icefalldb(make_batch(rows).schema());
            schema.row_group_target_rows = rows.max(1);
            let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
                .await
                .unwrap();
            writer.insert_batch(make_batch(rows)).await.unwrap();
            writer.commit().await.unwrap();
            storage
        }

        let tmp = tempfile::tempdir().unwrap();
        let storage = create_table(tmp.path(), 100).await;

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 200,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        };
        let provider = IcefallDBTableProvider::new(storage, "t", config)
            .await
            .unwrap();

        let ctx = crate::icefalldb_session(1, 1024);
        let projection = Some(vec![0usize]);
        let filter = col("a").gt(lit(49i32));

        let plan1 = provider
            .scan(
                &ctx.state(),
                projection.as_ref(),
                std::slice::from_ref(&filter),
                None,
            )
            .await
            .unwrap();
        let batches1 = collect_plan(Arc::clone(&plan1), Arc::new(TaskContext::default()))
            .await
            .unwrap();

        // The cache should now hold the loaded MemTable.
        {
            let cache = provider.tiny_table_cache.read().await;
            assert!(
                cache.is_some(),
                "tiny table cache should be populated after first scan"
            );
        }

        let plan2 = provider
            .scan(
                &ctx.state(),
                projection.as_ref(),
                std::slice::from_ref(&filter),
                None,
            )
            .await
            .unwrap();
        let batches2 = collect_plan(Arc::clone(&plan2), Arc::new(TaskContext::default()))
            .await
            .unwrap();

        // Results must be identical across the cached and freshly-built paths.
        assert_eq!(batches1.len(), batches2.len());
        for (b1, b2) in batches1.iter().zip(batches2.iter()) {
            assert_eq!(b1.num_rows(), b2.num_rows());
        }

        // The second plan should be a MemTable scan, not the custom IcefallDB scan.
        assert!(
            plan2
                .downcast_ref::<crate::scan::IcefallDBScanExec>()
                .is_none(),
            "second scan should reuse the cached MemTable"
        );
    }

    #[tokio::test]
    async fn test_tiny_cache_results_match_normal_scan() {
        use arrow::array::{Int32Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::Writer;

        fn make_batch(n: usize) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ]));
            let a: Vec<Option<i32>> = (0..n).map(|i| Some(i as i32)).collect();
            let b: Vec<Option<&str>> = (0..n).map(|_| Some("x")).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(a)),
                    Arc::new(StringArray::from(b)),
                ],
            )
            .unwrap()
        }

        async fn create_table(root: &std::path::Path, rows: usize) -> Arc<dyn Storage> {
            let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
            let mut schema = arrow_schema_to_icefalldb(make_batch(rows).schema());
            schema.row_group_target_rows = rows.max(1);
            let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
                .await
                .unwrap();
            writer.insert_batch(make_batch(rows)).await.unwrap();
            writer.commit().await.unwrap();
            storage
        }

        async fn run_query(provider: Arc<IcefallDBTableProvider>, sql: &str) -> Vec<RecordBatch> {
            let ctx = crate::icefalldb_session(1, 1024);
            ctx.register_table("t", provider).unwrap();
            ctx.sql(sql).await.unwrap().collect().await.unwrap()
        }

        let tmp = tempfile::tempdir().unwrap();
        let storage = create_table(tmp.path(), 100).await;

        let tiny_config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 200,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        };
        let normal_config = ProviderConfig {
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
            ..tiny_config
        };

        let tiny_provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", tiny_config)
                .await
                .unwrap(),
        );
        let normal_provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", normal_config)
                .await
                .unwrap(),
        );

        let sql = "SELECT a, b FROM t WHERE a > 24 AND a < 76 ORDER BY a LIMIT 10";
        let tiny_batches = run_query(tiny_provider, sql).await;
        let normal_batches = run_query(normal_provider, sql).await;

        assert_eq!(tiny_batches.len(), normal_batches.len());
        for (tb, nb) in tiny_batches.iter().zip(normal_batches.iter()) {
            assert_eq!(tb.schema(), nb.schema());
            assert_eq!(tb.num_rows(), nb.num_rows());
            for col_idx in 0..tb.num_columns() {
                assert_eq!(
                    tb.column(col_idx).as_ref(),
                    nb.column(col_idx).as_ref(),
                    "column {col_idx} differs between cached and normal scan"
                );
            }
        }
    }

    #[tokio::test]
    async fn test_large_table_does_not_use_tiny_cache() {
        use arrow::array::{Int32Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::Writer;

        fn make_batch(n: usize) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ]));
            let a: Vec<Option<i32>> = (0..n).map(|i| Some(i as i32)).collect();
            let b: Vec<Option<&str>> = (0..n).map(|_i| Some("x")).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(a)),
                    Arc::new(StringArray::from(b)),
                ],
            )
            .unwrap()
        }

        async fn create_table(root: &std::path::Path, rows: usize) -> Arc<dyn Storage> {
            let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
            let mut schema = arrow_schema_to_icefalldb(make_batch(rows).schema());
            schema.row_group_target_rows = rows.max(1);
            let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
                .await
                .unwrap();
            writer.insert_batch(make_batch(rows)).await.unwrap();
            writer.commit().await.unwrap();
            storage
        }

        let tmp = tempfile::tempdir().unwrap();
        let storage = create_table(tmp.path(), 1000).await;

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 200,
            tiny_table_cache_threshold_bytes: 1,
            wal_mode: true,
        };
        let provider = IcefallDBTableProvider::new(storage, "t", config)
            .await
            .unwrap();

        let ctx = crate::icefalldb_session(1, 1024);
        let _ = provider
            .scan(&ctx.state(), Some(&vec![0]), &[], None)
            .await
            .unwrap();

        let cache = provider.tiny_table_cache.read().await;
        assert!(
            cache.is_none(),
            "large table should not be loaded into the tiny cache"
        );
    }

    #[tokio::test]
    async fn test_tiny_cache_invalidated_on_manifest_sequence_change() {
        use arrow::array::{Int32Array, RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::Writer;

        fn make_batch(n: usize, offset: i32) -> RecordBatch {
            let schema = Arc::new(ArrowSchema::new(vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ]));
            let a: Vec<Option<i32>> = (0..n).map(|i| Some((i as i32) + offset)).collect();
            let b: Vec<Option<&str>> = (0..n).map(|_| Some("x")).collect();
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(a)),
                    Arc::new(StringArray::from(b)),
                ],
            )
            .unwrap()
        }

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let mut schema = arrow_schema_to_icefalldb(make_batch(50, 0).schema());
        schema.row_group_target_rows = 50;

        let mut writer = Writer::create(Arc::clone(&storage), "t", schema.clone())
            .await
            .unwrap();
        writer.insert_batch(make_batch(50, 0)).await.unwrap();
        writer.commit().await.unwrap();

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 200,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        };
        let provider = IcefallDBTableProvider::new(Arc::clone(&storage), "t", config)
            .await
            .unwrap();

        let ctx = crate::icefalldb_session(1, 1024);
        let _ = provider
            .scan(&ctx.state(), Some(&vec![0]), &[], None)
            .await
            .unwrap();
        {
            let cache = provider.tiny_table_cache.read().await;
            assert!(
                cache.is_some(),
                "cache should be populated for first snapshot"
            );
        }

        // Commit a second snapshot and refresh the provider in-place. The new
        // snapshot sequence invalidates the tiny-table cache.
        let mut writer = Writer::new(Arc::clone(&storage), "t", schema.clone())
            .await
            .unwrap();
        writer.insert_batch(make_batch(50, 1000)).await.unwrap();
        let delta = writer.commit().await.unwrap();
        provider.apply_committed_delta(&delta).await.unwrap();

        let _ = provider
            .scan(&ctx.state(), Some(&vec![0]), &[], None)
            .await
            .unwrap();
        {
            let cache = provider.tiny_table_cache.read().await;
            assert!(
                cache.is_some(),
                "cache should be repopulated for the new snapshot"
            );
            let (_, mem_table) = cache.as_ref().unwrap();
            let mut total_rows = 0usize;
            for partition in &mem_table.batches {
                let batches = partition.read().await;
                total_rows += batches.iter().map(|b| b.num_rows()).sum::<usize>();
            }
            assert_eq!(
                total_rows, 100,
                "cached table should reflect the new snapshot"
            );
        }
    }

    /// Verify that the tiny-table cache does NOT serve deleted rows.
    ///
    /// The test builds a small table (comfortably under the tiny-table threshold),
    /// injects a deletion vector via a patched manifest, and then runs the same
    /// SELECT query TWICE through the provider. The first call materialises the
    /// MemTable; the second call hits the cache. Both must exclude the deleted rows.
    ///
    /// This guards against a future refactor of `build_tiny_mem_table` that
    /// bypasses `IcefallDBScanExec` and therefore stops applying deletion vectors.
    #[tokio::test]
    async fn tiny_table_cache_skips_deleted_rows() {
        use arrow::array::{Int32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::metadata::Manifest;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::DeletionVector;
        use icefalldb_core::Writer;

        // ── 1. Write a small table (10 rows: id = 0..9) ──────────────────────
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from((0i32..10).collect::<Vec<_>>()))],
        )
        .unwrap();

        let mut icefalldb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
        icefalldb_schema.row_group_target_rows = 10;

        let mut writer = Writer::create(Arc::clone(&storage), "t", icefalldb_schema.clone())
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        // ── 2. Read the current manifest and inject a deletion vector ─────────
        // The pointer `t/_manifest.json` contains {"latest": N}. Read it to find N.
        let pointer_bytes = storage.read("t/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();

        let manifest_path = format!("t/{}", Manifest::filename(seq));
        let manifest_bytes = storage.read(&manifest_path).await.unwrap();
        let mut manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();

        // Write the deletion vector: delete row offsets 2 and 7.
        let del_rel_path = "_deletions/rg0__v1.del";
        let del_storage_path = format!("t/{}", del_rel_path);
        let mut dv = DeletionVector::default();
        dv.union_offsets([2u32, 7u32]);
        let del_bytes = dv.serialize();
        storage.write(&del_storage_path, &del_bytes).await.unwrap();

        // Patch the first row-group entry with the deletion-vector reference.
        assert!(
            !manifest.row_groups.is_empty(),
            "expected at least one row group after commit"
        );
        manifest.row_groups[0].deletes = Some(del_rel_path.to_string());
        manifest.row_groups[0].deleted_count = 2;

        // Write a new manifest at seq+1 with the patched entry and a valid checksum.
        let next_seq = seq + 1;
        manifest.sequence = next_seq;
        manifest.checksum = String::new();
        manifest.checksum = manifest.compute_checksum().unwrap();

        let new_manifest_path = format!("t/{}", Manifest::filename(next_seq));
        let new_manifest_bytes = serde_json::to_vec(&manifest).unwrap();
        storage
            .write(&new_manifest_path, &new_manifest_bytes)
            .await
            .unwrap();

        // Update the manifest pointer to the new sequence.
        let new_pointer = serde_json::json!({"latest": next_seq});
        storage
            .write(
                "t/_manifest.json",
                &serde_json::to_vec(&new_pointer).unwrap(),
            )
            .await
            .unwrap();

        // ── 3. Create the provider (tiny-table threshold >> 10 rows) ─────────
        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX, // force custom scan path
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 200, // 10 rows << 200: takes tiny path
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        };
        let provider = IcefallDBTableProvider::new(Arc::clone(&storage), "t", config)
            .await
            .unwrap();

        // Use the session only for its `Session` state, not for table registration,
        // so we can call `provider.scan(...)` directly and retain access to its
        // internal `tiny_table_cache` field.
        let ctx = crate::icefalldb_session(1, 1024);
        let state = ctx.state();

        // ── 4. First query: must materialise the MemTable without deleted rows ─
        let plan1 = provider.scan(&state, None, &[], None).await.unwrap();
        let batches1 = collect_plan(Arc::clone(&plan1), Arc::new(TaskContext::default()))
            .await
            .unwrap();

        let ids1: Vec<i32> = batches1
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
                    .flatten()
            })
            .collect();

        assert_eq!(
            ids1.len(),
            8,
            "first scan (cold cache): expected 8 rows after deleting 2"
        );
        assert!(
            !ids1.contains(&2),
            "first scan: deleted id=2 must be absent"
        );
        assert!(
            !ids1.contains(&7),
            "first scan: deleted id=7 must be absent"
        );

        // The cache must now be populated.
        {
            let cache = provider.tiny_table_cache.read().await;
            assert!(
                cache.is_some(),
                "tiny-table cache must be populated after first scan"
            );
        }

        // ── 5. Second query: must serve the same (DV-correct) result from cache ─
        let plan2 = provider.scan(&state, None, &[], None).await.unwrap();
        let batches2 = collect_plan(Arc::clone(&plan2), Arc::new(TaskContext::default()))
            .await
            .unwrap();

        let ids2: Vec<i32> = batches2
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .iter()
                    .flatten()
            })
            .collect();

        assert_eq!(
            ids2.len(),
            8,
            "second scan (cached MemTable): expected 8 rows after deleting 2"
        );
        assert!(
            !ids2.contains(&2),
            "cached scan: deleted id=2 must remain absent"
        );
        assert!(
            !ids2.contains(&7),
            "cached scan: deleted id=7 must remain absent"
        );
    }

    // ── SQL-surface acceptance tests ──────────────────────────────────

    /// `SELECT _rowid FROM t` resolves `_rowid` through the full SQL/DataFrame
    /// path and returns only the live row-ids (deleted rows absent).
    ///
    /// This test exercises the complete pipeline: DataFusion planner resolves
    /// `_rowid` via `TableProvider::schema()`, projects index N into the scan,
    /// `IcefallDBScanExec` synthesizes the column, and the deletion vector
    /// removes the deleted physical offsets before id lookup.
    ///
    /// Row IDs are synthesized as 0 when a fragment has no allocated row-id
    /// segments (legacy fragment), so this test asserts on the live-row COUNT
    /// rather than specific id values.  The key correctness property is that
    /// deleted rows do not appear in the output.
    #[tokio::test]
    async fn select_rowid_via_sql_skips_deleted() {
        use arrow::array::{Int32Array, RecordBatch, UInt64Array};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::metadata::Manifest;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::{DeletionVector, Writer};

        // ── 1. Write a 6-row table (id = 0..5) ──────────────────────────────
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from((0i32..6).collect::<Vec<_>>()))],
        )
        .unwrap();

        let mut icefalldb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
        icefalldb_schema.row_group_target_rows = 6;

        let mut writer = Writer::create(Arc::clone(&storage), "t", icefalldb_schema.clone())
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        // ── 2. Inject a deletion vector (delete physical offsets 1 and 3) ───
        let pointer_bytes = storage.read("t/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();

        let manifest_path = format!("t/{}", Manifest::filename(seq));
        let manifest_bytes = storage.read(&manifest_path).await.unwrap();
        let mut manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();

        let del_rel_path = "_deletions/rg0__v1.del";
        let del_storage_path = format!("t/{}", del_rel_path);
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32, 3u32]);
        let del_bytes = dv.serialize();
        storage.write(&del_storage_path, &del_bytes).await.unwrap();

        manifest.row_groups[0].deletes = Some(del_rel_path.to_string());
        manifest.row_groups[0].deleted_count = dv.cardinality();

        // Write at next sequence.
        let next_seq = seq + 1;
        manifest.sequence = next_seq;
        manifest.checksum = String::new();
        manifest.checksum = manifest.compute_checksum().unwrap();

        let new_manifest_path = format!("t/{}", Manifest::filename(next_seq));
        storage
            .write(&new_manifest_path, &serde_json::to_vec(&manifest).unwrap())
            .await
            .unwrap();
        storage
            .write(
                "t/_manifest.json",
                &serde_json::to_vec(&serde_json::json!({"latest": next_seq})).unwrap(),
            )
            .await
            .unwrap();

        // ── 3. Register table, run `SELECT _rowid FROM t` ────────────────────
        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX, // force custom scan
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0, // bypass tiny-table cache
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        };
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", config)
                .await
                .unwrap(),
        );

        let ctx = crate::icefalldb_session(1, 1024);
        ctx.register_table("t", provider).unwrap();

        let batches = ctx
            .sql("SELECT _rowid FROM t")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // Verify the output is a UInt64 column.
        let result_schema = batches[0].schema();
        assert_eq!(result_schema.fields().len(), 1);
        assert_eq!(result_schema.field(0).name(), "_rowid");
        assert_eq!(
            result_schema.field(0).data_type(),
            &arrow::datatypes::DataType::UInt64
        );

        // 6 total rows minus 2 deleted = 4 live rows.
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 4,
            "SELECT _rowid must return 4 live rows (2 deleted out of 6)"
        );

        // All returned values should be UInt64 (value 0 is fine for legacy
        // fragments without allocated row-id segments).
        let all_values: Vec<u64> = batches
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .iter()
                    .flatten()
            })
            .collect();
        assert_eq!(all_values.len(), 4, "must have exactly 4 UInt64 values");
    }

    /// Verify the `SELECT *` behaviour with pseudo-columns.
    ///
    /// DataFusion 54 has no system-column / hidden-column mechanism, so
    /// `_rowid` and `_rowaddr` ARE included in `SELECT *`.  This test documents
    /// and asserts that actual behaviour.  If a future DataFusion version adds
    /// a way to exclude columns from wildcard expansion the implementation can
    /// be updated and this test changed to assert exclusion.
    #[tokio::test]
    async fn select_star_includes_pseudo_columns() {
        use arrow::array::{Int32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::Writer;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1i32, 2, 3]))],
        )
        .unwrap();

        let mut icefalldb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
        icefalldb_schema.row_group_target_rows = 3;

        let mut writer = Writer::create(Arc::clone(&storage), "t2", icefalldb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        };
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t2", config)
                .await
                .unwrap(),
        );

        let ctx = crate::icefalldb_session(1, 1024);
        ctx.register_table("t2", provider).unwrap();

        let batches = ctx
            .sql("SELECT * FROM t2 LIMIT 1")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();

        // DataFusion 54 has no hidden-column mechanism; `SELECT *` includes
        // `_rowid` and `_rowaddr`.  The output schema should have 3 columns:
        // `id`, `_rowid`, `_rowaddr`.
        let result_schema = batches[0].schema();
        let col_names: Vec<&str> = result_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(
            col_names.contains(&"id"),
            "SELECT * must include data column 'id'"
        );
        assert!(
            col_names.contains(&"_rowid"),
            "SELECT * includes _rowid (DataFusion 54 has no hidden-column mechanism)"
        );
        assert!(
            col_names.contains(&"_rowaddr"),
            "SELECT * includes _rowaddr (DataFusion 54 has no hidden-column mechanism)"
        );
        assert_eq!(
            col_names.len(),
            3,
            "SELECT * on a 1-data-column table must return exactly 3 columns (id + 2 pseudo)"
        );
    }

    /// Storage wrapper that counts reads of `.meta` sidecars and `_manifest.json`.
    struct CountingStorage {
        inner: Arc<dyn Storage>,
        meta_reads: std::sync::atomic::AtomicUsize,
        manifest_reads: std::sync::atomic::AtomicUsize,
    }

    impl CountingStorage {
        fn new(inner: Arc<dyn Storage>) -> Arc<Self> {
            Arc::new(Self {
                inner,
                meta_reads: std::sync::atomic::AtomicUsize::new(0),
                manifest_reads: std::sync::atomic::AtomicUsize::new(0),
            })
        }

        fn reset(&self) {
            self.meta_reads
                .store(0, std::sync::atomic::Ordering::SeqCst);
            self.manifest_reads
                .store(0, std::sync::atomic::Ordering::SeqCst);
        }

        fn meta_reads(&self) -> usize {
            self.meta_reads.load(std::sync::atomic::Ordering::SeqCst)
        }

        fn manifest_reads(&self) -> usize {
            self.manifest_reads
                .load(std::sync::atomic::Ordering::SeqCst)
        }

        fn count_read(&self, path: &str) {
            if path.ends_with(".meta") {
                self.meta_reads
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
            if path.ends_with("_manifest.json") {
                self.manifest_reads
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    #[async_trait]
    impl Storage for CountingStorage {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn local_root(&self) -> Option<&std::path::Path> {
            self.inner.local_root()
        }

        async fn read(&self, path: &str) -> icefalldb_core::Result<Vec<u8>> {
            self.count_read(path);
            self.inner.read(path).await
        }

        async fn size(&self, path: &str) -> icefalldb_core::Result<u64> {
            self.inner.size(path).await
        }

        async fn read_range(
            &self,
            path: &str,
            offset: u64,
            len: u64,
        ) -> icefalldb_core::Result<Vec<u8>> {
            self.count_read(path);
            self.inner.read_range(path, offset, len).await
        }

        async fn write(&self, path: &str, data: &[u8]) -> icefalldb_core::Result<()> {
            self.inner.write(path, data).await
        }

        async fn delete(&self, path: &str) -> icefalldb_core::Result<()> {
            self.inner.delete(path).await
        }

        async fn rename(&self, from: &str, to: &str) -> icefalldb_core::Result<()> {
            self.inner.rename(from, to).await
        }

        async fn list(&self, prefix: &str) -> icefalldb_core::Result<Vec<String>> {
            self.inner.list(prefix).await
        }

        async fn exists(&self, path: &str) -> icefalldb_core::Result<bool> {
            self.inner.exists(path).await
        }

        async fn lock_exclusive(
            &self,
            path: &str,
            timeout: std::time::Duration,
        ) -> icefalldb_core::Result<Box<dyn icefalldb_core::storage::LockGuard>> {
            self.inner.lock_exclusive(path, timeout).await
        }

        async fn sync(&self, path: &str) -> icefalldb_core::Result<()> {
            self.inner.sync(path).await
        }

        async fn sync_data(&self, path: &str) -> icefalldb_core::Result<()> {
            self.inner.sync_data(path).await
        }

        async fn sync_root(&self) -> icefalldb_core::Result<()> {
            self.inner.sync_root().await
        }

        async fn append(&self, path: &str, data: &[u8]) -> icefalldb_core::Result<()> {
            self.inner.append(path, data).await
        }
    }

    /// Incremental in-memory refresh (`apply_committed_delta`) produces the same
    /// query results as a full provider reload, and performs no `.meta` or
    /// `_manifest.json` reads while doing so.
    #[tokio::test]
    async fn apply_committed_delta_matches_full_reload_without_sidecar_reads() {
        use arrow::array::{AsArray, Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Int64Type, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::Writer;

        let tmp = tempfile::tempdir().unwrap();
        let local_storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let counting_storage = CountingStorage::new(Arc::clone(&local_storage));
        let storage: Arc<dyn Storage> = counting_storage.clone();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 10_000;

        // Two fragments: 15k rows + 10k rows = 25k rows total.
        let mut writer = Writer::create(storage.clone(), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch0 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from_iter(0i64..15_000)),
                Arc::new(Int64Array::from_iter((0i64..15_000).map(|i| i * 10))),
            ],
        )
        .unwrap();
        writer.insert_batch(batch0).await.unwrap();
        writer.commit().await.unwrap();

        let mut writer = Writer::new(storage.clone(), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch1 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from_iter(15_000i64..25_000)),
                Arc::new(Int64Array::from_iter((15_000i64..25_000).map(|i| i * 10))),
            ],
        )
        .unwrap();
        writer.insert_batch(batch1).await.unwrap();
        writer.commit().await.unwrap();

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        };
        let provider = Arc::new(
            IcefallDBTableProvider::new(storage.clone(), "t", config)
                .await
                .unwrap(),
        );

        let ctx = crate::icefalldb_session(1, 1024);
        ctx.register_table("t", provider.clone()).unwrap();

        // Warm the lazy-open scan-plan cache at the pre-delta sequence so the
        // apply below exercises the incremental (no-read) path, as on the real
        // mutation hot loop where a prior read has already materialized the plan.
        let _ = provider.scan_plan().await.unwrap();

        // Delete 1_000 rows from fragment 0 (offsets 5_000..5_999).
        let mut by_fragment: std::collections::HashMap<u64, Vec<u32>> =
            std::collections::HashMap::new();
        by_fragment.insert(0, (5_000u32..6_000).collect());

        let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "t")
            .await
            .unwrap();
        let schema = catalog.latest_schema().cloned().unwrap();
        let mut writer = Writer::new(storage.clone(), "t", schema).await.unwrap();
        let delta = writer.commit_deletes(by_fragment).await.unwrap();

        // Incremental refresh must not touch `.meta` or `_manifest.json`.
        counting_storage.reset();
        provider.apply_committed_delta(&delta).await.unwrap();
        assert_eq!(
            counting_storage.meta_reads(),
            0,
            "apply_committed_delta must not read .meta sidecars"
        );
        assert_eq!(
            counting_storage.manifest_reads(),
            0,
            "apply_committed_delta must not read _manifest.json"
        );

        // Compare against a freshly built provider on the same post-mutation table.
        // A subsequent query through the incrementally-refreshed provider must not
        // read `.meta` or `_manifest.json` either.
        counting_storage.reset();

        let fresh_provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&local_storage), "t", config)
                .await
                .unwrap(),
        );
        ctx.register_table("t_fresh", fresh_provider).unwrap();

        async fn scalar_i64(ctx: &datafusion::prelude::SessionContext, sql: &str) -> i64 {
            let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            let batch = batches.first().expect("no batches returned");
            batch.column(0).as_primitive::<Int64Type>().value(0)
        }

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t").await,
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_fresh").await,
            "COUNT(*) must match after incremental refresh"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT SUM(v) FROM t WHERE id >= 4000 AND id < 7000").await,
            scalar_i64(
                &ctx,
                "SELECT SUM(v) FROM t_fresh WHERE id >= 4000 AND id < 7000"
            )
            .await,
            "filtered SUM must match after incremental refresh"
        );

        assert_eq!(
            counting_storage.meta_reads(),
            0,
            "subsequent query must not read .meta sidecars"
        );
        assert_eq!(
            counting_storage.manifest_reads(),
            0,
            "subsequent query must not read _manifest.json"
        );
    }

    /// Incremental refresh after an UPDATE produces the same results as a full
    /// reload and performs no `.meta` or `_manifest.json` reads.
    #[tokio::test]
    async fn apply_committed_delta_update_matches_full_reload_without_sidecar_reads() {
        use arrow::array::{AsArray, Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Int64Type, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::{MatchLoc, Writer};

        let tmp = tempfile::tempdir().unwrap();
        let local_storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let counting_storage = CountingStorage::new(Arc::clone(&local_storage));
        let storage: Arc<dyn Storage> = counting_storage.clone();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 10_000;

        // Two fragments: 15k rows + 10k rows = 25k rows total.
        let mut writer = Writer::create(storage.clone(), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch0 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from_iter(0i64..15_000)),
                Arc::new(Int64Array::from_iter((0i64..15_000).map(|i| i * 10))),
            ],
        )
        .unwrap();
        writer.insert_batch(batch0).await.unwrap();
        writer.commit().await.unwrap();

        let mut writer = Writer::new(storage.clone(), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch1 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from_iter(15_000i64..25_000)),
                Arc::new(Int64Array::from_iter((15_000i64..25_000).map(|i| i * 10))),
            ],
        )
        .unwrap();
        writer.insert_batch(batch1).await.unwrap();
        writer.commit().await.unwrap();

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        };
        let provider = Arc::new(
            IcefallDBTableProvider::new(storage.clone(), "t", config)
                .await
                .unwrap(),
        );

        let ctx = crate::icefalldb_session(1, 1024);
        ctx.register_table("t", provider.clone()).unwrap();

        // Warm the lazy-open scan-plan cache at the pre-delta sequence (see the
        // delete test): the incremental apply must not re-read manifest/.meta.
        let _ = provider.scan_plan().await.unwrap();

        // Update rows 1_000..2_000 in fragment 0: set v = v + 1.
        let update_ids: Vec<i64> = (1_000i64..2_000).collect();
        let updated_batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(update_ids.clone())),
                Arc::new(Int64Array::from(
                    update_ids.iter().map(|id| id * 10 + 1).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let locs: Vec<MatchLoc> = update_ids
            .iter()
            .map(|id| MatchLoc {
                fragment_id: 0,
                offset: *id as u32,
                row_id: *id as u64,
            })
            .collect();

        let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "t")
            .await
            .unwrap();
        let schema = catalog.latest_schema().cloned().unwrap();
        let mut writer = Writer::new(storage.clone(), "t", schema).await.unwrap();
        let delta = writer
            .commit_update(updated_batch, locs, &["v".to_string()])
            .await
            .unwrap();

        // Incremental refresh must not touch `.meta` or `_manifest.json`.
        counting_storage.reset();
        provider.apply_committed_delta(&delta).await.unwrap();
        assert_eq!(
            counting_storage.meta_reads(),
            0,
            "apply_committed_delta must not read .meta sidecars"
        );
        assert_eq!(
            counting_storage.manifest_reads(),
            0,
            "apply_committed_delta must not read _manifest.json"
        );

        // Compare against a freshly built provider on the same post-mutation table.
        counting_storage.reset();

        let fresh_provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&local_storage), "t", config)
                .await
                .unwrap(),
        );
        ctx.register_table("t_fresh", fresh_provider).unwrap();

        async fn scalar_i64(ctx: &datafusion::prelude::SessionContext, sql: &str) -> i64 {
            let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            let batch = batches.first().expect("no batches returned");
            batch.column(0).as_primitive::<Int64Type>().value(0)
        }

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t").await,
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_fresh").await,
            "COUNT(*) must match after incremental UPDATE refresh"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT SUM(v) FROM t").await,
            scalar_i64(&ctx, "SELECT SUM(v) FROM t_fresh").await,
            "SUM(v) must match after incremental UPDATE refresh"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT SUM(v) FROM t WHERE id >= 1500 AND id < 2500").await,
            scalar_i64(
                &ctx,
                "SELECT SUM(v) FROM t_fresh WHERE id >= 1500 AND id < 2500"
            )
            .await,
            "filtered SUM must match after incremental UPDATE refresh"
        );

        assert_eq!(
            counting_storage.meta_reads(),
            0,
            "subsequent query must not read .meta sidecars"
        );
        assert_eq!(
            counting_storage.manifest_reads(),
            0,
            "subsequent query must not read _manifest.json"
        );
    }

    /// Incremental refresh after a MERGE (matched update + not-matched insert)
    /// produces the same results as a full reload and performs no `.meta` or
    /// `_manifest.json` reads.
    #[tokio::test]
    async fn apply_committed_delta_merge_matches_full_reload_without_sidecar_reads() {
        use arrow::array::{AsArray, Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Int64Type, Schema as ArrowSchema};
        use icefalldb_core::arrow_schema_to_icefalldb;
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::{MatchLoc, Writer};
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let local_storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let counting_storage = CountingStorage::new(Arc::clone(&local_storage));
        let storage: Arc<dyn Storage> = counting_storage.clone();

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 10;

        // Register a UNIQUE btree index on `id` before writing rows so MERGE has
        // a unique-key constraint to maintain.
        let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_index_definition_with_options(&guard, "id_uniq", "t_merge", "id", "btree", true)
            .await
            .unwrap();
        drop(guard);

        // 10 rows in a single fragment, ids 0..9, v = id * 10.
        let mut writer = Writer::create(storage.clone(), "t_merge", mdb_schema.clone())
            .await
            .unwrap();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from_iter(0i64..10)),
                Arc::new(Int64Array::from_iter((0i64..10).map(|i| i * 10))),
            ],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        };
        let provider = Arc::new(
            IcefallDBTableProvider::new(storage.clone(), "t_merge", config)
                .await
                .unwrap(),
        );

        let ctx = crate::icefalldb_session(1, 1024);
        ctx.register_table("t_merge", provider.clone()).unwrap();

        // Warm the lazy-open scan-plan cache at the pre-delta sequence (see the
        // delete test): the incremental apply must not re-read manifest/.meta.
        let _ = provider.scan_plan().await.unwrap();

        // MERGE source: update existing id=7 to v=777, insert new id=99 with v=9900.
        let matched_batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![7i64])),
                Arc::new(Int64Array::from(vec![777i64])),
            ],
        )
        .unwrap();
        let matched_locs = vec![MatchLoc {
            fragment_id: 0,
            offset: 7,
            row_id: 7,
        }];
        let insert_batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![99i64])),
                Arc::new(Int64Array::from(vec![9900i64])),
            ],
        )
        .unwrap();

        let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "t_merge")
            .await
            .unwrap();
        let schema = catalog.latest_schema().cloned().unwrap();
        let mut writer = Writer::new(storage.clone(), "t_merge", schema)
            .await
            .unwrap();
        let delta = writer
            .commit_merge(
                matched_batch,
                matched_locs,
                &["id".to_string(), "v".to_string()],
                insert_batch,
            )
            .await
            .unwrap();

        // Incremental refresh must not touch `.meta` or `_manifest.json`.
        counting_storage.reset();
        provider.apply_committed_delta(&delta).await.unwrap();
        assert_eq!(
            counting_storage.meta_reads(),
            0,
            "apply_committed_delta must not read .meta sidecars"
        );
        assert_eq!(
            counting_storage.manifest_reads(),
            0,
            "apply_committed_delta must not read _manifest.json"
        );

        // Compare against a freshly built provider on the same post-mutation table.
        counting_storage.reset();

        let fresh_provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&local_storage), "t_merge", config)
                .await
                .unwrap(),
        );
        ctx.register_table("t_merge_fresh", fresh_provider).unwrap();

        async fn scalar_i64(ctx: &datafusion::prelude::SessionContext, sql: &str) -> i64 {
            let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            let batch = batches.first().expect("no batches returned");
            batch.column(0).as_primitive::<Int64Type>().value(0)
        }

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge").await,
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_fresh").await,
            "COUNT(*) must match after incremental MERGE refresh"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge WHERE id = 7").await,
            scalar_i64(&ctx, "SELECT v FROM t_merge_fresh WHERE id = 7").await,
            "matched-update lookup must match after incremental MERGE refresh"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge WHERE id = 99").await,
            scalar_i64(&ctx, "SELECT v FROM t_merge_fresh WHERE id = 99").await,
            "not-matched-insert lookup must match after incremental MERGE refresh"
        );

        assert_eq!(
            counting_storage.meta_reads(),
            0,
            "subsequent query must not read .meta sidecars"
        );
        assert_eq!(
            counting_storage.manifest_reads(),
            0,
            "subsequent query must not read _manifest.json"
        );
    }

    // ── mmap'd binary index on open ─────────────────────────────────

    /// Build a single-fragment table with an `id`/`v` schema and a unique index
    /// on `id` via the legacy (unversioned) save path — mirroring
    /// `icefalldb create-index`, which now also writes the `.idx` binary sibling.
    /// Returns the storage and the row count.
    #[cfg(test)]
    async fn build_legacy_indexed_table(root: &std::path::Path) -> (Arc<dyn Storage>, i64) {
        use arrow::array::{Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::catalog::Catalog;
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::{arrow_schema_to_icefalldb, Writer};

        let rows = 2000i64;
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
        let schema_arrow = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut schema = arrow_schema_to_icefalldb(Arc::clone(&schema_arrow));
        schema.row_group_target_rows = (rows as usize).max(1);
        let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
            .await
            .unwrap();
        // A non-affine id set (an outlier breaks the linear fit) so this table
        // exercises the BINARY index path; the affine/contiguous case is covered
        // by `open_uses_learned_model_for_affine_key` below.
        let mut ids: Vec<i64> = (0..rows).collect();
        *ids.last_mut().unwrap() = 100_000;
        let vs: Vec<i64> = ids.iter().map(|i| i * 10).collect();
        let batch = RecordBatch::try_new(
            schema_arrow,
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(vs)),
            ],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let manifest = Catalog::load(storage.as_ref(), "t")
            .await
            .unwrap()
            .latest_manifest()
            .cloned()
            .unwrap();
        let def = IndexDefinition {
            name: "t_id_idx".into(),
            table: "t".into(),
            column: "id".into(),
            unique: true,
        };
        build_btree_index(storage.as_ref(), &def, &manifest)
            .await
            .unwrap()
            .save(storage.as_ref())
            .await
            .unwrap();
        (storage, rows)
    }

    #[cfg(test)]
    fn test_provider_config() -> ProviderConfig {
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        }
    }

    #[cfg(test)]
    async fn locate_id(provider: &IcefallDBTableProvider, id: i64) -> Option<Vec<MatchLoc>> {
        use datafusion::prelude::{col, lit};
        provider
            .try_locate_by_index(&col("id").eq(lit(id)))
            .await
            .unwrap()
    }

    /// Regression: a lazily-opened provider's point-locate path must re-pin its
    /// index when the manifest advances externally, so a `DELETE`/`UPDATE`/`MERGE`
    /// locates rows committed after open instead of silently under-matching them.
    #[tokio::test]
    async fn locate_repins_index_on_external_manifest_advance() {
        use arrow::array::{Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::catalog::Catalog;
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::Writer;

        let tmp = tempfile::tempdir().unwrap();
        let (storage, _rows) = build_legacy_indexed_table(tmp.path()).await;

        // Open the provider LAZILY at S, before the external advance.
        let provider =
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", test_provider_config())
                .await
                .unwrap();

        // EXTERNAL writer appends id=500_000 (absent from the open-time index) and
        // advances the manifest to S+1, then refreshes the flat index file.
        let schema = Catalog::load(storage.as_ref(), "t")
            .await
            .unwrap()
            .latest_schema()
            .cloned()
            .unwrap();
        let schema_arrow = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut writer = Writer::new(Arc::clone(&storage), "t", schema)
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    schema_arrow,
                    vec![
                        Arc::new(Int64Array::from(vec![500_000i64])),
                        Arc::new(Int64Array::from(vec![5_000_000i64])),
                    ],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();
        let manifest = Catalog::load(storage.as_ref(), "t")
            .await
            .unwrap()
            .latest_manifest()
            .cloned()
            .unwrap();
        let def = IndexDefinition {
            name: "t_id_idx".into(),
            table: "t".into(),
            column: "id".into(),
            unique: true,
        };
        build_btree_index(storage.as_ref(), &def, &manifest)
            .await
            .unwrap()
            .save(storage.as_ref())
            .await
            .unwrap();

        // The lazy provider must re-pin (index@S+1) and locate the new id.
        // Without the re-pin its open-time index@S has no id=500_000 → None.
        let locs = locate_id(&provider, 500_000)
            .await
            .expect("re-pin must surface the externally-added id");
        assert_eq!(locs.len(), 1, "exactly one live location for id=500_000");
    }

    /// Open uses the mmap binary index, and the located row is correct.
    #[tokio::test]
    async fn open_uses_binary_index_and_locates() {
        let tmp = tempfile::tempdir().unwrap();
        let (storage, _rows) = build_legacy_indexed_table(tmp.path()).await;
        let provider = IcefallDBTableProvider::new(storage, "t", test_provider_config())
            .await
            .unwrap();
        assert_eq!(
            provider.test_indexes_are_binary(),
            vec![true],
            "open must load the index via the mmap binary path"
        );
        let locs = locate_id(&provider, 1234).await.expect("id=1234 indexed");
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].row_id, 1234, "row_id for id=1234 must be 1234");
    }

    /// Proof that the binary path does NOT parse the JSON index: corrupt the
    /// JSON base bytes (keeping its filename so the name is still discovered),
    /// keep the valid `.idx`, and assert open + locate still succeed.
    #[tokio::test]
    async fn open_does_not_parse_json_when_binary_present() {
        let tmp = tempfile::tempdir().unwrap();
        let (storage, _rows) = build_legacy_indexed_table(tmp.path()).await;

        // Replace the JSON index contents with garbage; the `.idx` is untouched.
        storage
            .write(
                "t/_indexes/t_id_idx.json",
                b"{ this is not valid index json",
            )
            .await
            .unwrap();

        let provider = IcefallDBTableProvider::new(storage, "t", test_provider_config())
            .await
            .unwrap();
        assert_eq!(provider.test_indexes_are_binary(), vec![true]);
        let locs = locate_id(&provider, 42).await.expect("id=42 indexed");
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].row_id, 42);
    }

    /// Without a binary sibling, open falls back to parsing the JSON index and
    /// still locates correctly.
    #[tokio::test]
    async fn open_falls_back_to_json_without_binary() {
        let tmp = tempfile::tempdir().unwrap();
        let (storage, _rows) = build_legacy_indexed_table(tmp.path()).await;

        // Remove the derived binary cache; the canonical JSON remains.
        storage.delete("t/_indexes/t_id_idx.idx").await.unwrap();

        let provider = IcefallDBTableProvider::new(storage, "t", test_provider_config())
            .await
            .unwrap();
        assert_eq!(
            provider.test_indexes_are_binary(),
            vec![false],
            "missing .idx must fall back to the JSON path"
        );
        let locs = locate_id(&provider, 777).await.expect("id=777 indexed");
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].row_id, 777);
    }

    /// A contiguous (affine) `id` index loads as a learned model — a
    /// point locate works with NO postings present (delete both the JSON and the
    /// binary index, keeping only the tiny `.model`).
    #[tokio::test]
    async fn open_uses_learned_model_for_affine_key() {
        use arrow::array::{Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::{arrow_schema_to_icefalldb, Writer};

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let mut schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        schema.row_group_target_rows = 100;
        let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
            .await
            .unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    arrow_schema,
                    vec![Arc::new(Int64Array::from((0..100i64).collect::<Vec<_>>()))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        let catalog = DatabaseCatalog::new(Arc::clone(&storage));
        let lock: CatalogLockGuard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_index_definition(&lock, "t_id_idx", "t", "id", "btree")
            .await
            .unwrap();
        drop(lock);
        let ptr: serde_json::Value =
            serde_json::from_slice(&storage.read("t/_manifest.json").await.unwrap()).unwrap();
        let seq = ptr["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("t/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        let def = IndexDefinition {
            name: "t_id_idx".into(),
            table: "t".into(),
            column: "id".into(),
            unique: true,
        };
        build_btree_index(storage.as_ref(), &def, &manifest)
            .await
            .unwrap()
            .save(storage.as_ref())
            .await
            .unwrap();

        let provider =
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", test_provider_config())
                .await
                .unwrap();
        assert_eq!(
            provider.test_indexes_are_learned(),
            vec![true],
            "an affine id index must load as a learned model"
        );
        let locs = locate_id(&provider, 42).await.expect("id=42");
        assert_eq!(locs.len(), 1);
        assert_eq!(locs[0].row_id, 42);

        // No postings needed: corrupt the JSON index (kept only so the name is
        // still discovered) and delete the binary index; the tiny `.model` alone
        // answers the locate without ever parsing the garbage postings.
        storage
            .write("t/_indexes/t_id_idx.json", b"{ not valid index json")
            .await
            .unwrap();
        let _ = storage.delete("t/_indexes/t_id_idx.idx").await;
        let provider2 =
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", test_provider_config())
                .await
                .unwrap();
        assert_eq!(provider2.test_indexes_are_learned(), vec![true]);
        let locs = locate_id(&provider2, 7).await.expect("id=7 via model only");
        assert_eq!(locs[0].row_id, 7);
    }

    // ── new_at_snapshot must not load stale legacy indexes ─

    /// `new_at_snapshot` must restrict its index set to the indexes recorded in
    /// the historical manifest's `index_generations` map and must NOT surface
    /// additional legacy unversioned `_indexes/<name>.json` files discovered by
    /// `list_index_names`.
    ///
    /// Setup:
    ///   1. Create a table with a versioned "id_idx" index built from the commit
    ///      path (i.e. it appears in `manifest.index_generations`).
    ///   2. Plant a stale legacy flat index `_indexes/stale_idx.json` on disk —
    ///      a file that `list_index_names` would discover but that is NOT in
    ///      `index_generations` for the pinned snapshot.
    ///   3. Open a snapshot-pinned provider via `new_at_snapshot`.
    ///   4. Assert the provider has exactly 1 loaded index (the versioned id_idx),
    ///      not 2 (versioned id_idx + stale stale_idx).
    ///
    /// Without the fix `load_snapshot_indexes` calls `list_index_names` even on
    /// the time-travel path and would load "stale_idx", yielding count == 2.
    #[tokio::test]
    async fn at_snapshot_skips_legacy_flat_indexes() {
        use arrow::array::{Int64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use icefalldb_core::storage::local::LocalStorage;
        use icefalldb_core::storage::Storage;
        use icefalldb_core::{arrow_schema_to_icefalldb, Writer};

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let mut schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        schema.row_group_target_rows = 1000;

        // Use the DatabaseCatalog to register a versioned index definition so
        // the commit path writes it to index_generations.
        let dbcat = icefalldb_core::database_catalog::DatabaseCatalog::new(storage.clone());
        let guard = dbcat
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        dbcat.create_table(&guard, "t2", &schema).await.unwrap();
        dbcat
            .create_index_definition(&guard, "id_idx", "t2", "id", "btree")
            .await
            .unwrap();
        drop(guard);

        let mut writer = Writer::new(storage.clone(), "t2", schema).await.unwrap();
        writer
            .insert_batch(
                RecordBatch::try_new(
                    Arc::clone(&arrow_schema),
                    vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
                )
                .unwrap(),
            )
            .await
            .unwrap();
        writer.commit().await.unwrap();

        // Read the pinned sequence from the manifest pointer.
        let pointer_bytes = storage.read("t2/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let s1: u64 = pointer["latest"].as_u64().unwrap();

        // Plant a stale legacy flat index under _indexes/.  `list_index_names`
        // lists top-level .json files in the _indexes directory, so this file
        // would be discovered and loaded via the legacy path unless the fix
        // (versioned_only=true) prevents it.
        let stale_json = serde_json::json!({
            "definition": {
                "name": "stale_idx",
                "table": "t2",
                "column": "id",
                "unique": false
            },
            "snapshot_sequence": 0,
            "entries": { "1": [9999u64], "2": [9998u64] }
        });
        storage
            .write(
                "t2/_indexes/stale_idx.json",
                &serde_json::to_vec(&stale_json).unwrap(),
            )
            .await
            .unwrap();

        // Build a snapshot-pinned provider at S1.
        let provider =
            IcefallDBTableProvider::new_at_snapshot(storage, "t2", test_provider_config(), s1)
                .await
                .unwrap();

        // The S1 manifest has exactly 1 entry in index_generations (id_idx).
        // The fix ensures new_at_snapshot loads only that — not the additional
        // stale_idx found by list_index_names.
        assert_eq!(
            provider.test_loaded_index_count(),
            1,
            "new_at_snapshot must load only indexes from manifest.index_generations; \
             stale legacy flat indexes must be excluded (fix: versioned_only=true)"
        );
    }
}
