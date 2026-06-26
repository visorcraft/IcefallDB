//! Custom DataFusion `ExecutionPlan` for reading IcefallDB tables.
//!
//! Each partition reads one [`PlannedRowGroup`]. When sidecar `column_offsets`
//! cover all projected columns, the reader issues sparse `Storage::read_range`
//! calls for the column chunks plus the Parquet footer, reconstructs a
//! contiguous in-memory file buffer, and decodes it with the standard Parquet
//! arrow reader. Otherwise it falls back to reading the whole Parquet file.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{BooleanArray, RecordBatch, UInt64Array};
use arrow::datatypes::{Field, Schema, SchemaRef};
use bytes::Bytes;
use datafusion::common::stats::Precision;
use datafusion::common::utils::project_schema;
use datafusion::error::DataFusionError;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{
    EquivalenceProperties, Partitioning, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    Statistics,
};
use futures::StreamExt;
use futures::TryStreamExt;
use icefalldb_core::storage::Storage;
use icefalldb_core::PlannedRowGroup;
use parquet::arrow::arrow_reader::{
    ArrowPredicateFn, ArrowReaderOptions, ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder, RowFilter, RowSelection, RowSelector,
};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaDataReader};
use std::any::Any;

use crate::coalesce::coalesce_ranges;
use crate::error::QueryError;
use crate::metadata_cache::ParquetMetadataCache;
use crate::Result;

/// Name of the synthetic row-id pseudo-column.
pub const PSEUDO_COL_ROWID: &str = "_rowid";

/// Name of the synthetic row-address pseudo-column.
pub const PSEUDO_COL_ROWADDR: &str = "_rowaddr";

/// Returns `true` if `name` is a pseudo-column that must NOT be looked up in
/// the Parquet schema.
#[inline]
fn is_pseudo_column(name: &str) -> bool {
    name == PSEUDO_COL_ROWID || name == PSEUDO_COL_ROWADDR
}

/// Number of bytes in the Parquet footer tail (metadata length + magic).
const PARQUET_FOOTER_SIZE: u64 = 8;

/// Number of bytes in the Parquet magic header at the start of the file.
const PARQUET_MAGIC_SIZE: u64 = 4;

/// A single scan partition: one slice of one row group.
#[derive(Clone)]
pub(crate) struct ScanPartition {
    row_group: PlannedRowGroup,
    /// Row selection within the row group. `None` means read all rows.
    row_selection: Option<RowSelection>,
}

impl ScanPartition {
    fn row_count(&self) -> usize {
        match &self.row_selection {
            Some(sel) => sel.iter().filter(|s| !s.skip).map(|s| s.row_count).sum(),
            // When no explicit row selection is set, return the live row count:
            // physical rows minus those logically deleted via a deletion vector.
            // This keeps `partition_statistics()` accurate after DELETE commits so
            // that DataFusion's `AggregateStatistics` rule folds COUNT(*) to the
            // correct live count rather than the stale physical count.
            None => self
                .row_group
                .meta
                .rows
                .saturating_sub(self.row_group.deleted_count as usize),
        }
    }
}

/// Custom scan execution plan for IcefallDB tables.
#[derive(Clone)]
pub struct IcefallDBScanExec {
    storage: Arc<dyn Storage>,
    schema: SchemaRef,
    table_schema: SchemaRef,
    partitions: Vec<ScanPartition>,
    projection: Option<Vec<usize>>,
    filters: Vec<Arc<dyn PhysicalExpr>>,
    limit: Option<usize>,
    properties: Arc<PlanProperties>,
    batch_size: usize,
    io_coalesce_window: u64,
    io_concurrency: usize,
    /// Declared low-cardinality GROUP BY keys for warm partial aggregation
    /// (`Schema.agg_group_keys`).  Empty when the table declares no
    /// grouping keys.  Threaded from the provider's schema so the
    /// `MetadataAggregate` rule can recognise a GROUP BY on a declared key and
    /// compose the result from cached per-group partials.
    agg_group_keys: Arc<Vec<String>>,
    /// Optional Parquet Modular Encryption decryption properties. `None` for
    /// plaintext tables. When `Some`, every `ParquetRecordBatchReaderBuilder`
    /// built by this scan gets these properties via
    /// `ArrowReaderOptions::with_file_decryption_properties`.
    #[cfg(feature = "encryption")]
    decryption_properties: Option<Arc<parquet::encryption::decrypt::FileDecryptionProperties>>,
}

impl fmt::Debug for IcefallDBScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IcefallDBScanExec")
            .field(
                "table",
                &self.partitions.first().map(|p| &p.row_group.data_path),
            )
            .field("partitions", &self.partitions.len())
            .field("projection", &self.projection)
            .field("filters", &self.filters.len())
            .field("limit", &self.limit)
            .field("batch_size", &self.batch_size)
            .finish_non_exhaustive()
    }
}

impl IcefallDBScanExec {
    /// Create a new `IcefallDBScanExec`.
    ///
    /// `arrow_schema` is the full table schema; the output schema is the
    /// projection of `arrow_schema` by `projection`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        storage: Arc<dyn Storage>,
        arrow_schema: SchemaRef,
        planned_row_groups: Vec<PlannedRowGroup>,
        projection: Option<Vec<usize>>,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        limit: Option<usize>,
        batch_size: usize,
        io_coalesce_window: u64,
        io_concurrency: usize,
    ) -> Result<Self> {
        Self::new_with_target_partitions(
            storage,
            arrow_schema,
            planned_row_groups,
            projection,
            filters,
            limit,
            batch_size,
            io_coalesce_window,
            io_concurrency,
            1,
        )
    }

    /// Create a scan with intra-row-group parallelism.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_target_partitions(
        storage: Arc<dyn Storage>,
        arrow_schema: SchemaRef,
        planned_row_groups: Vec<PlannedRowGroup>,
        projection: Option<Vec<usize>>,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        limit: Option<usize>,
        batch_size: usize,
        io_coalesce_window: u64,
        io_concurrency: usize,
        target_partitions: usize,
    ) -> Result<Self> {
        Self::new_impl(
            storage,
            arrow_schema,
            planned_row_groups,
            projection,
            filters,
            limit,
            batch_size,
            io_coalesce_window,
            io_concurrency,
            target_partitions,
            #[cfg(feature = "encryption")]
            None,
        )
    }

    /// Create an encrypted scan. Requires the `encryption` feature. The
    /// decryption properties are propagated to every `ParquetRecordBatchReader`
    /// built by this scan.
    #[cfg(feature = "encryption")]
    #[allow(clippy::too_many_arguments)]
    pub fn new_encrypted(
        storage: Arc<dyn Storage>,
        arrow_schema: SchemaRef,
        planned_row_groups: Vec<PlannedRowGroup>,
        projection: Option<Vec<usize>>,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        limit: Option<usize>,
        batch_size: usize,
        io_coalesce_window: u64,
        io_concurrency: usize,
        target_partitions: usize,
        decryption_properties: Arc<parquet::encryption::decrypt::FileDecryptionProperties>,
    ) -> Result<Self> {
        Self::new_impl(
            storage,
            arrow_schema,
            planned_row_groups,
            projection,
            filters,
            limit,
            batch_size,
            io_coalesce_window,
            io_concurrency,
            target_partitions,
            Some(decryption_properties),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_impl(
        storage: Arc<dyn Storage>,
        arrow_schema: SchemaRef,
        planned_row_groups: Vec<PlannedRowGroup>,
        projection: Option<Vec<usize>>,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        limit: Option<usize>,
        batch_size: usize,
        io_coalesce_window: u64,
        io_concurrency: usize,
        target_partitions: usize,
        #[cfg(feature = "encryption")] decryption_properties: Option<
            Arc<parquet::encryption::decrypt::FileDecryptionProperties>,
        >,
    ) -> Result<Self> {
        let table_schema = Arc::clone(&arrow_schema);
        let output_schema = project_schema(&arrow_schema, projection.as_ref())?;
        // Intra-row-group splitting only pays off when each partition can skip the
        // other partitions' pages via the Parquet page index. `build_parquet_reader`
        // loads that index only for whole-file reads *with* filters
        // (`want_page_index = !is_sparse && !filters.is_empty()`). IcefallDB's sparse
        // column-chunk reads — taken whenever a row group has a `column_offsets`
        // sidecar — leave hole-filled buffers that disable the page index, so a
        // split there makes every partition re-decode the WHOLE group (an N-way
        // re-decode that pinned all cores for ~111s on a 1M single-row-group
        // filtered scan). Only split when the page index will actually be usable.
        let page_index_usable = !filters.is_empty()
            && planned_row_groups
                .iter()
                .all(|rg| rg.meta.column_offsets.is_none());
        let partitions =
            split_row_groups(&planned_row_groups, target_partitions, page_index_usable);
        let properties = compute_properties(&output_schema, &partitions);
        Ok(Self {
            storage,
            schema: output_schema,
            table_schema,
            partitions,
            projection,
            filters,
            limit,
            properties: Arc::new(properties),
            batch_size,
            io_coalesce_window,
            io_concurrency,
            agg_group_keys: Arc::new(Vec::new()),
            #[cfg(feature = "encryption")]
            decryption_properties,
        })
    }

    /// Return the output schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Return the declared low-cardinality GROUP BY keys for warm partial
    /// aggregation (`Schema.agg_group_keys`).  Empty when the table declares no
    /// grouping keys.
    pub fn agg_group_keys(&self) -> &[String] {
        &self.agg_group_keys
    }

    /// Return a clone of this scan with the declared GROUP BY keys attached.
    /// Used by the provider to thread `Schema.agg_group_keys` into the scan so
    /// the `MetadataAggregate` rule can compose warm GROUP BY results.
    pub fn with_agg_group_keys(mut self, keys: Vec<String>) -> Self {
        self.agg_group_keys = Arc::new(keys);
        self
    }

    /// Return the planned row groups (unsplit, with split clones collapsed).
    ///
    /// A single physical row group may be split across several scan partitions
    /// for intra-row-group parallelism (see [`split_row_groups`]); every such
    /// partition holds a clone of the SAME [`PlannedRowGroup`] and carries a
    /// `row_selection` (`ScanPartition::row_selection.is_some()`).  This method
    /// collapses those contiguous split clones back to one entry per physical
    /// row group so that metadata-only consumers (e.g. the `MetadataAggregate`
    /// COUNT(*)/SUM fast path, which sums `meta.rows - deleted_count` per row
    /// group) count each row group exactly once.  Without this an N-way split of
    /// a deletion-bearing row group would over-count its live rows N-fold.
    ///
    /// Only consecutive SPLIT partitions (`row_selection.is_some()`) that share
    /// the same physical row group are collapsed; unsplit partitions
    /// (`row_selection.is_none()`) are always emitted as-is, so genuinely
    /// distinct row groups — which may share a `data_path` (multiple row groups
    /// in one Parquet file) — are never merged.
    pub fn planned_row_groups(&self) -> Vec<PlannedRowGroup> {
        let mut out: Vec<PlannedRowGroup> = Vec::with_capacity(self.partitions.len());
        for part in &self.partitions {
            // A split partition repeats its row group; collapse it into the
            // immediately-preceding identical split entry.  Identity is the
            // physical row-group coordinates plus its deletion state, which are
            // identical across all splits of one row group and differ between
            // distinct row groups.
            if part.row_selection.is_some() {
                if let Some(prev) = out.last() {
                    if prev.data_path == part.row_group.data_path
                        && prev.meta_path == part.row_group.meta_path
                        && prev.fragment_id == part.row_group.fragment_id
                        && prev.meta.rows == part.row_group.meta.rows
                        && prev.deletes == part.row_group.deletes
                    {
                        continue;
                    }
                }
            }
            out.push(part.row_group.clone());
        }
        out
    }

    /// Return the limit.
    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    /// Test-only: whether any partition carries a `_rowid` row-selection. Used to
    /// assert that [`RowIdSelectionPushdown`](crate::rules::RowIdSelectionPushdown)
    /// actually rewrote this scan (`with_rowid_targets` sets `Some(..)` on every
    /// partition; an unfiltered scan leaves them `None`).
    #[cfg(test)]
    pub(crate) fn any_partition_has_row_selection(&self) -> bool {
        self.partitions.iter().any(|p| p.row_selection.is_some())
    }

    /// Return the physical filter expressions pushed into the scan.
    pub fn filters(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.filters
    }

    /// Build a new scan that reads only the supplied subset of fragments,
    /// keeping every other config (storage, schema, projection, filters, limit,
    /// I/O tuning, encryption) identical to this scan.
    ///
    /// Used by the partial-aggregate-pushdown rule to construct a boundary scan
    /// that applies the same pushed filter `F` but over only the BOUNDARY
    /// fragments (those straddling the filter edge), so the fully-covered
    /// fragments are never read from Parquet.  Re-derives `partitions` /
    /// `properties` from `subset` via `Self::new_impl`.
    pub fn with_planned_row_groups(&self, subset: Vec<PlannedRowGroup>) -> Result<Arc<Self>> {
        // target_partitions=1: the boundary scan is tiny by construction and the
        // partial-aggregate rule wraps it in an AggregateExec, so a single
        // partition keeps the rebuilt plan deterministic.
        let scan = Self::new_impl(
            Arc::clone(&self.storage),
            Arc::clone(&self.table_schema),
            subset,
            self.projection.clone(),
            self.filters.clone(),
            self.limit,
            self.batch_size,
            self.io_coalesce_window,
            self.io_concurrency,
            1,
            #[cfg(feature = "encryption")]
            self.decryption_properties.clone(),
        )?;
        Ok(Arc::new(
            scan.with_agg_group_keys(self.agg_group_keys.as_ref().clone()),
        ))
    }

    /// Return a new scan with `filters` replacing the current pushed filters.
    pub fn with_filters(&self, filters: Vec<Arc<dyn PhysicalExpr>>) -> Arc<Self> {
        Arc::new(Self {
            storage: Arc::clone(&self.storage),
            schema: Arc::clone(&self.schema),
            table_schema: Arc::clone(&self.table_schema),
            partitions: self.partitions.clone(),
            projection: self.projection.clone(),
            filters,
            limit: self.limit,
            properties: Arc::clone(&self.properties),
            batch_size: self.batch_size,
            io_coalesce_window: self.io_coalesce_window,
            io_concurrency: self.io_concurrency,
            agg_group_keys: Arc::clone(&self.agg_group_keys),
            #[cfg(feature = "encryption")]
            decryption_properties: self.decryption_properties.clone(),
        })
    }

    /// Return a new scan that decodes only the physical rows whose `_rowid` is in
    /// `targets`. Each fragment becomes one un-split partition carrying a
    /// row-id-derived [`rowid_set_to_row_selection`]; this full-length selection
    /// composes safely with the deletion vector (no short-partition × DV
    /// intersection). The pushed `_rowid IN (...)` filter stays as a post-scan
    /// correctness guard, so an over-broad selection can only ever read extra
    /// rows, never return wrong ones.
    pub fn with_rowid_targets(&self, targets: std::collections::HashSet<u64>) -> Arc<Self> {
        let partitions: Vec<ScanPartition> = self
            .planned_row_groups()
            .into_iter()
            .map(|rg| {
                let row_selection = Some(rowid_set_to_row_selection(
                    &rg.meta.row_ids,
                    &targets,
                    rg.meta.rows,
                ));
                ScanPartition {
                    row_selection,
                    row_group: rg,
                }
            })
            .collect();
        let properties = compute_properties(&self.schema, &partitions);
        Arc::new(Self {
            storage: Arc::clone(&self.storage),
            schema: Arc::clone(&self.schema),
            table_schema: Arc::clone(&self.table_schema),
            partitions,
            projection: self.projection.clone(),
            filters: self.filters.clone(),
            limit: self.limit,
            properties: Arc::new(properties),
            batch_size: self.batch_size,
            io_coalesce_window: self.io_coalesce_window,
            io_concurrency: self.io_concurrency,
            agg_group_keys: Arc::clone(&self.agg_group_keys),
            #[cfg(feature = "encryption")]
            decryption_properties: self.decryption_properties.clone(),
        })
    }
}

/// Minimum rows per intra-row-group partition.  Splitting below this wastes
/// more thread-coordination overhead than it saves.
const MIN_ROWS_PER_PARTITION: usize = 10_000;

/// Split row groups into scan partitions.  Row groups are split when the table
/// has fewer row groups than the target partition count, allowing parallel
/// decoding of a single large row group.
///
/// `allow_intra_split` gates the intra-row-group split.  It MUST be false for
/// filter-free scans: intra-row-group parallelism only pays off when the reader
/// can skip the other partitions' pages via the Parquet offset (page) index, and
/// `build_parquet_reader` only loads that index when there are pushed filters
/// (`want_page_index = !is_sparse && !filters.is_empty()`).  Without it every
/// split partition decompresses the WHOLE row group, so an unfiltered `SELECT *`
/// over one giant row group would pin all cores re-decoding the same bytes N
/// times.  With the gate off such a scan stays one partition per row group:
/// single-threaded for a lone giant group, but no duplicated decode.
fn split_row_groups(
    planned_row_groups: &[PlannedRowGroup],
    target_partitions: usize,
    allow_intra_split: bool,
) -> Vec<ScanPartition> {
    let total_rows: usize = planned_row_groups.iter().map(|rg| rg.meta.rows).sum();
    let desired = target_partitions.max(1);
    if !allow_intra_split
        || planned_row_groups.is_empty()
        || planned_row_groups.len() >= desired
        || total_rows < MIN_ROWS_PER_PARTITION * 2
    {
        return planned_row_groups
            .iter()
            .cloned()
            .map(|row_group| ScanPartition {
                row_group,
                row_selection: None,
            })
            .collect();
    }

    // Compute proportional split counts and round so the total equals `desired`.
    let mut splits: Vec<usize> = planned_row_groups
        .iter()
        .map(|rg| {
            let frac = rg.meta.rows as f64 / total_rows as f64 * desired as f64;
            frac.max(1.0).round() as usize
        })
        .collect();

    // Adjust so the sum equals `desired` (rounding can push it over/under).
    let current: usize = splits.iter().sum();
    if current > desired {
        let excess = current - desired;
        // Trim from the largest splits.
        let mut indexed: Vec<(usize, usize)> =
            splits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
        for (idx, _) in indexed.into_iter().take(excess) {
            if splits[idx] > 1 {
                splits[idx] -= 1;
            }
        }
    } else if current < desired {
        let deficit = desired - current;
        // Add to the largest row groups.
        let mut indexed: Vec<(usize, usize)> = planned_row_groups
            .iter()
            .enumerate()
            .map(|(i, rg)| (i, rg.meta.rows))
            .collect();
        indexed.sort_by_key(|&(_, rows)| std::cmp::Reverse(rows));
        for (idx, _) in indexed.into_iter().take(deficit) {
            splits[idx] += 1;
        }
    }

    let mut partitions = Vec::with_capacity(desired);
    for (rg, &splits_for_rg) in planned_row_groups.iter().zip(splits.iter()) {
        let rows = rg.meta.rows;
        if splits_for_rg == 1 {
            partitions.push(ScanPartition {
                row_group: rg.clone(),
                row_selection: None,
            });
            continue;
        }
        let base = rows / splits_for_rg;
        let rem = rows % splits_for_rg;
        let mut offset = 0;
        for i in 0..splits_for_rg {
            let chunk = base + if i < rem { 1 } else { 0 };
            if chunk == 0 {
                continue;
            }
            // The selection MUST span all `rows` (skip the prefix, select this
            // partition's slice, skip the trailing remainder).  Without the
            // trailing skip the selection covers only `offset + chunk` rows, and
            // `RowSelection::intersection` with the (full-length) deletion-vector
            // selection appends the DV's trailing selectors verbatim — making
            // each partition read a suffix to the end of the row group and
            // grossly duplicating live rows on any table with deletions.
            let mut selectors = vec![RowSelector::skip(offset), RowSelector::select(chunk)];
            let tail = rows - offset - chunk;
            if tail > 0 {
                selectors.push(RowSelector::skip(tail));
            }
            partitions.push(ScanPartition {
                row_group: rg.clone(),
                row_selection: Some(RowSelection::from(selectors)),
            });
            offset += chunk;
        }
    }

    if partitions.is_empty() {
        planned_row_groups
            .iter()
            .cloned()
            .map(|row_group| ScanPartition {
                row_group,
                row_selection: None,
            })
            .collect()
    } else {
        partitions
    }
}

/// Build a `RowSelection` that selects live rows and skips deleted ones.
///
/// Iterates `[0, total_rows)` and emits run-length encoded `select`/`skip`
/// selectors. Adjacent live rows are coalesced into a single `select(n)`.
/// The resulting selection can be intersected with an existing split selection
/// to combine intra-row-group parallelism with deletion filtering.
pub fn deletion_to_row_selection(
    dv: &icefalldb_core::DeletionVector,
    total_rows: usize,
) -> RowSelection {
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut pos: usize = 0;
    for offset in 0..total_rows as u32 {
        let offset_usize = offset as usize;
        if dv.contains(offset) {
            if offset_usize > pos {
                selectors.push(RowSelector::select(offset_usize - pos));
            }
            selectors.push(RowSelector::skip(1));
            pos = offset_usize + 1;
        }
    }
    if pos < total_rows {
        selectors.push(RowSelector::select(total_rows - pos));
    }
    RowSelection::from(selectors)
}

/// Build a full-length `RowSelection` that selects the physical offsets whose
/// row-id is in `targets` and skips the rest, run-length encoded over
/// `[0, total_rows)`. Spanning the full row-group (like
/// [`deletion_to_row_selection`]) lets it intersect safely with the
/// deletion-vector selection; the caller forces a single (un-split) partition
/// when this is present so the intersection is never between a short partition
/// selection and a full-length DV selection.
pub fn rowid_set_to_row_selection(
    row_ids: &[icefalldb_core::RowIdSegment],
    targets: &std::collections::HashSet<u64>,
    total_rows: usize,
) -> RowSelection {
    use icefalldb_core::segment_ids;
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut run_select = 0usize;
    let mut run_skip = 0usize;
    let mut phys = 0usize;
    'outer: for seg in row_ids {
        for id in segment_ids(seg) {
            if phys >= total_rows {
                break 'outer;
            }
            if targets.contains(&id) {
                if run_skip > 0 {
                    selectors.push(RowSelector::skip(run_skip));
                    run_skip = 0;
                }
                run_select += 1;
            } else {
                if run_select > 0 {
                    selectors.push(RowSelector::select(run_select));
                    run_select = 0;
                }
                run_skip += 1;
            }
            phys += 1;
        }
    }
    if run_select > 0 {
        selectors.push(RowSelector::select(run_select));
    }
    if run_skip > 0 {
        selectors.push(RowSelector::skip(run_skip));
    }
    // Physical rows past the row-id segments carry no row-id, so they can never
    // match a target: skip them to keep the selection full-length.
    if phys < total_rows {
        selectors.push(RowSelector::skip(total_rows - phys));
    }
    RowSelection::from(selectors)
}

/// Expand a `RowSelection` into the ordered list of physical row offsets that
/// are selected (i.e. not skipped).  When `row_selection` is `None`, all
/// offsets `0..total_rows` are returned.
///
/// This is used to compute `_rowid` and `_rowaddr` pseudo-column values: the
/// i-th element of the returned vector is the physical offset of the i-th live
/// row and corresponds directly to the i-th output row of the Parquet reader.
fn live_physical_offsets(row_selection: Option<&RowSelection>, total_rows: usize) -> Vec<usize> {
    match row_selection {
        None => (0..total_rows).collect(),
        Some(sel) => {
            let mut offsets = Vec::with_capacity(total_rows);
            let mut phys = 0usize;
            for selector in sel.iter() {
                if selector.skip {
                    phys += selector.row_count;
                } else {
                    for off in phys..phys + selector.row_count {
                        offsets.push(off);
                    }
                    phys += selector.row_count;
                }
            }
            offsets
        }
    }
}

/// Given the fragment's `row_ids` segments, look up the row-id at `offset`
/// (physical index within the fragment).  Returns `None` when `row_ids` is
/// empty (legacy fragment with no allocated ids) or when `offset` is out of
/// range.
/// Map a physical row offset within a fragment to its stable row id.
///
/// Walks the row-id segments by length and indexes the target segment directly.
/// MUST NOT materialize segment id lists: this runs once per output row, so an
/// earlier `segment_ids(seg).collect()` per call made pseudo-column synthesis
/// over an N-row group O(N^2) (it allocated the whole id list on every lookup,
/// pinning all cores for ~minutes on a 1M-row single-segment group). Both
/// variants are O(1) to index, so this is O(segments) per call.
fn row_id_at_offset(row_ids: &[icefalldb_core::RowIdSegment], offset: usize) -> Option<u64> {
    use icefalldb_core::RowIdSegment;
    let mut remaining = offset;
    for seg in row_ids {
        match seg {
            RowIdSegment::Range { start, count } => {
                if (remaining as u64) < *count {
                    return Some(start + remaining as u64);
                }
                remaining -= *count as usize;
            }
            RowIdSegment::Sorted { ids } => {
                if remaining < ids.len() {
                    return Some(ids[remaining]);
                }
                remaining -= ids.len();
            }
        }
    }
    None
}

fn compute_properties(output_schema: &SchemaRef, partitions: &[ScanPartition]) -> PlanProperties {
    let mut eq_properties = EquivalenceProperties::new(Arc::clone(output_schema));
    if let Some(sort) = shared_sort(partitions) {
        let sort_exprs: Vec<PhysicalSortExpr> = sort
            .iter()
            .filter_map(|name| {
                output_schema.index_of(name).ok().map(|idx| {
                    PhysicalSortExpr::new_default(Arc::new(Column::new(name, idx))).asc()
                })
            })
            .collect();
        if !sort_exprs.is_empty() {
            eq_properties.add_ordering(sort_exprs);
        }
    }
    let partitioning = Partitioning::UnknownPartitioning(partitions.len().max(1));
    PlanProperties::new(
        eq_properties,
        partitioning,
        EmissionType::Incremental,
        Boundedness::Bounded,
    )
}

fn shared_sort(partitions: &[ScanPartition]) -> Option<&[String]> {
    let mut iter = partitions.iter();
    let first = iter.next()?.row_group.meta.sort.as_deref()?;
    if iter.any(|p| p.row_group.meta.sort.as_deref() != Some(first)) {
        return None;
    }
    Some(first)
}

impl DisplayAs for IcefallDBScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "IcefallDBScanExec: partitions={}, projection={:?}, filters={}, limit={:?}",
            self.partitions.len(),
            self.projection,
            self.filters.len(),
            self.limit
        )
    }
}

impl ExecutionPlan for IcefallDBScanExec {
    fn name(&self) -> &str {
        "IcefallDBScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        assert!(children.is_empty(), "IcefallDBScanExec has no children");
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion::common::Result<SendableRecordBatchStream> {
        let partition_count = self.properties.partitioning.partition_count();
        if partition >= partition_count {
            return Err(DataFusionError::Internal(format!(
                "Invalid partition index {partition}, partition count {partition_count}"
            )));
        }

        let stream_schema = Arc::clone(&self.schema);
        if self.partitions.is_empty() {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                stream_schema,
                futures::stream::empty(),
            )));
        }

        let part = self.partitions[partition].clone();
        let storage = Arc::clone(&self.storage);
        let reader_schema = Arc::clone(&self.table_schema);

        // Determine which output columns are pseudo-columns and at which
        // positions they appear in the output schema.  Pseudo-columns are NOT
        // in the Parquet file; they must be synthesised after reading.
        let output_fields: Vec<(usize, String)> = stream_schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| (i, f.name().clone()))
            .collect();
        let need_rowid = output_fields.iter().any(|(_, n)| n == PSEUDO_COL_ROWID);
        let need_rowaddr = output_fields.iter().any(|(_, n)| n == PSEUDO_COL_ROWADDR);
        let has_pseudo = need_rowid || need_rowaddr;

        // Build a reduced schema that contains only real (Parquet) columns.
        // This is what gets passed into build_parquet_reader.
        let parquet_schema: SchemaRef = if has_pseudo {
            Arc::new(Schema::new(
                stream_schema
                    .fields()
                    .iter()
                    .filter(|f| !is_pseudo_column(f.name()))
                    .cloned()
                    .collect::<Vec<_>>(),
            ))
        } else {
            Arc::clone(&stream_schema)
        };

        // For the sparse I/O path, only pass real column names.
        let projected_names: Vec<String> = parquet_schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let filters = self.filters.clone();

        // When pseudo-columns are projected, the Parquet RowFilter approach breaks
        // pseudo-column offset tracking: RowFilter skips rows inside the Parquet
        // reader, making the `consumed` index diverge from physical row offsets in
        // `live_offsets`.  To fix this, when `has_pseudo = true` and filters are
        // present we:
        //   1. Expand `parquet_schema` to include any filter-only columns so the
        //      Parquet reader can evaluate them post-injection.
        //   2. Pass an empty filter list to `build_parquet_reader` (no RowFilter).
        //   3. After injecting pseudo-columns, evaluate the filter expressions
        //      against the full batch and apply the resulting boolean mask.
        //   4. Project the filtered batch to `stream_schema` (dropping filter-only
        //      columns).
        //
        // When `has_pseudo = false` the existing RowFilter path is used unchanged,
        // so there is no regression for non-pseudo-column queries.
        let (parquet_schema, parquet_read_filters, post_filters) =
            if has_pseudo && !filters.is_empty() {
                // Collect filter-only column names (those not already in parquet_schema).
                let existing_names: std::collections::HashSet<String> = parquet_schema
                    .fields()
                    .iter()
                    .map(|f| f.name().clone())
                    .collect();
                let mut filter_col_names: std::collections::HashSet<String> =
                    std::collections::HashSet::new();
                for f in &filters {
                    collect_column_names(f.as_ref(), &mut filter_col_names);
                }
                // Build the expanded Parquet schema: projected real cols + filter-only cols.
                let extra_fields: Vec<_> = reader_schema
                    .fields()
                    .iter()
                    .filter(|f| {
                        filter_col_names.contains(f.name())
                            && !existing_names.contains(f.name())
                            && !is_pseudo_column(f.name())
                    })
                    .cloned()
                    .collect();
                let expanded_schema: SchemaRef = if extra_fields.is_empty() {
                    Arc::clone(&parquet_schema)
                } else {
                    let mut all_fields: Vec<_> = parquet_schema.fields().iter().cloned().collect();
                    all_fields.extend(extra_fields);
                    Arc::new(Schema::new(all_fields))
                };
                // Move filters to the post-scan step; read without RowFilter.
                let post = filters.clone();
                (expanded_schema, vec![], Some(post))
            } else {
                (parquet_schema, filters.clone(), None)
            };
        let limit = self.limit;
        let batch_size = self.batch_size;
        let io_coalesce_window = self.io_coalesce_window;
        let io_concurrency = self.io_concurrency;
        #[cfg(feature = "encryption")]
        let decryption_properties = self.decryption_properties.clone();

        // Positions of pseudo-columns in the output schema (for re-insertion).
        let rowid_out_pos: Option<usize> = if need_rowid {
            output_fields
                .iter()
                .find(|(_, n)| n == PSEUDO_COL_ROWID)
                .map(|(i, _)| *i)
        } else {
            None
        };
        let rowaddr_out_pos: Option<usize> = if need_rowaddr {
            output_fields
                .iter()
                .find(|(_, n)| n == PSEUDO_COL_ROWADDR)
                .map(|(i, _)| *i)
        } else {
            None
        };

        let fut = {
            let stream_schema = Arc::clone(&stream_schema);
            async move {
                let mut needed: std::collections::HashSet<String> =
                    projected_names.into_iter().collect();
                for f in &filters {
                    collect_column_names(f.as_ref(), &mut needed);
                }
                let needed_refs: Vec<&str> = needed.iter().map(|s| s.as_str()).collect();
                let (bytes, is_sparse) = read_row_group_bytes(
                    storage.as_ref(),
                    &part.row_group,
                    &needed_refs,
                    io_coalesce_window,
                    io_concurrency,
                )
                .await?;

                // Apply deletion vector if this row group carries one.
                let effective_row_selection = if let Some(ref del_path) = part.row_group.deletes {
                    let del_bytes = storage
                        .read(del_path)
                        .await
                        .map_err(|e| DataFusionError::External(Box::new(QueryError::Core(e))))?;
                    let dv =
                        icefalldb_core::DeletionVector::deserialize(&del_bytes).map_err(|e| {
                            DataFusionError::External(Box::new(QueryError::Other(e.to_string())))
                        })?;
                    let total_rows = part.row_group.meta.rows;
                    let del_sel = deletion_to_row_selection(&dv, total_rows);
                    Some(match part.row_selection {
                        Some(existing) => existing.intersection(&del_sel),
                        None => del_sel,
                    })
                } else {
                    part.row_selection
                };

                // Pre-compute the live physical offsets when pseudo-columns are
                // needed.  The list is in physical order; the i-th element is
                // the physical offset of the i-th output row of the Parquet
                // reader.  We consume this slice batch-by-batch below.
                let live_offsets: Option<Arc<Vec<usize>>> = if has_pseudo {
                    Some(Arc::new(live_physical_offsets(
                        effective_row_selection.as_ref(),
                        part.row_group.meta.rows,
                    )))
                } else {
                    None
                };
                let row_ids = Arc::new(part.row_group.meta.row_ids.clone());
                let fragment_id = part.row_group.fragment_id;

                let reader = build_parquet_reader(
                    bytes,
                    is_sparse,
                    &parquet_schema,
                    &reader_schema,
                    parquet_read_filters,
                    limit,
                    batch_size,
                    effective_row_selection,
                    #[cfg(feature = "encryption")]
                    decryption_properties.as_deref(),
                )?;

                // Track how many output rows we have consumed so far so we can
                // slice `live_offsets` to the correct window per batch.
                let mut consumed: usize = 0;
                let iter_stream = futures::stream::iter(reader).map(
                    move |r| -> datafusion::common::Result<RecordBatch> {
                        let parquet_batch: RecordBatch = r.map_err(DataFusionError::from)?;
                        if !has_pseudo {
                            return Ok(parquet_batch);
                        }

                        // Slice the relevant window of physical offsets for this
                        // batch's rows.
                        let n = parquet_batch.num_rows();
                        let offsets_window: &[usize] = live_offsets
                            .as_ref()
                            .map(|v| &v[consumed..consumed + n])
                            .unwrap_or(&[]);
                        consumed += n;

                        // When pseudo-columns are present and filters could not be
                        // pushed into the Parquet RowFilter (to preserve offset
                        // tracking), apply the filters as a post-scan step and
                        // project down to `stream_schema` at the end.
                        if let Some(ref post) = post_filters {
                            // The parquet_batch has columns from the expanded
                            // parquet_schema (real projected cols + filter-only
                            // cols).  Synthesize the pseudo-column arrays directly
                            // from the physical offsets and append them to produce a
                            // "wide batch" that the filter expressions can see.
                            let num_rows = parquet_batch.num_rows();
                            let rowid_vals: Vec<u64> = offsets_window
                                .iter()
                                .map(|&off| row_id_at_offset(&row_ids, off).unwrap_or(0))
                                .collect();
                            let rowaddr_vals: Vec<u64> = offsets_window
                                .iter()
                                .map(|&off| (fragment_id << 32) | (off as u64))
                                .collect();
                            debug_assert_eq!(rowid_vals.len(), num_rows);

                            // Build wide batch: parquet columns + _rowid + _rowaddr.
                            let rowid_col: Arc<dyn arrow::array::Array> =
                                Arc::new(UInt64Array::from(rowid_vals));
                            let rowaddr_col: Arc<dyn arrow::array::Array> =
                                Arc::new(UInt64Array::from(rowaddr_vals));

                            let mut wide_fields: Vec<arrow::datatypes::FieldRef> =
                                parquet_schema.fields().iter().cloned().collect();
                            wide_fields.push(Arc::new(arrow::datatypes::Field::new(
                                PSEUDO_COL_ROWID,
                                arrow::datatypes::DataType::UInt64,
                                false,
                            )));
                            wide_fields.push(Arc::new(arrow::datatypes::Field::new(
                                PSEUDO_COL_ROWADDR,
                                arrow::datatypes::DataType::UInt64,
                                false,
                            )));
                            let wide_schema = Arc::new(Schema::new(wide_fields));

                            let mut wide_cols: Vec<Arc<dyn arrow::array::Array>> =
                                parquet_batch.columns().to_vec();
                            wide_cols.push(rowid_col);
                            wide_cols.push(rowaddr_col);

                            let wide_batch =
                                RecordBatch::try_new(Arc::clone(&wide_schema), wide_cols)
                                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

                            // Evaluate each filter against the wide batch, remapping
                            // column indices from reader_schema to wide_schema.
                            let mut mask: Option<BooleanArray> = None;
                            for f in post {
                                let remapped = remap_expr_to_schema(
                                    Arc::clone(f),
                                    &reader_schema,
                                    &wide_schema,
                                )?;
                                let result = evaluate_filter(&remapped, &wide_batch, &wide_schema)
                                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;
                                mask = Some(match mask {
                                    None => result,
                                    Some(prev) => {
                                        arrow::compute::and(&prev, &result).map_err(|e| {
                                            DataFusionError::ArrowError(Box::new(e), None)
                                        })?
                                    }
                                });
                            }
                            let filtered = match mask {
                                None => wide_batch,
                                Some(m) => arrow::compute::filter_record_batch(&wide_batch, &m)
                                    .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?,
                            };

                            // Project wide batch to stream_schema (removes filter-
                            // only columns, keeps projected real cols + pseudo-cols).
                            let columns: Vec<Arc<dyn arrow::array::Array>> = stream_schema
                                .fields()
                                .iter()
                                .map(|f| {
                                    let idx =
                                        filtered.schema().index_of(f.name()).map_err(|e| {
                                            DataFusionError::ArrowError(Box::new(e), None)
                                        })?;
                                    Ok(Arc::clone(filtered.column(idx)))
                                })
                                .collect::<datafusion::common::Result<Vec<_>>>()?;
                            return RecordBatch::try_new(Arc::clone(&stream_schema), columns)
                                .map_err(|e| DataFusionError::ArrowError(Box::new(e), None));
                        }

                        inject_pseudo_columns(
                            parquet_batch,
                            &stream_schema,
                            offsets_window,
                            &row_ids,
                            fragment_id,
                            rowid_out_pos,
                            rowaddr_out_pos,
                        )
                    },
                );
                Ok::<_, DataFusionError>(iter_stream)
            }
        };

        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            stream,
        )))
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> datafusion::common::Result<Arc<Statistics>> {
        // When filters are pushed down, the per-partition row count is no longer
        // the raw row-group count.  Returning exact unfiltered statistics here
        // would let DataFusion's AggregateStatistics rule fold COUNT(*) into the
        // unfiltered total, which is incorrect for filtered scans.
        if !self.filters.is_empty() {
            return Ok(Arc::new(Statistics::new_unknown(&self.schema)));
        }

        let stats = match partition {
            Some(idx) => {
                let part = self.partitions.get(idx).ok_or_else(|| {
                    DataFusionError::Internal(format!("Invalid partition index {idx}"))
                })?;
                // A split partition (`row_selection.is_some()`) covers only a
                // slice of a row group, so the row group's whole-group
                // `deleted_count` does NOT apply to it.  When such a partition
                // also carries a deletion vector, the exact post-deletion count
                // of its slice is unknowable without reading the DV, so report
                // unknown statistics and let the scan compute COUNT(*) honestly
                // (an exact-but-wrong count here folds COUNT(*) to the inflated
                // physical total via AggregateStatistics).
                if part.row_selection.is_some() && part.row_group.deletes.is_some() {
                    Statistics::new_unknown(&self.schema)
                } else {
                    row_group_statistics(&part.row_group, &self.schema)
                }
            }
            None => {
                // Same reasoning for the combined statistics: if any partition is
                // a split slice of a deletion-bearing row group, the per-slice
                // live count is not cheaply exact, so fall back to a real scan.
                if self
                    .partitions
                    .iter()
                    .any(|p| p.row_selection.is_some() && p.row_group.deletes.is_some())
                {
                    Statistics::new_unknown(&self.schema)
                } else {
                    let mut combined = Statistics::new_unknown(&self.schema);
                    let num_rows: usize = self.partitions.iter().map(|p| p.row_count()).sum();
                    combined.num_rows = Precision::Exact(num_rows);
                    combined
                }
            }
        };
        Ok(Arc::new(stats))
    }

    fn supports_limit_pushdown(&self) -> bool {
        true
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        Some(Arc::new(Self {
            storage: Arc::clone(&self.storage),
            schema: Arc::clone(&self.schema),
            table_schema: Arc::clone(&self.table_schema),
            partitions: self.partitions.clone(),
            projection: self.projection.clone(),
            filters: self.filters.clone(),
            limit,
            properties: Arc::clone(&self.properties),
            batch_size: self.batch_size,
            io_coalesce_window: self.io_coalesce_window,
            io_concurrency: self.io_concurrency,
            agg_group_keys: Arc::clone(&self.agg_group_keys),
            #[cfg(feature = "encryption")]
            decryption_properties: self.decryption_properties.clone(),
        }))
    }

    fn fetch(&self) -> Option<usize> {
        self.limit
    }
}

/// Read the bytes for a single row group, using sparse range reads when
/// sidecar offsets are available and falling back to a full-file read otherwise.
///
async fn read_row_group_bytes(
    storage: &dyn Storage,
    rg: &PlannedRowGroup,
    needed_names: &[&str],
    io_coalesce_window: u64,
    io_concurrency: usize,
) -> Result<(Bytes, bool)> {
    let path = &rg.data_path;
    let file_size = storage.size(path).await?;

    if let Some(ref offsets) = rg.meta.column_offsets {
        if needed_names.iter().all(|name| offsets.contains_key(*name)) {
            // Try the process-wide metadata cache first. A hit avoids reading
            // the Parquet footer and decoding metadata entirely.
            let (metadata, metadata_len) = if let Some((cached, len)) =
                ParquetMetadataCache::global().get_with_footer(path, &rg.meta.checksum)
            {
                (cached, len)
            } else {
                // Read footer tail to discover metadata length.
                let footer_tail = storage
                    .read_range(path, file_size - PARQUET_FOOTER_SIZE, PARQUET_FOOTER_SIZE)
                    .await?;
                let metadata_len = parse_footer_metadata_len(&footer_tail)? as u64;
                let footer_start = file_size - metadata_len - PARQUET_FOOTER_SIZE;

                let footer_bytes = storage
                    .read_range(path, footer_start, metadata_len + PARQUET_FOOTER_SIZE)
                    .await?;
                let metadata =
                    ParquetMetaDataReader::decode_metadata(&footer_bytes[..metadata_len as usize])?;
                let metadata = Arc::new(metadata);
                ParquetMetadataCache::global().put(
                    path,
                    &rg.meta.checksum,
                    Arc::clone(&metadata),
                    metadata_len,
                );
                (metadata, metadata_len)
            };
            let footer_start = file_size - metadata_len - PARQUET_FOOTER_SIZE;

            // The sidecar column_offsets describe a single row group. If the
            // Parquet file contains multiple row groups, the offsets are only
            // valid for the first one and sparse reconstruction would leave the
            // remaining row groups as zero-filled holes. Fall back to a whole-file
            // read so every row group is present.
            if metadata.num_row_groups() > 1 {
                let bytes = storage.read(path).await?;
                return Ok((Bytes::from(bytes), false));
            }

            // Read only the column chunks we actually need.  The in-memory file
            // buffer is reconstructed by leaving holes for unneeded columns;
            // Parquet's metadata still points to the correct offsets.
            let mut ranges: Vec<Range<u64>> = needed_names
                .iter()
                .filter_map(|name| offsets.get(*name))
                .map(|off| off.offset..off.offset + off.length)
                .collect();
            ranges.push(0..PARQUET_MAGIC_SIZE);
            ranges.push(footer_start..file_size);

            // NOTE: We intentionally do NOT fetch page indexes for sparse
            // reconstructed buffers. The zero-filled holes left for unneeded
            // column chunks can cause Parquet's page-index decoder to fail with
            // errors such as "Required field type_ is missing". DataFusion's
            // RowFilter still evaluates pushed predicates correctly without page
            // indexes; we simply lose page-level pruning inside the row group.
            let coalesced = coalesce_ranges(&mut ranges, io_coalesce_window);
            let fetched = read_ranges_concurrent(storage, path, &coalesced, io_concurrency).await?;

            let mut buffer = vec![0u8; file_size as usize];
            for (offset, bytes) in fetched {
                buffer[offset as usize..offset as usize + bytes.len()].copy_from_slice(&bytes);
            }
            return Ok((Bytes::from(buffer), true));
        }
    }

    // Fallback: read the whole Parquet file. Page indexes are present and valid
    // because the buffer is a complete copy of the file.
    let bytes = storage.read(path).await?;
    Ok((Bytes::from(bytes), false))
}

fn parse_footer_metadata_len(tail: &[u8]) -> Result<u32> {
    if tail.len() != PARQUET_FOOTER_SIZE as usize {
        return Err(QueryError::Other(format!(
            "invalid parquet footer tail length: {}",
            tail.len()
        )));
    }
    let len = u32::from_le_bytes(tail[..4].try_into().unwrap());
    Ok(len)
}

async fn read_ranges_concurrent(
    storage: &dyn Storage,
    path: &str,
    ranges: &[Range<u64>],
    io_concurrency: usize,
) -> Result<Vec<(u64, Vec<u8>)>> {
    let stream = futures::stream::iter(ranges.iter().cloned())
        .map(|range| async move {
            let len = range.end - range.start;
            let bytes = storage.read_range(path, range.start, len).await?;
            Ok::<_, QueryError>((range.start, bytes))
        })
        .buffer_unordered(io_concurrency.max(1));
    stream.try_collect().await
}

/// Build a `ParquetRecordBatchReader` from in-memory Parquet bytes, applying
/// projection, optional row filters, limit, batch size, and (when encryption
/// is enabled) decryption properties.
#[allow(clippy::too_many_arguments)]
fn build_parquet_reader(
    bytes: Bytes,
    is_sparse: bool,
    output_schema: &SchemaRef,
    table_schema: &SchemaRef,
    filters: Vec<Arc<dyn PhysicalExpr>>,
    limit: Option<usize>,
    batch_size: usize,
    row_selection: Option<RowSelection>,
    #[cfg(feature = "encryption")] decryption_properties: Option<
        &parquet::encryption::decrypt::FileDecryptionProperties,
    >,
) -> Result<ParquetRecordBatchReader> {
    // For sparse reconstructed buffers, page-index decoding can fail at read
    // time because unneeded column chunks are left as zero-filled holes. Always
    // use the default reader for sparse buffers; RowFilter still applies
    // predicates correctly without page-level pruning.
    let want_page_index = !is_sparse && !filters.is_empty();
    #[allow(unused_mut)]
    let mut options = if want_page_index {
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional)
    } else {
        ArrowReaderOptions::default()
    };
    #[cfg(feature = "encryption")]
    if let Some(dec) = decryption_properties {
        options = options.with_file_decryption_properties(std::sync::Arc::new(dec.clone()));
    }

    // If page-index decoding fails (e.g. the buffer is incomplete), fall back to
    // the default reader so that row-filter evaluation still produces correct
    // results. We preserve decryption properties across the fallback.
    let mut builder =
        match ParquetRecordBatchReaderBuilder::try_new_with_options(bytes.clone(), options) {
            Ok(b) => b,
            Err(_) if want_page_index => {
                let fallback = ArrowReaderOptions::default();
                #[cfg(feature = "encryption")]
                let fallback = if let Some(dec) = decryption_properties {
                    fallback.with_file_decryption_properties(std::sync::Arc::new(dec.clone()))
                } else {
                    fallback
                };
                ParquetRecordBatchReaderBuilder::try_new_with_options(bytes, fallback)?
            }
            Err(e) => return Err(e.into()),
        };
    let schema_descr = builder.metadata().file_metadata().schema_descr().clone();

    let projection = projection_mask_from_names(&schema_descr, output_schema)?;
    builder = builder.with_projection(projection);

    if let Some(selection) = row_selection {
        builder = builder.with_row_selection(selection);
    }

    if !filters.is_empty() {
        let row_filter = build_row_filter(filters, table_schema, &schema_descr)?;
        builder = builder.with_row_filter(row_filter);
    }

    if let Some(limit) = limit {
        builder = builder.with_limit(limit);
    }

    builder = builder.with_batch_size(batch_size);
    Ok(builder.build()?)
}

fn projection_mask_from_names(
    schema_descr: &parquet::schema::types::SchemaDescriptor,
    output_schema: &SchemaRef,
) -> Result<ProjectionMask> {
    let root_fields = schema_descr.root_schema().get_fields();
    let mut indices = Vec::new();
    for name in output_schema.fields().iter().map(|f| f.name()) {
        let idx = root_fields
            .iter()
            .position(|f| f.name() == name)
            .ok_or_else(|| {
                QueryError::Other(format!("column '{}' not found in parquet schema", name))
            })?;
        indices.push(idx);
    }
    Ok(ProjectionMask::roots(schema_descr, indices))
}

fn build_row_filter(
    filters: Vec<Arc<dyn PhysicalExpr>>,
    table_schema: &SchemaRef,
    schema_descr: &parquet::schema::types::SchemaDescriptor,
) -> Result<RowFilter> {
    let root_fields = schema_descr.root_schema().get_fields();

    let predicates = filters
        .into_iter()
        .map(|expr| {
            // Determine the columns this filter actually references and build a
            // reduced schema / Parquet projection mask that reads only those
            // columns for predicate evaluation.
            let referenced = referenced_columns(Arc::clone(&expr), table_schema);
            let filter_schema = Arc::new(Schema::new(
                referenced
                    .iter()
                    .map(|idx| table_schema.field(*idx).clone())
                    .collect::<Vec<Field>>(),
            ));
            let mask_indices: Vec<usize> = referenced
                .iter()
                .map(|idx| {
                    let name = table_schema.field(*idx).name();
                    root_fields
                        .iter()
                        .position(|f| f.name() == name)
                        .ok_or_else(|| {
                            QueryError::Other(format!(
                                "filter column '{}' not found in parquet schema",
                                name
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            let mask = ProjectionMask::roots(schema_descr, mask_indices);
            let remapped = remap_expr_to_schema(Arc::clone(&expr), table_schema, &filter_schema)?;

            let predicate = ArrowPredicateFn::new(mask, move |batch: RecordBatch| {
                evaluate_filter(&remapped, &batch, &filter_schema)
            });
            Ok(Box::new(predicate) as Box<dyn parquet::arrow::arrow_reader::ArrowPredicate>)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(RowFilter::new(predicates))
}

/// Return the column indices (in `schema` order) referenced by `expr`.
fn referenced_columns(expr: Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> Vec<usize> {
    let mut set = std::collections::HashSet::new();
    collect_columns(expr.as_ref(), &mut set);
    (0..schema.fields().len())
        .filter(|idx| set.contains(idx))
        .collect()
}

fn collect_columns(expr: &dyn PhysicalExpr, out: &mut std::collections::HashSet<usize>) {
    if let Some(col) = (expr as &dyn Any).downcast_ref::<Column>() {
        out.insert(col.index());
        return;
    }
    for child in expr.children() {
        collect_columns(child.as_ref(), out);
    }
}

fn collect_column_names(expr: &dyn PhysicalExpr, out: &mut std::collections::HashSet<String>) {
    if let Some(col) = (expr as &dyn Any).downcast_ref::<Column>() {
        out.insert(col.name().to_string());
        return;
    }
    for child in expr.children() {
        collect_column_names(child.as_ref(), out);
    }
}

/// Rewrite `expr` so that its column indices refer to `new_schema` instead of
/// `old_schema`.  Both schemas must contain the referenced columns in the same
/// relative order.
fn remap_expr_to_schema(
    expr: Arc<dyn PhysicalExpr>,
    old_schema: &SchemaRef,
    new_schema: &SchemaRef,
) -> Result<Arc<dyn PhysicalExpr>> {
    if let Some(col) = (&*expr as &dyn Any).downcast_ref::<Column>() {
        let old_name = old_schema.field(col.index()).name();
        let new_index = new_schema
            .fields()
            .iter()
            .position(|f| f.name() == old_name)
            .ok_or_else(|| {
                QueryError::Other(format!(
                    "column '{}' not found in reduced filter schema",
                    old_name
                ))
            })?;
        return Ok(Arc::new(Column::new(old_name, new_index)) as Arc<dyn PhysicalExpr>);
    }

    let children = expr.children();
    if children.is_empty() {
        return Ok(Arc::clone(&expr));
    }

    let new_children: Vec<Arc<dyn PhysicalExpr>> = children
        .iter()
        .map(|c| remap_expr_to_schema(Arc::clone(c), old_schema, new_schema))
        .collect::<Result<Vec<_>>>()?;
    expr.with_new_children(new_children)
        .map_err(QueryError::DataFusion)
}

fn evaluate_filter(
    expr: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
    _output_schema: &SchemaRef,
) -> arrow::error::Result<BooleanArray> {
    // The batch already contains exactly the columns referenced by the filter,
    // in the reduced schema order, so evaluate directly.
    match expr.evaluate(batch) {
        Ok(ColumnarValue::Array(array)) => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .cloned()
            .ok_or_else(|| {
                arrow::error::ArrowError::ComputeError(
                    "filter expression did not evaluate to a boolean array".into(),
                )
            }),
        Ok(ColumnarValue::Scalar(scalar)) => {
            let array = scalar
                .to_array_of_size(batch.num_rows())
                .map_err(|e| arrow::error::ArrowError::ComputeError(e.to_string()))?;
            array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .cloned()
                .ok_or_else(|| {
                    arrow::error::ArrowError::ComputeError(
                        "filter scalar did not evaluate to a boolean".into(),
                    )
                })
        }
        Err(e) => Err(arrow::error::ArrowError::ComputeError(e.to_string())),
    }
}

/// Insert `_rowid` and/or `_rowaddr` pseudo-columns into a Parquet-sourced
/// batch at the positions they occupy in `output_schema`.
///
/// `offsets` is a slice of physical row offsets (one per output row) in the
/// same order as the rows of `parquet_batch`.
///
/// Columns that are not pseudo-columns are taken verbatim from `parquet_batch`
/// in their schema order.  The result has exactly `output_schema`'s field
/// sequence.
#[allow(clippy::too_many_arguments)]
fn inject_pseudo_columns(
    parquet_batch: RecordBatch,
    output_schema: &SchemaRef,
    offsets: &[usize],
    row_ids: &[icefalldb_core::RowIdSegment],
    fragment_id: u64,
    rowid_out_pos: Option<usize>,
    rowaddr_out_pos: Option<usize>,
) -> datafusion::common::Result<RecordBatch> {
    // Build the _rowid and _rowaddr arrays once (if needed), using the
    // physical offsets aligned to the surviving rows.
    let rowid_array: Option<Arc<UInt64Array>> = rowid_out_pos.map(|_| {
        let vals: Vec<u64> = offsets
            .iter()
            .map(|&off| row_id_at_offset(row_ids, off).unwrap_or(0))
            .collect();
        Arc::new(UInt64Array::from(vals))
    });
    let rowaddr_array: Option<Arc<UInt64Array>> = rowaddr_out_pos.map(|_| {
        let vals: Vec<u64> = offsets
            .iter()
            .map(|&off| (fragment_id << 32) | (off as u64))
            .collect();
        Arc::new(UInt64Array::from(vals))
    });

    // Rebuild columns in output_schema order.
    //
    // ORDERING ASSUMPTION: `parquet_batch` columns are in the same relative
    // order as the non-pseudo fields of `output_schema`.  This holds because
    // `parquet_schema` is constructed by filtering pseudo-columns out of
    // `stream_schema` (which IS `output_schema`) in field order — see the
    // `parquet_schema` construction in `execute()` above.  Any refactor that
    // changes how `parquet_schema` is built MUST preserve this ordering
    // invariant or the column stitching below will misalign real and synthetic
    // columns.
    let mut parquet_col_idx: usize = 0; // next column to consume from parquet_batch
    let mut columns: Vec<Arc<dyn arrow::array::Array>> =
        Vec::with_capacity(output_schema.fields().len());

    for (out_idx, _field) in output_schema.fields().iter().enumerate() {
        if Some(out_idx) == rowid_out_pos {
            columns.push(rowid_array.as_ref().unwrap().clone() as Arc<dyn arrow::array::Array>);
        } else if Some(out_idx) == rowaddr_out_pos {
            columns.push(rowaddr_array.as_ref().unwrap().clone() as Arc<dyn arrow::array::Array>);
        } else {
            // Real Parquet column — take from parquet_batch in order.
            columns.push(Arc::clone(parquet_batch.column(parquet_col_idx)));
            parquet_col_idx += 1;
        }
    }

    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

fn row_group_statistics(rg: &PlannedRowGroup, schema: &SchemaRef) -> Statistics {
    let mut stats = Statistics::new_unknown(schema);
    // Report live rows (physical minus deleted) so that DataFusion's statistics-
    // based optimizers see the correct post-deletion count.
    stats.num_rows = Precision::Exact(rg.meta.rows.saturating_sub(rg.deleted_count as usize));
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rowid_set_selection_picks_matching_offsets() {
        use icefalldb_core::RowIdSegment;
        use std::collections::HashSet;
        // Physical offsets 0..7 map to row_ids: Range[100..105) then Sorted[7, 9].
        //   offset 0->100 1->101 2->102 3->103 4->104 5->7 6->9
        let segs = vec![
            RowIdSegment::Range {
                start: 100,
                count: 5,
            },
            RowIdSegment::Sorted { ids: vec![7, 9] },
        ];
        let targets: HashSet<u64> = [101u64, 103, 9].into_iter().collect();
        let sel = rowid_set_to_row_selection(&segs, &targets, 7);
        // Exactly the offsets whose row_id is targeted: 101->1, 103->3, 9->6.
        assert_eq!(live_physical_offsets(Some(&sel), 7), vec![1, 3, 6]);
    }

    #[test]
    fn rowid_set_selection_empty_targets_selects_nothing() {
        use icefalldb_core::RowIdSegment;
        use std::collections::HashSet;
        let segs = vec![RowIdSegment::Range { start: 0, count: 4 }];
        let sel = rowid_set_to_row_selection(&segs, &HashSet::new(), 4);
        assert!(live_physical_offsets(Some(&sel), 4).is_empty());
    }

    #[test]
    fn rowid_set_selection_all_targets_selects_all() {
        use icefalldb_core::RowIdSegment;
        use std::collections::HashSet;
        let segs = vec![RowIdSegment::Range { start: 0, count: 4 }];
        let targets: HashSet<u64> = [0u64, 1, 2, 3].into_iter().collect();
        let sel = rowid_set_to_row_selection(&segs, &targets, 4);
        assert_eq!(live_physical_offsets(Some(&sel), 4), vec![0, 1, 2, 3]);
    }

    use crate::metadata_cache::ParquetMetadataCache;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{binary, col, lit};
    use futures::TryStreamExt;
    use icefalldb_core::metadata::{ColumnChunkOffset, ColumnStats, RowGroupMeta};
    use icefalldb_core::storage::memory::MemoryStorage;
    use parquet::arrow::ArrowWriter;
    use sha2::Digest;
    use std::collections::HashMap;

    fn make_test_batch() -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
            ],
        )
        .unwrap()
    }

    fn write_batch(batch: &RecordBatch) -> Vec<u8> {
        let props = parquet::file::properties::WriterProperties::builder()
            .set_dictionary_enabled(false)
            .build();
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
        buf
    }

    fn compute_offsets(bytes: &[u8], names: &[&str]) -> Option<HashMap<String, ColumnChunkOffset>> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes.to_vec())).ok()?;
        let metadata = builder.metadata();
        let rg = metadata.row_groups().first()?;
        let schema_descr = metadata.file_metadata().schema_descr();
        let mut map = HashMap::new();
        for name in names {
            let leaf_idx = schema_descr
                .columns()
                .iter()
                .position(|c| c.name() == *name)?;
            let col_meta = rg.column(leaf_idx);
            map.insert(
                (*name).to_string(),
                ColumnChunkOffset {
                    offset: col_meta.data_page_offset().max(0) as u64,
                    length: col_meta.compressed_size().max(0) as u64,
                },
            );
        }
        Some(map)
    }

    fn make_planned_row_group(
        data_path: &str,
        rows: usize,
        offsets: Option<HashMap<String, ColumnChunkOffset>>,
    ) -> PlannedRowGroup {
        make_planned_row_group_with_checksum(data_path, rows, offsets, "")
    }

    fn make_planned_row_group_with_checksum(
        data_path: &str,
        rows: usize,
        offsets: Option<HashMap<String, ColumnChunkOffset>>,
        checksum: &str,
    ) -> PlannedRowGroup {
        PlannedRowGroup {
            data_path: data_path.into(),
            meta_path: "test/rg.meta".into(),
            meta: RowGroupMeta {
                row_group: "rg".into(),
                schema_id: 1,
                rows,
                columns: [
                    (
                        "id".into(),
                        ColumnStats {
                            min: None,
                            max: None,
                            nulls: 0,
                        },
                    ),
                    (
                        "name".into(),
                        ColumnStats {
                            min: None,
                            max: None,
                            nulls: 0,
                        },
                    ),
                ]
                .into(),
                column_offsets: offsets,
                sort: None,
                row_ids: vec![],
                checksum: checksum.into(),
                meta_checksum: String::new(),
            },
            partition_values: None,
            snapshot: 0,
            fallback: false,
            deletes: None,
            deleted_count: 0,
            fragment_id: 0,
            agg_state: None,
        }
    }

    /// Splitting a SINGLE Parquet row group into N parallel `RowSelection`
    /// partitions and applying a pushed `RowFilter` MUST return exactly the same
    /// surviving rows as an unsplit single-partition scan — no row may be dropped
    /// or double-counted across the split boundary.
    ///
    /// This guards the parallel-scan routing for single-row-group filtered scans
    /// (`should_use_native_parquet` returns false for them so the custom scan's
    /// intra-row-group split runs). A bug where a partition's `RowSelection` and
    /// the `RowFilter` compose incorrectly would change the total COUNT/SUM.
    #[tokio::test]
    async fn test_single_row_group_split_filter_byte_equal_to_full_scan() {
        use arrow::array::{Float64Array, Int64Array};

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        // One Parquet row group with 50_000 rows. `id` is a dense counter and
        // `value` ramps across [0, 1) so the filter `value > 0.5` keeps a known,
        // contiguous-ish subset that straddles every split boundary.
        const N: usize = 50_000;
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
        ]));
        let ids: Vec<i64> = (0..N as i64).collect();
        let values: Vec<f64> = (0..N).map(|i| (i as f64) / (N as f64)).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Float64Array::from(values.clone())),
            ],
        )
        .unwrap();

        // Write as a single row group (row_group_size == N).
        let props = parquet::file::properties::WriterProperties::builder()
            .set_dictionary_enabled(false)
            .set_max_row_group_row_count(Some(N))
            .build();
        let mut buf = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut buf, Arc::clone(&schema), Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
        storage.write("test/split.parquet", &buf).await.unwrap();

        // Expected result computed directly from the source data.
        let expected_count = values.iter().filter(|&&v| v > 0.5).count();
        let expected_sum: f64 = values.iter().copied().filter(|&v| v > 0.5).sum();

        let make_rg = || PlannedRowGroup {
            data_path: "test/split.parquet".into(),
            meta_path: "test/rg.meta".into(),
            meta: RowGroupMeta {
                row_group: "rg".into(),
                schema_id: 1,
                rows: N,
                columns: [
                    (
                        "id".into(),
                        ColumnStats {
                            min: None,
                            max: None,
                            nulls: 0,
                        },
                    ),
                    (
                        "value".into(),
                        ColumnStats {
                            min: None,
                            max: None,
                            nulls: 0,
                        },
                    ),
                ]
                .into(),
                column_offsets: None,
                sort: None,
                row_ids: vec![],
                checksum: String::new(),
                meta_checksum: String::new(),
            },
            partition_values: None,
            snapshot: 0,
            fallback: false,
            deletes: None,
            deleted_count: 0,
            fragment_id: 0,
            agg_state: None,
        };

        // Pushed filter: value > 0.5 (column index 1 in the output/table schema).
        let build_filter = |sch: &SchemaRef| {
            let value_expr = col("value", sch).unwrap();
            binary(value_expr, Operator::Gt, lit(0.5f64), sch).unwrap()
        };

        // Helper: run the scan over all partitions and fold COUNT / SUM(value).
        async fn run(exec: IcefallDBScanExec) -> (usize, f64) {
            let parts = exec.properties().partitioning.partition_count();
            let mut count = 0usize;
            let mut sum = 0.0f64;
            for p in 0..parts {
                let stream = exec.execute(p, Arc::new(TaskContext::default())).unwrap();
                let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
                for b in &batches {
                    count += b.num_rows();
                    let value_idx = b.schema().index_of("value").unwrap();
                    let col = b
                        .column(value_idx)
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                        .unwrap();
                    for v in col.iter().flatten() {
                        sum += v;
                    }
                }
            }
            (count, sum)
        }

        // Unsplit reference scan (target_partitions = 1).
        let unsplit = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&storage),
            Arc::clone(&schema),
            vec![make_rg()],
            None,
            vec![build_filter(&schema)],
            None,
            1024,
            0,
            2,
            1,
        )
        .unwrap();
        assert_eq!(
            unsplit.properties().partitioning.partition_count(),
            1,
            "reference scan must be a single partition"
        );
        let (unsplit_count, unsplit_sum) = run(unsplit).await;

        // Parallel scan: split the single row group into 8 partitions.
        let split = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&storage),
            Arc::clone(&schema),
            vec![make_rg()],
            None,
            vec![build_filter(&schema)],
            None,
            1024,
            0,
            2,
            8,
        )
        .unwrap();
        assert!(
            split.properties().partitioning.partition_count() > 1,
            "single large row group must split into multiple partitions for parallelism"
        );
        let (split_count, split_sum) = run(split).await;

        // 1. The split scan must match the brute-force expected result.
        assert_eq!(
            split_count, expected_count,
            "parallel split filtered COUNT must equal the brute-force count"
        );
        assert!(
            (split_sum - expected_sum).abs() < 1e-6,
            "parallel split filtered SUM ({split_sum}) must equal expected ({expected_sum})"
        );

        // 2. The split scan must be byte-equal to the unsplit reference scan.
        assert_eq!(
            split_count, unsplit_count,
            "split and unsplit COUNT must be identical (no dropped/double-counted rows)"
        );
        assert!(
            (split_sum - unsplit_sum).abs() < 1e-9,
            "split and unsplit SUM must be identical"
        );
    }

    #[tokio::test]
    async fn test_scan_whole_file_fallback() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let batch = make_test_batch();
        let full_schema = batch.schema();
        let bytes = write_batch(&batch);
        storage.write("test/rg.parquet", &bytes).await.unwrap();

        let rg = make_planned_row_group("test/rg.parquet", batch.num_rows(), None);
        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&full_schema),
            vec![rg],
            None,
            vec![],
            None,
            1024,
            0,
            2,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 5);
    }

    #[tokio::test]
    async fn test_scan_range_read_with_projection_and_filter() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let batch = make_test_batch();
        let full_schema = batch.schema();
        let bytes = write_batch(&batch);
        storage.write("test/rg.parquet", &bytes).await.unwrap();

        let offsets = compute_offsets(&bytes, &["id", "name"]).unwrap();
        let rg = make_planned_row_group("test/rg.parquet", batch.num_rows(), Some(offsets));

        // Project only `id`.
        let projection = Some(vec![0]);
        let output_schema = project_schema(&full_schema, projection.as_ref()).unwrap();
        let id_expr = col("id", &output_schema).unwrap();
        let filter = binary(
            Arc::clone(&id_expr),
            Operator::Gt,
            lit(2i32),
            &output_schema,
        )
        .unwrap();

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&full_schema),
            vec![rg],
            projection,
            vec![filter],
            None,
            1024,
            0,
            2,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);
        assert_eq!(batches[0].schema().fields().len(), 1);
        assert_eq!(batches[0].schema().field(0).name(), "id");
    }

    #[tokio::test]
    async fn test_scan_limit_pushdown() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let batch = make_test_batch();
        let full_schema = batch.schema();
        let bytes = write_batch(&batch);
        storage.write("test/rg.parquet", &bytes).await.unwrap();

        let offsets = compute_offsets(&bytes, &["id", "name"]).unwrap();
        let rg = make_planned_row_group("test/rg.parquet", batch.num_rows(), Some(offsets));

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&full_schema),
            vec![rg],
            None,
            vec![],
            Some(2),
            1024,
            0,
            2,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 2);
    }

    #[tokio::test]
    async fn test_scan_uses_parquet_metadata_cache() {
        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let batch = make_test_batch();
        let full_schema = batch.schema();
        let bytes = write_batch(&batch);
        storage.write("cache/rg.parquet", &bytes).await.unwrap();

        let checksum = format!("sha256:{}", hex::encode(sha2::Sha256::digest(&bytes)));
        let offsets = compute_offsets(&bytes, &["id", "name"]).unwrap();
        let rg = make_planned_row_group_with_checksum(
            "cache/rg.parquet",
            batch.num_rows(),
            Some(offsets.clone()),
            &checksum,
        );

        // Before scanning, the cache entry for this file should be absent.
        assert!(ParquetMetadataCache::global()
            .get("cache/rg.parquet", &checksum)
            .is_none());

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&full_schema),
            vec![rg.clone()],
            None,
            vec![],
            None,
            1024,
            0,
            2,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 5);

        // After scanning with sidecar offsets, the Parquet metadata should be
        // cached under the content-addressed checksum.
        assert!(ParquetMetadataCache::global()
            .get("cache/rg.parquet", &checksum)
            .is_some());

        // A second scan over the same file with the same checksum should be
        // served from the cache.
        let exec2 = IcefallDBScanExec::new(
            storage,
            full_schema,
            vec![rg],
            None,
            vec![],
            None,
            1024,
            0,
            2,
        )
        .unwrap();
        let stream2 = exec2.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches2: Vec<RecordBatch> = stream2.try_collect().await.unwrap();
        assert_eq!(batches2.len(), 1);
        assert_eq!(batches2[0].num_rows(), 5);
    }

    // --- deletion_to_row_selection unit tests ---

    #[test]
    fn test_deletion_to_row_selection_basic() {
        use icefalldb_core::DeletionVector;
        let mut dv = DeletionVector::default();
        dv.union_offsets([2u32, 5u32]);
        let sel = deletion_to_row_selection(&dv, 10);
        let selected: usize = sel.iter().filter(|s| !s.skip).map(|s| s.row_count).sum();
        let skipped: usize = sel.iter().filter(|s| s.skip).map(|s| s.row_count).sum();
        assert_eq!(selected, 8, "should select 8 of 10 rows");
        assert_eq!(skipped, 2, "should skip exactly 2 rows");

        // Assert the exact run-length structure: select(2), skip(1), select(2),
        // skip(1), select(4). A bug that emits [select(8), skip(2)] would satisfy
        // the aggregate counts above but skip the WRONG rows.
        let runs: Vec<(bool, usize)> = sel.iter().map(|s| (!s.skip, s.row_count)).collect();
        assert_eq!(
            runs,
            vec![
                (true, 2),  // rows 0–1: live
                (false, 1), // row 2:    deleted
                (true, 2),  // rows 3–4: live
                (false, 1), // row 5:    deleted
                (true, 4),  // rows 6–9: live
            ],
            "selector runs must identify the exact deleted offsets"
        );
    }

    #[test]
    fn test_deletion_to_row_selection_empty_dv() {
        use icefalldb_core::DeletionVector;
        let dv = DeletionVector::default();
        let sel = deletion_to_row_selection(&dv, 5);
        let selected: usize = sel.iter().filter(|s| !s.skip).map(|s| s.row_count).sum();
        let skipped: usize = sel.iter().filter(|s| s.skip).map(|s| s.row_count).sum();
        assert_eq!(selected, 5, "no deletions: all rows selected");
        assert_eq!(skipped, 0);
    }

    #[test]
    fn test_deletion_to_row_selection_all_deleted() {
        use icefalldb_core::DeletionVector;
        let mut dv = DeletionVector::default();
        dv.union_offsets(0u32..4u32);
        let sel = deletion_to_row_selection(&dv, 4);
        let selected: usize = sel.iter().filter(|s| !s.skip).map(|s| s.row_count).sum();
        assert_eq!(selected, 0, "all rows deleted: nothing selected");
    }

    // --- acceptance test: scan honours deletion vector ---

    #[tokio::test]
    async fn scan_skips_deleted_rows() {
        use icefalldb_core::metadata::ColumnStats;
        use icefalldb_core::DeletionVector;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        // Build a 10-row batch: id = [0..9]
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let ids: Vec<i32> = (0..10).collect();
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(Int32Array::from(ids))])
                .unwrap();

        // Write the data file and the deletion vector.
        let bytes = write_batch(&batch);
        storage
            .write("test/del_test.parquet", &bytes)
            .await
            .unwrap();

        let mut dv = DeletionVector::default();
        dv.union_offsets([2u32, 5u32]);
        let del_bytes = dv.serialize();
        storage
            .write("test/_deletions/rg0__v1.del", &del_bytes)
            .await
            .unwrap();

        // Build a PlannedRowGroup with the deletion vector reference.
        let rg = PlannedRowGroup {
            data_path: "test/del_test.parquet".into(),
            meta_path: "test/rg.meta".into(),
            meta: icefalldb_core::metadata::RowGroupMeta {
                row_group: "rg".into(),
                schema_id: 1,
                rows: 10,
                columns: [(
                    "id".into(),
                    ColumnStats {
                        min: None,
                        max: None,
                        nulls: 0,
                    },
                )]
                .into(),
                column_offsets: None,
                sort: None,
                row_ids: vec![],
                checksum: String::new(),
                meta_checksum: String::new(),
            },
            partition_values: None,
            snapshot: 0,
            fallback: false,
            deletes: Some("test/_deletions/rg0__v1.del".into()),
            deleted_count: 2,
            fragment_id: 0,
            agg_state: None,
        };

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&schema),
            vec![rg],
            None,
            vec![],
            None,
            1024,
            0,
            2,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 8, "should return 8 rows after deleting 2");

        // Verify that the deleted row values (id=2 and id=5) are absent.
        let all_ids: Vec<i32> = batches
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
        assert!(!all_ids.contains(&2), "id=2 should be deleted");
        assert!(!all_ids.contains(&5), "id=5 should be deleted");
    }

    // ── _rowid / _rowaddr pseudo-column tests ──────────────────────

    /// Helper: build a PlannedRowGroup with row_ids and an optional deletion
    /// vector path.  The Parquet data file is NOT written by this helper — the
    /// caller is expected to write it separately.
    /// Build a `PlannedRowGroup` with row_ids and an optional deletion vector.
    ///
    /// `deleted_count` must equal the DV's actual cardinality (i.e. the number
    /// of offsets the caller deleted).  Pass `dv.len()` or the result of
    /// `dv.cardinality()` — do NOT hard-code a constant.
    fn make_pseudo_col_rg(
        data_path: &str,
        rows: usize,
        row_ids: Vec<icefalldb_core::RowIdSegment>,
        fragment_id: u64,
        deletes: Option<String>,
        deleted_count: u64,
    ) -> PlannedRowGroup {
        PlannedRowGroup {
            data_path: data_path.into(),
            meta_path: "test/rg.meta".into(),
            meta: RowGroupMeta {
                row_group: "rg".into(),
                schema_id: 1,
                rows,
                columns: [(
                    "id".into(),
                    ColumnStats {
                        min: None,
                        max: None,
                        nulls: 0,
                    },
                )]
                .into(),
                column_offsets: None,
                sort: None,
                row_ids,
                checksum: String::new(),
                meta_checksum: String::new(),
            },
            partition_values: None,
            snapshot: 0,
            fallback: false,
            deletes,
            deleted_count,
            fragment_id,
            agg_state: None,
        }
    }

    /// `SELECT _rowid FROM t` where every row is live returns the allocated ids
    /// in physical order.
    #[tokio::test]
    async fn select_rowid_returns_allocated_ids_no_deletions() {
        use arrow::array::UInt64Array;
        use icefalldb_core::rowid::allocate_range;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        // Write a 5-row id column.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&parquet_schema),
            vec![Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50]))],
        )
        .unwrap();
        let bytes = write_batch(&batch);
        storage
            .write("test/pc_no_del.parquet", &bytes)
            .await
            .unwrap();

        // Allocate row IDs 100..104 (contiguous range starting at 100).
        let mut next_id = 100u64;
        let seg = allocate_range(&mut next_id, 5);
        let rg = make_pseudo_col_rg(
            "test/pc_no_del.parquet",
            5,
            vec![seg],
            7, // fragment_id
            None,
            0, // deleted_count: no deletions
        );

        // Table schema includes _rowid and _rowaddr as virtual columns.
        let table_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_rowid", DataType::UInt64, false),
            Field::new("_rowaddr", DataType::UInt64, false),
        ]));

        // Project only _rowid (index 1 in table_schema).
        let projection = Some(vec![1usize]);

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&table_schema),
            vec![rg],
            projection,
            vec![],
            None,
            1024,
            0,
            1,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let all_rowids: Vec<u64> = batches
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

        assert_eq!(
            all_rowids,
            vec![100u64, 101, 102, 103, 104],
            "_rowid must equal the allocated range in physical order"
        );
    }

    /// `SELECT _rowid FROM t` skips deleted rows, returning only the live ids.
    #[tokio::test]
    async fn select_rowid_returns_allocated_ids_skipping_deleted() {
        use arrow::array::UInt64Array;
        use icefalldb_core::rowid::allocate_range;
        use icefalldb_core::DeletionVector;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        // Write a 6-row data file (offsets 0..5).
        let parquet_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&parquet_schema),
            vec![Arc::new(Int32Array::from(vec![0, 1, 2, 3, 4, 5]))],
        )
        .unwrap();
        let bytes = write_batch(&batch);
        storage.write("test/pc_del.parquet", &bytes).await.unwrap();

        // Delete rows at physical offsets 1 and 4.
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32, 4u32]);
        storage
            .write("test/_del/pc.del", &dv.serialize())
            .await
            .unwrap();

        // Row IDs 200..205 for all 6 physical rows.
        let mut next_id = 200u64;
        let seg = allocate_range(&mut next_id, 6);

        let rg = make_pseudo_col_rg(
            "test/pc_del.parquet",
            6,
            vec![seg],
            3, // fragment_id
            Some("test/_del/pc.del".into()),
            dv.cardinality(), // deleted_count from the actual DV
        );

        let table_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_rowid", DataType::UInt64, false),
            Field::new("_rowaddr", DataType::UInt64, false),
        ]));

        // Project _rowid (index 1).
        let projection = Some(vec![1usize]);

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&table_schema),
            vec![rg],
            projection,
            vec![],
            None,
            1024,
            0,
            1,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let all_rowids: Vec<u64> = batches
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

        // Physical offsets 0,2,3,5 survive; row IDs are 200,202,203,205.
        assert_eq!(
            all_rowids,
            vec![200u64, 202, 203, 205],
            "_rowid must skip deleted physical offsets"
        );
    }

    /// `SELECT _rowaddr FROM t` returns packed (fragment_id << 32) | offset for
    /// surviving rows.
    #[tokio::test]
    async fn select_rowaddr_returns_packed_address() {
        use arrow::array::UInt64Array;
        use icefalldb_core::rowid::allocate_range;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

        // 4-row fragment with fragment_id = 5.
        let parquet_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            Arc::clone(&parquet_schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4]))],
        )
        .unwrap();
        let bytes = write_batch(&batch);
        storage.write("test/pc_addr.parquet", &bytes).await.unwrap();

        let mut next_id = 0u64;
        let seg = allocate_range(&mut next_id, 4);

        let rg = make_pseudo_col_rg(
            "test/pc_addr.parquet",
            4,
            vec![seg],
            5, // fragment_id
            None,
            0, // deleted_count: no deletions
        );

        let table_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("_rowid", DataType::UInt64, false),
            Field::new("_rowaddr", DataType::UInt64, false),
        ]));

        // Project _rowaddr (index 2).
        let projection = Some(vec![2usize]);

        let exec = IcefallDBScanExec::new(
            Arc::clone(&storage),
            Arc::clone(&table_schema),
            vec![rg],
            projection,
            vec![],
            None,
            1024,
            0,
            1,
        )
        .unwrap();

        let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches: Vec<RecordBatch> = stream.try_collect().await.unwrap();

        let all_addrs: Vec<u64> = batches
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

        let expected: Vec<u64> = (0u64..4).map(|off| (5u64 << 32) | off).collect();
        assert_eq!(
            all_addrs, expected,
            "_rowaddr must equal (fragment_id << 32) | physical_offset"
        );
    }

    /// REGRESSION (duplication-on-deletion through the intra-row-group split).
    ///
    /// When a single row group of N >= `MIN_ROWS_PER_PARTITION * 2` rows is split
    /// into parallel scan partitions AND it carries a deletion vector, the
    /// per-partition `RowSelection` is intersected with the (full-length)
    /// deletion-vector selection (`scan.rs`: `existing.intersection(&del_sel)`).
    ///
    /// `RowSelection::intersection` is only well-defined when both selections
    /// span the SAME number of rows: when one runs out, arrow-rs appends the
    /// longer selection's trailing selectors verbatim.  If a partition selection
    /// covers only `offset + chunk` rows (no trailing skip), it is SHORTER than
    /// the DV selection (which covers all N rows), so the intersection appends the
    /// DV's suffix — making partition `i` read a suffix `[i*chunk, N)`.  The union
    /// over partitions then yields a triangular ~`N_partitions * live / 2`
    /// duplicated offsets (e.g. 25k rows / 5k deleted -> ~170k rows instead of
    /// 20k), which inflates COUNT/SUM/GROUP BY and breaks compaction.
    ///
    /// The fix appends a trailing `skip(rows - offset - chunk)` so every
    /// partition selection spans the full N rows and the intersection is exact.
    ///
    /// This test drives the real production path (`split_row_groups` +
    /// `deletion_to_row_selection` + `intersection` + `live_physical_offsets`) and
    /// asserts that the concatenation of every partition's live physical offsets,
    /// sorted, equals EXACTLY the non-deleted offset set — no duplicates, no gaps,
    /// no overlaps.  It fails hard without the trailing-skip fix.
    #[test]
    fn split_row_group_with_deletions_yields_each_live_row_exactly_once() {
        use icefalldb_core::DeletionVector;
        use std::collections::BTreeSet;

        // N is comfortably above the split trigger threshold so the row group is
        // split (1 row group < desired partitions, and
        // total_rows >= MIN_ROWS_PER_PARTITION * 2).
        const N: usize = 25_000;
        const _: () = assert!(
            N >= MIN_ROWS_PER_PARTITION * 2,
            "N must exceed the split threshold"
        );

        let planned = make_planned_row_group("test/big.parquet", N, None);

        // Without filters the page index is never loaded, so splitting a single
        // row group makes every partition re-decode the WHOLE group. The gate must
        // keep an unfiltered scan at one partition per row group (no row_selection)
        // regardless of how large the group is or how many partitions are wanted.
        let unfiltered = split_row_groups(std::slice::from_ref(&planned), 16, false);
        assert_eq!(
            unfiltered.len(),
            1,
            "unfiltered scan of one row group must not split (got {})",
            unfiltered.len()
        );
        assert!(
            unfiltered[0].row_selection.is_none(),
            "unfiltered partition must carry no RowSelection"
        );

        let partitions = split_row_groups(std::slice::from_ref(&planned), 16, true);
        assert!(
            partitions.len() > 1,
            "a single {N}-row row group must split into multiple partitions \
             (got {} — split path not exercised)",
            partitions.len()
        );

        // Delete a known, scattered set of offsets that crosses partition
        // boundaries (the bug duplicates surviving rows from EVERY partition, so
        // any non-trivial deletion set surfaces it).
        let deleted: BTreeSet<usize> = (0..N).filter(|off| off % 5 == 0).collect();
        assert_eq!(deleted.len(), N / 5, "expected exactly N/5 deletions");
        let mut dv = DeletionVector::default();
        dv.union_offsets(deleted.iter().map(|&o| o as u32));
        let del_sel = deletion_to_row_selection(&dv, N);

        // Mirror the production scan: intersect each partition's split selection
        // with the deletion-vector selection, then expand to physical offsets.
        let mut all_live: Vec<usize> = Vec::new();
        for part in &partitions {
            let effective = match &part.row_selection {
                Some(existing) => existing.intersection(&del_sel),
                None => del_sel.clone(),
            };
            all_live.extend(live_physical_offsets(Some(&effective), N));
        }

        // The union must be EXACTLY the live set, once each.
        let expected_live: Vec<usize> = (0..N).filter(|off| !deleted.contains(off)).collect();
        let mut sorted_live = all_live.clone();
        sorted_live.sort_unstable();

        assert_eq!(
            all_live.len(),
            expected_live.len(),
            "split scan must read each live row exactly once: got {} offsets, \
             expected {} live rows ({}x duplication indicates the missing \
             trailing-skip bug)",
            all_live.len(),
            expected_live.len(),
            all_live.len() / expected_live.len().max(1)
        );
        assert_eq!(
            sorted_live, expected_live,
            "the union of all partitions' live offsets must equal the \
             non-deleted offset set exactly (no duplicates, gaps, or overlaps)"
        );

        // Belt-and-suspenders: explicit no-duplicate check (a duplicate with a
        // compensating gap could otherwise slip past the length check).
        let unique: BTreeSet<usize> = all_live.into_iter().collect();
        assert_eq!(
            unique.len(),
            expected_live.len(),
            "no physical row offset may appear in more than one partition"
        );
    }

    /// REGRESSION (metadata-only COUNT(*)/SUM over a split row group).
    ///
    /// When a single >=20k-row row group is split into N scan partitions, every
    /// partition holds a clone of the SAME physical row group.
    /// [`IcefallDBScanExec::planned_row_groups`] feeds the `MetadataAggregate`
    /// fast path, which computes COUNT(*) as `Σ meta.rows - deleted_count` over
    /// the returned row groups.  If the split clones are not collapsed, a
    /// deletion-bearing row group is counted N times, so COUNT(*) returns
    /// `N * (rows - deleted)` instead of `rows - deleted` (the 25k/5k case
    /// returned 160 000 = 8 * 20 000 instead of 20 000).
    ///
    /// `planned_row_groups()` must return exactly ONE entry per physical row
    /// group regardless of how many partitions it was split into.
    #[test]
    fn planned_row_groups_collapses_split_clones_of_one_row_group() {
        // A single deletion-bearing row group large enough to be split.
        const N: usize = 25_000;
        let mut planned = make_planned_row_group("test/big.parquet", N, None);
        planned.deletes = Some("test/_deletions/rg0__v1.del".into());
        planned.deleted_count = 5_000;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        // A non-empty filter is required for the intra-row-group split to fire
        // (unfiltered scans stay one-partition-per-row-group — they load no page
        // index, so a split would only duplicate decode). The literal predicate is
        // never evaluated here; only the split + collapse paths are exercised.
        let exec = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&storage),
            schema,
            vec![planned],
            None,
            vec![datafusion::physical_expr::expressions::lit(true)],
            None,
            1024,
            0,
            1,
            8, // force the single row group to split
        )
        .unwrap();

        assert!(
            exec.partitions.len() > 1,
            "the row group must actually be split (got {} partitions)",
            exec.partitions.len()
        );

        let rgs = exec.planned_row_groups();
        assert_eq!(
            rgs.len(),
            1,
            "split clones of one physical row group must collapse to a single \
             entry (got {} — metadata COUNT(*) would over-count {}x)",
            rgs.len(),
            rgs.len()
        );
        // The single returned row group carries the correct live count for the
        // metadata fast path.
        let live = rgs[0]
            .meta
            .rows
            .saturating_sub(rgs[0].deleted_count as usize);
        assert_eq!(
            live,
            N - 5_000,
            "the collapsed row group must report N-deleted live rows"
        );
    }

    /// Guard against over-eager collapsing: genuinely DISTINCT row groups must
    /// never be merged by `planned_row_groups`, even when they are not split
    /// (each is its own partition, `row_selection = None`).
    #[test]
    fn planned_row_groups_keeps_distinct_row_groups() {
        let rg_a = make_planned_row_group("test/a.parquet", 10, None);
        let rg_b = make_planned_row_group("test/b.parquet", 20, None);
        let rg_c = make_planned_row_group("test/c.parquet", 30, None);

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        // target_partitions <= row-group count: no splitting, all `row_selection`
        // are None and every distinct row group must survive.
        let exec = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&storage),
            schema,
            vec![rg_a, rg_b, rg_c],
            None,
            vec![],
            None,
            1024,
            0,
            1,
            1,
        )
        .unwrap();

        let rgs = exec.planned_row_groups();
        assert_eq!(rgs.len(), 3, "three distinct row groups must not be merged");
        let total: usize = rgs.iter().map(|rg| rg.meta.rows).sum();
        assert_eq!(total, 60, "distinct row group rows must all be preserved");
    }

    /// REGRESSION: `RowIdSelectionPushdown` must rewrite the scan even when a
    /// `RepartitionExec` sits between the `FilterExec(_rowid = ..)` and the
    /// `IcefallDBScanExec`. DataFusion inserts that repartition for any
    /// `target_partitions > 1` plan, so a rule that matched only a *direct*
    /// filter→scan child silently never fired in production (the `_rowid IN`
    /// read fell back to a full multi-fragment scan, ~30x slower).
    #[test]
    fn rowid_pushdown_fires_through_repartition_exec() {
        use crate::rules::RowIdSelectionPushdown;
        use datafusion::common::config::ConfigOptions;
        use datafusion::logical_expr::Operator;
        use datafusion::physical_expr::expressions::{binary, lit, Column};
        use datafusion::physical_optimizer::PhysicalOptimizerRule;
        use datafusion::physical_plan::filter::FilterExec;
        use datafusion::physical_plan::repartition::RepartitionExec;
        use datafusion::physical_plan::Partitioning;

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        // Schema carries the `_rowid` pseudo-column so the filter can reference it.
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(PSEUDO_COL_ROWID, DataType::UInt64, false),
        ]));
        let planned = make_planned_row_group("test/rg.parquet", 100, None);
        // Unfiltered scan over a single fragment: no split, no row selection.
        let scan = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&storage),
            Arc::clone(&schema),
            vec![planned],
            None,
            vec![],
            None,
            1024,
            0,
            1,
            4,
        )
        .unwrap();
        assert!(
            !scan.any_partition_has_row_selection(),
            "baseline unfiltered scan must carry no row selection"
        );

        // FilterExec(_rowid = 5) -> RepartitionExec(16) -> scan: the exact shape
        // DataFusion produces for `WHERE _rowid IN (..)` with target_partitions>1.
        let scan_arc: Arc<dyn ExecutionPlan> = Arc::new(scan);
        let repart: Arc<dyn ExecutionPlan> = Arc::new(
            RepartitionExec::try_new(scan_arc, Partitioning::RoundRobinBatch(16)).unwrap(),
        );
        let rowid_idx = schema.index_of(PSEUDO_COL_ROWID).unwrap();
        let pred = binary(
            Arc::new(Column::new(PSEUDO_COL_ROWID, rowid_idx)),
            Operator::Eq,
            lit(5u64),
            schema.as_ref(),
        )
        .unwrap();
        let filter: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(pred, repart).unwrap());

        let optimized = RowIdSelectionPushdown::new()
            .optimize(filter, &ConfigOptions::default())
            .unwrap();

        // Descend FilterExec -> RepartitionExec -> IcefallDBScanExec; the scan must
        // now carry a `_rowid` row selection (proving the rule fired through the
        // repartition).
        let repart_node = Arc::clone(optimized.children()[0]);
        let scan_node = Arc::clone(repart_node.children()[0]);
        let rewritten = (scan_node.as_ref() as &dyn std::any::Any)
            .downcast_ref::<IcefallDBScanExec>()
            .expect("node beneath the repartition must still be a IcefallDBScanExec");
        assert!(
            rewritten.any_partition_has_row_selection(),
            "rule must push the _rowid selection into the scan through the RepartitionExec"
        );
    }

    /// REGRESSION: a filtered scan over a single large row group that has a
    /// `column_offsets` sidecar (so reads are sparse and the page index is
    /// unavailable) must NOT intra-split. Splitting it would make every partition
    /// re-decode the whole group — the ~111s footgun on a 1M single-row-group
    /// filtered scan. Even with filters present and target_partitions=16, a
    /// sparse-read row group stays a single partition.
    #[test]
    fn no_intra_split_when_page_index_unavailable() {
        use std::collections::HashMap;

        const N: usize = 50_000; // > MIN_ROWS_PER_PARTITION * 2, so it *could* split
        let offsets: HashMap<String, ColumnChunkOffset> = HashMap::from([
            (
                "id".to_string(),
                ColumnChunkOffset {
                    offset: 4,
                    length: 100,
                },
            ),
            (
                "name".to_string(),
                ColumnChunkOffset {
                    offset: 104,
                    length: 100,
                },
            ),
        ]);
        let planned = make_planned_row_group("test/big.parquet", N, Some(offsets));

        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        // Non-empty filter: the old gate (`!filters.is_empty()`) would have split.
        let exec = IcefallDBScanExec::new_with_target_partitions(
            Arc::clone(&storage),
            schema,
            vec![planned],
            None,
            vec![datafusion::physical_expr::expressions::lit(true)],
            None,
            1024,
            0,
            1,
            16,
        )
        .unwrap();

        assert_eq!(
            exec.partitions.len(),
            1,
            "a sparse-read (column_offsets) row group must not intra-split (got {})",
            exec.partitions.len()
        );
        assert!(
            !exec.any_partition_has_row_selection(),
            "no split => no intra-group row selection"
        );
    }

    /// `row_id_at_offset` must index segments directly (no per-call
    /// materialization) and stay correct across `Range` + `Sorted` segments.
    #[test]
    fn row_id_at_offset_indexes_segments() {
        use icefalldb_core::RowIdSegment;
        let segs = vec![
            RowIdSegment::Range {
                start: 100,
                count: 5,
            },
            RowIdSegment::Sorted {
                ids: vec![200, 201, 305],
            },
        ];
        // Range segment: offsets 0..5 -> 100..105.
        assert_eq!(row_id_at_offset(&segs, 0), Some(100));
        assert_eq!(row_id_at_offset(&segs, 4), Some(104));
        // Sorted segment: offsets 5..8 -> 200, 201, 305.
        assert_eq!(row_id_at_offset(&segs, 5), Some(200));
        assert_eq!(row_id_at_offset(&segs, 6), Some(201));
        assert_eq!(row_id_at_offset(&segs, 7), Some(305));
        // Out of range / empty.
        assert_eq!(row_id_at_offset(&segs, 8), None);
        assert_eq!(row_id_at_offset(&[], 0), None);
    }
}
