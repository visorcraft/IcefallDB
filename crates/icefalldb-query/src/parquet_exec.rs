//! Build a DataFusion native `DataSourceExec` over IcefallDB Parquet files.
//!
//! When the table data is local and the query benefits from DataFusion's
//! highly-optimized Parquet reader (page-index pruning, vectorized filters,
//! morsel-based parallelism), this path can be significantly faster than the
//! custom `IcefallDBScanExec`.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::DFSchema;
use datafusion::datasource::physical_plan::ParquetSource;
use datafusion::execution::object_store::ObjectStoreUrl;
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::{Expr, Operator};
use datafusion::physical_expr::create_physical_expr;
use datafusion::physical_expr::expressions::BinaryExpr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource::FileRange;
use datafusion_datasource::PartitionedFile;
use icefalldb_core::storage::Storage;
use icefalldb_core::{PlannedRowGroup, ScanPlan};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use std::io::{Read, Seek, SeekFrom};

use crate::metadata_cache::ParquetMetadataCache;
use crate::stats::scan_plan_statistics;
use crate::Result;

/// Configuration controlling when the native Parquet reader is used.
#[derive(Debug, Clone, Copy)]
pub struct NativeParquetConfig {
    /// Minimum number of columns that must be referenced by the query before
    /// the native reader is preferred over the custom scan.
    pub min_filter_columns: usize,
}

impl Default for NativeParquetConfig {
    fn default() -> Self {
        Self {
            min_filter_columns: 1,
        }
    }
}

/// Return true if the native Parquet reader should be used for this scan.
///
/// The native reader is selected for local tables whose query references at
/// least `config.min_filter_columns` columns (wide-schema selective scans),
/// because it can exploit page-index pruning and row-group-granular parallelism.
///
/// It is NOT selected for a FILTERED scan whose surviving data is a single
/// Parquet row group that the native path cannot split: the native reader
/// parallelizes only at row-group granularity (`build_file_groups` only emits
/// multiple byte-range file groups for files with >1 row group), so such a scan
/// would run single-threaded.  The custom `IcefallDBScanExec` instead splits a
/// single row group into `target_partitions` parallel `RowSelection` tasks
/// (`split_row_groups`), which is measurably faster for these selective
/// single-row-group scans.  See `should_route_single_row_group_to_custom_scan`.
pub fn should_use_native_parquet(
    storage: &Arc<dyn Storage>,
    scan_plan: &ScanPlan,
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
    arrow_schema: &SchemaRef,
    target_partitions: usize,
    config: NativeParquetConfig,
) -> bool {
    // Native reader currently only supports local filesystem-backed storage.
    if storage.local_root().is_none() {
        return false;
    }

    let mut referenced = std::collections::HashSet::new();
    if let Some(proj) = projection {
        for idx in proj {
            referenced.insert(*idx);
        }
    }
    for filter in filters {
        collect_expr_columns(filter, arrow_schema, &mut referenced);
    }
    if referenced.len() < config.min_filter_columns {
        return false;
    }

    // Prefer the custom parallel scan when the native path would be stuck on a
    // single un-splittable scan task for a filtered scan (see fn docs).
    if should_route_single_row_group_to_custom_scan(scan_plan, filters, target_partitions) {
        return false;
    }

    true
}

/// Return true when a FILTERED scan over `scan_plan` would be a single
/// un-splittable scan task on the native Parquet path but can be parallelized by
/// the custom scan's intra-row-group `RowSelection` split.
///
/// The native `ParquetExec` parallelizes only across files and Parquet row
/// groups: a scan that prunes down to a SINGLE surviving file becomes one scan
/// task, and if that file holds a single Parquet row group `build_file_groups`
/// cannot split it by byte range either (it only emits multiple byte-range
/// groups for files with `>1` row group).  Such a scan runs single-threaded.
/// The custom `IcefallDBScanExec::split_row_groups` instead carves the single
/// row group into `target_partitions` parallel `RowSelection` tasks, letting the
/// selective Parquet decode run on all cores.  This is the `warm_filtered_scan`
/// shape: `WHERE category='cat_0'` partition-prunes the `events` table to one
/// ~1 M-row, single-row-group file plus a residual `value > 0.5` filter.
///
/// This only fires when:
///   * `target_partitions > 1` (there is parallelism to gain),
///   * the scan carries at least one pushable filter (an unfiltered single-file
///     scan pays no selective-decode cost — the native bulk decode is already
///     competitive — and folds to metadata for COUNT(*)),
///   * the scan prunes to exactly ONE surviving file (with several files the
///     native path already has file-level parallelism and its page-index path is
///     faster for the wide multi-column scans), AND
///   * that file holds a single Parquet row group (so the native path has
///     nothing to split) and is large enough that `split_row_groups` actually
///     produces multiple partitions (`MIN_ROWS_FOR_SPLIT`).
///
/// When a single file holds multiple Parquet row groups the native path can
/// already split it, so the custom scan is not preferred.  This function does
/// not read Parquet metadata: the single-physical-row-group-per-file property is
/// a writer invariant (one `ArrowWriter`/one batch per fragment file), and the
/// "exactly one surviving file" check is decided from the `ScanPlan` entries
/// (shared `data_path` + summed `meta.rows`).
fn should_route_single_row_group_to_custom_scan(
    scan_plan: &ScanPlan,
    filters: &[Expr],
    target_partitions: usize,
) -> bool {
    /// Minimum surviving rows before intra-row-group splitting is worthwhile.
    /// Mirrors `scan::MIN_ROWS_PER_PARTITION * 2`: below this, `split_row_groups`
    /// produces a single partition anyway, so routing to the custom scan would
    /// give up the native page-index path for no parallelism gain.
    const MIN_ROWS_FOR_SPLIT: usize = 20_000;

    if target_partitions <= 1 || filters.is_empty() {
        return false;
    }

    // Only the single-surviving-file shape is an un-splittable native task.
    // Multiple distinct files already give the native path file-level
    // parallelism, so keep the native page-index reader for those.
    let mut paths = scan_plan.row_groups.iter().map(|rg| rg.data_path.as_str());
    let Some(first_path) = paths.next() else {
        return false;
    };
    if paths.any(|p| p != first_path) {
        return false;
    }

    let total_rows: usize = scan_plan.row_groups.iter().map(|rg| rg.meta.rows).sum();
    total_rows >= MIN_ROWS_FOR_SPLIT
}

/// Decide whether the native Parquet row filter (the selective-decode path)
/// should be skipped in favour of bulk decode + a `FilterExec`.
///
/// Returns `Some(scan_projection)` — the ascending, deduped set of column
/// indices the scan must read (projection ∪ filter columns) — when every
/// projected column is also a filter column. In that case the row filter can
/// defer no other column's decode, so it is pure selective-decode overhead
/// (measured ~1.5× slower than bulk decode + a vectorized `FilterExec` on
/// uniform `wide_filter` data, where page-index pruning skips nothing).
///
/// Returns `None` (keep the row filter, the default) when there are non-filter
/// projected columns to late-materialize (e.g. `wide_agg`, whose aggregate
/// columns are decoded only for surviving rows), when there are no filters, or
/// when the projection is open (`None` = all columns).
///
/// Deliberately NOT selectivity-aware. A multi-column filter with a very
/// selective LEADING predicate (e.g. `id > 999990 AND a = 1 AND b = 2`) could in
/// theory still prefer the row filter — it would decode the leading column, reject
/// almost everything, and barely decode the rest — whereas this gate bulk-decodes
/// every filter column. We considered gating on an estimated selectivity from the
/// sidecar min/max stats and rejected it: those stats describe the value *range*,
/// not the *distribution*, so on skewed data a range like `> 999990` can look
/// 0.001%-selective while actually matching half the rows. Acting on that false
/// "selective" signal would switch back to the slower selective-decode path and
/// REGRESS exactly the uniform `wide_filter` shape that was fixed. With no reliable
/// plan-time estimate and no measured workload that regresses under the simple
/// rule, the gate stays projection-structural and is disableable per session via
/// `icefalldb.native_bulk_decode` for the rare skewed-selective case.
pub fn native_bulk_decode_projection(
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
    arrow_schema: &SchemaRef,
) -> Option<Vec<usize>> {
    if filters.is_empty() {
        return None;
    }
    // An open projection reads every column; the row filter still helps there.
    let proj = projection?;
    let mut filter_cols = std::collections::HashSet::new();
    for f in filters {
        collect_expr_columns(f, arrow_schema, &mut filter_cols);
    }
    if filter_cols.is_empty() {
        return None;
    }
    // Fire only when nothing can be late-materialized: every projected column is
    // already decoded for the filter anyway.
    if !proj.iter().all(|p| filter_cols.contains(p)) {
        return None;
    }
    // proj ⊆ filter_cols, so the scan reads exactly the filter columns.
    let mut scan_proj: std::collections::BTreeSet<usize> = filter_cols.into_iter().collect();
    scan_proj.extend(proj.iter().copied());
    Some(scan_proj.into_iter().collect())
}

/// Build a native DataFusion `DataSourceExec` over the Parquet files referenced
/// by `scan_plan`.
///
/// `enable_row_filter` controls whether the predicate is applied as a Parquet
/// `RowFilter` during decode (the selective-decode path). The predicate is
/// always supplied for statistics/page-index pruning regardless; when
/// `enable_row_filter` is `false` the caller must re-apply the filter with a
/// `FilterExec` over the bulk-decoded output. Disabling the row filter is faster
/// when nothing can be late-materialized (the projection is a subset of the
/// filter columns) because DataFusion's bulk decode avoids the selective-decode
/// overhead — see `should_disable_native_row_filter`.
#[allow(clippy::too_many_arguments)]
pub fn build_native_parquet_exec(
    storage: &Arc<dyn Storage>,
    scan_plan: &ScanPlan,
    arrow_schema: &SchemaRef,
    projection: Option<&Vec<usize>>,
    filters: &[Expr],
    limit: Option<usize>,
    batch_size: usize,
    target_partitions: usize,
    enable_row_filter: bool,
) -> Result<Arc<dyn ExecutionPlan>> {
    let root = storage.local_root().ok_or_else(|| {
        crate::QueryError::Other("native Parquet reader requires local storage".into())
    })?;
    let file_groups = build_file_groups(scan_plan, root, target_partitions)?;

    let predicate = if filters.is_empty() {
        None
    } else {
        let df_schema = DFSchema::try_from(arrow_schema.as_ref().clone())?;
        let props = ExecutionProps::new();
        let physical_filters: Vec<_> = filters
            .iter()
            .map(|f| create_physical_expr(f, &df_schema, &props))
            .collect::<datafusion::common::Result<Vec<_>>>()?;
        Some(combine_physical_filters(physical_filters))
    };

    let mut source = ParquetSource::new(Arc::clone(arrow_schema))
        .with_enable_page_index(true)
        .with_pushdown_filters(enable_row_filter)
        .with_reorder_filters(true)
        .with_bloom_filter_on_read(false);

    if let Some(pred) = predicate {
        source = source.with_predicate(pred);
    }

    let source: Arc<dyn FileSource> = Arc::new(source);

    let statistics = scan_plan_statistics(scan_plan, arrow_schema);

    let mut builder = FileScanConfigBuilder::new(ObjectStoreUrl::local_filesystem(), source)
        .with_file_groups(file_groups)
        .with_batch_size(Some(batch_size))
        .with_statistics(statistics);

    if let Some(proj) = projection {
        builder = builder
            .with_projection_indices(Some(proj.clone()))
            .map_err(crate::QueryError::DataFusion)?;
    }

    if let Some(limit) = limit {
        builder = builder.with_limit(Some(limit));
    }

    // target_partitions is consumed by build_file_groups to cap the number of
    // byte-range partitions created for large Parquet files.

    let config = builder.build();
    Ok(DataSourceExec::from_data_source(config))
}

/// Minimum bytes per native Parquet scan partition.
///
/// Files smaller than this are scanned with a single partition, avoiding the
/// fixed overhead of multi-partition scheduling for tiny row groups.  Larger
/// files are split along Parquet row-group boundaries so each row group can be
/// pruned independently.
const MIN_PARTITION_BYTES: u64 = 4 * 1024 * 1024;

fn build_file_groups(
    scan_plan: &ScanPlan,
    root: &std::path::Path,
    target_partitions: usize,
) -> Result<Vec<FileGroup>> {
    // Collect every row-group file with its size and decide whether it should
    // be split (multi-row-group files larger than the minimum partition size)
    // or treated as a small file that can be coalesced with other small files.
    struct Candidate {
        path: String,
        size: u64,
        ranges: Vec<(u64, u64)>,
    }

    let mut candidates = Vec::with_capacity(scan_plan.row_groups.len());
    for rg in &scan_plan.row_groups {
        let absolute = root.join(&rg.data_path);
        let size = std::fs::metadata(&absolute).map(|m| m.len()).map_err(|e| {
            crate::QueryError::Other(format!("metadata error for {}: {e}", absolute.display()))
        })?;
        let path = absolute.to_string_lossy().to_string();
        let ranges = row_group_ranges(rg, &absolute, size)?;
        candidates.push(Candidate { path, size, ranges });
    }

    let mut groups: Vec<FileGroup> = Vec::with_capacity(candidates.len());
    let mut small_files: Vec<PartitionedFile> = Vec::new();

    for cand in &candidates {
        // Only split files that actually contain multiple row groups and are
        // large enough to benefit from parallel range scans.
        if cand.ranges.len() > 1 && cand.size >= MIN_PARTITION_BYTES {
            for (start, end) in &cand.ranges {
                let mut file = PartitionedFile::new(cand.path.clone(), cand.size);
                file.range = Some(FileRange {
                    start: *start as i64,
                    end: *end as i64,
                });
                groups.push(FileGroup::new(vec![file]));
            }
        } else {
            small_files.push(PartitionedFile::new(cand.path.clone(), cand.size));
        }
    }

    if !small_files.is_empty() {
        // Coalesce small files into balanced groups.  We want enough groups to
        // keep all cores busy without creating one task per tiny file.  Using
        // several groups per requested partition lets DataFusion's thread pool
        // absorb scheduling jitter while still parallelizing large tables.
        let num_groups = (target_partitions.saturating_mul(4)).clamp(1, small_files.len());
        let mut grouped: Vec<Vec<PartitionedFile>> = vec![Vec::new(); num_groups];
        let mut group_sizes: Vec<u64> = vec![0; num_groups];

        // Sort descending by size and best-fit into the currently smallest
        // group so partitions stay balanced even when file sizes vary.
        small_files.sort_by_key(|b| std::cmp::Reverse(b.object_meta.size));
        for file in small_files {
            let idx = group_sizes
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| *s)
                .map(|(i, _)| i)
                .unwrap_or(0);
            group_sizes[idx] += file.object_meta.size;
            grouped[idx].push(file);
        }
        for g in grouped {
            if !g.is_empty() {
                groups.push(FileGroup::new(g));
            }
        }
    }

    Ok(groups)
}

/// Return the byte ranges of each Parquet row group in `path`.
///
/// The checksum from `rg.meta.checksum` is used to lookup decoded metadata in
/// the process-wide [`ParquetMetadataCache`], avoiding repeated footer decoding
/// across scans.
fn row_group_ranges(
    rg: &PlannedRowGroup,
    path: &std::path::Path,
    file_size: u64,
) -> Result<Vec<(u64, u64)>> {
    let metadata = load_parquet_metadata(rg, path)?;

    let mut ranges = Vec::with_capacity(metadata.num_row_groups());
    for rg_meta in metadata.row_groups() {
        let mut start = file_size;
        let mut end = 0u64;
        for col in rg_meta.columns() {
            let col_start = col
                .dictionary_page_offset()
                .map(|o| o as u64)
                .unwrap_or_else(|| col.data_page_offset() as u64);
            let col_end = col_start + col.compressed_size() as u64;
            start = start.min(col_start);
            end = end.max(col_end);
        }
        ranges.push((start, end));
    }
    Ok(ranges)
}

/// Size of the Parquet footer tail: 4-byte metadata length + 4-byte magic.
const PARQUET_FOOTER_SIZE: u64 = 8;

/// Load metadata for a row group, using the process-wide cache on hit and
/// falling back to reading the footer with [`ParquetMetaDataReader`] on miss.
fn load_parquet_metadata(
    rg: &PlannedRowGroup,
    path: &std::path::Path,
) -> Result<Arc<ParquetMetaData>> {
    if let Some(cached) = ParquetMetadataCache::global().get(&rg.data_path, &rg.meta.checksum) {
        return Ok(cached);
    }

    let mut file = std::fs::File::open(path)
        .map_err(|e| crate::QueryError::Other(format!("open {}: {e}", path.display())))?;

    // Read the footer tail to discover the metadata length.
    let mut footer_tail = [0u8; PARQUET_FOOTER_SIZE as usize];
    file.seek(SeekFrom::End(-(PARQUET_FOOTER_SIZE as i64)))
        .map_err(|e| crate::QueryError::Other(format!("seek {}: {e}", path.display())))?;
    file.read_exact(&mut footer_tail)
        .map_err(|e| crate::QueryError::Other(format!("read footer {}: {e}", path.display())))?;
    let metadata_len = u32::from_le_bytes(footer_tail[..4].try_into().unwrap()) as u64;

    // Read the thrift-encoded metadata bytes.
    let mut metadata_bytes = vec![0u8; metadata_len as usize];
    file.seek(SeekFrom::End(
        -((PARQUET_FOOTER_SIZE + metadata_len) as i64),
    ))
    .map_err(|e| crate::QueryError::Other(format!("seek metadata {}: {e}", path.display())))?;
    file.read_exact(&mut metadata_bytes)
        .map_err(|e| crate::QueryError::Other(format!("read metadata {}: {e}", path.display())))?;

    let metadata = Arc::new(
        ParquetMetaDataReader::decode_metadata(&metadata_bytes).map_err(|e| {
            crate::QueryError::Other(format!("decode metadata {}: {e}", path.display()))
        })?,
    );
    ParquetMetadataCache::global().put(
        &rg.data_path,
        &rg.meta.checksum,
        Arc::clone(&metadata),
        metadata_len,
    );
    Ok(metadata)
}

fn combine_physical_filters(
    filters: Vec<Arc<dyn datafusion::physical_plan::PhysicalExpr>>,
) -> Arc<dyn datafusion::physical_plan::PhysicalExpr> {
    filters
        .into_iter()
        .reduce(|acc, f| {
            Arc::new(BinaryExpr::new(acc, Operator::And, f))
                as Arc<dyn datafusion::physical_plan::PhysicalExpr>
        })
        .expect("filters is non-empty")
}

fn collect_expr_columns(
    expr: &Expr,
    schema: &SchemaRef,
    out: &mut std::collections::HashSet<usize>,
) {
    for col in expr.column_refs() {
        if let Ok(idx) = schema.index_of(col.name()) {
            out.insert(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::logical_expr::{col, lit};
    use icefalldb_core::metadata::{RowGroupMeta, Schema as IcefallDBSchema};

    /// Build a `ScanPlan` whose row groups have the given `(data_path, rows)`.
    fn make_scan_plan(row_groups: &[(&str, usize)]) -> ScanPlan {
        let planned = row_groups
            .iter()
            .map(|(path, rows)| PlannedRowGroup {
                data_path: (*path).to_string(),
                meta: RowGroupMeta {
                    rows: *rows,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();
        ScanPlan {
            table: "t".into(),
            schema: IcefallDBSchema {
                schema_id: 1,
                columns: vec![],
                partition_by: None,
                sort: None,
                agg_group_keys: None,
                row_group_target_rows: 1_000_000,
                row_group_target_bytes: 1 << 27,
                dropped_columns: vec![],
                max_field_id: 0,
            },
            row_groups: planned,
        }
    }

    fn value_filter() -> Vec<Expr> {
        vec![col("value").gt(lit(0.5f64))]
    }

    #[test]
    fn single_large_row_group_filtered_routes_to_custom_scan() {
        // One file, one row group, large: native cannot split → custom scan.
        let plan = make_scan_plan(&[("events/cat_0.parquet", 1_000_000)]);
        assert!(should_route_single_row_group_to_custom_scan(
            &plan,
            &value_filter(),
            16
        ));
    }

    #[test]
    fn multiple_files_stay_native() {
        // Several distinct files give the native path file-level parallelism and
        // its page-index reader is faster for wide multi-column scans, so the
        // routing predicate must NOT divert these to the custom scan. This is the
        // `events_wide` / `clustered_wide_filter` shape (dozens of files).
        let plan = make_scan_plan(&[
            ("events_wide/a.parquet", 600_000),
            ("events_wide/b.parquet", 600_000),
            ("events_wide/c.parquet", 600_000),
        ]);
        assert!(!should_route_single_row_group_to_custom_scan(
            &plan,
            &value_filter(),
            16
        ));
    }

    #[test]
    fn single_file_split_into_planned_row_groups_routes_to_custom_scan() {
        // A single physical file can back several planned row groups (same
        // data_path). The native path still scans it as one task per row group,
        // so the single-file selective shape routes to the custom parallel scan.
        let plan = make_scan_plan(&[
            ("events/cat_0.parquet", 500_000),
            ("events/cat_0.parquet", 500_000),
        ]);
        assert!(should_route_single_row_group_to_custom_scan(
            &plan,
            &value_filter(),
            16
        ));
    }

    #[test]
    fn unfiltered_scan_stays_native() {
        let plan = make_scan_plan(&[("events/cat_0.parquet", 1_000_000)]);
        assert!(!should_route_single_row_group_to_custom_scan(
            &plan,
            &[],
            16
        ));
    }

    #[test]
    fn single_partition_target_stays_native() {
        let plan = make_scan_plan(&[("events/cat_0.parquet", 1_000_000)]);
        assert!(!should_route_single_row_group_to_custom_scan(
            &plan,
            &value_filter(),
            1
        ));
    }

    #[test]
    fn tiny_single_row_group_stays_native() {
        // Below MIN_ROWS_FOR_SPLIT the custom scan would not split anyway, so
        // keep the native page-index path.
        let plan = make_scan_plan(&[("events/tiny.parquet", 5_000)]);
        assert!(!should_route_single_row_group_to_custom_scan(
            &plan,
            &value_filter(),
            16
        ));
    }

    fn wide_schema() -> SchemaRef {
        use arrow::datatypes::{DataType, Field, Schema};
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Int64, false),
            Field::new("d", DataType::Int64, false),
        ]))
    }

    fn ab_filters() -> Vec<Expr> {
        vec![col("a").gt(lit(1i64)), col("b").lt(lit(2i64))]
    }

    #[test]
    fn bulk_decode_fires_when_projection_subset_of_filters() {
        let s = wide_schema();
        // COUNT(*): empty projection ⊆ {a, b} → scan the filter columns.
        assert_eq!(
            native_bulk_decode_projection(Some(&vec![]), &ab_filters(), &s),
            Some(vec![0, 1])
        );
        // Projecting a filter column also fires; result is the deduped union.
        assert_eq!(
            native_bulk_decode_projection(Some(&vec![0]), &ab_filters(), &s),
            Some(vec![0, 1])
        );
        assert_eq!(
            native_bulk_decode_projection(Some(&vec![1, 0]), &ab_filters(), &s),
            Some(vec![0, 1])
        );
    }

    #[test]
    fn bulk_decode_skipped_when_non_filter_column_projected() {
        let s = wide_schema();
        // `c` is not a filter column → late-materialization helps → keep the row filter.
        assert_eq!(
            native_bulk_decode_projection(Some(&vec![0, 2]), &ab_filters(), &s),
            None
        );
    }

    #[test]
    fn bulk_decode_skipped_without_filters_or_open_projection() {
        let s = wide_schema();
        assert_eq!(native_bulk_decode_projection(Some(&vec![]), &[], &s), None);
        // Open projection (None) reads every column → keep the row filter.
        assert_eq!(native_bulk_decode_projection(None, &ab_filters(), &s), None);
    }
}
