use crate::agg_cache::{
    deleted_contribution, deserialize_agg_state, retract, retract_grouped, AggStateCache,
    FragmentAggState, GroupedPartials,
};
use crate::catalog::Catalog;
use crate::deletion::DeletionVector;
use crate::meta_cache::MetaCache;
use crate::metadata::{RowGroupMeta, Schema, SnapshotCheckpoint};
use crate::storage::Storage;
use crate::{IcefallDBError, Result};
use arrow::array::RecordBatch;
use bytes::Bytes;
use futures::stream::{BoxStream, Stream, StreamExt};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::task::{Context, Poll};

/// A literal value used in scan predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int64(i64),
    Float64(f64),
    String(String),
    Bool(bool),
}

impl From<i64> for Literal {
    fn from(value: i64) -> Self {
        Literal::Int64(value)
    }
}

impl From<f64> for Literal {
    fn from(value: f64) -> Self {
        Literal::Float64(value)
    }
}

impl From<&str> for Literal {
    fn from(value: &str) -> Self {
        Literal::String(value.to_string())
    }
}

impl From<String> for Literal {
    fn from(value: String) -> Self {
        Literal::String(value)
    }
}

impl From<bool> for Literal {
    fn from(value: bool) -> Self {
        Literal::Bool(value)
    }
}

impl Literal {
    /// Convert the literal to a JSON value for comparison with stored statistics.
    fn to_value(&self) -> Value {
        match self {
            Literal::Int64(v) => Value::from(*v),
            Literal::Float64(v) => Value::from(*v),
            Literal::String(v) => Value::from(v.clone()),
            Literal::Bool(v) => Value::from(*v),
        }
    }
}

/// A predicate applied to a scan plan for partition or statistical pruning.
#[derive(Debug, Clone, PartialEq)]
pub enum Predicate {
    Eq {
        column: String,
        value: Literal,
    },
    Lt {
        column: String,
        value: Literal,
    },
    Lte {
        column: String,
        value: Literal,
    },
    Gt {
        column: String,
        value: Literal,
    },
    Gte {
        column: String,
        value: Literal,
    },
    InList {
        column: String,
        values: Vec<Literal>,
    },
    IsNull {
        column: String,
    },
    IsNotNull {
        column: String,
    },
}

/// A row group selected by a scan, with its data path, metadata path, and
/// decoded metadata.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlannedRowGroup {
    pub data_path: String,
    pub meta_path: String,
    pub meta: RowGroupMeta,
    pub partition_values: Option<HashMap<String, Value>>,
    /// Manifest sequence (snapshot ID) that selected this row group.
    pub snapshot: u64,
    /// If true, the sidecar metadata was missing or corrupt and `meta` only
    /// contains a placeholder. The DataFusion provider should fall back to
    /// reading the Parquet footer for statistics.
    pub fallback: bool,
    /// Absolute path to the deletion vector file for this row group, if any.
    /// Derived from `RowGroupEntry.deletes` by prefixing the table directory.
    pub deletes: Option<String>,
    /// Number of logically deleted rows in this row group.
    pub deleted_count: u64,
    /// Stable fragment identifier from the manifest entry. Used to compute
    /// `_rowaddr` pseudo-column values: `(fragment_id << 32) | physical_offset`.
    pub fragment_id: u64,
    /// Per-fragment additive aggregate partials loaded from the `.agg` sidecar,
    /// if one exists and is valid. `None` when the fragment has no `.agg` file
    /// (e.g. UPDATE/MERGE/compaction fragments) or when the sidecar could not be
    /// read. The metadata-aggregate rule falls back for any fragment where this is `None`.
    // Loaded on every scan of a clean fragment and cached after first touch.
    pub agg_state: Option<std::sync::Arc<FragmentAggState>>,
}

/// A scan plan describing the row groups that make up a table snapshot.
#[derive(Debug, Clone)]
pub struct ScanPlan {
    pub table: String,
    pub schema: Schema,
    pub row_groups: Vec<PlannedRowGroup>,
}

impl ScanPlan {
    /// Apply a set of predicates to produce a pruned copy of this plan.
    ///
    /// Row groups are dropped only when the available metadata proves that no
    /// row in the group can satisfy the predicate. If a column is not present
    /// in the row group metadata, the row group is kept.
    ///
    /// Returns [`IcefallDBError::TypeNotSupported`] when a predicate literal cannot
    /// be compared to the stored min/max statistics for its column.
    pub fn prune(&self, predicates: &[Predicate]) -> Result<ScanPlan> {
        let partition_by: Vec<&str> = self
            .schema
            .partition_by
            .as_deref()
            .map(|cols| cols.iter().map(String::as_str).collect())
            .unwrap_or_default();

        let mut row_groups = Vec::with_capacity(self.row_groups.len());
        for rg in &self.row_groups {
            let mut keep = true;
            for pred in predicates {
                if !row_group_satisfies(pred, rg, &partition_by)? {
                    keep = false;
                    break;
                }
            }
            if keep {
                row_groups.push(rg.clone());
            }
        }

        Ok(ScanPlan {
            table: self.table.clone(),
            schema: self.schema.clone(),
            row_groups,
        })
    }

    /// Keep only row groups whose id (filename stem) is in `keep`.
    pub fn retain_row_groups(&mut self, keep: &HashSet<String>) {
        self.row_groups.retain(|rg| {
            let rg_id = rg
                .data_path
                .strip_suffix(".parquet")
                .and_then(|p| std::path::Path::new(p).file_name())
                .and_then(|s| s.to_str())
                .unwrap_or("");
            keep.contains(rg_id)
        });
    }

    /// Keep only row groups that contain at least one row ID from `row_ids`.
    ///
    /// A row group "contains" a row ID when at least one of its
    /// `meta.row_ids` segments covers that ID.  Row groups with empty
    /// `row_ids` (legacy fragments without allocated IDs) are always retained
    /// so they are not incorrectly pruned.
    pub fn retain_row_groups_by_row_ids(&mut self, row_ids: &HashSet<u64>) {
        // Sort the matching IDs once so each row-group segment can be tested for
        // overlap in O(log N) via binary search. The previous implementation
        // expanded every `Range { count }` segment into `count` individual
        // `HashSet::contains` probes with no short-circuit on misses, which made
        // a 10M-row table pay ~9M hash lookups on the indexed-equality path
        // (queries grew ~8x slower than the unindexed scan).
        let mut sorted: Vec<u64> = row_ids.iter().copied().collect();
        sorted.sort_unstable();
        let overlaps = |start: u64, end_exclusive: u64| -> bool {
            // Index of the first matching id >= start; it overlaps the segment
            // iff that id also falls below the segment's end.
            let i = sorted.partition_point(|&x| x < start);
            i < sorted.len() && sorted[i] < end_exclusive
        };
        self.row_groups.retain(|rg| {
            // Legacy fragment: no row IDs allocated yet — do not prune.
            if rg.meta.row_ids.is_empty() {
                return true;
            }
            // Keep the row group if any of its row-ID segments overlaps the
            // matching set.
            rg.meta.row_ids.iter().any(|seg| match seg {
                crate::rowid::RowIdSegment::Range { start, count } => {
                    overlaps(*start, start + count)
                }
                crate::rowid::RowIdSegment::Sorted { ids } => {
                    ids.iter().any(|id| sorted.binary_search(id).is_ok())
                }
            })
        });
    }
}

/// Returns `true` if the row group may contain rows matching the predicate.
fn row_group_satisfies(
    pred: &Predicate,
    rg: &PlannedRowGroup,
    partition_by: &[&str],
) -> Result<bool> {
    match pred {
        Predicate::Eq { column, value } => {
            // Partition pruning: an equality predicate on a partition column can
            // drop row groups whose partition value is known and different.
            if partition_by.contains(&column.as_str()) {
                if let Some(values) = &rg.partition_values {
                    if let Some(partition_value) = values.get(column) {
                        if partition_value != &value.to_value() {
                            return Ok(false);
                        }
                    }
                }
                return Ok(true);
            }

            // Stats pruning for non-partition columns: drop the row group if the
            // literal lies outside the recorded [min, max] interval.
            let Some(stats) = rg.meta.columns.get(column) else {
                return Ok(true);
            };
            if let (Some(min), Some(max)) = (&stats.min, &stats.max) {
                let v = value.to_value();
                if value_lt(&v, min)? || value_gt(&v, max)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Predicate::Lt { column, value } => {
            Ok(!stats_outside_range(rg, column, value, |min, _max, v| {
                // Drop if every value >= v, i.e. min >= v.
                value_ge(min, v)
            })?)
        }
        Predicate::Lte { column, value } => {
            Ok(!stats_outside_range(rg, column, value, |min, _max, v| {
                // Drop if every value > v, i.e. min > v.
                value_gt(min, v)
            })?)
        }
        Predicate::Gt { column, value } => {
            Ok(!stats_outside_range(rg, column, value, |_min, max, v| {
                // Drop if every value <= v, i.e. max <= v.
                value_le(max, v)
            })?)
        }
        Predicate::Gte { column, value } => {
            Ok(!stats_outside_range(rg, column, value, |_min, max, v| {
                // Drop if every value < v, i.e. max < v.
                value_lt(max, v)
            })?)
        }
        Predicate::IsNull { column } => {
            if let Some(stats) = rg.meta.columns.get(column) {
                if stats.nulls == 0 {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Predicate::IsNotNull { column } => {
            if let Some(stats) = rg.meta.columns.get(column) {
                if stats.nulls == rg.meta.rows {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Predicate::InList { column, values } => {
            // Partition pruning: drop the row group if its single partition value
            // is known and is not contained in the list.
            if partition_by.contains(&column.as_str()) {
                if let Some(values_map) = &rg.partition_values {
                    if let Some(partition_value) = values_map.get(column) {
                        let value_set: std::collections::HashSet<Value> =
                            values.iter().map(|v| v.to_value()).collect();
                        return Ok(value_set.contains(partition_value));
                    }
                }
                return Ok(true);
            }

            // Stats pruning for non-partition columns: keep the row group if any
            // list value falls within the recorded [min, max] interval.
            let Some(stats) = rg.meta.columns.get(column) else {
                return Ok(true);
            };
            if let (Some(min), Some(max)) = (&stats.min, &stats.max) {
                for value in values {
                    let v = value.to_value();
                    if !value_lt(&v, min)? && !value_gt(&v, max)? {
                        return Ok(true);
                    }
                }
                return Ok(false);
            }
            Ok(true)
        }
    }
}

/// Invokes `check` when both min and max statistics are present. If `check`
/// returns `true`, the row group cannot contain any matching rows.
fn stats_outside_range<F>(
    rg: &PlannedRowGroup,
    column: &str,
    value: &Literal,
    check: F,
) -> Result<bool>
where
    F: FnOnce(&Value, &Value, &Value) -> Result<bool>,
{
    let Some(stats) = rg.meta.columns.get(column) else {
        return Ok(false);
    };
    match (&stats.min, &stats.max) {
        (Some(min), Some(max)) => check(min, max, &value.to_value()),
        _ => Ok(false),
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn compare_values(a: &Value, b: &Value) -> Result<Option<std::cmp::Ordering>> {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            if let (Some(ia), Some(ib)) = (na.as_i64(), nb.as_i64()) {
                Ok(ia.partial_cmp(&ib))
            } else {
                Ok(na
                    .as_f64()
                    .and_then(|fa| nb.as_f64().and_then(|fb| fa.partial_cmp(&fb))))
            }
        }
        (Value::String(sa), Value::String(sb)) => Ok(sa.partial_cmp(sb)),
        (Value::Bool(ba), Value::Bool(bb)) => Ok(ba.partial_cmp(bb)),
        _ => Err(IcefallDBError::TypeNotSupported(format!(
            "cannot compare {} literal with {} statistics",
            json_type_name(a),
            json_type_name(b)
        ))),
    }
}

fn value_lt(a: &Value, b: &Value) -> Result<bool> {
    Ok(compare_values(a, b)?
        .map(|o| o == std::cmp::Ordering::Less)
        .unwrap_or(false))
}

fn value_le(a: &Value, b: &Value) -> Result<bool> {
    Ok(compare_values(a, b)?
        .map(|o| o != std::cmp::Ordering::Greater)
        .unwrap_or(false))
}

fn value_gt(a: &Value, b: &Value) -> Result<bool> {
    Ok(compare_values(a, b)?
        .map(|o| o == std::cmp::Ordering::Greater)
        .unwrap_or(false))
}

fn value_ge(a: &Value, b: &Value) -> Result<bool> {
    Ok(compare_values(a, b)?
        .map(|o| o != std::cmp::Ordering::Less)
        .unwrap_or(false))
}

/// Sound per-fragment coverage classification of a range/equality predicate set
/// against a single column's sidecar `[min, max]` statistics.
///
/// Used by the partial-aggregate-pushdown optimizer rule to decide
/// which fragments are FULLY COVERED by a filter `F` (so their cached aggregate
/// partials can be composed without reading Parquet) and which are DISJOINT
/// (contribute nothing).  Soundness is the absolute requirement: a fragment is
/// reported COVERED only when EVERY value in `[min, max]` provably satisfies the
/// predicate.  Null handling is the caller's responsibility (a fragment with
/// nulls in the filter column is never fully covered).
pub mod predicate_eval {
    use super::{compare_values, Predicate};
    use crate::metadata::ColumnStats;
    use std::cmp::Ordering;

    /// Compare the JSON literal `a` against `b`, returning `None` when the two
    /// are type-incompatible (so the caller can conservatively treat the
    /// fragment as BOUNDARY rather than risk an unsound decision).
    fn cmp(a: &serde_json::Value, b: &serde_json::Value) -> Option<Ordering> {
        compare_values(a, b).ok().flatten()
    }

    /// Returns `true` when EVERY non-null value in `[min, max]` satisfies `pred`.
    ///
    /// Conservative: any missing min/max bound, or a type-incompatible
    /// comparison, yields `false` (not covered).  The predicate's column name is
    /// assumed to match the column `stats` describe (the caller enforces this).
    fn predicate_covers(pred: &Predicate, stats: &ColumnStats) -> bool {
        let (Some(min), Some(max)) = (&stats.min, &stats.max) else {
            return false;
        };
        match pred {
            // [min,max] ⊆ {v}  ⇔  min == v == max.
            Predicate::Eq { value, .. } => {
                let v = value.to_value();
                matches!(cmp(min, &v), Some(Ordering::Equal))
                    && matches!(cmp(max, &v), Some(Ordering::Equal))
            }
            // col < v covers everything iff max < v.
            Predicate::Lt { value, .. } => {
                matches!(cmp(max, &value.to_value()), Some(Ordering::Less))
            }
            // col <= v covers everything iff max <= v.
            Predicate::Lte { value, .. } => {
                matches!(
                    cmp(max, &value.to_value()),
                    Some(Ordering::Less | Ordering::Equal)
                )
            }
            // col > v covers everything iff min > v.
            Predicate::Gt { value, .. } => {
                matches!(cmp(min, &value.to_value()), Some(Ordering::Greater))
            }
            // col >= v covers everything iff min >= v.
            Predicate::Gte { value, .. } => {
                matches!(
                    cmp(min, &value.to_value()),
                    Some(Ordering::Greater | Ordering::Equal)
                )
            }
            // IN / IS NULL / IS NOT NULL are not used by the rule; never covered.
            _ => false,
        }
    }

    /// Returns `true` when NO non-null value in `[min, max]` satisfies `pred`
    /// (the fragment is provably disjoint from the predicate's accepted range).
    ///
    /// Conservative: missing bounds or incompatible comparisons yield `false`
    /// (cannot prove disjointness → treat as overlapping).
    fn predicate_disjoint(pred: &Predicate, stats: &ColumnStats) -> bool {
        let (Some(min), Some(max)) = (&stats.min, &stats.max) else {
            return false;
        };
        match pred {
            // Disjoint iff v < min or v > max.
            Predicate::Eq { value, .. } => {
                let v = value.to_value();
                matches!(cmp(&v, min), Some(Ordering::Less))
                    || matches!(cmp(&v, max), Some(Ordering::Greater))
            }
            // col < v disjoint iff min >= v.
            Predicate::Lt { value, .. } => {
                matches!(
                    cmp(min, &value.to_value()),
                    Some(Ordering::Greater | Ordering::Equal)
                )
            }
            // col <= v disjoint iff min > v.
            Predicate::Lte { value, .. } => {
                matches!(cmp(min, &value.to_value()), Some(Ordering::Greater))
            }
            // col > v disjoint iff max <= v.
            Predicate::Gt { value, .. } => {
                matches!(
                    cmp(max, &value.to_value()),
                    Some(Ordering::Less | Ordering::Equal)
                )
            }
            // col >= v disjoint iff max < v.
            Predicate::Gte { value, .. } => {
                matches!(cmp(max, &value.to_value()), Some(Ordering::Less))
            }
            _ => false,
        }
    }

    /// Returns `true` when EVERY non-null value in the fragment's `[min, max]`
    /// satisfies ALL `predicates` (a conjunction).  An empty predicate set is
    /// never treated as covering (the caller always passes ≥1 predicate).
    pub fn fragment_fully_covered(predicates: &[Predicate], stats: &ColumnStats) -> bool {
        !predicates.is_empty() && predicates.iter().all(|p| predicate_covers(p, stats))
    }

    /// Returns `true` when the fragment is provably disjoint from the
    /// conjunction (ANY single predicate excludes the whole `[min, max]`).
    pub fn fragment_disjoint(predicates: &[Predicate], stats: &ColumnStats) -> bool {
        predicates.iter().any(|p| predicate_disjoint(p, stats))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::reader::Literal;
        use serde_json::Value;

        fn stats(min: i64, max: i64, nulls: usize) -> ColumnStats {
            ColumnStats {
                min: Some(Value::from(min)),
                max: Some(Value::from(max)),
                nulls,
            }
        }

        fn between(lo: i64, hi: i64) -> Vec<Predicate> {
            vec![
                Predicate::Gte {
                    column: "d".into(),
                    value: Literal::Int64(lo),
                },
                Predicate::Lte {
                    column: "d".into(),
                    value: Literal::Int64(hi),
                },
            ]
        }

        #[test]
        fn interior_fragment_is_covered() {
            // [10,20] ⊆ [5,30] → covered, not disjoint.
            let s = stats(10, 20, 0);
            assert!(fragment_fully_covered(&between(5, 30), &s));
            assert!(!fragment_disjoint(&between(5, 30), &s));
        }

        #[test]
        fn straddling_fragment_is_boundary() {
            // [10,20] straddles upper edge 15 → neither fully covered nor disjoint.
            let s = stats(10, 20, 0);
            assert!(!fragment_fully_covered(&between(5, 15), &s));
            assert!(!fragment_disjoint(&between(5, 15), &s));
        }

        #[test]
        fn outside_fragment_is_disjoint() {
            // [10,20] vs [30,40] → disjoint.
            let s = stats(10, 20, 0);
            assert!(fragment_disjoint(&between(30, 40), &s));
            assert!(!fragment_fully_covered(&between(30, 40), &s));
        }

        #[test]
        fn exact_edge_inclusive_is_covered() {
            // [10,20] ⊆ [10,20] inclusive → covered.
            let s = stats(10, 20, 0);
            assert!(fragment_fully_covered(&between(10, 20), &s));
        }

        #[test]
        fn missing_bounds_never_covered() {
            let s = ColumnStats {
                min: None,
                max: None,
                nulls: 0,
            };
            assert!(!fragment_fully_covered(&between(5, 30), &s));
            assert!(!fragment_disjoint(&between(5, 30), &s));
        }

        #[test]
        fn equality_covers_only_single_valued_fragment() {
            let eq = vec![Predicate::Eq {
                column: "d".into(),
                value: Literal::Int64(7),
            }];
            assert!(fragment_fully_covered(&eq, &stats(7, 7, 0)));
            assert!(!fragment_fully_covered(&eq, &stats(7, 8, 0)));
            assert!(fragment_disjoint(&eq, &stats(8, 9, 0)));
        }
    }
}

/// A stream of [`RecordBatch`]es produced by reading a single row group.
pub struct RowGroupStream {
    inner: BoxStream<'static, Result<RecordBatch>>,
}

impl std::fmt::Debug for RowGroupStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RowGroupStream").finish_non_exhaustive()
    }
}

impl Stream for RowGroupStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

/// A read-only reader for a single IcefallDB table.
pub struct Reader<'a> {
    storage: &'a dyn Storage,
    catalog: Catalog<'a>,
}

impl<'a> std::fmt::Debug for Reader<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("table", &self.catalog.table())
            .field(
                "schema_id",
                &self.catalog.latest_schema().map(|s| s.schema_id),
            )
            .finish_non_exhaustive()
    }
}

impl<'a> Reader<'a> {
    /// Open a reader for `table`, loading the latest catalog snapshot.
    ///
    /// Returns [`IcefallDBError::EmptyTable`] if the table exists but has no
    /// committed manifests (including when the table does not exist at all).
    /// Returns [`IcefallDBError::MissingManifestPointer`] if `_schema.json` exists
    /// but `_manifest.json` does not, indicating a partially initialized table.
    pub async fn new(storage: &'a dyn Storage, table: &str) -> Result<Self> {
        let catalog = Catalog::load(storage, table).await?;
        if catalog.latest_manifest().is_none() {
            return Err(IcefallDBError::EmptyTable(table.to_string()));
        }
        Ok(Self { storage, catalog })
    }

    /// Refresh the catalog snapshot from storage.
    pub async fn refresh(&mut self) -> Result<()> {
        self.catalog.refresh().await
    }

    /// Return a reference to the loaded catalog.
    pub fn catalog(&self) -> &Catalog<'a> {
        &self.catalog
    }

    /// Build a scan plan over the latest catalog snapshot.
    pub async fn scan(&self) -> Result<ScanPlan> {
        self.scan_internal(false).await
    }

    /// Build a scan plan that tolerates missing or corrupt `.meta` sidecars.
    ///
    /// Missing sidecars produce a [`PlannedRowGroup`] with `fallback` set to
    /// `true` and a placeholder [`RowGroupMeta`]. When the manifest has
    /// denormalized `row_counts`, the placeholder row count is taken from there;
    /// otherwise it is zero.
    pub async fn scan_allow_missing_meta(&self) -> Result<ScanPlan> {
        self.scan_internal(true).await
    }

    async fn scan_internal(&self, allow_missing_meta: bool) -> Result<ScanPlan> {
        let manifest = self
            .catalog
            .latest_manifest()
            .ok_or_else(|| IcefallDBError::ManifestNotFound(self.catalog.table().to_string()))?;
        // Borrow the catalog's schema (like `manifest` above); `build_scan_plan_from`
        // takes `&Schema` and clones once internally, so an extra clone here is
        // redundant.
        let schema = self
            .catalog
            .latest_schema()
            .ok_or_else(|| IcefallDBError::ManifestNotFound(self.catalog.table().to_string()))?;
        let table = self.catalog.table();
        build_scan_plan_from(self.storage, table, manifest, schema, allow_missing_meta).await
    }
}

/// Build a [`ScanPlan`] from an explicit `manifest` + `schema` pair.
///
/// This is the single plan-building implementation shared by the latest-snapshot
/// path ([`Reader::scan`]) and the time-travel path ([`build_scan_plan_at`]).
async fn build_scan_plan_from(
    storage: &dyn Storage,
    table: &str,
    manifest: &crate::metadata::Manifest,
    schema: &crate::metadata::Schema,
    allow_missing_meta: bool,
) -> Result<ScanPlan> {
    let mut row_groups = Vec::with_capacity(manifest.row_groups.len());

    let cache = MetaCache::global();
    let agg_cache = AggStateCache::global();

    // If the manifest references a snapshot checkpoint, try to load it once.
    // A valid checkpoint lets us build the scan plan without reading every
    // per-fragment `.meta` sidecar. If the checkpoint is missing or malformed,
    // fall back to the sidecar path rather than failing the whole scan.
    let checkpoint: Option<SnapshotCheckpoint> = if let Some(cp_rel) = &manifest.checkpoint {
        let is_valid =
            |cp: &SnapshotCheckpoint| -> bool {
                cp.sequence == manifest.sequence
                    && cp.schema_id == manifest.schema_id
                    && cp.fragments.len() == manifest.row_groups.len()
                    && cp.fragments.iter().zip(manifest.row_groups.iter()).all(
                        |(summary, entry)| {
                            summary.data == entry.data
                                && summary.meta == entry.meta
                                && summary.fragment_id == entry.fragment_id
                                && summary.agg == entry.agg
                                && summary.deletes == entry.deletes
                        },
                    )
                    && (cp.checksum.is_empty() || cp.verify_checksum().unwrap_or(false))
            };
        // Prefer the derived zero-copy rkyv archive (no serde_json structural
        // parse — the O(fragments) win); fall back to the canonical JSON
        // checkpoint, then (if neither validates) to the per-fragment sidecars.
        let arch_abs = format!(
            "{}/{}",
            table,
            SnapshotCheckpoint::archive_filename(manifest.sequence)
        );
        let from_archive = storage
            .read(&arch_abs)
            .await
            .ok()
            .and_then(|bytes| SnapshotCheckpoint::from_archive_bytes(&bytes))
            .filter(&is_valid);
        match from_archive {
            Some(cp) => Some(cp),
            None => {
                let cp_abs = format!("{}/{}", table, cp_rel);
                storage
                    .read(&cp_abs)
                    .await
                    .ok()
                    .and_then(|bytes| serde_json::from_slice::<SnapshotCheckpoint>(&bytes).ok())
                    .filter(&is_valid)
            }
        }
    } else {
        None
    };

    for (idx, entry) in manifest.row_groups.iter().enumerate() {
        let data_path = format!("{}/{}", table, entry.data);
        let meta_path = format!("{}/{}", table, entry.meta);

        let (meta, fallback) =
            if let Some(summary) = checkpoint.as_ref().and_then(|cp| cp.fragments.get(idx)) {
                // O(1) path: reconstruct the row-group metadata from the checkpoint
                // summary. The manifest checksum already covers the checkpoint
                // reference, so we trust a checkpoint that passes the filter above.
                let rg_id = if summary.row_group.is_empty() {
                    entry
                        .meta
                        .strip_suffix(".meta")
                        .unwrap_or(&entry.meta)
                        .to_string()
                } else {
                    summary.row_group.clone()
                };
                let meta = RowGroupMeta {
                    row_group: rg_id,
                    schema_id: schema.schema_id,
                    rows: summary.rows,
                    columns: summary.columns.clone(),
                    column_offsets: summary.column_offsets.clone(),
                    sort: summary.sort.clone(),
                    row_ids: summary.row_ids.clone(),
                    checksum: summary.checksum.clone(),
                    meta_checksum: summary.meta_checksum.clone(),
                };
                (meta, false)
            } else {
                let meta_result: Result<RowGroupMeta> = async {
                    // Cache hit: the sidecar was already read, parsed, and verified
                    // on a prior scan. Fragment files are write-once (rg_{uuid}), so
                    // the path→content mapping is permanent and the cached value is
                    // always valid. Schema-id was verified at insert time.
                    //
                    // Keep PlannedRowGroup.meta as RowGroupMeta (value) for now;
                    // cloning from Arc is far cheaper than read+parse+checksum.
                    // Upgrade PlannedRowGroup.meta to Arc<RowGroupMeta> in the next
                    // pass if this clone shows up in a profile.
                    if let Some(cached) = cache.get(&meta_path) {
                        return Ok((*cached).clone());
                    }

                    // Cache miss: read, parse, verify, then populate the cache.
                    let meta_bytes = match storage.read(&meta_path).await {
                        Ok(bytes) => bytes,
                        Err(IcefallDBError::NotFound(_)) => {
                            return Err(IcefallDBError::MissingRowGroupFile {
                                snapshot: manifest.sequence,
                                path: meta_path.clone(),
                            });
                        }
                        Err(e) => return Err(e),
                    };
                    let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
                    if !meta.verify_meta_checksum()? {
                        return Err(IcefallDBError::RowGroupChecksumMismatch {
                            path: meta_path.clone(),
                        });
                    }
                    if meta.schema_id != schema.schema_id {
                        return Err(IcefallDBError::SchemaMismatch {
                            column: "schema_id".into(),
                            expected: schema.schema_id.to_string(),
                            path: meta_path.clone(),
                        });
                    }
                    // Only cache the success path (checksum+schema verified).
                    // The allow_missing_meta placeholder branch is NOT cached.
                    cache.put(meta_path.clone(), std::sync::Arc::new(meta.clone()));
                    Ok(meta)
                }
                .await;

                match meta_result {
                    Ok(meta) => (meta, false),
                    Err(_) if allow_missing_meta => {
                        let row_count = manifest
                            .row_counts
                            .as_ref()
                            .and_then(|counts| counts.get(idx).copied())
                            .unwrap_or(0);
                        let rg_id = entry
                            .data
                            .strip_suffix(".parquet")
                            .unwrap_or(&entry.data)
                            .to_string();
                        let placeholder = RowGroupMeta {
                            row_group: rg_id,
                            schema_id: schema.schema_id,
                            rows: row_count,
                            columns: HashMap::new(),
                            column_offsets: None,
                            sort: None,
                            row_ids: vec![],
                            checksum: String::new(),
                            meta_checksum: String::new(),
                        };
                        (placeholder, true)
                    }
                    Err(e) => return Err(e),
                }
            };

        let partition_values = manifest
            .partition_values
            .as_ref()
            .and_then(|values| values.get(&entry.data).cloned());

        // Load the `.agg` sidecar when the manifest entry points to one.
        // Cache hit: immediately available; cache miss: read + deserialize.
        // On any error (not found, parse failure), silently use None so the
        // The metadata-aggregate rule falls back to a real scan rather than failing.
        let full_agg_state: Option<std::sync::Arc<FragmentAggState>> =
            if let Some(agg_rel) = &entry.agg {
                let agg_path = format!("{}/{}", table, agg_rel);
                if let Some(cached) = agg_cache.get(&agg_path) {
                    Some(cached)
                } else {
                    match storage.read(&agg_path).await {
                        Ok(bytes) => match deserialize_agg_state(&bytes) {
                            Ok(state) => {
                                let arc = std::sync::Arc::new(state);
                                agg_cache.put(agg_path, std::sync::Arc::clone(&arc));
                                Some(arc)
                            }
                            Err(_) => None,
                        },
                        Err(_) => None,
                    }
                }
            } else {
                None
            };

        // For dirty fragments, compute or fetch from cache the live partial
        // by retracting the deleted rows' contribution from the full partial.
        // The live partial is cached under the composite key
        // "{agg_path}@{dv_path}" — both files are write-once, so the key is
        // permanently valid for the lifetime of the process.
        let agg_state: Option<std::sync::Arc<FragmentAggState>> = if entry.deleted_count > 0 {
            if let (Some(full), Some(del_rel), Some(agg_rel)) =
                (&full_agg_state, &entry.deletes, &entry.agg)
            {
                let dv_path = format!("{}/{}", table, del_rel);
                let agg_path = format!("{}/{}", table, agg_rel);
                let live_key = format!("{}@{}", agg_path, dv_path);

                if let Some(cached_live) = agg_cache.get(&live_key) {
                    Some(cached_live)
                } else {
                    let live_opt: Option<std::sync::Arc<FragmentAggState>> =
                        match storage.read(&dv_path).await {
                            Ok(dv_bytes) => match DeletionVector::deserialize(&dv_bytes) {
                                Ok(dv) => {
                                    let numeric_cols: Vec<String> =
                                        full.cols.keys().cloned().collect();
                                    match deleted_contribution(
                                        storage,
                                        table,
                                        entry,
                                        &dv,
                                        &numeric_cols,
                                    )
                                    .await
                                    {
                                        Ok(del_contrib) => {
                                            let live = retract(full, &del_contrib);
                                            // Resolve live min/max per integer column.
                                            let mut live = resolve_live_extrema(
                                                storage, table, entry, &dv, &meta, live,
                                            )
                                            .await;
                                            // Resolve LIVE grouped partials when a
                                            // group key is declared and this fragment carries
                                            // grouped partials.  `retract` copied the FULL
                                            // (pre-deletion) grouped map; replace it with the
                                            // deletion-aware version routed by group key.  On
                                            // any failure leave `grouped = None` so the rule
                                            // falls back for grouped queries on this fragment.
                                            live.grouped = resolve_live_grouped(
                                                storage,
                                                table,
                                                entry,
                                                &dv,
                                                schema,
                                                full.grouped(),
                                            )
                                            .await;
                                            Some(std::sync::Arc::new(live))
                                        }
                                        Err(_) => None,
                                    }
                                }
                                Err(_) => None,
                            },
                            Err(_) => None,
                        };
                    if let Some(ref live_arc) = live_opt {
                        agg_cache.put(live_key, std::sync::Arc::clone(live_arc));
                    }
                    live_opt
                }
            } else {
                // Dirty fragment without agg sidecar or deletion file ref —
                // no live partial available; rule falls back.
                None
            }
        } else {
            // Clean fragment: use the full partial as-is.
            full_agg_state
        };

        row_groups.push(PlannedRowGroup {
            data_path,
            meta_path,
            meta,
            partition_values,
            snapshot: manifest.sequence,
            fallback,
            deletes: entry.deletes.as_ref().map(|p| format!("{}/{}", table, p)),
            deleted_count: entry.deleted_count,
            fragment_id: entry.fragment_id,
            agg_state,
        });
    }

    Ok(ScanPlan {
        table: table.to_string(),
        schema: schema.clone(),
        row_groups,
    })
}

/// A summary of a single committed snapshot, returned by [`list_snapshots`].
#[derive(Debug, Clone)]
pub struct SnapshotInfo {
    pub sequence: u64,
    pub committed_at: Option<String>,
    /// Live rows (physical rows minus `deleted_count`). For the current
    /// checkpoint snapshot this folds any pending mutation WAL (DELETE,
    /// UPDATE, and MERGE) so it matches the live query count (`SELECT
    /// COUNT(*)`); older snapshots reflect their committed deletion state.
    pub rows: u64,
    pub fragments: usize,
    pub parent_hash: Option<String>,
    /// `true` when the row count was computed by folding pending WAL records
    /// (DELETE/UPDATE/MERGE) into the current checkpoint snapshot.
    pub wal_folded: bool,
}

/// Build a [`ScanPlan`] for the table as of the snapshot at `sequence`.
///
/// Loads `_manifests/<sequence>.json` from storage and resolves the schema
/// that was current at that snapshot via `manifest.schema_id`. Returns
/// [`IcefallDBError::SnapshotNotFound`] if the manifest file is absent or
/// cannot be parsed.
///
/// This reads the raw historical manifest without applying any in-flight WAL
/// mutations, which is the correct behaviour for time-travel reads.
pub async fn build_scan_plan_at(
    storage: &dyn Storage,
    table: &str,
    sequence: u64,
) -> Result<ScanPlan> {
    let path = format!(
        "{}/{}",
        table,
        crate::metadata::Manifest::filename(sequence)
    );
    let bytes = storage
        .read(&path)
        .await
        .map_err(|_| IcefallDBError::SnapshotNotFound(sequence))?;
    let manifest: crate::metadata::Manifest =
        serde_json::from_slice(&bytes).map_err(|_| IcefallDBError::SnapshotNotFound(sequence))?;

    // Verify the historical manifest's integrity.  Legacy manifests have an
    // empty checksum field AND no `committed_at` (both fields were introduced
    // together with hash chaining); skip verification for those so time-travel
    // reads on legacy tables remain functional.  A non-empty checksum that does
    // not match indicates a corrupt or truncated manifest file.  An empty
    // checksum with a timestamp present is post-chain tampering and must still
    // be rejected.
    let is_legacy_anchor = manifest.checksum.is_empty() && manifest.committed_at.is_none();
    if !is_legacy_anchor && !manifest.verify_checksum()? {
        return Err(IcefallDBError::ChecksumMismatch { path });
    }

    // Load the schema that was current at this snapshot (schema-as-of).
    let schema_path = format!(
        "{}/{}",
        table,
        crate::metadata::Schema::filename(manifest.schema_id)
    );
    let schema_bytes = storage.read(&schema_path).await?;
    let mut schema: crate::metadata::Schema = serde_json::from_slice(&schema_bytes)?;
    if !schema.has_field_ids() {
        schema.repair_field_ids();
    }

    build_scan_plan_from(storage, table, &manifest, &schema, false).await
}

/// Physical (pre-deletion) rows for a manifest.
///
/// Prefers the denormalized `row_counts`. When it is absent — WAL-folded
/// checkpoints and UPDATE/MERGE commits publish manifests without it — falls
/// back to summing each row group's canonical `.meta` `rows`, so the count is
/// correct rather than silently zero.
async fn manifest_physical_rows_resolved(
    storage: &dyn Storage,
    table: &str,
    m: &crate::metadata::Manifest,
) -> Result<u64> {
    if let Some(counts) = &m.row_counts {
        return Ok(counts.iter().map(|&r| r as u64).sum());
    }
    let mut total = 0u64;
    for entry in &m.row_groups {
        let meta_path = format!("{}/{}", table, entry.meta);
        let bytes = match storage.read(&meta_path).await {
            Ok(b) => b,
            Err(e) if crate::is_not_found(&e) => continue,
            Err(e) => return Err(e),
        };
        let meta: RowGroupMeta = serde_json::from_slice(&bytes)?;
        total += meta.rows as u64;
    }
    Ok(total)
}

/// Live rows for a manifest: physical rows minus `deleted_count` on each row
/// group entry.
async fn manifest_live_rows_resolved(
    storage: &dyn Storage,
    table: &str,
    m: &crate::metadata::Manifest,
) -> Result<u64> {
    let physical = manifest_physical_rows_resolved(storage, table, m).await?;
    let deleted: u64 = m.row_groups.iter().map(|e| e.deleted_count).sum();
    Ok(physical.saturating_sub(deleted))
}

/// Read the `_manifest.json` pointer sequence, if present and valid.
async fn read_pointer_sequence(storage: &dyn Storage, table: &str) -> Option<u64> {
    let data = storage
        .read(&format!("{}/_manifest.json", table))
        .await
        .ok()?;
    let pointer: serde_json::Value = serde_json::from_slice(&data).ok()?;
    pointer.get("latest").and_then(|v| v.as_u64())
}

/// Return the list of all on-disk snapshots for `table`, sorted ascending by
/// sequence number.
///
/// Each entry is populated from the manifest file for that sequence. Manifests
/// that cannot be read or parsed are silently skipped (consistent with the GC
/// and doctor behaviour for missing/in-flight files).
///
/// Row counts are *live* counts: physical rows minus `deleted_count`. Physical
/// rows come from the denormalized `row_counts` when present and otherwise from
/// each row group's canonical `.meta` sidecar, so a manifest published without
/// `row_counts` (WAL fold, UPDATE/MERGE) is counted correctly rather than as
/// zero. For the current checkpoint snapshot, any pending mutation WAL records
/// are folded — including patch fragments appended by UPDATE/MERGE — so the
/// displayed count matches a live `SELECT COUNT(*)`.
pub async fn list_snapshots(storage: &dyn Storage, table: &str) -> Result<Vec<SnapshotInfo>> {
    require_table_exists(storage, table).await?;

    let manifests_dir = format!("{}/_manifests", table);
    let entries = match storage.list(&manifests_dir).await {
        Ok(e) => e,
        Err(e) if crate::is_not_found(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };

    // Parse sequence numbers from filenames (e.g. `_manifests/000000001.json`).
    let mut seqs: Vec<u64> = Vec::new();
    for entry in &entries {
        let filename = std::path::Path::new(entry)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        if let Some(stem) = filename.strip_suffix(".json") {
            if let Ok(seq) = stem.parse::<u64>() {
                seqs.push(seq);
            }
        }
    }
    seqs.sort_unstable();

    let current_seq = read_pointer_sequence(storage, table).await.unwrap_or(0);

    let mut out = Vec::with_capacity(seqs.len());
    for seq in seqs {
        let path = format!("{}/{}", table, crate::metadata::Manifest::filename(seq));
        let bytes = match storage.read(&path).await {
            Ok(b) => b,
            Err(e) if crate::is_not_found(&e) => continue, // race: removed between list+read
            Err(e) => return Err(e),
        };
        let m: crate::metadata::Manifest = match serde_json::from_slice(&bytes) {
            Ok(m) => m,
            Err(_) => continue, // skip invalid manifests
        };

        // For the current checkpoint snapshot, fold any pending WAL so the row
        // count reflects live, committed state. Physical rows are taken from the
        // folded manifest's row groups (which include any UPDATE/MERGE patch
        // fragments the WAL appended), so the displayed count matches a live
        // `SELECT COUNT(*)` exactly rather than undercounting.
        let (rows, wal_folded) = if seq == current_seq {
            let live = crate::mutation_wal::live_manifest(storage, table, m.clone()).await?;
            let rows = manifest_live_rows_resolved(storage, table, &live).await?;
            (rows, live.sequence != m.sequence)
        } else {
            (
                manifest_live_rows_resolved(storage, table, &m).await?,
                false,
            )
        };

        out.push(SnapshotInfo {
            sequence: m.sequence,
            committed_at: m.committed_at.clone(),
            rows,
            fragments: m.row_groups.len(),
            parent_hash: m.parent_hash.clone(),
            wal_folded,
        });
    }
    Ok(out)
}

/// Verify that `table` exists by checking that its schema pointer and the schema
/// file it references are present.
///
/// `_schema.json` (plus the schema snapshot it names) is the authoritative
/// existence marker for a table: the writer creates it first, and `doctor` can
/// rebuild a lost `_manifest.json` pointer from the retained `_manifests/`
/// snapshots. Gating on the manifest pointer here would therefore make a
/// recoverable pointer loss look like a missing table and block its repair.
///
/// This is the shared table-existence gate for maintenance commands. It returns
/// [`IcefallDBError::TableNotFound`] when the schema pointer is missing or points
/// to a schema snapshot that does not exist.
pub async fn require_table_exists(storage: &dyn Storage, table: &str) -> Result<()> {
    let schema_pointer_path = format!("{}/_schema.json", table);
    let schema_id = match storage.exists(&schema_pointer_path).await? {
        true => {
            let data = storage.read(&schema_pointer_path).await?;
            let pointer: serde_json::Value = serde_json::from_slice(&data)?;
            pointer.get("latest").and_then(|v| v.as_u64())
        }
        false => None,
    };
    let Some(schema_id) = schema_id else {
        return Err(IcefallDBError::TableNotFound(table.to_string()));
    };

    let schema_path = format!("{}/{}", table, Schema::filename(schema_id));
    if !storage.exists(&schema_path).await? {
        return Err(IcefallDBError::TableNotFound(table.to_string()));
    }

    Ok(())
}

impl<'a> Reader<'a> {
    /// Read a planned row group and return a stream of Arrow record batches.
    ///
    /// Verifies both the row-group metadata checksum and the Parquet data
    /// checksum before streaming any batches. The current [`Storage`] trait
    /// returns the whole Parquet file as a `Vec<u8>`, so the file is read into
    /// memory once; batches are then produced lazily from the synchronous
    /// Parquet reader rather than collected upfront.
    pub async fn read_row_group(&self, prg: &PlannedRowGroup) -> Result<RowGroupStream> {
        let meta_bytes = match self.storage.read(&prg.meta_path).await {
            Ok(bytes) => bytes,
            Err(IcefallDBError::NotFound(_)) => {
                return Err(IcefallDBError::MissingRowGroupFile {
                    snapshot: prg.snapshot,
                    path: prg.meta_path.clone(),
                });
            }
            Err(e) => return Err(e),
        };
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes)?;
        if !meta.verify_meta_checksum()? {
            return Err(IcefallDBError::RowGroupChecksumMismatch {
                path: prg.meta_path.clone(),
            });
        }

        let data_bytes = match self.storage.read(&prg.data_path).await {
            Ok(bytes) => bytes,
            Err(IcefallDBError::NotFound(_)) => {
                return Err(IcefallDBError::MissingRowGroupFile {
                    snapshot: prg.snapshot,
                    path: prg.data_path.clone(),
                });
            }
            Err(e) => return Err(e),
        };
        if !meta.verify_against_data(&data_bytes) {
            return Err(IcefallDBError::RowGroupChecksumMismatch {
                path: prg.data_path.clone(),
            });
        }

        let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data_bytes))
            .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;
        let reader = builder
            .build()
            .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;

        let stream = futures::stream::iter(reader)
            .map(|res| res.map_err(|e| IcefallDBError::ParquetDecode(e.to_string())))
            .boxed();

        Ok(RowGroupStream { inner: stream })
    }
}

/// Resolve LIVE per-group partials for a dirty fragment, accounting for the
/// current deletion vector.
///
/// Returns `Some(GroupedPartials)` only when:
///
/// - the table declares at least one `agg_group_keys`,
/// - the fragment carries full grouped partials (`full_grouped` is `Some`),
/// - that fragment's `key_col` equals the first declared key, and
/// - [`retract_grouped`] succeeds.
///
/// On any mismatch or failure returns `None`, so the warm-GROUP-BY rule falls back to a
/// full GROUP BY scan for this fragment rather than risk a stale result.
async fn resolve_live_grouped(
    storage: &dyn Storage,
    table: &str,
    entry: &crate::metadata::RowGroupEntry,
    dv: &crate::deletion::DeletionVector,
    schema: &Schema,
    full_grouped: Option<&GroupedPartials>,
) -> Option<GroupedPartials> {
    let key_col = schema.agg_group_keys.as_deref().and_then(|v| v.first())?;
    let gp = full_grouped?;
    if &gp.key_col != key_col {
        return None;
    }
    // Measure columns = the union of all per-group measure column names.
    let mut measure_set: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for cols in gp.groups.values() {
        for name in cols.keys() {
            measure_set.insert(name.clone());
        }
    }
    let measure_cols: Vec<String> = measure_set.into_iter().collect();
    match retract_grouped(storage, table, entry, gp, dv, &measure_cols).await {
        Ok(live_groups) => Some(GroupedPartials {
            key_col: gp.key_col.clone(),
            groups: live_groups,
        }),
        Err(_) => None,
    }
}

/// For each integer column in `live`, resolve the live min and max JSON
/// values by checking whether the extremal row was deleted.
///
/// - If the extremal row survives (not in DV): use the existing `ColumnStats`
///   min/max from `meta.columns` — zero I/O.
/// - If the extremal row was deleted: call `scoped_recompute_extremum` to scan
///   only the survivor rows — one Parquet read per column that needs it.
/// - Float columns and columns without `min_off`/`max_off` (old `.agg` files):
///   leave `live_min_json`/`live_max_json` as `None` so the optimizer falls back.
async fn resolve_live_extrema(
    storage: &dyn Storage,
    table: &str,
    entry: &crate::metadata::RowGroupEntry,
    dv: &crate::deletion::DeletionVector,
    meta: &crate::metadata::RowGroupMeta,
    mut live: crate::agg_cache::FragmentAggState,
) -> crate::agg_cache::FragmentAggState {
    use crate::agg_cache::{extremum_deleted, scoped_recompute_extremum, AggScalar, ExtremumKind};

    for (col_name, col_agg) in &mut live.cols {
        // Skip float columns — float guard is unconditional.
        if matches!(&col_agg.sum, AggScalar::Float(_)) {
            continue;
        }

        // Resolve live_min_json.
        col_agg.live_min_json = if extremum_deleted(col_agg, dv, ExtremumKind::Min) {
            // Extremal row was deleted → scan survivors for new min.
            scoped_recompute_extremum(
                storage,
                table,
                entry,
                dv,
                meta.rows,
                col_name,
                ExtremumKind::Min,
            )
            .await
            .unwrap_or_default()
        } else {
            // Extremal row survived → zero-I/O: use ColumnStats min.
            meta.columns.get(col_name).and_then(|s| s.min.clone())
        };

        // Resolve live_max_json.
        col_agg.live_max_json = if extremum_deleted(col_agg, dv, ExtremumKind::Max) {
            scoped_recompute_extremum(
                storage,
                table,
                entry,
                dv,
                meta.rows,
                col_name,
                ExtremumKind::Max,
            )
            .await
            .unwrap_or_default()
        } else {
            meta.columns.get(col_name).and_then(|s| s.max.clone())
        };
    }

    live
}
