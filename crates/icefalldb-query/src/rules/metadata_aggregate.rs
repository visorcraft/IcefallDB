//! Physical optimizer rule that answers simple aggregates from IcefallDB metadata.
//!
//! When an [`AggregateExec`] sits directly on a [`IcefallDBScanExec`] and every
//! aggregate expression is `COUNT(*)`, `COUNT(col)`, `MIN(col)`, `MAX(col)`,
//! `SUM(col)`, `AVG(col)`, `VAR_POP(col)`, `VAR_SAMP(col)`, `STDDEV_POP(col)`,
//! or `STDDEV_SAMP(col)`, the rule replaces the aggregate with a single-row
//! memory source containing the computed scalars.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::physical_expr::expressions::{CastExpr, Column, Literal};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::PhysicalGroupBy;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::memory::MemorySourceConfig;
use icefalldb_core::agg_cache::{merge_grouped, AggScalar, ColAgg, GroupedPartials};
use icefalldb_core::PlannedRowGroup;

use crate::error::QueryError;
use crate::scalar_codec::json_to_scalar_value;
use crate::scan::IcefallDBScanExec;
use crate::session::IcefallDBConfig;
use std::collections::BTreeMap;

/// Physical optimizer rule that folds metadata-readable aggregates into a
/// single constant row.
#[derive(Debug, Default)]
pub struct MetadataAggregate;

impl MetadataAggregate {
    /// Create a new `MetadataAggregate` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for MetadataAggregate {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let enabled = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.metadata_aggregate)
            .unwrap_or(true);
        if !enabled {
            return Ok(plan);
        }

        // Recursively apply the rule bottom-up.
        let plan = crate::rules::optimize_children(plan, config, |plan, config| {
            self.optimize(plan, config)
        })?;
        transform_aggregate(plan)
    }

    fn name(&self) -> &str {
        "metadata_aggregate"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Peel through any [`CoalescePartitionsExec`],
/// [`datafusion::physical_plan::coop::CooperativeExec`], and — for the grouped
/// two-phase HASH plan — [`RepartitionExec`] wrappers to reach the inner child.
/// Used when navigating from the Final aggregate's input to the Partial
/// aggregate in a two-phase plan.
///
/// CRITICAL: a multi-fragment `GROUP BY k` builds a TWO-PHASE HASH plan
/// `AggregateExec(FinalPartitioned,group=[k]) → [CooperativeExec] →
/// RepartitionExec(Hash[k]) → [CooperativeExec] → AggregateExec(Partial,
/// group=[k]) → … → scan`.  Without peeling `RepartitionExec` the rule would
/// no-op on every real grouped table.  Ungrouped two-phase plans use
/// `CoalescePartitionsExec` instead and never hit the RepartitionExec arm.
fn find_inner_input(input: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    use datafusion::physical_plan::coop::CooperativeExec;
    use datafusion::physical_plan::repartition::RepartitionExec;
    let mut current = Arc::clone(input);
    loop {
        if let Some(coop) = current.downcast_ref::<CooperativeExec>() {
            current = Arc::clone(coop.input());
        } else if let Some(coalesce) = current.downcast_ref::<CoalescePartitionsExec>() {
            current = Arc::clone(coalesce.input());
        } else if let Some(repart) = current.downcast_ref::<RepartitionExec>() {
            current = Arc::clone(repart.input());
        } else {
            break;
        }
    }
    current
}

/// Peel through any [`CooperativeExec`] and a single optional [`ProjectionExec`]
/// that contains only numeric→float cast-of-column expressions (the coercion
/// DataFusion 54 lifts out of AVG/VAR/STDDEV aggregates into a separate node).
///
/// Returns `(scan_candidate, col_map)` where:
/// - `scan_candidate` is the innermost node after peeling wrappers.
/// - `col_map` maps projected-alias → original-scan column name for every
///   `CastExpr(Column → Float32|Float64)` entry in the projection.
///
/// If no `ProjectionExec` is present the map is empty.  If the projection
/// contains any expression that is NOT a float-cast-of-column, the whole
/// function returns `None` (caller falls back).
fn peel_to_scan(
    input: &Arc<dyn ExecutionPlan>,
) -> Option<(Arc<dyn ExecutionPlan>, HashMap<String, String>)> {
    use datafusion::physical_plan::coop::CooperativeExec;
    use datafusion::physical_plan::repartition::RepartitionExec;
    let mut current = Arc::clone(input);

    // Peel CooperativeExec and RepartitionExec wrappers.  A RepartitionExec
    // (RoundRobinBatch/Hash) appears between the Partial aggregate and the scan
    // when DataFusion parallelizes a single-partition scan; it only redistributes
    // rows across partitions and never alters column identity or values, so the
    // composed-from-sidecar result is unchanged by peeling it.
    loop {
        if let Some(coop) = current.downcast_ref::<CooperativeExec>() {
            current = Arc::clone(coop.input());
        } else if let Some(repart) = current.downcast_ref::<RepartitionExec>() {
            current = Arc::clone(repart.input());
        } else {
            break;
        }
    }

    // If the next node is a ProjectionExec, inspect it.
    if let Some(proj) = current.downcast_ref::<ProjectionExec>() {
        let mut col_map: HashMap<String, String> = HashMap::with_capacity(proj.expr().len());
        for proj_expr in proj.expr() {
            if let Some(col) = proj_expr.expr.downcast_ref::<Column>() {
                // Passthrough column: identity mapping (alias → col name).
                col_map.insert(proj_expr.alias.clone(), col.name().to_string());
            } else if let Some(cast) = proj_expr.expr.downcast_ref::<CastExpr>() {
                // Only accept casts to float types — those are the coercions DF
                // inserts for integer inputs to AVG/VAR/STDDEV.
                if !matches!(
                    cast.cast_type(),
                    arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64
                ) {
                    return None;
                }
                let inner_col = cast.expr().downcast_ref::<Column>()?;
                col_map.insert(proj_expr.alias.clone(), inner_col.name().to_string());
            } else {
                // Any other expression shape (arithmetic, literal, multi-column,
                // function, …) → fall back.
                return None;
            }
        }
        // The projection's input is the actual scan (peel Coop/Repartition below).
        let mut inner = Arc::clone(proj.input());
        loop {
            if let Some(coop) = inner.downcast_ref::<CooperativeExec>() {
                inner = Arc::clone(coop.input());
            } else if let Some(repart) = inner.downcast_ref::<RepartitionExec>() {
                inner = Arc::clone(repart.input());
            } else {
                break;
            }
        }
        return Some((inner, col_map));
    }

    // No projection — bare scan (or something the caller will reject).
    Some((current, HashMap::new()))
}

fn transform_aggregate(plan: Arc<dyn ExecutionPlan>) -> DFResult<Arc<dyn ExecutionPlan>> {
    let Some(agg) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(plan);
    };

    // Grouped aggregates take the warm-GROUP-BY path; ungrouped ones the
    // single-row path below.
    if !agg.group_expr().is_empty() {
        return transform_grouped_aggregate(&plan, agg);
    }

    match agg.mode() {
        // ── Single-phase path (Bug A: gate to Single/SinglePartitioned so a bare
        //    Partial node in the bottom-up traversal is never incorrectly folded
        //    into a final scalar value). ──────────────────────────────────────────
        AggregateMode::Single | AggregateMode::SinglePartitioned => {
            // Aggregates with filters are not handled.
            let filter_exprs = agg.filter_expr();
            if agg.aggr_expr().len() != filter_exprs.len()
                || filter_exprs.iter().any(|f| f.is_some())
            {
                return Ok(plan);
            }
            // Peel Coop/Projection wrappers to reach the scan, collecting any
            // numeric→float coercion column-name mapping from a projection.
            let Some((scan_arc, col_map)) = peel_to_scan(agg.input()) else {
                return Ok(plan);
            };
            let Some(scan) = scan_arc.downcast_ref::<IcefallDBScanExec>() else {
                return Ok(plan);
            };
            // Range/equality-filtered SUM/COUNT: compose covered fragments and
            // scan only the boundary fragments.  Falls through to the
            // unfiltered fast path when there are no pushed filters.
            if !scan.filters().is_empty() {
                return match compose_filtered_from_scan(
                    agg.aggr_expr(),
                    &agg.schema(),
                    &agg.input_schema(),
                    scan,
                    &col_map,
                ) {
                    Some(result) => result,
                    None => Ok(plan),
                };
            }
            match compose_from_scan(
                agg.aggr_expr(),
                &agg.schema(),
                &agg.input_schema(),
                scan,
                &col_map,
            ) {
                Some(result) => result,
                None => Ok(plan),
            }
        }

        // ── Two-phase path (Bug B): Final → [Coalesce/Coop] → Partial → scan ───
        //    For no-group-by aggregates DataFusion builds:
        //
        //    With float columns (bare column in aggregate arg):
        //      AggregateExec(Final) → CoalescePartitionsExec
        //                           → AggregateExec(Partial) → IcefallDBScanExec
        //
        //    With integer columns (DataFusion 54 lifts the cast into a projection):
        //      AggregateExec(Final) → CoalescePartitionsExec
        //                           → AggregateExec(Partial)
        //                             → ProjectionExec [CastExpr(col → Float64)]
        //                               → IcefallDBScanExec
        //
        //    We detect both shapes and replace the entire subtree with a constant.
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            // Peel CoalescePartitionsExec (and any CooperativeExec) between
            // the Final and the Partial aggregate.
            let partial_candidate = find_inner_input(agg.input());
            let Some(partial_agg) = partial_candidate.downcast_ref::<AggregateExec>() else {
                return Ok(plan);
            };
            if !matches!(partial_agg.mode(), AggregateMode::Partial) {
                return Ok(plan);
            }
            // The Partial must also be ungrouped.
            if !partial_agg.group_expr().is_empty() {
                return Ok(plan);
            }
            // Aggregates with filters are not handled.
            let filter_exprs = partial_agg.filter_expr();
            if partial_agg.aggr_expr().len() != filter_exprs.len()
                || filter_exprs.iter().any(|f| f.is_some())
            {
                return Ok(plan);
            }
            // Peel Coop/Projection wrappers below the Partial to reach the scan.
            // A ProjectionExec is present when DataFusion coerced integer columns
            // to Float64 — `peel_to_scan` returns the alias→scan-col mapping.
            let Some((scan_arc, col_map)) = peel_to_scan(partial_agg.input()) else {
                return Ok(plan);
            };
            let Some(scan) = scan_arc.downcast_ref::<IcefallDBScanExec>() else {
                return Ok(plan);
            };
            // Range/equality-filtered SUM/COUNT over a two-phase plan: compose
            // covered fragments + boundary scan, matching the FINAL output schema.
            if !scan.filters().is_empty() {
                return match compose_filtered_from_scan(
                    partial_agg.aggr_expr(),
                    &agg.schema(),
                    &partial_agg.input_schema(),
                    scan,
                    &col_map,
                ) {
                    Some(result) => result,
                    None => Ok(plan),
                };
            }
            // Use the Partial's aggr_exprs (they reference the original data
            // columns, possibly via the projection alias) and the Final's schema.
            match compose_from_scan(
                partial_agg.aggr_expr(),
                &agg.schema(),
                &partial_agg.input_schema(),
                scan,
                &col_map,
            ) {
                Some(result) => result,
                None => Ok(plan),
            }
        }

        // Any other mode (Partial, PartialReduce, …) — never fold.
        _ => Ok(plan),
    }
}

/// Try to compose all aggregate expressions from the scan's `.agg` sidecar and
/// return a single-row `MemorySourceConfig` that replaces the aggregate subtree.
///
/// Returns `None` if any precondition is not met (caller must leave plan unchanged).
///
/// Arguments:
/// - `aggr_exprs`: the aggregate function expressions referencing the *original*
///   data columns.  For single-phase this is the root aggregate's `aggr_expr()`;
///   for two-phase it is the Partial aggregate's `aggr_expr()`.
/// - `output_schema`: the final output schema (Final or Single aggregate's schema).
/// - `input_schema`: the original input column schema (used by MIN/MAX type lookup).
/// - `scan`: the validated `IcefallDBScanExec` with no filters.
/// - `col_map`: alias → scan-column-name mapping collected by `peel_to_scan` when
///   DataFusion inserted a `ProjectionExec` for numeric→float coercion.  Empty
///   when the aggregate args reference scan columns directly.
fn compose_from_scan(
    aggr_exprs: &[Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>],
    output_schema: &arrow::datatypes::Schema,
    input_schema: &arrow::datatypes::Schema,
    scan: &IcefallDBScanExec,
    col_map: &HashMap<String, String>,
) -> Option<DFResult<Arc<dyn ExecutionPlan>>> {
    // Validate that every aggregate is supported (distinct / unsupported → fall back).
    for aggr in aggr_exprs {
        if aggr.is_distinct() {
            return None;
        }
        let name = aggr.fun().name().to_lowercase();
        // `approx_distinct` is handled separately in the sketches path below.
        // Let it pass the validation check so the match arm can route it.
        #[cfg(feature = "sketches")]
        let is_approx_distinct = name == "approx_distinct";
        #[cfg(not(feature = "sketches"))]
        let is_approx_distinct = false;

        // `approx_percentile_cont` takes two args (col, q); handled separately.
        #[cfg(feature = "sketches")]
        let is_approx_percentile = name == "approx_percentile_cont";
        #[cfg(not(feature = "sketches"))]
        let is_approx_percentile = false;

        // Only the canonical names that DataFusion 54's fun().name() actually
        // produces.  Dead aliases ("mean", "variance", "var_samp", "stddev_samp",
        // "std") are omitted — they never match in practice and would widen the
        // fast path to functions that might change behaviour.
        let supported = [
            "count",
            "min",
            "max",
            "sum",
            "avg",
            "var",
            "var_pop",
            "stddev",
            "stddev_pop",
        ];
        // Single-arg functions: exactly 1 expression required.
        // approx_percentile_cont: 2 expressions (col + q literal).
        let n_args = aggr.expressions().len();
        let ok = ((supported.contains(&name.as_str()) || is_approx_distinct) && n_args == 1)
            || (is_approx_percentile && n_args == 2);
        if !ok {
            return None;
        }
    }

    let mut output_values: Vec<ScalarValue> = Vec::with_capacity(aggr_exprs.len());
    for (i, aggr) in aggr_exprs.iter().enumerate() {
        let name = aggr.fun().name().to_lowercase();
        let args = aggr.expressions();
        let value = match name.as_str() {
            "count" => compute_count(&args[0], scan),
            "min" => compute_min(&args[0], scan, input_schema),
            "max" => compute_max(&args[0], scan, input_schema),
            "sum" => compose_sum(&args[0], scan, output_schema, i),
            "avg" => compose_avg(&args[0], scan, output_schema, i, col_map),
            "var" => compose_variance(&args[0], scan, output_schema, i, false, col_map),
            "var_pop" => compose_variance(&args[0], scan, output_schema, i, true, col_map),
            "stddev" => compose_stddev(&args[0], scan, output_schema, i, false, col_map),
            "stddev_pop" => compose_stddev(&args[0], scan, output_schema, i, true, col_map),
            #[cfg(feature = "sketches")]
            "approx_distinct" => compose_approx_distinct(&args[0], scan),
            #[cfg(feature = "sketches")]
            "approx_percentile_cont" => compose_approx_percentile(&args[0], &args[1], scan),
            _ => return None,
        };
        match value {
            Ok(v) => output_values.push(v),
            Err(_) => return None,
        }
    }

    // Build a single-row batch that matches the aggregate's output schema.
    let output_schema_ref: SchemaRef = Arc::new(output_schema.clone());
    let arrays: Vec<arrow::array::ArrayRef> = match output_values
        .iter()
        .map(|v| v.to_array_of_size(1))
        .collect::<DFResult<_>>()
    {
        Ok(a) => a,
        Err(e) => return Some(Err(e)),
    };
    let batch = match RecordBatch::try_new(Arc::clone(&output_schema_ref), arrays) {
        Ok(b) => b,
        Err(e) => return Some(Err(e.into())),
    };

    match MemorySourceConfig::try_new_from_batches(output_schema_ref, vec![batch]) {
        Ok(exec) => Some(Ok(exec as Arc<dyn ExecutionPlan>)),
        Err(e) => Some(Err(e)),
    }
}

// ── partial-aggregate pushdown for range/equality-filtered SUM/COUNT ──
//
// For `AggregateExec(SUM(v) | COUNT(*) | COUNT(v), group=[])` over a
// `IcefallDBScanExec` carrying a single decidable range/equality filter `F`, we
// classify the scan's fragments by their sidecar [min,max] for the filter
// column vs `F`'s accepted range:
//   * COVERED  — `[min,max] ⊆ accepted-range` AND `nulls==0` AND the fragment is
//                CLEAN (deleted_count==0, agg_state present, hash matches): every
//                row provably passes `F`, so its cached partial contributes a
//                constant with ZERO Parquet I/O.
//   * BOUNDARY — overlaps the edge (some rows pass): scanned by a reconstructed
//                boundary scan that re-applies `F`.
//   * DISJOINT — no overlap: contributes nothing (usually already pruned out).
//
// Rewrite:
//   ProjectionExec[ boundary_agg + covered_const  AS  <orig output> ]
//     ( AggregateExec(SUM/COUNT, group=[]) ( boundary_scan_with_F ) )
// or, when there are no boundary fragments, a single-row constant source.
//
// Falls back (returns `None`) on ANY uncertainty: unsupported aggregate, a
// filter that is not a single-column decidable range/equality form, a covered
// fragment that is dirty / missing-partial / hash-mismatched, no covered
// fragment (no benefit), or an output type that cannot represent the addition.

/// A decidable single-column predicate decoded from the scan's PHYSICAL filter
/// list, expressed in IcefallDB's `Predicate` form so the existing sidecar
/// comparison helpers can decide per-fragment coverage.
struct DecodedFilter {
    column: String,
    predicates: Vec<icefalldb_core::Predicate>,
}

/// Decode a physical `Literal` into a IcefallDB scan `Literal`.  Mirrors the
/// logical `scalar_to_literal` in `crate::predicate`; only the scalar variants
/// that compare cleanly against JSON sidecar stats are accepted.
fn phys_scalar_to_literal(value: &ScalarValue) -> Option<icefalldb_core::Literal> {
    use icefalldb_core::Literal as L;
    match value {
        ScalarValue::Int8(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::Int16(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::Int32(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::Int64(Some(v)) => Some(L::Int64(*v)),
        ScalarValue::UInt8(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::UInt16(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::UInt32(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::UInt64(Some(v)) => Some(L::Int64(*v as i64)),
        ScalarValue::Float32(Some(v)) => Some(L::Float64(*v as f64)),
        ScalarValue::Float64(Some(v)) => Some(L::Float64(*v)),
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => Some(L::String(v.clone())),
        ScalarValue::Boolean(Some(v)) => Some(L::Bool(*v)),
        _ => None,
    }
}

/// Translate one physical comparison expression (`col OP lit` or `lit OP col`)
/// into a `(column, Predicate)`.  Returns `None` for any other shape so the
/// caller falls back.
fn phys_comparison_to_predicate(
    expr: &Arc<dyn PhysicalExpr>,
) -> Option<(String, icefalldb_core::Predicate)> {
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::BinaryExpr;
    use icefalldb_core::Predicate;

    let bin = expr.downcast_ref::<BinaryExpr>()?;
    let op = *bin.op();

    // Identify which side is the column and which is the literal.
    let (col_name, lit, swapped) = if let (Some(col), Some(lit)) = (
        bin.left().downcast_ref::<Column>(),
        bin.right().downcast_ref::<Literal>(),
    ) {
        (col.name().to_string(), lit.value(), false)
    } else if let (Some(lit), Some(col)) = (
        bin.left().downcast_ref::<Literal>(),
        bin.right().downcast_ref::<Column>(),
    ) {
        (col.name().to_string(), lit.value(), true)
    } else {
        return None;
    };

    let value = phys_scalar_to_literal(lit)?;
    let column = col_name.clone();
    let pred = match (op, swapped) {
        (Operator::Eq, _) => Predicate::Eq { column, value },
        (Operator::Lt, false) | (Operator::Gt, true) => Predicate::Lt { column, value },
        (Operator::LtEq, false) | (Operator::GtEq, true) => Predicate::Lte { column, value },
        (Operator::Gt, false) | (Operator::Lt, true) => Predicate::Gt { column, value },
        (Operator::GtEq, false) | (Operator::LtEq, true) => Predicate::Gte { column, value },
        _ => return None,
    };
    Some((col_name, pred))
}

/// Collect the conjuncts of a physical filter expression, splitting top-level
/// `AND`s (so a `BETWEEN`, lowered to `col >= lo AND col <= hi`, yields two
/// comparisons).  Any non-AND, non-comparison node aborts decoding.
fn collect_conjuncts(expr: &Arc<dyn PhysicalExpr>, out: &mut Vec<Arc<dyn PhysicalExpr>>) -> bool {
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::BinaryExpr;
    if let Some(bin) = expr.downcast_ref::<BinaryExpr>() {
        if matches!(bin.op(), Operator::And) {
            return collect_conjuncts(bin.left(), out) && collect_conjuncts(bin.right(), out);
        }
    }
    out.push(Arc::clone(expr));
    true
}

/// Decode the scan's pushed PHYSICAL filters into a single-column decidable
/// range/equality predicate set.  Returns `None` (caller falls back) when the
/// filters reference more than one column, contain a non-comparison conjunct, a
/// non-`(col OP lit)` shape, or a literal that cannot be compared to sidecar
/// stats.
fn decode_scan_filter(scan: &IcefallDBScanExec) -> Option<DecodedFilter> {
    let mut conjuncts: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
    for f in scan.filters() {
        if !collect_conjuncts(f, &mut conjuncts) {
            return None;
        }
    }
    if conjuncts.is_empty() {
        return None;
    }

    let mut column: Option<String> = None;
    let mut predicates: Vec<icefalldb_core::Predicate> = Vec::with_capacity(conjuncts.len());
    for c in &conjuncts {
        let (col, pred) = phys_comparison_to_predicate(c)?;
        match &column {
            None => column = Some(col),
            // Multi-column filters are not decidable from a single column's
            // [min,max]; fall back.
            Some(existing) if existing != &col => return None,
            _ => {}
        }
        predicates.push(pred);
    }

    Some(DecodedFilter {
        column: column.unwrap(),
        predicates,
    })
}

/// Coverage classification of one fragment against a decoded filter.
#[derive(PartialEq)]
enum Coverage {
    /// Every row provably passes `F`.
    Covered,
    /// Some rows may pass `F` (straddles the filter edge).
    Boundary,
    /// No row passes `F` (provably disjoint).
    Disjoint,
}

/// Classify a fragment against the decoded filter using ONLY sidecar min/max and
/// the null count for the filter column.
///
/// SOUNDNESS: a fragment is `Covered` only when, for EVERY predicate, the
/// fragment's `[min,max]` for the filter column lies entirely inside the
/// predicate's accepted range AND the column has `nulls == 0` in the fragment (a
/// NULL filter-column value fails any range/equality predicate, so a fragment
/// with nulls cannot be fully covered).  When min/max is missing, or any value
/// comparison is type-incompatible, we cannot prove coverage and conservatively
/// return `Boundary` (it will be scanned, never silently composed).
fn classify_fragment(rg: &PlannedRowGroup, filter: &DecodedFilter) -> Coverage {
    use icefalldb_core::predicate_eval::{fragment_disjoint, fragment_fully_covered};

    let Some(stats) = rg.meta.columns.get(&filter.column) else {
        // No sidecar stats for the filter column → cannot prove anything.
        return Coverage::Boundary;
    };

    // A null filter-column value never satisfies a range/equality predicate, so
    // any fragment with nulls in the filter column is at best BOUNDARY.
    let has_nulls = stats.nulls > 0;

    // Provably disjoint? (every row fails F) — these contribute nothing.  We only
    // classify disjoint when there are NO nulls ambiguity issues for the range,
    // but disjoint is sound regardless: if [min,max] is outside the accepted
    // range, no non-null row passes, and null rows never pass either.
    if fragment_disjoint(&filter.predicates, stats) {
        return Coverage::Disjoint;
    }

    if !has_nulls && fragment_fully_covered(&filter.predicates, stats) {
        return Coverage::Covered;
    }

    Coverage::Boundary
}

/// Entry point for a range/equality-filtered SUM / COUNT over a scan.  Returns
/// `None` on any precondition violation (caller falls back to the original plan).
fn compose_filtered_from_scan(
    aggr_exprs: &[Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>],
    output_schema: &arrow::datatypes::Schema,
    input_schema: &arrow::datatypes::Schema,
    scan: &IcefallDBScanExec,
    col_map: &HashMap<String, String>,
) -> Option<DFResult<Arc<dyn ExecutionPlan>>> {
    // Only a SINGLE aggregate, and only SUM(col) / COUNT(*) / COUNT(col).
    if aggr_exprs.len() != 1 || output_schema.fields().len() != 1 {
        return None;
    }
    let aggr = &aggr_exprs[0];
    if aggr.is_distinct() {
        return None;
    }
    let name = aggr.fun().name().to_lowercase();
    if aggr.expressions().len() != 1 || !matches!(name.as_str(), "sum" | "count") {
        return None;
    }

    // The numeric→float coercion path (col_map non-empty) only happens for
    // AVG/VAR/STDDEV; SUM/COUNT take the column directly.  If a projection alias
    // map is present we conservatively fall back (the filtered path is scoped to
    // the clean, partial-state-free SUM/COUNT case).
    if !col_map.is_empty() {
        return None;
    }

    // Decode the pushed filter into a single decidable range/equality predicate.
    let filter = decode_scan_filter(scan)?;

    // Classify every fragment.
    let rgs = scan.planned_row_groups();
    let mut covered: Vec<&PlannedRowGroup> = Vec::new();
    let mut boundary: Vec<PlannedRowGroup> = Vec::new();
    for rg in &rgs {
        if rg.fallback {
            // Unknown stats → cannot classify soundly.
            return None;
        }
        match classify_fragment(rg, &filter) {
            Coverage::Covered => covered.push(rg),
            Coverage::Boundary => boundary.push(rg.clone()),
            Coverage::Disjoint => { /* contributes nothing, skip */ }
        }
    }

    // No covered fragment → no benefit over the existing filtered scan.
    if covered.is_empty() {
        return None;
    }

    // Every covered fragment must be CLEAN with a trustworthy live partial: the
    // composed constant is read from `agg_state`, so it must match the on-disk
    // data exactly (deleted_count==0 AND content_hash==meta.checksum).
    for rg in &covered {
        if rg.deleted_count > 0 {
            return None;
        }
        let Some(agg_state) = &rg.agg_state else {
            return None;
        };
        if agg_state.content_hash != rg.meta.checksum {
            return None;
        }
    }

    // Compose the covered contribution as a constant scalar of the output type.
    let out_type = output_schema.field(0).data_type();
    let covered_const = match name.as_str() {
        "sum" => covered_sum_const(&aggr.expressions()[0], &covered, out_type)?,
        "count" => covered_count_const(&aggr.expressions()[0], &covered, out_type)?,
        _ => return None,
    };

    // Build the boundary aggregate + projection, or just the constant.
    let is_sum = name == "sum";
    Some(build_filtered_rewrite(
        aggr,
        output_schema,
        input_schema,
        scan,
        boundary,
        covered_const,
        out_type,
        is_sum,
    ))
}

/// Σ of the covered fragments' cached SUM(col) partial, as a scalar of `out_type`.
fn covered_sum_const(
    arg: &Arc<dyn PhysicalExpr>,
    covered: &[&PlannedRowGroup],
    out_type: &arrow::datatypes::DataType,
) -> Option<ScalarValue> {
    use arrow::datatypes::DataType;
    let col = arg.downcast_ref::<Column>()?;
    let col_name = col.name();

    // Determine int vs float from the first covered fragment's partial.
    let first = covered[0].agg_state.as_ref()?.cols.get(col_name)?;
    let is_int = matches!(&first.sum, AggScalar::Int(_) | AggScalar::Null);

    if is_int && matches!(out_type, DataType::Int64 | DataType::Int32) {
        let mut acc: i128 = 0;
        let mut any = false;
        for rg in covered {
            let c = rg.agg_state.as_ref()?.cols.get(col_name)?;
            match &c.sum {
                AggScalar::Int(s) => {
                    acc = acc.checked_add(*s)?;
                    any = true;
                }
                AggScalar::Null => {}
                AggScalar::Float(_) => return None,
            }
        }
        if !any {
            // Every covered fragment all-null contributes nothing to SUM.
            return ScalarValue::try_from(out_type).ok();
        }
        match out_type {
            DataType::Int64 => Some(ScalarValue::Int64(Some(i64::try_from(acc).ok()?))),
            DataType::Int32 => Some(ScalarValue::Int32(Some(i32::try_from(acc).ok()?))),
            _ => None,
        }
    } else if matches!(out_type, DataType::Float64 | DataType::Float32) {
        let mut acc: f64 = 0.0;
        for rg in covered {
            let c = rg.agg_state.as_ref()?.cols.get(col_name)?;
            match &c.sum {
                AggScalar::Float(s) => acc += s,
                AggScalar::Int(s) => acc += *s as f64,
                AggScalar::Null => {}
            }
        }
        match out_type {
            DataType::Float64 => Some(ScalarValue::Float64(Some(acc))),
            DataType::Float32 => Some(ScalarValue::Float32(Some(acc as f32))),
            _ => None,
        }
    } else {
        None
    }
}

/// Σ of the covered fragments' row contribution for COUNT, as a scalar of
/// `out_type` (always `Int64` for COUNT in DataFusion).
///
/// COUNT(*)  → Σ meta.rows (covered fragments are clean → no deletions).
/// COUNT(col)→ Σ count_non_null from the cached partial.
fn covered_count_const(
    arg: &Arc<dyn PhysicalExpr>,
    covered: &[&PlannedRowGroup],
    out_type: &arrow::datatypes::DataType,
) -> Option<ScalarValue> {
    if !matches!(out_type, arrow::datatypes::DataType::Int64) {
        return None;
    }
    if arg.downcast_ref::<Literal>().is_some() {
        // COUNT(*) — every covered row passes F (covered means clean, no deletes).
        let total: i128 = covered.iter().map(|rg| rg.meta.rows as i128).sum();
        return Some(ScalarValue::Int64(Some(i64::try_from(total).ok()?)));
    }
    let col = arg.downcast_ref::<Column>()?;
    let col_name = col.name();
    let mut total: i128 = 0;
    for rg in covered {
        let c = rg.agg_state.as_ref()?.cols.get(col_name)?;
        total = total.checked_add(c.count_non_null as i128)?;
    }
    Some(ScalarValue::Int64(Some(i64::try_from(total).ok()?)))
}

/// Build the final rewrite: either a single-row constant source (no boundary
/// fragments), or `ProjectionExec[ boundary_agg + covered_const ](
/// AggregateExec(SUM/COUNT)( boundary_scan ) )` matching `output_schema` exactly.
#[allow(clippy::too_many_arguments)]
fn build_filtered_rewrite(
    aggr: &Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
    output_schema: &arrow::datatypes::Schema,
    input_schema: &arrow::datatypes::Schema,
    scan: &IcefallDBScanExec,
    boundary: Vec<PlannedRowGroup>,
    covered_const: ScalarValue,
    out_type: &arrow::datatypes::DataType,
    is_sum: bool,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{is_null, BinaryExpr, CaseExpr, CastExpr};
    use datafusion::physical_plan::aggregates::PhysicalGroupBy;

    let output_schema_ref: SchemaRef = Arc::new(output_schema.clone());
    let out_field = output_schema.field(0);

    // No boundary fragments: the whole answer is the covered constant.
    if boundary.is_empty() {
        let array = covered_const.to_array_of_size(1)?;
        let batch = RecordBatch::try_new(Arc::clone(&output_schema_ref), vec![array])?;
        return MemorySourceConfig::try_new_from_batches(output_schema_ref, vec![batch])
            .map(|e| e as Arc<dyn ExecutionPlan>);
    }

    // Build the boundary scan (same filter F, only boundary fragments).  A
    // multi-fragment boundary scan exposes one partition per fragment; coalesce
    // them into a single partition so the Single-mode aggregate emits exactly one
    // row (a per-partition Single aggregate would emit one row per partition).
    let boundary_scan = scan
        .with_planned_row_groups(boundary)
        .map_err(|e| datafusion::error::DataFusionError::External(Box::new(e)))?;
    let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(
        boundary_scan as Arc<dyn ExecutionPlan>,
    ));

    // Build a fresh single-phase AggregateExec(SUM|COUNT, group=[]) over the
    // coalesced boundary scan, re-creating the aggregate against the boundary
    // scan's (identical) input schema.
    let boundary_input_schema: SchemaRef = Arc::new(input_schema.clone());
    let rebuilt_aggr = rebuild_aggregate_against(aggr, &boundary_input_schema)?;
    let aggr_arc = Arc::new(rebuilt_aggr);
    let boundary_agg = Arc::new(AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::default(),
        vec![Arc::clone(&aggr_arc)],
        vec![None],
        coalesced,
        Arc::clone(&boundary_input_schema),
    )?);

    // The boundary aggregate's single output column (index 0) holds the boundary
    // partial.  Its DataType is the aggregate's own output type, which equals the
    // final output type for SUM/COUNT(group=[]).
    let agg_out_schema = boundary_agg.schema();
    let boundary_col: Arc<dyn PhysicalExpr> =
        Arc::new(Column::new(agg_out_schema.field(0).name(), 0));
    let agg_out_type = agg_out_schema.field(0).data_type().clone();

    // Combine the boundary partial with the covered constant.  COUNT never
    // produces NULL and the covered constant is a concrete count, so plain `+`
    // is exact.  SUM, however, returns NULL over an empty/all-null input, and
    // `NULL + const` would WRONGLY discard the covered contribution.  We honour
    // SQL SUM null-semantics exactly:
    //   * covered_const is NULL (every covered `v` is NULL) → the answer is just
    //     the boundary SUM (NULL iff the boundary is also empty/all-null).
    //   * covered_const is non-NULL → the answer is non-NULL; treat a NULL
    //     boundary SUM as the additive identity via
    //     `CASE WHEN boundary IS NULL THEN const ELSE boundary + const END`.
    let const_expr: Arc<dyn PhysicalExpr> = Arc::new(Literal::new(covered_const.clone()));
    let mut combined: Arc<dyn PhysicalExpr> = if is_sum {
        if covered_const.is_null() {
            // Pass the boundary SUM through unchanged.
            boundary_col
        } else {
            let when_null = is_null(Arc::clone(&boundary_col))?;
            let plus: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
                boundary_col,
                Operator::Plus,
                Arc::clone(&const_expr),
            ));
            Arc::new(CaseExpr::try_new(
                None,
                vec![(when_null, Arc::clone(&const_expr))],
                Some(plus),
            )?)
        }
    } else {
        // COUNT: boundary count (never NULL) + covered count.
        Arc::new(BinaryExpr::new(boundary_col, Operator::Plus, const_expr))
    };
    if &agg_out_type != out_type {
        combined = Arc::new(CastExpr::new(combined, out_type.clone(), None));
    }

    let projection = Arc::new(ProjectionExec::try_new(
        vec![(combined, out_field.name().to_string())],
        boundary_agg as Arc<dyn ExecutionPlan>,
    )?);

    // Defend the byte-equal contract: the projection's output field must match
    // the original final aggregate's output field (name + type) exactly.
    let proj_schema = projection.schema();
    if proj_schema.fields().len() != 1
        || proj_schema.field(0).data_type() != out_type
        || proj_schema.field(0).name() != out_field.name()
    {
        return Err(datafusion::error::DataFusionError::Internal(
            "partial-aggregate pushdown produced a mismatched output schema".into(),
        ));
    }

    Ok(projection as Arc<dyn ExecutionPlan>)
}

/// Re-create an aggregate function expression against a new (identical) input
/// schema so it can sit on top of the reconstructed boundary scan.
fn rebuild_aggregate_against(
    aggr: &Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
    schema: &SchemaRef,
) -> DFResult<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    AggregateExprBuilder::new(Arc::new(aggr.fun().clone()), aggr.expressions().to_vec())
        .schema(Arc::clone(schema))
        .alias(aggr.name().to_string())
        .build()
}

fn compute_count(arg: &Arc<dyn PhysicalExpr>, scan: &IcefallDBScanExec) -> DFResult<ScalarValue> {
    let rgs = scan.planned_row_groups();
    if let Some(col) = arg.downcast_ref::<Column>() {
        // COUNT(col): for clean fragments use meta null counts; for dirty fragments
        // with a live agg_state use the retracted count_non_null (exact).  If any
        // fragment is dirty without a live partial, fall back.
        for rg in &rgs {
            if rg.fallback {
                return Err(QueryError::StatsUnavailable.into());
            }
            if rg.deleted_count > 0 {
                // Dirty fragment: require a live partial with the column present.
                let Some(agg_state) = &rg.agg_state else {
                    return Err(QueryError::StatsUnavailable.into());
                };
                if !agg_state.cols.contains_key(col.name()) {
                    return Err(QueryError::StatsUnavailable.into());
                }
            }
        }
        let mut count: i64 = 0;
        for rg in &rgs {
            if rg.deleted_count > 0 {
                // Live partial is guaranteed present and has the column (checked above).
                let col_agg = rg.agg_state.as_ref().unwrap().cols.get(col.name()).unwrap();
                count += col_agg.count_non_null as i64;
            } else {
                let nulls = rg
                    .meta
                    .columns
                    .get(col.name())
                    .map(|s| s.nulls)
                    .ok_or_else(|| QueryError::StatsUnavailable)?;
                count += rg.meta.rows.saturating_sub(nulls) as i64;
            }
        }
        Ok(ScalarValue::Int64(Some(count)))
    } else if arg.downcast_ref::<Literal>().is_some() {
        // COUNT(*) lowered to COUNT(1).
        // Subtract deleted rows from each fragment's row count — zero I/O because
        // `deleted_count` comes from the manifest, not the Parquet data files.
        let total: usize = rgs
            .iter()
            .map(|rg| rg.meta.rows.saturating_sub(rg.deleted_count as usize))
            .sum();
        Ok(ScalarValue::Int64(Some(total as i64)))
    } else {
        Err(QueryError::StatsUnavailable.into())
    }
}

fn compute_min(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
    input_schema: &arrow::datatypes::Schema,
) -> DFResult<ScalarValue> {
    let col = arg
        .downcast_ref::<Column>()
        .ok_or(QueryError::StatsUnavailable)?;
    let data_type = input_schema
        .field_with_name(col.name())
        .map_err(|_| QueryError::StatsUnavailable)?
        .data_type()
        .clone();

    // Sidecar statistics for floats skip non-finite values, so they cannot be
    // trusted to answer MIN/MAX for the actual column data.
    if matches!(
        data_type,
        arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64
    ) {
        return Err(QueryError::StatsUnavailable.into());
    }

    let mut current: Option<ScalarValue> = None;
    for rg in scan.planned_row_groups() {
        if rg.fallback {
            return Err(QueryError::StatsUnavailable.into());
        }
        if rg.deleted_count > 0 {
            // Dirty fragment: use live_min_json if available (set by scan_internal).
            let Some(agg_state) = &rg.agg_state else {
                return Err(QueryError::StatsUnavailable.into());
            };
            let Some(col_agg) = agg_state.cols.get(col.name()) else {
                return Err(QueryError::StatsUnavailable.into());
            };
            let Some(min_json) = &col_agg.live_min_json else {
                return Err(QueryError::StatsUnavailable.into());
            };
            let rg_min =
                json_to_scalar_value(min_json, &data_type).ok_or(QueryError::StatsUnavailable)?;
            current = Some(match current {
                Some(c) => pick_min(&c, &rg_min),
                None => rg_min,
            });
            continue;
        }
        let stats = rg
            .meta
            .columns
            .get(col.name())
            .ok_or(QueryError::StatsUnavailable)?;
        let min_json = stats.min.as_ref().ok_or(QueryError::StatsUnavailable)?;
        let rg_min =
            json_to_scalar_value(min_json, &data_type).ok_or(QueryError::StatsUnavailable)?;
        current = Some(match current {
            Some(c) => pick_min(&c, &rg_min),
            None => rg_min,
        });
    }
    current.ok_or_else(|| QueryError::StatsUnavailable.into())
}

fn compute_max(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
    input_schema: &arrow::datatypes::Schema,
) -> DFResult<ScalarValue> {
    let col = arg
        .downcast_ref::<Column>()
        .ok_or(QueryError::StatsUnavailable)?;
    let data_type = input_schema
        .field_with_name(col.name())
        .map_err(|_| QueryError::StatsUnavailable)?
        .data_type()
        .clone();

    // Sidecar statistics for floats skip non-finite values, so they cannot be
    // trusted to answer MIN/MAX for the actual column data.
    if matches!(
        data_type,
        arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64
    ) {
        return Err(QueryError::StatsUnavailable.into());
    }

    let mut current: Option<ScalarValue> = None;
    for rg in scan.planned_row_groups() {
        if rg.fallback {
            return Err(QueryError::StatsUnavailable.into());
        }
        if rg.deleted_count > 0 {
            // Dirty fragment: use live_max_json if available (set by scan_internal).
            let Some(agg_state) = &rg.agg_state else {
                return Err(QueryError::StatsUnavailable.into());
            };
            let Some(col_agg) = agg_state.cols.get(col.name()) else {
                return Err(QueryError::StatsUnavailable.into());
            };
            let Some(max_json) = &col_agg.live_max_json else {
                return Err(QueryError::StatsUnavailable.into());
            };
            let rg_max =
                json_to_scalar_value(max_json, &data_type).ok_or(QueryError::StatsUnavailable)?;
            current = Some(match current {
                Some(c) => pick_max(&c, &rg_max),
                None => rg_max,
            });
            continue;
        }
        let stats = rg
            .meta
            .columns
            .get(col.name())
            .ok_or(QueryError::StatsUnavailable)?;
        let max_json = stats.max.as_ref().ok_or(QueryError::StatsUnavailable)?;
        let rg_max =
            json_to_scalar_value(max_json, &data_type).ok_or(QueryError::StatsUnavailable)?;
        current = Some(match current {
            Some(c) => pick_max(&c, &rg_max),
            None => rg_max,
        });
    }
    current.ok_or_else(|| QueryError::StatsUnavailable.into())
}

fn pick_min(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    match a.partial_cmp(b) {
        Some(std::cmp::Ordering::Greater) => b.clone(),
        _ => a.clone(),
    }
}

fn pick_max(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    match a.partial_cmp(b) {
        Some(std::cmp::Ordering::Less) => b.clone(),
        _ => a.clone(),
    }
}

/// Check preconditions for all fragments and collect per-column partial sums.
///
/// Returns `(n, S, Q)` where `n` is the total non-null count and `S`/`Q` are
/// the accumulated sum and sum-of-squares as `AggScalar`.
///
/// Preconditions checked per fragment:
/// 1. `!rg.fallback`
/// 2. `rg.agg_state.is_some()` (the live partial must have been stored by scan_internal)
/// 3. For CLEAN fragments (`rg.deleted_count == 0`): `agg_state.content_hash == rg.meta.checksum`.
///    For DIRTY fragments the live partial was produced by `scan_internal` via
///    `retract(full, deleted_contribution(…))`, so a present `agg_state` is sufficient.
/// 4. `col_name` is present in `agg_state.cols`
///
/// Returns `Err(StatsUnavailable)` on any violation so the caller falls back.
fn collect_col_partials(
    col_name: &str,
    scan: &IcefallDBScanExec,
) -> DFResult<(u64, AggScalar, AggScalar)> {
    let rgs = scan.planned_row_groups();

    // First pass: validate preconditions.
    for rg in &rgs {
        if rg.fallback {
            return Err(QueryError::StatsUnavailable.into());
        }
        let Some(agg_state) = &rg.agg_state else {
            return Err(QueryError::StatsUnavailable.into());
        };
        // For clean fragments, verify the content hash guards against a stale sidecar.
        // For dirty fragments the live partial was computed by scan_internal from the
        // trusted full partial + the current deletion vector; a present agg_state is
        // sufficient.
        if rg.deleted_count == 0 && agg_state.content_hash != rg.meta.checksum {
            return Err(QueryError::StatsUnavailable.into());
        }
        if !agg_state.cols.contains_key(col_name) {
            return Err(QueryError::StatsUnavailable.into());
        }
    }

    if rgs.is_empty() {
        return Ok((0, AggScalar::Null, AggScalar::Null));
    }

    // Second pass: accumulate.  All sum/sumsq scalars must be the same variant
    // (all Int or all Float).  A mismatch falls back defensively.
    let mut total_n: u64 = 0;

    // Detect int vs float from the first fragment's ColAgg.
    let first_col = rgs[0]
        .agg_state
        .as_ref()
        .unwrap()
        .cols
        .get(col_name)
        .unwrap();
    let is_int = matches!(&first_col.sum, AggScalar::Int(_));

    if is_int {
        let mut sum_i: i128 = 0;
        let mut sumsq_i: i128 = 0;
        for rg in &rgs {
            let col = rg.agg_state.as_ref().unwrap().cols.get(col_name).unwrap();
            total_n += col.count_non_null;
            match (&col.sum, &col.sumsq) {
                (AggScalar::Int(s), AggScalar::Int(q)) => {
                    sum_i = sum_i.checked_add(*s).ok_or(QueryError::StatsUnavailable)?;
                    sumsq_i = sumsq_i
                        .checked_add(*q)
                        .ok_or(QueryError::StatsUnavailable)?;
                }
                // All-null fragment: contributes zero to sums.
                (AggScalar::Null, AggScalar::Null) => {}
                _ => return Err(QueryError::StatsUnavailable.into()),
            }
        }
        Ok((total_n, AggScalar::Int(sum_i), AggScalar::Int(sumsq_i)))
    } else {
        let mut sum_f: f64 = 0.0;
        let mut sumsq_f: f64 = 0.0;
        for rg in &rgs {
            let col = rg.agg_state.as_ref().unwrap().cols.get(col_name).unwrap();
            total_n += col.count_non_null;
            match (&col.sum, &col.sumsq) {
                (AggScalar::Float(s), AggScalar::Float(q)) => {
                    sum_f += s;
                    sumsq_f += q;
                }
                (AggScalar::Null, AggScalar::Null) => {}
                _ => return Err(QueryError::StatsUnavailable.into()),
            }
        }
        Ok((total_n, AggScalar::Float(sum_f), AggScalar::Float(sumsq_f)))
    }
}

fn compose_sum(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
    output_schema: &arrow::datatypes::Schema,
    output_idx: usize,
) -> DFResult<ScalarValue> {
    let col = arg
        .downcast_ref::<Column>()
        .ok_or(QueryError::StatsUnavailable)?;
    let (_n, sum, _sumsq) = collect_col_partials(col.name(), scan)?;

    let out_type = output_schema.field(output_idx).data_type();
    match (sum, out_type) {
        (AggScalar::Int(s), arrow::datatypes::DataType::Int64) => {
            let v = i64::try_from(s).map_err(|_| QueryError::StatsUnavailable)?;
            Ok(ScalarValue::Int64(Some(v)))
        }
        (AggScalar::Int(s), arrow::datatypes::DataType::Int32) => {
            let v = i32::try_from(s).map_err(|_| QueryError::StatsUnavailable)?;
            Ok(ScalarValue::Int32(Some(v)))
        }
        (AggScalar::Float(s), arrow::datatypes::DataType::Float64) => {
            Ok(ScalarValue::Float64(Some(s)))
        }
        (AggScalar::Float(s), arrow::datatypes::DataType::Float32) => {
            Ok(ScalarValue::Float32(Some(s as f32)))
        }
        (AggScalar::Null, dt) => {
            // All-null column: SUM is NULL with the appropriate type.
            Ok(ScalarValue::try_from(dt).map_err(|_| QueryError::StatsUnavailable)?)
        }
        _ => Err(QueryError::StatsUnavailable.into()),
    }
}

/// Resolve a `Column` physical expression's name through an optional alias map.
///
/// When DataFusion 54 inserts a `ProjectionExec` for numeric→float coercion, the
/// aggregate argument is `Column("__common_expr_N")` — an alias of the original
/// column in the projection.  `col_map` contains the alias→original mapping
/// collected by `peel_to_scan`.  If `arg` is not a bare `Column`, or the alias is
/// not in `col_map` (and the map is non-empty), returns `None`.
fn resolve_col_name(
    arg: &Arc<dyn PhysicalExpr>,
    col_map: &HashMap<String, String>,
) -> Option<String> {
    // Unwrap a numeric→float coercion cast inlined in the aggregate argument
    // (e.g. single-phase `AVG(int_col)` whose arg is `Cast(Column → Float64)`,
    // not lifted into a separate ProjectionExec).  Only float-target casts of a
    // bare Column are accepted, matching `peel_to_scan`'s projection policy.
    if let Some(cast) = arg.downcast_ref::<CastExpr>() {
        if !matches!(
            cast.cast_type(),
            arrow::datatypes::DataType::Float32 | arrow::datatypes::DataType::Float64
        ) {
            return None;
        }
        return resolve_col_name(cast.expr(), col_map);
    }
    let col = arg.downcast_ref::<Column>()?;
    if col_map.is_empty() {
        return Some(col.name().to_string());
    }
    col_map.get(col.name()).cloned()
}

fn compose_avg(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
    output_schema: &arrow::datatypes::Schema,
    output_idx: usize,
    col_map: &HashMap<String, String>,
) -> DFResult<ScalarValue> {
    let col_name = resolve_col_name(arg, col_map).ok_or(QueryError::StatsUnavailable)?;
    let (n, sum, _sumsq) = collect_col_partials(&col_name, scan)?;

    let out_type = output_schema.field(output_idx).data_type();
    // AVG always returns Float64 in DataFusion.
    if !matches!(out_type, arrow::datatypes::DataType::Float64) {
        return Err(QueryError::StatsUnavailable.into());
    }

    if n == 0 {
        return Ok(ScalarValue::Float64(None));
    }

    let s_f64 = match sum {
        AggScalar::Int(s) => s as f64,
        AggScalar::Float(s) => s,
        AggScalar::Null => return Ok(ScalarValue::Float64(None)),
    };
    Ok(ScalarValue::Float64(Some(s_f64 / n as f64)))
}

fn compose_variance(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
    output_schema: &arrow::datatypes::Schema,
    output_idx: usize,
    population: bool,
    col_map: &HashMap<String, String>,
) -> DFResult<ScalarValue> {
    let col_name = resolve_col_name(arg, col_map).ok_or(QueryError::StatsUnavailable)?;
    let (n, sum, sumsq) = collect_col_partials(&col_name, scan)?;

    let out_type = output_schema.field(output_idx).data_type();
    if !matches!(out_type, arrow::datatypes::DataType::Float64) {
        return Err(QueryError::StatsUnavailable.into());
    }

    let denom = if population {
        if n == 0 {
            return Ok(ScalarValue::Float64(None));
        }
        n as f64
    } else {
        if n < 2 {
            return Ok(ScalarValue::Float64(None));
        }
        (n - 1) as f64
    };

    let s_f64 = match sum {
        AggScalar::Int(s) => s as f64,
        AggScalar::Float(s) => s,
        AggScalar::Null => return Ok(ScalarValue::Float64(None)),
    };
    let q_f64 = match sumsq {
        AggScalar::Int(q) => q as f64,
        AggScalar::Float(q) => q,
        AggScalar::Null => return Ok(ScalarValue::Float64(None)),
    };

    let n_f = n as f64;
    // Clamp tiny negatives due to floating-point reassociation to zero.
    let variance = ((q_f64 - s_f64 * s_f64 / n_f) / denom).max(0.0);
    Ok(ScalarValue::Float64(Some(variance)))
}

fn compose_stddev(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
    output_schema: &arrow::datatypes::Schema,
    output_idx: usize,
    population: bool,
    col_map: &HashMap<String, String>,
) -> DFResult<ScalarValue> {
    let v = compose_variance(arg, scan, output_schema, output_idx, population, col_map)?;
    match v {
        ScalarValue::Float64(Some(var)) => Ok(ScalarValue::Float64(Some(var.sqrt()))),
        ScalarValue::Float64(None) => Ok(ScalarValue::Float64(None)),
        _ => Err(QueryError::StatsUnavailable.into()),
    }
}

// ── warm approx_distinct composition ───────────────────────────────────

/// Answer `approx_distinct(col)` from cached per-fragment CPC sketches.
///
/// Preconditions (any violation → `Err(StatsUnavailable)` → caller falls back):
/// 1. Every fragment must be CLEAN (`deleted_count == 0`).  Dirty fragments
///    over-count (the sketch includes deleted rows), so we fall back to an exact
///    scan.  Deletion-aware handling (Theta A-not-B) is deferred.
/// 2. Every fragment must have a `distinct` sketch map with the requested column.
/// 3. Content hash must match for each clean fragment (stale-sidecar guard).
#[cfg(feature = "sketches")]
fn compose_approx_distinct(
    arg: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
) -> DFResult<ScalarValue> {
    use icefalldb_core::agg_cache::cpc_distinct_estimate;

    let col = arg
        .downcast_ref::<Column>()
        .ok_or(QueryError::StatsUnavailable)?;
    let col_name = col.name();
    let rgs = scan.planned_row_groups();

    // Collect sketch maps; any missing sketch or dirty fragment → fall back.
    let mut sketch_maps: Vec<&std::collections::BTreeMap<String, Vec<u8>>> =
        Vec::with_capacity(rgs.len());
    for rg in &rgs {
        if rg.fallback {
            return Err(QueryError::StatsUnavailable.into());
        }
        // Dirty fragment: sketch over-counts deleted rows → fall back.
        if rg.deleted_count > 0 {
            return Err(QueryError::StatsUnavailable.into());
        }
        let Some(agg_state) = &rg.agg_state else {
            return Err(QueryError::StatsUnavailable.into());
        };
        // Stale-sidecar guard.
        if agg_state.content_hash != rg.meta.checksum {
            return Err(QueryError::StatsUnavailable.into());
        }
        let Some(distinct_map) = &agg_state.distinct else {
            return Err(QueryError::StatsUnavailable.into());
        };
        sketch_maps.push(distinct_map);
    }

    if rgs.is_empty() {
        return Ok(ScalarValue::UInt64(Some(0)));
    }

    let estimate =
        cpc_distinct_estimate(&sketch_maps, col_name).ok_or(QueryError::StatsUnavailable)?;
    Ok(ScalarValue::UInt64(Some(estimate.round() as u64)))
}

// ── warm approx_percentile_cont composition ────────────────────────────

/// Answer `approx_percentile_cont(col, q)` from cached per-fragment T-Digest sketches.
///
/// Preconditions (any violation → `Err(StatsUnavailable)` → caller falls back):
/// 1. `arg_q` must be a `Literal` containing a `Float64` or `Float32` value in [0, 1].
///    If `q` is not a literal in [0, 1] → fall back (runtime quantile values are
///    not supported — the sketch is queried with a fixed `q` at plan time).
/// 2. Every fragment must be CLEAN (`deleted_count == 0`).  Dirty fragments
///    over-count deleted rows in the sketch, so we fall back to an exact scan.
///    Deletion-aware handling is deferred.
/// 3. Every fragment must have a `quantile` sketch map with the requested column.
/// 4. Content hash must match for each clean fragment (stale-sidecar guard).
///
/// Sketch: T-Digest k=200 (datasketches default).  Error type: rank error, i.e.
/// the rank of the returned value is within ~1-2% of `q`; we assert ≤3% in tests.
/// The returned value is a `Float64` scalar.
#[cfg(feature = "sketches")]
fn compose_approx_percentile(
    arg_col: &Arc<dyn PhysicalExpr>,
    arg_q: &Arc<dyn PhysicalExpr>,
    scan: &IcefallDBScanExec,
) -> DFResult<ScalarValue> {
    use icefalldb_core::agg_cache::tdigest_quantile_estimate;

    // Extract the column name.
    let col = arg_col
        .downcast_ref::<Column>()
        .ok_or(QueryError::StatsUnavailable)?;
    let col_name = col.name();

    // Extract the quantile literal — must be a Float64 or Float32 in [0, 1].
    let lit = arg_q
        .downcast_ref::<Literal>()
        .ok_or(QueryError::StatsUnavailable)?;
    let q: f64 = match lit.value() {
        ScalarValue::Float64(Some(v)) => *v,
        ScalarValue::Float32(Some(v)) => *v as f64,
        _ => return Err(QueryError::StatsUnavailable.into()),
    };
    if !(0.0..=1.0).contains(&q) {
        return Err(QueryError::StatsUnavailable.into());
    }

    let rgs = scan.planned_row_groups();

    // Collect quantile sketch maps; any missing sketch or dirty fragment → fall back.
    let mut sketch_maps: Vec<&std::collections::BTreeMap<String, Vec<u8>>> =
        Vec::with_capacity(rgs.len());
    for rg in &rgs {
        if rg.fallback {
            return Err(QueryError::StatsUnavailable.into());
        }
        // Dirty fragment: sketch includes deleted rows → fall back.
        if rg.deleted_count > 0 {
            return Err(QueryError::StatsUnavailable.into());
        }
        let Some(agg_state) = &rg.agg_state else {
            return Err(QueryError::StatsUnavailable.into());
        };
        // Stale-sidecar guard.
        if agg_state.content_hash != rg.meta.checksum {
            return Err(QueryError::StatsUnavailable.into());
        }
        let Some(quantile_map) = &agg_state.quantile else {
            return Err(QueryError::StatsUnavailable.into());
        };
        sketch_maps.push(quantile_map);
    }

    if rgs.is_empty() {
        return Ok(ScalarValue::Float64(None));
    }

    let estimate =
        tdigest_quantile_estimate(&sketch_maps, col_name, q).ok_or(QueryError::StatsUnavailable)?;
    Ok(ScalarValue::Float64(Some(estimate)))
}

// ── warm GROUP BY composition ─────────────────────────────────────────

/// Resolve the single group-by key column name from a [`PhysicalGroupBy`],
/// through the optional coercion-projection alias map.
///
/// Returns `None` (caller falls back) when:
/// - the group-by uses GROUPING SETS / ROLLUP / CUBE (`has_grouping_set`),
/// - there is not EXACTLY one group expression,
/// - the single group expression is not a bare `Column`, or
/// - the alias is not in `col_map` (when the map is non-empty).
fn single_group_key_name(
    group_by: &PhysicalGroupBy,
    col_map: &HashMap<String, String>,
) -> Option<String> {
    if group_by.has_grouping_set() {
        return None;
    }
    let exprs = group_by.expr();
    if exprs.len() != 1 {
        return None;
    }
    resolve_col_name(&exprs[0].0, col_map)
}

/// Entry point for grouped aggregates.  Handles both the single-phase grouped
/// plan and the two-phase HASH grouped plan
/// (`FinalPartitioned → RepartitionExec(Hash[k]) → Partial`).  Returns the
/// original `plan` unchanged on any unsupported shape (correctness over
/// coverage).
fn transform_grouped_aggregate(
    plan: &Arc<dyn ExecutionPlan>,
    agg: &AggregateExec,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    match agg.mode() {
        // Single-phase grouped: AggregateExec(Single, group=[k]) → [proj] → scan.
        AggregateMode::Single | AggregateMode::SinglePartitioned => {
            let filter_exprs = agg.filter_expr();
            if agg.aggr_expr().len() != filter_exprs.len()
                || filter_exprs.iter().any(|f| f.is_some())
            {
                return Ok(Arc::clone(plan));
            }
            let Some((scan_arc, col_map)) = peel_to_scan(agg.input()) else {
                return Ok(Arc::clone(plan));
            };
            let scan = match scan_arc.downcast_ref::<IcefallDBScanExec>() {
                Some(s) if s.filters().is_empty() => s,
                _ => return Ok(Arc::clone(plan)),
            };
            let Some(key_name) = single_group_key_name(agg.group_expr(), &col_map) else {
                return Ok(Arc::clone(plan));
            };
            match compose_grouped_from_scan(
                &key_name,
                agg.aggr_expr(),
                &agg.schema(),
                &agg.input_schema(),
                scan,
                &col_map,
            ) {
                Some(result) => result,
                None => Ok(Arc::clone(plan)),
            }
        }

        // Two-phase HASH grouped: FinalPartitioned/Final → RepartitionExec(Hash[k])
        // → Partial(group=[k]) → [proj] → scan.  Take the group key + measures
        // from the PARTIAL, the output schema from the FINAL.
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            let partial_candidate = find_inner_input(agg.input());
            let Some(partial_agg) = partial_candidate.downcast_ref::<AggregateExec>() else {
                return Ok(Arc::clone(plan));
            };
            if !matches!(partial_agg.mode(), AggregateMode::Partial) {
                return Ok(Arc::clone(plan));
            }
            // The Partial must also be grouped (mirrors the Final).
            if partial_agg.group_expr().is_empty() {
                return Ok(Arc::clone(plan));
            }
            let filter_exprs = partial_agg.filter_expr();
            if partial_agg.aggr_expr().len() != filter_exprs.len()
                || filter_exprs.iter().any(|f| f.is_some())
            {
                return Ok(Arc::clone(plan));
            }
            let Some((scan_arc, col_map)) = peel_to_scan(partial_agg.input()) else {
                return Ok(Arc::clone(plan));
            };
            let scan = match scan_arc.downcast_ref::<IcefallDBScanExec>() {
                Some(s) if s.filters().is_empty() => s,
                _ => return Ok(Arc::clone(plan)),
            };
            let Some(key_name) = single_group_key_name(partial_agg.group_expr(), &col_map) else {
                return Ok(Arc::clone(plan));
            };
            match compose_grouped_from_scan(
                &key_name,
                partial_agg.aggr_expr(),
                &agg.schema(),
                &partial_agg.input_schema(),
                scan,
                &col_map,
            ) {
                Some(result) => result,
                None => Ok(Arc::clone(plan)),
            }
        }

        _ => Ok(Arc::clone(plan)),
    }
}

/// Compose a warm GROUP BY result from the scan's cached per-group partials.
///
/// `key_name` is the resolved group-by key column.  `aggr_exprs` are the
/// measure aggregates (SUM/COUNT/AVG/VAR/STDDEV over single columns).
/// `output_schema` is the FINAL aggregate's output schema: column 0 is the
/// group key, columns 1..N are the aggregates in `aggr_exprs` order.
///
/// Returns `None` on any precondition violation (caller falls back).  Builds a
/// multi-row `MemorySourceConfig` (one row per surviving group) on success.
fn compose_grouped_from_scan(
    key_name: &str,
    aggr_exprs: &[Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>],
    output_schema: &arrow::datatypes::Schema,
    input_schema: &arrow::datatypes::Schema,
    scan: &IcefallDBScanExec,
    col_map: &HashMap<String, String>,
) -> Option<DFResult<Arc<dyn ExecutionPlan>>> {
    // 1. The key must be a DECLARED agg_group_key of the scan.
    if !scan.agg_group_keys().iter().any(|k| k == key_name) {
        return None;
    }

    // 2. The output schema must be: [key, agg_0, agg_1, …] — exactly one group
    //    key column followed by one column per aggregate, in order.
    if output_schema.fields().len() != aggr_exprs.len() + 1 {
        return None;
    }

    // 3. Validate every aggregate is supported and over a single column.
    for aggr in aggr_exprs {
        if aggr.is_distinct() {
            return None;
        }
        let name = aggr.fun().name().to_lowercase();
        let supported = [
            "count",
            "sum",
            "avg",
            "var",
            "var_pop",
            "stddev",
            "stddev_pop",
        ];
        if !supported.contains(&name.as_str()) || aggr.expressions().len() != 1 {
            return None;
        }
    }

    // 4. Resolve the key column's Arrow data type for decoding group keys.
    let key_type = input_schema
        .field_with_name(key_name)
        .ok()?
        .data_type()
        .clone();

    // 5. Collect & merge the per-fragment LIVE grouped partials.  Every
    //    referenced fragment must have grouped partials keyed by `key_name`;
    //    clean fragments must have a matching content hash.
    let rgs = scan.planned_row_groups();
    let mut owned: Vec<GroupedPartials> = Vec::with_capacity(rgs.len());
    for rg in &rgs {
        if rg.fallback {
            return None;
        }
        let agg_state = rg.agg_state.as_ref()?;
        // Clean fragments: guard against a stale sidecar.  Dirty fragments'
        // grouped partials were made live by scan_internal.
        if rg.deleted_count == 0 && agg_state.content_hash != rg.meta.checksum {
            return None;
        }
        let gp = agg_state.grouped()?;
        if gp.key_col != key_name {
            return None;
        }
        owned.push(gp.clone());
    }

    let merged: BTreeMap<String, BTreeMap<String, ColAgg>> = if owned.is_empty() {
        BTreeMap::new()
    } else {
        let refs: Vec<&GroupedPartials> = owned.iter().collect();
        merge_grouped(&refs)
    };

    // 6. Build one output row per surviving group.
    let mut group_keys: Vec<ScalarValue> = Vec::with_capacity(merged.len());
    // Per output aggregate column, the column of scalars across groups.
    let mut agg_columns: Vec<Vec<ScalarValue>> =
        vec![Vec::with_capacity(merged.len()); aggr_exprs.len()];

    for (group_key_str, group_cols) in &merged {
        // Decode the canonical-JSON group key back to a ScalarValue.
        let key_scalar = decode_group_key(group_key_str, &key_type)?;

        // Compute each aggregate for this group; track whether the group has any
        // surviving rows (total count > 0) so empty groups are dropped.
        let mut row_values: Vec<ScalarValue> = Vec::with_capacity(aggr_exprs.len());
        let mut max_count: u64 = 0;
        for (i, aggr) in aggr_exprs.iter().enumerate() {
            // output field is at i+1 (column 0 is the group key).
            let out_field_type = output_schema.field(i + 1).data_type();
            let (value, count) =
                compose_aggregate_for_group(aggr, group_cols, out_field_type, col_map)?;
            max_count = max_count.max(count);
            row_values.push(value);
        }

        // A group with zero total non-null measure rows is AMBIGUOUS from the
        // per-measure partials alone: it could be a clean group whose every
        // measure value is NULL (a full GROUP BY scan WOULD emit it with a NULL
        // measure) OR a group whose every row was deleted (a full scan would NOT
        // emit it).  `GroupedPartials` does not carry a per-group total row count
        // to disambiguate, so fall back to a real GROUP BY scan rather than risk
        // a missing or phantom group.  Real low-cardinality dimensions rarely hit
        // this; correctness over coverage.
        if max_count == 0 {
            return None;
        }

        group_keys.push(key_scalar);
        for (i, v) in row_values.into_iter().enumerate() {
            agg_columns[i].push(v);
        }
    }

    // 7. Assemble the multi-row batch in output-schema column order.
    let output_schema_ref: SchemaRef = Arc::new(output_schema.clone());
    let mut arrays: Vec<arrow::array::ArrayRef> = Vec::with_capacity(output_schema.fields().len());

    // Column 0: group key.
    match scalars_to_array(&group_keys, output_schema.field(0).data_type()) {
        Ok(a) => arrays.push(a),
        Err(e) => return Some(Err(e)),
    }
    // Columns 1..N: aggregates.
    for (i, col) in agg_columns.iter().enumerate() {
        match scalars_to_array(col, output_schema.field(i + 1).data_type()) {
            Ok(a) => arrays.push(a),
            Err(e) => return Some(Err(e)),
        }
    }

    let batch = match RecordBatch::try_new(Arc::clone(&output_schema_ref), arrays) {
        Ok(b) => b,
        Err(e) => return Some(Err(e.into())),
    };

    match MemorySourceConfig::try_new_from_batches(output_schema_ref, vec![batch]) {
        Ok(exec) => Some(Ok(exec as Arc<dyn ExecutionPlan>)),
        Err(e) => Some(Err(e)),
    }
}

/// Build an Arrow array from a slice of `ScalarValue`s, or an empty typed array
/// when the slice is empty (so a zero-group result is still well-typed).
fn scalars_to_array(
    values: &[ScalarValue],
    data_type: &arrow::datatypes::DataType,
) -> DFResult<arrow::array::ArrayRef> {
    if values.is_empty() {
        return Ok(arrow::array::new_empty_array(data_type));
    }
    ScalarValue::iter_to_array(values.iter().cloned())
}

/// Decode a canonical-JSON group-key string into a `ScalarValue` of the
/// key column's Arrow type.  Returns `None` on any decode failure.
fn decode_group_key(
    group_key_str: &str,
    key_type: &arrow::datatypes::DataType,
) -> Option<ScalarValue> {
    let json: serde_json::Value = serde_json::from_str(group_key_str).ok()?;
    if json.is_null() {
        // NULL group: a typed NULL scalar.
        return ScalarValue::try_from(key_type).ok();
    }
    json_to_scalar_value(&json, key_type)
}

/// Compute one aggregate's value for a single group from its `ColAgg` map.
///
/// Returns `(scalar_value, total_count)` where `total_count` is the group's
/// non-null measure count (used to drop fully-retracted groups).  Returns
/// `None` on any unsupported shape so the caller falls back.
fn compose_aggregate_for_group(
    aggr: &Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
    group_cols: &BTreeMap<String, ColAgg>,
    out_type: &arrow::datatypes::DataType,
    col_map: &HashMap<String, String>,
) -> Option<(ScalarValue, u64)> {
    use arrow::datatypes::DataType;

    let name = aggr.fun().name().to_lowercase();
    let arg = &aggr.expressions()[0];

    // COUNT(col): count_non_null of the measure column.  COUNT(*) (a literal
    // arg) is the count of rows in the group — but the per-group ColAgg only
    // tracks non-null measure counts, so COUNT(*) cannot be answered exactly
    // from grouped partials without a per-group row count; fall back.
    if name == "count" {
        let col_name = resolve_col_name(arg, col_map)?;
        let col_agg = group_cols.get(&col_name)?;
        let cnt = col_agg.count_non_null;
        return Some((ScalarValue::Int64(Some(cnt as i64)), cnt));
    }

    // SUM / AVG / VAR / STDDEV all read the measure column's ColAgg.
    let col_name = resolve_col_name(arg, col_map)?;
    let col_agg = group_cols.get(&col_name)?;
    let n = col_agg.count_non_null;

    let value = match name.as_str() {
        "sum" => match (&col_agg.sum, out_type) {
            (AggScalar::Int(s), DataType::Int64) => {
                ScalarValue::Int64(Some(i64::try_from(*s).ok()?))
            }
            (AggScalar::Int(s), DataType::Int32) => {
                ScalarValue::Int32(Some(i32::try_from(*s).ok()?))
            }
            (AggScalar::Float(s), DataType::Float64) => ScalarValue::Float64(Some(*s)),
            (AggScalar::Float(s), DataType::Float32) => ScalarValue::Float32(Some(*s as f32)),
            (AggScalar::Null, dt) => ScalarValue::try_from(dt).ok()?,
            _ => return None,
        },
        "avg" => {
            if !matches!(out_type, DataType::Float64) {
                return None;
            }
            if n == 0 {
                ScalarValue::Float64(None)
            } else {
                let s = match &col_agg.sum {
                    AggScalar::Int(s) => *s as f64,
                    AggScalar::Float(s) => *s,
                    AggScalar::Null => return Some((ScalarValue::Float64(None), n)),
                };
                ScalarValue::Float64(Some(s / n as f64))
            }
        }
        "var" | "var_pop" | "stddev" | "stddev_pop" => {
            if !matches!(out_type, DataType::Float64) {
                return None;
            }
            let population = matches!(name.as_str(), "var_pop" | "stddev_pop");
            let var = compose_group_variance(col_agg, population);
            let out = match var {
                Some(v) if name.starts_with("stddev") => Some(v.sqrt()),
                other => other,
            };
            ScalarValue::Float64(out)
        }
        _ => return None,
    };

    Some((value, n))
}

/// Compute population or sample variance for a single group's `ColAgg` from its
/// additive primitives.  Returns `None` when undefined (n==0 for population,
/// n<2 for sample, or null sums).
fn compose_group_variance(col_agg: &ColAgg, population: bool) -> Option<f64> {
    let n = col_agg.count_non_null;
    let denom = if population {
        if n == 0 {
            return None;
        }
        n as f64
    } else {
        if n < 2 {
            return None;
        }
        (n - 1) as f64
    };
    let s = match &col_agg.sum {
        AggScalar::Int(s) => *s as f64,
        AggScalar::Float(s) => *s,
        AggScalar::Null => return None,
    };
    let q = match &col_agg.sumsq {
        AggScalar::Int(q) => *q as f64,
        AggScalar::Float(q) => *q,
        AggScalar::Null => return None,
    };
    let n_f = n as f64;
    Some(((q - s * s / n_f) / denom).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::execution::TaskContext;
    use datafusion::functions_aggregate::count;
    use datafusion::functions_aggregate::min_max::min_udaf;
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    use datafusion::physical_expr::expressions::{col as phys_col, Literal};
    use datafusion::physical_plan::aggregates::AggregateMode;
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::scan::IcefallDBScanExec;
    use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
    use icefalldb_core::storage::memory::MemoryStorage;
    use icefalldb_core::PlannedRowGroup;
    use serde_json::Value;

    fn make_scan_exec(
        rows_and_stats: Vec<(usize, HashMap<String, ColumnStats>)>,
    ) -> IcefallDBScanExec {
        let schema = Arc::new(ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        let planned: Vec<PlannedRowGroup> = rows_and_stats
            .into_iter()
            .map(|(rows, columns)| PlannedRowGroup {
                meta: RowGroupMeta {
                    rows,
                    columns,
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect();
        IcefallDBScanExec::new(
            Arc::new(MemoryStorage::new()),
            schema,
            planned,
            None,
            vec![],
            None,
            1024,
            0,
            1,
        )
        .unwrap()
    }

    fn count_star_expr() -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let lit_one = Arc::new(Literal::new(ScalarValue::Int64(Some(1)))) as Arc<dyn PhysicalExpr>;
        AggregateExprBuilder::new(count::count_udaf(), vec![lit_one])
            .schema(schema)
            .alias("COUNT(*)")
            .build()
            .unwrap()
    }

    fn count_a_expr() -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let col_a = phys_col("a", &schema).unwrap();
        AggregateExprBuilder::new(count::count_udaf(), vec![col_a])
            .schema(schema)
            .alias("COUNT(a)")
            .build()
            .unwrap()
    }

    fn min_a_expr() -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )]));
        let col_a = phys_col("a", &schema).unwrap();
        AggregateExprBuilder::new(min_udaf(), vec![col_a])
            .schema(schema)
            .alias("MIN(a)")
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn test_count_star_from_metadata() {
        let scan = make_scan_exec(vec![
            (
                10,
                [(
                    "a".into(),
                    ColumnStats {
                        min: Some(Value::from(1)),
                        max: Some(Value::from(5)),
                        nulls: 2,
                    },
                )]
                .into(),
            ),
            (
                20,
                [(
                    "a".into(),
                    ColumnStats {
                        min: Some(Value::from(6)),
                        max: Some(Value::from(10)),
                        nulls: 0,
                    },
                )]
                .into(),
            ),
        ]);
        let agg = build_aggregate(scan, vec![count_star_expr()]);
        let rule = MetadataAggregate::new();
        let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();
        let stream = optimized
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        let count = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(count, 30);
    }

    #[tokio::test]
    async fn test_count_column_and_min_max_from_metadata() {
        let scan = make_scan_exec(vec![
            (
                10,
                [(
                    "a".into(),
                    ColumnStats {
                        min: Some(Value::from(1)),
                        max: Some(Value::from(5)),
                        nulls: 2,
                    },
                )]
                .into(),
            ),
            (
                20,
                [(
                    "a".into(),
                    ColumnStats {
                        min: Some(Value::from(6)),
                        max: Some(Value::from(10)),
                        nulls: 0,
                    },
                )]
                .into(),
            ),
        ]);
        let agg = build_aggregate(scan, vec![count_a_expr(), min_a_expr()]);
        let rule = MetadataAggregate::new();
        let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();
        let stream = optimized
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
    }

    fn build_aggregate(
        scan: IcefallDBScanExec,
        aggr_exprs: Vec<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
    ) -> AggregateExec {
        let input_schema = scan.schema();
        let aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> =
            aggr_exprs.into_iter().map(Arc::new).collect();
        let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
            aggr_exprs.iter().map(|_| None).collect();
        AggregateExec::try_new(
            AggregateMode::Single,
            datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
            aggr_exprs,
            filter_exprs,
            Arc::new(scan),
            input_schema,
        )
        .unwrap()
    }

    /// Build a `IcefallDBScanExec` over a schema with column `v` (Int32) where
    /// each entry also carries a `deleted_count`.
    fn make_scan_exec_with_deletions(
        fragments: Vec<(usize, u64, HashMap<String, ColumnStats>)>,
    ) -> IcefallDBScanExec {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int32,
            true,
        )]));
        let planned: Vec<PlannedRowGroup> = fragments
            .into_iter()
            .map(|(rows, deleted_count, columns)| PlannedRowGroup {
                meta: RowGroupMeta {
                    rows,
                    columns,
                    ..Default::default()
                },
                deleted_count,
                ..Default::default()
            })
            .collect();
        IcefallDBScanExec::new(
            Arc::new(MemoryStorage::new()),
            schema,
            planned,
            None,
            vec![],
            None,
            1024,
            0,
            1,
        )
        .unwrap()
    }

    /// Returns true when the optimized plan IS the constant-folded fast path
    /// (i.e. a `DataSourceExec` produced by the `MetadataAggregate` rule).
    fn plan_is_metadata_fast_path(plan: &Arc<dyn ExecutionPlan>) -> bool {
        use datafusion_datasource::source::DataSourceExec;
        plan.downcast_ref::<DataSourceExec>().is_some()
    }

    fn count_star_expr_for_schema(
        schema: &ArrowSchema,
    ) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
        let lit_one = Arc::new(Literal::new(ScalarValue::Int64(Some(1)))) as Arc<dyn PhysicalExpr>;
        AggregateExprBuilder::new(count::count_udaf(), vec![lit_one])
            .schema(Arc::new(schema.clone()))
            .alias("COUNT(*)")
            .build()
            .unwrap()
    }

    /// COUNT(*) must subtract deleted_count from each fragment — zero I/O
    /// because deleted_count comes from the manifest, not the Parquet data file.
    #[tokio::test]
    async fn count_star_subtracts_deleted() {
        // 10-row fragment, 2 logically deleted rows → COUNT(*) must be 8.
        let stats: HashMap<String, ColumnStats> = [(
            "v".into(),
            ColumnStats {
                min: Some(Value::from(1)),
                max: Some(Value::from(10)),
                nulls: 0,
            },
        )]
        .into();
        let scan = make_scan_exec_with_deletions(vec![(10, 2, stats)]);
        let agg = {
            let input_schema = scan.schema();
            let expr = count_star_expr_for_schema(&input_schema);
            let aggr_exprs = vec![Arc::new(expr)];
            let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
                aggr_exprs.iter().map(|_| None).collect();
            AggregateExec::try_new(
                AggregateMode::Single,
                datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
                aggr_exprs,
                filter_exprs,
                Arc::new(scan),
                input_schema,
            )
            .unwrap()
        };
        let rule = MetadataAggregate::new();
        let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

        // The fast path must have fired (no scan needed).
        assert!(
            plan_is_metadata_fast_path(&optimized),
            "COUNT(*) should still use the metadata fast path even with deletions"
        );

        let stream = optimized
            .execute(0, Arc::new(TaskContext::default()))
            .unwrap();
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        let count = batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap()
            .value(0);
        assert_eq!(
            count, 8,
            "COUNT(*) must be rows − deleted_count = 10 − 2 = 8"
        );
    }

    /// MIN/MAX fast path must NOT fire when any fragment has deletions.
    /// The sidecar min could be the deleted row, so we fall back to a real scan.
    ///
    /// Here we verify the optimizer correctly leaves the plan unchanged (returning
    /// an `AggregateExec`, not a constant `DataSourceExec`).
    #[tokio::test]
    async fn min_on_dirty_fragment_reflects_survivors() {
        // Fragment: 5 rows, min sidecar=1, but 1 deletion (the row with value 1).
        // Correct survivor MIN would be 2, but sidecar says 1.
        // The rule must NOT fire → plan stays as AggregateExec.
        let stats: HashMap<String, ColumnStats> = [(
            "v".into(),
            ColumnStats {
                min: Some(Value::from(1)),
                max: Some(Value::from(5)),
                nulls: 0,
            },
        )]
        .into();
        let scan = make_scan_exec_with_deletions(vec![(5, 1, stats)]);
        let schema = scan.schema();
        let col_v = phys_col("v", &schema).unwrap();
        let min_expr = AggregateExprBuilder::new(
            datafusion::functions_aggregate::min_max::min_udaf(),
            vec![col_v],
        )
        .schema(Arc::clone(&schema))
        .alias("MIN(v)")
        .build()
        .unwrap();
        let aggr_exprs = vec![Arc::new(min_expr)];
        let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
            aggr_exprs.iter().map(|_| None).collect();
        let agg = AggregateExec::try_new(
            AggregateMode::Single,
            datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
            aggr_exprs,
            filter_exprs,
            Arc::new(scan),
            schema,
        )
        .unwrap();
        let rule = MetadataAggregate::new();
        let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

        // The rule must NOT have fired — the plan must still be an AggregateExec
        // so that the real scan (which honours deletion vectors) runs.
        assert!(
            !plan_is_metadata_fast_path(&optimized),
            "MIN fast path must not fire when any fragment has deleted rows"
        );
        assert!(
            optimized.downcast_ref::<AggregateExec>().is_some(),
            "plan must remain an AggregateExec so the real scan runs"
        );
    }

    #[cfg(feature = "sketches")]
    mod sketch_tests {
        use super::*;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use datafusion::common::ScalarValue;
        use datafusion::execution::TaskContext;
        use datafusion::functions_aggregate::approx_distinct::approx_distinct_udaf;
        use datafusion::physical_expr::aggregate::AggregateExprBuilder;
        use datafusion::physical_expr::expressions::col as phys_col;
        use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
        use datasketches::cpc::CpcSketch;
        use icefalldb_core::agg_cache::FragmentAggState;
        use icefalldb_core::metadata::RowGroupMeta;
        use icefalldb_core::storage::memory::MemoryStorage;
        use icefalldb_core::PlannedRowGroup;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        fn make_scan_with_sketches(
            fragments: Vec<(usize, BTreeMap<String, Vec<u8>>, String)>,
        ) -> crate::scan::IcefallDBScanExec {
            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "v",
                DataType::Int64,
                false,
            )]));
            let planned: Vec<PlannedRowGroup> = fragments
                .into_iter()
                .enumerate()
                .map(|(i, (rows, sketch_map, checksum))| {
                    let agg_state = Arc::new(FragmentAggState {
                        fragment_id: i as u64,
                        content_hash: checksum.clone(),
                        cols: BTreeMap::new(),
                        grouped: None,
                        distinct: Some(sketch_map),
                        quantile: None,
                    });
                    PlannedRowGroup {
                        meta: RowGroupMeta {
                            rows,
                            checksum,
                            ..Default::default()
                        },
                        deleted_count: 0,
                        agg_state: Some(agg_state),
                        ..Default::default()
                    }
                })
                .collect();
            crate::scan::IcefallDBScanExec::new(
                Arc::new(MemoryStorage::new()),
                schema,
                planned,
                None,
                vec![],
                None,
                1024,
                0,
                1,
            )
            .unwrap()
        }

        fn build_approx_distinct_agg(scan: crate::scan::IcefallDBScanExec) -> AggregateExec {
            let schema = scan.schema();
            let col_v = phys_col("v", &schema).unwrap();
            let expr = AggregateExprBuilder::new(approx_distinct_udaf(), vec![col_v])
                .schema(Arc::clone(&schema))
                .alias("approx_distinct(v)")
                .build()
                .unwrap();
            let aggr_exprs = vec![Arc::new(expr)];
            let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
                aggr_exprs.iter().map(|_| None).collect();
            AggregateExec::try_new(
                AggregateMode::Single,
                datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
                aggr_exprs,
                filter_exprs,
                Arc::new(scan),
                schema,
            )
            .unwrap()
        }

        /// Clean-table test: 3 fragments with ~51k distinct values total.
        /// approx_distinct must be within 3% of the exact count AND the fast path fires.
        #[tokio::test]
        async fn approx_distinct_clean_table_within_error() {
            const N_PER_FRAG: u64 = 17_000;
            const N_FRAGS: u64 = 3;
            const TRUE_DISTINCT: u64 = N_PER_FRAG * N_FRAGS; // 51000, non-overlapping

            // Build 3 non-overlapping CPC sketches.
            let mut frags = vec![];
            for frag_idx in 0..N_FRAGS {
                let mut sketch = CpcSketch::new(12);
                for i in 0..N_PER_FRAG {
                    sketch.update(frag_idx * N_PER_FRAG + i);
                }
                let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
                map.insert("v".to_string(), sketch.serialize());
                let checksum = format!("sha256:frag{frag_idx}");
                frags.push((N_PER_FRAG as usize, map, checksum));
            }

            let scan = make_scan_with_sketches(frags);
            let agg = build_approx_distinct_agg(scan);
            let rule = MetadataAggregate::new();
            let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

            // Fast path must have fired: optimized plan is a DataSourceExec (constant).
            use datafusion_datasource::source::DataSourceExec;
            assert!(
                optimized.downcast_ref::<DataSourceExec>().is_some(),
                "approx_distinct fast path must fire for clean table with complete sketches"
            );

            let stream = optimized
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap();
            let batches = datafusion::physical_plan::common::collect(stream)
                .await
                .unwrap();
            let estimate = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .unwrap()
                .value(0) as f64;

            let error = (estimate - TRUE_DISTINCT as f64).abs() / TRUE_DISTINCT as f64;
            assert!(
                error <= 0.03,
                "CPC approx_distinct estimate {estimate:.0} for true {TRUE_DISTINCT}: \
                 relative error {error:.4} must be ≤ 3% (CPC lg_k=12 nominal ≈ 1.04%)"
            );
        }

        /// Fallback: a fragment missing its sketch → rule falls back, AggregateExec remains.
        #[tokio::test]
        async fn approx_distinct_missing_sketch_falls_back() {
            // Fragment 0: has sketch. Fragment 1: NO sketch (distinct = None in agg_state).
            let mut sketch = CpcSketch::new(12);
            for i in 0..100u64 {
                sketch.update(i);
            }
            let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            map.insert("v".to_string(), sketch.serialize());

            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "v",
                DataType::Int64,
                false,
            )]));

            let agg_state_with = Arc::new(FragmentAggState {
                fragment_id: 0,
                content_hash: "sha256:a".to_string(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: Some(map),
                quantile: None,
            });
            let agg_state_without = Arc::new(FragmentAggState {
                fragment_id: 1,
                content_hash: "sha256:b".to_string(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: None, // <-- missing sketch
                quantile: None,
            });

            let planned = vec![
                PlannedRowGroup {
                    meta: RowGroupMeta {
                        rows: 100,
                        checksum: "sha256:a".into(),
                        ..Default::default()
                    },
                    deleted_count: 0,
                    agg_state: Some(agg_state_with),
                    ..Default::default()
                },
                PlannedRowGroup {
                    meta: RowGroupMeta {
                        rows: 100,
                        checksum: "sha256:b".into(),
                        ..Default::default()
                    },
                    deleted_count: 0,
                    agg_state: Some(agg_state_without),
                    ..Default::default()
                },
            ];
            let scan = crate::scan::IcefallDBScanExec::new(
                Arc::new(MemoryStorage::new()),
                schema,
                planned,
                None,
                vec![],
                None,
                1024,
                0,
                1,
            )
            .unwrap();

            let agg = build_approx_distinct_agg(scan);
            let rule = MetadataAggregate::new();
            let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

            // Must fall back — plan stays as AggregateExec.
            assert!(
                optimized.downcast_ref::<AggregateExec>().is_some(),
                "missing sketch must cause fallback to AggregateExec"
            );
        }

        /// Fallback: a dirty fragment (deleted_count > 0) → rule falls back.
        #[tokio::test]
        async fn approx_distinct_dirty_fragment_falls_back() {
            let mut sketch = CpcSketch::new(12);
            for i in 0..100u64 {
                sketch.update(i);
            }
            let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            map.insert("v".to_string(), sketch.serialize());

            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "v",
                DataType::Int64,
                false,
            )]));
            let agg_state = Arc::new(FragmentAggState {
                fragment_id: 0,
                content_hash: "sha256:dirty".to_string(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: Some(map),
                quantile: None,
            });
            let planned = vec![PlannedRowGroup {
                meta: RowGroupMeta {
                    rows: 100,
                    checksum: "sha256:dirty".into(),
                    ..Default::default()
                },
                deleted_count: 5, // <-- dirty
                agg_state: Some(agg_state),
                ..Default::default()
            }];
            let scan = crate::scan::IcefallDBScanExec::new(
                Arc::new(MemoryStorage::new()),
                schema,
                planned,
                None,
                vec![],
                None,
                1024,
                0,
                1,
            )
            .unwrap();

            let agg = build_approx_distinct_agg(scan);
            let rule = MetadataAggregate::new();
            let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

            assert!(
                optimized.downcast_ref::<AggregateExec>().is_some(),
                "dirty fragment must cause fallback to AggregateExec"
            );
        }

        // ── approx_percentile_cont tests ───────────────────────────────

        /// Build a scan with T-Digest quantile sketches pre-populated.
        fn make_scan_with_quantile_sketches(
            fragments: Vec<(usize, std::collections::BTreeMap<String, Vec<u8>>, String)>,
        ) -> crate::scan::IcefallDBScanExec {
            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "v",
                DataType::Float64,
                false,
            )]));
            let planned: Vec<PlannedRowGroup> = fragments
                .into_iter()
                .enumerate()
                .map(|(i, (rows, quantile_map, checksum))| {
                    let agg_state = Arc::new(FragmentAggState {
                        fragment_id: i as u64,
                        content_hash: checksum.clone(),
                        cols: BTreeMap::new(),
                        grouped: None,
                        distinct: None,
                        quantile: Some(quantile_map),
                    });
                    PlannedRowGroup {
                        meta: RowGroupMeta {
                            rows,
                            checksum,
                            ..Default::default()
                        },
                        deleted_count: 0,
                        agg_state: Some(agg_state),
                        ..Default::default()
                    }
                })
                .collect();
            crate::scan::IcefallDBScanExec::new(
                Arc::new(MemoryStorage::new()),
                schema,
                planned,
                None,
                vec![],
                None,
                1024,
                0,
                1,
            )
            .unwrap()
        }

        fn build_approx_percentile_agg(
            scan: crate::scan::IcefallDBScanExec,
            q: f64,
        ) -> AggregateExec {
            use datafusion::functions_aggregate::approx_percentile_cont::approx_percentile_cont_udaf;
            let schema = scan.schema();
            let col_v = phys_col("v", &schema).unwrap();
            let q_lit = Arc::new(datafusion::physical_expr::expressions::Literal::new(
                ScalarValue::Float64(Some(q)),
            )) as Arc<dyn datafusion::physical_expr::PhysicalExpr>;
            let expr = AggregateExprBuilder::new(approx_percentile_cont_udaf(), vec![col_v, q_lit])
                .schema(Arc::clone(&schema))
                .alias("approx_percentile_cont(v,0.95)")
                .build()
                .unwrap();
            let aggr_exprs = vec![Arc::new(expr)];
            let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
                aggr_exprs.iter().map(|_| None).collect();
            AggregateExec::try_new(
                AggregateMode::Single,
                datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
                aggr_exprs,
                filter_exprs,
                Arc::new(scan),
                schema,
            )
            .unwrap()
        }

        /// Clean-table test: ≥3 fragments with skewed ~300k-row data total.
        ///
        /// `approx_percentile_cont(v, 0.95)` is answered from merged cached T-Digest
        /// sketches.  Asserts:
        /// 1. Fast path fired (DataSourceExec, zero Parquet I/O).
        /// 2. Returned value's rank over the data is within T-Digest's bound (≤3%
        ///    rank error — NOT byte-equal).
        ///
        /// Sketch: T-Digest k=200.  Error type: rank error.  Bound: ≤0.03.
        #[tokio::test]
        async fn approx_percentile_clean_table_fast_path_and_within_error() {
            use datasketches::tdigest::TDigestMut;

            const N_PER_FRAG: usize = 100_000;
            const N_FRAGS: usize = 3;
            const Q: f64 = 0.95;

            // Build 3 non-overlapping T-Digest sketches (skewed distribution:
            // fragment 2 covers the upper range where p95 lands).
            let mut frags = vec![];
            let mut all_values: Vec<f64> = Vec::with_capacity(N_PER_FRAG * N_FRAGS);

            for frag_idx in 0..N_FRAGS {
                let base = frag_idx * N_PER_FRAG;
                let mut sketch = TDigestMut::default();
                for i in 0..N_PER_FRAG {
                    let v = (base + i) as f64;
                    sketch.update(v);
                    all_values.push(v);
                }
                let mut map: std::collections::BTreeMap<String, Vec<u8>> =
                    std::collections::BTreeMap::new();
                map.insert("v".to_string(), sketch.serialize());
                let checksum = format!("sha256:frag{frag_idx}");
                frags.push((N_PER_FRAG, map, checksum));
            }

            let scan = make_scan_with_quantile_sketches(frags);
            let agg = build_approx_percentile_agg(scan, Q);
            let rule = MetadataAggregate::new();
            let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

            // Fast path must have fired: optimized plan is a DataSourceExec (constant).
            use datafusion_datasource::source::DataSourceExec;
            assert!(
                optimized.downcast_ref::<DataSourceExec>().is_some(),
                "approx_percentile_cont fast path must fire for clean table with complete sketches"
            );

            let stream = optimized
                .execute(0, Arc::new(TaskContext::default()))
                .unwrap();
            let batches = datafusion::physical_plan::common::collect(stream)
                .await
                .unwrap();
            let estimate = batches[0]
                .column(0)
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .unwrap()
                .value(0);

            // Compute rank of the returned value over the combined data.
            all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let rank_pos = all_values.partition_point(|&x| x <= estimate);
            let actual_rank = rank_pos as f64 / all_values.len() as f64;
            let rank_err = (actual_rank - Q).abs();

            assert!(
                rank_err <= 0.03,
                "T-Digest approx_percentile_cont({Q}) estimate {estimate:.1}: \
                 rank {actual_rank:.4} vs target {Q}, rank error {rank_err:.4} must be ≤ 0.03 \
                 (T-Digest k=200 bound)"
            );
        }

        /// Fallback: a fragment missing its quantile sketch → rule falls back.
        #[tokio::test]
        async fn approx_percentile_missing_sketch_falls_back() {
            use datasketches::tdigest::TDigestMut;

            let mut sketch = TDigestMut::default();
            for i in 0..100i64 {
                sketch.update(i as f64);
            }
            let mut map: std::collections::BTreeMap<String, Vec<u8>> =
                std::collections::BTreeMap::new();
            map.insert("v".to_string(), sketch.serialize());

            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "v",
                DataType::Float64,
                false,
            )]));

            let agg_state_with = Arc::new(FragmentAggState {
                fragment_id: 0,
                content_hash: "sha256:qa".to_string(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: None,
                quantile: Some(map),
            });
            let agg_state_without = Arc::new(FragmentAggState {
                fragment_id: 1,
                content_hash: "sha256:qb".to_string(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: None,
                quantile: None, // <-- missing quantile sketch
            });

            let planned = vec![
                PlannedRowGroup {
                    meta: RowGroupMeta {
                        rows: 100,
                        checksum: "sha256:qa".into(),
                        ..Default::default()
                    },
                    deleted_count: 0,
                    agg_state: Some(agg_state_with),
                    ..Default::default()
                },
                PlannedRowGroup {
                    meta: RowGroupMeta {
                        rows: 100,
                        checksum: "sha256:qb".into(),
                        ..Default::default()
                    },
                    deleted_count: 0,
                    agg_state: Some(agg_state_without),
                    ..Default::default()
                },
            ];
            let scan = crate::scan::IcefallDBScanExec::new(
                Arc::new(MemoryStorage::new()),
                schema,
                planned,
                None,
                vec![],
                None,
                1024,
                0,
                1,
            )
            .unwrap();

            let agg = build_approx_percentile_agg(scan, 0.5);
            let rule = MetadataAggregate::new();
            let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

            assert!(
                optimized.downcast_ref::<AggregateExec>().is_some(),
                "missing quantile sketch must cause fallback to AggregateExec"
            );
        }

        /// Fallback: a dirty fragment (deleted_count > 0) → rule falls back.
        #[tokio::test]
        async fn approx_percentile_dirty_fragment_falls_back() {
            use datasketches::tdigest::TDigestMut;

            let mut sketch = TDigestMut::default();
            for i in 0..100i64 {
                sketch.update(i as f64);
            }
            let mut map: std::collections::BTreeMap<String, Vec<u8>> =
                std::collections::BTreeMap::new();
            map.insert("v".to_string(), sketch.serialize());

            let schema = Arc::new(ArrowSchema::new(vec![Field::new(
                "v",
                DataType::Float64,
                false,
            )]));
            let agg_state = Arc::new(FragmentAggState {
                fragment_id: 0,
                content_hash: "sha256:qdirty".to_string(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: None,
                quantile: Some(map),
            });
            let planned = vec![PlannedRowGroup {
                meta: RowGroupMeta {
                    rows: 100,
                    checksum: "sha256:qdirty".into(),
                    ..Default::default()
                },
                deleted_count: 3, // <-- dirty
                agg_state: Some(agg_state),
                ..Default::default()
            }];
            let scan = crate::scan::IcefallDBScanExec::new(
                Arc::new(MemoryStorage::new()),
                schema,
                planned,
                None,
                vec![],
                None,
                1024,
                0,
                1,
            )
            .unwrap();

            let agg = build_approx_percentile_agg(scan, 0.5);
            let rule = MetadataAggregate::new();
            let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

            assert!(
                optimized.downcast_ref::<AggregateExec>().is_some(),
                "dirty fragment must cause fallback to AggregateExec for approx_percentile_cont"
            );
        }
    }

    /// Pre-existing guard: on a CLEAN (zero-deletion) float table,
    /// MIN(float_col) must still NOT use the sidecar fast path — the pre-existing
    /// float guard must be preserved unconditionally.
    #[tokio::test]
    async fn float_min_still_scans_even_on_clean_fragment() {
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "f",
            DataType::Float64,
            true,
        )]));
        let stats: HashMap<String, ColumnStats> = [(
            "f".into(),
            ColumnStats {
                min: Some(Value::from(1.0f64)),
                max: Some(Value::from(9.9f64)),
                nulls: 0,
            },
        )]
        .into();
        // deleted_count = 0 → clean fragment, so the deletion guard is not the reason
        // the fast path is bypassed; the float-type guard must be.
        let planned: Vec<PlannedRowGroup> = vec![PlannedRowGroup {
            meta: RowGroupMeta {
                rows: 5,
                columns: stats,
                ..Default::default()
            },
            deleted_count: 0,
            ..Default::default()
        }];
        let scan = IcefallDBScanExec::new(
            Arc::new(MemoryStorage::new()),
            Arc::clone(&schema),
            planned,
            None,
            vec![],
            None,
            1024,
            0,
            1,
        )
        .unwrap();
        let col_f = phys_col("f", &schema).unwrap();
        let min_expr = AggregateExprBuilder::new(
            datafusion::functions_aggregate::min_max::min_udaf(),
            vec![col_f],
        )
        .schema(Arc::clone(&schema))
        .alias("MIN(f)")
        .build()
        .unwrap();
        let aggr_exprs = vec![Arc::new(min_expr)];
        let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
            aggr_exprs.iter().map(|_| None).collect();
        let agg = AggregateExec::try_new(
            AggregateMode::Single,
            datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
            aggr_exprs,
            filter_exprs,
            Arc::new(scan),
            schema,
        )
        .unwrap();
        let rule = MetadataAggregate::new();
        let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

        // The float guard must prevent the fast path from firing even on a clean fragment.
        assert!(
            !plan_is_metadata_fast_path(&optimized),
            "MIN(float) must never use the sidecar fast path — float guard must be preserved"
        );
        assert!(
            optimized.downcast_ref::<AggregateExec>().is_some(),
            "plan must remain an AggregateExec for float MIN"
        );
    }
}
