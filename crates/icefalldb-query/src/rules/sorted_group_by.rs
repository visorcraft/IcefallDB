//! Physical optimizer rule that specializes sorted group-by aggregates.
//!
//! When an [`AggregateExec`] sits directly on a [`IcefallDBScanExec`] whose
//! partitions advertise a shared sort order matching the group key(s), and all
//! aggregate expressions are supported, the rule replaces the aggregate with a
//! single-pass [`StreamingGroupByExec`].

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::config::ConfigOptions;
use datafusion::common::Result as DFResult;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_expr::{PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::execution::streaming_group_by::{AggType, StreamingGroupByExec};
use crate::scan::IcefallDBScanExec;
use crate::session::IcefallDBConfig;

/// Physical optimizer rule that specializes sorted group-by aggregates.
#[derive(Debug, Default)]
pub struct SortedGroupBy;

impl SortedGroupBy {
    /// Create a new `SortedGroupBy` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for SortedGroupBy {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let enabled = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.sorted_group_by)
            .unwrap_or(true);
        if !enabled {
            return Ok(plan);
        }

        // Recursively apply the rule bottom-up.
        let plan = crate::rules::optimize_children(plan, config, |plan, config| {
            self.optimize(plan, config)
        })?;
        transform(plan)
    }

    fn name(&self) -> &str {
        "sorted_group_by"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Try to rewrite an `AggregateExec` over a sorted `IcefallDBScanExec` into a
/// `StreamingGroupByExec`.
fn transform(plan: Arc<dyn ExecutionPlan>) -> DFResult<Arc<dyn ExecutionPlan>> {
    let Some(agg) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(plan);
    };

    let group_by = agg.group_expr();
    if group_by.expr().is_empty() {
        return Ok(plan);
    }

    // Pick the aggregate that actually evaluates raw input data.  For a Final
    // aggregate this is the Partial child; for Single it is the aggregate
    // itself.  Partial-only aggregators are left untouched.
    let (source_agg, output_schema, chain_root): (
        &AggregateExec,
        SchemaRef,
        Arc<dyn ExecutionPlan>,
    ) = match agg.mode() {
        AggregateMode::Single | AggregateMode::SinglePartitioned => {
            (agg, agg.schema(), Arc::clone(agg.input()))
        }
        AggregateMode::Final | AggregateMode::FinalPartitioned => {
            let child = agg.input();
            let Some(partial) = child.downcast_ref::<AggregateExec>() else {
                return Ok(plan);
            };
            if !matches!(
                partial.mode(),
                AggregateMode::Partial | AggregateMode::PartialReduce
            ) {
                return Ok(plan);
            }
            (partial, agg.schema(), Arc::clone(partial.input()))
        }
        AggregateMode::Partial | AggregateMode::PartialReduce => return Ok(plan),
    };

    // Walk through any FilterExec chain to reach the IcefallDBScanExec.
    let Some((scan, filters)) = find_scan_with_filters(&chain_root) else {
        return Ok(plan);
    };

    // The scan must advertise an ordering whose prefix covers the group keys.
    let group_exprs: Vec<Arc<dyn PhysicalExpr>> = source_agg
        .group_expr()
        .expr()
        .iter()
        .map(|(expr, _name)| Arc::clone(expr))
        .collect();
    if !ordering_covers_groups(scan, &group_exprs) {
        return Ok(plan);
    }

    // Verify every aggregate expression is supported.
    let mut streaming_aggrs = Vec::with_capacity(source_agg.aggr_expr().len());
    let filter_exprs = source_agg.filter_expr();
    if source_agg.aggr_expr().len() != filter_exprs.len() {
        return Ok(plan);
    }
    for (aggr, filter) in source_agg.aggr_expr().iter().zip(filter_exprs.iter()) {
        if aggr.is_distinct() || filter.is_some() {
            return Ok(plan);
        }
        let args = aggr.expressions();
        if args.len() != 1 {
            return Ok(plan);
        }
        let name = aggr.fun().name().to_lowercase();
        let (agg_type, arg) = match name.as_str() {
            "count" => {
                let arg = &args[0];
                if arg.downcast_ref::<Column>().is_some() || arg.downcast_ref::<Literal>().is_some()
                {
                    (AggType::Count, Arc::clone(arg))
                } else {
                    return Ok(plan);
                }
            }
            "sum" => (AggType::Sum, Arc::clone(&args[0])),
            "avg" => (AggType::Avg, Arc::clone(&args[0])),
            _ => return Ok(plan),
        };
        streaming_aggrs.push((agg_type, arg));
    }

    // Rebuild the FilterExec chain with the scan at the bottom.
    let mut streaming_input: Arc<dyn ExecutionPlan> = Arc::new(scan.clone());
    for predicate in filters.iter().rev() {
        let filter = match FilterExec::try_new(Arc::clone(predicate), streaming_input) {
            Ok(f) => Arc::new(f),
            Err(_) => return Ok(plan),
        };
        streaming_input = filter;
    }

    match StreamingGroupByExec::try_new(
        streaming_input,
        group_exprs,
        streaming_aggrs,
        output_schema,
    ) {
        Ok(exec) => Ok(Arc::new(exec)),
        Err(_) => Ok(plan),
    }
}

/// Walk through a chain of `FilterExec` nodes above a `IcefallDBScanExec` and
/// return the scan plus the predicates in outermost-to-innermost order.
fn find_scan_with_filters(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<(&IcefallDBScanExec, Vec<Arc<dyn PhysicalExpr>>)> {
    let mut current = plan;
    let mut filters = Vec::new();
    loop {
        if let Some(filter) = current.downcast_ref::<FilterExec>() {
            filters.push(Arc::clone(filter.predicate()));
            current = filter.input();
        } else {
            let scan = current.downcast_ref::<IcefallDBScanExec>()?;
            return Some((scan, filters));
        }
    }
}

/// Return true if the scan's advertised ordering has the group expressions as
/// an ascending prefix.
fn ordering_covers_groups(scan: &IcefallDBScanExec, group_exprs: &[Arc<dyn PhysicalExpr>]) -> bool {
    let Some(ordering) = scan.properties().equivalence_properties().output_ordering() else {
        return false;
    };
    if ordering.len() < group_exprs.len() {
        return false;
    }

    for (group_expr, sort_expr) in group_exprs.iter().zip(ordering.iter()) {
        if !group_expr_matches_sort(group_expr, sort_expr) {
            return false;
        }
    }
    true
}

/// Return true if `group_expr` is a column expression that matches the sort
/// expression in both column identity and ascending order.
fn group_expr_matches_sort(
    group_expr: &Arc<dyn PhysicalExpr>,
    sort_expr: &PhysicalSortExpr,
) -> bool {
    let group_col = match group_expr.downcast_ref::<Column>() {
        Some(c) => c,
        None => return false,
    };
    let sort_col = match sort_expr.expr.downcast_ref::<Column>() {
        Some(c) => c,
        None => return false,
    };

    if group_col.index() != sort_col.index() || group_col.name() != sort_col.name() {
        return false;
    }

    // StreamingGroupByExec encodes group keys with default ascending sort
    // options, so only ascending orderings are eligible.
    !sort_expr.options.descending && sort_expr.options.nulls_first
}
