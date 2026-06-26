//! Physical optimizer rule that turns a `_rowid` equality filter directly above
//! a `IcefallDBScanExec` into a row-selection pushdown.
//!
//! `_rowid` is a synthesized pseudo-column, so DataFusion 54 cannot push a
//! `_rowid` filter through the `TableProvider` (it fails to resolve the column
//! against the table source schema during logical filter pushdown). Instead, on
//! the fully-resolved *physical* plan, this rule matches
//! `FilterExec(_rowid = a OR _rowid = b OR ...) -> IcefallDBScanExec`, extracts
//! the target row-id set, and rebuilds the scan with a per-fragment
//! [`IcefallDBScanExec::with_rowid_targets`] selection so only the matching rows
//! are decoded. The `FilterExec` is preserved as the correctness guard, so the
//! selection can only ever reduce I/O — never change results.

use std::any::Any;
use std::collections::HashSet;
use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::{Result as DFResult, ScalarValue};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, InListExpr, Literal};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::rules::optimize_children;
use crate::scan::{IcefallDBScanExec, PSEUDO_COL_ROWID};

/// Physical optimizer rule for the `_rowid` row-selection pushdown.
#[derive(Debug, Default)]
pub struct RowIdSelectionPushdown;

impl RowIdSelectionPushdown {
    /// Create a new `RowIdSelectionPushdown` rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for RowIdSelectionPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        optimize_internal(plan, config)
    }

    fn name(&self) -> &str {
        "rowid_selection_pushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn optimize_internal(
    plan: Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    // Optimize children first (bottom-up), so the scan child is final.
    let plan = optimize_children(plan, config, optimize_internal)?;

    let Some(filter) = (plan.as_ref() as &dyn Any).downcast_ref::<FilterExec>() else {
        return Ok(plan);
    };

    let mut targets: HashSet<u64> = HashSet::new();
    if !collect_rowid_targets(filter.predicate(), &mut targets) || targets.is_empty() {
        return Ok(plan);
    }

    // Rebuild the scan beneath this FilterExec so it decodes only the targeted
    // rows. DataFusion inserts a `RepartitionExec`/`CoalesceBatchesExec` between
    // the filter and the scan for parallelism (any `target_partitions > 1` plan),
    // so matching only a *direct* `FilterExec -> IcefallDBScanExec` child misses
    // every real multi-partition plan and the selection never applies. Descend
    // through partition-only pass-throughs (which preserve every row and its
    // `_rowid`) to reach the scan. The `FilterExec` is preserved as the
    // correctness guard, so the selection can only ever reduce I/O.
    match rewrite_scan_with_targets(Arc::clone(filter.input()), &targets) {
        Some(new_input) => plan.with_new_children(vec![new_input]),
        None => Ok(plan),
    }
}

/// Descend through single-child, partition-only pass-through operators to find a
/// `IcefallDBScanExec` and rebuild it with the `_rowid` row-selection. Returns
/// `None` when no scan is reachable that way. Only operators that preserve every
/// input row and its `_rowid` (repartition / coalesce / cooperative-yield) are
/// traversed, so applying the selection below them is always safe.
fn rewrite_scan_with_targets(
    plan: Arc<dyn ExecutionPlan>,
    targets: &HashSet<u64>,
) -> Option<Arc<dyn ExecutionPlan>> {
    let any = plan.as_ref() as &dyn Any;
    if let Some(scan) = any.downcast_ref::<IcefallDBScanExec>() {
        return Some(scan.with_rowid_targets(targets.clone()));
    }
    let is_passthrough = any.is::<RepartitionExec>()
        || any.is::<CoalescePartitionsExec>()
        || any.is::<CooperativeExec>();
    if is_passthrough {
        let children = plan.children();
        if children.len() == 1 {
            let child = Arc::clone(children[0]);
            if let Some(new_child) = rewrite_scan_with_targets(child, targets) {
                return plan.with_new_children(vec![new_child]).ok();
            }
        }
    }
    None
}

/// Collect the `_rowid` values from a physical predicate composed entirely of
/// `_rowid = lit`, `_rowid IN (lits)`, or `OR`-chains of those (DataFusion
/// rewrites a small `IN` list into `= a OR = b OR ...`). Returns `false` (and may
/// leave `out` partially filled) if any sub-term is anything else.
fn collect_rowid_targets(expr: &Arc<dyn PhysicalExpr>, out: &mut HashSet<u64>) -> bool {
    let any = expr.as_ref() as &dyn Any;
    if let Some(b) = any.downcast_ref::<BinaryExpr>() {
        return match b.op() {
            Operator::Or => {
                collect_rowid_targets(b.left(), out) && collect_rowid_targets(b.right(), out)
            }
            Operator::Eq => {
                if is_rowid_column(b.left()) {
                    if let Some(v) = lit_to_u64(b.right()) {
                        out.insert(v);
                        return true;
                    }
                }
                if is_rowid_column(b.right()) {
                    if let Some(v) = lit_to_u64(b.left()) {
                        out.insert(v);
                        return true;
                    }
                }
                false
            }
            _ => false,
        };
    }
    if let Some(il) = any.downcast_ref::<InListExpr>() {
        if il.negated() || !is_rowid_column(il.expr()) {
            return false;
        }
        for item in il.list() {
            match lit_to_u64(item) {
                Some(v) => {
                    out.insert(v);
                }
                None => return false,
            }
        }
        return true;
    }
    false
}

fn is_rowid_column(expr: &Arc<dyn PhysicalExpr>) -> bool {
    (expr.as_ref() as &dyn Any)
        .downcast_ref::<Column>()
        .is_some_and(|c| c.name() == PSEUDO_COL_ROWID)
}

fn lit_to_u64(expr: &Arc<dyn PhysicalExpr>) -> Option<u64> {
    let lit = (expr.as_ref() as &dyn Any).downcast_ref::<Literal>()?;
    match lit.value() {
        ScalarValue::UInt64(Some(v)) => Some(*v),
        ScalarValue::Int64(Some(v)) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}
