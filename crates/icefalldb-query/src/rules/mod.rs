//! DataFusion optimizer rules for IcefallDB.

pub mod broadcast_tiny_join;
pub mod dynamic_filter_pushdown;
pub mod lookup_join;
pub mod low_cardinality_group_by;
pub mod metadata_aggregate;
pub mod rowid_selection_pushdown;
pub mod simplify_cast_predicates;
pub mod sorted_group_by;
pub mod tiny_build_join;

pub use broadcast_tiny_join::BroadcastTinyJoin;
pub use dynamic_filter_pushdown::DynamicFilterPushdown;
pub use lookup_join::LookupJoin;
pub use low_cardinality_group_by::LowCardinalityGroupBy;
pub use metadata_aggregate::MetadataAggregate;
pub use rowid_selection_pushdown::RowIdSelectionPushdown;
pub use simplify_cast_predicates::SimplifyCastPredicates;
pub use sorted_group_by::SortedGroupBy;
pub use tiny_build_join::TinyBuildJoin;

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::Result as DFResult;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::execution::SessionState;
use datafusion::physical_plan::ExecutionPlan;

/// Recursively optimize the children of `plan` bottom-up.
///
/// If `plan` has no children it is returned unchanged. Otherwise each child is
/// passed to `optimize` and the plan is rebuilt with the optimized children.
pub(crate) fn optimize_children(
    plan: Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
    optimize: impl Fn(Arc<dyn ExecutionPlan>, &ConfigOptions) -> DFResult<Arc<dyn ExecutionPlan>>,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }
    let new_children: Vec<Arc<dyn ExecutionPlan>> = children
        .iter()
        .map(|child| optimize(Arc::clone(*child), config))
        .collect::<DFResult<_>>()?;
    plan.with_new_children(new_children)
}

/// Register all IcefallDB physical optimizer rules on a fresh `SessionState`.
///
/// Returns a new [`SessionState`] because DataFusion's builder API consumes the
/// old state. Rules are appended in the order they should run: metadata
/// aggregate first, low-cardinality group-by, sorted group-by, tiny-build-side
/// join, lookup join, and dynamic filter pushdown.
pub fn register_icefalldb_rules(state: SessionState) -> SessionState {
    SessionStateBuilder::new_from_existing(state)
        .with_physical_optimizer_rule(std::sync::Arc::new(MetadataAggregate::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(LowCardinalityGroupBy::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(SortedGroupBy::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(BroadcastTinyJoin::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(TinyBuildJoin::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(LookupJoin::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(DynamicFilterPushdown::new()))
        .with_physical_optimizer_rule(std::sync::Arc::new(RowIdSelectionPushdown::new()))
        .build()
}
