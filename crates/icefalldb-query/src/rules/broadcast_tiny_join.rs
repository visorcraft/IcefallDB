//! Physical optimizer rule that converts tiny-build partitioned hash joins
//! into broadcast hash joins.
//!
//! DataFusion's physical planner sometimes keeps a hash join in
//! `PartitionMode::Partitioned` even when the build side is tiny (e.g. a 10-row
//! dimension table), because a `RepartitionExec` has already split the build
//! side into many partitions. This rule detects that situation and switches
//! the join to `PartitionMode::CollectLeft`, which collects the small build
//! side once and broadcasts it to every probe partition.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Precision;
use datafusion::common::Result as DFResult;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::joins::PartitionMode;
use datafusion::physical_plan::repartition::RepartitionExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::session::IcefallDBConfig;

/// Physical optimizer rule that broadcasts tiny-build-side hash joins.
#[derive(Debug, Default)]
pub struct BroadcastTinyJoin;

impl BroadcastTinyJoin {
    /// Create a new `BroadcastTinyJoin` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for BroadcastTinyJoin {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let enabled = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.tiny_build_join)
            .unwrap_or(true);
        if !enabled {
            return Ok(plan);
        }

        let threshold = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.tiny_build_join_threshold)
            .unwrap_or(4096);

        optimize_internal(plan, config, threshold)
    }

    fn name(&self) -> &str {
        "broadcast_tiny_join"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn optimize_internal(
    plan: Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
    threshold: usize,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    // Recurse bottom-up.
    let plan = crate::rules::optimize_children(plan, config, |plan, config| {
        optimize_internal(plan, config, threshold)
    })?;

    let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
        return Ok(plan);
    };

    // Only rewrite partitioned inner joins with a tiny build side.
    if !matches!(join.partition_mode(), PartitionMode::Partitioned) {
        return Ok(plan);
    }
    if !matches!(join.join_type(), datafusion::common::JoinType::Inner) {
        return Ok(plan);
    }

    let Some(build_rows) = build_side_row_count(join.left()) else {
        return Ok(plan);
    };
    if build_rows > threshold {
        return Ok(plan);
    }

    let new_join = join
        .builder()
        .with_partition_mode(PartitionMode::CollectLeft)
        .reset_state()
        .build_exec()?;
    Ok(new_join)
}

/// Return the exact build-side row count if known.
///
/// `RepartitionExec` and `CoalescePartitionsExec` do not change the total
/// number of rows, so we look through them to find a child with exact
/// statistics.
fn build_side_row_count(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    // Recurse through no-op (row-count-preserving) operators first; their own
    // statistics are often inexact even when the child's statistics are exact.
    if let Some(repart) = plan.downcast_ref::<RepartitionExec>() {
        return build_side_row_count(repart.input());
    }
    if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
        return build_side_row_count(coalesce.input());
    }

    // Then try the plan's global/partition statistics.
    if let Ok(stats) = plan.partition_statistics(None) {
        if let Precision::Exact(n) = stats.num_rows {
            return Some(n);
        }
    }

    None
}
