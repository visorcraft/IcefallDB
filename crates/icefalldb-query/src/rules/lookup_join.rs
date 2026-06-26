//! Physical optimizer rule that rewrites tiny-build-side hash joins into lookups.
//!
//! When the build side of an inner equi-join is small (by default at most
//! `lookup_join_threshold` rows), this rule materializes the build side into a
//! `HashMap` keyed by join-key `ScalarValue` and replaces the `HashJoinExec`
//! with a `LookupJoinExec`.  This avoids building a hash table at execution time
//! and lets the probe side stream through a simple key lookup.

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::DataType;
use datafusion::common::config::ConfigOptions;
use datafusion::common::NullEquality;
use datafusion::common::Result as DFResult;
use datafusion::common::ScalarValue;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::projection::ProjectionExpr;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::ExecutionPlan;
use futures::StreamExt;

use crate::execution::LookupJoinExec;
use crate::rules::dynamic_filter_pushdown::{build_side_row_count, extract_equi_keys};
use crate::session::IcefallDBConfig;

/// Physical optimizer rule that rewrites tiny-build-side hash joins into lookups.
#[derive(Debug, Default)]
pub struct LookupJoin;

impl LookupJoin {
    /// Create a new `LookupJoin` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for LookupJoin {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let enabled = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.lookup_join)
            .unwrap_or(true);
        if !enabled {
            return Ok(plan);
        }

        let threshold = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.lookup_join_threshold)
            .unwrap_or(4096);

        // Recursively apply the rule bottom-up.
        let plan = crate::rules::optimize_children(plan, config, |plan, config| {
            self.optimize(plan, config)
        })?;

        transform(plan, threshold)
    }

    fn name(&self) -> &str {
        "lookup_join"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn transform(plan: Arc<dyn ExecutionPlan>, threshold: usize) -> DFResult<Arc<dyn ExecutionPlan>> {
    let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
        return Ok(plan);
    };

    if join.filter().is_some() {
        tracing::debug!("LookupJoin rule bailing: HashJoinExec has a non-equi JoinFilter");
        return Ok(plan);
    }

    if join.null_equality() != NullEquality::NullEqualsNothing {
        tracing::debug!(
            null_equality = ?join.null_equality(),
            "LookupJoin rule bailing: unsupported null equality"
        );
        return Ok(plan);
    }

    let Some((build_key, probe_key)) = extract_equi_keys(join)? else {
        return Ok(plan);
    };

    // DataFusion's HashJoinExec uses left as build and right as probe.
    let build = join.left();
    let probe = join.right();

    let Some(build_rows) = build_side_row_count(build) else {
        return Ok(plan);
    };
    if build_rows > threshold {
        return Ok(plan);
    }

    let Some(build_batches) = execute_build_side(build) else {
        return Ok(plan);
    };
    let Some(build_keys) = extract_build_keys(&build_batches, &build_key) else {
        return Ok(plan);
    };

    if (probe_key.as_ref() as &dyn std::any::Any)
        .downcast_ref::<Column>()
        .is_none()
    {
        return Ok(plan);
    }
    let probe_key_expr = Arc::clone(&probe_key);

    // Use the full join schema (build columns followed by probe columns).  If
    // the original `HashJoinExec` had a projection pushed into it, wrap the
    // lookup join in a matching `ProjectionExec` so the output schema is
    // preserved.
    let full_output_schema = Arc::clone(join.join_schema());
    let build_schema = Arc::clone(&build.schema());

    let exec = LookupJoinExec::try_new(
        Arc::clone(probe),
        probe_key_expr,
        build_keys,
        build_schema,
        build_batches,
        full_output_schema,
    )?;
    let mut plan: Arc<dyn ExecutionPlan> = Arc::new(exec);

    if join.contains_projection() {
        let projected_schema = join.schema();
        let projection = join
            .projection
            .as_ref()
            .expect("contains_projection returned true");
        let exprs: Vec<ProjectionExpr> = projection
            .iter()
            .enumerate()
            .map(|(out_idx, &in_idx)| {
                let input_name = plan.schema().field(in_idx).name().clone();
                let output_name = projected_schema.field(out_idx).name().clone();
                ProjectionExpr {
                    expr: Arc::new(Column::new(&input_name, in_idx)) as Arc<dyn PhysicalExpr>,
                    alias: output_name,
                }
            })
            .collect();
        plan = Arc::new(ProjectionExec::try_new(exprs, plan)?);
    }

    Ok(plan)
}

/// Execute the build side and collect all batches.
///
/// Runs the build plan on a dedicated thread with a fresh current-thread Tokio
/// runtime so the optimizer can be invoked from within an async task without
/// deadlocking.
fn execute_build_side(build: &Arc<dyn ExecutionPlan>) -> Option<Vec<RecordBatch>> {
    let build = Arc::clone(build);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().build().ok()?;
        rt.block_on(async {
            let runtime = Arc::new(RuntimeEnv::default());
            let ctx: Arc<TaskContext> = Arc::new(TaskContext::default().with_runtime(runtime));
            let mut batches = Vec::new();
            for partition in 0..build.properties().partitioning.partition_count() {
                let mut stream = build.execute(partition, Arc::clone(&ctx)).ok()?;
                while let Some(batch) = stream.next().await {
                    batches.push(batch.ok()?);
                }
            }
            Some(batches)
        })
    })
    .join()
    .ok()?
}

/// Extract one join-key scalar value per build row.
///
/// Only `Int32`, `Int64`, `Utf8`, and `LargeUtf8` keys are supported; other
/// types cause the rule to bail out.
fn extract_build_keys(
    batches: &[RecordBatch],
    key_expr: &Arc<dyn PhysicalExpr>,
) -> Option<Vec<ScalarValue>> {
    let col = (key_expr.as_ref() as &dyn std::any::Any).downcast_ref::<Column>()?;
    if batches.is_empty() {
        return Some(Vec::new());
    }

    let schema = batches[0].schema();
    let key_type = schema.field(col.index()).data_type();
    if !matches!(
        key_type,
        DataType::Int32 | DataType::Int64 | DataType::Utf8 | DataType::LargeUtf8
    ) {
        return None;
    }

    let mut keys = Vec::new();
    for batch in batches {
        let array: &ArrayRef = batch.column(col.index());
        for row in 0..array.len() {
            keys.push(ScalarValue::try_from_array(array, row).ok()?);
        }
    }
    Some(keys)
}
