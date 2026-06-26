//! Physical optimizer rule that rewrites low-cardinality group-by aggregates
//! into single-phase aggregates.
//!
//! When a two-phase `FinalPartitioned -> Partial` aggregate has a small
//! estimated group count, the rule collapses it into a `Single` aggregate over
//! a coalesced input, avoiding repartition and merge overhead.

use std::sync::Arc;

use datafusion::common::config::ConfigOptions;
use datafusion::common::{ColumnStatistics, Result as DFResult, ScalarValue, Statistics};
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::session::IcefallDBConfig;

/// Physical optimizer rule that rewrites low-cardinality group-by aggregates
/// into single-phase aggregates.
#[derive(Debug, Default)]
pub struct LowCardinalityGroupBy;

impl LowCardinalityGroupBy {
    /// Create a new `LowCardinalityGroupBy` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for LowCardinalityGroupBy {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let icefalldb_config = config.extensions.get::<IcefallDBConfig>();
        let enabled = icefalldb_config
            .map(|c| c.low_cardinality_group_by)
            .unwrap_or(true);
        if !enabled {
            return Ok(plan);
        }
        let threshold = icefalldb_config
            .map(|c| c.low_cardinality_group_by_threshold)
            .unwrap_or(4096);

        // Recursively apply the rule bottom-up.
        let plan = crate::rules::optimize_children(plan, config, |plan, config| {
            self.optimize(plan, config)
        })?;
        transform(plan, threshold)
    }

    fn name(&self) -> &str {
        "low_cardinality_group_by"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

/// Try to rewrite a `FinalPartitioned -> Partial` aggregate into a single-phase
/// aggregate when the group-by cardinality is known to be low.
fn transform(plan: Arc<dyn ExecutionPlan>, threshold: usize) -> DFResult<Arc<dyn ExecutionPlan>> {
    let Some(final_agg) = plan.downcast_ref::<AggregateExec>() else {
        return Ok(plan);
    };
    if !matches!(final_agg.mode(), AggregateMode::FinalPartitioned) {
        return Ok(plan);
    }

    let child = final_agg.input();
    let Some(partial_agg) = child.downcast_ref::<AggregateExec>() else {
        return Ok(plan);
    };
    if !matches!(partial_agg.mode(), AggregateMode::Partial) {
        return Ok(plan);
    }

    // Avoid rewriting aggregates that are already single-phase, distinct, or
    // otherwise unsupported for collapse into a single `AggregateExec`.
    if !can_rewrite_aggregate(partial_agg) {
        return Ok(plan);
    }

    let partial_input = partial_agg.input();

    // Require exact row count statistics; otherwise be conservative and leave
    // the plan unchanged.
    let Ok(stats) = partial_input.partition_statistics(None) else {
        return Ok(plan);
    };
    let Some(num_rows) = stats.num_rows.get_value().copied() else {
        return Ok(plan);
    };
    if stats.num_rows.is_exact() != Some(true) {
        return Ok(plan);
    }

    let estimated_groups =
        estimate_group_count(partial_agg.group_expr(), partial_input, &stats, num_rows);
    let Some(estimated_groups) = estimated_groups else {
        return Ok(plan);
    };

    if estimated_groups > threshold {
        return Ok(plan);
    }

    let coalesced = Arc::new(CoalescePartitionsExec::new(Arc::clone(partial_input)));
    let single: Option<Arc<dyn ExecutionPlan>> = AggregateExec::try_new(
        AggregateMode::Single,
        partial_agg.group_expr().clone(),
        partial_agg.aggr_expr().to_vec(),
        partial_agg.filter_expr().to_vec(),
        coalesced,
        partial_agg.input_schema(),
    )
    .ok()
    .map(|agg| agg.with_limit_options(final_agg.limit_options()))
    .map(|agg| Arc::new(agg) as Arc<dyn ExecutionPlan>);

    Ok(single.unwrap_or(plan))
}

/// Return `false` for aggregates that cannot be safely collapsed into a single
/// `AggregateExec` over the raw input.
fn can_rewrite_aggregate(agg: &AggregateExec) -> bool {
    // Distinct aggregates require special per-partition handling and are not
    // candidates for this rewrite.
    if agg.aggr_expr().iter().any(|a| a.is_distinct()) {
        return false;
    }
    // Empty grouping is already handled by other rules; leave it untouched.
    if agg.group_expr().is_empty() {
        return false;
    }
    true
}

/// Estimate the number of distinct groups produced by `group_by` over
/// `input` given its `Statistics`.
///
/// Returns `None` if any required statistic is unknown, signalling the caller
/// to be conservative and skip the rewrite.
fn estimate_group_count(
    group_by: &datafusion::physical_plan::aggregates::PhysicalGroupBy,
    input: &Arc<dyn ExecutionPlan>,
    stats: &Statistics,
    num_rows: usize,
) -> Option<usize> {
    let schema = input.schema();
    let column_stats = &stats.column_statistics;

    let mut estimate: usize = 1;
    for (expr, _name) in group_by.expr() {
        let column_estimate = if let Some(col) = expr.downcast_ref::<Column>() {
            let field_idx = schema.index_of(col.name()).ok()?;
            let col_stats = column_stats.get(field_idx)?;
            estimate_column_groups(col_stats, schema.field(field_idx).data_type(), num_rows)?
        } else {
            num_rows
        };
        estimate = estimate.checked_mul(column_estimate)?;
    }

    // The number of distinct groups can never exceed the number of input rows.
    Some(estimate.min(num_rows))
}

/// Estimate the number of distinct values for a single group expression.
///
/// Integer columns with exact min/max statistics use `max - min + 1`. String
/// columns and columns without exact min/max fall back to the row count.
fn estimate_column_groups(
    col_stats: &ColumnStatistics,
    data_type: &arrow::datatypes::DataType,
    num_rows: usize,
) -> Option<usize> {
    if is_integer_type(data_type)
        && col_stats.min_value.is_exact() == Some(true)
        && col_stats.max_value.is_exact() == Some(true)
    {
        let min = col_stats.min_value.get_value()?;
        let max = col_stats.max_value.get_value()?;
        let min_i128 = scalar_to_i128(min)?;
        let max_i128 = scalar_to_i128(max)?;
        let range = max_i128.checked_sub(min_i128)?.checked_add(1)?;
        if range < 0 {
            return Some(0);
        }
        return Some((range as usize).min(num_rows));
    }
    // For non-integer or unbounded columns, assume every row could be distinct.
    Some(num_rows)
}

fn is_integer_type(data_type: &arrow::datatypes::DataType) -> bool {
    matches!(
        data_type,
        arrow::datatypes::DataType::Int8
            | arrow::datatypes::DataType::Int16
            | arrow::datatypes::DataType::Int32
            | arrow::datatypes::DataType::Int64
            | arrow::datatypes::DataType::UInt8
            | arrow::datatypes::DataType::UInt16
            | arrow::datatypes::DataType::UInt32
            | arrow::datatypes::DataType::UInt64
    )
}

fn scalar_to_i128(value: &ScalarValue) -> Option<i128> {
    match value {
        ScalarValue::Int8(Some(v)) => Some(i128::from(*v)),
        ScalarValue::Int16(Some(v)) => Some(i128::from(*v)),
        ScalarValue::Int32(Some(v)) => Some(i128::from(*v)),
        ScalarValue::Int64(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt8(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt16(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt32(Some(v)) => Some(i128::from(*v)),
        ScalarValue::UInt64(Some(v)) => Some(i128::from(*v)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion::functions_aggregate::sum::sum_udaf;
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    use datafusion::physical_expr::expressions::{col as phys_col, lit as phys_lit, Column};
    use datafusion::physical_expr::PhysicalExpr;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::aggregates::PhysicalGroupBy;
    use datafusion_datasource::memory::MemorySourceConfig;

    fn input_schema() -> Arc<ArrowSchema> {
        Arc::new(ArrowSchema::new(vec![
            Field::new("g", DataType::Int32, false),
            Field::new("v", DataType::Float64, true),
        ]))
    }

    fn make_batch(low: i32, high: i32) -> RecordBatch {
        let schema = input_schema();
        let values: Vec<i32> = (low..=high).collect();
        // Duplicate every value so the row count is larger than the cardinality.
        let g: Vec<i32> = values.iter().flat_map(|v| [*v, *v]).collect();
        let v: Vec<Option<f64>> = g.iter().map(|_| Some(1.0)).collect();
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(g)),
                Arc::new(Float64Array::from(v)),
            ],
        )
        .unwrap()
    }

    fn memory_source(batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
        MemorySourceConfig::try_new_exec(
            std::slice::from_ref(&vec![batch.clone()]),
            batch.schema(),
            None,
        )
        .unwrap()
    }

    fn sum_v_expr(
        schema: &Arc<ArrowSchema>,
    ) -> Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
        let col = phys_col("v", schema).unwrap();
        Arc::new(
            AggregateExprBuilder::new(sum_udaf(), vec![col])
                .schema(Arc::clone(schema))
                .alias("SUM(v)")
                .build()
                .unwrap(),
        )
    }

    fn count_star_expr(
        schema: &Arc<ArrowSchema>,
    ) -> Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
        let lit_one = phys_lit(ScalarValue::Int64(Some(1)));
        Arc::new(
            AggregateExprBuilder::new(count_udaf(), vec![lit_one])
                .schema(Arc::clone(schema))
                .alias("COUNT(*)")
                .build()
                .unwrap(),
        )
    }

    fn build_two_phase_aggregate(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let input_schema = input.schema();
        let aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> =
            vec![sum_v_expr(&input_schema), count_star_expr(&input_schema)];
        let group_exprs: Vec<(Arc<dyn PhysicalExpr>, String)> =
            vec![(phys_col("g", &input_schema).unwrap(), "g".into())];
        let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
            aggr_exprs.iter().map(|_| None).collect();

        let partial = AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::new_single(group_exprs.clone()),
            aggr_exprs.clone(),
            filter_exprs.clone(),
            input,
            Arc::clone(&input_schema),
        )
        .unwrap();
        let partial_schema = partial.schema();

        let mut final_aggr_exprs = Vec::with_capacity(aggr_exprs.len());
        let mut col_idx_base = group_exprs.len();
        for aggr in &aggr_exprs {
            let state_fields = aggr.state_fields().unwrap();
            let args: Vec<Arc<dyn PhysicalExpr>> = state_fields
                .iter()
                .enumerate()
                .map(|(idx, f)| Arc::new(Column::new(f.name(), col_idx_base + idx)) as _)
                .collect();
            col_idx_base += state_fields.len();

            let alias = aggr.field().name().to_string();
            let final_udaf = match aggr.fun().name().to_lowercase().as_str() {
                "count" => count_udaf(),
                "sum" => sum_udaf(),
                other => panic!("unsupported aggregate for two-phase test: {other}"),
            };
            final_aggr_exprs.push(Arc::new(
                AggregateExprBuilder::new(final_udaf, args)
                    .schema(Arc::clone(&partial_schema))
                    .alias(alias)
                    .build()
                    .unwrap(),
            ));
        }
        let final_filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
            final_aggr_exprs.iter().map(|_| None).collect();
        let final_group_exprs = vec![(phys_col("g", &partial_schema).unwrap(), "g".into())];

        Arc::new(
            AggregateExec::try_new(
                AggregateMode::FinalPartitioned,
                PhysicalGroupBy::new_single(final_group_exprs),
                final_aggr_exprs,
                final_filter_exprs,
                Arc::new(partial),
                Arc::clone(&partial_schema),
            )
            .unwrap(),
        )
    }

    fn apply_rule(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let rule = LowCardinalityGroupBy::new();
        rule.optimize(plan, &Default::default()).unwrap()
    }

    fn plan_contains_final_partitioned(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if let Some(agg) = plan.downcast_ref::<AggregateExec>() {
            if matches!(agg.mode(), AggregateMode::FinalPartitioned) {
                return true;
            }
        }
        plan.children()
            .iter()
            .any(|child| plan_contains_final_partitioned(child))
    }

    fn plan_contains_coalesce_partitions(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if (*plan).downcast_ref::<CoalescePartitionsExec>().is_some() {
            return true;
        }
        plan.children()
            .iter()
            .any(|child| plan_contains_coalesce_partitions(child))
    }

    #[test]
    fn test_low_cardinality_group_by_rewrites() {
        let input = memory_source(&make_batch(1, 4));
        let plan = build_two_phase_aggregate(input);
        let optimized = apply_rule(plan);
        assert!(
            !plan_contains_final_partitioned(&optimized),
            "FinalPartitioned should be removed"
        );
        assert!(
            plan_contains_coalesce_partitions(&optimized),
            "CoalescePartitionsExec should be present"
        );
    }

    #[test]
    fn test_high_cardinality_group_by_left_unchanged() {
        let input = memory_source(&make_batch(1, 8192));
        let plan = build_two_phase_aggregate(input);
        let optimized = apply_rule(plan);
        assert!(
            plan_contains_final_partitioned(&optimized),
            "FinalPartitioned should be preserved for high-cardinality groups"
        );
        assert!(
            !plan_contains_coalesce_partitions(&optimized),
            "CoalescePartitionsExec should not be introduced"
        );
    }

    #[test]
    fn test_rule_disabled_config() {
        let input = memory_source(&make_batch(1, 4));
        let plan = build_two_phase_aggregate(input);

        let mut config = ConfigOptions::default();
        let icefalldb_config = IcefallDBConfig {
            low_cardinality_group_by: false,
            ..Default::default()
        };
        config.extensions.insert(icefalldb_config);

        let rule = LowCardinalityGroupBy::new();
        let optimized = rule.optimize(plan, &config).unwrap();
        assert!(
            plan_contains_final_partitioned(&optimized),
            "rule should not fire when disabled in config"
        );
    }

    #[tokio::test]
    async fn test_rewritten_plan_matches_original() {
        let input = memory_source(&make_batch(1, 4));
        let plan = build_two_phase_aggregate(input);
        let original = collect_plan(plan.clone()).await;
        let optimized = collect_plan(apply_rule(plan)).await;
        assert_batches_eq(&original, &optimized);
    }

    async fn collect_plan(plan: Arc<dyn ExecutionPlan>) -> RecordBatch {
        use datafusion::execution::TaskContext;
        let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
        let batches = datafusion::physical_plan::common::collect(stream)
            .await
            .unwrap();
        arrow::compute::concat_batches(&plan.schema(), &batches).unwrap()
    }

    fn assert_batches_eq(left: &RecordBatch, right: &RecordBatch) {
        assert_eq!(left.num_rows(), right.num_rows(), "row count differs");
        assert_eq!(
            left.num_columns(),
            right.num_columns(),
            "column count differs"
        );
        for col_idx in 0..left.num_columns() {
            let l_col = left.column(col_idx);
            let r_col = right.column(col_idx);
            assert_eq!(l_col.data_type(), r_col.data_type(), "column type differs");
            assert_eq!(
                batch_col_to_strings(l_col),
                batch_col_to_strings(r_col),
                "column {col_idx} values differ"
            );
        }
    }

    fn batch_col_to_strings(col: &arrow::array::ArrayRef) -> Vec<String> {
        (0..col.len())
            .map(|i| ScalarValue::try_from_array(col, i).unwrap().to_string())
            .collect()
    }
}
