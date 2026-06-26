//! Integration tests for the `LookupJoin` physical optimizer rule.

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::compute::{lexsort_to_indices, take, SortColumn};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::{JoinType, NullEquality, ScalarValue};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::joins::{HashJoinExec, JoinOn, PartitionMode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::memory::MemorySourceConfig;
use icefalldb_query::rules::LookupJoin;

fn build_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn probe_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn build_batches() -> Vec<RecordBatch> {
    let schema = build_schema();
    vec![RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 3])),
            Arc::new(StringArray::from(vec!["one", "two", "three-a", "three-b"])),
        ],
    )
    .unwrap()]
}

fn probe_batches() -> Vec<RecordBatch> {
    let schema = probe_schema();
    vec![RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap()]
}

fn build_memory_plan(
    batches: Vec<RecordBatch>,
    schema: Arc<ArrowSchema>,
) -> Arc<dyn ExecutionPlan> {
    MemorySourceConfig::try_new_from_batches(schema, batches).unwrap()
}

fn build_hash_join(
    build_plan: Arc<dyn ExecutionPlan>,
    probe_plan: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
) -> Arc<dyn ExecutionPlan> {
    build_hash_join_with_projection(build_plan, probe_plan, join_type, None)
}

fn build_hash_join_with_projection(
    build_plan: Arc<dyn ExecutionPlan>,
    probe_plan: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
    projection: Option<Vec<usize>>,
) -> Arc<dyn ExecutionPlan> {
    let on: JoinOn = vec![(
        Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
        Arc::new(Column::new("id", 0)) as Arc<dyn PhysicalExpr>,
    )];

    Arc::new(
        HashJoinExec::try_new(
            Arc::clone(&build_plan),
            Arc::clone(&probe_plan),
            on,
            None,
            &join_type,
            projection,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

fn apply_rule(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let rule = LookupJoin::new();
    rule.optimize(plan, &Default::default()).unwrap()
}

fn apply_rule_with_threshold(
    plan: Arc<dyn ExecutionPlan>,
    threshold: usize,
) -> Arc<dyn ExecutionPlan> {
    let mut config = datafusion::common::config::ConfigOptions::default();
    let mut icefalldb_config = icefalldb_query::IcefallDBConfig::default();
    icefalldb_config.lookup_join_threshold = threshold;
    config.extensions.insert(icefalldb_config);

    let rule = LookupJoin::new();
    rule.optimize(plan, &config).unwrap()
}

fn plan_contains_lookup_join(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.name() == "LookupJoinExec" {
        return true;
    }
    plan.children()
        .iter()
        .any(|child| plan_contains_lookup_join(child))
}

async fn collect_sorted(
    exec: Arc<dyn ExecutionPlan>,
) -> Result<RecordBatch, datafusion::error::DataFusionError> {
    let ctx = Arc::new(TaskContext::default());
    let partitioned =
        datafusion::physical_plan::collect_partitioned(Arc::clone(&exec), ctx).await?;
    let batches: Vec<RecordBatch> = partitioned.into_iter().flatten().collect();
    let combined = arrow::compute::concat_batches(&exec.schema(), &batches)?;
    Ok(sort_batch(combined))
}

fn sort_batch(batch: RecordBatch) -> RecordBatch {
    let sort_columns: Vec<SortColumn> = (0..batch.num_columns())
        .map(|i| SortColumn {
            values: batch.column(i).clone(),
            options: None,
        })
        .collect();
    let indices = lexsort_to_indices(&sort_columns, None).unwrap();
    let columns: Vec<_> = batch
        .columns()
        .iter()
        .map(|col| take(col.as_ref(), &indices, None).unwrap())
        .collect();
    RecordBatch::try_new(batch.schema(), columns).unwrap()
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
            col_to_strings(l_col),
            col_to_strings(r_col),
            "column {col_idx} values differ"
        );
    }
}

fn col_to_strings(col: &arrow::array::ArrayRef) -> Vec<String> {
    (0..col.len())
        .map(|i| ScalarValue::try_from_array(col, i).unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn test_lookup_join_rule_fires_on_tiny_build() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_memory_plan(build, build_schema());
    let probe_plan = build_memory_plan(probe, probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan, JoinType::Inner);
    let optimized = apply_rule(hash_join);

    assert!(
        plan_contains_lookup_join(&optimized),
        "optimized plan should contain LookupJoinExec for a tiny build side"
    );
}

#[tokio::test]
async fn test_lookup_join_rule_does_not_fire_when_build_too_large() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_memory_plan(build, build_schema());
    let probe_plan = build_memory_plan(probe, probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan, JoinType::Inner);
    // Build side has 4 rows; set threshold below that.
    let optimized = apply_rule_with_threshold(hash_join, 2);

    assert!(
        !plan_contains_lookup_join(&optimized),
        "rule should not fire when build side exceeds threshold"
    );
}

#[tokio::test]
async fn test_lookup_join_rule_does_not_fire_for_non_inner_join() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_memory_plan(build, build_schema());
    let probe_plan = build_memory_plan(probe, probe_schema());

    let left_join = build_hash_join(build_plan, probe_plan, JoinType::Left);
    let optimized = apply_rule(left_join);

    assert!(
        !plan_contains_lookup_join(&optimized),
        "rule should not fire for non-inner joins"
    );
}

#[tokio::test]
async fn test_lookup_join_rule_matches_hash_join() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_memory_plan(build.clone(), build_schema());
    let probe_plan = build_memory_plan(probe.clone(), probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan.clone(), JoinType::Inner);
    let optimized = apply_rule(hash_join);
    assert!(plan_contains_lookup_join(&optimized));

    // Compare the optimized join against a fresh hash join to verify correctness.
    let original_hash_join = build_hash_join(
        build_memory_plan(build, build_schema()),
        build_memory_plan(probe, probe_schema()),
        JoinType::Inner,
    );

    let expected = collect_sorted(original_hash_join).await.unwrap();
    let actual = collect_sorted(optimized).await.unwrap();
    assert_batches_eq(&expected, &actual);
}

#[tokio::test]
async fn test_lookup_join_rule_disabled_config() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_memory_plan(build, build_schema());
    let probe_plan = build_memory_plan(probe, probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan, JoinType::Inner);

    let mut config = datafusion::common::config::ConfigOptions::default();
    let mut icefalldb_config = icefalldb_query::IcefallDBConfig::default();
    icefalldb_config.lookup_join = false;
    config.extensions.insert(icefalldb_config);

    let rule = LookupJoin::new();
    let optimized = rule.optimize(hash_join, &config).unwrap();

    assert!(
        !plan_contains_lookup_join(&optimized),
        "rule should not fire when disabled in config"
    );
}

#[tokio::test]
async fn test_lookup_join_rule_preserves_projected_output() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_memory_plan(build, build_schema());
    let probe_plan = build_memory_plan(probe, probe_schema());

    // Full join schema is [build.id, build.name, probe.id, probe.value].
    // Project to [probe.id, build.id] to verify the rule preserves both the
    // output schema and the results.
    let projection = Some(vec![2, 0]);
    let projected_join =
        build_hash_join_with_projection(build_plan, probe_plan, JoinType::Inner, projection);
    let optimized = apply_rule(Arc::clone(&projected_join));

    assert!(
        plan_contains_lookup_join(&optimized),
        "rule should fire for a projected tiny build side"
    );

    let expected = collect_sorted(projected_join).await.unwrap();
    let actual = collect_sorted(optimized).await.unwrap();

    assert_eq!(
        expected.schema(),
        actual.schema(),
        "optimized projected schema should match HashJoinExec output schema"
    );
    assert_batches_eq(&expected, &actual);
}
