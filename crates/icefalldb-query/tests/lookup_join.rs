//! Integration tests for `LookupJoinExec`.

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::compute::{lexsort_to_indices, take, SortColumn};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::{JoinType, NullEquality, ScalarValue};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::joins::{HashJoinExec, JoinOn, PartitionMode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::memory::MemorySourceConfig;
use icefalldb_query::LookupJoinExec;

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

fn build_keys(batches: &[RecordBatch]) -> Vec<ScalarValue> {
    batches
        .iter()
        .flat_map(|batch| {
            let col = batch.column(0);
            (0..batch.num_rows())
                .map(|i| ScalarValue::try_from_array(col, i).unwrap())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn build_probe_plan(batches: Vec<RecordBatch>, schema: Arc<ArrowSchema>) -> Arc<dyn ExecutionPlan> {
    MemorySourceConfig::try_new_from_batches(schema, batches).unwrap()
}

fn build_probe_plan_partitioned(
    partitions: Vec<Vec<RecordBatch>>,
    schema: Arc<ArrowSchema>,
) -> Arc<dyn ExecutionPlan> {
    MemorySourceConfig::try_new_exec(&partitions, schema, None).unwrap()
}

fn build_hash_join(
    build_plan: Arc<dyn ExecutionPlan>,
    probe_plan: Arc<dyn ExecutionPlan>,
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
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )
        .unwrap(),
    )
}

fn build_lookup_join(
    probe_plan: Arc<dyn ExecutionPlan>,
    build_batches: Vec<RecordBatch>,
    output_schema: Arc<ArrowSchema>,
) -> Arc<dyn ExecutionPlan> {
    let build_schema = build_schema();
    let probe_key_expr: Arc<dyn PhysicalExpr> = Arc::new(Column::new("id", 0));
    let keys = build_keys(&build_batches);

    Arc::new(
        LookupJoinExec::try_new(
            probe_plan,
            probe_key_expr,
            keys,
            build_schema,
            build_batches,
            output_schema,
        )
        .unwrap(),
    )
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
async fn test_lookup_join_matches_hash_join() {
    let build = build_batches();
    let probe = probe_batches();

    let build_plan = build_probe_plan(build.clone(), build_schema());
    let probe_plan = build_probe_plan(probe.clone(), probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan.clone());
    let output_schema = hash_join.schema();

    let lookup_join = build_lookup_join(probe_plan, build, output_schema);

    let expected = collect_sorted(hash_join).await.unwrap();
    let actual = collect_sorted(lookup_join).await.unwrap();
    assert_batches_eq(&expected, &actual);
}

#[tokio::test]
async fn test_lookup_join_multi_match_build_key() {
    // Build side has duplicate keys; probe rows with key 3 produce two output rows.
    let build = build_batches();
    let probe = vec![RecordBatch::try_new(
        probe_schema(),
        vec![
            Arc::new(Int32Array::from(vec![3, 3])),
            Arc::new(Int64Array::from(vec![30, 31])),
        ],
    )
    .unwrap()];

    let build_plan = build_probe_plan(build.clone(), build_schema());
    let probe_plan = build_probe_plan(probe.clone(), probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan.clone());
    let output_schema = hash_join.schema();

    let lookup_join = build_lookup_join(probe_plan, build, output_schema);

    let expected = collect_sorted(hash_join).await.unwrap();
    let actual = collect_sorted(lookup_join).await.unwrap();
    assert_batches_eq(&expected, &actual);
    assert_eq!(actual.num_rows(), 4, "expected four output rows");
}

#[tokio::test]
async fn test_lookup_join_empty_build_side() {
    let build_schema = build_schema();
    let empty_build = vec![RecordBatch::try_new(
        Arc::clone(&build_schema),
        vec![
            Arc::new(Int32Array::from(Vec::<i32>::new())),
            Arc::new(StringArray::from(Vec::<&str>::new())),
        ],
    )
    .unwrap()];

    let probe = probe_batches();
    let build_plan = build_probe_plan(empty_build.clone(), build_schema);
    let probe_plan = build_probe_plan(probe.clone(), probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan.clone());
    let output_schema = hash_join.schema();

    let lookup_join = build_lookup_join(probe_plan, empty_build, output_schema);

    let expected = collect_sorted(hash_join).await.unwrap();
    let actual = collect_sorted(lookup_join).await.unwrap();
    assert_batches_eq(&expected, &actual);
    assert_eq!(actual.num_rows(), 0, "expected zero output rows");
}

#[tokio::test]
async fn test_lookup_join_multiple_probe_partitions() {
    let build = build_batches();

    // Split probe rows across two partitions.
    let probe_batch1 = RecordBatch::try_new(
        probe_schema(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![10, 20])),
        ],
    )
    .unwrap();
    let probe_batch2 = RecordBatch::try_new(
        probe_schema(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(Int64Array::from(vec![30, 40])),
        ],
    )
    .unwrap();
    let probe_partitions = vec![vec![probe_batch1], vec![probe_batch2]];

    let build_plan = build_probe_plan(build.clone(), build_schema());
    let probe_plan = build_probe_plan_partitioned(probe_partitions, probe_schema());

    let hash_join = build_hash_join(build_plan, probe_plan.clone());
    let output_schema = hash_join.schema();

    let lookup_join = build_lookup_join(probe_plan, build, output_schema);

    let expected = collect_sorted(hash_join).await.unwrap();
    let actual = collect_sorted(lookup_join).await.unwrap();
    assert_batches_eq(&expected, &actual);
}
