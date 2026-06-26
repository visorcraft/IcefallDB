//! Integration tests for `StreamingGroupByExec`.

use std::sync::Arc;

use arrow::array::{Decimal128Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::compute::{lexsort_to_indices, take, SortColumn};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::execution::TaskContext;
use datafusion::functions_aggregate::average::avg_udaf;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_expr::expressions::{col as phys_col, lit as phys_lit};
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::memory::MemorySourceConfig;
use icefalldb_query::{AggType, StreamingGroupByExec};

fn input_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Float64, true),
    ]))
}

fn multi_key_input_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("sub", DataType::Utf8, false),
        Field::new("value", DataType::Float64, true),
    ]))
}

/// Build sorted input batches that split groups across batch boundaries.
fn sorted_batches() -> Vec<RecordBatch> {
    let schema = input_schema();
    vec![
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None])),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["b", "b", "c"])),
                Arc::new(Float64Array::from(vec![Some(3.0), Some(4.0), Some(5.0)])),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["c", "c"])),
                Arc::new(Float64Array::from(vec![Some(6.0), None])),
            ],
        )
        .unwrap(),
    ]
}

/// Build sorted input with multiple group keys.
fn multi_key_batches() -> Vec<RecordBatch> {
    let schema = multi_key_input_schema();
    vec![
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "b"])),
                Arc::new(StringArray::from(vec!["x", "y", "x"])),
                Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), None])),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["b", "b", "c"])),
                Arc::new(StringArray::from(vec!["x", "y", "x"])),
                Arc::new(Float64Array::from(vec![Some(3.0), Some(4.0), Some(5.0)])),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["c", "c"])),
                Arc::new(StringArray::from(vec!["x", "y"])),
                Arc::new(Float64Array::from(vec![Some(6.0), None])),
            ],
        )
        .unwrap(),
    ]
}

fn build_aggregate_exec(
    schema: Arc<ArrowSchema>,
    batches: Vec<RecordBatch>,
    group_names: Vec<&str>,
    aggr_specs: Vec<(
        Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
        &str,
    )>,
) -> Arc<dyn ExecutionPlan> {
    let input: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_from_batches(Arc::clone(&schema), batches).unwrap();

    let group_exprs: Vec<(Arc<dyn datafusion::physical_expr::PhysicalExpr>, String)> = group_names
        .iter()
        .map(|name| {
            let expr = phys_col(name, &schema).unwrap();
            (expr, name.to_string())
        })
        .collect();

    let aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> =
        aggr_specs
            .iter()
            .map(|(expr, _)| Arc::clone(expr))
            .collect();
    let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        aggr_exprs.iter().map(|_| None).collect();

    let aggregate = AggregateExec::try_new(
        AggregateMode::Single,
        PhysicalGroupBy::new_single(group_exprs),
        aggr_exprs,
        filter_exprs,
        input,
        Arc::clone(&schema),
    )
    .unwrap();

    Arc::new(aggregate)
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

fn count_value_expr(
    schema: &Arc<ArrowSchema>,
) -> Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
    let col = phys_col("value", schema).unwrap();
    Arc::new(
        AggregateExprBuilder::new(count_udaf(), vec![col])
            .schema(Arc::clone(schema))
            .alias("COUNT(value)")
            .build()
            .unwrap(),
    )
}

fn sum_value_expr(
    schema: &Arc<ArrowSchema>,
) -> Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
    let col = phys_col("value", schema).unwrap();
    Arc::new(
        AggregateExprBuilder::new(sum_udaf(), vec![col])
            .schema(Arc::clone(schema))
            .alias("SUM(value)")
            .build()
            .unwrap(),
    )
}

fn avg_value_expr(
    schema: &Arc<ArrowSchema>,
) -> Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
    let col = phys_col("value", schema).unwrap();
    Arc::new(
        AggregateExprBuilder::new(avg_udaf(), vec![col])
            .schema(Arc::clone(schema))
            .alias("AVG(value)")
            .build()
            .unwrap(),
    )
}

fn build_streaming_exec(
    schema: Arc<ArrowSchema>,
    batches: Vec<RecordBatch>,
    group_names: Vec<&str>,
    aggr_specs: Vec<(AggType, &str, &str)>,
    output_schema: Arc<ArrowSchema>,
) -> Arc<dyn ExecutionPlan> {
    let input: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_from_batches(Arc::clone(&schema), batches).unwrap();

    let group_exprs: Vec<Arc<dyn datafusion::physical_expr::PhysicalExpr>> = group_names
        .iter()
        .map(|name| phys_col(name, &schema).unwrap())
        .collect();

    let aggr_exprs: Vec<(AggType, Arc<dyn datafusion::physical_expr::PhysicalExpr>)> = aggr_specs
        .iter()
        .map(|(typ, col_name, _alias)| {
            let expr: Arc<dyn datafusion::physical_expr::PhysicalExpr> = if *col_name == "*" {
                phys_lit(ScalarValue::Int64(Some(1)))
            } else {
                phys_col(col_name, &schema).unwrap()
            };
            (*typ, expr)
        })
        .collect();

    Arc::new(StreamingGroupByExec::try_new(input, group_exprs, aggr_exprs, output_schema).unwrap())
}

async fn collect_sorted(exec: Arc<dyn ExecutionPlan>, group_key_count: usize) -> RecordBatch {
    let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();

    let batch = arrow::compute::concat_batches(&exec.schema(), &batches).unwrap();
    sort_batch_by_prefix(batch, group_key_count)
}

fn sort_batch_by_prefix(batch: RecordBatch, prefix_len: usize) -> RecordBatch {
    let sort_columns: Vec<SortColumn> = (0..prefix_len)
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

#[tokio::test]
async fn test_streaming_group_by_matches_hash_aggregate() {
    let schema = input_schema();
    let batches = sorted_batches();

    let aggregate_exec = build_aggregate_exec(
        Arc::clone(&schema),
        batches.clone(),
        vec!["category"],
        vec![
            (count_star_expr(&schema), "COUNT(*)"),
            (count_value_expr(&schema), "COUNT(value)"),
            (sum_value_expr(&schema), "SUM(value)"),
            (avg_value_expr(&schema), "AVG(value)"),
        ],
    );
    let output_schema = aggregate_exec.schema();

    let streaming_exec = build_streaming_exec(
        Arc::clone(&schema),
        batches,
        vec!["category"],
        vec![
            (AggType::Count, "*", "COUNT(*)"),
            (AggType::Count, "value", "COUNT(value)"),
            (AggType::Sum, "value", "SUM(value)"),
            (AggType::Avg, "value", "AVG(value)"),
        ],
        output_schema,
    );

    let expected = collect_sorted(aggregate_exec, 1).await;
    let actual = collect_sorted(streaming_exec, 1).await;
    assert_batches_eq(&expected, &actual);
}

#[tokio::test]
async fn test_streaming_group_by_multiple_keys() {
    let schema = multi_key_input_schema();
    let batches = multi_key_batches();

    let aggregate_exec = build_aggregate_exec(
        Arc::clone(&schema),
        batches.clone(),
        vec!["category", "sub"],
        vec![
            (count_star_expr(&schema), "COUNT(*)"),
            (sum_value_expr(&schema), "SUM(value)"),
            (avg_value_expr(&schema), "AVG(value)"),
        ],
    );
    let output_schema = aggregate_exec.schema();

    let streaming_exec = build_streaming_exec(
        Arc::clone(&schema),
        batches,
        vec!["category", "sub"],
        vec![
            (AggType::Count, "*", "COUNT(*)"),
            (AggType::Sum, "value", "SUM(value)"),
            (AggType::Avg, "value", "AVG(value)"),
        ],
        output_schema,
    );

    let expected = collect_sorted(aggregate_exec, 2).await;
    let actual = collect_sorted(streaming_exec, 2).await;
    assert_batches_eq(&expected, &actual);
}

fn int_input_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Int64, true),
    ]))
}

fn decimal_input_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Decimal128(10, 2), true),
    ]))
}

async fn collect_batches(exec: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
    let stream = exec.execute(0, Arc::new(TaskContext::default())).unwrap();
    datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap()
}

async fn total_rows(exec: Arc<dyn ExecutionPlan>) -> usize {
    collect_batches(exec)
        .await
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

#[tokio::test]
async fn test_streaming_group_by_integer_sum_and_avg_exactness() {
    let schema = int_input_schema();
    // Values around 2^53 that are not exactly representable in f64.
    let base: i64 = 9_007_199_254_740_993; // 2^53 + 1
    let batches = vec![
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "a", "a"])),
                Arc::new(Int64Array::from(vec![
                    Some(base),
                    Some(base + 1),
                    Some(base + 2),
                ])),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "b"])),
                Arc::new(Int64Array::from(vec![Some(base + 3), Some(10), Some(20)])),
            ],
        )
        .unwrap(),
        RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(vec!["b", "c"])),
                Arc::new(Int64Array::from(vec![Some(30), Some(base * 2)])),
            ],
        )
        .unwrap(),
    ];

    // DataFusion 54 does not support AVG over Int64 in this configuration, so
    // we build the output schema manually and verify both SUM and AVG against
    // manually computed expectations.
    let output_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("SUM(value)", DataType::Int64, true),
        Field::new("AVG(value)", DataType::Float64, true),
    ]));

    let streaming_exec = build_streaming_exec(
        Arc::clone(&schema),
        batches,
        vec!["category"],
        vec![
            (AggType::Sum, "value", "SUM(value)"),
            (AggType::Avg, "value", "AVG(value)"),
        ],
        output_schema,
    );

    let actual = collect_sorted(streaming_exec, 1).await;
    assert_eq!(actual.num_rows(), 3, "expected three groups");

    let sum_array = actual
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let avg_array = actual
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();

    // Group a: base, base+1, base+2, base+3 -> sum = base*4 + 6, avg = sum/4
    let expected_a_sum = base * 4 + 6;
    assert_eq!(
        sum_array.value(0),
        expected_a_sum,
        "group a SUM must be exact"
    );
    assert!(
        (avg_array.value(0) - (expected_a_sum as f64 / 4.0)).abs() < f64::EPSILON,
        "group a AVG must be exact"
    );

    // Group b: 10, 20, 30 -> sum = 60, avg = 20.0
    assert_eq!(sum_array.value(1), 60, "group b SUM must be exact");
    assert!(
        (avg_array.value(1) - 20.0).abs() < f64::EPSILON,
        "group b AVG must be exact"
    );

    // Group c: base*2 -> sum = base*2, avg = base*2 as f64
    let expected_c_sum = base * 2;
    assert_eq!(
        sum_array.value(2),
        expected_c_sum,
        "group c SUM must be exact"
    );
    assert!(
        (avg_array.value(2) - expected_c_sum as f64).abs() < f64::EPSILON,
        "group c AVG must be exact"
    );
}

#[tokio::test]
async fn test_streaming_group_by_empty_input() {
    let schema = input_schema();
    let empty_batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(Vec::<&str>::new())),
            Arc::new(Float64Array::from(Vec::<Option<f64>>::new())),
        ],
    )
    .unwrap();
    let batches = vec![empty_batch];

    let aggregate_exec = build_aggregate_exec(
        Arc::clone(&schema),
        batches.clone(),
        vec!["category"],
        vec![(sum_value_expr(&schema), "SUM(value)")],
    );
    let output_schema = aggregate_exec.schema();

    let streaming_exec = build_streaming_exec(
        Arc::clone(&schema),
        batches,
        vec!["category"],
        vec![(AggType::Sum, "value", "SUM(value)")],
        output_schema,
    );

    assert_eq!(total_rows(streaming_exec).await, 0);
}

#[tokio::test]
async fn test_streaming_group_by_groups_span_many_batches() {
    let schema = input_schema();
    let mut batches: Vec<RecordBatch> = Vec::new();
    // 10 batches, each up to 3 rows, maintaining global sorted order.
    // Groups a, b, c span multiple consecutive batches.
    let group_sequence = [
        ("a", "a", None::<f64>),
        ("a", "b", Some(1.0)),
        ("b", "b", Some(2.0)),
        ("b", "c", Some(3.0)),
        ("c", "c", Some(4.0)),
        ("c", "d", Some(5.0)),
        ("d", "d", Some(6.0)),
        ("d", "e", Some(7.0)),
        ("e", "e", Some(8.0)),
        ("e", "e", Some(9.0)),
    ];
    for (g1, g2, v2) in group_sequence {
        let v1: Option<f64> = Some(1.0);
        batches.push(
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(vec![g1, g2])),
                    Arc::new(Float64Array::from(vec![v1, v2])),
                ],
            )
            .unwrap(),
        );
    }

    let aggregate_exec = build_aggregate_exec(
        Arc::clone(&schema),
        batches.clone(),
        vec!["category"],
        vec![
            (count_star_expr(&schema), "COUNT(*)"),
            (sum_value_expr(&schema), "SUM(value)"),
            (avg_value_expr(&schema), "AVG(value)"),
        ],
    );
    let output_schema = aggregate_exec.schema();

    let streaming_exec = build_streaming_exec(
        Arc::clone(&schema),
        batches,
        vec!["category"],
        vec![
            (AggType::Count, "*", "COUNT(*)"),
            (AggType::Sum, "value", "SUM(value)"),
            (AggType::Avg, "value", "AVG(value)"),
        ],
        output_schema,
    );

    let expected = collect_sorted(aggregate_exec, 1).await;
    let actual = collect_sorted(streaming_exec, 1).await;
    assert_batches_eq(&expected, &actual);
}

#[tokio::test]
async fn test_streaming_group_by_unsupported_decimal_type() {
    let schema = decimal_input_schema();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a"])),
            Arc::new(
                Decimal128Array::from(vec![Some(100), Some(200)])
                    .with_precision_and_scale(10, 2)
                    .unwrap(),
            ),
        ],
    )
    .unwrap();

    let input: Arc<dyn ExecutionPlan> =
        MemorySourceConfig::try_new_from_batches(Arc::clone(&schema), vec![batch]).unwrap();
    let group_exprs: Vec<Arc<dyn datafusion::physical_expr::PhysicalExpr>> =
        vec![phys_col("category", &schema).unwrap()];
    let aggr_exprs: Vec<(AggType, Arc<dyn datafusion::physical_expr::PhysicalExpr>)> =
        vec![(AggType::Sum, phys_col("value", &schema).unwrap())];

    let output_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("SUM(value)", DataType::Decimal128(10, 2), true),
    ]));

    let result = StreamingGroupByExec::try_new(input, group_exprs, aggr_exprs, output_schema);
    assert!(
        result.is_err(),
        "SUM over decimal should be rejected as unsupported"
    );
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("unsupported numeric type"),
        "error should mention unsupported numeric type, got: {err}"
    );
}
