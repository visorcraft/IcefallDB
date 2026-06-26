//! Integration tests for the `SortedGroupBy` physical optimizer rule.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::execution::TaskContext;
use datafusion::functions_aggregate::average::avg_udaf;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::min_max::min_udaf;
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_expr::expressions::{
    col as phys_col, is_not_null, lit as phys_lit, Column,
};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::ExecutionPlan;
use icefalldb_core::metadata::{ColumnChunkOffset, ColumnStats, RowGroupMeta};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::PlannedRowGroup;
use icefalldb_query::rules::SortedGroupBy;

fn input_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Float64, true),
    ]))
}

fn sorted_batch() -> RecordBatch {
    let schema = input_schema();
    RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "a", "b", "b", "c", "c"])),
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(2.0),
                Some(3.0),
                None,
                Some(5.0),
                Some(6.0),
            ])),
        ],
    )
    .unwrap()
}

fn write_parquet(batch: &RecordBatch) -> Vec<u8> {
    let props = parquet::file::properties::WriterProperties::builder()
        .set_dictionary_enabled(false)
        .build();
    let mut buf = Vec::new();
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    buf
}

fn compute_offsets(bytes: &[u8], names: &[&str]) -> Option<HashMap<String, ColumnChunkOffset>> {
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        bytes::Bytes::from(bytes.to_vec()),
    )
    .ok()?;
    let metadata = builder.metadata();
    let rg = metadata.row_groups().first()?;
    let schema_descr = metadata.file_metadata().schema_descr();
    let mut map = HashMap::new();
    for name in names {
        let leaf_idx = schema_descr
            .columns()
            .iter()
            .position(|c| c.name() == *name)?;
        let col_meta = rg.column(leaf_idx);
        map.insert(
            (*name).to_string(),
            ColumnChunkOffset {
                offset: col_meta.data_page_offset().max(0) as u64,
                length: col_meta.compressed_size().max(0) as u64,
            },
        );
    }
    Some(map)
}

async fn make_scan_exec(sort: Option<Vec<String>>) -> icefalldb_query::scan::IcefallDBScanExec {
    let storage: Arc<dyn icefalldb_core::storage::Storage> = Arc::new(MemoryStorage::new());
    let batch = sorted_batch();
    let schema = batch.schema();
    let bytes = write_parquet(&batch);
    storage.write("test/rg.parquet", &bytes).await.unwrap();

    let offsets = compute_offsets(&bytes, &["category", "value"]);
    let rg = PlannedRowGroup {
        data_path: "test/rg.parquet".into(),
        meta_path: "test/rg.meta".into(),
        meta: RowGroupMeta {
            row_group: "rg".into(),
            schema_id: 1,
            rows: batch.num_rows(),
            columns: [
                (
                    "category".into(),
                    ColumnStats {
                        min: None,
                        max: None,
                        nulls: 0,
                    },
                ),
                (
                    "value".into(),
                    ColumnStats {
                        min: None,
                        max: None,
                        nulls: 1,
                    },
                ),
            ]
            .into(),
            column_offsets: offsets,
            sort,
            row_ids: vec![],
            checksum: String::new(),
            meta_checksum: String::new(),
        },
        partition_values: None,
        snapshot: 0,
        fallback: false,
        deletes: None,
        deleted_count: 0,
        fragment_id: 0,
        agg_state: None,
    };

    icefalldb_query::scan::IcefallDBScanExec::new(
        storage,
        Arc::clone(&schema),
        vec![rg],
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap()
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

fn min_value_expr(
    schema: &Arc<ArrowSchema>,
) -> Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr> {
    let col = phys_col("value", schema).unwrap();
    Arc::new(
        AggregateExprBuilder::new(min_udaf(), vec![col])
            .schema(Arc::clone(schema))
            .alias("MIN(value)")
            .build()
            .unwrap(),
    )
}

fn build_aggregate_exec(
    scan: icefalldb_query::scan::IcefallDBScanExec,
    mode: AggregateMode,
    aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>>,
) -> Arc<dyn ExecutionPlan> {
    build_aggregate_exec_with_input(Arc::new(scan), mode, aggr_exprs)
}

fn build_aggregate_exec_with_input(
    input: Arc<dyn ExecutionPlan>,
    mode: AggregateMode,
    aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>>,
) -> Arc<dyn ExecutionPlan> {
    let input_schema = input.schema();
    let group_exprs: Vec<(Arc<dyn datafusion::physical_expr::PhysicalExpr>, String)> = vec![(
        phys_col("category", &input_schema).unwrap(),
        "category".into(),
    )];
    let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        aggr_exprs.iter().map(|_| None).collect();

    Arc::new(
        AggregateExec::try_new(
            mode,
            PhysicalGroupBy::new_single(group_exprs),
            aggr_exprs,
            filter_exprs,
            input,
            input_schema,
        )
        .unwrap(),
    )
}

fn build_two_phase_aggregate_exec(
    scan: icefalldb_query::scan::IcefallDBScanExec,
    aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>>,
) -> Arc<dyn ExecutionPlan> {
    build_two_phase_aggregate_exec_with_input(Arc::new(scan), aggr_exprs)
}

fn build_two_phase_aggregate_exec_with_input(
    input: Arc<dyn ExecutionPlan>,
    aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>>,
) -> Arc<dyn ExecutionPlan> {
    let input_schema = input.schema();
    let group_exprs: Vec<(Arc<dyn datafusion::physical_expr::PhysicalExpr>, String)> = vec![(
        phys_col("category", &input_schema).unwrap(),
        "category".into(),
    )];
    let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
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
        let args: Vec<Arc<dyn datafusion::physical_expr::PhysicalExpr>> = state_fields
            .iter()
            .enumerate()
            .map(|(idx, f)| Arc::new(Column::new(f.name(), col_idx_base + idx)) as _)
            .collect();
        col_idx_base += state_fields.len();

        let alias = aggr.field().name().to_string();
        let final_udaf = match aggr.fun().name().to_lowercase().as_str() {
            "count" => count_udaf(),
            "sum" => sum_udaf(),
            "avg" => avg_udaf(),
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
    let final_filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        final_aggr_exprs.iter().map(|_| None).collect();
    let final_group_exprs = vec![(
        phys_col("category", &partial_schema).unwrap(),
        "category".into(),
    )];

    Arc::new(
        AggregateExec::try_new(
            AggregateMode::Final,
            PhysicalGroupBy::new_single(final_group_exprs),
            final_aggr_exprs,
            final_filter_exprs,
            Arc::new(partial),
            Arc::clone(&partial_schema),
        )
        .unwrap(),
    )
}

fn wrap_filter(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let predicate = is_not_null(phys_col("value", &input.schema()).unwrap()).unwrap();
    Arc::new(FilterExec::try_new(predicate, input).unwrap())
}

fn apply_rule(plan: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    let rule = SortedGroupBy::new();
    rule.optimize(plan, &Default::default()).unwrap()
}

fn plan_contains_streaming_group_by(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.name() == "StreamingGroupByExec" {
        return true;
    }
    plan.children()
        .iter()
        .any(|child| plan_contains_streaming_group_by(child))
}

async fn collect_plan(plan: Arc<dyn ExecutionPlan>) -> RecordBatch {
    let stream = plan.execute(0, Arc::new(TaskContext::default())).unwrap();
    let batches = datafusion::physical_plan::common::collect(stream)
        .await
        .unwrap();
    arrow::compute::concat_batches(&plan.schema(), &batches).unwrap()
}

fn sort_batch_by_prefix(batch: RecordBatch, prefix_len: usize) -> RecordBatch {
    use arrow::compute::{lexsort_to_indices, take, SortColumn};
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
async fn test_sorted_group_by_rule_single_fires() {
    let scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&scan.schema()),
        count_value_expr(&scan.schema()),
        sum_value_expr(&scan.schema()),
        avg_value_expr(&scan.schema()),
    ];
    let plan = build_aggregate_exec(scan, AggregateMode::Single, aggr_exprs);
    let optimized = apply_rule(plan);
    assert!(
        plan_contains_streaming_group_by(&optimized),
        "optimized plan should contain StreamingGroupByExec for Single aggregate"
    );
}

#[tokio::test]
async fn test_sorted_group_by_rule_final_partial_fires() {
    let scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&scan.schema()),
        count_value_expr(&scan.schema()),
        sum_value_expr(&scan.schema()),
        avg_value_expr(&scan.schema()),
    ];
    let plan = build_two_phase_aggregate_exec(scan, aggr_exprs);
    let optimized = apply_rule(plan);
    assert!(
        plan_contains_streaming_group_by(&optimized),
        "optimized plan should contain StreamingGroupByExec for Final -> Partial aggregate"
    );
}

#[tokio::test]
async fn test_sorted_group_by_rule_single_with_filter_fires() {
    let scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&scan.schema()),
        count_value_expr(&scan.schema()),
        sum_value_expr(&scan.schema()),
        avg_value_expr(&scan.schema()),
    ];
    let filtered = wrap_filter(Arc::new(scan));
    let plan = build_aggregate_exec_with_input(filtered, AggregateMode::Single, aggr_exprs);
    let optimized = apply_rule(plan);
    assert!(
        plan_contains_streaming_group_by(&optimized),
        "optimized plan should contain StreamingGroupByExec for Single -> Filter -> Scan"
    );
}

#[tokio::test]
async fn test_sorted_group_by_rule_final_partial_with_filter_fires() {
    let scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&scan.schema()),
        count_value_expr(&scan.schema()),
        sum_value_expr(&scan.schema()),
        avg_value_expr(&scan.schema()),
    ];
    let filtered = wrap_filter(Arc::new(scan));
    let plan = build_two_phase_aggregate_exec_with_input(filtered, aggr_exprs);
    let optimized = apply_rule(plan);
    assert!(
        plan_contains_streaming_group_by(&optimized),
        "optimized plan should contain StreamingGroupByExec for Final -> Partial -> Filter -> Scan"
    );
}

#[tokio::test]
async fn test_sorted_group_by_rule_does_not_fire_without_sort() {
    let scan = make_scan_exec(None).await;
    let aggr_exprs = vec![
        count_star_expr(&scan.schema()),
        sum_value_expr(&scan.schema()),
    ];
    let plan = build_aggregate_exec(scan, AggregateMode::Single, aggr_exprs);
    let optimized = apply_rule(plan);
    assert!(
        !plan_contains_streaming_group_by(&optimized),
        "rule should not fire when scan has no sort metadata"
    );
}

#[tokio::test]
async fn test_sorted_group_by_rule_does_not_fire_for_unsupported_aggregate() {
    let scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![min_value_expr(&scan.schema())];
    let plan = build_aggregate_exec(scan, AggregateMode::Single, aggr_exprs);
    let optimized = apply_rule(plan);
    assert!(
        !plan_contains_streaming_group_by(&optimized),
        "rule should not fire for unsupported aggregates like MIN"
    );
}

#[tokio::test]
async fn test_sorted_group_by_matches_hash_aggregate() {
    let sorted_scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&sorted_scan.schema()),
        count_value_expr(&sorted_scan.schema()),
        sum_value_expr(&sorted_scan.schema()),
        avg_value_expr(&sorted_scan.schema()),
    ];
    let optimized = apply_rule(build_two_phase_aggregate_exec(sorted_scan, aggr_exprs));

    let unsorted_scan = make_scan_exec(None).await;
    let aggr_exprs = vec![
        count_star_expr(&unsorted_scan.schema()),
        count_value_expr(&unsorted_scan.schema()),
        sum_value_expr(&unsorted_scan.schema()),
        avg_value_expr(&unsorted_scan.schema()),
    ];
    let reference = build_aggregate_exec(unsorted_scan, AggregateMode::Single, aggr_exprs);

    let expected = sort_batch_by_prefix(collect_plan(reference).await, 1);
    let actual = sort_batch_by_prefix(collect_plan(optimized).await, 1);
    assert_batches_eq(&expected, &actual);
}

#[tokio::test]
async fn test_sorted_group_by_matches_hash_aggregate_with_filter() {
    let sorted_scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&sorted_scan.schema()),
        count_value_expr(&sorted_scan.schema()),
        sum_value_expr(&sorted_scan.schema()),
        avg_value_expr(&sorted_scan.schema()),
    ];
    let sorted_filtered = wrap_filter(Arc::new(sorted_scan));
    let optimized = apply_rule(build_aggregate_exec_with_input(
        sorted_filtered,
        AggregateMode::Single,
        aggr_exprs,
    ));

    let unsorted_scan = make_scan_exec(None).await;
    let aggr_exprs = vec![
        count_star_expr(&unsorted_scan.schema()),
        count_value_expr(&unsorted_scan.schema()),
        sum_value_expr(&unsorted_scan.schema()),
        avg_value_expr(&unsorted_scan.schema()),
    ];
    let unsorted_filtered = wrap_filter(Arc::new(unsorted_scan));
    let reference =
        build_aggregate_exec_with_input(unsorted_filtered, AggregateMode::Single, aggr_exprs);

    let expected = sort_batch_by_prefix(collect_plan(reference).await, 1);
    let actual = sort_batch_by_prefix(collect_plan(optimized).await, 1);
    assert_batches_eq(&expected, &actual);
}

#[tokio::test]
async fn test_sorted_group_by_rule_disabled_config() {
    let scan = make_scan_exec(Some(vec!["category".to_string()])).await;
    let aggr_exprs = vec![
        count_star_expr(&scan.schema()),
        sum_value_expr(&scan.schema()),
    ];
    let plan = build_aggregate_exec(scan, AggregateMode::Single, aggr_exprs);

    let mut config = datafusion::common::config::ConfigOptions::default();
    let mut icefalldb_config = icefalldb_query::IcefallDBConfig::default();
    icefalldb_config.sorted_group_by = false;
    config.extensions.insert(icefalldb_config);

    let rule = SortedGroupBy::new();
    let optimized = rule.optimize(plan, &config).unwrap();
    assert!(
        !plan_contains_streaming_group_by(&optimized),
        "rule should not fire when disabled in config"
    );
}
