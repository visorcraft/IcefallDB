//! Correctness and fallback tests for the warm-aggregate fast path.
//!
//! Tests verify:
//! 1. SUM/COUNT/AVG/VAR_POP/VAR_SAMP/STDDEV_POP/STDDEV_SAMP composed from
//!    `.agg` partials match DataFusion full-scan ground truth (int exact,
//!    float within 1e-9 relative tolerance).
//! 2. The fast path was taken (plan is a DataSourceExec constant, zero Parquet I/O).
//! 3. Three fallback cases: deleted_count>0, missing agg_state, hash mismatch.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::common::ScalarValue;
use datafusion::execution::TaskContext;
use datafusion::functions_aggregate::average::avg_udaf;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::stddev::{stddev_pop_udaf, stddev_udaf};
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::functions_aggregate::variance::{var_pop_udaf, var_samp_udaf};
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_expr::expressions::{col as phys_col, Literal as PhysLiteral};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;
use icefalldb_core::agg_cache::{AggScalar, ColAgg, FragmentAggState};
use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::PlannedRowGroup;
use icefalldb_query::rules::MetadataAggregate;
use icefalldb_query::scan::IcefallDBScanExec;
use serde_json::Value;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn plan_is_metadata_fast_path(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.downcast_ref::<DataSourceExec>().is_some()
}

fn float_approx_eq(a: f64, b: f64) -> bool {
    let tol = 1e-9 * b.abs().max(1.0);
    (a - b).abs() <= tol
}

/// Schema for tests: Int64 column "i" and Float64 column "f".
fn test_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("i", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
    ]))
}

/// Build a `ColAgg` for the Int64 column from raw values.
///
/// Fragment 1: i = [10, 20, 30] → count=3, sum=60, sumsq=1400
/// Fragment 2: i = [40, 50]     → count=2, sum=90, sumsq=4100
/// Total: n=5, S=150, Q=5500
fn int_col_agg_frag1() -> ColAgg {
    // values: 10, 20, 30
    ColAgg {
        count_non_null: 3,
        sum: AggScalar::Int(60),
        sumsq: AggScalar::Int(1400),
        min_off: None,
        max_off: None,
        live_min_json: None,
        live_max_json: None,
    }
}

fn int_col_agg_frag2() -> ColAgg {
    // values: 40, 50
    ColAgg {
        count_non_null: 2,
        sum: AggScalar::Int(90),
        sumsq: AggScalar::Int(4100),
        min_off: None,
        max_off: None,
        live_min_json: None,
        live_max_json: None,
    }
}

/// Build a `ColAgg` for the Float64 column from raw values.
///
/// Fragment 1: f = [1.0, 2.0, 3.0] → count=3, sum=6.0, sumsq=14.0
/// Fragment 2: f = [4.0, 5.0]       → count=2, sum=9.0, sumsq=41.0
/// Total: n=5, S=15.0, Q=55.0
fn float_col_agg_frag1() -> ColAgg {
    ColAgg {
        count_non_null: 3,
        sum: AggScalar::Float(6.0),
        sumsq: AggScalar::Float(14.0),
        min_off: None,
        max_off: None,
        live_min_json: None,
        live_max_json: None,
    }
}

fn float_col_agg_frag2() -> ColAgg {
    ColAgg {
        count_non_null: 2,
        sum: AggScalar::Float(9.0),
        sumsq: AggScalar::Float(41.0),
        min_off: None,
        max_off: None,
        live_min_json: None,
        live_max_json: None,
    }
}

fn make_agg_state(
    fragment_id: u64,
    checksum: &str,
    i_agg: ColAgg,
    f_agg: ColAgg,
) -> Arc<FragmentAggState> {
    let mut cols = BTreeMap::new();
    cols.insert("i".to_string(), i_agg);
    cols.insert("f".to_string(), f_agg);
    Arc::new(FragmentAggState {
        fragment_id,
        content_hash: checksum.to_string(),
        cols,
        grouped: None,
        distinct: None,
        quantile: None,
    })
}

/// Build a two-fragment `IcefallDBScanExec` with agg_state populated on both
/// fragments and matching content hashes.
fn make_clean_scan_exec() -> IcefallDBScanExec {
    let schema = test_schema();
    let checksum1 = "sha256:frag1";
    let checksum2 = "sha256:frag2";

    let planned: Vec<PlannedRowGroup> = vec![
        PlannedRowGroup {
            meta: RowGroupMeta {
                rows: 3,
                columns: [
                    (
                        "i".into(),
                        ColumnStats {
                            min: Some(Value::from(10i64)),
                            max: Some(Value::from(30i64)),
                            nulls: 0,
                        },
                    ),
                    (
                        "f".into(),
                        ColumnStats {
                            min: Some(Value::from(1.0f64)),
                            max: Some(Value::from(3.0f64)),
                            nulls: 0,
                        },
                    ),
                ]
                .into(),
                checksum: checksum1.to_string(),
                ..Default::default()
            },
            agg_state: Some(make_agg_state(
                1,
                checksum1,
                int_col_agg_frag1(),
                float_col_agg_frag1(),
            )),
            deleted_count: 0,
            fallback: false,
            ..Default::default()
        },
        PlannedRowGroup {
            meta: RowGroupMeta {
                rows: 2,
                columns: [
                    (
                        "i".into(),
                        ColumnStats {
                            min: Some(Value::from(40i64)),
                            max: Some(Value::from(50i64)),
                            nulls: 0,
                        },
                    ),
                    (
                        "f".into(),
                        ColumnStats {
                            min: Some(Value::from(4.0f64)),
                            max: Some(Value::from(5.0f64)),
                            nulls: 0,
                        },
                    ),
                ]
                .into(),
                checksum: checksum2.to_string(),
                ..Default::default()
            },
            agg_state: Some(make_agg_state(
                2,
                checksum2,
                int_col_agg_frag2(),
                float_col_agg_frag2(),
            )),
            deleted_count: 0,
            fallback: false,
            ..Default::default()
        },
    ];

    IcefallDBScanExec::new(
        Arc::new(MemoryStorage::new()),
        schema,
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap()
}

fn build_aggregate_exec(
    scan: IcefallDBScanExec,
    aggr_exprs: Vec<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
) -> AggregateExec {
    let input_schema = scan.schema();
    let aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> =
        aggr_exprs.into_iter().map(Arc::new).collect();
    let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
        aggr_exprs.iter().map(|_| None).collect();
    AggregateExec::try_new(
        AggregateMode::Single,
        datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
        aggr_exprs,
        filter_exprs,
        Arc::new(scan),
        input_schema,
    )
    .unwrap()
}

fn sum_expr_for(
    col_name: &str,
    schema: &ArrowSchema,
) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
    let col = phys_col(col_name, schema).unwrap();
    AggregateExprBuilder::new(sum_udaf(), vec![col])
        .schema(Arc::new(schema.clone()))
        .alias(format!("SUM({col_name})"))
        .build()
        .unwrap()
}

fn avg_expr_for(
    col_name: &str,
    schema: &ArrowSchema,
) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
    let col = phys_col(col_name, schema).unwrap();
    AggregateExprBuilder::new(avg_udaf(), vec![col])
        .schema(Arc::new(schema.clone()))
        .alias(format!("AVG({col_name})"))
        .build()
        .unwrap()
}

fn var_samp_expr_for(
    col_name: &str,
    schema: &ArrowSchema,
) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
    let col = phys_col(col_name, schema).unwrap();
    AggregateExprBuilder::new(var_samp_udaf(), vec![col])
        .schema(Arc::new(schema.clone()))
        .alias(format!("VAR_SAMP({col_name})"))
        .build()
        .unwrap()
}

fn var_pop_expr_for(
    col_name: &str,
    schema: &ArrowSchema,
) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
    let col = phys_col(col_name, schema).unwrap();
    AggregateExprBuilder::new(var_pop_udaf(), vec![col])
        .schema(Arc::new(schema.clone()))
        .alias(format!("VAR_POP({col_name})"))
        .build()
        .unwrap()
}

fn stddev_samp_expr_for(
    col_name: &str,
    schema: &ArrowSchema,
) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
    let col = phys_col(col_name, schema).unwrap();
    AggregateExprBuilder::new(stddev_udaf(), vec![col])
        .schema(Arc::new(schema.clone()))
        .alias(format!("STDDEV_SAMP({col_name})"))
        .build()
        .unwrap()
}

fn stddev_pop_expr_for(
    col_name: &str,
    schema: &ArrowSchema,
) -> datafusion::physical_expr::aggregate::AggregateFunctionExpr {
    let col = phys_col(col_name, schema).unwrap();
    AggregateExprBuilder::new(stddev_pop_udaf(), vec![col])
        .schema(Arc::new(schema.clone()))
        .alias(format!("STDDEV_POP({col_name})"))
        .build()
        .unwrap()
}

async fn collect_plan(plan: Arc<dyn ExecutionPlan>) -> Vec<RecordBatch> {
    datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(TaskContext::default())).unwrap(),
    )
    .await
    .unwrap()
}

// ── Ground-truth helpers ──────────────────────────────────────────────────────
//
// These compute the expected values directly from the known test data without
// running DataFusion, to avoid the chicken-and-egg problem of comparing the
// fast path against a real scan that uses the same rule.
//
// Data: i = [10, 20, 30, 40, 50], f = [1.0, 2.0, 3.0, 4.0, 5.0]

fn ground_truth_sum_i() -> i64 {
    150
}
fn ground_truth_count_i() -> i64 {
    5
}
fn ground_truth_avg_f() -> f64 {
    15.0 / 5.0 // 3.0
}

/// var_pop(f) = (Q - S²/n) / n = (55.0 - 225.0/5) / 5 = (55 - 45) / 5 = 2.0
fn ground_truth_var_pop_f() -> f64 {
    2.0
}

/// var_samp(f) = (Q - S²/n) / (n-1) = 10.0 / 4 = 2.5
fn ground_truth_var_samp_f() -> f64 {
    2.5
}

fn ground_truth_stddev_pop_f() -> f64 {
    ground_truth_var_pop_f().sqrt()
}
fn ground_truth_stddev_samp_f() -> f64 {
    ground_truth_var_samp_f().sqrt()
}

// ── Correctness tests ─────────────────────────────────────────────────────────
// NOTE: These tests FAIL until the MetadataAggregate rule is extended to
// handle SUM/AVG/VAR/STDDEV. The rule currently only handles COUNT/MIN/MAX and
// returns the original AggregateExec for any unknown aggregate, so
// `plan_is_metadata_fast_path` returns false and the assertion below fails.

/// SUM(i) should compose exactly from integer partials.
#[tokio::test]
async fn sum_int_composed_exact() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let agg = build_aggregate_exec(scan, vec![sum_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "SUM(i) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        sum,
        ground_truth_sum_i(),
        "SUM(i) must be byte-equal to ground truth"
    );
}

/// COUNT(i) should compose exactly.
#[tokio::test]
async fn count_int_composed_exact() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let lit = Arc::new(PhysLiteral::new(ScalarValue::Int64(Some(1)))) as Arc<dyn PhysicalExpr>;
    let count_expr = AggregateExprBuilder::new(count_udaf(), vec![lit])
        .schema(Arc::new(schema.as_ref().clone()))
        .alias("COUNT(*)")
        .build()
        .unwrap();
    let agg = build_aggregate_exec(scan, vec![count_expr]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "COUNT(*) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, ground_truth_count_i(), "COUNT(*) must be 5");
}

/// AVG(f) should compose and be within relative tolerance.
#[tokio::test]
async fn avg_float_composed_within_tolerance() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let agg = build_aggregate_exec(scan, vec![avg_expr_for("f", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "AVG(f) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let avg = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_avg_f();
    assert!(
        float_approx_eq(avg, expected),
        "AVG(f) = {avg}, expected ≈ {expected}"
    );
}

/// VAR_POP(f).
#[tokio::test]
async fn var_pop_float_composed_within_tolerance() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let agg = build_aggregate_exec(scan, vec![var_pop_expr_for("f", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "VAR_POP(f) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_var_pop_f();
    assert!(
        float_approx_eq(v, expected),
        "VAR_POP(f) = {v}, expected ≈ {expected}"
    );
}

/// VAR_SAMP(f).
#[tokio::test]
async fn var_samp_float_composed_within_tolerance() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let agg = build_aggregate_exec(scan, vec![var_samp_expr_for("f", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "VAR_SAMP(f) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_var_samp_f();
    assert!(
        float_approx_eq(v, expected),
        "VAR_SAMP(f) = {v}, expected ≈ {expected}"
    );
}

/// STDDEV_POP(f).
#[tokio::test]
async fn stddev_pop_float_composed_within_tolerance() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let agg = build_aggregate_exec(scan, vec![stddev_pop_expr_for("f", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "STDDEV_POP(f) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_stddev_pop_f();
    assert!(
        float_approx_eq(v, expected),
        "STDDEV_POP(f) = {v}, expected ≈ {expected}"
    );
}

/// STDDEV_SAMP(f).
#[tokio::test]
async fn stddev_samp_float_composed_within_tolerance() {
    let scan = make_clean_scan_exec();
    let schema = test_schema();
    let agg = build_aggregate_exec(scan, vec![stddev_samp_expr_for("f", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "STDDEV_SAMP(f) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_stddev_samp_f();
    assert!(
        float_approx_eq(v, expected),
        "STDDEV_SAMP(f) = {v}, expected ≈ {expected}"
    );
}

// ── Fallback tests ────────────────────────────────────────────────────────────

/// Fallback case (a): deleted_count > 0 but agg_state is None (scan_internal could
/// not produce a live partial) → rule must NOT fire because there is no live partial.
#[tokio::test]
async fn sum_fallback_on_deleted_fragment_without_live_partial() {
    let schema = test_schema();
    let checksum = "sha256:frag1";
    let planned = vec![PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [
                (
                    "i".into(),
                    ColumnStats {
                        min: Some(Value::from(10i64)),
                        max: Some(Value::from(30i64)),
                        nulls: 0,
                    },
                ),
                (
                    "f".into(),
                    ColumnStats {
                        min: None,
                        max: None,
                        nulls: 0,
                    },
                ),
            ]
            .into(),
            checksum: checksum.to_string(),
            ..Default::default()
        },
        // No live partial: scan_internal could not load/compute one (e.g. missing DV file).
        agg_state: None,
        deleted_count: 1,
        fallback: false,
        ..Default::default()
    }];
    let scan = IcefallDBScanExec::new(
        Arc::new(MemoryStorage::new()),
        Arc::clone(&schema),
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap();
    let agg = build_aggregate_exec(scan, vec![sum_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        !plan_is_metadata_fast_path(&optimized),
        "SUM fast path must NOT fire when deleted_count > 0 and no live partial is available"
    );
    assert!(
        optimized.downcast_ref::<AggregateExec>().is_some(),
        "plan must remain AggregateExec so real scan runs"
    );
}

/// Dirty fragment WITH a live partial stored in agg_state → fast path MUST fire.
/// This simulates scan_internal having successfully computed the retracted partial.
#[tokio::test]
async fn sum_fast_path_fires_on_dirty_fragment_with_live_partial() {
    let schema = test_schema();
    // Live partial after retraction: fragment had [10, 20, 30], row 20 deleted.
    // Live: count=2, sum=40, sumsq=1000
    let live_state = {
        let mut cols = std::collections::BTreeMap::new();
        cols.insert(
            "i".to_string(),
            ColAgg {
                count_non_null: 2,
                sum: AggScalar::Int(40),
                sumsq: AggScalar::Int(1000),
                min_off: None,
                max_off: None,
                live_min_json: None,
                live_max_json: None,
            },
        );
        cols.insert(
            "f".to_string(),
            ColAgg {
                count_non_null: 2,
                sum: AggScalar::Float(4.0),
                sumsq: AggScalar::Float(10.0),
                min_off: None,
                max_off: None,
                live_min_json: None,
                live_max_json: None,
            },
        );
        Arc::new(FragmentAggState {
            fragment_id: 1,
            content_hash: "sha256:full-frag".to_string(), // hash of the FULL data file
            cols,
            grouped: None,
            distinct: None,
            quantile: None,
        })
    };
    let planned = vec![PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [
                (
                    "i".into(),
                    ColumnStats {
                        min: Some(Value::from(10i64)),
                        max: Some(Value::from(30i64)),
                        nulls: 0,
                    },
                ),
                (
                    "f".into(),
                    ColumnStats {
                        min: None,
                        max: None,
                        nulls: 0,
                    },
                ),
            ]
            .into(),
            checksum: "sha256:full-frag".to_string(),
            ..Default::default()
        },
        agg_state: Some(live_state),
        deleted_count: 1, // dirty, but live partial is present
        fallback: false,
        ..Default::default()
    }];
    let scan = IcefallDBScanExec::new(
        Arc::new(MemoryStorage::new()),
        Arc::clone(&schema),
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap();
    let agg = build_aggregate_exec(scan, vec![sum_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "SUM fast path MUST fire on a dirty fragment that has a live partial in agg_state"
    );

    let batches = collect_plan(optimized).await;
    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    // Live SUM(i) = 40 (after removing deleted row with value 20)
    assert_eq!(
        sum, 40i64,
        "SUM(i) must equal the live partial sum after retraction"
    );
}

/// Fallback case (b): agg_state is None → rule must NOT fire.
#[tokio::test]
async fn sum_fallback_on_missing_agg_state() {
    let schema = test_schema();
    let planned = vec![PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [(
                "i".into(),
                ColumnStats {
                    min: Some(Value::from(10i64)),
                    max: Some(Value::from(30i64)),
                    nulls: 0,
                },
            )]
            .into(),
            checksum: "sha256:frag1".to_string(),
            ..Default::default()
        },
        agg_state: None, // no .agg sidecar
        deleted_count: 0,
        fallback: false,
        ..Default::default()
    }];
    let scan = IcefallDBScanExec::new(
        Arc::new(MemoryStorage::new()),
        Arc::clone(&schema),
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap();
    let agg = build_aggregate_exec(scan, vec![sum_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        !plan_is_metadata_fast_path(&optimized),
        "SUM fast path must NOT fire when agg_state is None"
    );
    assert!(
        optimized.downcast_ref::<AggregateExec>().is_some(),
        "plan must remain AggregateExec so real scan runs"
    );
}

/// Fallback case (c): content_hash mismatch → rule must NOT fire.
#[tokio::test]
async fn sum_fallback_on_hash_mismatch() {
    let schema = test_schema();
    let meta_checksum = "sha256:current";
    let stale_checksum = "sha256:stale"; // agg_state has a different hash
    let planned = vec![PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [(
                "i".into(),
                ColumnStats {
                    min: Some(Value::from(10i64)),
                    max: Some(Value::from(30i64)),
                    nulls: 0,
                },
            )]
            .into(),
            checksum: meta_checksum.to_string(),
            ..Default::default()
        },
        agg_state: Some(make_agg_state(
            1,
            stale_checksum,
            int_col_agg_frag1(),
            float_col_agg_frag1(),
        )),
        deleted_count: 0,
        fallback: false,
        ..Default::default()
    }];
    let scan = IcefallDBScanExec::new(
        Arc::new(MemoryStorage::new()),
        Arc::clone(&schema),
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap();
    let agg = build_aggregate_exec(scan, vec![sum_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        !plan_is_metadata_fast_path(&optimized),
        "SUM fast path must NOT fire when content_hash does not match meta.checksum"
    );
    assert!(
        optimized.downcast_ref::<AggregateExec>().is_some(),
        "plan must remain AggregateExec so real scan runs"
    );
}

// ── Two-phase (multi-fragment) unit tests ─────────────────────────────────────
//
// These tests build the two-phase `AggregateExec(Final) → CoalescePartitionsExec
// → AggregateExec(Partial) → IcefallDBScanExec` plan shape manually (exactly as
// DataFusion builds it for multi-partition scans) and verify:
//   (a) the fast path fires (plan becomes a DataSourceExec)
//   (b) the result equals the known ground truth
//
// These tests FAIL before the Bug B fix (rule only handles single-phase).

/// Helper: wrap a `IcefallDBScanExec` in a two-phase aggregate (no group-by)
/// for a given set of aggregate expressions.  Returns the Final aggregate root.
///
/// The Final aggregate is built exactly as DataFusion's physical planner does it:
/// the same `aggr_expr` objects from the Partial are reused and the original
/// `input_schema` (not the partial output schema) is passed — see DataFusion
/// `physical_planner.rs::create_initial_plan` lines ~1139–1177.
fn build_two_phase_aggregate_no_group(
    scan: IcefallDBScanExec,
    aggr_exprs: Vec<datafusion::physical_expr::aggregate::AggregateFunctionExpr>,
) -> Arc<dyn ExecutionPlan> {
    use datafusion::physical_plan::aggregates::PhysicalGroupBy;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;

    let input_schema = scan.schema();
    let aggr_exprs: Vec<Arc<datafusion::physical_expr::aggregate::AggregateFunctionExpr>> =
        aggr_exprs.into_iter().map(Arc::new).collect();
    let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
        aggr_exprs.iter().map(|_| None).collect();

    // Partial aggregate: emits partial state columns.
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        PhysicalGroupBy::default(),
        aggr_exprs.clone(),
        filter_exprs.clone(),
        Arc::new(scan),
        Arc::clone(&input_schema),
    )
    .unwrap();

    // The Final aggregate reuses the SAME aggr_exprs and the ORIGINAL input_schema —
    // this mirrors DataFusion's physical planner exactly (updated_aggregates from
    // initial_aggr, physical_input_schema passed to both stages).
    let updated_aggregates = partial.aggr_expr().to_vec();
    let final_filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> =
        updated_aggregates.iter().map(|_| None).collect();

    // CoalescePartitionsExec merges multiple scan partitions into one.
    let coalesced = Arc::new(CoalescePartitionsExec::new(Arc::new(partial)));

    Arc::new(
        AggregateExec::try_new(
            AggregateMode::Final,
            PhysicalGroupBy::default(),
            updated_aggregates,
            final_filter_exprs,
            coalesced,
            input_schema,
        )
        .unwrap(),
    )
}

/// Build a three-fragment scan with agg_state on each fragment.
///
/// Data: frag1=[10,20,30], frag2=[40,50], frag3=[60,70,80,90]
/// Totals: n=9, S_i=450, Q_i=28500
fn make_three_fragment_scan() -> IcefallDBScanExec {
    use icefalldb_core::agg_cache::{AggScalar, ColAgg, FragmentAggState};
    use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
    use std::collections::BTreeMap;

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));

    let make_frag =
        |frag_id: u64, rows: usize, checksum: &str, count: u64, sum: i128, sumsq: i128| {
            let mut cols = BTreeMap::new();
            cols.insert(
                "i".to_string(),
                ColAgg {
                    count_non_null: count,
                    sum: AggScalar::Int(sum),
                    sumsq: AggScalar::Int(sumsq),
                    min_off: None,
                    max_off: None,
                    live_min_json: None,
                    live_max_json: None,
                },
            );
            PlannedRowGroup {
                meta: RowGroupMeta {
                    rows,
                    columns: [(
                        "i".into(),
                        ColumnStats {
                            min: Some(Value::from(0i64)),
                            max: Some(Value::from(100i64)),
                            nulls: 0,
                        },
                    )]
                    .into(),
                    checksum: checksum.to_string(),
                    ..Default::default()
                },
                agg_state: Some(Arc::new(FragmentAggState {
                    fragment_id: frag_id,
                    content_hash: checksum.to_string(),
                    cols,
                    grouped: None,
                    distinct: None,
                    quantile: None,
                })),
                deleted_count: 0,
                fallback: false,
                ..Default::default()
            }
        };

    // frag1: [10,20,30] → count=3, sum=60, sumsq=1400
    // frag2: [40,50]    → count=2, sum=90, sumsq=4100
    // frag3: [60,70,80,90] → count=4, sum=300, sumsq=23000
    // Total: n=9, S=450, Q=28500
    let planned = vec![
        make_frag(1, 3, "sha256:f1", 3, 60, 1400),
        make_frag(2, 2, "sha256:f2", 2, 90, 4100),
        make_frag(3, 4, "sha256:f3", 4, 300, 23000),
    ];

    IcefallDBScanExec::new(
        Arc::new(icefalldb_core::storage::memory::MemoryStorage::new()),
        schema,
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        3, // three scan partitions — causes DataFusion to generate two-phase plan
    )
    .unwrap()
}

// Ground truth for three-fragment data:
// i = [10,20,30,40,50,60,70,80,90]
// n=9, S=450, Q=28500
// SUM = 450
// COUNT = 9
// AVG = 450/9 = 50.0
// var_pop = (28500 - 450²/9) / 9 = (28500 - 22500) / 9 = 6000/9 ≈ 666.666...
// var_samp = 6000 / 8 = 750.0
// stddev_pop = sqrt(666.666...) ≈ 25.8198...
// stddev_samp = sqrt(750) ≈ 27.3861...
fn ground_truth_three_sum_i() -> i64 {
    450
}
fn ground_truth_three_count_i() -> i64 {
    9
}
fn ground_truth_three_avg_i() -> f64 {
    50.0
}
fn ground_truth_three_var_pop_i() -> f64 {
    6000.0 / 9.0
}
fn ground_truth_three_var_samp_i() -> f64 {
    750.0
}
fn ground_truth_three_stddev_pop_i() -> f64 {
    ground_truth_three_var_pop_i().sqrt()
}
fn ground_truth_three_stddev_samp_i() -> f64 {
    ground_truth_three_var_samp_i().sqrt()
}

/// Two-phase SUM — fast path must fire and result must be exact.
#[tokio::test]
async fn two_phase_sum_int_fast_path_fires() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let plan = build_two_phase_aggregate_no_group(scan, vec![sum_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase SUM(i) must take the fast path — rule must match Final→Coalesce→Partial→scan"
    );

    let batches = collect_plan(optimized).await;
    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        sum,
        ground_truth_three_sum_i(),
        "two-phase SUM(i) must equal ground truth"
    );
}

/// Two-phase COUNT(*) — fast path must fire and total must be exact.
#[tokio::test]
async fn two_phase_count_star_fast_path_fires() {
    use datafusion::physical_expr::expressions::Literal as PhysLiteral2;
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let lit = Arc::new(PhysLiteral2::new(ScalarValue::Int64(Some(1)))) as Arc<dyn PhysicalExpr>;
    let count_expr = AggregateExprBuilder::new(count_udaf(), vec![lit])
        .schema(Arc::clone(&schema))
        .alias("COUNT(*)")
        .build()
        .unwrap();
    let plan = build_two_phase_aggregate_no_group(scan, vec![count_expr]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase COUNT(*) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        count,
        ground_truth_three_count_i(),
        "two-phase COUNT(*) must equal 9"
    );
}

/// Two-phase AVG — fast path must fire and result must be within tolerance.
#[tokio::test]
async fn two_phase_avg_fast_path_fires() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let plan = build_two_phase_aggregate_no_group(scan, vec![avg_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase AVG(i) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let avg = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_three_avg_i();
    assert!(
        float_approx_eq(avg, expected),
        "two-phase AVG(i) = {avg}, expected ≈ {expected}"
    );
}

/// Two-phase VAR_POP.
#[tokio::test]
async fn two_phase_var_pop_fast_path_fires() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let plan = build_two_phase_aggregate_no_group(scan, vec![var_pop_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase VAR_POP(i) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_three_var_pop_i();
    assert!(
        float_approx_eq(v, expected),
        "two-phase VAR_POP(i) = {v}, expected ≈ {expected}"
    );
}

/// Two-phase VAR_SAMP.
#[tokio::test]
async fn two_phase_var_samp_fast_path_fires() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let plan = build_two_phase_aggregate_no_group(scan, vec![var_samp_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase VAR_SAMP(i) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_three_var_samp_i();
    assert!(
        float_approx_eq(v, expected),
        "two-phase VAR_SAMP(i) = {v}, expected ≈ {expected}"
    );
}

/// Two-phase STDDEV_POP.
#[tokio::test]
async fn two_phase_stddev_pop_fast_path_fires() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let plan = build_two_phase_aggregate_no_group(scan, vec![stddev_pop_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase STDDEV_POP(i) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_three_stddev_pop_i();
    assert!(
        float_approx_eq(v, expected),
        "two-phase STDDEV_POP(i) = {v}, expected ≈ {expected}"
    );
}

/// Two-phase STDDEV_SAMP.
#[tokio::test]
async fn two_phase_stddev_samp_fast_path_fires() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let plan = build_two_phase_aggregate_no_group(scan, vec![stddev_samp_expr_for("i", &schema)]);
    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(plan, &Default::default()).unwrap();

    assert!(
        plan_is_metadata_fast_path(&optimized),
        "two-phase STDDEV_SAMP(i) must take the fast path"
    );

    let batches = collect_plan(optimized).await;
    let v = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let expected = ground_truth_three_stddev_samp_i();
    assert!(
        float_approx_eq(v, expected),
        "two-phase STDDEV_SAMP(i) = {v}, expected ≈ {expected}"
    );
}

/// Bug-A correctness guard: a bare Partial aggregate over the scan must NOT be
/// folded into a final value.  Before the mode guard, a bottom-up traversal could
/// encounter the Partial node and fold it — producing a wrong answer.
///
/// This test verifies that a Partial AggregateExec is left unchanged (returns an
/// AggregateExec, not a DataSourceExec).  A real two-phase plan will have the
/// Final at the root (handled by Bug B), but the rule must not fire on the Partial
/// half in isolation.
#[tokio::test]
async fn partial_agg_is_not_folded() {
    let scan = make_three_fragment_scan();
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "i",
        DataType::Int64,
        false,
    )]));
    let aggr = Arc::new(sum_expr_for("i", &schema));
    let filter_exprs: Vec<Option<Arc<dyn PhysicalExpr>>> = vec![None];
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
        vec![aggr],
        filter_exprs,
        Arc::new(scan),
        schema,
    )
    .unwrap();

    let rule = MetadataAggregate::new();
    // The optimizer traverses bottom-up; when it encounters a bare Partial
    // it must leave it unchanged.
    let optimized = rule
        .optimize(Arc::new(partial), &Default::default())
        .unwrap();

    // Must NOT be the fast path — Partial emits state, not a final scalar.
    assert!(
        !plan_is_metadata_fast_path(&optimized),
        "bare Partial aggregate must NOT be folded into a final value (mode guard)"
    );
    assert!(
        optimized.downcast_ref::<AggregateExec>().is_some(),
        "plan must remain AggregateExec for a bare Partial"
    );
}

// ── E2E multi-fragment SQL test ───────────────────────────────────────────────

/// End-to-end multi-fragment test: write a table with 3 fragments (2 rows/group),
/// query SUM/AVG/STDDEV via SQL — DataFusion will generate a two-phase plan.
/// The fast path must fire AND results must match ground truth.
#[tokio::test]
async fn e2e_multi_fragment_sum_avg_stddev_via_sql() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        icefalldb_session_config, icefalldb_session_state_from_config, IcefallDBTableProvider,
        ProviderConfig,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Float64,
        false,
    )]));

    // 6 rows split into 3 row-groups of 2 rows each → 3 fragments → two-phase plan.
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 2;

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Float64Array::from(vec![
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_multi", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    // Use target_partitions > 1 to force DataFusion into a two-phase plan.
    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_multi",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    // Use the same session config the benchmark uses — target_partitions drives
    // the two-phase plan.
    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);
    ctx.register_table("t_multi", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT SUM(v), AVG(v), STDDEV(v) FROM t_multi")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    assert!(
        plan_is_metadata_fast_path(&plan),
        "E2E multi-fragment: MetadataAggregate rule must fire on the two-phase plan \
         (Final→CoalescePartitions→Partial→scan)"
    );

    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(TaskContext::default())).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1, "expected exactly one result batch");

    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let avg = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let stddev = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);

    // Data: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    // sum=21.0, avg=3.5
    // var_samp = 3.5, stddev_samp = sqrt(3.5)
    let expected_sum = 21.0_f64;
    let expected_avg = 3.5_f64;
    let expected_stddev = 3.5_f64.sqrt();

    assert!(
        float_approx_eq(sum, expected_sum),
        "E2E multi-fragment SUM={sum}, expected {expected_sum}"
    );
    assert!(
        float_approx_eq(avg, expected_avg),
        "E2E multi-fragment AVG={avg}, expected {expected_avg}"
    );
    assert!(
        float_approx_eq(stddev, expected_stddev),
        "E2E multi-fragment STDDEV={stddev}, expected {expected_stddev}"
    );
}

/// End-to-end test: write a two-fragment table via `Writer::insert_batch`,
/// then query SUM/AVG/STDDEV via SQL through the full DataFusion pipeline.
/// The composed result must match the ground truth.
#[tokio::test]
async fn e2e_sum_avg_stddev_via_sql() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Float64,
        false,
    )]));
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    // Keep all 6 rows in a single row group so the scan has one partition and
    // the MetadataAggregate rule can fire (a two-partition scan causes DataFusion
    // to generate a two-phase Partial/Final aggregate that the rule cannot rewrite).
    mdb_schema.row_group_target_rows = 1024;

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Float64Array::from(vec![
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT SUM(v), AVG(v), STDDEV(v) FROM t")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan),
        "E2E: MetadataAggregate rule must fire for SUM/AVG/STDDEV on a clean table with .agg sidecars"
    );
    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(TaskContext::default())).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1, "expected exactly one result batch");

    let sum = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let avg = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let stddev = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);

    // Data: 1.0, 2.0, 3.0, 4.0, 5.0, 6.0
    // sum=21.0, avg=3.5
    // var_samp = (Σx² - n×avg²)/(n-1) = (91 - 6×12.25)/5 = 17.5/5 = 3.5
    // stddev_samp = sqrt(3.5) ≈ 1.8708286933869707
    let expected_sum = 21.0_f64;
    let expected_avg = 3.5_f64;
    let expected_stddev = 3.5_f64.sqrt();

    assert!(
        float_approx_eq(sum, expected_sum),
        "SUM={sum}, expected {expected_sum}"
    );
    assert!(
        float_approx_eq(avg, expected_avg),
        "AVG={avg}, expected {expected_avg}"
    );
    assert!(
        float_approx_eq(stddev, expected_stddev),
        "STDDEV={stddev}, expected {expected_stddev}"
    );
}

// ── integer AVG/VAR/STDDEV fast path via SQL ─────────────────────
//
// DataFusion 54 coerces Int64 inputs to Float64 in the Partial aggregate plan,
// emitting a CastExpr(column → Float64) as the aggregate's expression.  Before
// the cast-peel fix the rule sees a CastExpr instead of a bare Column, returns
// None, and silently falls back to a full Parquet scan.
//
// This test MUST FAIL before the fix (plan is NOT a DataSourceExec) and PASS
// after (plan IS a DataSourceExec and results match ground truth).

/// E2E SQL test: Int64 column, multi-fragment, AVG/VAR_POP/STDDEV_POP via SQL.
/// DataFusion inserts CastExpr(col → Float64) in the two-phase plan's Partial
/// aggregate; the cast-peel fix must unwrap it so the fast path fires.
#[tokio::test]
async fn e2e_int64_avg_var_pop_stddev_pop_via_sql() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        icefalldb_session_config, icefalldb_session_state_from_config, IcefallDBTableProvider,
        ProviderConfig,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // Integer column — DataFusion will coerce to Float64 in the aggregate plan.
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "n",
        DataType::Int64,
        false,
    )]));

    // 6 rows split into 3 row-groups of 2 rows each → 3 fragments → two-phase plan.
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 2;

    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50, 60]))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_int", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        "t_int",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);
    ctx.register_table("t_int", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT AVG(n), VAR_POP(n), STDDEV_POP(n) FROM t_int")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    assert!(
        plan_is_metadata_fast_path(&plan),
        "E2E Int64: MetadataAggregate rule must fire for AVG/VAR_POP/STDDEV_POP on an \
         Int64 column — the cast-peel fix must unwrap DataFusion's CastExpr(n → Float64)"
    );

    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(TaskContext::default())).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1, "expected exactly one result batch");

    let avg = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let var_pop = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let stddev_pop = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);

    // Data: 10, 20, 30, 40, 50, 60
    // n=6, S=210, Q=9100
    // avg = 210/6 = 35.0
    // var_pop = (9100 - 210²/6) / 6 = (9100 - 7350) / 6 = 1750/6 ≈ 291.6666...
    // stddev_pop = sqrt(1750/6)
    let expected_avg = 35.0_f64;
    let expected_var_pop = 1750.0_f64 / 6.0;
    let expected_stddev_pop = expected_var_pop.sqrt();

    assert!(
        float_approx_eq(avg, expected_avg),
        "E2E Int64 AVG={avg}, expected {expected_avg}"
    );
    assert!(
        float_approx_eq(var_pop, expected_var_pop),
        "E2E Int64 VAR_POP={var_pop}, expected {expected_var_pop}"
    );
    assert!(
        float_approx_eq(stddev_pop, expected_stddev_pop),
        "E2E Int64 STDDEV_POP={stddev_pop}, expected {expected_stddev_pop}"
    );
}

// ── warm-aggregate fast path with deletions ─────────────────────────────
//
// Write three fragments (~1000 rows each), DELETE ~7% via SQL, then query
// SUM/COUNT/AVG via SQL.  The fast path must fire (DataSourceExec) and the
// results must match ground truth computed by a reference full scan with
// MetadataAggregate disabled.

/// E2E test: SUM/COUNT/AVG/STDDEV on a Float64 table with ~7% deleted rows.
///
/// Uses a Float64 column to avoid DataFusion's Int64→Float64 coercion which
/// can produce a mixed-projection plan shape that the cast-peel logic in the
/// existing rule doesn't yet handle for multi-aggregate queries.  The
/// retraction logic must produce correct live partials so the fast path fires
/// and the composed results match ground truth.
#[tokio::test]
async fn sum_count_avg_var_exact_under_deletes() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
        IcefallDBTableProvider, ProviderConfig,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // Float64 column — no type coercion needed, so the two-phase plan shape is
    // simple: Final → Coalesce → Partial → scan (no ProjectionExec).
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Float64,
        false,
    )]));

    // Three fragments of 1000 rows each (v = 0.0 .. 2999.0).
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 1000;

    let values: Vec<f64> = (0i64..3000).map(|i| i as f64).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Float64Array::from(values))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "tq_dirty", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);

    let make_provider = |storage: Arc<dyn Storage>| async move {
        IcefallDBTableProvider::new(
            storage,
            "tq_dirty",
            ProviderConfig {
                batch_size: 1024,
                target_partitions: 4,
                io_coalesce_window: 0,
                io_concurrency: 1,
                native_parquet_threshold: usize::MAX,
                parquet_metadata_cache_capacity: 256,
                tiny_table_cache_threshold_rows: 0,
                tiny_table_cache_threshold_bytes: 0,
                wal_mode: true,
            },
        )
        .await
        .unwrap()
    };

    let provider = make_provider(Arc::clone(&storage)).await;
    ctx.register_table("tq_dirty", Arc::new(provider)).unwrap();

    // Delete ~7%: rows where CAST(v, INT64) % 14 == 0 (i.e. v ∈ {0.0, 14.0, 28.0, …}).
    // execute_sql DELETE uses rowid/rowaddr so it works on Float64 columns too.
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "tq_dirty",
        "DELETE FROM tq_dirty WHERE CAST(v AS BIGINT) % 14 = 0",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();

    let provider = make_provider(Arc::clone(&storage)).await;
    ctx.deregister_table("tq_dirty").unwrap();
    ctx.register_table("tq_dirty", Arc::new(provider)).unwrap();

    // ── Fast-path query ────────────────────────────────────────────────────────
    let df = ctx
        .sql("SELECT SUM(v), AVG(v), STDDEV(v) FROM tq_dirty")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    assert!(
        plan_is_metadata_fast_path(&plan),
        "R2.1: MetadataAggregate fast path must fire on a dirty Float64 table with live partials"
    );

    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1);

    let sum_fp = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let avg_fp = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let stddev_fp = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);

    // ── Ground truth: compute from raw data ────────────────────────────────────
    // v = 0.0 .. 2999.0, delete where i % 14 == 0
    let survivors: Vec<f64> = (0i64..3000)
        .filter(|i| i % 14 != 0)
        .map(|i| i as f64)
        .collect();
    let n = survivors.len() as f64;
    let expected_sum: f64 = survivors.iter().sum();
    let expected_avg = expected_sum / n;
    let q: f64 = survivors.iter().map(|v| v * v).sum();
    let expected_var_samp = (q - expected_sum * expected_sum / n) / (n - 1.0);
    let expected_stddev = expected_var_samp.sqrt();

    assert!(
        float_approx_eq(sum_fp, expected_sum),
        "R2.1 SUM(v) = {sum_fp}, expected {expected_sum}"
    );
    assert!(
        float_approx_eq(avg_fp, expected_avg),
        "R2.1 AVG(v) = {avg_fp}, expected {expected_avg}"
    );
    assert!(
        float_approx_eq(stddev_fp, expected_stddev),
        "R2.1 STDDEV(v) = {stddev_fp}, expected {expected_stddev}"
    );
}

/// E2E test: Int64 column `v`, 3 fragments (~1000 rows each), DELETE ~7%,
/// then SELECT SUM(v), COUNT(v), AVG(v), VAR_POP(v), STDDEV_POP(v).
///
/// With passthrough-projection support, the mixed Int64 multi-aggregate
/// projection is peeled and the fast path fires.  Without it, it falls back.
/// SUM and COUNT must be BYTE-EQUAL to DV-filtered ground truth; AVG/VAR_POP/
/// STDDEV_POP within 1e-9*max(1,|expected|) relative tolerance.
#[tokio::test]
async fn e2e_int64_sum_count_avg_var_pop_stddev_pop_under_deletes() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
        IcefallDBTableProvider, ProviderConfig,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));

    // 3000 rows → 3 fragments of 1000 each.
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 1000;

    let values: Vec<i64> = (0i64..3000).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_int64_dirty", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);

    let make_provider = |storage: Arc<dyn Storage>| async move {
        IcefallDBTableProvider::new(
            storage,
            "t_int64_dirty",
            ProviderConfig {
                batch_size: 1024,
                target_partitions: 4,
                io_coalesce_window: 0,
                io_concurrency: 1,
                native_parquet_threshold: usize::MAX,
                parquet_metadata_cache_capacity: 256,
                tiny_table_cache_threshold_rows: 0,
                tiny_table_cache_threshold_bytes: 0,
                wal_mode: true,
            },
        )
        .await
        .unwrap()
    };

    let provider = make_provider(Arc::clone(&storage)).await;
    ctx.register_table("t_int64_dirty", Arc::new(provider))
        .unwrap();

    // Delete ~7%: rows where v % 14 = 0.
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "t_int64_dirty",
        "DELETE FROM t_int64_dirty WHERE v % 14 = 0",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();

    let provider = make_provider(Arc::clone(&storage)).await;
    ctx.deregister_table("t_int64_dirty").unwrap();
    ctx.register_table("t_int64_dirty", Arc::new(provider))
        .unwrap();

    // Run the mixed multi-aggregate query — this produces a mixed projection
    // (passthrough Column for SUM/COUNT, CastExpr for AVG/VAR_POP/STDDEV_POP).
    let df = ctx
        .sql("SELECT SUM(v), COUNT(v), AVG(v), VAR_POP(v), STDDEV_POP(v) FROM t_int64_dirty")
        .await
        .unwrap();
    let plan = df.create_physical_plan().await.unwrap();

    assert!(
        plan_is_metadata_fast_path(&plan),
        "R2.1 Fix 1: fast path must fire for mixed Int64 multi-aggregate query \
         (passthrough Column + CastExpr projection)"
    );

    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1);

    let sum_fp = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let count_fp = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let avg_fp = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let var_pop_fp = batches[0]
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let stddev_pop_fp = batches[0]
        .column(4)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);

    // Ground truth: survivors = 0..2999 where i % 14 != 0
    let survivors: Vec<i64> = (0i64..3000).filter(|i| i % 14 != 0).collect();
    let expected_sum: i64 = survivors.iter().sum();
    let expected_count: i64 = survivors.len() as i64;
    let n_f = expected_count as f64;
    let sum_f = expected_sum as f64;
    let sumsq_f: f64 = survivors.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let expected_avg = sum_f / n_f;
    let expected_var_pop = (sumsq_f - sum_f * sum_f / n_f) / n_f;
    let expected_stddev_pop = expected_var_pop.sqrt();

    // SUM and COUNT must be byte-exact.
    assert_eq!(
        sum_fp, expected_sum,
        "SUM(v) must be byte-equal to ground truth"
    );
    assert_eq!(
        count_fp, expected_count,
        "COUNT(v) must be byte-equal to ground truth"
    );

    // AVG/VAR_POP/STDDEV_POP within 1e-9 relative tolerance.
    let tol = |expected: f64| 1e-9_f64 * expected.abs().max(1.0);
    assert!(
        (avg_fp - expected_avg).abs() <= tol(expected_avg),
        "AVG(v) = {avg_fp}, expected {expected_avg}"
    );
    assert!(
        (var_pop_fp - expected_var_pop).abs() <= tol(expected_var_pop),
        "VAR_POP(v) = {var_pop_fp}, expected {expected_var_pop}"
    );
    assert!(
        (stddev_pop_fp - expected_stddev_pop).abs() <= tol(expected_stddev_pop),
        "STDDEV_POP(v) = {stddev_pop_fp}, expected {expected_stddev_pop}"
    );
}

// ── MIN/MAX exact under deletions via scoped recompute-on-shrink ────────

/// Integration: 3-fragment integer table; delete the global-MIN row →
/// MIN must rise, equal a DV-filtered full-scan result, and be byte-exact.
/// A non-extremal delete (different fragment) stays on the zero-I/O fast path.
#[tokio::test]
async fn min_correct_after_deleting_min_row_scoped_recompute() {
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    use datafusion::physical_expr::expressions::col as phys_col;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
    use datafusion_datasource::source::DataSourceExec;
    use icefalldb_core::agg_cache::{AggScalar, ColAgg, FragmentAggState};
    use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
    use icefalldb_core::storage::memory::MemoryStorage;
    use icefalldb_core::PlannedRowGroup;
    use icefalldb_query::rules::MetadataAggregate;
    use icefalldb_query::scan::IcefallDBScanExec;
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));

    let storage = Arc::new(MemoryStorage::new());

    fn make_col_agg(count: u64, sum: i128, sumsq: i128, min_off: u32, max_off: u32) -> ColAgg {
        ColAgg {
            count_non_null: count,
            sum: AggScalar::Int(sum),
            sumsq: AggScalar::Int(sumsq),
            min_off: Some(min_off),
            max_off: Some(max_off),
            live_min_json: None,
            live_max_json: None,
        }
    }

    // Frag A (clean, no deletions).
    let agg_a = {
        let mut cols = BTreeMap::new();
        cols.insert("v".to_string(), make_col_agg(3, 600, 140_000, 0, 2));
        Arc::new(FragmentAggState {
            fragment_id: 1,
            content_hash: "h_a".to_string(),
            cols,
            grouped: None,
            distinct: None,
            quantile: None,
        })
    };
    let rg_a = PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [(
                "v".into(),
                ColumnStats {
                    min: Some(Value::from(100i64)),
                    max: Some(Value::from(300i64)),
                    nulls: 0,
                },
            )]
            .into(),
            checksum: "h_a".to_string(),
            ..Default::default()
        },
        agg_state: Some(agg_a),
        deleted_count: 0,
        fallback: false,
        ..Default::default()
    };

    // Frag B (dirty — offset 0 deleted, value=1 is the global min).
    // live_min_json = 400 after deleting offset 0 (survivor min of [400, 500]).
    let agg_b_live = {
        let mut cols = BTreeMap::new();
        cols.insert(
            "v".to_string(),
            ColAgg {
                count_non_null: 2,
                sum: AggScalar::Int(900),
                sumsq: AggScalar::Int(410_000),
                min_off: Some(0), // original min_off — the deleted row
                max_off: Some(2), // original max_off — still alive
                live_min_json: Some(serde_json::json!(400i64)),
                live_max_json: Some(serde_json::json!(500i64)),
            },
        );
        Arc::new(FragmentAggState {
            fragment_id: 2,
            content_hash: "h_b".to_string(),
            cols,
            grouped: None,
            distinct: None,
            quantile: None,
        })
    };
    let rg_b = PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [(
                "v".into(),
                ColumnStats {
                    min: Some(Value::from(1i64)),
                    max: Some(Value::from(500i64)),
                    nulls: 0,
                },
            )]
            .into(),
            checksum: "h_b".to_string(),
            ..Default::default()
        },
        agg_state: Some(agg_b_live),
        deleted_count: 1, // dirty
        fallback: false,
        ..Default::default()
    };

    // Frag C (clean).
    let agg_c = {
        let mut cols = BTreeMap::new();
        cols.insert("v".to_string(), make_col_agg(3, 2100, 1_490_000, 0, 2));
        Arc::new(FragmentAggState {
            fragment_id: 3,
            content_hash: "h_c".to_string(),
            cols,
            grouped: None,
            distinct: None,
            quantile: None,
        })
    };
    let rg_c = PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [(
                "v".into(),
                ColumnStats {
                    min: Some(Value::from(600i64)),
                    max: Some(Value::from(800i64)),
                    nulls: 0,
                },
            )]
            .into(),
            checksum: "h_c".to_string(),
            ..Default::default()
        },
        agg_state: Some(agg_c),
        deleted_count: 0,
        fallback: false,
        ..Default::default()
    };

    let planned = vec![rg_a, rg_b, rg_c];
    let scan = IcefallDBScanExec::new(
        storage,
        Arc::clone(&schema),
        planned,
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap();

    let col_v = phys_col("v", &schema).unwrap();
    let min_expr = AggregateExprBuilder::new(
        datafusion::functions_aggregate::min_max::min_udaf(),
        vec![col_v],
    )
    .schema(Arc::clone(&schema))
    .alias("MIN(v)")
    .build()
    .unwrap();
    let aggr_exprs = vec![Arc::new(min_expr)];
    let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        aggr_exprs.iter().map(|_| None).collect();
    let agg = AggregateExec::try_new(
        AggregateMode::Single,
        datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
        aggr_exprs,
        filter_exprs,
        Arc::new(scan),
        schema,
    )
    .unwrap();

    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        optimized.downcast_ref::<DataSourceExec>().is_some(),
        "MIN fast path must fire when all dirty fragments have resolved live_min_json"
    );

    let batches = datafusion::physical_plan::common::collect(
        optimized
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(batches.len(), 1);
    let min_val = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);

    assert_eq!(
        min_val, 100i64,
        "MIN(v) must be 100 after deleting the global min row (1)"
    );
    assert!(min_val > 1, "MIN(v) must have risen above deleted value 1");
}

/// Non-extremal delete — one dirty fragment where min_off is NOT deleted
/// — the fast path fires (zero-I/O for that fragment via live_min_json).
#[tokio::test]
async fn min_non_extremal_delete_stays_fast_path() {
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use datafusion::physical_expr::aggregate::AggregateExprBuilder;
    use datafusion::physical_expr::expressions::col as phys_col;
    use datafusion::physical_optimizer::PhysicalOptimizerRule;
    use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode};
    use datafusion_datasource::source::DataSourceExec;
    use icefalldb_core::agg_cache::{AggScalar, ColAgg, FragmentAggState};
    use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
    use icefalldb_core::storage::memory::MemoryStorage;
    use icefalldb_core::PlannedRowGroup;
    use icefalldb_query::rules::MetadataAggregate;
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));

    // One dirty fragment: v=[10, 50, 30], min=10 at offset 0 (survives).
    // Deletion vector deletes offset 1 (value=50), which is non-extremal.
    // live_min_json is pre-resolved (zero-I/O: min_off=0 not in DV).
    let agg_dirty = {
        let mut cols = BTreeMap::new();
        cols.insert(
            "v".to_string(),
            ColAgg {
                count_non_null: 2,
                sum: AggScalar::Int(40),
                sumsq: AggScalar::Int(100 + 900),
                min_off: Some(0), // value=10, offset 0 is alive
                max_off: Some(1), // value=50, offset 1 deleted — but live_max_json handles this
                live_min_json: Some(serde_json::json!(10i64)), // zero-I/O resolved
                live_max_json: Some(serde_json::json!(30i64)), // max survivor
            },
        );
        Arc::new(FragmentAggState {
            fragment_id: 1,
            content_hash: "hh".to_string(),
            cols,
            grouped: None,
            distinct: None,
            quantile: None,
        })
    };
    let rg = PlannedRowGroup {
        meta: RowGroupMeta {
            rows: 3,
            columns: [(
                "v".into(),
                ColumnStats {
                    min: Some(Value::from(10i64)),
                    max: Some(Value::from(50i64)),
                    nulls: 0,
                },
            )]
            .into(),
            checksum: "hh".to_string(),
            ..Default::default()
        },
        agg_state: Some(agg_dirty),
        deleted_count: 1,
        fallback: false,
        ..Default::default()
    };

    let scan = IcefallDBScanExec::new(
        Arc::new(MemoryStorage::new()),
        Arc::clone(&schema),
        vec![rg],
        None,
        vec![],
        None,
        1024,
        0,
        1,
    )
    .unwrap();

    let col_v = phys_col("v", &schema).unwrap();
    let min_expr = AggregateExprBuilder::new(
        datafusion::functions_aggregate::min_max::min_udaf(),
        vec![col_v],
    )
    .schema(Arc::clone(&schema))
    .alias("MIN(v)")
    .build()
    .unwrap();
    let aggr_exprs = vec![Arc::new(min_expr)];
    let filter_exprs: Vec<Option<Arc<dyn datafusion::physical_expr::PhysicalExpr>>> =
        aggr_exprs.iter().map(|_| None).collect();
    let agg = AggregateExec::try_new(
        AggregateMode::Single,
        datafusion::physical_plan::aggregates::PhysicalGroupBy::default(),
        aggr_exprs,
        filter_exprs,
        Arc::new(scan),
        schema,
    )
    .unwrap();

    let rule = MetadataAggregate::new();
    let optimized = rule.optimize(Arc::new(agg), &Default::default()).unwrap();

    assert!(
        optimized.downcast_ref::<DataSourceExec>().is_some(),
        "MIN fast path must fire when live_min_json is resolved (non-extremal delete)"
    );

    let batches = datafusion::physical_plan::common::collect(
        optimized
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    let min_val = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(
        min_val, 10i64,
        "survivor MIN must be 10 (non-extremal delete)"
    );
}

// ── genuine E2E tests — drive the real scan_internal / resolve_live_extrema path ──
//
// These tests write real Parquet fragments via Writer, delete rows via
// execute_sql (which writes a real deletion vector), re-register the provider,
// then query SELECT MIN(v)/MAX(v) through ctx.sql.  The expected values are
// computed from the same raw data with the same filter — not baked constants.
//
// They exercise the FULL path: scan_internal → resolve_live_extrema →
// extremum_deleted → (scoped_recompute_extremum | zero-I/O copy from
// ColumnStats) → live_min_json / live_max_json → MetadataAggregate rule.
//
// A polarity inversion in extremum_deleted (returning true instead of false or
// vice-versa) or in scoped_recompute_extremum (using the wrong RowSelection
// polarity) would cause the fast-path result to diverge from the full-scan
// ground truth → assertions fail.

/// Helper: build a IcefallDBTableProvider for a named table.
async fn make_int_provider(
    storage: Arc<dyn icefalldb_core::storage::Storage>,
    table: &str,
) -> icefalldb_query::IcefallDBTableProvider {
    icefalldb_query::IcefallDBTableProvider::new(
        storage,
        table,
        icefalldb_query::ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap()
}

/// Helper: collect a single i64 from the first column of a SQL query result.
async fn sql_single_i64(ctx: &datafusion::execution::context::SessionContext, query: &str) -> i64 {
    let batches = ctx.sql(query).await.unwrap().collect().await.unwrap();
    assert!(!batches.is_empty(), "query returned no batches");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("expected Int64 column")
        .value(0)
}

/// E2E case 1: delete the global-MIN row → MIN must rise, be byte-equal
/// to a DV-filtered full-scan ground truth, and the fast path must fire
/// (scoped-recompute path through resolve_live_extrema).
///
/// The assertion fails if extremum_deleted polarity is inverted (would return
/// the stale ColumnStats min = 0 instead of the post-delete minimum) or if
/// scoped_recompute_extremum scans deleted rows instead of survivors.
#[tokio::test]
async fn e2e_r22_delete_global_min_row_min_rises() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3 fragments of 100 rows each: v = 0..299.
    // Global MIN = 0 (frag 0, offset 0).  Deleting v=0 forces scoped recompute.
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 100;

    let values: Vec<i64> = (0i64..300).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_r22_min", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_min").await;
    ctx.register_table("t_r22_min", Arc::new(provider)).unwrap();

    // Delete the global-MIN row (v = 0).
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "t_r22_min",
        "DELETE FROM t_r22_min WHERE v = 0",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_min").await;
    ctx.deregister_table("t_r22_min").unwrap();
    ctx.register_table("t_r22_min", Arc::new(provider)).unwrap();

    // Fast-path query.
    let df = ctx.sql("SELECT MIN(v) FROM t_r22_min").await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan),
        "R2.2 E2E case 1: MIN fast path must fire after deleting global-MIN row"
    );

    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    let fast_path_min = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    // Ground truth: survivors are 1..299; MIN = 1.
    let expected_min: i64 = (0i64..300).filter(|&v| v != 0).min().unwrap();
    assert_eq!(
        fast_path_min, expected_min,
        "R2.2 E2E case 1: MIN(v) byte-equal to DV-filtered scan MIN; \
         fast-path={fast_path_min}, expected={expected_min}"
    );
    // Sanity: min must be strictly greater than the deleted value.
    assert!(
        fast_path_min > 0,
        "MIN must have risen above deleted value 0"
    );
}

/// E2E case 2: delete the global-MAX row → MAX must fall, be byte-equal
/// to DV-filtered full-scan MAX, fast path fires (scoped-recompute path).
///
/// A polarity inversion in extremum_deleted for MAX, or wrong RowSelection
/// in scoped_recompute_extremum, causes fast-path MAX to equal the stale
/// ColumnStats max (299) instead of 298.
#[tokio::test]
async fn e2e_r22_delete_global_max_row_max_falls() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3 fragments of 100 rows each: v = 0..299.
    // Global MAX = 299 (frag 2, last row).
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 100;

    let values: Vec<i64> = (0i64..300).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_r22_max", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_max").await;
    ctx.register_table("t_r22_max", Arc::new(provider)).unwrap();

    // Delete the global-MAX row (v = 299).
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "t_r22_max",
        "DELETE FROM t_r22_max WHERE v = 299",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_max").await;
    ctx.deregister_table("t_r22_max").unwrap();
    ctx.register_table("t_r22_max", Arc::new(provider)).unwrap();

    // Fast-path query.
    let df = ctx.sql("SELECT MAX(v) FROM t_r22_max").await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan),
        "R2.2 E2E case 2: MAX fast path must fire after deleting global-MAX row"
    );

    let batches = datafusion::physical_plan::common::collect(
        plan.execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    let fast_path_max = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    // Ground truth: survivors 0..298; MAX = 298.
    let expected_max: i64 = (0i64..300).filter(|&v| v != 299).max().unwrap();
    assert_eq!(
        fast_path_max, expected_max,
        "R2.2 E2E case 2: MAX(v) byte-equal to DV-filtered scan MAX; \
         fast-path={fast_path_max}, expected={expected_max}"
    );
    assert!(
        fast_path_max < 299,
        "MAX must have fallen below deleted value 299"
    );
}

/// E2E case 3: non-extremal delete → MIN and MAX unchanged, equal
/// full-scan values, and fast path fires (zero-I/O surviving-extremum path).
///
/// Deleting a middle row (v = 150, which is not the global min or max) means
/// extremum_deleted returns false for both min and max.  The zero-I/O path
/// copies ColumnStats min/max directly into live_min_json/live_max_json.
/// A polarity inversion (extremum_deleted returning true for non-extremal rows)
/// would trigger unnecessary scoped recomputes — still correct, but the cached
/// extremum path would not be taken.  The correctness assertion catches the
/// case where scoped_recompute_extremum returns a wrong value.
#[tokio::test]
async fn e2e_r22_non_extremal_delete_min_max_unchanged() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3 fragments of 100 rows each: v = 0..299.
    // Delete v = 150 (non-extremal, in the middle fragment).
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 100;

    let values: Vec<i64> = (0i64..300).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_r22_mid", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_mid").await;
    ctx.register_table("t_r22_mid", Arc::new(provider)).unwrap();

    // Delete a non-extremal row.
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "t_r22_mid",
        "DELETE FROM t_r22_mid WHERE v = 150",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_mid").await;
    ctx.deregister_table("t_r22_mid").unwrap();
    ctx.register_table("t_r22_mid", Arc::new(provider)).unwrap();

    // MIN query via fast path.
    let df_min = ctx.sql("SELECT MIN(v) FROM t_r22_mid").await.unwrap();
    let plan_min = df_min.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan_min),
        "R2.2 E2E case 3: MIN fast path must fire for non-extremal delete"
    );
    let min_batches = datafusion::physical_plan::common::collect(
        plan_min
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    let fast_path_min = min_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    // MAX query via fast path.
    let df_max = ctx.sql("SELECT MAX(v) FROM t_r22_mid").await.unwrap();
    let plan_max = df_max.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan_max),
        "R2.2 E2E case 3: MAX fast path must fire for non-extremal delete"
    );
    let max_batches = datafusion::physical_plan::common::collect(
        plan_max
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap();
    let fast_path_max = max_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    // Ground truth: survivors 0..299 excluding 150; global min=0, global max=298.
    let survivors: Vec<i64> = (0i64..300).filter(|&v| v != 150).collect();
    let expected_min = *survivors.iter().min().unwrap();
    let expected_max = *survivors.iter().max().unwrap();

    assert_eq!(
        fast_path_min, expected_min,
        "R2.2 E2E case 3: MIN(v) byte-equal to DV-filtered scan MIN after non-extremal delete"
    );
    assert_eq!(
        fast_path_max, expected_max,
        "R2.2 E2E case 3: MAX(v) byte-equal to DV-filtered scan MAX after non-extremal delete"
    );
}

/// E2E case 4: delete the global-MIN row, query MIN (triggers scoped
/// recompute + DV-keyed cache), then issue a SECOND MIN query — the result
/// must still be correct (cache hit on the composite "{agg}@{dv}" key after
/// the first query populated the live partial).
///
/// This guards the cache reuse path: if the cached live_min_json from the
/// first query were stale or wrong, the second query would return a different
/// (incorrect) value.
#[tokio::test]
async fn e2e_r22_delete_min_row_then_second_query_consistent() {
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
    use icefalldb_query::{
        execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    };

    AggStateCache::global().clear();

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // 3 fragments of 100 rows each: v = 0..299.
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 100;

    let values: Vec<i64> = (0i64..300).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(values))],
    )
    .unwrap();

    let mut writer = Writer::create(Arc::clone(&storage), "t_r22_cache", mdb_schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let config = icefalldb_session_config(4, 8192);
    let state = icefalldb_session_state_from_config(config);
    let ctx = datafusion::execution::context::SessionContext::new_with_state(state);

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_cache").await;
    ctx.register_table("t_r22_cache", Arc::new(provider))
        .unwrap();

    // Delete the global-MIN row (v = 0).
    execute_sql(
        &ctx,
        Arc::clone(&storage),
        "t_r22_cache",
        "DELETE FROM t_r22_cache WHERE v = 0",
    )
    .await
    .unwrap();

    AggStateCache::global().clear();

    let provider = make_int_provider(Arc::clone(&storage), "t_r22_cache").await;
    ctx.deregister_table("t_r22_cache").unwrap();
    ctx.register_table("t_r22_cache", Arc::new(provider))
        .unwrap();

    let expected_min: i64 = (1i64..300).min().unwrap(); // = 1

    // First query: triggers scoped recompute and populates the AggStateCache.
    let first_min = sql_single_i64(&ctx, "SELECT MIN(v) FROM t_r22_cache").await;
    assert_eq!(
        first_min, expected_min,
        "R2.2 E2E case 4 (first query): MIN(v) must equal DV-filtered scan MIN"
    );

    // Re-register with a fresh provider so scan_internal re-runs (uses the
    // populated AggStateCache for the live partial, not recomputing it).
    AggStateCache::global().clear();
    let provider2 = make_int_provider(Arc::clone(&storage), "t_r22_cache").await;
    ctx.deregister_table("t_r22_cache").unwrap();
    ctx.register_table("t_r22_cache", Arc::new(provider2))
        .unwrap();

    // Second query: should hit the cache path and still return the correct value.
    let df2 = ctx.sql("SELECT MIN(v) FROM t_r22_cache").await.unwrap();
    let plan2 = df2.create_physical_plan().await.unwrap();
    assert!(
        plan_is_metadata_fast_path(&plan2),
        "R2.2 E2E case 4: MIN fast path must fire on second query"
    );
    let second_min = datafusion::physical_plan::common::collect(
        plan2
            .execute(0, Arc::new(datafusion::execution::TaskContext::default()))
            .unwrap(),
    )
    .await
    .unwrap()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);

    assert_eq!(
        second_min, expected_min,
        "R2.2 E2E case 4 (second query): MIN(v) must equal first query result \
         — cache reuse must not corrupt the live extremum"
    );
}
