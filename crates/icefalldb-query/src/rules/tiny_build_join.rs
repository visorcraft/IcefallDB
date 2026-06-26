//! Physical optimizer rule that specializes hash joins with a tiny build side.
//!
//! When the build side of an inner equi-join is small (by default at most
//! `tiny_build_join_threshold` rows), this rule executes the build side, materializes the
//! distinct join-key values, and pushes an `IN (...)` list plus a derived
//! min/max range filter into the probe-side `IcefallDBScanExec`.  This turns a
//! hash join into a selective scan followed by a cheap verification join.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array, Int64Array, LargeStringArray, StringArray};
use arrow::compute::cast;
use arrow::datatypes::DataType;
use datafusion::common::config::ConfigOptions;
use datafusion::common::Result as DFResult;
use datafusion::common::ScalarValue;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::TaskContext;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_datasource::source::DataSourceExec;
use futures::StreamExt;

use crate::rules::dynamic_filter_pushdown::{
    build_side_row_count, column_name, extract_equi_keys, push_filter_specs_into_scan, FilterSpec,
};
use crate::session::IcefallDBConfig;

/// Physical optimizer rule that specializes tiny-build-side hash joins.
#[derive(Debug, Default)]
pub struct TinyBuildJoin;

impl TinyBuildJoin {
    /// Create a new `TinyBuildJoin` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for TinyBuildJoin {
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
        "tiny_build_join"
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
    // Recurse bottom-up so children are optimized first.
    let plan = crate::rules::optimize_children(plan, config, |plan, config| {
        optimize_internal(plan, config, threshold)
    })?;

    let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
        return Ok(plan);
    };

    let Some((build_key, probe_key)) = extract_equi_keys(join)? else {
        return Ok(plan);
    };

    let build = join.left();
    let Some(build_rows) = build_side_row_count(build) else {
        return Ok(plan);
    };
    if build_rows > threshold {
        return Ok(plan);
    }

    let Some(build_col_name) = column_name(&build_key) else {
        return Ok(plan);
    };
    let Some(probe_col_name) = column_name(&probe_key) else {
        return Ok(plan);
    };

    // Skip native DataSourceExec probe sides. Filters can only be pushed into
    // IcefallDBScanExec, and rebuilding the join for native Parquet scans adds
    // overhead without benefit.
    if contains_native_datasource(join.right()) {
        return Ok(plan);
    }

    let Some(values) = execute_build_side_values(build, &build_col_name) else {
        return Ok(plan);
    };
    if values.is_empty() {
        return Ok(plan);
    }

    let specs = build_specs(&probe_col_name, values, build_rows);
    let new_probe = push_filter_specs_into_scan(Arc::clone(join.right()), &specs)?;

    Arc::clone(&plan).with_new_children(vec![Arc::clone(build), new_probe])
}

/// Execute the build side and collect distinct non-null key values.
fn execute_build_side_values(
    build: &Arc<dyn ExecutionPlan>,
    key_name: &str,
) -> Option<Vec<ScalarValue>> {
    // Run the build plan on a dedicated thread so we can block on a fresh
    // current-thread Tokio runtime.  This avoids deadlocking when the optimizer
    // is invoked from within an async task.
    let build = Arc::clone(build);
    let key_name = key_name.to_string();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread().build().ok()?;
        rt.block_on(async {
            let runtime = Arc::new(RuntimeEnv::default());
            let ctx: Arc<TaskContext> = Arc::new(TaskContext::default().with_runtime(runtime));
            let mut values = HashSet::new();
            for partition in 0..build.properties().partitioning.partition_count() {
                let mut stream = build.execute(partition, Arc::clone(&ctx)).ok()?;
                while let Some(batch) = stream.next().await {
                    let batch = batch.ok()?;
                    let array = batch.column_by_name(&key_name)?;
                    extend_values(&mut values, array)?;
                }
            }
            Some(values.into_iter().collect())
        })
    })
    .join()
    .ok()?
}

fn extend_values(set: &mut HashSet<ScalarValue>, array: &ArrayRef) -> Option<()> {
    match array.data_type() {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>()?;
            for v in arr.iter().flatten() {
                set.insert(ScalarValue::Int32(Some(v)));
            }
        }
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>()?;
            for v in arr.iter().flatten() {
                set.insert(ScalarValue::Int64(Some(v)));
            }
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>()?;
            for v in arr.iter().flatten() {
                set.insert(ScalarValue::Utf8(Some(v.to_string())));
            }
        }
        DataType::LargeUtf8 => {
            let arr = array.as_any().downcast_ref::<LargeStringArray>()?;
            for v in arr.iter().flatten() {
                set.insert(ScalarValue::LargeUtf8(Some(v.to_string())));
            }
        }
        _ => {
            // For other integer-like types, fall back to casting to Int64.
            // String types are handled above; anything else is unsupported.
            let casted = cast(array.as_ref(), &DataType::Int64).ok()?;
            extend_values(set, &casted)?;
        }
    }
    Some(())
}

/// Return true if `plan` (or any descendant) is a DataFusion native DataSourceExec.
fn contains_native_datasource(plan: &Arc<dyn ExecutionPlan>) -> bool {
    if plan.downcast_ref::<DataSourceExec>().is_some() {
        return true;
    }
    plan.children()
        .iter()
        .any(|child| contains_native_datasource(child))
}

fn build_specs(column: &str, values: Vec<ScalarValue>, _build_rows: usize) -> Vec<FilterSpec> {
    let mut specs = Vec::new();
    specs.push(FilterSpec::InList {
        column: column.to_string(),
        values: values.clone(),
    });

    if let Some((min, max)) = scalar_min_max(&values) {
        specs.push(FilterSpec::Range {
            column: column.to_string(),
            min,
            max,
        });
    }
    specs
}

fn scalar_min_max(values: &[ScalarValue]) -> Option<(ScalarValue, ScalarValue)> {
    if values.is_empty() {
        return None;
    }
    let mut min = values[0].clone();
    let mut max = values[0].clone();
    for v in &values[1..] {
        if let Some(std::cmp::Ordering::Less) = v.partial_cmp(&min) {
            min = v.clone();
        }
        if let Some(std::cmp::Ordering::Greater) = v.partial_cmp(&max) {
            max = v.clone();
        }
    }
    Some((min, max))
}
