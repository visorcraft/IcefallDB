//! Physical optimizer rule that pushes build-side min/max filters into the
//! probe-side `IcefallDBScanExec`.
//!
//! For every `HashJoinExec` with a single-column equi-condition and a build side
//! whose cardinality is known, the rule extracts the minimum and maximum value of
//! the build-side join key and adds a range predicate (`probe_key >= min AND
//! probe_key <= max`) to the probe side scan. This lets IcefallDB's manifest-level
//! pruning skip files/row groups that cannot match before any Parquet data is
//! decoded.

use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use datafusion::common::config::ConfigOptions;
use datafusion::common::stats::Precision;
use datafusion::common::{JoinType, Result as DFResult, ScalarValue};
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{binary, in_list, lit, Column};
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::joins::HashJoinExec;
use datafusion::physical_plan::ExecutionPlan;

use crate::error::QueryError;
use crate::scalar_codec::json_to_scalar_value;
use crate::scan::IcefallDBScanExec;
use crate::session::IcefallDBConfig;
use crate::Result;

/// A filter that can be pushed into a `IcefallDBScanExec`.
#[derive(Debug, Clone)]
pub(crate) enum FilterSpec {
    /// `column IN (values...)`.
    InList {
        column: String,
        values: Vec<ScalarValue>,
    },
    /// `column >= min AND column <= max`.
    Range {
        column: String,
        min: ScalarValue,
        max: ScalarValue,
    },
}

/// Pair of physical expressions representing the build-side and probe-side
/// equi-join keys.
pub(crate) type EquiKeyPair = (Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>);

/// Physical optimizer rule that pushes build-side min/max predicates into the
/// probe-side scan.
#[derive(Debug, Default)]
pub struct DynamicFilterPushdown;

impl DynamicFilterPushdown {
    /// Create a new `DynamicFilterPushdown` optimizer rule.
    pub fn new() -> Self {
        Self
    }
}

impl PhysicalOptimizerRule for DynamicFilterPushdown {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let enabled = config
            .extensions
            .get::<IcefallDBConfig>()
            .map(|c| c.dynamic_filter_pushdown)
            .unwrap_or(true);
        if !enabled {
            return Ok(plan);
        }
        optimize_internal(plan, config)
    }

    fn name(&self) -> &str {
        "dynamic_filter_pushdown"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn optimize_internal(
    plan: Arc<dyn ExecutionPlan>,
    config: &ConfigOptions,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    // Recurse bottom-up so children are optimized first.
    let plan = crate::rules::optimize_children(plan, config, |plan, config| {
        optimize_internal(plan, config)
    })?;

    let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
        return Ok(plan);
    };

    let Some((build_key, probe_key)) = extract_equi_keys(join)? else {
        return Ok(plan);
    };

    let Some((min, max)) = build_side_min_max(join.left(), &build_key)? else {
        return Ok(plan);
    };

    let probe_col = match column_name(&probe_key) {
        Some(name) => name,
        None => return Ok(plan),
    };

    let spec = FilterSpec::Range {
        column: probe_col,
        min,
        max,
    };

    let new_probe = push_filter_specs_into_scan(Arc::clone(join.right()), &[spec])?;
    let mut new_children: Vec<Arc<dyn ExecutionPlan>> =
        plan.children().iter().map(|c| Arc::clone(*c)).collect();
    new_children[1] = new_probe;
    plan.with_new_children(new_children)
}

/// Extract a single-column equi-join key pair from a hash join.
///
/// Returns `Ok(None)` for multi-column join conditions or non-inner join types.
pub(crate) fn extract_equi_keys(join: &HashJoinExec) -> DFResult<Option<EquiKeyPair>> {
    match join.join_type() {
        JoinType::Inner => {}
        _ => return Ok(None),
    }
    if join.on().len() != 1 {
        return Ok(None);
    }
    let (build_key, probe_key) = &join.on()[0];
    Ok(Some((Arc::clone(build_key), Arc::clone(probe_key))))
}

/// Return the build-side row count if it is known exactly.
pub(crate) fn build_side_row_count(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    // Recurse through no-op (row-count-preserving) operators first; their own
    // statistics are often inexact even when the child's statistics are exact.
    if let Some(repart) =
        plan.downcast_ref::<datafusion::physical_plan::repartition::RepartitionExec>()
    {
        return build_side_row_count(repart.input());
    }
    if let Some(coalesce) = plan
        .downcast_ref::<datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec>(
    ) {
        return build_side_row_count(coalesce.input());
    }

    match plan.partition_statistics(None) {
        Ok(stats) => match stats.num_rows {
            Precision::Exact(n) => Some(n),
            _ => None,
        },
        Err(_) => None,
    }
}

/// Return the column name if the expression is a simple column reference.
pub(crate) fn column_name(expr: &Arc<dyn PhysicalExpr>) -> Option<String> {
    (expr.as_ref() as &dyn std::any::Any)
        .downcast_ref::<Column>()
        .map(|c| c.name().to_string())
}

/// Compute the global min/max of the build-side join key.
///
/// Tries sidecar stats first (cheapest), then falls back to the child plan's
/// computed statistics.
fn build_side_min_max(
    plan: &Arc<dyn ExecutionPlan>,
    key_expr: &Arc<dyn PhysicalExpr>,
) -> Result<Option<(ScalarValue, ScalarValue)>> {
    if let Some(name) = column_name(key_expr) {
        if let Some(scan) = plan.downcast_ref::<IcefallDBScanExec>() {
            if let Some(mm) = min_max_from_row_groups(scan, &name)? {
                return Ok(Some(mm));
            }
        }
    }
    min_max_from_statistics(plan, key_expr)
}

/// Fold per-row-group sidecar min/max statistics for the build key.
fn min_max_from_row_groups(
    scan: &IcefallDBScanExec,
    col_name: &str,
) -> Result<Option<(ScalarValue, ScalarValue)>> {
    let schema = scan.schema();
    let field = schema
        .field_with_name(col_name)
        .map_err(|_| QueryError::StatsUnavailable)?;
    let data_type = field.data_type().clone();

    let mut min: Option<ScalarValue> = None;
    let mut max: Option<ScalarValue> = None;

    for rg in scan.planned_row_groups() {
        if rg.fallback {
            return Ok(None);
        }
        let stats = rg
            .meta
            .columns
            .get(col_name)
            .ok_or(QueryError::StatsUnavailable)?;
        let rg_min = match stats
            .min
            .as_ref()
            .and_then(|v| json_to_scalar_value(v, &data_type))
        {
            Some(v) => v,
            None => return Ok(None),
        };
        let rg_max = match stats
            .max
            .as_ref()
            .and_then(|v| json_to_scalar_value(v, &data_type))
        {
            Some(v) => v,
            None => return Ok(None),
        };

        min = Some(match min {
            Some(current) => pick_min(&current, &rg_min),
            None => rg_min,
        });
        max = Some(match max {
            Some(current) => pick_max(&current, &rg_max),
            None => rg_max,
        });
    }

    Ok(min.zip(max))
}

/// Extract min/max from the build side's computed statistics.
fn min_max_from_statistics(
    plan: &Arc<dyn ExecutionPlan>,
    key_expr: &Arc<dyn PhysicalExpr>,
) -> Result<Option<(ScalarValue, ScalarValue)>> {
    let col = match (key_expr.as_ref() as &dyn std::any::Any).downcast_ref::<Column>() {
        Some(c) => c,
        None => return Ok(None),
    };
    let stats = plan
        .partition_statistics(None)
        .map_err(QueryError::DataFusion)?;
    let col_stats = match stats.column_statistics.get(col.index()) {
        Some(s) => s,
        None => return Ok(None),
    };
    let min = match &col_stats.min_value {
        Precision::Exact(v) => v.clone(),
        _ => return Ok(None),
    };
    let max = match &col_stats.max_value {
        Precision::Exact(v) => v.clone(),
        _ => return Ok(None),
    };
    Ok(Some((min, max)))
}

/// Push a list of filter specs into the first `IcefallDBScanExec` found under
/// `plan`.
///
/// The filters are applied via a [`FilterExec`] placed immediately above the
/// scan.  This lets DataFusion handle type coercion between the literal values
/// (which may be `Utf8`) and the decoded Parquet columns (which may be
/// `LargeUtf8`/`Utf8View` depending on the file), avoiding errors inside
/// Parquet's row-filter evaluator.
pub(crate) fn push_filter_specs_into_scan(
    plan: Arc<dyn ExecutionPlan>,
    specs: &[FilterSpec],
) -> DFResult<Arc<dyn ExecutionPlan>> {
    if let Some(scan) = plan.downcast_ref::<IcefallDBScanExec>() {
        let schema = scan.schema();
        let mut built = Vec::with_capacity(specs.len());
        for spec in specs {
            match build_filter_expr(spec, &schema) {
                Ok(expr) if filter_fits_schema(&expr, &schema) => built.push(expr),
                Ok(_) | Err(_) => {
                    // Skip specs that cannot be evaluated in this scan's schema.
                }
            }
        }
        if built.is_empty() {
            return Ok(plan);
        }
        // Combine specs with AND.
        let combined = built
            .into_iter()
            .reduce(|a, b| {
                binary(a, Operator::And, b, &schema).expect("AND of two bool expressions")
            })
            .expect("built is non-empty");
        return datafusion::physical_plan::filter::FilterExec::try_new(
            combined,
            Arc::new(scan.clone()),
        )
        .map(|f| Arc::new(f) as Arc<dyn ExecutionPlan>);
    }

    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }
    let new_children: Vec<Arc<dyn ExecutionPlan>> = children
        .iter()
        .map(|child| push_filter_specs_into_scan(Arc::clone(*child), specs))
        .collect::<DFResult<_>>()?;
    plan.with_new_children(new_children)
}

/// Build a physical expression for `spec` against `schema`.
fn build_filter_expr(spec: &FilterSpec, schema: &Schema) -> Result<Arc<dyn PhysicalExpr>> {
    let col_name = match spec {
        FilterSpec::InList { column, .. } | FilterSpec::Range { column, .. } => column,
    };
    let col_idx = schema
        .index_of(col_name)
        .map_err(|_| QueryError::Other(format!("column '{col_name}' not in scan schema")))?;
    let col_expr = Arc::new(Column::new(col_name, col_idx)) as Arc<dyn PhysicalExpr>;
    let col_type = schema.field(col_idx).data_type();

    match spec {
        FilterSpec::InList { values, .. } => {
            let list: Vec<Arc<dyn PhysicalExpr>> = values
                .iter()
                .map(|v| {
                    cast_scalar_for_column(v, col_type).map(|s| lit(s) as Arc<dyn PhysicalExpr>)
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(in_list(col_expr, list, &false, schema).map_err(QueryError::DataFusion)?)
        }
        FilterSpec::Range { min, max, .. } => {
            let min = cast_scalar_for_column(min, col_type)?;
            let max = cast_scalar_for_column(max, col_type)?;
            let ge = binary(col_expr.clone(), Operator::GtEq, lit(min), schema)
                .map_err(QueryError::DataFusion)?;
            let le = binary(col_expr, Operator::LtEq, lit(max), schema)
                .map_err(QueryError::DataFusion)?;
            Ok(binary(ge, Operator::And, le, schema).map_err(QueryError::DataFusion)?)
        }
    }
}

/// Cast a scalar value to the column's data type so filter expressions are type-safe.
fn cast_scalar_for_column(value: &ScalarValue, target: &DataType) -> Result<ScalarValue> {
    if value.data_type() == *target {
        return Ok(value.clone());
    }
    value.cast_to(target).map_err(|e| {
        QueryError::Other(format!(
            "cannot cast scalar {:?} to {:?}: {}",
            value, target, e
        ))
    })
}

/// Return true when every column referenced by `filter` exists in `schema`.
fn filter_fits_schema(
    filter: &Arc<dyn PhysicalExpr>,
    schema: &arrow::datatypes::SchemaRef,
) -> bool {
    collect_columns(filter)
        .iter()
        .all(|c| c.index() < schema.fields().len())
}

fn pick_min(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    match a.partial_cmp(b) {
        Some(std::cmp::Ordering::Greater) => b.clone(),
        _ => a.clone(),
    }
}

fn pick_max(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    match a.partial_cmp(b) {
        Some(std::cmp::Ordering::Less) => b.clone(),
        _ => a.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};

    fn test_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int64, true),
        ])
    }

    #[test]
    fn test_build_range_filter_expr() {
        let schema = test_schema();
        let spec = FilterSpec::Range {
            column: "a".into(),
            min: ScalarValue::Int32(Some(1)),
            max: ScalarValue::Int32(Some(10)),
        };
        let expr = build_filter_expr(&spec, &schema).unwrap();
        let cols = collect_columns(&expr);
        assert_eq!(cols.len(), 1);
        let col = cols.iter().next().unwrap();
        assert_eq!(col.name(), "a");
        assert_eq!(col.index(), 0);
    }

    #[test]
    fn test_build_in_list_filter_expr() {
        let schema = test_schema();
        let spec = FilterSpec::InList {
            column: "b".into(),
            values: vec![ScalarValue::Int64(Some(1)), ScalarValue::Int64(Some(2))],
        };
        let expr = build_filter_expr(&spec, &schema).unwrap();
        let cols = collect_columns(&expr);
        assert_eq!(cols.len(), 1);
        let col = cols.iter().next().unwrap();
        assert_eq!(col.name(), "b");
        assert_eq!(col.index(), 1);
    }
}
