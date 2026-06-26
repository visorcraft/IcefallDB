//! Logical optimizer rule that strips redundant widening casts from filter
//! predicates.
//!
//! DataFusion's logical type coercion frequently turns `float32_col > 0.5` into
//! `CAST(float32_col AS Float64) > Float64(0.5)`. Evaluating the cast per row
//! inside the Parquet reader is slower than comparing the column's native type
//! directly. The [`SimplifyCastPredicates`] rule rewrites such comparisons back
//! to the source column type before filters are pushed into table scans.

use arrow::datatypes::{DataType, Schema as ArrowSchema};
use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, DataFusionError, Result as DFResult, ScalarValue};
use datafusion::logical_expr::logical_plan::{Filter, LogicalPlan};
use datafusion::logical_expr::{BinaryExpr, Cast, Expr, Operator};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};

/// Logical optimizer rule that removes redundant widening casts from filter
/// predicates.
#[derive(Debug, Default)]
pub struct SimplifyCastPredicates;

impl SimplifyCastPredicates {
    /// Create a new `SimplifyCastPredicates` rule.
    pub fn new() -> Self {
        Self
    }
}

impl OptimizerRule for SimplifyCastPredicates {
    fn name(&self) -> &str {
        "simplify_cast_predicates"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>, DataFusionError> {
        match plan {
            LogicalPlan::Filter(filter) => {
                let schema = filter.input.schema().inner().as_ref().clone();
                if let Some(new_predicate) = simplify_cast_predicates(&filter.predicate, &schema) {
                    Ok(Transformed::yes(LogicalPlan::Filter(Filter::try_new(
                        new_predicate,
                        filter.input,
                    )?)))
                } else {
                    Ok(Transformed::no(LogicalPlan::Filter(filter)))
                }
            }
            LogicalPlan::TableScan(scan) => {
                let schema = scan.source.schema();
                let mut new_filters = Vec::with_capacity(scan.filters.len());
                let mut changed = false;
                for filter in &scan.filters {
                    if let Some(new_filter) = simplify_cast_predicates(filter, schema.as_ref()) {
                        new_filters.push(new_filter);
                        changed = true;
                    } else {
                        new_filters.push(filter.clone());
                    }
                }
                if changed {
                    let mut new_scan = scan;
                    new_scan.filters = new_filters;
                    Ok(Transformed::yes(LogicalPlan::TableScan(new_scan)))
                } else {
                    Ok(Transformed::no(LogicalPlan::TableScan(scan)))
                }
            }
            _ => Ok(Transformed::no(plan)),
        }
    }
}

/// Strip redundant widening casts from comparison predicates so that filters
/// evaluate on the original (often narrower) column type.
///
/// Returns `Some(new_expr)` when the expression was rewritten, or `None` when
/// no simplification was possible.
pub fn simplify_cast_predicates(expr: &Expr, schema: &ArrowSchema) -> Option<Expr> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if let Some(new_left) = simplify_cast_predicates(left, schema) {
                let new_right =
                    simplify_cast_predicates(right, schema).unwrap_or_else(|| *right.clone());
                return Some(Expr::BinaryExpr(BinaryExpr::new(
                    Box::new(new_left),
                    *op,
                    Box::new(new_right),
                )));
            }
            if let Some(new_right) = simplify_cast_predicates(right, schema) {
                return Some(Expr::BinaryExpr(BinaryExpr::new(
                    left.clone(),
                    *op,
                    Box::new(new_right),
                )));
            }
            simplify_cast_comparison(left, *op, right, schema)
        }
        _ => None,
    }
}

fn simplify_cast_comparison(
    left: &Expr,
    op: Operator,
    right: &Expr,
    schema: &ArrowSchema,
) -> Option<Expr> {
    if !matches!(
        op,
        Operator::Eq | Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq
    ) {
        return None;
    }

    // Find a comparison between a CAST(column) and a literal.
    let (cast_expr, literal_expr) = if is_cast_column(left, schema) && is_literal(right) {
        (left, right)
    } else if is_cast_column(right, schema) && is_literal(left) {
        (right, left)
    } else {
        return None;
    };

    let col = extract_column_from_cast(cast_expr)?;
    let source_type = schema.field_with_name(col.name()).ok()?.data_type();
    let lit_value = extract_literal(literal_expr)?;

    let cast_lit = cast_literal_to_source(lit_value, source_type)?;
    let new_lit = Expr::Literal(cast_lit, None);

    let (new_left, new_right) = if std::ptr::eq(cast_expr, left) {
        (Expr::Column(col.clone()), new_lit)
    } else {
        (new_lit, Expr::Column(col.clone()))
    };

    Some(Expr::BinaryExpr(BinaryExpr::new(
        Box::new(new_left),
        op,
        Box::new(new_right),
    )))
}

fn is_cast_column(expr: &Expr, schema: &ArrowSchema) -> bool {
    match expr {
        Expr::Cast(Cast { expr: inner, .. }) => {
            matches!(inner.as_ref(), Expr::Column(c) if schema.field_with_name(c.name()).is_ok())
        }
        _ => false,
    }
}

fn extract_column_from_cast(expr: &Expr) -> Option<&Column> {
    match expr {
        Expr::Cast(Cast { expr: inner, .. }) => match inner.as_ref() {
            Expr::Column(c) => Some(c),
            _ => None,
        },
        _ => None,
    }
}

fn is_literal(expr: &Expr) -> bool {
    matches!(expr, Expr::Literal(_, _))
}

fn extract_literal(expr: &Expr) -> Option<&ScalarValue> {
    match expr {
        Expr::Literal(v, _) => Some(v),
        _ => None,
    }
}

fn cast_literal_to_source(value: &ScalarValue, source_type: &DataType) -> Option<ScalarValue> {
    use arrow::datatypes::DataType as DT;

    // Only rewrite comparisons where casting the literal to the column's
    // source type preserves the intended ordering/equality semantics.
    let compatible = match (value.data_type(), source_type) {
        (
            DT::Float64 | DT::Float32 | DT::Int64 | DT::Int32 | DT::Int16 | DT::Int8,
            DT::Float64 | DT::Float32,
        ) => true,
        (
            DT::Int64 | DT::Int32 | DT::Int16 | DT::Int8,
            DT::Int64 | DT::Int32 | DT::Int16 | DT::Int8,
        ) => true,
        (DT::Utf8 | DT::LargeUtf8, DT::Utf8 | DT::LargeUtf8) => true,
        // Timestamp/Date/Decimal could be added, but the hot path is numeric.
        _ => false,
    };
    if !compatible {
        return None;
    }

    // Reject casts from floating-point literals to integer columns: truncating
    // 0.5 to 0 would change the comparison semantics.
    if matches!(value.data_type(), DT::Float64 | DT::Float32)
        && matches!(source_type, DT::Int64 | DT::Int32 | DT::Int16 | DT::Int8)
    {
        return None;
    }

    // Reject casts that change interval/time units or signedness in ways that
    // could reorder values.
    if matches!(
        source_type,
        DT::Interval(_) | DT::Timestamp(_, _) | DT::Date32 | DT::Date64
    ) && value.data_type() != *source_type
    {
        return None;
    }

    value.cast_to(source_type).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema as ArrowSchema};
    use datafusion::logical_expr::{col, lit};

    fn cast_expr(expr: Expr, data_type: DataType) -> Expr {
        Expr::Cast(Cast::new(Box::new(expr), data_type))
    }

    fn numeric_schema() -> ArrowSchema {
        ArrowSchema::new(vec![
            Field::new("f32", DataType::Float32, true),
            Field::new("f64", DataType::Float64, true),
            Field::new("i32", DataType::Int32, true),
            Field::new("i64", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ])
    }

    #[test]
    fn test_simplify_float32_cast_to_float64_literal() {
        let schema = numeric_schema();
        // DataFusion coercion turns f32 > 0.5 into CAST(f32 AS Float64) > 0.5.
        let expr = cast_expr(col("f32"), DataType::Float64).gt(lit(0.5f64));
        let simplified = simplify_cast_predicates(&expr, &schema).unwrap();
        assert_eq!(simplified, col("f32").gt(lit(0.5f32)));
    }

    #[test]
    fn test_simplify_int32_cast_to_int64_literal() {
        let schema = numeric_schema();
        let expr = cast_expr(col("i32"), DataType::Int64).gt(lit(20i64));
        let simplified = simplify_cast_predicates(&expr, &schema).unwrap();
        assert_eq!(simplified, col("i32").gt(lit(20i32)));
    }

    #[test]
    fn test_does_not_simplify_float_literal_to_int_column() {
        let schema = numeric_schema();
        // CAST(i32 AS Float64) > 1.5 cannot be rewritten to i32 > 1.5 because
        // the cast would truncate.
        let expr = cast_expr(col("i32"), DataType::Float64).gt(lit(1.5f64));
        assert!(simplify_cast_predicates(&expr, &schema).is_none());
    }

    #[test]
    fn test_does_not_simplify_unsafe_cast() {
        let schema = numeric_schema();
        // i64 > i32(20) is not a redundant cast; the literal cannot be safely
        // narrowed to i32 without potential overflow.
        let expr = col("i64").gt(lit(20i32));
        assert!(simplify_cast_predicates(&expr, &schema).is_none());
    }

    #[test]
    fn test_simplify_reorders_literal_on_left() {
        let schema = numeric_schema();
        let expr = lit(0.5f64).lt(cast_expr(col("f32"), DataType::Float64));
        let simplified = simplify_cast_predicates(&expr, &schema).unwrap();
        assert_eq!(simplified, lit(0.5f32).lt(col("f32")));
    }

    #[test]
    fn test_simplify_nested_in_and() {
        let schema = numeric_schema();
        let left = cast_expr(col("f32"), DataType::Float64).gt(lit(0.5f64));
        let right = cast_expr(col("i32"), DataType::Int64).lt(lit(100i64));
        let expr = left.and(right);
        let simplified = simplify_cast_predicates(&expr, &schema).unwrap();
        assert_eq!(
            simplified,
            col("f32").gt(lit(0.5f32)).and(col("i32").lt(lit(100i32)))
        );
    }

    #[test]
    fn test_no_change_when_no_cast() {
        let schema = numeric_schema();
        let expr = col("f32").gt(lit(0.5f32));
        assert!(simplify_cast_predicates(&expr, &schema).is_none());
    }
}
