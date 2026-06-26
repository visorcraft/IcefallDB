//! Translate DataFusion logical expressions into IcefallDB `Predicate` filters.
//!
//! Only simple predicates that can be evaluated from sidecar statistics are
//! recognized. Everything else returns `None` so the provider can declare the
//! filter as inexact or unsupported.

use arrow::datatypes::Schema;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{BinaryExpr, Expr, Operator};
use icefalldb_core::{Literal, Predicate};

/// Convert a DataFusion `Expr` into a IcefallDB column predicate.
///
/// Recognizes binary comparisons of the form `col OP literal` and
/// `literal OP col`, plus `IsNull` and `IsNotNull`. Returns `None` for any
/// expression that cannot be represented as a single-column predicate.
pub fn expr_to_predicate(expr: &Expr, schema: &Schema) -> Option<(String, Predicate)> {
    match expr {
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            if let Some((col, literal)) = extract_column_literal(left, right, schema) {
                build_comparison(col, *op, literal, false)
            } else if let Some((col, literal)) = extract_column_literal(right, left, schema) {
                build_comparison(col, *op, literal, true)
            } else {
                None
            }
        }
        Expr::IsNull(expr) => {
            let col = expr_as_column(expr, schema)?;
            Some((col.clone(), Predicate::IsNull { column: col }))
        }
        Expr::IsNotNull(expr) => {
            let col = expr_as_column(expr, schema)?;
            Some((col.clone(), Predicate::IsNotNull { column: col }))
        }
        Expr::InList(inlist) if !inlist.negated => {
            let col = expr_as_column(&inlist.expr, schema)?;
            let values: Vec<Literal> = inlist
                .list
                .iter()
                .map(expr_as_literal)
                .collect::<Option<_>>()?;
            if values.is_empty() {
                return None;
            }
            Some((
                col.clone(),
                Predicate::InList {
                    column: col,
                    values,
                },
            ))
        }
        _ => None,
    }
}

/// If `maybe_col` is a column in `schema` and `maybe_lit` is a literal, return
/// the column name and the literal value.
fn extract_column_literal(
    maybe_col: &Expr,
    maybe_lit: &Expr,
    schema: &Schema,
) -> Option<(String, Literal)> {
    let col = expr_as_column(maybe_col, schema)?;
    let lit = expr_as_literal(maybe_lit)?;
    Some((col, lit))
}

/// Return the column name if `expr` references a top-level column in `schema`.
fn expr_as_column(expr: &Expr, schema: &Schema) -> Option<String> {
    match expr {
        Expr::Column(c) => {
            let name = c.name();
            if schema.fields().iter().any(|f| f.name() == name) {
                Some(name.to_string())
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Convert a DataFusion `ScalarValue` literal into a IcefallDB `Literal`.
///
/// Null literals are rejected because they cannot be used in the current
/// predicate representation.
fn scalar_to_literal(value: &ScalarValue) -> Option<Literal> {
    match value {
        ScalarValue::Int8(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::Int16(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::Int32(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::Int64(Some(v)) => Some(Literal::Int64(*v)),
        ScalarValue::UInt8(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::UInt16(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::UInt32(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::UInt64(Some(v)) => Some(Literal::Int64(*v as i64)),
        ScalarValue::Float32(Some(v)) => Some(Literal::Float64(*v as f64)),
        ScalarValue::Float64(Some(v)) => Some(Literal::Float64(*v)),
        ScalarValue::Utf8(Some(v)) | ScalarValue::LargeUtf8(Some(v)) => {
            Some(Literal::String(v.clone()))
        }
        ScalarValue::Boolean(Some(v)) => Some(Literal::Bool(*v)),
        _ => None,
    }
}

fn expr_as_literal(expr: &Expr) -> Option<Literal> {
    match expr {
        Expr::Literal(v, _) => scalar_to_literal(v),
        _ => None,
    }
}

/// Build a predicate from a column, operator, and literal.
///
/// When `swapped` is `true` the original expression was `literal OP col`, so the
/// operator is mirrored (e.g. `5 > col` becomes `col < 5`).
fn build_comparison(
    column: String,
    op: Operator,
    value: Literal,
    swapped: bool,
) -> Option<(String, Predicate)> {
    let result_column = column.clone();
    let pred = match (op, swapped) {
        (Operator::Eq, _) => Predicate::Eq { column, value },
        (Operator::Lt, false) | (Operator::Gt, true) => Predicate::Lt { column, value },
        (Operator::LtEq, false) | (Operator::GtEq, true) => Predicate::Lte { column, value },
        (Operator::Gt, false) | (Operator::Lt, true) => Predicate::Gt { column, value },
        (Operator::GtEq, false) | (Operator::LtEq, true) => Predicate::Gte { column, value },
        _ => return None,
    };
    Some((result_column, pred))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use datafusion::logical_expr::col;

    fn test_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
            Field::new("score", DataType::Float64, true),
        ])
    }

    #[test]
    fn test_eq_and_ordering_predicates() {
        let schema = test_schema();

        let expr = col("id").eq(Expr::Literal(ScalarValue::Int64(Some(42)), None));
        let (col_name, pred) = expr_to_predicate(&expr, &schema).unwrap();
        assert_eq!(col_name, "id");
        assert_eq!(
            pred,
            Predicate::Eq {
                column: "id".into(),
                value: Literal::Int64(42)
            }
        );

        let expr = col("score").gt(Expr::Literal(ScalarValue::Float64(Some(3.5)), None));
        let (_, pred) = expr_to_predicate(&expr, &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::Gt {
                column: "score".into(),
                value: Literal::Float64(3.5)
            }
        );

        // Literal on the left side: 100 <= id becomes id >= 100.
        let expr = Expr::Literal(ScalarValue::Int64(Some(100)), None).lt_eq(col("id"));
        let (_, pred) = expr_to_predicate(&expr, &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::Gte {
                column: "id".into(),
                value: Literal::Int64(100)
            }
        );
    }

    #[test]
    fn test_null_predicates() {
        let schema = test_schema();

        let expr = col("name").is_null();
        let (_, pred) = expr_to_predicate(&expr, &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::IsNull {
                column: "name".into()
            }
        );

        let expr = col("name").is_not_null();
        let (_, pred) = expr_to_predicate(&expr, &schema).unwrap();
        assert_eq!(
            pred,
            Predicate::IsNotNull {
                column: "name".into()
            }
        );
    }

    #[test]
    fn test_unknown_column_returns_none() {
        let schema = test_schema();
        let expr = col("missing").eq(Expr::Literal(ScalarValue::Int64(Some(1)), None));
        assert!(expr_to_predicate(&expr, &schema).is_none());
    }

    #[test]
    fn test_unsupported_expression_returns_none() {
        let schema = test_schema();
        let expr = col("id") + Expr::Literal(ScalarValue::Int64(Some(1)), None);
        assert!(expr_to_predicate(&expr, &schema).is_none());
    }
}
