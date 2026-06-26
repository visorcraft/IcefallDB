use datafusion::common::ScalarValue;
use datafusion::logical_expr::Operator;
use datafusion::prelude::Expr;
use icefalldb_core::index::BTreeIndex;
use std::collections::HashSet;

/// A secondary index this selector can probe: it exposes its indexed column and
/// resolves a value to candidate row ids. Implemented for the fully-parsed
/// [`BTreeIndex`] (JSON path) and for the provider's mmap-backed binary index,
/// so `index_equality_lookup` works uniformly over both.
pub trait IndexLookup {
    /// Name of the indexed column.
    fn column(&self) -> &str;
    /// Row ids that may carry `value` (empty if the value is absent).
    ///
    /// Returns `None` when a derived index cache is malformed and the caller
    /// must fall back to the canonical scan path.
    fn lookup_ids(&self, value: &str) -> Option<Vec<u64>>;
}

impl IndexLookup for BTreeIndex {
    fn column(&self) -> &str {
        &self.definition.column
    }
    fn lookup_ids(&self, value: &str) -> Option<Vec<u64>> {
        Some(self.lookup(value).to_vec())
    }
}

/// If `expr` is `<column> = <literal>` or a non-negated `<column> IN (...)`
/// predicate and `column` matches the index, return the set of row IDs that
/// may contain the value(s).
pub fn index_equality_lookup<I: IndexLookup + ?Sized>(
    index: &I,
    expr: &Expr,
) -> Option<HashSet<u64>> {
    match expr {
        Expr::BinaryExpr(binary) if binary.op == Operator::Eq => {
            let col_name = match binary.left.as_ref() {
                Expr::Column(col) => col.name(),
                _ => return None,
            };
            if col_name != index.column() {
                return None;
            }
            let key = scalar_to_string(binary.right.as_ref())?;
            Some(index.lookup_ids(&key)?.into_iter().collect())
        }
        Expr::InList(inlist) if !inlist.negated => {
            let col_name = match inlist.expr.as_ref() {
                Expr::Column(col) => col.name(),
                _ => return None,
            };
            if col_name != index.column() {
                return None;
            }
            let mut result = HashSet::new();
            for item in &inlist.list {
                let key = scalar_to_string(item)?;
                result.extend(index.lookup_ids(&key)?);
            }
            Some(result)
        }
        _ => None,
    }
}

fn scalar_to_string(value: &Expr) -> Option<String> {
    match value {
        Expr::Literal(ScalarValue::Int64(Some(v)), _) => Some(v.to_string()),
        Expr::Literal(ScalarValue::Utf8(Some(v)), _) => Some(v.clone()),
        Expr::Literal(ScalarValue::LargeUtf8(Some(v)), _) => Some(v.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::Column;
    use datafusion::logical_expr::expr::InList;
    use icefalldb_core::index::IndexDefinition;

    fn make_index() -> BTreeIndex {
        let mut entries = std::collections::BTreeMap::new();
        // Value "a" → row_ids [1], "b" → row_ids [2, 3]
        entries.insert("a".to_string(), vec![1u64]);
        entries.insert("b".to_string(), vec![2u64, 3u64]);
        BTreeIndex {
            definition: IndexDefinition {
                name: "idx".to_string(),
                table: "t".to_string(),
                column: "c".to_string(),
                unique: false,
            },
            snapshot_sequence: 1,
            entries,
        }
    }

    #[test]
    fn test_index_equality_lookup() {
        let index = make_index();
        let expr = Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
            left: Box::new(Expr::Column(Column::from_name("c"))),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some("b".to_string())),
                None,
            )),
        });
        let ids = index_equality_lookup(&index, &expr).unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&2u64));
        assert!(ids.contains(&3u64));
    }

    #[test]
    fn test_index_in_list_lookup() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("c"))),
            list: vec![
                Expr::Literal(ScalarValue::Utf8(Some("a".to_string())), None),
                Expr::Literal(ScalarValue::Utf8(Some("b".to_string())), None),
            ],
            negated: false,
        });
        let ids = index_equality_lookup(&index, &expr).unwrap();
        assert_eq!(ids.len(), 3);
        assert!(ids.contains(&1u64));
        assert!(ids.contains(&2u64));
        assert!(ids.contains(&3u64));
    }

    #[test]
    fn test_index_in_list_empty() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("c"))),
            list: vec![],
            negated: false,
        });
        let ids = index_equality_lookup(&index, &expr).unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_index_in_list_negated() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("c"))),
            list: vec![Expr::Literal(
                ScalarValue::Utf8(Some("a".to_string())),
                None,
            )],
            negated: true,
        });
        assert!(index_equality_lookup(&index, &expr).is_none());
    }

    #[test]
    fn test_index_in_list_wrong_column() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("other_col"))),
            list: vec![Expr::Literal(
                ScalarValue::Utf8(Some("a".to_string())),
                None,
            )],
            negated: false,
        });
        assert!(index_equality_lookup(&index, &expr).is_none());
    }

    #[test]
    fn test_index_in_list_missing_key() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("c"))),
            list: vec![Expr::Literal(
                ScalarValue::Utf8(Some("z".to_string())),
                None,
            )],
            negated: false,
        });
        let ids = index_equality_lookup(&index, &expr).unwrap();
        assert!(ids.is_empty());
    }

    #[test]
    fn test_index_in_list_duplicate_values() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("c"))),
            list: vec![
                Expr::Literal(ScalarValue::Utf8(Some("a".to_string())), None),
                Expr::Literal(ScalarValue::Utf8(Some("a".to_string())), None),
            ],
            negated: false,
        });
        let ids = index_equality_lookup(&index, &expr).unwrap();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&1u64));
    }

    #[test]
    fn test_index_in_list_unsupported_literal() {
        let index = make_index();
        let expr = Expr::InList(InList {
            expr: Box::new(Expr::Column(Column::from_name("c"))),
            list: vec![Expr::Literal(ScalarValue::Null, None)],
            negated: false,
        });
        assert!(index_equality_lookup(&index, &expr).is_none());
    }
}
