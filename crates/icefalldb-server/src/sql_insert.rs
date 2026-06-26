use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use datafusion::sql::parser::DFParser;
use icefalldb_core::catalog::Catalog;
use icefalldb_core::metadata::Schema;
use icefalldb_core::storage::Storage;
use icefalldb_core::{IcefallDBError, Result};
use sqlparser::ast::{Expr, SetExpr, Statement as SqlStatement, Value, ValueWithSpan};
use std::sync::Arc;

pub async fn parse_insert_values(
    storage: &dyn Storage,
    sql: &str,
) -> Result<(String, RecordBatch)> {
    let statements = DFParser::parse_sql(sql).map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    let statement = statements
        .into_iter()
        .next()
        .ok_or_else(|| IcefallDBError::Other("empty SQL".into()))?;

    let datafusion::sql::parser::Statement::Statement(stmt) = statement else {
        return Err(IcefallDBError::Other("expected INSERT".into()));
    };
    let SqlStatement::Insert(insert) = *stmt else {
        return Err(IcefallDBError::Other("expected INSERT".into()));
    };

    let table_name = insert.table.to_string();
    let schema = load_table_schema(storage, &table_name).await?;
    let arrow_schema = icefalldb_core::schema_util::arrow_schema_from_icefalldb(&schema)
        .ok_or_else(|| IcefallDBError::Other("failed to convert schema".into()))?;

    let source = insert
        .source
        .as_deref()
        .ok_or_else(|| IcefallDBError::Other("expected VALUES source".into()))?;
    let SetExpr::Values(values) = &*source.body else {
        return Err(IcefallDBError::Other("expected VALUES".into()));
    };

    let mut column_arrays: Vec<(usize, Vec<&Expr>)> = Vec::new();
    for row in &values.rows {
        for (col_idx, expr) in row.content.iter().enumerate() {
            if column_arrays.len() <= col_idx {
                column_arrays.push((col_idx, Vec::new()));
            }
            column_arrays[col_idx].1.push(expr);
        }
    }

    if column_arrays.len() != arrow_schema.fields().len() {
        return Err(IcefallDBError::InvalidSchema {
            reason: format!(
                "INSERT provides {} columns but table has {}",
                column_arrays.len(),
                arrow_schema.fields().len()
            ),
            path: table_name,
        });
    }

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    for (col_idx, exprs) in column_arrays {
        let field = arrow_schema.field(col_idx);
        let array = exprs_to_array(field.data_type(), &exprs)?;
        arrays.push(array);
    }

    let batch = RecordBatch::try_new(arrow_schema, arrays)
        .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    Ok((table_name, batch))
}

async fn load_table_schema(storage: &dyn Storage, table: &str) -> Result<Schema> {
    // If a manifest exists, Catalog already has the schema loaded.
    if let Ok(catalog) = Catalog::load(storage, table).await {
        if let Some(schema) = catalog.latest_schema() {
            return Ok(schema.clone());
        }
    }

    // Otherwise read the schema pointer directly (empty table, no manifests yet).
    let pointer_path = format!("{}/_schema.json", table);
    let data = storage
        .read(&pointer_path)
        .await
        .map_err(|_| IcefallDBError::SchemaNotFound {
            path: pointer_path.clone(),
        })?;
    let pointer: serde_json::Value = serde_json::from_slice(&data)?;
    let schema_id = pointer
        .get("latest")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| IcefallDBError::InvalidSchemaPointer {
            path: pointer_path.clone(),
        })?;
    let schema_path = format!("{}/{}", table, Schema::filename(schema_id));
    let schema: Schema = serde_json::from_slice(&storage.read(&schema_path).await?)?;
    Ok(schema)
}

fn exprs_to_array(data_type: &DataType, exprs: &[&Expr]) -> Result<ArrayRef> {
    match data_type {
        DataType::Int64 => {
            let values: Vec<Option<i64>> = exprs
                .iter()
                .map(|e| match e {
                    Expr::Value(ValueWithSpan {
                        value: Value::Number(n, _),
                        ..
                    }) => n.parse().ok(),
                    _ => None,
                })
                .collect();
            Ok(Arc::new(Int64Array::from(values)))
        }
        DataType::Utf8 => {
            let values: Vec<Option<String>> = exprs
                .iter()
                .map(|e| match e {
                    Expr::Value(ValueWithSpan {
                        value: Value::SingleQuotedString(s),
                        ..
                    }) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            Ok(Arc::new(StringArray::from(values)))
        }
        other => Err(IcefallDBError::TypeNotSupported(other.to_string())),
    }
}
