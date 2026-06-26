use crate::metadata::schema::icefalldb_type_to_arrow;
use crate::metadata::Schema as IcefallDBSchema;
use crate::{IcefallDBError, Result};
use arrow::datatypes::TimeUnit;
use serde_json::{json, Value};

/// Convert a IcefallDB schema into an Iceberg schema JSON object.
///
/// Iceberg schemas are arrays of fields with monotonically increasing integer
/// IDs. The conversion reuses the IcefallDB `field_id` when present; otherwise it
/// assigns temporary IDs starting from 1. Column IDs in Iceberg must be
/// positive.
pub fn to_iceberg_schema(schema: &IcefallDBSchema) -> Result<Value> {
    let mut fields = Vec::with_capacity(schema.columns.len());
    for (idx, col) in schema.columns.iter().enumerate() {
        let field_id = if col.field_id > 0 {
            col.field_id
        } else {
            (idx + 1) as i32
        };
        let arrow_type = icefalldb_type_to_arrow(&col.r#type).ok_or_else(|| {
            IcefallDBError::TypeNotSupported(format!(
                "column '{}' has unsupported IcefallDB type '{}'",
                col.name, col.r#type
            ))
        })?;
        let iceberg_type = arrow_type_to_iceberg(&arrow_type).ok_or_else(|| {
            IcefallDBError::TypeNotSupported(format!(
                "column '{}' has unsupported Arrow type {:?}",
                col.name, arrow_type
            ))
        })?;
        fields.push(json!({
            "id": field_id,
            "name": col.name,
            "type": iceberg_type,
            "required": !col.nullable,
        }));
    }
    Ok(json!({
        "type": "struct",
        "schema-id": schema.schema_id as i64,
        "fields": fields,
    }))
}

/// Map an Arrow [`DataType`] to an Iceberg type JSON value.
///
/// Returns `None` for types that cannot be represented in Iceberg v2.
pub fn arrow_type_to_iceberg(dt: &arrow::datatypes::DataType) -> Option<Value> {
    use arrow::datatypes::DataType;
    match dt {
        DataType::Boolean => Some(json!("boolean")),
        DataType::Int8 | DataType::Int16 | DataType::Int32 => Some(json!("int")),
        DataType::Int64 => Some(json!("long")),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => None,
        DataType::Float32 => Some(json!("float")),
        DataType::Float64 => Some(json!("double")),
        DataType::Utf8 | DataType::LargeUtf8 => Some(json!("string")),
        DataType::Binary | DataType::LargeBinary => Some(json!("binary")),
        DataType::FixedSizeBinary(n) => Some(json!({"type": "fixed", "length": n})),
        DataType::Timestamp(TimeUnit::Microsecond, _) => Some(json!("timestamp")),
        DataType::Timestamp(_, _) => None,
        DataType::Date32 | DataType::Date64 => Some(json!("date")),
        DataType::Decimal128(precision, scale) => Some(json!({
            "type": "decimal",
            "precision": precision,
            "scale": scale,
        })),
        DataType::List(field) => {
            let item = arrow_type_to_iceberg(field.data_type())?;
            Some(json!({
                "type": "list",
                "element-id": 1,
                "element": item,
                "element-required": !field.is_nullable(),
            }))
        }
        DataType::LargeList(field) => {
            let item = arrow_type_to_iceberg(field.data_type())?;
            Some(json!({
                "type": "list",
                "element-id": 1,
                "element": item,
                "element-required": !field.is_nullable(),
            }))
        }
        DataType::Struct(fields) => {
            let mut iceberg_fields = Vec::with_capacity(fields.len());
            for (idx, field) in fields.iter().enumerate() {
                iceberg_fields.push(json!({
                    "id": (idx + 1) as i64,
                    "name": field.name(),
                    "type": arrow_type_to_iceberg(field.data_type())?,
                    "required": !field.is_nullable(),
                }));
            }
            Some(json!({
                "type": "struct",
                "fields": iceberg_fields,
            }))
        }
        DataType::Map(field, _) => {
            let entry_fields = if let DataType::Struct(children) = field.data_type() {
                children
            } else {
                return None;
            };
            let key_field = entry_fields.iter().find(|f| f.name() == "key")?;
            let value_field = entry_fields.iter().find(|f| f.name() == "value")?;
            Some(json!({
                "type": "map",
                "key-id": 1,
                "key": arrow_type_to_iceberg(key_field.data_type())?,
                "value-id": 2,
                "value": arrow_type_to_iceberg(value_field.data_type())?,
                "value-required": !value_field.is_nullable(),
            }))
        }
        _ => None,
    }
}
