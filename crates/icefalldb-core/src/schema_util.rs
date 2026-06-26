use crate::metadata::{Column, Schema};
#[cfg(test)]
use arrow::datatypes::Fields;
use arrow::datatypes::{DataType, Field, SchemaRef, TimeUnit};

/// Default target number of rows per row group.
pub const DEFAULT_ROW_GROUP_TARGET_ROWS: usize = 1_000_000;

/// Default target uncompressed byte size per row group.
pub const DEFAULT_ROW_GROUP_TARGET_BYTES: usize = 134_217_728;

/// Convert an Arrow [`DataType`] into a IcefallDB type string.
///
/// This is the inverse of [`icefalldb_type_to_arrow`].
pub fn arrow_type_to_icefalldb(dt: &DataType) -> String {
    match dt {
        DataType::Int8 => "int8".into(),
        DataType::Int16 => "int16".into(),
        DataType::Int32 => "int32".into(),
        DataType::Int64 => "int64".into(),
        DataType::UInt8 => "uint8".into(),
        DataType::UInt16 => "uint16".into(),
        DataType::UInt32 => "uint32".into(),
        DataType::UInt64 => "uint64".into(),
        DataType::Float32 => "float32".into(),
        DataType::Float64 => "float64".into(),
        DataType::Boolean => "bool".into(),
        DataType::Utf8 => "utf8".into(),
        DataType::LargeUtf8 => "large_utf8".into(),
        DataType::Binary => "binary".into(),
        DataType::LargeBinary => "large_binary".into(),
        DataType::FixedSizeBinary(n) => format!("fixed_size_binary({})", n),
        DataType::Timestamp(TimeUnit::Microsecond, None) => "timestamp[us]".into(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(_)) => "timestamp[us]".into(),
        DataType::Decimal128(precision, scale) => format!("decimal128({},{})", precision, scale),
        DataType::List(field) => format!("list<{}>", arrow_type_to_icefalldb(field.data_type())),
        DataType::LargeList(field) => {
            format!("large_list<{}>", arrow_type_to_icefalldb(field.data_type()))
        }
        DataType::Struct(fields) => {
            let inner: Vec<String> = fields
                .iter()
                .map(|f| format!("{}: {}", f.name(), arrow_type_to_icefalldb(f.data_type())))
                .collect();
            format!("struct<{}>", inner.join(", "))
        }
        DataType::Map(field, _) => {
            let entry_fields = if let DataType::Struct(fields) = field.data_type() {
                fields
            } else {
                // Map fields in Arrow are always wrapped in a struct named "entries".
                return "map<utf8,utf8>".into();
            };
            let key = entry_fields
                .iter()
                .find(|f| f.name() == "key")
                .map(|f| arrow_type_to_icefalldb(f.data_type()))
                .unwrap_or_else(|| "utf8".into());
            let value = entry_fields
                .iter()
                .find(|f| f.name() == "value")
                .map(|f| arrow_type_to_icefalldb(f.data_type()))
                .unwrap_or_else(|| "utf8".into());
            format!("map<{},{}>", key, value)
        }
        other => format!("{:?}", other),
    }
}

/// Convert an Arrow [`SchemaRef`] into a IcefallDB [`Schema`].
///
/// The returned schema has `schema_id` set to `1`, no partitioning or sort
/// order, and default row-group targets. Column `field_id`s are unassigned
/// (`0`) and should be assigned before the schema is persisted.
pub fn arrow_schema_to_icefalldb(schema: SchemaRef) -> Schema {
    let columns = schema
        .fields()
        .iter()
        .map(|f| arrow_field_to_column(f))
        .collect();
    Schema {
        schema_id: 1,
        columns,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: DEFAULT_ROW_GROUP_TARGET_ROWS,
        row_group_target_bytes: DEFAULT_ROW_GROUP_TARGET_BYTES,
        dropped_columns: vec![],
        max_field_id: 0,
    }
}

/// Convert an Arrow [`Field`] into a IcefallDB [`Column`].
///
/// The column's `field_id` is unassigned (`0`).
pub fn arrow_field_to_column(field: &Field) -> Column {
    Column::new(
        field.name(),
        arrow_type_to_icefalldb(field.data_type()),
        field.is_nullable(),
    )
}

/// Convert a IcefallDB type string into an Arrow [`DataType`].
///
/// This delegates to the canonical parser in [`crate::metadata::schema`].
pub fn icefalldb_type_to_arrow(type_str: &str) -> Option<DataType> {
    crate::metadata::schema::icefalldb_type_to_arrow(type_str)
}

/// Build an Arrow [`SchemaRef`] from a IcefallDB [`Schema`].
///
/// Returns `None` if any column declares an unsupported type.
pub fn arrow_schema_from_icefalldb(schema: &Schema) -> Option<SchemaRef> {
    let fields: Vec<_> = schema
        .columns
        .iter()
        .map(|c| icefalldb_type_to_arrow(&c.r#type).map(|dt| Field::new(&c.name, dt, c.nullable)))
        .collect::<Option<_>>()?;
    Some(std::sync::Arc::new(arrow::datatypes::Schema::new(fields)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_arrow_type_to_icefalldb_primitives() {
        assert_eq!(arrow_type_to_icefalldb(&DataType::Int64), "int64");
        assert_eq!(arrow_type_to_icefalldb(&DataType::Utf8), "utf8");
        assert_eq!(arrow_type_to_icefalldb(&DataType::Boolean), "bool");
    }

    #[test]
    fn test_arrow_type_to_icefalldb_complex() {
        let dt = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
        assert_eq!(arrow_type_to_icefalldb(&dt), "list<int64>");

        let fields = Fields::from(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Utf8, true),
        ]);
        let dt = DataType::Struct(fields);
        assert_eq!(arrow_type_to_icefalldb(&dt), "struct<x: int64, y: utf8>");
    }

    #[test]
    fn test_arrow_schema_to_icefalldb() {
        let arrow = Arc::new(arrow::datatypes::Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let schema = arrow_schema_to_icefalldb(arrow);
        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "id");
        assert_eq!(schema.columns[0].r#type, "int64");
        assert!(!schema.columns[0].nullable);
    }
}
