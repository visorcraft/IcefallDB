use crate::metadata::Schema as IcefallDBSchema;
use serde_json::{json, Value};

/// Build an Iceberg partition spec from a IcefallDB schema.
///
/// Identity transforms are used for all partition columns. Returns `None` if
/// the schema is unpartitioned.
pub fn to_partition_spec(schema: &IcefallDBSchema) -> Option<Value> {
    let partition_by = schema.partition_by.as_ref()?;
    if partition_by.is_empty() {
        return None;
    }

    let mut fields = Vec::with_capacity(partition_by.len());
    let mut last_partition_id: i32 = 0;
    for (idx, col_name) in partition_by.iter().enumerate() {
        let source_id = schema
            .columns
            .iter()
            .find(|c| &c.name == col_name)
            .map(|c| c.field_id)
            .unwrap_or(0)
            .max(0);
        let field_id = (idx + 1) as i32;
        last_partition_id = last_partition_id.max(field_id);
        fields.push(json!({
            "source-id": source_id,
            "field-id": field_id,
            "name": col_name,
            "transform": "identity",
        }));
    }

    Some(json!({
        "spec-id": 0,
        "fields": fields,
    }))
}

/// Return the last partition field ID, or 0 if unpartitioned.
pub fn last_partition_id(schema: &IcefallDBSchema) -> i32 {
    let Some(spec) = to_partition_spec(schema) else {
        return 0;
    };
    spec.get("fields")
        .and_then(|f| f.as_array())
        .map(|fields| {
            fields
                .iter()
                .filter_map(|f| f.get("field-id").and_then(|v| v.as_i64()))
                .max()
                .unwrap_or(0) as i32
        })
        .unwrap_or(0)
}
