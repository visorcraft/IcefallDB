use crate::metadata::Schema as IcefallDBSchema;
use serde_json::{json, Value};

/// Build an Iceberg sort order from a IcefallDB schema.
///
/// Returns `None` if the schema declares no sort order.
pub fn to_sort_order(schema: &IcefallDBSchema) -> Option<Value> {
    let sort = schema.sort.as_ref()?;
    if sort.is_empty() {
        return None;
    }

    let mut fields = Vec::with_capacity(sort.len());
    for col_name in sort {
        let source_id = schema
            .columns
            .iter()
            .find(|c| &c.name == col_name)
            .map(|c| c.field_id)
            .unwrap_or(0)
            .max(0);
        fields.push(json!({
            "source-id": source_id,
            "transform": "identity",
            "direction": "asc",
            "null-order": "nulls-first",
        }));
    }

    Some(json!({
        "order-id": 1,
        "fields": fields,
    }))
}
