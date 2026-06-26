use crate::metadata::Schema as IcefallDBSchema;
use crate::Result;
use serde_json::{json, Value};

/// Build an Iceberg v2 table metadata JSON object.
#[allow(clippy::too_many_arguments)]
pub fn build_table_metadata(
    icefalldb_schema: &IcefallDBSchema,
    table_uuid: &str,
    location: &str,
    last_sequence_number: i64,
    current_snapshot_id: i64,
    snapshots: Vec<Value>,
    partition_spec: Option<Value>,
    last_partition_id: i32,
    sort_order: Option<Value>,
    schema_field_ids_start: i32,
) -> Result<Value> {
    let iceberg_schema = super::schema::to_iceberg_schema(icefalldb_schema)?;
    let schemas = json!([iceberg_schema]);

    let partition_specs = if let Some(spec) = partition_spec {
        json!([spec])
    } else {
        json!([{
            "spec-id": 0,
            "fields": [],
        }])
    };

    let sort_orders = if let Some(order) = sort_order {
        json!([order])
    } else {
        json!([{
            "order-id": 0,
            "fields": [],
        }])
    };

    Ok(json!({
        "format-version": 2,
        "table-uuid": table_uuid,
        "location": location,
        "last-sequence-number": last_sequence_number,
        "last-updated-ms": chrono::Utc::now().timestamp_millis(),
        "last-column-id": schema_field_ids_start - 1,
        "schemas": schemas,
        "current-schema-id": icefalldb_schema.schema_id as i64,
        "partition-specs": partition_specs,
        "default-spec-id": 0,
        "last-partition-id": last_partition_id,
        "sort-orders": sort_orders,
        "default-sort-order-id": 0,
        "properties": {},
        "current-snapshot-id": current_snapshot_id,
        "snapshots": snapshots,
        "snapshot-log": [],
        "metadata-log": [],
        "refs": {
            "main": {
                "snapshot-id": current_snapshot_id,
                "type": "branch"
            }
        },
    }))
}

/// Build a snapshot JSON object for Iceberg metadata.
pub fn build_snapshot(
    snapshot_id: i64,
    sequence_number: i64,
    schema_id: u64,
    manifest_list_path: &str,
    added_files: i32,
    added_rows: i64,
) -> Value {
    json!({
        "snapshot-id": snapshot_id,
        "sequence-number": sequence_number,
        "timestamp-ms": chrono::Utc::now().timestamp_millis(),
        "summary": {
            "operation": "append",
            "added-files-size": "0",
            "added-data-files": added_files.to_string(),
            "added-records": added_rows.to_string(),
        },
        "manifest-list": manifest_list_path,
        "schema-id": schema_id as i64,
    })
}
