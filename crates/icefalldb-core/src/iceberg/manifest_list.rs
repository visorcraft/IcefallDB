use crate::{IcefallDBError, Result};
use apache_avro::types::Value as AvroValue;
use apache_avro::{Codec, Schema as AvroSchema, Writer as AvroWriter};

const MANIFEST_LIST_SCHEMA_JSON: &str = r#"{
    "type": "record",
    "name": "manifest_file",
    "fields": [
        {"name": "manifest_path", "type": "string"},
        {"name": "manifest_length", "type": "long"},
        {"name": "partition_spec_id", "type": "int"},
        {"name": "content", "type": "int"},
        {"name": "sequence_number", "type": "long"},
        {"name": "min_sequence_number", "type": "long"},
        {"name": "added_snapshot_id", "type": "long"},
        {"name": "added_files_count", "type": "int"},
        {"name": "existing_files_count", "type": "int"},
        {"name": "deleted_files_count", "type": "int"},
        {"name": "added_rows_count", "type": "long"},
        {"name": "existing_rows_count", "type": "long"},
        {"name": "deleted_rows_count", "type": "long"},
        {"name": "partitions", "type": ["null", {"type": "array", "items": {"type": "record", "name": "partition_summary", "fields": [{"name": "contains_null", "type": "boolean"}, {"name": "contains_nan", "type": ["null", "boolean"]}, {"name": "lower_bound", "type": ["null", "bytes"]}, {"name": "upper_bound", "type": ["null", "bytes"]}]}}]}
    ]
}"#;

/// Entry describing a manifest file in a manifest list.
#[derive(Debug, Clone)]
pub struct ManifestListEntry {
    pub manifest_path: String,
    pub manifest_length: i64,
    pub partition_spec_id: i32,
    pub content: i32,
    pub sequence_number: i64,
    pub min_sequence_number: i64,
    pub added_snapshot_id: i64,
    pub added_files_count: i32,
    pub existing_files_count: i32,
    pub deleted_files_count: i32,
    pub added_rows_count: i64,
    pub existing_rows_count: i64,
    pub deleted_rows_count: i64,
}

/// Write a manifest list Avro file with Deflate compression.
pub fn write_manifest_list(entries: &[ManifestListEntry]) -> Result<Vec<u8>> {
    let avro_schema = AvroSchema::parse_str(MANIFEST_LIST_SCHEMA_JSON)
        .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    let mut avro_writer =
        AvroWriter::with_codec(&avro_schema, Vec::new(), Codec::Deflate(Default::default()));

    for entry in entries {
        let record = build_manifest_list_entry_value(entry);
        avro_writer
            .append(record)
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    }

    avro_writer
        .into_inner()
        .map_err(|e| IcefallDBError::Other(Box::new(e)))
}

fn build_manifest_list_entry_value(entry: &ManifestListEntry) -> AvroValue {
    AvroValue::Record(vec![
        (
            "manifest_path".into(),
            AvroValue::String(entry.manifest_path.clone()),
        ),
        (
            "manifest_length".into(),
            AvroValue::Long(entry.manifest_length),
        ),
        (
            "partition_spec_id".into(),
            AvroValue::Int(entry.partition_spec_id),
        ),
        ("content".into(), AvroValue::Int(entry.content)),
        (
            "sequence_number".into(),
            AvroValue::Long(entry.sequence_number),
        ),
        (
            "min_sequence_number".into(),
            AvroValue::Long(entry.min_sequence_number),
        ),
        (
            "added_snapshot_id".into(),
            AvroValue::Long(entry.added_snapshot_id),
        ),
        (
            "added_files_count".into(),
            AvroValue::Int(entry.added_files_count),
        ),
        (
            "existing_files_count".into(),
            AvroValue::Int(entry.existing_files_count),
        ),
        (
            "deleted_files_count".into(),
            AvroValue::Int(entry.deleted_files_count),
        ),
        (
            "added_rows_count".into(),
            AvroValue::Long(entry.added_rows_count),
        ),
        (
            "existing_rows_count".into(),
            AvroValue::Long(entry.existing_rows_count),
        ),
        (
            "deleted_rows_count".into(),
            AvroValue::Long(entry.deleted_rows_count),
        ),
        (
            "partitions".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
    ])
}
