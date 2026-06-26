use super::data_file::DataFile;
use crate::{IcefallDBError, Result};
use apache_avro::types::Value as AvroValue;
use apache_avro::{Codec, Schema as AvroSchema, Writer as AvroWriter};
use std::collections::HashMap;

const MANIFEST_SCHEMA_JSON: &str = r#"{
    "type": "record",
    "name": "manifest_entry",
    "fields": [
        {"name": "status", "type": "int"},
        {"name": "snapshot_id", "type": ["null", "long"]},
        {"name": "sequence_number", "type": ["null", "long"]},
        {"name": "file_sequence_number", "type": ["null", "long"]},
        {"name": "data_file", "type": {
            "type": "record",
            "name": "r2",
            "fields": [
                {"name": "content", "type": "int"},
                {"name": "file_path", "type": "string"},
                {"name": "file_format", "type": "string"},
                {"name": "partition", "type": {"type": "record", "name": "partition", "fields": []}},
                {"name": "record_count", "type": "long"},
                {"name": "file_size_in_bytes", "type": "long"},
                {"name": "column_sizes", "type": ["null", {"type": "array", "items": {"type": "record", "name": "k117_v117", "fields": [{"name": "key", "type": "int"}, {"name": "value", "type": "long"}]}}]},
                {"name": "value_counts", "type": ["null", {"type": "array", "items": "k117_v117"}]},
                {"name": "null_value_counts", "type": ["null", {"type": "array", "items": "k117_v117"}]},
                {"name": "nan_value_counts", "type": ["null", {"type": "array", "items": "k117_v117"}]},
                {"name": "lower_bounds", "type": ["null", {"type": "array", "items": {"type": "record", "name": "k117_v118", "fields": [{"name": "key", "type": "int"}, {"name": "value", "type": "bytes"}]}}]},
                {"name": "upper_bounds", "type": ["null", {"type": "array", "items": "k117_v118"}]},
                {"name": "key_metadata", "type": ["null", "bytes"]},
                {"name": "split_offsets", "type": ["null", {"type": "array", "items": "long"}]},
                {"name": "equality_ids", "type": ["null", {"type": "array", "items": "int"}]},
                {"name": "sort_order_id", "type": ["null", "int"]}
            ]
        }}
    ]
}"#;

/// Write a manifest Avro file containing the provided data files.
///
/// The manifest is written with Deflate compression. Only unpartitioned tables
/// are supported in v1; the partition record is always empty.
pub fn write_manifest(snapshot_id: i64, data_files: &[DataFile]) -> Result<Vec<u8>> {
    let avro_schema = AvroSchema::parse_str(MANIFEST_SCHEMA_JSON)
        .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    let mut avro_writer =
        AvroWriter::with_codec(&avro_schema, Vec::new(), Codec::Deflate(Default::default()));

    for data_file in data_files {
        let record = build_manifest_entry_value(snapshot_id, data_file)?;
        avro_writer
            .append(record)
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    }

    avro_writer
        .into_inner()
        .map_err(|e| IcefallDBError::Other(Box::new(e)))
}

fn build_manifest_entry_value(snapshot_id: i64, data_file: &DataFile) -> Result<AvroValue> {
    Ok(AvroValue::Record(vec![
        ("status".into(), AvroValue::Int(1)),
        (
            "snapshot_id".into(),
            AvroValue::Union(1, Box::new(AvroValue::Long(snapshot_id))),
        ),
        (
            "sequence_number".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
        (
            "file_sequence_number".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
        ("data_file".into(), build_data_file_value(data_file)?),
    ]))
}

fn build_data_file_value(data_file: &DataFile) -> Result<AvroValue> {
    Ok(AvroValue::Record(vec![
        ("content".into(), AvroValue::Int(data_file.content)),
        (
            "file_path".into(),
            AvroValue::String(data_file.file_path.clone()),
        ),
        (
            "file_format".into(),
            AvroValue::String(data_file.file_format.clone()),
        ),
        ("partition".into(), AvroValue::Record(vec![])),
        (
            "record_count".into(),
            AvroValue::Long(data_file.record_count),
        ),
        (
            "file_size_in_bytes".into(),
            AvroValue::Long(data_file.file_size_in_bytes),
        ),
        (
            "column_sizes".into(),
            AvroValue::Union(
                1,
                Box::new(AvroValue::Array(map_to_avro_array(&data_file.column_sizes))),
            ),
        ),
        (
            "value_counts".into(),
            AvroValue::Union(
                1,
                Box::new(AvroValue::Array(map_to_avro_array(&data_file.value_counts))),
            ),
        ),
        (
            "null_value_counts".into(),
            AvroValue::Union(
                1,
                Box::new(AvroValue::Array(map_to_avro_array(
                    &data_file.null_value_counts,
                ))),
            ),
        ),
        (
            "nan_value_counts".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
        (
            "lower_bounds".into(),
            AvroValue::Union(
                1,
                Box::new(AvroValue::Array(bytes_map_to_avro_array(
                    &data_file.lower_bounds,
                ))),
            ),
        ),
        (
            "upper_bounds".into(),
            AvroValue::Union(
                1,
                Box::new(AvroValue::Array(bytes_map_to_avro_array(
                    &data_file.upper_bounds,
                ))),
            ),
        ),
        (
            "key_metadata".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
        (
            "split_offsets".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
        (
            "equality_ids".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
        (
            "sort_order_id".into(),
            AvroValue::Union(0, Box::new(AvroValue::Null)),
        ),
    ]))
}

fn map_to_avro_array(map: &HashMap<i32, i64>) -> Vec<AvroValue> {
    let mut entries: Vec<(i32, i64)> = map.iter().map(|(&k, &v)| (k, v)).collect();
    entries.sort_by_key(|(k, _)| *k);
    entries
        .into_iter()
        .map(|(k, v)| {
            AvroValue::Record(vec![
                ("key".into(), AvroValue::Int(k)),
                ("value".into(), AvroValue::Long(v)),
            ])
        })
        .collect()
}

fn bytes_map_to_avro_array(map: &HashMap<i32, Vec<u8>>) -> Vec<AvroValue> {
    let mut entries: Vec<(i32, Vec<u8>)> = map.iter().map(|(&k, v)| (k, v.clone())).collect();
    entries.sort_by_key(|(k, _)| *k);
    entries
        .into_iter()
        .map(|(k, v)| {
            AvroValue::Record(vec![
                ("key".into(), AvroValue::Int(k)),
                ("value".into(), AvroValue::Bytes(v)),
            ])
        })
        .collect()
}
