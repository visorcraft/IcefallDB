use crate::catalog::Catalog;
use crate::metadata::{Manifest, RowGroupMeta};
use crate::storage::Storage;
use crate::{IcefallDBError, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::data_file::DataFile;
use super::manifest::write_manifest;
use super::manifest_list::{write_manifest_list, ManifestListEntry};
use super::metadata::{build_snapshot, build_table_metadata};

/// Export a IcefallDB table snapshot to Iceberg-compatible metadata.
///
/// The output directory will contain:
/// - `metadata/<seq>-<uuid>.metadata.json`
/// - `metadata/<seq>-<uuid>-manifests.avro`
/// - `metadata/<seq>-<uuid>-<name>.avro` manifest files
/// - `metadata/version-hint.text`
///
/// The existing Parquet files are referenced by absolute `file://` URI.
pub async fn export_table(
    storage: &dyn Storage,
    table: &str,
    output: &Path,
    snapshot: Option<u64>,
    _table_root_uri: &str,
) -> Result<PathBuf> {
    let catalog = Catalog::load(storage, table).await?;
    let schema = catalog
        .latest_schema()
        .ok_or_else(|| IcefallDBError::InvalidSchema {
            reason: "table has no schema".into(),
            path: table.into(),
        })?;

    // Reject unsupported types early.
    for col in &schema.columns {
        if super::schema::arrow_type_to_iceberg(
            &crate::metadata::schema::icefalldb_type_to_arrow(&col.r#type).ok_or_else(|| {
                IcefallDBError::TypeNotSupported(format!(
                    "unsupported IcefallDB type: {}",
                    col.r#type
                ))
            })?,
        )
        .is_none()
        {
            return Err(IcefallDBError::TypeNotSupported(format!(
                "column '{}' has unsupported type '{}' for Iceberg export",
                col.name, col.r#type
            )));
        }
    }

    // Only unpartitioned tables are supported in v1.
    if schema
        .partition_by
        .as_ref()
        .map(|v| !v.is_empty())
        .unwrap_or(false)
    {
        return Err(IcefallDBError::TypeNotSupported(
            "partitioned tables are not supported by the v1 Iceberg export bridge".into(),
        ));
    }

    let manifest = match snapshot {
        Some(seq) => {
            let manifest_path = format!("{}/{}", table, Manifest::filename(seq));
            let manifest_data = storage.read(&manifest_path).await?;
            let manifest: Manifest = serde_json::from_slice(&manifest_data)?;
            if !manifest.verify_checksum()? {
                return Err(IcefallDBError::ChecksumMismatch {
                    path: manifest_path,
                });
            }
            if manifest.sequence != seq {
                return Err(IcefallDBError::InvalidManifestPointer(format!(
                    "sequence mismatch: requested {}, manifest has {}",
                    seq, manifest.sequence
                )));
            }
            if manifest.schema_id != schema.schema_id {
                return Err(IcefallDBError::SchemaMismatch {
                    column: "schema_id".into(),
                    expected: schema.schema_id.to_string(),
                    path: manifest_path,
                });
            }
            manifest
        }
        None => catalog.latest_manifest().cloned().ok_or_else(|| {
            IcefallDBError::ManifestNotFound(format!("no snapshots available for table {}", table))
        })?,
    };

    let sequence = manifest.sequence as i64;
    let snapshot_id = sequence;

    let field_ids: HashMap<String, i32> = schema
        .columns
        .iter()
        .map(|c| (c.name.clone(), c.field_id.max(1)))
        .collect();
    let max_field_id = schema
        .columns
        .iter()
        .map(|c| c.field_id.max(1))
        .max()
        .unwrap_or(0)
        + 1;

    // Load metadata for each row group and build DataFile entries.
    let mut data_files = Vec::with_capacity(manifest.row_groups.len());
    let mut total_rows = 0i64;
    for entry in &manifest.row_groups {
        let meta_path = format!("{}/{}", table, entry.meta);
        let meta_data = storage.read(&meta_path).await?;
        let meta: RowGroupMeta = serde_json::from_slice(&meta_data)?;
        let data_file =
            DataFile::from_icefalldb(storage, table, &entry.data, &meta, &field_ids).await?;
        total_rows += data_file.record_count;
        data_files.push(data_file);
    }

    // Write manifest Avro.
    let manifest_filename = format!("{}-manifest.avro", snapshot_id);
    let manifest_path = output.join("metadata").join(&manifest_filename);
    let manifest_rel_path = format!("metadata/{}", manifest_filename);

    tokio::fs::create_dir_all(manifest_path.parent().unwrap())
        .await
        .map_err(IcefallDBError::Io)?;
    let manifest_bytes = write_manifest(snapshot_id, &data_files)?;
    tokio::fs::write(&manifest_path, &manifest_bytes)
        .await
        .map_err(IcefallDBError::Io)?;

    // Write manifest list Avro.
    let manifest_list_filename = format!("{}-manifests.avro", snapshot_id);
    let manifest_list_path = output.join("metadata").join(&manifest_list_filename);
    let manifest_list_rel_path = format!("metadata/{}", manifest_list_filename);
    let manifest_list_entry = ManifestListEntry {
        manifest_path: manifest_rel_path,
        manifest_length: manifest_bytes.len() as i64,
        partition_spec_id: 0,
        content: 0,
        sequence_number: sequence,
        min_sequence_number: sequence,
        added_snapshot_id: snapshot_id,
        added_files_count: data_files.len() as i32,
        existing_files_count: 0,
        deleted_files_count: 0,
        added_rows_count: total_rows,
        existing_rows_count: 0,
        deleted_rows_count: 0,
    };
    let manifest_list_bytes = write_manifest_list(&[manifest_list_entry])?;
    tokio::fs::write(&manifest_list_path, &manifest_list_bytes)
        .await
        .map_err(IcefallDBError::Io)?;

    // Write metadata.json.
    let table_uuid = uuid::Uuid::new_v4().to_string();
    let output_abs = std::path::absolute(output).map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    let location = format!("file://{}", output_abs.to_string_lossy().replace('\\', "/"));
    let snapshot_json = build_snapshot(
        snapshot_id,
        sequence,
        schema.schema_id,
        &manifest_list_rel_path,
        data_files.len() as i32,
        total_rows,
    );
    let metadata = build_table_metadata(
        schema,
        &table_uuid,
        &location,
        sequence,
        snapshot_id,
        vec![snapshot_json],
        None,
        0,
        None,
        max_field_id,
    )?;
    let metadata_filename = format!("{}-{}.metadata.json", sequence, table_uuid);
    let metadata_path = output.join("metadata").join(&metadata_filename);
    tokio::fs::write(
        &metadata_path,
        serde_json::to_vec_pretty(&metadata).map_err(IcefallDBError::Serialization)?,
    )
    .await
    .map_err(IcefallDBError::Io)?;

    // Write version-hint.text.
    tokio::fs::write(
        output.join("metadata").join("version-hint.text"),
        metadata_filename.as_bytes(),
    )
    .await
    .map_err(IcefallDBError::Io)?;

    Ok(metadata_path)
}
