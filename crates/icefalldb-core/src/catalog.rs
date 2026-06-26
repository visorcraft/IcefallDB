use crate::metadata::{Manifest, Schema};
use crate::storage::Storage;
use crate::{is_not_found, Result};

/// A catalog provides access to a table's latest manifest and schema.
///
/// The catalog reads the manifest pointer (`_manifest.json`) from storage and
/// resolves the referenced manifest and schema files on `load` and `refresh`.
/// If the pointer is missing and `_schema.json` is also missing, the catalog is
/// empty. If `_schema.json` exists but `_manifest.json` does not, the table is
/// partially initialized and an error is returned.
pub struct Catalog<'a> {
    storage: &'a dyn Storage,
    table: String,
    latest_manifest: Option<Manifest>,
    latest_schema: Option<Schema>,
}

impl<'a> Catalog<'a> {
    /// Load a table catalog from storage.
    ///
    /// Reads the manifest pointer and resolves the latest manifest and schema.
    /// Returns an empty catalog if neither `_manifest.json` nor `_schema.json`
    /// exists. Returns an error if `_schema.json` exists but `_manifest.json`
    /// does not, because that indicates a partially initialized table.
    ///
    /// # Errors
    ///
    /// Returns an error if the pointer, manifest, or schema cannot be read or
    /// parsed, or if checksum or schema validation fails.
    pub async fn load(storage: &'a dyn Storage, table: &str) -> Result<Self> {
        let mut cat = Self {
            storage,
            table: table.to_string(),
            latest_manifest: None,
            latest_schema: None,
        };
        cat.refresh().await?;
        Ok(cat)
    }

    /// Refresh the catalog state from storage.
    ///
    /// Re-reads the manifest pointer and resolves the latest manifest and
    /// schema. If `_manifest.json` is missing but `_schema.json` exists, the
    /// table is partially initialized and an error is returned. If both pointers
    /// are missing, any cached state is retained. If any validation step fails,
    /// the previously cached state is retained.
    ///
    /// # Errors
    ///
    /// Returns an error if the pointer, manifest, or schema cannot be read or
    /// parsed, if `_schema.json` exists without `_manifest.json`, or if checksum
    /// or schema validation fails.
    pub async fn refresh(&mut self) -> Result<()> {
        let pointer_path = format!("{}/_manifest.json", self.table);
        let data = match self.storage.read(&pointer_path).await {
            Ok(data) => data,
            Err(crate::IcefallDBError::NotFound(_)) => {
                let schema_pointer_path = format!("{}/_schema.json", self.table);
                match self.storage.exists(&schema_pointer_path).await {
                    Ok(true) => {
                        return Err(crate::IcefallDBError::MissingManifestPointer {
                            path: pointer_path,
                        });
                    }
                    Ok(false) => return Ok(()),
                    Err(e) if is_not_found(&e) => return Ok(()),
                    Err(e) => return Err(e),
                }
            }
            Err(e) => return Err(e),
        };

        let pointer: serde_json::Value = serde_json::from_slice(&data)?;
        let latest = pointer
            .get("latest")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                crate::IcefallDBError::InvalidManifestPointer("missing or invalid 'latest'".into())
            })?;
        if latest == 0 {
            // A `latest` value of 0 indicates an empty table with no committed
            // manifests. This is the state left by table creation. Load the
            // schema if it has been written, but tolerate a missing schema
            // pointer (some tests create a manifest pointer in isolation).
            let schema_pointer_path = format!("{}/_schema.json", self.table);
            let schema_data = match self.storage.read(&schema_pointer_path).await {
                Ok(d) => d,
                Err(crate::IcefallDBError::NotFound(_)) => return Ok(()),
                Err(e) => return Err(e),
            };
            let schema_pointer: serde_json::Value = serde_json::from_slice(&schema_data)?;
            let schema_id = schema_pointer
                .get("latest")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| crate::IcefallDBError::InvalidSchemaPointer {
                    path: schema_pointer_path,
                })?;
            let schema_path = format!("{}/{}", self.table, Schema::filename(schema_id));
            let mut schema: Schema =
                serde_json::from_slice(&self.storage.read(&schema_path).await?)?;
            if !schema.has_field_ids() {
                schema.repair_field_ids();
            }
            self.latest_schema = Some(schema);
            return Ok(());
        }

        let manifest_path = format!("{}/{}", self.table, Manifest::filename(latest));
        let manifest: Manifest = serde_json::from_slice(&self.storage.read(&manifest_path).await?)?;
        if !manifest.verify_checksum()? {
            return Err(crate::IcefallDBError::ChecksumMismatch {
                path: manifest_path,
            });
        }
        if manifest.sequence != latest {
            return Err(crate::IcefallDBError::InvalidManifestPointer(format!(
                "sequence mismatch: pointer expects {}, manifest has {}",
                latest, manifest.sequence
            )));
        }

        let schema_path = format!("{}/{}", self.table, Schema::filename(manifest.schema_id));
        let schema_data = match self.storage.read(&schema_path).await {
            Ok(d) => d,
            Err(crate::IcefallDBError::NotFound(_)) => {
                return Err(crate::IcefallDBError::SchemaNotFound { path: schema_path });
            }
            Err(e) => return Err(e),
        };
        let mut schema: Schema = serde_json::from_slice(&schema_data)?;
        if schema.schema_id != manifest.schema_id {
            return Err(crate::IcefallDBError::SchemaMismatch {
                column: "schema_id".into(),
                expected: manifest.schema_id.to_string(),
                path: schema_path,
            });
        }

        // Backward compatibility: schemas written before field IDs were
        // introduced may have missing or zero IDs. Repair them in memory so
        // callers always see a valid schema; the IDs will be persisted on the
        // next write.
        if !schema.has_field_ids() {
            schema.repair_field_ids();
        }

        // Replay the mutation WAL (if any) on top of the checkpointed manifest so
        // readers see deferred mutations. A no-op (returns the manifest unchanged)
        // when no `_wal/` log exists — the default, non-WAL case.
        let manifest =
            crate::mutation_wal::live_manifest(self.storage, &self.table, manifest).await?;

        self.latest_manifest = Some(manifest);
        self.latest_schema = Some(schema);
        Ok(())
    }

    /// Returns the table name.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Returns a reference to the latest manifest, if any.
    pub fn latest_manifest(&self) -> Option<&Manifest> {
        self.latest_manifest.as_ref()
    }

    /// Returns a reference to the latest schema, if any.
    pub fn latest_schema(&self) -> Option<&Schema> {
        self.latest_schema.as_ref()
    }
}
