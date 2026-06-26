//! A `'static` catalog wrapper for IcefallDB tables.
//!
//! `icefalldb_core::catalog::Catalog` holds a borrowed `&'a dyn Storage`, which makes
//! it awkward to own from a DataFusion `TableProvider`. This wrapper owns an
//! `Arc<dyn Storage>` and reconstructs a short-lived core catalog on each load
//! call so the provider can keep it around.

use std::sync::Arc;

use icefalldb_core::metadata::{Manifest, Schema};
use icefalldb_core::storage::Storage;
use icefalldb_core::{IcefallDBError, Reader, ScanPlan};

use crate::Result;

/// Owns the storage handle and table name for a IcefallDB table.
#[derive(Clone)]
pub struct IcefallDBCatalog {
    storage: Arc<dyn Storage>,
    table: String,
}

impl IcefallDBCatalog {
    /// Create a new catalog wrapper for `table` backed by `storage`.
    pub fn new(storage: Arc<dyn Storage>, table: impl Into<String>) -> Self {
        Self {
            storage,
            table: table.into(),
        }
    }

    /// Load the latest snapshot, returning the scan plan and IcefallDB schema.
    ///
    /// This always reads the current manifest pointer from storage, so repeated
    /// calls observe newly committed snapshots.
    pub async fn load_snapshot(&self) -> Result<(ScanPlan, Schema)> {
        let (plan, schema, _manifest) = self.load_snapshot_internal(false).await?;
        Ok((plan, schema))
    }

    /// Load the latest snapshot, tolerating missing or corrupt `.meta` sidecars.
    ///
    /// Missing sidecars produce [`PlannedRowGroup`]s with `fallback` set to
    /// `true`. The caller should fall back to reading Parquet footer metadata
    /// when needed.
    pub async fn load_snapshot_allow_missing_meta(&self) -> Result<(ScanPlan, Schema)> {
        let (plan, schema, _manifest) = self.load_snapshot_internal(true).await?;
        Ok((plan, schema))
    }

    /// Load the latest snapshot, tolerating an empty table (no committed manifests).
    pub async fn load_snapshot_allow_empty(&self) -> Result<(ScanPlan, Schema)> {
        let (plan, schema, _manifest) = self.load_snapshot_allow_empty_with_manifest().await?;
        Ok((plan, schema))
    }

    /// Load the latest snapshot (tolerating an empty table) and also return the
    /// pinned `Manifest` that was read.
    ///
    /// The returned `Option<Manifest>` is `None` only for empty tables that have
    /// no committed manifests yet.  All three values come from the **same**
    /// manifest-pointer read so a concurrent commit cannot split the scan plan
    /// and the manifest's `index_generations` map.
    pub async fn load_snapshot_allow_empty_with_manifest(
        &self,
    ) -> Result<(ScanPlan, Schema, Option<Manifest>)> {
        match self.load_snapshot_internal(true).await {
            Ok((plan, schema, manifest)) => Ok((plan, schema, Some(manifest))),
            Err(crate::QueryError::Core(IcefallDBError::EmptyTable(_))) => {
                let catalog =
                    icefalldb_core::catalog::Catalog::load(self.storage.as_ref(), &self.table)
                        .await
                        .map_err(crate::QueryError::Core)?;
                let schema = catalog.latest_schema().cloned().ok_or_else(|| {
                    crate::QueryError::Core(IcefallDBError::SchemaNotFound {
                        path: format!("{}/_schema.json", self.table),
                    })
                })?;
                Ok((
                    ScanPlan {
                        table: self.table.clone(),
                        schema: schema.clone(),
                        row_groups: vec![],
                    },
                    schema,
                    None,
                ))
            }
            Err(e) => Err(e),
        }
    }

    /// Load only the IcefallDB schema and the pinned manifest, **without** building
    /// the per-fragment scan plan.
    ///
    /// This is the O(1) (in fragment count) open path: it reads `_schema.json`
    /// and the `_manifest.json` pointer but skips `Reader::scan()`'s per-fragment
    /// `RowGroupMeta` reconstruction. The scan plan is built lazily on first
    /// query via `IcefallDBTableProvider::current_scan_plan`. Returns `None` for the
    /// manifest only when the table has no committed manifests yet (empty table).
    pub async fn load_schema_and_manifest_allow_empty(&self) -> Result<(Schema, Option<Manifest>)> {
        let catalog = icefalldb_core::catalog::Catalog::load(self.storage.as_ref(), &self.table)
            .await
            .map_err(crate::QueryError::Core)?;
        let schema = catalog.latest_schema().cloned().ok_or_else(|| {
            crate::QueryError::Core(IcefallDBError::SchemaNotFound {
                path: format!("{}/_schema.json", self.table),
            })
        })?;
        let manifest = catalog.latest_manifest().cloned();
        Ok((schema, manifest))
    }

    async fn load_snapshot_internal(
        &self,
        allow_missing_meta: bool,
    ) -> Result<(ScanPlan, Schema, Manifest)> {
        let reader = Reader::new(self.storage.as_ref(), &self.table).await?;
        let scan_plan = if allow_missing_meta {
            reader.scan_allow_missing_meta().await?
        } else {
            reader.scan().await?
        };
        let schema = reader
            .catalog()
            .latest_schema()
            .cloned()
            .ok_or_else(|| icefalldb_core::IcefallDBError::EmptyTable(self.table.clone()))?;
        // SAFETY: `Reader::new` returns `EmptyTable` when `latest_manifest` is
        // `None`, so by the time we reach this line the manifest is always `Some`.
        let manifest = reader
            .catalog()
            .latest_manifest()
            .cloned()
            .ok_or_else(|| icefalldb_core::IcefallDBError::EmptyTable(self.table.clone()))?;
        Ok((scan_plan, schema, manifest))
    }

    /// Return the table name.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Return a reference to the underlying storage.
    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }
}

impl std::fmt::Debug for IcefallDBCatalog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcefallDBCatalog")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}
