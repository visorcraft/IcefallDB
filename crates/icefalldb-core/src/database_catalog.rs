use crate::metadata::Schema;
use crate::storage::Storage;
use crate::{is_not_found, IcefallDBError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

const CATALOG_PATH: &str = "_catalog.json";
const CATALOG_LOCK_PATH: &str = "_catalog.lock";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DatabaseCatalogData {
    pub tables: BTreeMap<String, TableEntry>,
    pub indexes: BTreeMap<String, IndexEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEntry {
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub table: String,
    pub column: String,
    pub index_type: String,
    /// Whether this index enforces a uniqueness constraint (one live row per
    /// distinct key).  Defaults to `false` for backward compatibility with
    /// catalog entries written before the unique-index feature was added.
    #[serde(default)]
    pub unique: bool,
}

/// A database-wide catalog manager.
#[derive(Clone)]
pub struct DatabaseCatalog {
    pub storage: Arc<dyn Storage>,
}

impl DatabaseCatalog {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }

    /// Load existing catalog or return an empty one if it does not exist.
    pub async fn load(&self) -> Result<DatabaseCatalogData> {
        match self.storage.read(CATALOG_PATH).await {
            Ok(data) => Ok(serde_json::from_slice(&data)?),
            Err(e) if is_not_found(&e) => Ok(DatabaseCatalogData::default()),
            Err(e) => Err(e),
        }
    }

    /// Acquire the global catalog lock.
    pub async fn lock(&self, timeout: Duration) -> Result<Box<dyn crate::storage::LockGuard>> {
        self.storage
            .lock_exclusive(CATALOG_LOCK_PATH, timeout)
            .await
    }

    /// Atomically write the catalog.
    pub async fn save(&self, catalog: &DatabaseCatalogData) -> Result<()> {
        let data = serde_json::to_vec_pretty(catalog)?;
        let tmp = format!("{}.tmp", CATALOG_PATH);
        self.storage.write(&tmp, &data).await?;
        self.storage.sync_data(&tmp).await?;
        self.storage.rename(&tmp, CATALOG_PATH).await?;
        self.storage.sync_root().await?;
        Ok(())
    }
}

use crate::writer::Writer;
use chrono::Utc;

impl DatabaseCatalog {
    pub async fn create_table(
        &self,
        _guard: &CatalogLockGuard,
        name: &str,
        schema: &Schema,
    ) -> Result<()> {
        validate_table_name(name)?;
        let mut catalog = self.load().await?;
        if catalog.tables.contains_key(name) {
            return Err(IcefallDBError::TableAlreadyExists(name.to_string()));
        }

        // Write the actual table files while holding the catalog lock.
        Writer::create(Arc::clone(&self.storage), name, schema.clone()).await?;

        catalog.tables.insert(
            name.to_string(),
            TableEntry {
                created_at: Utc::now().to_rfc3339(),
            },
        );
        self.save(&catalog).await?;
        Ok(())
    }

    /// Register a table that already exists on disk into the catalog.
    ///
    /// Unlike [`create_table`], this method does **not** create any table files — it
    /// only inserts the catalog entry.  If the table is already registered the call
    /// is a no-op (idempotent), which makes it safe to call after [`Writer::create`]
    /// even on a re-`create` or a partially-completed prior run.
    pub async fn register_existing_table(
        &self,
        _guard: &CatalogLockGuard,
        name: &str,
    ) -> Result<()> {
        validate_table_name(name)?;
        let mut catalog = self.load().await?;
        if catalog.tables.contains_key(name) {
            return Ok(()); // already registered — idempotent
        }
        catalog.tables.insert(
            name.to_string(),
            TableEntry {
                created_at: Utc::now().to_rfc3339(),
            },
        );
        self.save(&catalog).await?;
        Ok(())
    }

    pub async fn drop_table(&self, _guard: &CatalogLockGuard, name: &str) -> Result<()> {
        validate_table_name(name)?;
        let mut catalog = self.load().await?;
        if catalog.tables.remove(name).is_none() {
            return Err(IcefallDBError::TableNotFound(name.to_string()));
        }
        self.save(&catalog).await?;
        // Note: table data files are intentionally left in place so the drop is
        // reversible by doctor. A future `DELETE TABLE FILES` command can remove them.
        Ok(())
    }

    pub async fn list_tables(&self) -> Result<Vec<String>> {
        let catalog = self.load().await?;
        Ok(catalog.tables.keys().cloned().collect())
    }

    pub async fn table_exists(&self, name: &str) -> Result<bool> {
        let catalog = self.load().await?;
        Ok(catalog.tables.contains_key(name))
    }
}

/// Opaque token proving the caller holds `_catalog.lock`.
#[allow(dead_code)]
pub struct CatalogLockGuard(Box<dyn crate::storage::LockGuard>);

impl DatabaseCatalog {
    pub async fn acquire_lock(&self, timeout: Duration) -> Result<CatalogLockGuard> {
        Ok(CatalogLockGuard(self.lock(timeout).await?))
    }
}

fn validate_table_name(name: &str) -> Result<()> {
    crate::writer::validate_table(name)?;
    if name.starts_with('_') {
        return Err(IcefallDBError::InvalidPath(
            "table names may not start with '_'".into(),
        ));
    }
    Ok(())
}

impl DatabaseCatalog {
    pub async fn create_index_definition(
        &self,
        _guard: &CatalogLockGuard,
        name: &str,
        table: &str,
        column: &str,
        index_type: &str,
    ) -> Result<()> {
        self.create_index_definition_with_options(_guard, name, table, column, index_type, false)
            .await
    }

    /// Like [`create_index_definition`] but also records whether the index is
    /// unique.
    pub async fn create_index_definition_with_options(
        &self,
        _guard: &CatalogLockGuard,
        name: &str,
        table: &str,
        column: &str,
        index_type: &str,
        unique: bool,
    ) -> Result<()> {
        let mut catalog = self.load().await?;
        if catalog.indexes.contains_key(name) {
            return Err(IcefallDBError::TableAlreadyExists(name.to_string()));
        }
        catalog.indexes.insert(
            name.to_string(),
            IndexEntry {
                table: table.to_string(),
                column: column.to_string(),
                index_type: index_type.to_string(),
                unique,
            },
        );
        self.save(&catalog).await?;
        Ok(())
    }

    pub async fn drop_index_definition(&self, _guard: &CatalogLockGuard, name: &str) -> Result<()> {
        let mut catalog = self.load().await?;
        if catalog.indexes.remove(name).is_none() {
            return Err(IcefallDBError::TableNotFound(name.to_string()));
        }
        self.save(&catalog).await?;
        Ok(())
    }

    pub async fn list_index_definitions(&self) -> Result<Vec<(String, IndexEntry)>> {
        let catalog = self.load().await?;
        Ok(catalog.indexes.into_iter().collect())
    }
}
