//! Factory implementation backed by a `KeySource`.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::execution::context::SessionContext;
use datafusion_common::config::EncryptionFactoryOptions;
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::parquet_encryption::EncryptionFactory;
use object_store::path::Path as ObjectStorePath;
use parquet::encryption::decrypt::FileDecryptionProperties;

use crate::keys::KeySource;

/// Factory id used when registering on the `RuntimeEnv`. Reference it from
/// session config via `format.crypto.factory_id = "encrypted-parquet"`.
pub const FACTORY_ID: &str = "encrypted-parquet";

/// A turnkey wrapper around DataFusion's `ParquetFormat` that pre-registers
/// an [`EncryptionFactory`] on the session.
///
/// Construct via [`register_encryption_factory`].
pub struct EncryptedParquetFormat {
    /// The underlying `ParquetFormat`. Wrapped so callers can configure it
    /// (e.g. `.with_enable_page_index(true)`) exactly as they would without
    /// encryption.
    pub inner: ParquetFormat,
}

/// Register an [`EncryptionFactory`] on the session's `RuntimeEnv` that
/// resolves keys via the given [`KeySource`].
///
/// After this call, any `ParquetFormat`/`ParquetSource` configured with
/// `format.crypto.factory_id = "encrypted-parquet"` will decrypt files using
/// keys from `keys`.
///
/// Tables opt into decryption via:
///
/// ```sql
/// CREATE EXTERNAL TABLE my_tbl ... STORED AS PARQUET LOCATION '...'
/// OPTIONS ('format.crypto.factory_id' 'encrypted-parquet')
/// ```
///
/// or programmatically via `TableParquetOptions::crypto.configure_factory(...)`.
pub fn register_encryption_factory(ctx: &SessionContext, keys: Arc<dyn KeySource>) -> Result<()> {
    let factory = FactoryAdapter { keys };
    ctx.runtime_env()
        .register_parquet_encryption_factory(FACTORY_ID, Arc::new(factory));
    Ok(())
}

/// Internal adapter from `KeySource` to DataFusion's `EncryptionFactory`.
struct FactoryAdapter {
    keys: Arc<dyn KeySource>,
}

impl std::fmt::Debug for FactoryAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FactoryAdapter")
            .field("keys", &"<dyn KeySource>")
            .finish()
    }
}

#[async_trait]
impl EncryptionFactory for FactoryAdapter {
    async fn get_file_encryption_properties(
        &self,
        _options: &EncryptionFactoryOptions,
        _schema: &arrow::datatypes::SchemaRef,
        _file_path: &ObjectStorePath,
    ) -> Result<Option<std::sync::Arc<parquet::encryption::encrypt::FileEncryptionProperties>>>
    {
        // This crate only handles decryption; writers should construct their
        // own `FileEncryptionProperties` directly (via the parquet crate's
        // builder) or use IcefallDB's writer.
        Ok(None)
    }

    async fn get_file_decryption_properties(
        &self,
        options: &EncryptionFactoryOptions,
        _file_path: &ObjectStorePath,
    ) -> Result<Option<std::sync::Arc<FileDecryptionProperties>>> {
        let footer_kid = options
            .options
            .get("footer_key_id")
            .cloned()
            .unwrap_or_default();
        if footer_kid.is_empty() {
            return Err(DataFusionError::Configuration(
                "format.crypto.factory_options.footer_key_id is required when using \
                 the encrypted-parquet factory"
                    .into(),
            ));
        }
        let footer_key = self.keys.get(&footer_kid)?;

        let mut builder = FileDecryptionProperties::builder(footer_key);
        // Parse `column_key_ids` = "col1=kid1,col2=kid2".
        if let Some(col_spec) = options.options.get("column_key_ids") {
            for pair in col_spec.split(',') {
                let pair = pair.trim();
                if pair.is_empty() {
                    continue;
                }
                if let Some((col, kid)) = pair.split_once('=') {
                    let key = self.keys.get(kid.trim())?;
                    builder = builder.with_column_key(col.trim(), key);
                }
            }
        }
        if let Some(aad) = options.options.get("aad_prefix_b64") {
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(aad) {
                builder = builder.with_aad_prefix(bytes);
            }
        }
        let props = builder
            .build()
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(Some(props))
    }
}
