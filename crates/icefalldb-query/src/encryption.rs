//! DataFusion-side integration for Parquet Modular Encryption.
//!
//! This module is the *single* bridge between IcefallDB's
//! [`KeyProvider`](icefalldb_core::encryption::provider::KeyProvider) trait
//! and DataFusion's
//! [`EncryptionFactory`](datafusion_execution::parquet_encryption::EncryptionFactory)
//! trait. Wiring it together gives the engine, transparently:
//!
//! - **Reader:** `ParquetSource::with_encryption_factory(...)` calls back into
//!   our factory at scan time, which calls into our `KeyProvider`, which
//!   resolves keys from env / file / KMS.
//! - **Custom scan (`IcefallDBScanExec`):** consumes an
//!   `Option<Arc<FileDecryptionProperties>>` directly — built from the same
//!   provider at session-build time — and threads it into
//!   `ArrowReaderOptions::with_file_decryption_properties`.
//!
//! Both paths share the same [`build_decryption_properties_for_table`] helper,
//! which is the only function in this crate that constructs
//! `FileDecryptionProperties`.
//!
//! # Feature gate
//!
//! Everything in this module requires the `encryption` feature on both
//! `icefalldb-core` and `icefalldb-query`. When the feature is off, the module
//! is absent and the reader operates entirely in plaintext mode with zero
//! overhead.

#![cfg(feature = "encryption")]

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::common::config::EncryptionFactoryOptions;
use datafusion::common::DataFusionError;
use datafusion_execution::parquet_encryption::EncryptionFactory;
use icefalldb_core::encryption::{
    build_decryption_properties, table_aad_prefix, EncryptionKeySet, KeyIdentifier, KeyProvider,
};
use object_store::path::Path as ObjectStorePath;

use crate::Result;

/// Configuration carried on the DataFusion `SessionConfig` so SQL users can
/// opt into encryption without touching Rust code.
///
/// Exposed as `datafusion.common.extensions_options`-generated fields under
/// the `icefalldb.encryption.*` namespace.
#[derive(Debug, Default, Clone)]
pub struct IcefallDBdbEncryptionConfig {
    /// Key identifier for the footer key, e.g. `"events-v1"`. Resolved via
    /// the registered [`KeyProvider`].
    pub footer_key_id: String,
    /// Comma-separated `col=KeyId` pairs, e.g. `"ssn=ssn-v1,email=email-v1"`.
    /// Empty means footer-only encryption.
    pub column_key_ids: String,
    /// Optional table-name hint used to disambiguate AAD when multiple
    /// encrypted tables share a session.
    pub table_hint: String,
    /// Whether the Parquet footer is left unencrypted. Should match the
    /// writer's setting. Defaults to `true` (the IcefallDB writer default).
    pub plaintext_footer: bool,
}

impl IcefallDBdbEncryptionConfig {
    /// Parse `column_key_ids` into a `BTreeMap<column_name, KeyIdentifier>`.
    pub fn parse_column_keys(&self) -> std::collections::BTreeMap<String, KeyIdentifier> {
        let mut out = std::collections::BTreeMap::new();
        for pair in self.column_key_ids.split(',') {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            if let Some((col, kid)) = pair.split_once('=') {
                out.insert(col.trim().to_string(), KeyIdentifier::new(kid.trim()));
            }
        }
        out
    }
}

/// Look up a complete key set for a table from a `KeyProvider`, given the
/// table identity and the per-table config.
///
/// `table_hint` is used to scope the AAD prefix; `schema_id` is encoded in
/// the AAD so different schema revisions of the same table produce different
/// prefixes (preventing file-swap attacks across revisions).
pub async fn load_table_keys(
    provider: &dyn KeyProvider,
    footer_kid: &KeyIdentifier,
    column_kids: &std::collections::BTreeMap<String, KeyIdentifier>,
    aad: &[u8],
) -> Result<EncryptionKeySet> {
    let footer = provider.get(footer_kid, aad).await?;
    let mut cols = std::collections::BTreeMap::new();
    for (col, kid) in column_kids {
        cols.insert(col.clone(), provider.get(kid, aad).await?);
    }
    EncryptionKeySet::with_columns(footer, cols, aad.to_vec()).map_err(map_enc_err)
}

/// Build `FileDecryptionProperties` for a table from a key set. This is the
/// helper used by `IcefallDBScanExec` to thread decryption into the Parquet
/// reader.
pub fn build_decryption_properties_for_table(
    keys: &EncryptionKeySet,
) -> Result<Arc<parquet::encryption::decrypt::FileDecryptionProperties>> {
    build_decryption_properties(keys).map_err(map_enc_err)
}

fn map_enc_err(e: icefalldb_core::IcefallDBError) -> crate::QueryError {
    crate::QueryError::Core(e)
}

/// IcefallDB's implementation of DataFusion's `EncryptionFactory`.
///
/// One instance is registered on the `RuntimeEnv` per session, under the id
/// `"icefalldb"`. When DataFusion's `ParquetSource` needs to decrypt a file
/// (because `format.crypto.factory_id = "icefalldb"` was set on the session
/// config), it calls back into this factory.
///
/// The factory delegates to a [`KeyProvider`] for the actual key bytes. The
/// provider can be `EnvKeyProvider`, `FileKeyProvider`, `StaticKeyProvider`,
/// or a custom KMS-backed implementation.
pub struct IcefallDBdbEncryptionFactory {
    /// Shared key provider. Must be `Send + Sync` (the `KeyProvider` trait
    /// enforces this) because DataFusion partitions read concurrently.
    pub provider: Arc<dyn KeyProvider>,
}

impl std::fmt::Debug for IcefallDBdbEncryptionFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcefallDBdbEncryptionFactory")
            .field("provider", &"<dyn KeyProvider>")
            .finish()
    }
}

impl IcefallDBdbEncryptionFactory {
    pub fn new(provider: Arc<dyn KeyProvider>) -> Self {
        Self { provider }
    }

    /// The factory id under which this is registered on the `RuntimeEnv`.
    pub const FACTORY_ID: &'static str = "icefalldb";
}

#[async_trait]
impl EncryptionFactory for IcefallDBdbEncryptionFactory {
    async fn get_file_encryption_properties(
        &self,
        _options: &EncryptionFactoryOptions,
        _schema: &arrow::datatypes::SchemaRef,
        _file_path: &ObjectStorePath,
    ) -> std::result::Result<
        Option<Arc<parquet::encryption::encrypt::FileEncryptionProperties>>,
        DataFusionError,
    > {
        // Writer-side factory hooks are not used by IcefallDB today: the
        // writer builds `FileEncryptionProperties` directly via
        // `icefalldb_core::encryption::build_encryption_properties`. We return
        // `None` so DataFusion's `ParquetSink` (which we do not use) does not
        // silently produce encrypted files. The IcefallDB writer is the only
        // producer of encrypted files.
        Ok(None)
    }

    async fn get_file_decryption_properties(
        &self,
        options: &EncryptionFactoryOptions,
        file_path: &ObjectStorePath,
    ) -> std::result::Result<
        Option<Arc<parquet::encryption::decrypt::FileDecryptionProperties>>,
        DataFusionError,
    > {
        let cfg = parse_factory_options(options);

        let column_kids = cfg.parse_column_keys();
        let footer_kid = KeyIdentifier::new(cfg.footer_key_id);
        let schema_hint = cfg.table_hint;
        let aad = if schema_hint.is_empty() {
            // No hint → derive a weak AAD from the file path so different files
            // still get different AADs.
            let path_str = file_path.to_string();
            table_aad_prefix(&path_str, 0)
        } else {
            // Use the configured table name; assume schema_id 1 for now. A
            // future revision will thread the actual schema_id through.
            table_aad_prefix(&schema_hint, 1)
        };

        let keys = load_table_keys(self.provider.as_ref(), &footer_kid, &column_kids, &aad)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let props = build_decryption_properties_for_table(&keys)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(Some(props))
    }
}

/// Parse the factory options into our typed config struct.
fn parse_factory_options(options: &EncryptionFactoryOptions) -> IcefallDBdbEncryptionConfig {
    // `EncryptionFactoryOptions.options` is a free-form `HashMap<String, String>`
    // populated from `format.crypto.factory_options.<key>` session config
    // values.
    let opts = &options.options;
    let footer_key_id = opts
        .get("footer_key_id")
        .or_else(|| opts.get("icefalldb.encryption.footer_key_id"))
        .cloned()
        .unwrap_or_default();
    let column_key_ids = opts
        .get("column_key_ids")
        .or_else(|| opts.get("icefalldb.encryption.column_key_ids"))
        .cloned()
        .unwrap_or_default();
    let table_hint = opts
        .get("table_hint")
        .or_else(|| opts.get("icefalldb.encryption.table_hint"))
        .cloned()
        .unwrap_or_default();
    let plaintext_footer_str = opts
        .get("plaintext_footer")
        .or_else(|| opts.get("icefalldb.encryption.plaintext_footer"))
        .map(|s| s.as_str())
        .unwrap_or("true");
    let plaintext_footer = !matches!(
        plaintext_footer_str.to_lowercase().as_str(),
        "false" | "0" | "no" | "off"
    );

    IcefallDBdbEncryptionConfig {
        footer_key_id,
        column_key_ids,
        table_hint,
        plaintext_footer,
    }
}
