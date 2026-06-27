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
    Zeroizing,
};
use object_store::path::Path as ObjectStorePath;

use crate::Result;

/// Configuration carried on the DataFusion `SessionConfig` so SQL users can
/// opt into encryption without touching Rust code.
///
/// Exposed as `datafusion.common.extensions_options`-generated fields under
/// the `icefalldb.encryption.*` namespace.
#[derive(Debug, Clone)]
pub struct IcefallDBEncryptionConfig {
    /// Key identifier for the footer key, e.g. `"events-v1"`. Resolved via
    /// the registered [`KeyProvider`].
    pub footer_key_id: String,
    /// Comma-separated `col=KeyId` pairs, e.g. `"ssn=ssn-v1,email=email-v1"`.
    /// Empty means footer-only encryption.
    pub column_key_ids: String,
    /// Optional table-name hint used to derive the AAD prefix (with
    /// [`Self::schema_id`]) via [`table_aad_prefix`]. When unset, no AAD is
    /// supplied to the reader (see [`Self::derive_aad`]).
    pub table_hint: String,
    /// Schema id folded into the derived AAD prefix when `table_hint` is set.
    /// Defaults to `1` to match the writer's first schema revision.
    pub schema_id: u64,
    /// Explicit AAD prefix override (base64). When set, used verbatim and
    /// `table_hint`/`schema_id` are ignored. Read from the table's
    /// `_encryption.json` marker when the caller knows it.
    pub aad_prefix_b64: Option<String>,
}

impl Default for IcefallDBEncryptionConfig {
    fn default() -> Self {
        Self {
            footer_key_id: String::new(),
            column_key_ids: String::new(),
            table_hint: String::new(),
            schema_id: 1,
            aad_prefix_b64: None,
        }
    }
}

impl IcefallDBEncryptionConfig {
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

    /// Derive the AAD prefix to supply for decryption, in priority order:
    ///
    /// 1. [`Self::aad_prefix_b64`](struct@Self::aad_prefix_b64) — explicit base64
    ///    override.
    /// 2. [`Self::table_hint`] + [`Self::schema_id`] — recomputed via
    ///    [`table_aad_prefix`], matching the writer's derivation.
    /// 3. neither — empty, i.e. *no* AAD is supplied to the Parquet reader.
    ///
    /// Returning empty (case 3) is correct, not a fallback-of-last-resort: the
    /// IcefallDB writer always stores the AAD prefix in the file footer
    /// (`store_aad_prefix = true` default), so the reader uses the file's own
    /// AAD and decryption succeeds. We only forgo cross-table file-swap
    /// *verification*, which is impossible anyway when the reader does not know
    /// the table identity. We deliberately never fabricate an AAD from the file
    /// path — a made-up prefix would mismatch the stored one and fail every
    /// read's GCM authentication.
    pub fn derive_aad(&self) -> std::result::Result<Vec<u8>, DataFusionError> {
        use base64::Engine;
        if let Some(b64) = self.aad_prefix_b64.as_deref() {
            let b64 = b64.trim();
            if !b64.is_empty() {
                return base64::engine::general_purpose::STANDARD
                    .decode(b64)
                    .map_err(|e| {
                        DataFusionError::Configuration(format!("invalid aad_prefix_b64: {e}"))
                    });
            }
        }
        if !self.table_hint.is_empty() {
            return Ok(table_aad_prefix(&self.table_hint, self.schema_id));
        }
        Ok(Vec::new())
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
        cols.insert(
            col.clone(),
            Zeroizing::new(provider.get(kid, aad).await?.as_slice().to_vec()),
        );
    }
    EncryptionKeySet::with_columns_zeroizing(
        Zeroizing::new(footer.as_slice().to_vec()),
        cols,
        aad.to_vec(),
    )
    .map_err(map_enc_err)
}

/// Build `FileDecryptionProperties` for a table from a key set. This is the
/// helper used by `IcefallDBScanExec` to thread decryption into the Parquet
/// reader.
pub fn build_decryption_properties_for_table(
    keys: &EncryptionKeySet,
) -> Result<Arc<parquet::encryption::decrypt::FileDecryptionProperties>> {
    build_decryption_properties(keys).map_err(map_enc_err)
}

/// Lazy resolver for an encrypted table's decryption properties.
///
/// Column keys are fetched from the [`KeyProvider`] only when the column is
/// actually referenced by a query's projection or filter. This lets
/// plaintext-column queries on a per-column-encrypted table succeed without
/// providing keys for columns that are not read.
#[derive(Clone)]
pub struct EncryptionKeyResolver {
    provider: Arc<dyn KeyProvider>,
    footer_key_id: KeyIdentifier,
    column_key_ids: std::collections::BTreeMap<String, KeyIdentifier>,
    aad: Vec<u8>,
    plaintext_footer: bool,
}

impl std::fmt::Debug for EncryptionKeyResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionKeyResolver")
            .field("footer_key_id", &self.footer_key_id)
            .field(
                "column_key_ids",
                &self.column_key_ids.keys().collect::<Vec<_>>(),
            )
            .field("plaintext_footer", &self.plaintext_footer)
            .finish_non_exhaustive()
    }
}

impl EncryptionKeyResolver {
    /// Create a resolver for a table's encryption keys.
    pub fn new(
        provider: Arc<dyn KeyProvider>,
        footer_key_id: KeyIdentifier,
        column_key_ids: std::collections::BTreeMap<String, KeyIdentifier>,
        aad: Vec<u8>,
        plaintext_footer: bool,
    ) -> Self {
        Self {
            provider,
            footer_key_id,
            column_key_ids,
            aad,
            plaintext_footer,
        }
    }

    /// Whether the table uses a plaintext Parquet footer.
    pub fn plaintext_footer(&self) -> bool {
        self.plaintext_footer
    }

    /// Resolve the footer key and only the per-column keys for columns named
    /// in `needed_columns`, then build the corresponding
    /// `FileDecryptionProperties`.
    ///
    /// Columns that are not encrypted (no entry in `column_key_ids`) require no
    /// key. Missing keys for encrypted columns that *are* needed surface as
    /// [`IcefallDBError::EncryptionKeyNotFound`].
    ///
    /// When the table has a plaintext footer and no encrypted column is
    /// referenced, the footer key is not required: a dummy key is used and
    /// footer-signature verification is disabled. This lets plaintext-column
    /// queries (including `COUNT(*)`) succeed even if the caller only has
    /// access to the encrypted columns they actually read.
    pub async fn resolve_for_columns(
        &self,
        needed_columns: &std::collections::HashSet<String>,
    ) -> Result<Arc<parquet::encryption::decrypt::FileDecryptionProperties>> {
        // Does this query need a key beyond plaintext columns?
        //
        // - Per-column encryption (`column_key_ids` non-empty): only the listed
        //   encrypted columns require a key; plaintext columns (and `COUNT(*)`,
        //   which needs no column) do not.
        // - Uniform / whole-table encryption (`column_key_ids` empty): the footer
        //   key encrypts every data column, so any query touching a real column
        //   needs it. Without this, a uniform table read of a data column with a
        //   missing footer key wrongly took the no-key dummy path and failed with
        //   an opaque Parquet error instead of a clear missing-key error.
        //   (`COUNT(*)` — an empty needed set — is still answered from the
        //   plaintext-footer metadata without a key, matching prior behavior.)
        let needs_any_encrypted_column = if self.column_key_ids.is_empty() {
            !needed_columns.is_empty()
        } else {
            needed_columns
                .iter()
                .any(|name| self.column_key_ids.contains_key(name))
        };

        // Footer key. When the table has a plaintext footer and no encrypted
        // column is needed, a missing footer key is non-fatal: use a dummy key
        // and skip signature verification. This branch is detected from the
        // not-found error (not the key bytes), so a real all-zero footer key
        // still verifies its signature. No secret material is involved here.
        let footer_bytes = match self.provider.get(&self.footer_key_id, &self.aad).await {
            Ok(footer) => Zeroizing::new(footer.as_slice().to_vec()),
            Err(e) => {
                let is_key_not_found =
                    matches!(e, icefalldb_core::IcefallDBError::EncryptionKeyNotFound(_));
                if self.plaintext_footer && !needs_any_encrypted_column && is_key_not_found {
                    let mut builder =
                        parquet::encryption::decrypt::FileDecryptionProperties::builder(vec![
                            0u8;
                            16
                        ])
                        .disable_footer_signature_verification();
                    if !self.aad.is_empty() {
                        builder = builder.with_aad_prefix(self.aad.clone());
                    }
                    return builder.build().map_err(|e| {
                        map_enc_err(icefalldb_core::IcefallDBError::Encryption(e.to_string()))
                    });
                }
                return Err(map_enc_err(e));
            }
        };

        // Resolve only the per-column keys actually needed by this query. Keys
        // are held in `Zeroizing` buffers from the moment they are copied out of
        // the provider so the material stays wiped on drop (including across a
        // panic between resolution and key-set construction).
        let mut column_keys: std::collections::BTreeMap<String, Zeroizing<Vec<u8>>> =
            std::collections::BTreeMap::new();
        for name in needed_columns {
            if let Some(kid) = self.column_key_ids.get(name) {
                let key = self
                    .provider
                    .get(kid, &self.aad)
                    .await
                    .map_err(map_enc_err)?;
                column_keys.insert(name.clone(), Zeroizing::new(key.as_slice().to_vec()));
            }
        }

        // Build through `EncryptionKeySet` so all key material is held in
        // `Zeroizing` buffers (wiped on drop) and the shared builder is reused,
        // restoring the project-wide invariant the bespoke builder bypassed.
        let key_set =
            EncryptionKeySet::with_columns_zeroizing(footer_bytes, column_keys, self.aad.clone())
                .map_err(map_enc_err)?;
        build_decryption_properties(&key_set).map_err(map_enc_err)
    }
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
pub struct IcefallDBEncryptionFactory {
    /// Shared key provider. Must be `Send + Sync` (the `KeyProvider` trait
    /// enforces this) because DataFusion partitions read concurrently.
    pub provider: Arc<dyn KeyProvider>,
}

impl std::fmt::Debug for IcefallDBEncryptionFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcefallDBEncryptionFactory")
            .field("provider", &"<dyn KeyProvider>")
            .finish()
    }
}

impl IcefallDBEncryptionFactory {
    pub fn new(provider: Arc<dyn KeyProvider>) -> Self {
        Self { provider }
    }

    /// The factory id under which this is registered on the `RuntimeEnv`.
    pub const FACTORY_ID: &'static str = "icefalldb";
}

#[async_trait]
impl EncryptionFactory for IcefallDBEncryptionFactory {
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
        _file_path: &ObjectStorePath,
    ) -> std::result::Result<
        Option<Arc<parquet::encryption::decrypt::FileDecryptionProperties>>,
        DataFusionError,
    > {
        let cfg = parse_factory_options(options);

        let aad = cfg.derive_aad()?;
        let column_kids = cfg.parse_column_keys();
        let footer_kid = KeyIdentifier::new(cfg.footer_key_id);

        let keys = load_table_keys(self.provider.as_ref(), &footer_kid, &column_kids, &aad)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        let props = build_decryption_properties_for_table(&keys)
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

        Ok(Some(props))
    }
}

/// Parse the factory options into our typed config struct.
fn parse_factory_options(options: &EncryptionFactoryOptions) -> IcefallDBEncryptionConfig {
    // `EncryptionFactoryOptions.options` is a free-form `HashMap<String, String>`
    // populated from `format.crypto.factory_options.<key>` session config
    // values. Each key is accepted either bare (`footer_key_id`) or under the
    // `icefalldb.encryption.*` namespace.
    let opts = &options.options;
    let get = |key: &str| -> Option<String> {
        opts.get(key)
            .or_else(|| opts.get(&format!("icefalldb.encryption.{key}")))
            .cloned()
    };

    let schema_id = get("schema_id")
        .map(|s| s.trim().parse::<u64>().unwrap_or(1))
        .unwrap_or(1);

    IcefallDBEncryptionConfig {
        footer_key_id: get("footer_key_id").unwrap_or_default(),
        column_key_ids: get("column_key_ids").unwrap_or_default(),
        table_hint: get("table_hint").unwrap_or_default(),
        schema_id,
        aad_prefix_b64: get("aad_prefix_b64"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icefalldb_core::encryption::StaticKeyProvider;

    #[test]
    fn derive_aad_empty_when_no_identity() {
        // No table_hint and no explicit AAD → empty. The reader then uses the
        // file's stored AAD. This must NOT be a fabricated path-derived value,
        // which would mismatch the stored prefix and fail authentication.
        assert!(IcefallDBEncryptionConfig::default()
            .derive_aad()
            .unwrap()
            .is_empty());
    }

    #[test]
    fn derive_aad_from_table_hint() {
        let c = IcefallDBEncryptionConfig {
            table_hint: "events".into(),
            ..Default::default()
        };
        assert_eq!(c.derive_aad().unwrap(), table_aad_prefix("events", 1));
    }

    #[test]
    fn derive_aad_from_table_hint_and_schema_id() {
        let c = IcefallDBEncryptionConfig {
            table_hint: "events".into(),
            schema_id: 7,
            ..Default::default()
        };
        assert_eq!(c.derive_aad().unwrap(), table_aad_prefix("events", 7));
    }

    #[test]
    fn derive_aad_explicit_override_wins() {
        let c = IcefallDBEncryptionConfig {
            table_hint: "events".into(),
            aad_prefix_b64: Some("Y3VzdG9tLWFhZA==".into()), // b"custom-aad"
            ..Default::default()
        };
        assert_eq!(c.derive_aad().unwrap(), b"custom-aad");
    }

    #[test]
    fn derive_aad_blank_override_falls_through() {
        // A whitespace-only explicit override is treated as unset so the
        // table_hint path still applies.
        let c = IcefallDBEncryptionConfig {
            table_hint: "events".into(),
            aad_prefix_b64: Some("  ".into()),
            ..Default::default()
        };
        assert_eq!(c.derive_aad().unwrap(), table_aad_prefix("events", 1));
    }

    #[test]
    fn derive_aad_rejects_bad_base64() {
        let c = IcefallDBEncryptionConfig {
            aad_prefix_b64: Some("!!!not-base64!!!".into()),
            ..Default::default()
        };
        let err = c.derive_aad().unwrap_err();
        assert!(err.to_string().contains("aad_prefix_b64"));
    }

    #[tokio::test]
    async fn zero_byte_footer_key_still_verifies_signature() {
        // A real footer key that happens to be 16 zero bytes must not be
        // confused with the dummy key used when the caller lacks the footer key.
        // Footer-signature verification must remain enabled.
        let keys = std::collections::HashMap::from([(KeyIdentifier::new("footer"), vec![0u8; 16])]);
        let provider = Arc::new(StaticKeyProvider::new(keys).unwrap());
        let resolver = EncryptionKeyResolver::new(
            provider,
            KeyIdentifier::new("footer"),
            std::collections::BTreeMap::new(),
            Vec::new(),
            true, // plaintext_footer
        );

        let props = resolver
            .resolve_for_columns(&std::collections::HashSet::new())
            .await
            .unwrap();

        assert!(props.check_plaintext_footer_integrity());
    }

    #[tokio::test]
    async fn missing_footer_key_disables_verification_for_plaintext_query() {
        // When the footer key is missing, no encrypted column is referenced, and
        // the table has a plaintext footer, the resolver uses a dummy key and
        // disables footer verification so the read can proceed.
        let provider = Arc::new(StaticKeyProvider::new(std::iter::empty()).unwrap());
        let mut column_keys = std::collections::BTreeMap::new();
        column_keys.insert("secret".to_string(), KeyIdentifier::new("secret-key"));
        let resolver = EncryptionKeyResolver::new(
            provider,
            KeyIdentifier::new("footer"),
            column_keys,
            Vec::new(),
            true, // plaintext_footer
        );

        let needed = std::collections::HashSet::from(["plain".to_string()]);
        let props = resolver.resolve_for_columns(&needed).await.unwrap();

        assert!(!props.check_plaintext_footer_integrity());
    }
}
