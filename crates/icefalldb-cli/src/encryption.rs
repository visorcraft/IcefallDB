//! CLI helpers for Parquet Modular Encryption.
//!
//! Keys are resolved from a `--key-file` (JSON `{"keys": {"id": "hex"}}`) or,
//! by default, from `ICEFALLDB_KEY_*` environment variables. Key identifiers
//! follow the writer convention: the footer key is `<table>-v<schema_id>` and a
//! per-column key is `<footer-id>:<column>`. Callers therefore only ever supply
//! key *bytes*; the identifiers are derived here and on the read side from the
//! `_encryption.json` marker, so both sides agree without extra bookkeeping.
#![cfg(feature = "encryption")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use icefalldb_core::encryption::{
    build_decryption_properties, table_aad_prefix, EncryptionKeySet, EncryptionWriteConfig,
    EnvKeyProvider, FileKeyProvider, KeyIdentifier, KeyProvider, SchemaEncryptionMarker,
};
use icefalldb_core::storage::Storage;
use icefalldb_query::{IcefallDBTableProvider, ProviderConfig};

/// How the user asked to encrypt a table on `import`.
#[derive(Debug, Default, Clone)]
pub struct EncryptSpec {
    /// `--encrypt`: encrypt the whole table with the footer key.
    pub whole_table: bool,
    /// `--encrypt-column`: encrypt these columns with their own keys.
    pub columns: Vec<String>,
    /// `--encrypt-footer`: also encrypt the Parquet footer (default: plaintext).
    pub encrypt_footer: bool,
    /// `--key-file`: JSON key file; absent means `ICEFALLDB_KEY_*` env vars.
    pub key_file: Option<PathBuf>,
}

/// Build a key provider: a JSON key file if supplied, else `ICEFALLDB_KEY_*`.
pub fn provider_from(key_file: Option<&Path>) -> Arc<dyn KeyProvider> {
    match key_file {
        Some(p) => Arc::new(FileKeyProvider::new(p)) as Arc<dyn KeyProvider>,
        None => Arc::new(EnvKeyProvider) as Arc<dyn KeyProvider>,
    }
}

/// Read a table's `_encryption.json` marker, if present.
pub async fn read_marker(
    storage: &Arc<dyn Storage>,
    table: &str,
) -> Result<Option<SchemaEncryptionMarker>> {
    let path = format!("{table}/_encryption.json");
    if !storage.exists(&path).await? {
        return Ok(None);
    }
    let bytes = storage.read(&path).await?;
    let marker: SchemaEncryptionMarker =
        serde_json::from_slice(&bytes).with_context(|| format!("parsing {path}"))?;
    marker
        .validate()
        .with_context(|| format!("validating {path}"))?;
    Ok(Some(marker))
}

fn footer_key_id(table: &str, schema_id: u64) -> String {
    format!("{table}-v{schema_id}")
}

/// Resolve keys and build a write config to CREATE (or append to) an encrypted
/// table. The footer key is read from the provider under `<table>-v<schema_id>`;
/// each per-column key under `<footer-id>:<column>`.
pub async fn write_config(
    spec: &EncryptSpec,
    table: &str,
    schema_id: u64,
) -> Result<EncryptionWriteConfig> {
    let provider = provider_from(spec.key_file.as_deref());
    let aad = table_aad_prefix(table, schema_id);
    let footer_id = KeyIdentifier::new(footer_key_id(table, schema_id));
    let footer = provider
        .get(&footer_id, &aad)
        .await
        .with_context(|| format!("resolving footer key '{footer_id}'"))?;

    let keyset = if spec.columns.is_empty() {
        EncryptionKeySet::footer_only(footer.as_slice().to_vec(), aad.clone())
            .map_err(|e| anyhow!("invalid footer key: {e}"))?
    } else {
        let mut cols = BTreeMap::new();
        for col in &spec.columns {
            let kid = KeyIdentifier::new(format!("{footer_id}:{col}"));
            let bytes = provider
                .get(&kid, &aad)
                .await
                .with_context(|| format!("resolving column key '{kid}'"))?;
            cols.insert(col.clone(), bytes.as_slice().to_vec());
        }
        EncryptionKeySet::with_columns(footer.as_slice().to_vec(), cols, aad.clone())
            .map_err(|e| anyhow!("invalid key set: {e}"))?
    };

    let mut cfg = EncryptionWriteConfig::new(keyset).with_plaintext_footer(!spec.encrypt_footer);
    // When the user named specific columns *without* `--encrypt`, encrypt only
    // those columns and leave the rest plaintext (column-level access control).
    // With `--encrypt` too, every column is encrypted (named ones with their
    // own key), so leave `encrypted_columns` empty.
    if !spec.columns.is_empty() && !spec.whole_table {
        cfg = cfg.with_encrypted_columns(spec.columns.iter().cloned());
    }
    Ok(cfg)
}

fn marker_to_key_ids(marker: &SchemaEncryptionMarker) -> BTreeMap<String, KeyIdentifier> {
    marker
        .column_key_ids
        .iter()
        .map(|(col, kid)| (col.clone(), KeyIdentifier::new(kid.clone())))
        .collect()
}

/// Open an encrypted table provider for reading, resolving its key identifiers
/// from the on-disk marker and the supplied key provider.
pub async fn open_encrypted_provider(
    storage: &Arc<dyn Storage>,
    table: &str,
    provider_config: ProviderConfig,
    key_provider: Arc<dyn KeyProvider>,
    marker: &SchemaEncryptionMarker,
) -> Result<IcefallDBTableProvider> {
    let footer_id = KeyIdentifier::new(marker.footer_key_id.clone());
    let column_key_ids = marker_to_key_ids(marker);
    IcefallDBTableProvider::new_encrypted(
        Arc::clone(storage),
        table,
        provider_config,
        key_provider,
        footer_id,
        column_key_ids,
    )
    .await
    .map_err(|e| anyhow!("opening encrypted table '{table}': {e}"))
}

/// Open an encrypted table provider pinned to a historical snapshot.
pub async fn open_encrypted_provider_at_snapshot(
    storage: &Arc<dyn Storage>,
    table: &str,
    provider_config: ProviderConfig,
    key_provider: Arc<dyn KeyProvider>,
    marker: &SchemaEncryptionMarker,
    sequence: u64,
) -> Result<IcefallDBTableProvider> {
    let footer_id = KeyIdentifier::new(marker.footer_key_id.clone());
    let column_key_ids = marker_to_key_ids(marker);
    IcefallDBTableProvider::new_encrypted_at_snapshot(
        Arc::clone(storage),
        table,
        provider_config,
        key_provider,
        footer_id,
        column_key_ids,
        sequence,
    )
    .await
    .map_err(|e| anyhow!("opening encrypted table '{table}' at snapshot {sequence}: {e}"))
}

/// Build Parquet decryption properties for a projected index scan.
///
/// For whole-table encryption, the footer key is enough. For per-column
/// encryption, callers pass only the encrypted columns they need so unrelated
/// column keys are not required.
pub async fn decryption_properties_for_columns<I, S>(
    key_provider: Arc<dyn KeyProvider>,
    marker: &SchemaEncryptionMarker,
    encrypted_columns: I,
) -> Result<Arc<parquet::encryption::decrypt::FileDecryptionProperties>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let aad = match marker.aad_prefix.as_deref() {
        Some(b64) => base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| anyhow!("invalid AAD prefix in _encryption.json: {e}"))?,
        None => Vec::new(),
    };

    let footer_id = KeyIdentifier::new(marker.footer_key_id.clone());
    let footer = key_provider
        .get(&footer_id, &aad)
        .await
        .with_context(|| format!("resolving footer key '{footer_id}'"))?;

    let mut columns = BTreeMap::new();
    for col in encrypted_columns {
        let col = col.as_ref();
        if let Some(kid) = marker.column_key_ids.get(col) {
            let key_id = KeyIdentifier::new(kid.clone());
            let key = key_provider
                .get(&key_id, &aad)
                .await
                .with_context(|| format!("resolving column key '{key_id}'"))?;
            columns.insert(col.to_string(), key);
        }
    }

    let key_set = if columns.is_empty() {
        EncryptionKeySet::footer_only_zeroizing(footer, aad)
            .map_err(|e| anyhow!("invalid footer key: {e}"))?
    } else {
        EncryptionKeySet::with_columns_zeroizing(footer, columns, aad)
            .map_err(|e| anyhow!("invalid key set: {e}"))?
    };

    build_decryption_properties(&key_set)
        .map_err(|e| anyhow!("building decryption properties: {e}"))
}
