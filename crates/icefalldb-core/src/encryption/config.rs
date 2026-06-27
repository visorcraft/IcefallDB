//! Writer-side encryption configuration.
//!
//! [`EncryptionWriteConfig`] is consumed by [`crate::Writer`] when the
//! `encryption` feature is enabled. It binds an [`EncryptionKeySet`] to a set
//! of write-time options (plaintext footer, AAD storage, encrypted-column
//! selection).

use std::collections::{BTreeMap, BTreeSet};

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::encryption::EncryptionKeySet;
use crate::error::{IcefallDBError, Result};

/// Per-table write-time encryption configuration.
///
/// Stored on the writer; applied to every row group that the writer produces.
#[derive(Clone)]
pub struct EncryptionWriteConfig {
    /// Resolved key set: footer key + per-column keys + AAD prefix.
    pub keys: EncryptionKeySet,
    /// Leave the Parquet footer unencrypted so sidecar stats and page-index
    /// reads continue to work. Default `true`. Set to `false` only when the
    /// footer column names themselves are sensitive.
    pub plaintext_footer: bool,
    /// Write the AAD prefix into the file footer so readers do not need to
    /// provide it to read the data. Default `true`. Readers may still provide
    /// it for verification.
    pub store_aad_prefix: bool,
    /// Subset of columns to encrypt with their per-column keys. Columns not
    /// listed here are encrypted with the footer key (when their column key is
    /// absent from the key set) or left unencrypted (when their column key is
    /// present and the column is not in this set — this is the "column-level
    /// access control" mode).
    ///
    /// If empty, every column named in `keys.columns` is encrypted with its
    /// own key, and every column *not* named in `keys.columns` is encrypted
    /// with the footer key (this is the parquet crate's default behavior when
    /// only a footer key is provided).
    pub encrypted_columns: BTreeSet<String>,
}

impl std::fmt::Debug for EncryptionWriteConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptionWriteConfig")
            .field("keys", &"**REDACTED**")
            .field("plaintext_footer", &self.plaintext_footer)
            .field("store_aad_prefix", &self.store_aad_prefix)
            .field("encrypted_columns", &self.encrypted_columns)
            .finish()
    }
}

impl EncryptionWriteConfig {
    /// Construct with sensible defaults: plaintext footer + stored AAD prefix.
    pub fn new(keys: EncryptionKeySet) -> Self {
        Self {
            keys,
            plaintext_footer: true,
            store_aad_prefix: true,
            encrypted_columns: BTreeSet::new(),
        }
    }

    /// Builder-style: toggle the plaintext footer.
    pub fn with_plaintext_footer(mut self, enabled: bool) -> Self {
        self.plaintext_footer = enabled;
        self
    }

    /// Builder-style: toggle storing the AAD prefix.
    pub fn with_store_aad_prefix(mut self, enabled: bool) -> Self {
        self.store_aad_prefix = enabled;
        self
    }

    /// Builder-style: restrict encryption to a subset of columns.
    pub fn with_encrypted_columns<I>(mut self, cols: I) -> Self
    where
        I: IntoIterator<Item = String>,
    {
        self.encrypted_columns = cols.into_iter().collect();
        self
    }
}

/// Persisted marker that lives in `_schema.json` so the reader knows a table
/// is encrypted and which key identifiers to use. Contains *no key material*.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct SchemaEncryptionMarker {
    /// Algorithm identifier. Currently always
    /// `parquet-modular-encryption-v1`. The reader rejects unknown algorithms
    /// rather than silently reading the data.
    pub algorithm: String,
    /// Key identifier for the footer key. Resolved via a `KeyProvider`.
    pub footer_key_id: String,
    /// Map of column name → key identifier, for per-column encrypted columns.
    /// Empty when only footer-key encryption is used.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub column_key_ids: std::collections::BTreeMap<String, String>,
    /// Whether the Parquet footer is left unencrypted.
    pub plaintext_footer: bool,
    /// Base64-encoded AAD prefix. Bound to the table identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aad_prefix: Option<String>,
}

impl SchemaEncryptionMarker {
    /// Marker algorithm identifier for Parquet Modular Encryption v1.
    pub const ALGORITHM: &'static str = "parquet-modular-encryption-v1";

    /// Validate that this marker describes an encryption scheme the reader
    /// supports.
    ///
    /// Readers must call this after deserialising a marker so an unknown
    /// algorithm identifier is rejected up front — rather than being silently
    /// treated as decryptable and producing a confusing GCM auth-tag failure
    /// deep inside the Parquet decoder (or, worse, a plaintext read if the
    /// file is not actually encrypted). The writer-side reopen path already
    /// compares the algorithm field directly; this is the reader-side gate.
    pub fn validate(&self) -> Result<()> {
        if self.algorithm != Self::ALGORITHM {
            return Err(IcefallDBError::Encryption(format!(
                "unsupported encryption algorithm '{}'; expected '{}'. The reader \
                 refuses to decrypt a file with an unknown algorithm rather than \
                 guessing — rotate via an explicit migration once support lands.",
                self.algorithm,
                Self::ALGORITHM
            )));
        }
        Ok(())
    }

    /// Build a marker from a writer's config and a footer key identifier.
    pub fn for_write_config(
        footer_key_id: impl Into<String>,
        config: &EncryptionWriteConfig,
    ) -> Self {
        let footer_id_owned: String = footer_key_id.into();
        let footer_id_for_struct = footer_id_owned.clone();
        let column_key_ids = config
            .keys
            .column_pairs()
            .map(|(name, _)| {
                (
                    name.clone(),
                    footer_key_id_for_column(&footer_id_owned, name),
                )
            })
            .collect();
        let aad_prefix = if config.keys.aad_prefix.is_empty() {
            None
        } else {
            Some(base64::engine::general_purpose::STANDARD.encode(&config.keys.aad_prefix))
        };
        Self {
            algorithm: Self::ALGORITHM.to_string(),
            footer_key_id: footer_id_for_struct,
            column_key_ids,
            plaintext_footer: config.plaintext_footer,
            aad_prefix,
        }
    }
}

/// Convention for column key identifiers: `<footer-id>:<column-name>`.
/// This is not required by PME; it's just a stable convention so a single
/// `KeyProvider` lookup table can address every column key.
pub fn footer_key_id_for_column(footer_id: &str, column: &str) -> String {
    format!("{footer_id}:{column}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_keys() -> EncryptionKeySet {
        let mut cols = std::collections::BTreeMap::new();
        cols.insert("ssn".to_string(), b"0123456789abcdef".to_vec());
        EncryptionKeySet::with_columns(b"0123456789abcdef".to_vec(), cols, b"aad".to_vec()).unwrap()
    }

    #[test]
    fn debug_redacts_keys() {
        let cfg = EncryptionWriteConfig::new(sample_keys());
        let s = format!("{:?}", cfg);
        assert!(s.contains("REDACTED"));
        assert!(!s.contains("30313233"));
    }

    #[test]
    fn marker_round_trip_json() {
        let cfg = EncryptionWriteConfig::new(sample_keys());
        let marker = SchemaEncryptionMarker::for_write_config("events-v1", &cfg);
        let json = serde_json::to_string(&marker).unwrap();
        let parsed: SchemaEncryptionMarker = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.algorithm, SchemaEncryptionMarker::ALGORITHM);
        assert_eq!(parsed.footer_key_id, "events-v1");
        assert_eq!(
            parsed.column_key_ids.get("ssn").map(|s| s.as_str()),
            Some("events-v1:ssn")
        );
    }

    #[test]
    fn marker_rejects_unknown_algorithm() {
        let json = r#"{
            "algorithm": "rot13",
            "footer_key_id": "x",
            "plaintext_footer": true
        }"#;
        let marker: SchemaEncryptionMarker = serde_json::from_str(json).unwrap();
        let err = marker.validate().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported encryption algorithm"),
            "expected an unsupported-algorithm error, got: {msg}"
        );
        // Positive control: the canonical marker validates.
        let cfg = EncryptionWriteConfig::new(sample_keys());
        let good = SchemaEncryptionMarker::for_write_config("events-v1", &cfg);
        good.validate().expect("canonical marker validates");
    }
}
