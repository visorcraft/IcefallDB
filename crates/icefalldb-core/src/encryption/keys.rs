//! Encryption keys: identifiers, lengths, and a complete key set for one table.
//!
//! Keys are 16, 24, or 32 bytes (AES-128/192/256). The same length is required
//! across the footer and all per-column keys in a table, but different tables
//! may use different lengths.
//!
//! `EncryptionKeySet` owns its key bytes in a `Zeroizing<Vec<u8>>` so they are
//! wiped from memory on drop. Cloning is explicit (no `Clone` for the holding
//! struct by default — callers take references) to minimize the number of
//! copies.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

use crate::error::{IcefallDBError, Result};
use base64::Engine;

/// A short identifier for a key, stored in the Parquet footer as `key_metadata`
/// so the reader can look it up via a [`crate::encryption::provider::KeyProvider`].
///
/// The identifier is *not* secret; it is metadata. It must be unique within a
/// single IcefallDB deployment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KeyIdentifier(pub String);

impl KeyIdentifier {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for KeyIdentifier {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for KeyIdentifier {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl fmt::Display for KeyIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for KeyIdentifier {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Allowed AES key lengths for Parquet modular encryption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLength {
    /// 16 bytes.
    Aes128,
    /// 24 bytes.
    Aes192,
    /// 32 bytes.
    Aes256,
}

impl KeyLength {
    pub const fn byte_len(self) -> usize {
        match self {
            KeyLength::Aes128 => 16,
            KeyLength::Aes192 => 24,
            KeyLength::Aes256 => 32,
        }
    }

    /// Detect the key length from a byte slice, returning an error for any
    /// length other than 16/24/32.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        match bytes.len() {
            16 => Ok(Self::Aes128),
            24 => Ok(Self::Aes192),
            32 => Ok(Self::Aes256),
            n => Err(IcefallDBError::Encryption(format!(
                "invalid AES key length {n}; expected 16, 24, or 32 bytes"
            ))),
        }
    }
}

/// Validate that a byte slice is a legal AES key. Returns the length on success.
pub fn validate_key(bytes: &[u8]) -> Result<KeyLength> {
    KeyLength::from_bytes(bytes)
}

/// A complete set of encryption keys for a single table: one footer key plus
/// optional per-column keys.
///
/// Key bytes are held in [`Zeroizing`] buffers so they are wiped on drop.
/// The struct is intentionally `Clone` (used by the writer and the reader in
/// the same process) but `Debug` redacts key material.
#[derive(Clone)]
pub struct EncryptionKeySet {
    /// Key used to encrypt the Parquet footer (and any column data not listed
    /// in `columns`, when `encrypted_columns` is empty in the write config).
    pub footer: Zeroizing<Vec<u8>>,
    /// Per-column encryption keys, keyed by column name. Columns listed here
    /// are encrypted with their own key; columns not listed are encrypted with
    /// the footer key.
    pub columns: BTreeMap<String, Zeroizing<Vec<u8>>>,
    /// AAD prefix bound to the table identity. Not secret.
    pub aad_prefix: Vec<u8>,
}

impl EncryptionKeySet {
    /// Build a footer-only key set (no per-column keys).
    pub fn footer_only(footer: Vec<u8>, aad_prefix: Vec<u8>) -> Result<Self> {
        validate_key(&footer)?;
        Ok(Self {
            footer: Zeroizing::new(footer),
            columns: BTreeMap::new(),
            aad_prefix,
        })
    }

    /// Build a key set with per-column keys. All keys must have the same length.
    pub fn with_columns(
        footer: Vec<u8>,
        columns: BTreeMap<String, Vec<u8>>,
        aad_prefix: Vec<u8>,
    ) -> Result<Self> {
        let footer_len = validate_key(&footer)?;
        let mut cols: BTreeMap<String, Zeroizing<Vec<u8>>> = BTreeMap::new();
        for (name, key) in columns {
            let len = validate_key(&key)?;
            if len != footer_len {
                return Err(IcefallDBError::Encryption(format!(
                    "column '{name}' key length {:?} does not match footer key length {:?}; \
                     Parquet modular encryption requires consistent key lengths per file",
                    len, footer_len
                )));
            }
            cols.insert(name, Zeroizing::new(key));
        }
        Ok(Self {
            footer: Zeroizing::new(footer),
            columns: cols,
            aad_prefix,
        })
    }

    /// Return the (uniform) key length of this key set.
    pub fn key_length(&self) -> KeyLength {
        // SAFETY: constructor validated the length.
        KeyLength::from_bytes(&self.footer).expect("validated at construction")
    }

    /// Return the footer key bytes.
    pub fn footer_bytes(&self) -> &[u8] {
        &self.footer
    }

    /// Iterate over `(column_name, key_bytes)` pairs.
    pub fn column_pairs(&self) -> impl Iterator<Item = (&String, &[u8])> {
        self.columns.iter().map(|(k, v)| (k, v.as_slice()))
    }

    /// Decode a hex-encoded key into bytes, validating the length.
    pub fn decode_hex(hex_str: &str) -> Result<Vec<u8>> {
        let bytes = hex::decode(hex_str)
            .map_err(|e| IcefallDBError::Encryption(format!("invalid hex key ({e})")))?;
        validate_key(&bytes)?;
        Ok(bytes)
    }

    /// Load a key set from a JSON file of the form:
    ///
    /// ```json
    /// {
    ///   "footer_key":   "30313233343536373839616263646566",
    ///   "column_keys":  { "ssn": "30313233343536373839616263646566" },
    ///   "aad_prefix":   "b64..."
    /// }
    /// ```
    ///
    /// Keys are hex-encoded. `aad_prefix` is base64-encoded (it carries
    /// arbitrary bytes, not necessarily valid UTF-8).
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).map_err(|e| {
            IcefallDBError::Encryption(format!("failed to read key file {}: {e}", path.display()))
        })?;
        Self::from_json_str(&raw)
    }

    /// Same as [`Self::from_json_file`] but from an in-memory string.
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let parsed: KeyFile = serde_json::from_str(raw)?;
        let footer = Self::decode_hex(&parsed.footer_key)?;
        let mut columns = BTreeMap::new();
        for (name, hex) in parsed.column_keys.unwrap_or_default() {
            columns.insert(name, Self::decode_hex(&hex)?);
        }
        let aad_prefix = if let Some(b64) = parsed.aad_prefix {
            base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| {
                    IcefallDBError::Encryption(format!("invalid base64 aad_prefix ({e})"))
                })?
        } else {
            Vec::new()
        };
        Self::with_columns(footer, columns, aad_prefix)
    }
}

impl fmt::Debug for EncryptionKeySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncryptionKeySet")
            .field("footer", &"**REDACTED**")
            .field("columns", &self.columns.keys().collect::<Vec<_>>())
            .field("aad_prefix", &format_args!("{:?}", &self.aad_prefix))
            .finish()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct KeyFile {
    /// Hex-encoded footer key.
    pub footer_key: String,
    /// Map of column name → hex-encoded key.
    pub column_keys: Option<BTreeMap<String, String>>,
    /// Optional base64-encoded AAD prefix.
    pub aad_prefix: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k16(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[test]
    fn key_length_validation() {
        assert!(validate_key(&k16("0123456789abcdef")).is_ok());
        assert!(validate_key(&k16("0123456789abcdef01234567")).is_ok());
        assert!(validate_key(&k16("0123456789abcdef0123456789abcdef")).is_ok());
        let err = validate_key(b"too short").unwrap_err();
        assert!(matches!(err, IcefallDBError::Encryption(_)));
    }

    #[test]
    fn footer_only_round_trip() {
        let ks = EncryptionKeySet::footer_only(k16("0123456789abcdef"), b"aad".to_vec()).unwrap();
        assert_eq!(ks.column_pairs().count(), 0);
        assert_eq!(ks.footer_bytes(), &k16("0123456789abcdef"));
    }

    #[test]
    fn column_keys_must_match_footer_length() {
        let mut cols = BTreeMap::new();
        cols.insert("ssn".to_string(), k16("0123456789abcdef01234567")); // 24 bytes
        let err = EncryptionKeySet::with_columns(
            k16("0123456789abcdef"), // 16 bytes
            cols,
            Vec::new(),
        )
        .unwrap_err();
        assert!(matches!(err, IcefallDBError::Encryption(_)));
    }

    #[test]
    fn debug_redacts_keys() {
        let ks = EncryptionKeySet::footer_only(k16("0123456789abcdef"), Vec::new()).unwrap();
        let s = format!("{:?}", ks);
        assert!(s.contains("REDACTED"));
        assert!(!s.contains("30313233"));
    }

    #[test]
    fn json_round_trip() {
        let json = r#"{
            "footer_key": "30313233343536373839616263646566",
            "column_keys": { "ssn": "30313233343536373839616263646566" },
            "aad_prefix": "YWFk"
        }"#;
        let ks = EncryptionKeySet::from_json_str(json).unwrap();
        assert_eq!(ks.footer_bytes(), &k16("0123456789abcdef"));
        assert_eq!(ks.column_pairs().count(), 1);
        assert_eq!(ks.aad_prefix, b"aad".to_vec());
    }
}
