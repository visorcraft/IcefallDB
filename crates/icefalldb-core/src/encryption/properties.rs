//! Builders for `parquet::encryption` `FileEncryptionProperties` and
//! `FileDecryptionProperties`, bridging [`EncryptionKeySet`] into the parquet
//! crate's API.
//!
//! These are the only functions in `icefalldb-core` that touch `parquet::encryption`
//! directly; they are the single point of contact between our key representation
//! and the parquet crate's. Keeping them here means the writer, the reader,
//! and the future KMS factory all share one definition.

use std::collections::BTreeSet;
use std::sync::Arc;

use parquet::encryption::decrypt::FileDecryptionProperties;
use parquet::encryption::encrypt::FileEncryptionProperties;

use crate::encryption::EncryptionKeySet;
use crate::error::{IcefallDBError, Result};

/// Build `FileEncryptionProperties` for an `ArrowWriter` from a key set.
///
/// - `plaintext_footer = true` keeps the Parquet footer unencrypted while
///   encrypting column data. This preserves the byte-range + page-index reads
///   used by `IcefallDBScanExec`. Set to `false` to also encrypt the footer
///   (small confidentiality gain at a significant scan-speed cost).
/// - `store_aad_prefix = true` writes the AAD prefix into the file footer so
///   readers do not need to provide it to read the data. Readers may still
///   provide it for *verification*.
///
/// `encrypted_columns` selects the encryption mode (matches parquet-rs's
/// `FileEncryptionProperties::is_column_encrypted` semantics):
///
/// - **Uniform mode** (default; `encrypted_columns` empty AND `keys.columns`
///   empty): every column is encrypted with the footer key. The reader needs
///   only the footer key.
/// - **Per-column mode** (`encrypted_columns` non-empty, OR `keys.columns`
///   non-empty): only the listed columns are encrypted (each with its own key
///   from `keys.columns`); every other column is **plaintext**. This matches
///   parquet-rs's `ENCRYPTION_WITH_COLUMN_KEY` behavior — there is no
///   "encrypt unlisted columns with the footer key" middle ground. The reader
///   must supply a per-column key for every encrypted column.
///
/// To encrypt every column but give one its own key, callers must today use
/// uniform mode (one footer key for everything). True per-column-key-with-
/// footer-fallback requires a `KeyRetriever` on the reader side and is
/// planned for v1.x.
pub fn build_encryption_properties(
    keys: &EncryptionKeySet,
    plaintext_footer: bool,
    store_aad_prefix: bool,
    encrypted_columns: &BTreeSet<String>,
) -> Result<Arc<FileEncryptionProperties>> {
    let mut builder = FileEncryptionProperties::builder(keys.footer_bytes().to_vec())
        .with_plaintext_footer(plaintext_footer);

    if !keys.aad_prefix.is_empty() {
        builder = builder
            .with_aad_prefix(keys.aad_prefix.clone())
            .with_aad_prefix_storage(store_aad_prefix);
    }

    // Apply per-column keys. When `encrypted_columns` is set, restrict to
    // those columns; otherwise apply every per-column key in `keys.columns`.
    // Unlisted columns are left plaintext by parquet-rs — this is intentional
    // and documented above.
    let want: Vec<(String, Vec<u8>)> = if encrypted_columns.is_empty() {
        keys.column_pairs()
            .map(|(n, k)| (n.clone(), k.to_vec()))
            .collect()
    } else {
        keys.column_pairs()
            .filter(|(n, _)| encrypted_columns.contains(*n))
            .map(|(n, k)| (n.clone(), k.to_vec()))
            .collect()
    };
    for (name, key_bytes) in want {
        builder = builder.with_column_key(&name, key_bytes);
    }

    builder.build().map_err(map_parquet_enc_err)
}

/// Build `FileDecryptionProperties` for a reader from a key set.
///
/// The footer key and any per-column keys are provided directly. AAD prefix
/// is provided too, for files that opt to verify it.
pub fn build_decryption_properties(
    keys: &EncryptionKeySet,
) -> Result<Arc<FileDecryptionProperties>> {
    let mut builder = FileDecryptionProperties::builder(keys.footer_bytes().to_vec());
    if !keys.aad_prefix.is_empty() {
        builder = builder.with_aad_prefix(keys.aad_prefix.clone());
    }
    for (name, key_bytes) in keys.column_pairs() {
        builder = builder.with_column_key(name, key_bytes.to_vec());
    }
    builder.build().map_err(map_parquet_enc_err)
}

fn map_parquet_enc_err(e: parquet::errors::ParquetError) -> IcefallDBError {
    IcefallDBError::Encryption(e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_key_set() -> EncryptionKeySet {
        let mut cols = std::collections::BTreeMap::new();
        cols.insert("ssn".to_string(), b"0123456789abcdef".to_vec());
        EncryptionKeySet::with_columns(
            b"0123456789abcdef".to_vec(),
            cols,
            b"icefalldb:v1:events:1".to_vec(),
        )
        .unwrap()
    }

    #[test]
    fn uniform_mode_when_no_per_column_keys() {
        // Footer-only key set → uniform mode (parquet-rs encrypts every column
        // with the footer key). No per-column keys are emitted.
        let keys =
            EncryptionKeySet::footer_only(b"0123456789abcdef".to_vec(), b"aad".to_vec()).unwrap();
        let enc = build_encryption_properties(&keys, true, true, &BTreeSet::new()).unwrap();
        assert!(!enc.encrypt_footer()); // plaintext_footer=true
        let (names, _, _) = enc.column_keys();
        assert!(names.is_empty(), "uniform mode has no per-column keys");
    }

    #[test]
    fn per_column_mode_lists_only_encrypted_columns() {
        // Regression for the codex-found bug. When `keys.columns` is non-empty
        // (per-column mode), only the listed columns are encrypted; parquet-rs
        // leaves unlisted columns plaintext. The properties reflect this: only
        // the listed columns appear in column_keys().
        let ks = sample_key_set();
        let enc = build_encryption_properties(&ks, true, true, &BTreeSet::new()).unwrap();
        let (names, keys_for_cols, _) = enc.column_keys();
        assert_eq!(names, vec!["ssn".to_string()]);
        assert_eq!(keys_for_cols, vec![b"0123456789abcdef".to_vec()]);
    }

    #[test]
    fn encrypted_columns_filter_restricts_per_column_keys() {
        // If the caller restricts to a subset, only those per-column keys are
        // applied. Other columns in `keys.columns` (and all unlisted columns)
        // remain plaintext.
        let ks = sample_key_set();
        let mut filter = BTreeSet::new();
        // Filter excludes ssn → no per-column keys are applied at all → file
        // falls back to uniform mode (all columns encrypted with footer key).
        filter.insert("not_listed".to_string());
        let enc = build_encryption_properties(&ks, true, true, &filter).unwrap();
        let (names, _, _) = enc.column_keys();
        assert!(names.is_empty(), "filter excluded every per-column key");
    }

    #[test]
    fn aad_prefix_is_propagated() {
        let ks = sample_key_set();
        let enc = build_encryption_properties(&ks, true, true, &BTreeSet::new()).unwrap();
        assert_eq!(enc.aad_prefix(), Some(&b"icefalldb:v1:events:1".to_vec()));
        assert!(enc.store_aad_prefix());
    }

    #[test]
    fn decryption_properties_round_trip() {
        let ks = sample_key_set();
        let dec = build_decryption_properties(&ks).unwrap();
        let footer_key = dec.footer_key(None).unwrap();
        assert_eq!(footer_key.into_owned(), b"0123456789abcdef");
    }
}
