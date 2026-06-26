//! Key source trait for `datafusion-encrypted-parquet`.

use datafusion_common::Result;

/// Source of AES keys for the encryption factory.
///
/// Implementations resolve a string key identifier (the value passed to
/// `FileEncryptionProperties::with_column_key_and_metadata(.., key_metadata)`
/// on the write side, and `key_metadata` on the read side) into the raw key
/// bytes.
///
/// Implementations must be `Send + Sync` because DataFusion partitions read
/// concurrently.
pub trait KeySource: Send + Sync {
    /// Return the key bytes for `kid`. The returned `Vec<u8>` must be 16, 24,
    /// or 32 bytes (AES-128/192/256).
    fn get(&self, kid: &str) -> Result<Vec<u8>>;
}
