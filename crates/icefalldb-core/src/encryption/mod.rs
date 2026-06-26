//! Optional Parquet Modular Encryption (PME) support.
//!
//! Enabled by the `encryption` feature on `icefalldb-core`. When enabled, the
//! writer can produce Parquet files encrypted per the [Apache Parquet 2.9+
//! modular encryption specification](https://github.com/apache/parquet-format/blob/master/Encryption.md),
//! and the reader (in `icefalldb-query`) can transparently decrypt them.
//!
//! # Layout
//!
//! - [`keys`] — `EncryptionKeySet`, `KeyIdentifier`, key validation.
//! - [`provider`] — `KeyProvider` trait + `EnvKeyProvider`, `FileKeyProvider`,
//!   `StaticKeyProvider`.
//! - [`properties`] — builders for `FileEncryptionProperties` /
//!   `FileDecryptionProperties`.
//! - [`aad`] — additional-authenticated-data derivation bound to a table.
//! - [`config`] — `EncryptionWriteConfig` consumed by [`crate::Writer`].
//!
//! When the `encryption` feature is off, this module is empty and the writer
//! uses the existing plaintext Parquet code path with zero overhead.

#[cfg(feature = "encryption")]
pub mod aad;
#[cfg(feature = "encryption")]
pub mod config;
#[cfg(feature = "encryption")]
pub mod keys;
#[cfg(feature = "encryption")]
pub mod properties;
#[cfg(feature = "encryption")]
pub mod provider;

#[cfg(feature = "encryption")]
pub use aad::table_aad_prefix;
#[cfg(feature = "encryption")]
pub use config::{EncryptionWriteConfig, SchemaEncryptionMarker};
#[cfg(feature = "encryption")]
pub use keys::{EncryptionKeySet, KeyIdentifier, KeyLength};
#[cfg(feature = "encryption")]
pub use properties::{build_decryption_properties, build_encryption_properties};
#[cfg(feature = "encryption")]
pub use provider::{
    EnvKeyProvider, FileKeyProvider, KeyProvider, StaticKeyProvider, KEY_ID_ENV_PREFIX,
};
