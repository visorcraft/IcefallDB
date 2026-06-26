//! Turnkey Parquet Modular Encryption for DataFusion.
//!
//! DataFusion 54 already ships native Parquet Modular Encryption (PME) via the
//! `EncryptionFactory` trait on `datafusion-execution`. The mechanism is
//! powerful but verbose: callers must implement the factory, register it on
//! the `RuntimeEnv` by string id, then set `format.crypto.factory_id = "..."`
//! in their session config and SQL `OPTIONS (...)`.
//!
//! This crate wraps that machinery in a single call:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use datafusion::execution::context::SessionContext;
//! # use datafusion_encrypted_parquet::register_encryption_factory;
//! # fn main() -> datafusion_common::Result<()> {
//! let ctx = SessionContext::new();
//! register_encryption_factory(&ctx, Arc::new(MyKeys::default()))?;
//! // Now any ListingTable / ParquetFormat / ParquetSource created on `ctx`
//! // will use MyKeys when `format.crypto.factory_id = "encrypted-parquet"`
//! // is set on its table options.
//! # Ok(())
//! # }
//! # struct MyKeys;
//! # impl datafusion_encrypted_parquet::KeySource for MyKeys {
//! #     fn get(&self, _id: &str) -> datafusion_common::Result<Vec<u8>> { Ok(vec![]) }
//! # }
//! # impl Default for MyKeys { fn default() -> Self { Self } }
//! ```
//!
//! ## When to use this crate
//!
//! - You want to read/write Parquet Modular Encryption-encrypted files with
//!   DataFusion's `ListingTable` (e.g. you are not using IcefallDB's custom
//!   scan).
//! - You want a single, dependency-light crate that does not pull in
//!   IcefallDB.
//!
//! ## When *not* to use this crate
//!
//! - You are using IcefallDB. Use `icefalldb-query`'s `encryption` feature
//!   instead — it does the same thing but threads decryption through both
//!   the custom `IcefallDBScanExec` and the native `ParquetSource` paths.
//! - You want full control over the factory mechanism. Use
//!   `datafusion_execution::parquet_encryption::EncryptionFactory` directly.
//!
//! ## Feature flag
//!
//! The crate's API compiles without the `encryption` feature, but
//! [`register_encryption_factory`] returns an error at runtime in that mode.
//! Add `datafusion-encrypted-parquet/encryption` to your `Cargo.toml` to
//! enable real decryption.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "encryption")]
mod factory;
#[cfg(feature = "encryption")]
mod keys;

#[cfg(feature = "encryption")]
pub use factory::{register_encryption_factory, EncryptedParquetFormat, FACTORY_ID};
#[cfg(feature = "encryption")]
pub use keys::KeySource;

#[cfg(not(feature = "encryption"))]
mod stub;

#[cfg(not(feature = "encryption"))]
pub use stub::{register_encryption_factory, KeySource, FACTORY_ID};
