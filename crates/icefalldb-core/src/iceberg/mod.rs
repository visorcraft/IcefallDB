//! One-way offline Iceberg export bridge for IcefallDB.
//!
//! This module generates Iceberg-compatible `metadata.json`, manifest lists,
//! and manifest files in a separate output location, referencing the existing
//! IcefallDB Parquet files. IcefallDB's plain-JSON metadata remains the source of
//! truth.

mod data_file;
mod export;
mod manifest;
mod manifest_list;
mod metadata;
pub mod partition_spec;
pub mod schema;
pub mod sort_order;

pub use export::export_table;
