use crate::metadata::checksum::{checksum_bytes, checksum_json};
use crate::rowid::RowIdSegment;
use crate::{IcefallDBError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-column statistics for a row group.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnStats {
    /// Minimum value in the column, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<serde_json::Value>,
    /// Maximum value in the column, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<serde_json::Value>,
    /// Number of null values in the column.
    pub nulls: usize,
}

/// Byte offset and length of a column chunk within a Parquet data file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ColumnChunkOffset {
    /// Offset in bytes from the start of the data file.
    pub offset: u64,
    /// Length in bytes of the column chunk.
    pub length: u64,
}

/// Metadata describing a single row group.
///
/// The `row_group` field stores the row-group id (no extension).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RowGroupMeta {
    /// Row-group identifier (e.g., `rg_a7f3b2`), with no file extension.
    pub row_group: String,
    /// Id of the schema used to write this row group.
    pub schema_id: u64,
    /// Number of rows in the row group.
    pub rows: usize,
    /// Per-column statistics, keyed by column name.
    pub columns: HashMap<String, ColumnStats>,
    /// Optional per-column byte offsets into the Parquet data file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column_offsets: Option<HashMap<String, ColumnChunkOffset>>,
    /// Optional sort order for the row group.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<Vec<String>>,
    /// Stable row-ID segments describing which global row IDs are stored in
    /// this row group. An empty vec means the row IDs have not been assigned
    /// yet (legacy row groups written before row-ID allocation was introduced).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub row_ids: Vec<RowIdSegment>,
    /// SHA-256 checksum of the Parquet data file (not the row group metadata
    /// file). This is computed over the raw bytes of the file referenced by
    /// `row_group`.
    pub checksum: String,
    /// SHA-256 checksum of the metadata in this struct itself. This is
    /// computed over a canonical JSON form of the struct with `checksum` and
    /// `meta_checksum` excluded, so it detects tampering with the metadata
    /// fields without being circular.
    pub meta_checksum: String,
}

impl RowGroupMeta {
    /// Computes the SHA-256 checksum of the provided Parquet data bytes and
    /// stores it in the `checksum` field, then computes and stores the
    /// metadata's own checksum.
    ///
    /// The returned checksum has the form `sha256:<hex>`.
    pub fn compute_checksum(&mut self, parquet_bytes: &[u8]) -> Result<String> {
        let cs = checksum_bytes(parquet_bytes);
        self.checksum = cs.clone();
        self.compute_meta_checksum()?;
        Ok(cs)
    }

    /// Computes the SHA-256 checksum of this metadata struct itself.
    ///
    /// The checksum is taken over a canonical JSON form of the struct with
    /// `checksum` and `meta_checksum` cleared, so it is stable and not
    /// circular. The result is stored in `meta_checksum` and returned.
    pub fn compute_meta_checksum(&mut self) -> Result<String> {
        let saved_checksum = self.checksum.clone();
        self.checksum.clear();
        self.meta_checksum.clear();
        let value = serde_json::to_value(&*self).map_err(IcefallDBError::Serialization)?;
        let cs = checksum_json(&value);
        self.meta_checksum = cs.clone();
        self.checksum = saved_checksum;
        Ok(cs)
    }

    /// Verifies that the stored checksum matches the checksum of the provided
    /// Parquet data bytes.
    pub fn verify_against_data(&self, parquet_bytes: &[u8]) -> bool {
        checksum_bytes(parquet_bytes) == self.checksum
    }

    /// Verifies that `meta_checksum` matches the current metadata fields.
    pub fn verify_meta_checksum(&self) -> Result<bool> {
        if self.meta_checksum.is_empty() {
            return Ok(false);
        }
        let mut clone = self.clone();
        clone.checksum.clear();
        clone.meta_checksum.clear();
        let value = serde_json::to_value(&clone).map_err(IcefallDBError::Serialization)?;
        let cs = checksum_json(&value);
        Ok(cs == self.meta_checksum)
    }
}
