use thiserror::Error;

pub type Result<T> = std::result::Result<T, IcefallDBError>;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum IcefallDBError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("manifest not found: {0}")]
    ManifestNotFound(String),
    #[error("schema not found: {path}")]
    SchemaNotFound { path: String },
    #[error("checksum mismatch for {path}")]
    ChecksumMismatch { path: String },
    #[error("empty table: {0}")]
    EmptyTable(String),
    #[error("type not supported: {0}")]
    TypeNotSupported(String),
    #[error("row group checksum mismatch for {path}")]
    RowGroupChecksumMismatch { path: String },
    #[error("parquet decode error: {0}")]
    ParquetDecode(String),
    #[error("schema mismatch: column {column}, expected {expected}, path {path}")]
    SchemaMismatch {
        column: String,
        expected: String,
        path: String,
    },
    #[error("missing row group file: snapshot {snapshot}, path {path}")]
    MissingRowGroupFile { snapshot: u64, path: String },
    #[error("not found: {0}")]
    NotFound(String),
    #[error("invalid path: {0}")]
    InvalidPath(String),
    #[error("invalid manifest pointer: {0}")]
    InvalidManifestPointer(String),
    #[error("missing manifest pointer: {path}")]
    MissingManifestPointer { path: String },
    #[error("invalid schema pointer: {path}")]
    InvalidSchemaPointer { path: String },
    #[error("invalid schema: {reason}, path {path}")]
    InvalidSchema { reason: String, path: String },
    #[error("mixed partition values in row group: {path}")]
    MixedPartition { path: String },
    #[error("manifest sequence collision: sequence {0} already exists")]
    SequenceCollision(u64),
    #[error(
        "compaction conflict: source snapshot advanced from {pinned} to {current} \
         during the lock-free rewrite"
    )]
    CompactionConflict { pinned: u64, current: u64 },
    #[error("lock timeout: {0}")]
    LockTimeout(String),
    #[error("range read error for {path}: {reason}")]
    RangeReadError { path: String, reason: String },
    #[error("table already exists: {0}")]
    TableAlreadyExists(String),
    #[error("table not found: {0}")]
    TableNotFound(String),
    #[error("tsv error at line {line}, column {column}, value {value}: {reason}")]
    TsvError {
        line: usize,
        column: usize,
        value: String,
        reason: String,
    },
    #[error("encryption error: {0}")]
    Encryption(String),
    #[error("encryption key not found: {0}")]
    EncryptionKeyNotFound(String),
    #[error("decryption failed: {0}")]
    Decryption(String),
    #[error("index '{name}' on table '{table}' is in an obsolete format from an older IcefallDB version (it stores row-group identifiers as strings instead of stable row ids); rebuild it with `icefalldb create-index <db> {table} <columns>`")]
    LegacyIndex { table: String, name: String },
    #[error("snapshot {0} not found")]
    SnapshotNotFound(u64),
    #[error("{0}")]
    Other(#[source] Box<dyn std::error::Error + Send + Sync>),
}

/// Returns true if the error represents a missing object.
pub(crate) fn is_not_found(err: &IcefallDBError) -> bool {
    matches!(err, IcefallDBError::NotFound(_))
}

#[cfg(test)]
mod tests {
    use crate::{IcefallDBError, Result};

    #[test]
    fn test_result_type_aliased() {
        let _: Result<()> = Err(IcefallDBError::NotFound("test".into()));
    }

    #[test]
    fn test_is_not_found_matches_not_found() {
        assert!(super::is_not_found(&IcefallDBError::NotFound("x".into())));
        assert!(!super::is_not_found(&IcefallDBError::InvalidPath(
            "x".into()
        )));
    }
}
