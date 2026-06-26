use thiserror::Error;

#[derive(Error, Debug)]
pub enum QueryError {
    #[error("icefalldb core error: {0}")]
    Core(#[from] icefalldb_core::IcefallDBError),
    #[error("datafusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("unsupported type for sidecar stat: {0}")]
    UnsupportedStatType(String),
    #[error("stats unavailable")]
    StatsUnavailable,
    #[error("{0}")]
    Other(String),
}

impl From<QueryError> for datafusion::error::DataFusionError {
    fn from(err: QueryError) -> Self {
        datafusion::error::DataFusionError::External(Box::new(err))
    }
}

pub type Result<T> = std::result::Result<T, QueryError>;
