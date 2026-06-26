use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug)]
pub enum ServerError {
    Internal(String),
    BadRequest(String),
    NotFound(String),
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerError::Internal(m) => write!(f, "internal error: {m}"),
            ServerError::BadRequest(m) => write!(f, "bad request: {m}"),
            ServerError::NotFound(m) => write!(f, "not found: {m}"),
        }
    }
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ServerError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
            ServerError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            ServerError::NotFound(m) => (StatusCode::NOT_FOUND, m),
        };
        (status, Json(json!({"error": message}))).into_response()
    }
}

impl From<icefalldb_core::IcefallDBError> for ServerError {
    fn from(err: icefalldb_core::IcefallDBError) -> Self {
        use icefalldb_core::IcefallDBError;
        match err {
            IcefallDBError::NotFound(_)
            | IcefallDBError::TableNotFound(_)
            | IcefallDBError::SchemaNotFound { .. }
            | IcefallDBError::SnapshotNotFound(_) => ServerError::NotFound(err.to_string()),
            IcefallDBError::InvalidSchema { .. }
            | IcefallDBError::InvalidPath(_)
            | IcefallDBError::TypeNotSupported(_) => ServerError::BadRequest(err.to_string()),
            _ => ServerError::Internal(err.to_string()),
        }
    }
}

impl From<datafusion::error::DataFusionError> for ServerError {
    fn from(err: datafusion::error::DataFusionError) -> Self {
        ServerError::Internal(err.to_string())
    }
}

impl From<serde_json::Error> for ServerError {
    fn from(err: serde_json::Error) -> Self {
        ServerError::BadRequest(err.to_string())
    }
}

impl From<std::io::Error> for ServerError {
    fn from(err: std::io::Error) -> Self {
        ServerError::Internal(err.to_string())
    }
}

impl From<icefalldb_query::QueryError> for ServerError {
    fn from(err: icefalldb_query::QueryError) -> Self {
        match err {
            icefalldb_query::QueryError::Core(e) => ServerError::from(e),
            icefalldb_query::QueryError::DataFusion(e) => ServerError::Internal(e.to_string()),
            icefalldb_query::QueryError::Arrow(e) => ServerError::Internal(e.to_string()),
            icefalldb_query::QueryError::Parquet(e) => ServerError::Internal(e.to_string()),
            icefalldb_query::QueryError::UnsupportedStatType(_)
            | icefalldb_query::QueryError::StatsUnavailable
            | icefalldb_query::QueryError::Other(_) => ServerError::Internal(err.to_string()),
        }
    }
}
