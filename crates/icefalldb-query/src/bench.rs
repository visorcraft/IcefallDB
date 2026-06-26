//! Benchmark harness types and helpers for the DataFusion query layer.

use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Result of running a single query repeatedly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Human-readable query identifier.
    pub query_name: String,
    /// Engine name (e.g. `datafusion`, `duckdb`).
    pub engine: String,
    /// Raw iteration timings in milliseconds.
    pub iterations: Vec<f64>,
    /// Median iteration time in milliseconds.
    pub median_ms: f64,
    /// Minimum iteration time in milliseconds.
    pub min_ms: f64,
    /// Maximum iteration time in milliseconds.
    pub max_ms: f64,
}

/// A full benchmark matrix for one dataset/scale combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMatrix {
    /// Dataset name.
    pub dataset: String,
    /// Scale identifier (e.g. `1m`, `10m`, `100m`).
    pub scale: String,
    /// Per-query results.
    pub queries: Vec<QueryResult>,
}

/// Time `iters` executions of the async closure `f`, returning the elapsed time
/// of each iteration in milliseconds.
pub async fn time_query<F, Fut>(iters: usize, mut f: F) -> Vec<f64>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = ()>,
{
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f().await;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    times
}
