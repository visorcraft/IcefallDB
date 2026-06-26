//! Rust-native benchmark of the IcefallDB DataFusion query engine.
//!
//! Run against an existing IcefallDB benchmark database:
//!
//!   cargo run --release -p icefalldb-query --example bench_queries -- \
//!     target/tmp/datafusion_bench_db
//!
//! This bypasses Python entirely and measures in-process DataFusion performance.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;
use std::time::Instant;

use datafusion::physical_plan::{collect, displayable, ExecutionPlan};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn print_plan_metrics(plan: &dyn ExecutionPlan, depth: usize) {
    let pad = "  ".repeat(depth);
    if let Some(metrics) = plan.metrics() {
        println!("{pad}metrics for {}:\n{pad}{metrics}", plan.name());
    }
    for child in plan.children() {
        print_plan_metrics(child.as_ref(), depth + 1);
    }
}

#[tokio::main]
async fn main() {
    // Default bench DB lives under the workspace target/tmp dir, derived from the crate location.
    let db_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .ancestors()
                .nth(2)
                .unwrap()
                .join("target/tmp/datafusion_bench_db")
        });

    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(&db_path).unwrap());

    let cpus = available_parallelism().map(|n| n.get()).unwrap_or(1);
    let target_partitions = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(cpus);
    let batch_size = std::env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(65536usize);
    let native_parquet_threshold = std::env::args()
        .nth(4)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1usize);
    let config = ProviderConfig {
        batch_size,
        target_partitions,
        io_coalesce_window: 1024 * 1024,
        io_concurrency: target_partitions.max(1) * 2,
        native_parquet_threshold,
        parquet_metadata_cache_capacity: 256,
        tiny_table_cache_threshold_rows: 65_536,
        tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
        wal_mode: true,
    };

    let events = IcefallDBTableProvider::new(Arc::clone(&storage), "events", config)
        .await
        .unwrap();
    let categories = IcefallDBTableProvider::new(Arc::clone(&storage), "categories", config)
        .await
        .unwrap();
    let events_wide = IcefallDBTableProvider::new(Arc::clone(&storage), "events_wide", config)
        .await
        .unwrap();

    let ctx = icefalldb_session(config.target_partitions, config.batch_size);
    ctx.register_table("events", Arc::new(events)).unwrap();
    ctx.register_table("categories", Arc::new(categories))
        .unwrap();
    ctx.register_table("events_wide", Arc::new(events_wide))
        .unwrap();

    let queries: Vec<(&str, &str)> = vec![
        (
            "warm_agg_group",
            "SELECT category, COUNT(*) AS event_count, AVG(value) AS avg_value FROM events \
          WHERE ts >= TIMESTAMP '2024-03-01 00:00:00' AND ts < TIMESTAMP '2024-04-01 00:00:00' \
          GROUP BY category ORDER BY event_count DESC LIMIT 10",
        ),
        (
            "warm_filtered_scan",
            "SELECT COUNT(*) AS cnt, SUM(value) AS total FROM events \
          WHERE category = 'cat_0' AND value > 0.5",
        ),
        ("warm_full_count", "SELECT COUNT(*) AS cnt FROM events"),
        (
            "join_1m_x_10",
            "SELECT c.category_name, COUNT(*) AS event_count FROM events e \
          JOIN categories c ON e.category = c.category_name \
          GROUP BY c.category_name ORDER BY event_count DESC",
        ),
        (
            "wide_agg",
            "SELECT SUM(int_a) AS s_a, AVG(float_a) AS avg_a, SUM(int_b) AS s_b, \
          AVG(float_b) AS avg_b, AVG(float_c) AS avg_c, SUM(int_c) AS s_c \
          FROM events_wide WHERE int_d > 100 AND float_d < 0.5 AND status = 'ok'",
        ),
        (
            "wide_filter",
            "SELECT COUNT(*) AS cnt FROM events_wide \
          WHERE int_a > 500000 AND float_b > 0.5 AND int_c < 500000 \
            AND status = 'ok'",
        ),
    ];

    let iters = 5;
    println!(
        "Running {} queries, {} iterations each",
        queries.len(),
        iters
    );
    println!(
        "Target partitions: {}, batch size: {}, native_parquet_threshold: {}",
        config.target_partitions, config.batch_size, config.native_parquet_threshold
    );
    println!("args: db_path [target_partitions] [batch_size] [native_parquet_threshold]");
    println!();

    for (name, sql) in &queries {
        // Print optimized physical plan and pre-run metrics for every query.
        {
            let df = ctx.sql(sql).await.unwrap();
            let plan = df.clone().create_physical_plan().await.unwrap();
            println!("\n=== EXPLAIN: {name} ===");
            println!("{}", displayable(plan.as_ref()).indent(true));
            if let Some(metrics) = plan.metrics() {
                println!("=== METRICS (pre-run): {name} ===");
                println!("{metrics}");
            }
        }

        // Warm up once.
        let warmup = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        if *name == "wide_filter" || *name == "warm_filtered_scan" {
            use arrow::array::Array;
            let col = warmup[0].column(0);
            let arr = col
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            println!("{name} count = {}", arr.value(0));
        }

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = times[0];
        let max = times[times.len() - 1];
        let median = times[times.len() / 2];
        println!("{name:20} min={min:8.3}ms median={median:8.3}ms max={max:8.3}ms",);

        // Capture post-run metrics from a single execution for attribution.
        let df = ctx.sql(sql).await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let _ = collect(plan.clone(), ctx.task_ctx()).await.unwrap();
        println!("=== METRICS (post-run single): {name} ===");
        print_plan_metrics(plan.as_ref(), 0);
    }
}
