//! A/B profile: Parquet filter-pushdown ON (selective-decode path) vs OFF
//! (bulk-decode + FilterExec) over the real `events_wide` bench data.
//!
//! Mirrors IcefallDB's native session config (no view types, no auto file-scan
//! repartition, page index on). Run after generating the datafusion bench db:
//!   cargo run --release -p icefalldb-query --example bench_pushdown_ab -- on
//!   cargo run --release -p icefalldb-query --example bench_pushdown_ab -- off
//! Optional args: <on|off> [target_partitions] [batch_size]

use std::path::Path;
use std::thread::available_parallelism;
use std::time::Instant;

use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;

#[tokio::main]
async fn main() {
    let bench_db = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .join("target/tmp/datafusion_bench_db");

    let mut args = std::env::args().skip(1);
    let pushdown = matches!(args.next().as_deref(), Some("on"));
    let cpus = available_parallelism().map(|n| n.get()).unwrap_or(1);
    let target_partitions = args.next().and_then(|s| s.parse().ok()).unwrap_or(cpus);
    let batch_size = args.next().and_then(|s| s.parse().ok()).unwrap_or(8192);

    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(batch_size);
    config.options_mut().execution.parquet.pushdown_filters = pushdown;
    config.options_mut().execution.parquet.reorder_filters = true;
    config
        .options_mut()
        .execution
        .parquet
        .schema_force_view_types = false;
    config.options_mut().optimizer.repartition_file_scans = false;
    let ctx = SessionContext::new_with_config(config);

    for table in ["events_wide", "events"] {
        let dir = bench_db.join(table);
        // register_parquet over a dir picks up *.parquet only.
        let url = format!("file://{}/", dir.to_string_lossy());
        let opts = datafusion::datasource::file_format::options::ParquetReadOptions::default();
        if dir.exists() {
            ctx.register_parquet(table, &url, opts).await.unwrap();
        }
    }

    let queries: Vec<(&str, &str)> = vec![
        (
            "wide_filter",
            "SELECT COUNT(*) AS cnt FROM events_wide \
             WHERE int_a > 500000 AND float_b > 0.5 AND int_c < 500000 AND status = 'ok'",
        ),
        (
            "wide_agg",
            "SELECT SUM(int_a) AS s_a, AVG(float_a) AS avg_a, SUM(int_b) AS s_b, \
             AVG(float_b) AS avg_b, AVG(float_c) AS avg_c, SUM(int_c) AS s_c \
             FROM events_wide WHERE int_d > 100 AND float_d < 0.5 AND status = 'ok'",
        ),
        (
            "warm_filtered_scan",
            "SELECT COUNT(*) AS cnt, SUM(value) AS total FROM events \
             WHERE category = 'cat_0' AND value > 0.5",
        ),
    ];

    println!(
        "== pushdown={} target_partitions={target_partitions} batch_size={batch_size} ==",
        if pushdown { "ON" } else { "OFF" }
    );

    for (name, sql) in &queries {
        if ctx.sql(sql).await.is_err() {
            continue;
        }
        // warm up (2 discarded)
        for _ in 0..2 {
            let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        }
        let iters = 7;
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "{name:20} p50={:.2}ms min={:.2}ms max={:.2}ms",
            times[times.len() / 2],
            times[0],
            times[times.len() - 1]
        );
    }

    // EXPLAIN ANALYZE for wide_filter to expose the scan time breakdown.
    let analyze = format!("EXPLAIN ANALYZE {}", queries[0].1);
    if let Ok(df) = ctx.sql(&analyze).await {
        let batches = df.collect().await.unwrap();
        let txt = datafusion::arrow::util::pretty::pretty_format_batches(&batches).unwrap();
        // Print only lines mentioning the data source / scan metrics.
        for line in txt.to_string().lines() {
            if line.contains("DataSourceExec")
                || line.contains("time_elapsed")
                || line.contains("pushdown_rows")
                || line.contains("bytes_scanned")
                || line.contains("row_pushdown")
                || line.contains("FilterExec")
            {
                println!("  {}", line.trim());
            }
        }
    }
}
