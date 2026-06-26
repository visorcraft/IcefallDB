//! Benchmark DataFusion's native ParquetExec against the events_wide chunk files.

use std::path::Path;
use std::thread::available_parallelism;
use std::time::Instant;

use datafusion::execution::context::SessionContext;
use datafusion::prelude::SessionConfig;

#[tokio::main]
async fn main() {
    // Bench fixtures live under the workspace target/tmp dir, derived from the crate location.
    let bench_tmp = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .join("target/tmp");
    let chunk_dir = bench_tmp.join("datafusion_bench_chunks_sorted/events_wide");
    let cpus = available_parallelism().map(|n| n.get()).unwrap_or(1);
    let target_partitions = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(cpus);

    let mut config = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(65536);
    config.options_mut().execution.parquet.pushdown_filters = false;
    config.options_mut().execution.parquet.bloom_filter_on_read = false;
    let ctx = SessionContext::new_with_config(config);

    let wide_path = format!("file://{}/", chunk_dir.to_string_lossy());
    ctx.register_parquet(
        "events_wide",
        &wide_path,
        datafusion::datasource::file_format::options::ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let events_path = format!(
        "file://{}/",
        bench_tmp
            .join("datafusion_bench_chunks/events")
            .to_string_lossy()
    );
    ctx.register_parquet(
        "events",
        &events_path,
        datafusion::datasource::file_format::options::ParquetReadOptions::default(),
    )
    .await
    .unwrap();

    let queries: Vec<(&str, &str)> = vec![
        (
            "warm_filtered_scan",
            "SELECT COUNT(*) AS cnt, SUM(value) AS total FROM events \
          WHERE category = 'purchase' AND value > 50",
        ),
        (
            "wide_filter",
            "SELECT COUNT(*) AS cnt FROM events_wide \
        WHERE int_a > 100 AND float_b > 50 AND category = 'purchase' \
          AND status = 'ok' AND region = 'US'",
        ),
    ];

    // Print physical plan for wide_filter.
    let plan = ctx
        .sql(queries[1].1)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    println!(
        "\nEXPLAIN wide_filter:\n{}",
        datafusion::physical_plan::displayable(plan.as_ref()).indent(true)
    );

    for (name, sql) in &queries {
        // warm up
        let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();

        let iters = 5;
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Instant::now();
            let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "native parquet {name:20} target_partitions={target_partitions} min={:.3}ms median={:.3}ms max={:.3}ms",
            times[0],
            times[times.len() / 2],
            times[times.len() - 1]
        );
    }
}
