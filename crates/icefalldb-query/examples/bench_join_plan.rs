use datafusion::physical_plan::displayable;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::available_parallelism;

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
    let config = ProviderConfig {
        batch_size: 65536,
        target_partitions: cpus,
        io_coalesce_window: 1024 * 1024,
        io_concurrency: cpus * 2,
        native_parquet_threshold: 1,
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

    let ctx = icefalldb_session(config.target_partitions, config.batch_size);
    ctx.register_table("events", Arc::new(events)).unwrap();
    ctx.register_table("categories", Arc::new(categories))
        .unwrap();

    let sql = "SELECT c.category_name, COUNT(*) AS event_count FROM events e \
        JOIN categories c ON e.category = c.category_name \
        GROUP BY c.category_name ORDER BY event_count DESC";
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    println!("{}", displayable(plan.as_ref()).indent(true));
}
