use datafusion::physical_plan::displayable;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};
use std::path::Path;
use std::sync::Arc;
#[tokio::main]
async fn main() {
    // Default bench DB lives under the workspace target/tmp dir, derived from the crate location.
    let db_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .join("target/tmp/datafusion_bench_db");
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(&db_path).unwrap());
    let config = ProviderConfig::default();
    let categories = IcefallDBTableProvider::new(Arc::clone(&storage), "categories", config)
        .await
        .unwrap();
    let ctx = icefalldb_session(config.target_partitions, config.batch_size);
    ctx.register_table("categories", Arc::new(categories))
        .unwrap();
    let df = ctx.sql("SELECT * FROM categories").await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    println!("{}", displayable(plan.as_ref()).indent(true));
    let stats = plan.partition_statistics(None).unwrap();
    println!("num_rows: {:?}", stats.num_rows);
}
