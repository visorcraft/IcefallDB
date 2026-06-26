//! End-to-end tests for encrypted Parquet tables read through the IcefallDB
//! DataFusion query engine.
//!
//! Only compiled when the `encryption` feature is enabled:
//! ```sh
//! cargo test -p icefalldb-query --features encryption --test encryption_e2e
//! ```

#![cfg(feature = "encryption")]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::arrow_schema_to_icefalldb;
use icefalldb_core::encryption::{
    table_aad_prefix, EncryptionKeySet, EncryptionWriteConfig, KeyIdentifier, KeyProvider,
    StaticKeyProvider,
};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{Writer, WriterOptionsFull};
use icefalldb_query::{icefalldb_encrypted_session, IcefallDBTableProvider, ProviderConfig};

const FOOTER_KEY: &[u8; 16] = b"0123456789abcdef";

fn make_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]));
    let a = Int32Array::from(vec![Some(1), Some(2), Some(3), None, Some(5), Some(6)]);
    let b = StringArray::from(vec![
        Some("a"),
        Some("b"),
        Some("c"),
        Some("d"),
        None,
        Some("f"),
    ]);
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
}

fn make_key_set(table: &str) -> EncryptionKeySet {
    EncryptionKeySet::footer_only(FOOTER_KEY.to_vec(), table_aad_prefix(table, 1)).unwrap()
}

async fn create_encrypted_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let icefalldb_schema = arrow_schema_to_icefalldb(make_batch().schema());
    let cfg = EncryptionWriteConfig::new(make_key_set("t"));
    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "t",
        icefalldb_schema,
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .unwrap();
    writer.insert_batch(make_batch()).await.unwrap();
    writer.commit().await.unwrap();
    storage
}

fn make_provider(table: &str) -> Arc<dyn KeyProvider> {
    let mut keys: BTreeMap<KeyIdentifier, Vec<u8>> = BTreeMap::new();
    keys.insert(
        KeyIdentifier::new(format!("{table}-v1")),
        FOOTER_KEY.to_vec(),
    );
    Arc::new(StaticKeyProvider::new(keys))
}

#[tokio::test]
async fn encrypted_table_count_and_projection() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_encrypted_table(tmp.path()).await;

    let provider = IcefallDBTableProvider::new_encrypted(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        },
        make_provider("t"),
        KeyIdentifier::new("t-v1"),
        BTreeMap::new(),
    )
    .await
    .expect("open encrypted provider");

    let ctx = icefalldb_encrypted_session(1, 1024, make_provider("t"));
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batches = ctx
        .sql("SELECT a, b FROM t ORDER BY a")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 6);

    let count: i64 = ctx
        .sql("SELECT COUNT(*) FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(count, 6);
}

#[tokio::test]
async fn encrypted_table_filter_pushdown() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_encrypted_table(tmp.path()).await;

    let provider = IcefallDBTableProvider::new_encrypted(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        },
        make_provider("t"),
        KeyIdentifier::new("t-v1"),
        BTreeMap::new(),
    )
    .await
    .unwrap();

    let ctx = icefalldb_encrypted_session(1, 1024, make_provider("t"));
    ctx.register_table("t", Arc::new(provider)).unwrap();

    let batches = ctx
        .sql("SELECT a FROM t WHERE a > 3")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2); // 5 and 6
}
