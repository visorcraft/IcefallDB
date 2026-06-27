//! Round-trip tests for Parquet Modular Encryption in `icefalldb-core`.
//!
//! These tests are only compiled when the `encryption` feature is enabled:
//! ```sh
//! cargo test -p icefalldb-core --features encryption --test encryption_tests
//! ```

#![cfg(feature = "encryption")]

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use icefalldb_core::check::{CheckOptions, Severity};
use icefalldb_core::encryption::{
    build_decryption_properties, table_aad_prefix, EncryptionKeySet, EncryptionWriteConfig,
    SchemaEncryptionMarker, StaticKeyProvider,
};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{Writer, WriterOptionsFull};
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

fn write_plain_parquet_file(dir: &TempDir, batch: &RecordBatch) -> std::path::PathBuf {
    let path = dir.path().join("plain.parquet");
    let file = std::fs::File::create(&path).unwrap();
    let mut writer =
        ArrowWriter::try_new(file, batch.schema(), None).expect("create plain parquet writer");
    writer.write(batch).expect("write plain batch");
    writer.close().expect("close plain writer");
    path
}

fn make_two_col_batch() -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("ssn", DataType::Utf8, false),
    ]);
    let ids = Int64Array::from(vec![1i64, 2, 3, 4, 5]);
    let ssn = StringArray::from(vec!["111-1", "222-2", "333-3", "444-4", "555-5"]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids), Arc::new(ssn)]).unwrap()
}

fn make_two_col_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "ssn".into(),
                r#type: "utf8".into(),
                nullable: false,
                field_id: 1,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn sample_key_set(footer_key: &[u8]) -> EncryptionKeySet {
    let mut cols = BTreeMap::new();
    cols.insert("ssn".to_string(), footer_key.to_vec());
    EncryptionKeySet::with_columns(footer_key.to_vec(), cols, table_aad_prefix("events", 1))
        .unwrap()
}

const FOOTER_KEY: &[u8; 16] = b"0123456789abcdef";
const WRONG_KEY: &[u8; 16] = b"abcdef0123456789";

#[tokio::test]
async fn write_encrypted_round_trip() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let keys = sample_key_set(FOOTER_KEY);
    let cfg = EncryptionWriteConfig::new(keys);

    // Create an encrypted table and insert a row group.
    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg.clone()),
    )
    .await
    .expect("create encrypted writer");
    writer
        .insert_batch(make_two_col_batch())
        .await
        .expect("insert");
    writer.commit().await.expect("commit");

    // The data file is at events/<rg-id>.parquet. List the table root.
    let entries = storage.list("events").await.expect("list");
    let parquet_path = entries
        .iter()
        .find(|p| p.ends_with(".parquet"))
        .expect("parquet file exists")
        .clone();
    let bytes = storage.read(&parquet_path).await.expect("read");

    // Reading the encrypted file WITHOUT keys: opening the file succeeds
    // because `plaintext_footer = true` leaves the footer unencrypted (this
    // is the perf-preserving mode that keeps page-index reads working). But
    // actually iterating the reader — which decrypts column data — fails.
    let no_key_reader = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes.clone()))
        .expect("opening plaintext-footer file without keys succeeds");
    let no_key_iter = no_key_reader.build().expect("build reader struct");
    let read_result: Result<Vec<RecordBatch>, _> = no_key_iter.collect();
    assert!(
        read_result.is_err(),
        "reading encrypted column data without keys should fail"
    );

    // Reading WITH the correct keys returns the original data.
    let dec = build_decryption_properties(&cfg.keys).expect("build decryption");
    let opts = ArrowReaderOptions::new().with_file_decryption_properties(dec);
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(Bytes::from(bytes), opts)
        .expect("open with keys");
    let reader = builder.build().expect("build reader");
    let read: Vec<RecordBatch> = reader.collect::<Result<_, _>>().unwrap();
    let total_rows: usize = read.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 5);
}

#[tokio::test]
async fn rejects_encrypted_partition_column() {
    // Partition values are stored in plaintext (for pruning), so partitioning by
    // an encrypted column would leak it. Creating such a table must be rejected.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut schema = make_two_col_schema();
    schema.partition_by = Some(vec!["ssn".to_string()]);
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let result = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        schema,
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await;
    let msg = match result {
        Ok(_) => panic!("partitioning by an encrypted column must be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("partition column") && msg.contains("cannot be encrypted"),
        "unexpected error: {msg}"
    );
}

#[tokio::test]
async fn write_encrypted_wrong_key_fails() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let keys = sample_key_set(FOOTER_KEY);
    let cfg = EncryptionWriteConfig::new(keys);

    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create");
    writer.insert_batch(make_two_col_batch()).await.unwrap();
    writer.commit().await.unwrap();

    let entries = storage.list("events").await.unwrap();
    let parquet_path = entries
        .iter()
        .find(|p| p.ends_with(".parquet"))
        .unwrap()
        .clone();
    let bytes = storage.read(&parquet_path).await.unwrap();

    // Build decryption properties with the WRONG key.
    let wrong_keys = sample_key_set(WRONG_KEY);
    let dec = build_decryption_properties(&wrong_keys).expect("build decryption");
    let opts = ArrowReaderOptions::new().with_file_decryption_properties(dec);
    let result = ParquetRecordBatchReaderBuilder::try_new_with_options(Bytes::from(bytes), opts);
    assert!(
        result.is_err(),
        "decoding with the wrong key should fail authentication"
    );
}

#[tokio::test]
async fn encryption_marker_is_written() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let keys = sample_key_set(FOOTER_KEY);
    let cfg = EncryptionWriteConfig::new(keys);

    let _writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create");

    let marker_bytes = storage
        .read("events/_encryption.json")
        .await
        .expect("read marker");
    let marker: SchemaEncryptionMarker =
        serde_json::from_slice(&marker_bytes).expect("parse marker");
    assert_eq!(marker.algorithm, SchemaEncryptionMarker::ALGORITHM);
    assert_eq!(marker.footer_key_id, "events-v1");
    assert!(marker.plaintext_footer);
    assert!(marker.aad_prefix.is_some());
    assert_eq!(
        marker.column_key_ids.get("ssn").map(|s| s.as_str()),
        Some("events-v1:ssn")
    );
}

#[tokio::test]
async fn no_encryption_marker_when_encryption_off() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let _writer = Writer::create(Arc::clone(&storage), "plain", make_two_col_schema())
        .await
        .expect("create");

    // No _encryption.json should exist for plaintext tables.
    let result = storage.exists("plain/_encryption.json").await.unwrap();
    assert!(
        !result,
        "plaintext tables must not have an _encryption.json marker"
    );
}

#[tokio::test]
async fn encrypted_bytes_differ_from_plaintext() {
    // Write the same batch twice: once plaintext, once encrypted.
    let plain_storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut plain = Writer::create(Arc::clone(&plain_storage), "plain", make_two_col_schema())
        .await
        .unwrap();
    plain.insert_batch(make_two_col_batch()).await.unwrap();
    plain.commit().await.unwrap();

    let enc_storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let mut enc = Writer::create_with_full(
        Arc::clone(&enc_storage),
        "enc",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .unwrap();
    enc.insert_batch(make_two_col_batch()).await.unwrap();
    enc.commit().await.unwrap();

    let plain_bytes = plain_storage
        .read(
            plain_storage
                .list("plain")
                .await
                .unwrap()
                .iter()
                .find(|p| p.ends_with(".parquet"))
                .unwrap(),
        )
        .await
        .unwrap();
    let enc_bytes = enc_storage
        .read(
            enc_storage
                .list("enc")
                .await
                .unwrap()
                .iter()
                .find(|p| p.ends_with(".parquet"))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_ne!(
        plain_bytes, enc_bytes,
        "encrypted bytes must differ from plaintext bytes"
    );
    // Sanity: both should be at least 4 bytes (PAR1 magic) — encrypted files
    // are still valid Parquet at the magic-byte level.
    assert!(plain_bytes.len() > 4);
    assert!(enc_bytes.len() > 4);
}

#[tokio::test]
async fn insert_parquet_fast_path_is_disabled_when_encrypted() {
    // The zero-copy fast path would copy plaintext Parquet bytes into an
    // encrypted table, which is a security hole. Verify it's disabled.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create");

    let tmp = TempDir::new().unwrap();
    let path = write_plain_parquet_file(&tmp, &make_two_col_batch());

    use icefalldb_core::writer::InsertParquetOutcome;
    let outcome = writer.insert_parquet(path.to_str().unwrap()).await.unwrap();
    assert_eq!(
        outcome,
        InsertParquetOutcome::Incompatible,
        "fast path must be disabled for encrypted writers"
    );
}

#[tokio::test]
async fn opening_encrypted_table_without_keys_is_rejected() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    // Create the encrypted table.
    let _writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create");

    // Reopen without encryption options — must error.
    let result = Writer::new_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new(),
    )
    .await;
    let err = match result {
        Ok(_) => panic!("expected error: encrypted table opened without keys"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("encrypted") && msg.contains("_encryption.json"),
        "expected encryption-mismatch error, got: {msg}"
    );
}

#[tokio::test]
async fn opening_plain_table_with_keys_is_rejected() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    // Create a plaintext table.
    let _writer = Writer::create(Arc::clone(&storage), "plain", make_two_col_schema())
        .await
        .unwrap();

    // Try to reopen with encryption options — must error.
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let result = Writer::new_with_full(
        Arc::clone(&storage),
        "plain",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await;
    let err = match result {
        Ok(_) => panic!("expected error: plaintext table opened with keys"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("not encrypted"),
        "expected 'not encrypted' error, got: {msg}"
    );
}

#[tokio::test]
async fn reopening_encrypted_table_with_mismatched_plaintext_footer_is_rejected() {
    // Regression: an encrypted table's `_encryption.json`
    // marker pins the encryption parameters. A reopen that supplies a
    // different plaintext-footer setting must fail — otherwise new row groups
    // would be undecryptable by readers that follow the marker.
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let _writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create");

    // Reopen with plaintext_footer = false — must fail.
    let cfg_mismatched =
        EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY)).with_plaintext_footer(false);
    let result = Writer::new_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg_mismatched),
    )
    .await;
    let err = match result {
        Ok(_) => panic!("expected error: mismatched encryption config was accepted"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("does not match the marker"),
        "expected marker-mismatch error, got: {msg}"
    );
}

#[tokio::test]
async fn reopening_encrypted_table_with_matching_config_succeeds() {
    // Positive control: reopening with the SAME config must succeed (and
    // further commits must work).
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create");
    writer.insert_batch(make_two_col_batch()).await.unwrap();
    writer.commit().await.unwrap();

    let cfg2 = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));
    let mut writer2 = Writer::new_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg2),
    )
    .await
    .expect("reopen with matching config");
    writer2.insert_batch(make_two_col_batch()).await.unwrap();
    writer2.commit().await.expect("second commit");
}

#[tokio::test]
async fn check_encrypted_table_with_key_passes() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let keys = sample_key_set(FOOTER_KEY);
    let cfg = EncryptionWriteConfig::new(keys.clone());

    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg.clone()),
    )
    .await
    .expect("create encrypted writer");
    writer
        .insert_batch(make_two_col_batch())
        .await
        .expect("insert");
    writer.commit().await.expect("commit");

    let provider = Arc::new(StaticKeyProvider::from_key_set("events-v1", &cfg.keys));
    let checker = icefalldb_core::Checker::new_with_options(
        storage.as_ref(),
        "events",
        CheckOptions::new().with_key_provider(provider),
    );
    let result = checker.check().await.unwrap();
    assert!(result.passed, "unexpected issues: {:?}", result.issues);
    assert!(
        !result
            .issues
            .iter()
            .any(|i| i.code == "ENCRYPTION_KEY_UNAVAILABLE"),
        "should not skip data-page validation when keys are supplied"
    );
}

#[tokio::test]
async fn check_encrypted_table_without_key_skips_data_pages() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let cfg = EncryptionWriteConfig::new(sample_key_set(FOOTER_KEY));

    let mut writer = Writer::create_with_full(
        Arc::clone(&storage),
        "events",
        make_two_col_schema(),
        WriterOptionsFull::new().with_encryption(cfg),
    )
    .await
    .expect("create encrypted writer");
    writer
        .insert_batch(make_two_col_batch())
        .await
        .expect("insert");
    writer.commit().await.expect("commit");

    let checker = icefalldb_core::Checker::new(storage.as_ref(), "events");
    let result = checker.check().await.unwrap();
    assert!(result.passed, "unexpected issues: {:?}", result.issues);
    let issue = result
        .issues
        .iter()
        .find(|i| i.code == "ENCRYPTION_KEY_UNAVAILABLE")
        .expect("expected ENCRYPTION_KEY_UNAVAILABLE info issue");
    assert_eq!(issue.severity, Severity::Info);
    assert!(
        issue.message.contains("data-page validation skipped"),
        "message: {}",
        issue.message
    );
}
