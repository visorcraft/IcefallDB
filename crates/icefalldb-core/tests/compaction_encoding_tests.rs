use arrow::array::{
    Int64Array, LargeStringArray, RecordBatch, StringArray, TimestampMicrosecondArray,
};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use icefalldb_core::compaction::{CompactionOptions, Compactor};
use icefalldb_core::gc::GarbageCollector;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use parquet::basic::{Compression, Encoding, ZstdLevel};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::sync::Arc;

mod common;

async fn compact_and_gc(storage: &dyn Storage, table: &str, force: bool) {
    let result = Compactor::with_options(
        storage,
        table,
        CompactionOptions {
            force,
            ..CompactionOptions::default()
        },
    )
    .compact()
    .await
    .unwrap();
    assert!(result.rewrote);

    GarbageCollector::new(storage, table, 1)
        .run()
        .await
        .unwrap();
}

async fn read_output_metadata(
    dir: &tempfile::TempDir,
    storage: &dyn Storage,
    table: &str,
) -> parquet::file::metadata::ParquetMetaData {
    let entries = storage.list(table).await.unwrap();
    let parquet_path = dir
        .path()
        .join(common::find_parquet_path(&entries).unwrap());
    let file = std::fs::File::open(&parquet_path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    reader.metadata().clone()
}

/// Return every column chunk metadata whose descriptor name matches `column`.
fn column_chunks_for<'a>(
    metadata: &'a parquet::file::metadata::ParquetMetaData,
    column: &str,
) -> Vec<&'a parquet::file::metadata::ColumnChunkMetaData> {
    metadata
        .row_groups()
        .iter()
        .flat_map(|rg| rg.columns())
        .filter(|col| col.column_descr().name() == column)
        .collect()
}

fn make_int_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column::new("id", "int64", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 128 * 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_timestamp_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column::new("ts", "timestamp", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 128 * 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_utf8_schema(column: &str) -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column::new(column, "utf8", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 128 * 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

#[tokio::test]
async fn test_compaction_avoids_delta_for_unsorted_ints() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_int_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "unsorted_ids", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    // Shuffled ids: not non-decreasing.
    let ids: Vec<i64> = vec![512, 1, 999, 42, 7, 123, 888, 3, 600, 10];
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(Int64Array::from(ids))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "unsorted_ids", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "unsorted_ids").await;
    let id_chunks = column_chunks_for(&metadata, "id");
    assert!(
        !id_chunks.is_empty(),
        "expected at least one id column chunk"
    );
    for col in id_chunks {
        assert!(
            !col.encodings().any(|e| e == Encoding::DELTA_BINARY_PACKED),
            "expected unsorted id column to avoid DELTA_BINARY_PACKED, got {:?}",
            col.encodings().collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn test_compaction_uses_delta_for_sorted_ints() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_int_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "sorted_ids", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(Int64Array::from((0..1_000).collect::<Vec<_>>()))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "sorted_ids", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "sorted_ids").await;
    let id_chunks = column_chunks_for(&metadata, "id");
    assert!(
        !id_chunks.is_empty(),
        "expected at least one id column chunk"
    );

    let expected_compression = Compression::ZSTD(ZstdLevel::try_new(1).unwrap());
    for col in id_chunks {
        assert_eq!(
            col.compression(),
            expected_compression,
            "expected ZSTD compression for id column chunk"
        );
        assert!(
            col.encodings().any(|e| e == Encoding::DELTA_BINARY_PACKED),
            "expected sorted id column to use DELTA_BINARY_PACKED encoding, got {:?}",
            col.encodings().collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn test_compaction_uses_delta_for_sorted_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_timestamp_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "sorted_ts", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let timestamps: Vec<i64> = (0..1_000).map(|i| i * 1_000_000).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(TimestampMicrosecondArray::from(timestamps))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "sorted_ts", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "sorted_ts").await;
    let ts_chunks = column_chunks_for(&metadata, "ts");
    assert!(
        !ts_chunks.is_empty(),
        "expected at least one ts column chunk"
    );
    for col in ts_chunks {
        assert!(
            col.encodings().any(|e| e == Encoding::DELTA_BINARY_PACKED),
            "expected sorted timestamp column to use DELTA_BINARY_PACKED, got {:?}",
            col.encodings().collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn test_compaction_avoids_delta_for_unsorted_timestamps() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_timestamp_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "unsorted_ts", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "ts",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    // Shuffled timestamps: not non-decreasing.
    let timestamps: Vec<i64> = vec![
        5_000_000, 1_000_000, 9_000_000, 2_000_000, 7_000_000, 3_000_000,
    ];
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(TimestampMicrosecondArray::from(timestamps))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "unsorted_ts", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "unsorted_ts").await;
    let ts_chunks = column_chunks_for(&metadata, "ts");
    assert!(
        !ts_chunks.is_empty(),
        "expected at least one ts column chunk"
    );
    for col in ts_chunks {
        assert!(
            !col.encodings().any(|e| e == Encoding::DELTA_BINARY_PACKED),
            "expected unsorted timestamp column to avoid DELTA_BINARY_PACKED, got {:?}",
            col.encodings().collect::<Vec<_>>()
        );
    }
}

#[tokio::test]
async fn test_compaction_uses_dictionary_for_low_cardinality_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_utf8_schema("category");
    let mut writer = Writer::create(Arc::clone(&storage), "low_card", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "category",
        DataType::Utf8,
        false,
    )]));
    let categories: Vec<&str> = (0..1_000)
        .map(|i| match i % 4 {
            0 => "electronics",
            1 => "clothing",
            2 => "grocery",
            _ => "books",
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(StringArray::from(categories))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "low_card", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "low_card").await;
    let cat_chunks = column_chunks_for(&metadata, "category");
    assert!(
        !cat_chunks.is_empty(),
        "expected at least one category column chunk"
    );

    // parquet-rs reports dictionary encoding either via the encodings list or by
    // setting a dictionary page offset. We accept either signal because the
    // exact representation can vary by writer version/path.
    let has_dictionary = cat_chunks.iter().any(|col| {
        col.dictionary_page_offset().is_some()
            || col.encodings().any(|e| e == Encoding::RLE_DICTIONARY)
            || col.encodings().any(|e| e == Encoding::PLAIN_DICTIONARY)
    });
    assert!(
        has_dictionary,
        "expected low-cardinality category column to use dictionary encoding"
    );
}

#[tokio::test]
async fn test_compaction_avoids_dictionary_for_high_cardinality_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_utf8_schema("name");
    let mut writer = Writer::create(Arc::clone(&storage), "high_card", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "name",
        DataType::Utf8,
        false,
    )]));
    let names: Vec<String> = (0..1_000).map(|i| format!("user-{}", i)).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(StringArray::from(
            names.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "high_card", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "high_card").await;
    let name_chunks = column_chunks_for(&metadata, "name");
    assert!(
        !name_chunks.is_empty(),
        "expected at least one name column chunk"
    );
    for col in name_chunks {
        assert!(
            !col.encodings().any(|e| e == Encoding::RLE_DICTIONARY),
            "expected high-cardinality name column to avoid RLE_DICTIONARY, got {:?}",
            col.encodings().collect::<Vec<_>>()
        );
        assert!(
            !col.encodings().any(|e| e == Encoding::PLAIN_DICTIONARY),
            "expected high-cardinality name column to avoid PLAIN_DICTIONARY, got {:?}",
            col.encodings().collect::<Vec<_>>()
        );
        assert!(
            col.dictionary_page_offset().is_none(),
            "expected high-cardinality name column to have no dictionary page"
        );
    }
}

#[tokio::test]
async fn test_compaction_uses_dictionary_for_low_cardinality_large_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = Schema {
        schema_id: 1,
        columns: vec![Column::new("category", "large_utf8", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 128 * 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    let mut writer = Writer::create(Arc::clone(&storage), "low_card_large", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "category",
        DataType::LargeUtf8,
        false,
    )]));
    let categories: Vec<&str> = (0..1_000)
        .map(|i| match i % 4 {
            0 => "electronics",
            1 => "clothing",
            2 => "grocery",
            _ => "books",
        })
        .collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(LargeStringArray::from(categories))],
    )
    .unwrap();
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    compact_and_gc(storage.as_ref(), "low_card_large", true).await;

    let metadata = read_output_metadata(&dir, storage.as_ref(), "low_card_large").await;
    let cat_chunks = column_chunks_for(&metadata, "category");
    assert!(
        !cat_chunks.is_empty(),
        "expected at least one category column chunk"
    );

    let has_dictionary = cat_chunks.iter().any(|col| {
        col.dictionary_page_offset().is_some()
            || col.encodings().any(|e| e == Encoding::RLE_DICTIONARY)
            || col.encodings().any(|e| e == Encoding::PLAIN_DICTIONARY)
    });
    assert!(
        has_dictionary,
        "expected low-cardinality LargeUtf8 category column to use dictionary encoding"
    );
}
