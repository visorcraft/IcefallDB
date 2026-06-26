use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use futures::StreamExt;
use icefalldb_core::compaction::Compactor;
use icefalldb_core::gc::GarbageCollector;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::Reader;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::sync::Arc;

mod common;

fn make_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column::new("id", "int64", false),
            Column::new("value", "float64", true),
            Column::new("name", "utf8", true),
        ],
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

fn make_arrow_schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
        Field::new("name", DataType::Utf8, true),
    ]))
}

fn make_batch(ids: Vec<i64>, values: Vec<Option<f64>>, names: Vec<Option<&str>>) -> RecordBatch {
    RecordBatch::try_new(
        make_arrow_schema(),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

fn count_parquet_files(entries: &[String]) -> usize {
    entries
        .iter()
        .filter(|e| {
            let name = std::path::Path::new(e)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            name.ends_with(".parquet")
        })
        .count()
}

#[tokio::test]
async fn test_optimize_compacts_and_cleans_up_old_files() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let file1 = dir.path().join("input1.parquet");
    let batch1 = make_batch(
        (0..100).collect(),
        (0..100).map(|i| Some(i as f64)).collect(),
        (0..100).map(|_| Some("a")).collect(),
    );
    common::write_snappy_parquet(&file1, &batch1);
    writer
        .insert_parquet(file1.to_str().unwrap())
        .await
        .unwrap();

    let file2 = dir.path().join("input2.parquet");
    let batch2 = make_batch(
        (100..200).collect(),
        (100..200).map(|i| Some(i as f64)).collect(),
        (100..200).map(|_| Some("b")).collect(),
    );
    common::write_snappy_parquet(&file2, &batch2);
    writer
        .insert_parquet(file2.to_str().unwrap())
        .await
        .unwrap();

    let result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();
    assert_eq!(result.input_row_groups, 2);
    assert_eq!(result.output_row_groups, 1);
    assert_eq!(result.input_rows, 200);
    assert_eq!(result.output_rows, 200);
    assert!(result.rewrote);

    GarbageCollector::new(storage.as_ref(), "products", 1)
        .run()
        .await
        .unwrap();

    let entries = storage.list("products").await.unwrap();
    let parquet_count = count_parquet_files(&entries);
    assert_eq!(
        parquet_count, 1,
        "expected one parquet file after optimize, found {}",
        parquet_count
    );

    // Verify the remaining parquet file uses ZSTD compression for every column chunk.
    let parquet_path = dir
        .path()
        .join(common::find_parquet_path(&entries).unwrap());
    let file = std::fs::File::open(&parquet_path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let metadata = reader.metadata();
    let expected_compression = Compression::ZSTD(ZstdLevel::try_new(1).unwrap());
    for rg in metadata.row_groups() {
        for col in rg.columns() {
            assert_eq!(
                col.compression(),
                expected_compression,
                "expected ZSTD compression for column chunk"
            );
        }
    }

    // Verify all 200 rows round-trip through the IcefallDB Reader with expected values.
    let reader = Reader::new(storage.as_ref(), "products").await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut read_batches = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            read_batches.push(batch.unwrap());
        }
    }
    let actual = arrow::compute::concat_batches(&make_arrow_schema(), &read_batches).unwrap();
    assert_eq!(actual.num_rows(), 200, "expected 200 rows after optimize");

    let ids = actual
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let values = actual
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let names = actual
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    for i in 0..200 {
        assert_eq!(ids.value(i), i as i64);
        assert_eq!(values.value(i), i as f64);
        let expected_name = if i < 100 { "a" } else { "b" };
        assert_eq!(names.value(i), expected_name);
    }
}

#[tokio::test]
async fn test_optimize_retains_requested_snapshots() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let file1 = dir.path().join("input1.parquet");
    let batch1 = make_batch(
        (0..100).collect(),
        (0..100).map(|i| Some(i as f64)).collect(),
        (0..100).map(|_| Some("a")).collect(),
    );
    common::write_snappy_parquet(&file1, &batch1);
    writer
        .insert_parquet(file1.to_str().unwrap())
        .await
        .unwrap();

    let file2 = dir.path().join("input2.parquet");
    let batch2 = make_batch(
        (100..200).collect(),
        (100..200).map(|i| Some(i as f64)).collect(),
        (100..200).map(|_| Some("b")).collect(),
    );
    common::write_snappy_parquet(&file2, &batch2);
    writer
        .insert_parquet(file2.to_str().unwrap())
        .await
        .unwrap();

    let compaction_result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();
    assert_eq!(compaction_result.input_row_groups, 2);
    assert_eq!(compaction_result.output_row_groups, 1);

    let gc_result = GarbageCollector::new(storage.as_ref(), "products", 2)
        .run()
        .await
        .unwrap();

    assert_eq!(
        gc_result.retained_snapshots.len(),
        2,
        "expected two snapshots to be retained, got {:?}",
        gc_result.retained_snapshots
    );

    let manifests_dir = dir.path().join("products").join("_manifests");
    let manifest_count = std::fs::read_dir(&manifests_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_str()
                .map(|n| n.ends_with(".json"))
                .unwrap_or(false)
        })
        .count();
    assert!(
        manifest_count >= 2,
        "expected at least two manifest files to remain, found {}",
        manifest_count
    );
}

#[tokio::test]
async fn test_optimize_sorts_output_row_groups() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    // Intentionally unsorted ids.
    let ids = vec![5i64, 3, 9, 1, 7, 2, 8, 4, 6, 0];
    let file = dir.path().join("unsorted.parquet");
    let batch = make_batch(
        ids.clone(),
        ids.iter().map(|&i| Some(i as f64)).collect(),
        ids.iter().map(|_| Some("name")).collect(),
    );
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    let result = Compactor::with_options(
        storage.as_ref(),
        "products",
        icefalldb_core::CompactionOptions {
            sort_keys: vec!["id".to_string()],
            force: true,
            ..icefalldb_core::CompactionOptions::default()
        },
    )
    .compact()
    .await
    .unwrap();
    assert!(result.rewrote);

    let reader = Reader::new(storage.as_ref(), "products").await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut read_batches = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            read_batches.push(batch.unwrap());
        }
    }
    let actual = arrow::compute::concat_batches(&make_arrow_schema(), &read_batches).unwrap();
    assert_eq!(actual.num_rows(), 10);

    let id_col = actual
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let expected: Vec<i64> = (0..10).collect();
    let actual_ids: Vec<i64> = id_col.iter().map(|v| v.unwrap()).collect();
    assert_eq!(
        actual_ids, expected,
        "rows should be sorted ascending by id"
    );
}

#[tokio::test]
async fn test_optimize_missing_sort_key_returns_error() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let file = dir.path().join("input.parquet");
    let batch = make_batch(
        (0..10).collect(),
        (0..10).map(|i| Some(i as f64)).collect(),
        (0..10).map(|_| Some("a")).collect(),
    );
    common::write_snappy_parquet(&file, &batch);
    writer.insert_parquet(file.to_str().unwrap()).await.unwrap();

    let result = Compactor::with_options(
        storage.as_ref(),
        "products",
        icefalldb_core::CompactionOptions {
            sort_keys: vec!["does_not_exist".to_string()],
            force: true,
            ..icefalldb_core::CompactionOptions::default()
        },
    )
    .compact()
    .await;

    assert!(
        result.is_err(),
        "compaction with a missing sort key should fail"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("does_not_exist"),
        "error should name the missing key: {}",
        err_msg
    );
}

#[tokio::test]
async fn test_optimize_forces_rewrite_of_single_row_group() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());

    let schema = make_schema();
    let mut writer = Writer::create(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let file1 = dir.path().join("input1.parquet");
    let batch1 = make_batch(
        (0..100).collect(),
        (0..100).map(|i| Some(i as f64)).collect(),
        (0..100).map(|_| Some("a")).collect(),
    );
    common::write_snappy_parquet(&file1, &batch1);
    writer
        .insert_parquet(file1.to_str().unwrap())
        .await
        .unwrap();

    let result = Compactor::with_options(
        storage.as_ref(),
        "products",
        icefalldb_core::CompactionOptions {
            force: true,
            ..icefalldb_core::CompactionOptions::default()
        },
    )
    .compact()
    .await
    .unwrap();
    assert_eq!(result.input_row_groups, 1);
    assert_eq!(result.output_row_groups, 1);
    assert!(result.rewrote);

    GarbageCollector::new(storage.as_ref(), "products", 1)
        .run()
        .await
        .unwrap();

    let entries = storage.list("products").await.unwrap();
    let parquet_count = count_parquet_files(&entries);
    assert_eq!(
        parquet_count, 1,
        "expected one parquet file after forced optimize, found {}",
        parquet_count
    );

    // Verify the remaining parquet file uses ZSTD compression for every column chunk.
    let parquet_path = dir
        .path()
        .join(common::find_parquet_path(&entries).unwrap());
    let file = std::fs::File::open(&parquet_path).unwrap();
    let reader = SerializedFileReader::new(file).unwrap();
    let metadata = reader.metadata();
    let expected_compression = Compression::ZSTD(ZstdLevel::try_new(1).unwrap());
    for rg in metadata.row_groups() {
        for col in rg.columns() {
            assert_eq!(
                col.compression(),
                expected_compression,
                "expected ZSTD compression for column chunk"
            );
        }
    }
}
