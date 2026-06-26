use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use futures::stream::StreamExt;
use icefalldb_core::metadata::{Column, Manifest, RowGroupMeta, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::{CompactionOptions, Compactor, Reader};
use std::sync::Arc;

fn make_partitioned_schema(target_rows: usize) -> Schema {
    Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 1,
            },
            Column {
                name: "region".into(),
                r#type: "utf8".into(),
                nullable: false,
                field_id: 2,
            },
        ],
        partition_by: Some(vec!["region".into()]),
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 2,
        dropped_columns: vec![],
    }
}

fn make_partitioned_batch(ids: Vec<i64>, regions: Vec<&str>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
    ]);
    let ids = Int64Array::from(ids);
    let regions = StringArray::from(regions);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids), Arc::new(regions)]).unwrap()
}

async fn read_latest_manifest(storage: &Arc<dyn Storage>, table: &str) -> Manifest {
    let pointer_data = storage
        .read(&format!("{}/_manifest.json", table))
        .await
        .unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    let seq = pointer["latest"].as_u64().unwrap();
    let manifest_data = storage
        .read(&format!("{}/{}", table, Manifest::filename(seq)))
        .await
        .unwrap();
    serde_json::from_slice(&manifest_data).unwrap()
}

#[tokio::test]
async fn test_writer_splits_rows_by_partition_value() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_partitioned_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    // Mix of two regions, small enough that without splitting it would form a
    // single row group.
    writer
        .insert_batch(make_partitioned_batch(
            vec![1, 2, 3, 4],
            vec!["east", "west", "east", "west"],
        ))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 2);

    let mut seen_regions = Vec::new();
    for entry in &manifest.row_groups {
        let meta: RowGroupMeta = serde_json::from_slice(
            &storage
                .read(&format!("products/{}", entry.meta))
                .await
                .unwrap(),
        )
        .unwrap();
        let partition_values = manifest
            .partition_values
            .as_ref()
            .and_then(|m| m.get(&entry.data))
            .expect("each homogeneous row group must have partition values");
        let region = partition_values
            .get("region")
            .expect("region partition value must be present")
            .as_str()
            .unwrap();
        seen_regions.push(region.to_string());
        assert_eq!(meta.rows, 2);
        assert!(
            meta.column_offsets.is_some(),
            "sidecar column offsets must be present"
        );
    }
    seen_regions.sort_unstable();
    assert_eq!(seen_regions, vec!["east", "west"]);
}

#[tokio::test]
async fn test_writer_null_partition_values_form_own_group() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 1,
            },
            Column {
                name: "region".into(),
                r#type: "utf8".into(),
                nullable: true,
                field_id: 2,
            },
        ],
        partition_by: Some(vec!["region".into()]),
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 2,
        dropped_columns: vec![],
    };
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, true),
    ]));
    let ids = Int64Array::from(vec![1, 2, 3]);
    let regions = StringArray::from(vec![Some("east"), None, Some("west")]);
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![Arc::new(ids), Arc::new(regions)],
    )
    .unwrap();

    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 3);

    let mut null_count = 0;
    for entry in &manifest.row_groups {
        if let Some(values) = manifest
            .partition_values
            .as_ref()
            .and_then(|m| m.get(&entry.data))
        {
            assert!(values.get("region").is_some());
        } else {
            null_count += 1;
        }
    }
    assert_eq!(
        null_count, 1,
        "exactly one row group should have no partition values (the NULL group)"
    );
}

#[tokio::test]
async fn test_compaction_preserves_partition_values() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_partitioned_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    // Two commits, each spanning both regions.
    writer
        .insert_batch(make_partitioned_batch(vec![1, 2], vec!["east", "west"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_partitioned_batch(vec![3, 4], vec!["east", "west"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest_before = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest_before.row_groups.len(), 4);

    let compactor = Compactor::with_options(
        storage.as_ref(),
        "products",
        CompactionOptions {
            target_row_group_rows: 100,
            target_row_group_bytes: 128 * 1024 * 1024,
            lock_timeout: std::time::Duration::from_secs(30),
            force: true,
            sort_keys: vec![],
        },
    );
    compactor.compact().await.unwrap();

    let manifest_after = read_latest_manifest(&storage, "products").await;

    let mut seen_regions = Vec::new();
    for entry in &manifest_after.row_groups {
        let values = manifest_after
            .partition_values
            .as_ref()
            .and_then(|m| m.get(&entry.data))
            .expect("compacted row groups must retain partition values");
        let region = values
            .get("region")
            .expect("region must be present")
            .as_str()
            .unwrap();
        seen_regions.push(region.to_string());
    }
    seen_regions.sort_unstable();
    assert_eq!(seen_regions, vec!["east", "west"]);

    // Verify the table still contains all rows in the right partition.
    let reader = Reader::new(storage.as_ref(), "products").await.unwrap();
    let scan = reader.scan().await.unwrap();
    let mut total_rows = 0;
    for prg in &scan.row_groups {
        let mut stream = reader.read_row_group(prg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            total_rows += batch.num_rows();
        }
    }
    assert_eq!(total_rows, 4);
}

fn make_mixed_parquet(path: &std::path::Path) {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("region", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["east", "west"])),
        ],
    )
    .unwrap();
    let mut writer = parquet::arrow::ArrowWriter::try_new(
        std::fs::File::create(path).unwrap(),
        arrow_schema,
        None,
    )
    .unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

#[tokio::test]
async fn test_compaction_omits_partition_values_for_mixed_group() {
    let dir = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());
    let parquet_path = dir.path().join("mixed.parquet");
    make_mixed_parquet(&parquet_path);

    let schema = make_partitioned_schema(10);
    let mut writer = Writer::create(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();
    let outcome = writer
        .insert_parquet(parquet_path.to_str().unwrap())
        .await
        .unwrap();
    assert_eq!(
        outcome,
        icefalldb_core::writer::InsertParquetOutcome::FastPath { rows: 2 }
    );

    let compactor = Compactor::with_options(
        storage.as_ref(),
        "products",
        CompactionOptions {
            target_row_group_rows: 10,
            target_row_group_bytes: 1024 * 1024,
            lock_timeout: std::time::Duration::from_secs(30),
            force: true,
            sort_keys: vec![],
        },
    );
    compactor.compact().await.unwrap();

    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(
        manifest.partition_values, None,
        "a compacted heterogeneous group must not claim partition values"
    );
}
