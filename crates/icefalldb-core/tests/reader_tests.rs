use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use futures::StreamExt;
use icefalldb_core::meta_cache::MetaCache;
use icefalldb_core::metadata::{Column, ColumnStats, RowGroupMeta, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{
    IcefallDBError, Literal, PlannedRowGroup, Predicate, Reader, ScanPlan, Writer,
};
use std::sync::Arc;

fn make_int_schema(partition_by: Option<Vec<String>>) -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_int_batch(col: &str, ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new(col, DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn make_partition_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "part".into(),
                r#type: "string".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "value".into(),
                r#type: "int64".into(),
                nullable: true,
                field_id: 0,
            },
        ],
        partition_by: Some(vec!["part".into()]),
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_partition_batch(part: &str, values: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("part", DataType::Utf8, false),
        Field::new("value", DataType::Int64, true),
    ]);
    let part_array = StringArray::from(vec![part; values.len()]);
    let value_array = Int64Array::from(values);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(part_array), Arc::new(value_array)],
    )
    .unwrap()
}

#[tokio::test]
async fn test_reader_empty_table_fails() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let result = Reader::new(&*storage, "empty").await;
    assert!(
        matches!(result, Err(IcefallDBError::EmptyTable(ref t)) if t == "empty"),
        "expected EmptyTable for empty table, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_reader_rejects_partial_table_with_schema_but_no_manifest() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";

    storage
        .write(&format!("{}/_schema.json", table), br#"{"latest":1}"#)
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/{}", table, Schema::filename(1)),
            &serde_json::to_vec(&make_int_schema(None)).unwrap(),
        )
        .await
        .unwrap();

    let result = Reader::new(&*storage, table).await;
    assert!(
        matches!(result, Err(IcefallDBError::MissingManifestPointer { .. })),
        "expected MissingManifestPointer for partial table, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_reader_one_row_group() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.table, table);
    assert_eq!(plan.schema.schema_id, schema.schema_id);
    assert_eq!(plan.row_groups.len(), 1);
    assert_eq!(plan.row_groups[0].meta.rows, 3);

    let mut stream = reader.read_row_group(&plan.row_groups[0]).await.unwrap();
    let batch = stream.next().await.unwrap().unwrap();
    assert_eq!(batch.num_rows(), 3);
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn test_reader_multiple_row_groups() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let mut schema = make_int_schema(None);
    schema.row_group_target_rows = 2;

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3, 4, 5]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert!(plan.row_groups.len() > 1, "expected multiple row groups");

    let mut total_rows = 0;
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            total_rows += batch.unwrap().num_rows();
        }
    }
    assert_eq!(total_rows, 5);
}

#[tokio::test]
async fn test_reader_prunes_by_partition_values() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "events";
    let schema = make_partition_schema();

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_partition_batch("a", vec![1, 2]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 1);
    assert_eq!(
        plan.row_groups[0]
            .partition_values
            .as_ref()
            .unwrap()
            .get("part")
            .unwrap()
            .as_str(),
        Some("a")
    );

    let pruned = plan
        .prune(&[Predicate::Eq {
            column: "part".into(),
            value: Literal::String("b".into()),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());
}

#[tokio::test]
async fn test_reader_prunes_by_column_stats() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 1);

    let pruned = plan
        .prune(&[Predicate::Gt {
            column: "id".into(),
            value: Literal::Int64(5),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());
}

#[tokio::test]
async fn test_reader_checksum_mismatch_on_corrupt_parquet() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    let rg = &plan.row_groups[0];

    let mut bytes = storage.read(&rg.data_path).await.unwrap();
    let last = bytes.len() - 1;
    bytes[last] = bytes[last].wrapping_add(1);
    storage.write(&rg.data_path, &bytes).await.unwrap();

    let result = reader.read_row_group(rg).await;
    assert!(
        matches!(result, Err(IcefallDBError::RowGroupChecksumMismatch { ref path }) if path == &rg.data_path),
        "expected RowGroupChecksumMismatch for corrupt parquet, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_reader_meta_checksum_mismatch() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    let rg = &plan.row_groups[0];

    // Deliberately corrupt the sidecar in-place to test error detection.
    // This violates the write-once immutability invariant that the cache
    // relies on, so we must evict the now-stale cached entry first.
    MetaCache::global().clear();

    // Remove the snapshot checkpoint so the scan must read and validate the
    // per-fragment .meta sidecar.
    let manifest = reader.catalog().latest_manifest().unwrap();
    if let Some(cp_rel) = &manifest.checkpoint {
        storage
            .delete(&format!("{}/{}", table, cp_rel))
            .await
            .unwrap();
    }

    let mut meta: serde_json::Value =
        serde_json::from_slice(&storage.read(&rg.meta_path).await.unwrap()).unwrap();
    meta.as_object_mut().unwrap().insert(
        "meta_checksum".into(),
        "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
    );
    storage
        .write(
            &rg.meta_path,
            serde_json::to_vec_pretty(&meta).unwrap().as_slice(),
        )
        .await
        .unwrap();

    let result = reader.scan().await;
    assert!(
        matches!(result, Err(IcefallDBError::RowGroupChecksumMismatch { ref path }) if path == &rg.meta_path),
        "expected RowGroupChecksumMismatch for corrupt meta checksum, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_reader_refresh_picks_up_new_commit() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let mut reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 1);

    writer
        .insert_batch(make_int_batch("id", vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    reader.refresh().await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 2);
}

#[tokio::test]
async fn test_reader_schema_mismatch() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    let rg = &plan.row_groups[0];

    // Deliberately rewrite the sidecar in-place to simulate a schema_id change
    // (a violation of the write-once invariant in production). Evict the cached
    // entry so the reader re-reads the tampered file and detects the mismatch.
    MetaCache::global().clear();

    // Remove the snapshot checkpoint so the scan must read and validate the
    // per-fragment .meta sidecar.
    let manifest = reader.catalog().latest_manifest().unwrap();
    if let Some(cp_rel) = &manifest.checkpoint {
        storage
            .delete(&format!("{}/{}", table, cp_rel))
            .await
            .unwrap();
    }

    let mut meta: RowGroupMeta =
        serde_json::from_slice(&storage.read(&rg.meta_path).await.unwrap()).unwrap();
    meta.schema_id = 999;
    meta.compute_meta_checksum().unwrap();
    storage
        .write(
            &rg.meta_path,
            serde_json::to_vec_pretty(&meta).unwrap().as_slice(),
        )
        .await
        .unwrap();

    let result = reader.scan().await;
    assert!(
        matches!(
            result,
            Err(IcefallDBError::SchemaMismatch {
                ref column,
                ref expected,
                ref path,
            })
            if column == "schema_id"
                && expected == &schema.schema_id.to_string()
                && path == &rg.meta_path
        ),
        "expected SchemaMismatch with column, expected, path, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_reader_missing_row_group_file() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    let rg = &plan.row_groups[0];
    let snapshot = rg.snapshot;

    // Missing .meta file is detected during scan. Evict the cached entry so
    // the reader re-issues the read and observes the deletion (this test
    // deliberately violates the write-once invariant to exercise error paths).
    MetaCache::global().clear();

    // Remove the snapshot checkpoint so the scan must read the per-fragment
    // .meta sidecar and observe that it is missing.
    let manifest = reader.catalog().latest_manifest().unwrap();
    if let Some(cp_rel) = &manifest.checkpoint {
        storage
            .delete(&format!("{}/{}", table, cp_rel))
            .await
            .unwrap();
    }

    let original_meta = storage.read(&rg.meta_path).await.unwrap();
    storage.delete(&rg.meta_path).await.unwrap();
    let result = reader.scan().await;
    assert!(
        matches!(
            result,
            Err(IcefallDBError::MissingRowGroupFile {
                snapshot: s,
                ref path,
            })
            if s == snapshot && path == &rg.meta_path
        ),
        "expected MissingRowGroupFile for deleted meta, got {:?}",
        result
    );

    // Restore the meta file and delete the Parquet file instead.
    storage.write(&rg.meta_path, &original_meta).await.unwrap();
    storage.delete(&rg.data_path).await.unwrap();

    let result = reader.read_row_group(rg).await;
    assert!(
        matches!(
            result,
            Err(IcefallDBError::MissingRowGroupFile {
                snapshot: s,
                ref path,
            })
            if s == snapshot && path == &rg.data_path
        ),
        "expected MissingRowGroupFile for deleted parquet, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_reader_eq_stats_pruning() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let mut schema = make_int_schema(None);
    schema.row_group_target_rows = 2;

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 10, 11]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 2);

    let pruned = plan
        .prune(&[Predicate::Eq {
            column: "id".into(),
            value: Literal::Int64(10),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
    assert_eq!(pruned.row_groups[0].meta.rows, 2);
}

fn make_plan_with_stats(min: i64, max: i64, nulls: usize, rows: usize) -> ScanPlan {
    use serde_json::json;
    use std::collections::HashMap;

    let schema = make_int_schema(None);

    let mut columns = HashMap::new();
    columns.insert(
        "id".into(),
        ColumnStats {
            min: Some(json!(min)),
            max: Some(json!(max)),
            nulls,
        },
    );

    let rg = PlannedRowGroup {
        data_path: "rg_a.parquet".into(),
        meta_path: "rg_a.meta".into(),
        meta: RowGroupMeta {
            row_group: "rg_a".into(),
            schema_id: schema.schema_id,
            rows,
            columns,
            sort: None,
            checksum: "sha256:dummy".into(),
            meta_checksum: "sha256:dummy".into(),
            ..Default::default()
        },
        partition_values: None,
        snapshot: 1,
        ..Default::default()
    };

    ScanPlan {
        table: "t".into(),
        schema,
        row_groups: vec![rg],
    }
}

#[test]
fn test_retain_row_groups_by_row_ids_range_overlap() {
    // Guards the index-pruning fix: a row group with a large `Range` segment that
    // holds NO matching id must be pruned without expanding the range into per-id
    // probes (the prior O(total_rows) behavior that made indexed equality ~8x
    // slower than the unindexed scan).
    use icefalldb_core::rowid::RowIdSegment;
    use std::collections::HashSet;

    let schema = make_int_schema(None);
    let schema_id = schema.schema_id;
    let mk = |name: &str, row_ids: Vec<RowIdSegment>| PlannedRowGroup {
        data_path: format!("{name}.parquet"),
        meta_path: format!("{name}.meta"),
        meta: RowGroupMeta {
            row_group: name.into(),
            schema_id,
            rows: 1,
            columns: std::collections::HashMap::new(),
            sort: None,
            checksum: "sha256:dummy".into(),
            meta_checksum: "sha256:dummy".into(),
            row_ids,
            ..Default::default()
        },
        partition_values: None,
        snapshot: 1,
        ..Default::default()
    };

    let plan = ScanPlan {
        table: "t".into(),
        schema,
        row_groups: vec![
            mk(
                "rg_a",
                vec![RowIdSegment::Range {
                    start: 0,
                    count: 1_000_000,
                }],
            ),
            mk(
                "rg_b",
                vec![RowIdSegment::Range {
                    start: 2_000_000,
                    count: 500,
                }],
            ),
            mk(
                "rg_c",
                vec![RowIdSegment::Sorted {
                    ids: vec![5_000_000, 5_000_001],
                }],
            ),
            mk("rg_legacy", vec![]),
        ],
    };

    let mut ids = HashSet::new();
    // 1_500_000 is a decoy present in none of the segments.
    ids.insert(1_500_000);
    ids.insert(2_000_100); // inside rg_b's range
    ids.insert(5_000_001); // inside rg_c's sorted ids

    let mut plan = plan;
    plan.retain_row_groups_by_row_ids(&ids);
    let kept: Vec<&str> = plan
        .row_groups
        .iter()
        .map(|rg| rg.meta.row_group.as_str())
        .collect();
    assert_eq!(
        kept,
        vec!["rg_b", "rg_c", "rg_legacy"],
        "rg_a (1M-id range with no match) must be pruned; rg_legacy (no row ids) \
         is always retained; rg_b and rg_c overlap the match set"
    );
}

#[test]
fn test_reader_lt_stats_pruning() {
    let plan = make_plan_with_stats(5, 10, 0, 10);

    // Drop: min (5) >= 5, so no value is < 5.
    let pruned = plan
        .prune(&[Predicate::Lt {
            column: "id".into(),
            value: Literal::Int64(5),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());

    // Keep: min (5) < 6.
    let pruned = plan
        .prune(&[Predicate::Lt {
            column: "id".into(),
            value: Literal::Int64(6),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
}

#[test]
fn test_reader_lte_stats_pruning() {
    let plan = make_plan_with_stats(5, 10, 0, 10);

    // Drop: min (5) > 4.
    let pruned = plan
        .prune(&[Predicate::Lte {
            column: "id".into(),
            value: Literal::Int64(4),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());

    // Keep: min (5) <= 5.
    let pruned = plan
        .prune(&[Predicate::Lte {
            column: "id".into(),
            value: Literal::Int64(5),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
}

#[test]
fn test_reader_gt_stats_pruning() {
    let plan = make_plan_with_stats(5, 10, 0, 10);

    // Drop: max (10) <= 10.
    let pruned = plan
        .prune(&[Predicate::Gt {
            column: "id".into(),
            value: Literal::Int64(10),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());

    // Keep: max (10) > 9.
    let pruned = plan
        .prune(&[Predicate::Gt {
            column: "id".into(),
            value: Literal::Int64(9),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
}

#[test]
fn test_reader_gte_stats_pruning() {
    let plan = make_plan_with_stats(5, 10, 0, 10);

    // Drop: max (10) < 11.
    let pruned = plan
        .prune(&[Predicate::Gte {
            column: "id".into(),
            value: Literal::Int64(11),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());

    // Keep: max (10) >= 10.
    let pruned = plan
        .prune(&[Predicate::Gte {
            column: "id".into(),
            value: Literal::Int64(10),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
}

#[test]
fn test_reader_is_null_stats_pruning() {
    // Drop when there are no nulls.
    let plan = make_plan_with_stats(1, 10, 0, 10);
    let pruned = plan
        .prune(&[Predicate::IsNull {
            column: "id".into(),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());

    // Keep when there are nulls.
    let plan = make_plan_with_stats(1, 10, 2, 10);
    let pruned = plan
        .prune(&[Predicate::IsNull {
            column: "id".into(),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
}

#[test]
fn test_reader_is_not_null_stats_pruning() {
    // Drop when every value is null.
    let plan = make_plan_with_stats(1, 10, 10, 10);
    let pruned = plan
        .prune(&[Predicate::IsNotNull {
            column: "id".into(),
        }])
        .unwrap();
    assert!(pruned.row_groups.is_empty());

    // Keep when some values are non-null.
    let plan = make_plan_with_stats(1, 10, 3, 10);
    let pruned = plan
        .prune(&[Predicate::IsNotNull {
            column: "id".into(),
        }])
        .unwrap();
    assert_eq!(pruned.row_groups.len(), 1);
}

#[test]
fn test_reader_type_not_supported() {
    use serde_json::json;
    use std::collections::HashMap;

    let schema = make_int_schema(None);

    let mut columns = HashMap::new();
    columns.insert(
        "id".into(),
        ColumnStats {
            min: Some(json!(1)),
            max: Some(json!(10)),
            nulls: 0,
        },
    );

    let rg = PlannedRowGroup {
        data_path: "rg_a.parquet".into(),
        meta_path: "rg_a.meta".into(),
        meta: RowGroupMeta {
            row_group: "rg_a".into(),
            schema_id: schema.schema_id,
            rows: 10,
            columns,
            sort: None,
            checksum: "sha256:dummy".into(),
            meta_checksum: "sha256:dummy".into(),
            ..Default::default()
        },
        partition_values: None,
        snapshot: 1,
        ..Default::default()
    };

    let plan = ScanPlan {
        table: "t".into(),
        schema,
        row_groups: vec![rg],
    };

    let result = plan.prune(&[Predicate::Eq {
        column: "id".into(),
        value: Literal::Bool(true),
    }]);
    assert!(
        matches!(result, Err(IcefallDBError::TypeNotSupported(_))),
        "expected TypeNotSupported, got {:?}",
        result
    );
}

#[tokio::test]
async fn scan_populates_agg_state_for_insert_batch_fragments() {
    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use icefalldb_core::agg_cache::AggStateCache;
    use icefalldb_core::arrow_schema_to_icefalldb;
    use icefalldb_core::storage::memory::MemoryStorage;
    use icefalldb_core::{Reader, Writer};
    use std::sync::Arc;

    // Clear the global agg cache so this test is isolated.
    AggStateCache::global().clear();

    let storage = Arc::new(MemoryStorage::new());
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "v",
        DataType::Int64,
        false,
    )]));
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&schema));
    mdb_schema.row_group_target_rows = 3;

    let mut writer = Writer::create(
        Arc::clone(&storage) as Arc<dyn icefalldb_core::storage::Storage>,
        "t",
        mdb_schema,
    )
    .await
    .unwrap();
    let batch = arrow::record_batch::RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5, 6]))],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(storage.as_ref(), "t").await.unwrap();
    let plan = reader.scan().await.unwrap();

    // Every fragment produced by insert_batch has a .agg sidecar; they must
    // all have agg_state populated after scan().
    for rg in &plan.row_groups {
        assert!(
            rg.agg_state.is_some(),
            "agg_state must be Some for insert_batch fragment: {}",
            rg.data_path
        );
    }
}

#[tokio::test]
async fn scan_uses_checkpoint_when_meta_sidecars_are_missing() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 2);

    // Capture the ground-truth metadata from the on-disk `.meta` sidecars.
    let mut expected_meta = Vec::new();
    for rg in &plan.row_groups {
        let bytes = storage.read(&rg.meta_path).await.unwrap();
        expected_meta.push(serde_json::from_slice::<RowGroupMeta>(&bytes).unwrap());
    }

    // Clear caches and delete every .meta sidecar. The manifest still references
    // a snapshot checkpoint, so the next scan must succeed using only that file.
    MetaCache::global().clear();
    for rg in &plan.row_groups {
        storage.delete(&rg.meta_path).await.unwrap();
    }

    let plan_from_checkpoint = reader.scan().await.unwrap();
    assert_eq!(plan_from_checkpoint.row_groups.len(), 2);
    let actual_meta: Vec<RowGroupMeta> = plan_from_checkpoint
        .row_groups
        .iter()
        .map(|rg| rg.meta.clone())
        .collect();
    assert_eq!(expected_meta, actual_meta);
}

#[tokio::test]
async fn scan_falls_back_to_meta_when_checkpoint_is_missing_or_corrupt() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_int_schema(None);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 1);

    let mut expected_meta = Vec::new();
    for rg in &plan.row_groups {
        let bytes = storage.read(&rg.meta_path).await.unwrap();
        expected_meta.push(serde_json::from_slice::<RowGroupMeta>(&bytes).unwrap());
    }

    // Corrupt the checkpoint bytes and ensure the reader falls back to the
    // `.meta` sidecar rather than using invalid metadata.
    MetaCache::global().clear();
    let manifest = reader.catalog().latest_manifest().unwrap();
    let cp_rel = manifest.checkpoint.as_ref().unwrap();
    let cp_abs = format!("{}/{}", table, cp_rel);
    let original_checkpoint = storage.read(&cp_abs).await.unwrap();
    storage.write(&cp_abs, b"not-json").await.unwrap();

    let plan_corrupt = reader.scan().await.unwrap();
    assert_eq!(plan_corrupt.row_groups.len(), 1);
    assert_eq!(expected_meta, vec![plan_corrupt.row_groups[0].meta.clone()]);

    // Mutate only the checkpoint self-checksum. Valid JSON with a wrong
    // checksum must also be rejected so the scan falls back to `.meta`.
    MetaCache::global().clear();
    let mut cp_value: serde_json::Value = serde_json::from_slice(&original_checkpoint).unwrap();
    cp_value.as_object_mut().unwrap().insert(
        "checksum".into(),
        "sha256:0000000000000000000000000000000000000000000000000000000000000000".into(),
    );
    storage
        .write(&cp_abs, serde_json::to_vec(&cp_value).unwrap().as_slice())
        .await
        .unwrap();

    let plan_checksum = reader.scan().await.unwrap();
    assert_eq!(plan_checksum.row_groups.len(), 1);
    assert_eq!(
        expected_meta,
        vec![plan_checksum.row_groups[0].meta.clone()]
    );

    // Delete the checkpoint entirely and again verify the fallback path.
    MetaCache::global().clear();
    storage.delete(&cp_abs).await.unwrap();

    let plan_missing = reader.scan().await.unwrap();
    assert_eq!(plan_missing.row_groups.len(), 1);
    assert_eq!(expected_meta, vec![plan_missing.row_groups[0].meta.clone()]);
}

#[tokio::test]
async fn scan_checkpoint_preserves_sort_order() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let mut schema = make_int_schema(None);
    schema.sort = Some(vec!["id".into()]);

    let mut writer = Writer::new(Arc::clone(&storage), table, schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_int_batch("id", vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let reader = Reader::new(&*storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert_eq!(plan.row_groups.len(), 1);

    let mut expected_meta = Vec::new();
    for rg in &plan.row_groups {
        let bytes = storage.read(&rg.meta_path).await.unwrap();
        expected_meta.push(serde_json::from_slice::<RowGroupMeta>(&bytes).unwrap());
    }

    // Delete the sidecar and verify the checkpoint-derived metadata preserves
    // the declared sort order.
    MetaCache::global().clear();
    storage.delete(&plan.row_groups[0].meta_path).await.unwrap();

    let plan_from_checkpoint = reader.scan().await.unwrap();
    assert_eq!(plan_from_checkpoint.row_groups.len(), 1);
    assert_eq!(
        expected_meta,
        vec![plan_from_checkpoint.row_groups[0].meta.clone()]
    );
}
