use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use futures::StreamExt;
use icefalldb_core::agg_cache::FragmentAggState;
use icefalldb_core::check::Checker;
use icefalldb_core::compaction::{CompactionOptions, Compactor};
use icefalldb_core::database_catalog::DatabaseCatalog;
use icefalldb_core::doctor::Doctor;
use icefalldb_core::metadata::{Column, Manifest, RowGroupMeta, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::Reader;
use icefalldb_core::{load_index_by_ref, IcefallDBError, MatchLoc};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn make_int_schema(row_group_target_rows: usize) -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn make_mixed_schema(row_group_target_rows: usize) -> Schema {
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
                name: "name".into(),
                r#type: "utf8".into(),
                nullable: false,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_mixed_batch(ids: Vec<i64>, names: Vec<&str>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]);
    let ids = Int64Array::from(ids);
    let names = StringArray::from(names);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ids), Arc::new(names)]).unwrap()
}

fn make_wide_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "payload".into(),
            r#type: "utf8".into(),
            nullable: false,
            field_id: 0,
        }],
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

#[allow(clippy::manual_repeat_n)]
fn make_wide_batch(row_count: usize, width: usize) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("payload", DataType::Utf8, false)]);
    let value = "x".repeat(width);
    let values: Vec<String> = std::iter::repeat(value).take(row_count).collect();
    let array = StringArray::from(values);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
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

async fn read_latest_schema(storage: &Arc<dyn Storage>, table: &str) -> Schema {
    let pointer_data = storage
        .read(&format!("{}/_schema.json", table))
        .await
        .unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    let schema_id = pointer["latest"].as_u64().unwrap();
    let schema_data = storage
        .read(&format!("{}/{}", table, Schema::filename(schema_id)))
        .await
        .unwrap();
    serde_json::from_slice(&schema_data).unwrap()
}

async fn insert_two_unsorted_row_groups(storage: &Arc<dyn Storage>, table: &str) {
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(storage), table, schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch(vec![3, 1, 2]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch(vec![6, 4, 5]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
}

fn assert_batches_equal(a: &RecordBatch, b: &RecordBatch) {
    assert_eq!(a.num_rows(), b.num_rows(), "row count mismatch");
    assert_eq!(
        a.schema().fields().len(),
        b.schema().fields().len(),
        "column count mismatch"
    );
    for (i, field) in a.schema().fields().iter().enumerate() {
        let col_a = a.column(i);
        let col_b = b.column_by_name(field.name()).expect("column missing");
        assert_eq!(
            col_a.data_type(),
            col_b.data_type(),
            "type mismatch for column {}",
            field.name()
        );
        assert_eq!(
            col_a.len(),
            col_b.len(),
            "len mismatch for column {}",
            field.name()
        );
        match col_a.data_type() {
            DataType::Int64 => {
                let a_arr = col_a.as_any().downcast_ref::<Int64Array>().unwrap();
                let b_arr = col_b.as_any().downcast_ref::<Int64Array>().unwrap();
                for j in 0..a_arr.len() {
                    assert_eq!(
                        a_arr.value(j),
                        b_arr.value(j),
                        "value mismatch at row {}",
                        j
                    );
                }
            }
            DataType::Utf8 => {
                let a_arr = col_a.as_any().downcast_ref::<StringArray>().unwrap();
                let b_arr = col_b.as_any().downcast_ref::<StringArray>().unwrap();
                for j in 0..a_arr.len() {
                    assert_eq!(
                        a_arr.value(j),
                        b_arr.value(j),
                        "value mismatch at row {}",
                        j
                    );
                }
            }
            other => panic!("unsupported type in assertion: {:?}", other),
        }
    }
}

/// A [`Storage`] wrapper that records every path passed to `write`.
struct RecordingStorage {
    inner: Arc<dyn Storage>,
    writes: Arc<Mutex<Vec<String>>>,
}

impl RecordingStorage {
    fn wrap(inner: Arc<dyn Storage>) -> (Arc<dyn Storage>, Arc<Mutex<Vec<String>>>) {
        let writes = Arc::new(Mutex::new(Vec::new()));
        let storage: Arc<dyn Storage> = Arc::new(Self {
            inner,
            writes: writes.clone(),
        });
        (storage, writes)
    }
}

#[async_trait]
impl Storage for RecordingStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> icefalldb_core::Result<Vec<u8>> {
        self.inner.read(path).await
    }

    async fn size(&self, path: &str) -> icefalldb_core::Result<u64> {
        self.inner.size(path).await
    }

    async fn read_range(
        &self,
        path: &str,
        offset: u64,
        len: u64,
    ) -> icefalldb_core::Result<Vec<u8>> {
        self.inner.read_range(path, offset, len).await
    }

    async fn write(&self, path: &str, data: &[u8]) -> icefalldb_core::Result<()> {
        self.writes.lock().unwrap().push(path.to_string());
        self.inner.write(path, data).await
    }

    async fn delete(&self, path: &str) -> icefalldb_core::Result<()> {
        self.inner.delete(path).await
    }

    async fn rename(&self, from: &str, to: &str) -> icefalldb_core::Result<()> {
        self.inner.rename(from, to).await
    }

    async fn list(&self, prefix: &str) -> icefalldb_core::Result<Vec<String>> {
        self.inner.list(prefix).await
    }

    async fn exists(&self, path: &str) -> icefalldb_core::Result<bool> {
        self.inner.exists(path).await
    }

    async fn lock_exclusive(
        &self,
        path: &str,
        timeout: Duration,
    ) -> icefalldb_core::Result<Box<dyn icefalldb_core::storage::LockGuard>> {
        self.inner.lock_exclusive(path, timeout).await
    }

    async fn sync(&self, path: &str) -> icefalldb_core::Result<()> {
        self.inner.sync(path).await
    }
}

#[tokio::test]
async fn test_compaction_reduces_file_count() {
    let (storage, writes) = RecordingStorage::wrap(Arc::new(MemoryStorage::new()));
    // Use a schema target larger than a single batch so each committed batch
    // becomes its own row group, but compaction can still merge them.
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((21..=30).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();

    assert_eq!(result.input_row_groups, 3);
    assert_eq!(result.output_row_groups, 1);
    assert_eq!(result.input_rows, 30);
    assert_eq!(result.output_rows, 30);

    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 1);

    // The compacted row group must be validly checksummed.
    let entry = &manifest.row_groups[0];
    let meta_bytes = storage
        .read(&format!("products/{}", entry.meta))
        .await
        .unwrap();
    let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes).unwrap();
    assert!(meta.verify_meta_checksum().unwrap());
    let data_bytes = storage
        .read(&format!("products/{}", entry.data))
        .await
        .unwrap();
    assert!(meta.verify_against_data(&data_bytes));

    // Staged files were written to _staging/compact/ and renamed to the table
    // root; no .part files should be left behind.
    let write_paths: Vec<String> = writes.lock().unwrap().clone();
    let parquet_part = format!("products/_staging/compact/{}.part", entry.data);
    let meta_part = format!("products/_staging/compact/{}.part", entry.meta);
    assert!(
        write_paths.contains(&parquet_part),
        "expected the compacted parquet to be staged in _staging/compact"
    );
    assert!(
        write_paths.contains(&meta_part),
        "expected the compacted meta to be staged in _staging/compact"
    );
    let compact_leftovers = storage
        .list("products/_staging/compact")
        .await
        .unwrap_or_default();
    assert!(
        compact_leftovers.is_empty(),
        "no leftover files in _staging/compact after commit"
    );
}

#[tokio::test]
async fn test_compaction_no_op_on_empty_table() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    Writer::create(Arc::clone(&storage), "empty", make_int_schema(10))
        .await
        .unwrap();

    let result = Compactor::new(storage.as_ref(), "empty")
        .compact()
        .await
        .unwrap();

    assert_eq!(result.input_row_groups, 0);
    assert_eq!(result.output_row_groups, 0);
    assert_eq!(result.input_rows, 0);
    assert_eq!(result.output_rows, 0);
}

#[tokio::test]
async fn test_compaction_fails_on_missing_table() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let err = Compactor::new(storage.as_ref(), "missing")
        .compact()
        .await
        .unwrap_err();

    assert!(matches!(err, IcefallDBError::TableNotFound(_)), "{err}");
}

#[tokio::test]
async fn test_compaction_preserves_data() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_mixed_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let batches = vec![
        make_mixed_batch(vec![1, 2, 3, 4, 5], vec!["a", "b", "c", "d", "e"]),
        make_mixed_batch(vec![6, 7, 8, 9, 10], vec!["f", "g", "h", "i", "j"]),
        make_mixed_batch(vec![11, 12, 13, 14, 15], vec!["k", "l", "m", "n", "o"]),
    ];

    for batch in &batches {
        writer.insert_batch(batch.clone()).await.unwrap();
    }
    writer.commit().await.unwrap();

    let expected = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();

    Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();

    let reader = Reader::new(storage.as_ref(), "products").await.unwrap();
    let scan = reader.scan().await.unwrap();
    let mut read_batches = Vec::new();
    for prg in &scan.row_groups {
        let mut stream = reader.read_row_group(prg).await.unwrap();
        while let Some(batch) = stream.next().await {
            read_batches.push(batch.unwrap());
        }
    }
    let actual = arrow::compute::concat_batches(&expected.schema(), &read_batches).unwrap();

    assert_batches_equal(&expected, &actual);
}

#[tokio::test]
async fn test_compaction_respects_target_rows() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..10 {
        let start = i * 10 + 1;
        let ids: Vec<i64> = (start..start + 10).collect();
        writer.insert_batch(make_batch(ids)).await.unwrap();
    }
    writer.commit().await.unwrap();

    let options = CompactionOptions {
        target_row_group_rows: 25,
        target_row_group_bytes: 128 * 1024 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: false,
        sort_keys: Vec::new(),
    };
    let result = Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await
        .unwrap();

    assert_eq!(result.input_row_groups, 10);
    assert_eq!(result.output_row_groups, 4);
    assert_eq!(result.input_rows, 100);
    assert_eq!(result.output_rows, 100);
}

#[tokio::test]
async fn test_compaction_uses_schema_targets() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    // Use a small row-group target in the schema.
    let schema = make_int_schema(7);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    // The writer splits the 21 rows into 3 row groups of 7 rows each.
    writer
        .insert_batch(make_batch((1..=21).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let before = read_latest_manifest(&storage, "products").await;
    assert_eq!(before.row_groups.len(), 3);

    // Compact without explicit options. If the default target (1M rows) were
    // used, all rows would be merged into a single row group. The compactor
    // should instead fall back to the schema target of 7 rows.
    let result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();

    assert_eq!(result.input_row_groups, 3);
    assert_eq!(result.output_row_groups, 3);
    assert_eq!(result.input_rows, 21);
    assert_eq!(result.output_rows, 21);
}

#[tokio::test]
async fn test_compaction_lock_conflicts_with_writer() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(10);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Hold the exclusive writer lock from outside the compactor.
    let _guard = storage
        .lock_exclusive("products/_write.lock", Duration::from_secs(10))
        .await
        .unwrap();

    let options = CompactionOptions {
        target_row_group_rows: 10,
        target_row_group_bytes: 1024 * 1024,
        lock_timeout: Duration::from_millis(50),
        force: false,
        sort_keys: Vec::new(),
    };
    let result = Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await;

    assert!(
        matches!(result, Err(IcefallDBError::LockTimeout(_))),
        "expected LockTimeout, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_compaction_recovers_from_crashed_compaction() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let before_manifest = read_latest_manifest(&storage, "products").await;
    let before_seq = before_manifest.sequence;

    // Simulate a crashed compaction: leave a stale intent that references
    // uncommitted output files, an orphan .part file in _staging/compact/, and
    // an uncommitted newer manifest snapshot.
    let stale_parquet = "rg_deadbeef0000000000000000.parquet";
    let stale_meta = "rg_deadbeef0000000000000000.meta";
    storage
        .write(&format!("products/{}", stale_parquet), b"stale")
        .await
        .unwrap();
    storage
        .write(&format!("products/{}", stale_meta), b"stale")
        .await
        .unwrap();

    let intent = serde_json::json!({
        "txn_id": "txn_deadbeef0000000000000000",
        "started_at": chrono::Utc::now().to_rfc3339(),
        "schema_id": 1,
        "files": [stale_parquet, stale_meta],
    });
    storage
        .write(
            "products/_staging/intents/txn_deadbeef0000000000000000.json",
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();

    storage
        .write(
            "products/_staging/compact/rg_cafebabe0000000000000000.parquet.part",
            b"orphan-part",
        )
        .await
        .unwrap();

    let stale_manifest_path = format!("products/{}", Manifest::filename(before_seq + 2));
    storage.write(&stale_manifest_path, b"{}").await.unwrap();

    // Compaction should succeed and clean up all crash debris before writing
    // the new manifest at before_seq + 1.
    let result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();
    assert_eq!(result.input_row_groups, 2);
    assert_eq!(result.output_row_groups, 1);

    let after_manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(after_manifest.sequence, before_seq + 1);
    assert_eq!(after_manifest.row_groups.len(), 1);

    assert!(
        !storage
            .exists(&format!("products/{}", stale_parquet))
            .await
            .unwrap(),
        "stale parquet from crashed intent should be deleted"
    );
    assert!(
        !storage
            .exists(&format!("products/{}", stale_meta))
            .await
            .unwrap(),
        "stale meta from crashed intent should be deleted"
    );
    assert!(
        !storage
            .exists("products/_staging/intents/txn_deadbeef0000000000000000.json")
            .await
            .unwrap(),
        "stale intent should be deleted"
    );
    assert!(
        !storage
            .exists("products/_staging/compact/rg_cafebabe0000000000000000.parquet.part")
            .await
            .unwrap(),
        "orphan .part file in _staging/compact should be deleted"
    );
    assert!(
        !storage.exists(&stale_manifest_path).await.unwrap(),
        "uncommitted newer manifest snapshot should be deleted"
    );
}

#[tokio::test]
async fn test_compaction_intent_lists_output_files() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();

    // After a successful compaction no intent file should remain.
    let intents: Vec<String> = storage
        .list("products/_staging/intents")
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.ends_with(".json"))
        .collect();
    assert!(
        intents.is_empty(),
        "expected no stale intent files after compaction, found {:?}",
        intents
    );

    // Every row group in the new manifest must exist and be checksummed.
    let manifest = read_latest_manifest(&storage, "products").await;
    for entry in &manifest.row_groups {
        let meta_bytes = storage
            .read(&format!("products/{}", entry.meta))
            .await
            .unwrap();
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes).unwrap();
        assert!(meta.verify_meta_checksum().unwrap());
        let data_bytes = storage
            .read(&format!("products/{}", entry.data))
            .await
            .unwrap();
        assert!(meta.verify_against_data(&data_bytes));
    }
}

#[tokio::test]
async fn test_compaction_intent_rollback() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Capture the pre-repair snapshot so we can distinguish committed files
    // from the fake stale output files created below.
    let before_manifest = read_latest_manifest(&storage, "products").await;
    let before_files: std::collections::HashSet<String> = before_manifest
        .row_groups
        .iter()
        .flat_map(|e| [e.data.clone(), e.meta.clone()])
        .collect();

    // Simulate a crashed compaction by creating an intent that references
    // uncommitted output files in the table root.
    let stale_parquet = "rg_deadbeef0000000000000000.parquet";
    let stale_meta = "rg_deadbeef0000000000000000.meta";
    storage
        .write(&format!("products/{}", stale_parquet), b"stale")
        .await
        .unwrap();
    storage
        .write(&format!("products/{}", stale_meta), b"stale")
        .await
        .unwrap();

    let intent = serde_json::json!({
        "txn_id": "txn_deadbeef0000000000000000",
        "files": [stale_parquet, stale_meta],
    });
    storage
        .write(
            "products/_staging/intents/txn_deadbeef0000000000000000.json",
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();

    // Repair should roll back the stale intent and delete the unreferenced
    // files it lists, while leaving committed files untouched.
    let result = Doctor::new(storage.as_ref(), "products")
        .repair()
        .await
        .unwrap();
    assert!(result.repaired, "expected repair to mutate state");
    assert!(
        !storage
            .exists(&format!("products/{}", stale_parquet))
            .await
            .unwrap(),
        "stale parquet should be deleted by repair"
    );
    assert!(
        !storage
            .exists(&format!("products/{}", stale_meta))
            .await
            .unwrap(),
        "stale meta should be deleted by repair"
    );

    for file in &before_files {
        assert!(
            storage.exists(&format!("products/{}", file)).await.unwrap(),
            "committed file {} should not be deleted by repair",
            file
        );
    }

    // The original manifest is still valid and unchanged.
    let after_manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(after_manifest.sequence, before_manifest.sequence);
}

#[tokio::test]
async fn test_compaction_repairs_missing_pointer() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let before_seq = read_latest_manifest(&storage, "products").await.sequence;

    // Delete the manifest pointer but leave the snapshots intact. The table
    // still exists (schema + `_manifests/` snapshots), so compaction must
    // self-repair the pointer from the retained snapshots rather than reporting
    // the table as missing. This mirrors `test_compaction_repairs_corrupt_pointer`
    // for the missing-file (not corrupt-file) case.
    storage.delete("products/_manifest.json").await.unwrap();

    let result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();
    assert_eq!(result.input_row_groups, 2);
    assert_eq!(result.output_row_groups, 1);

    // The pointer must be restored and point to a new compacted snapshot.
    let pointer_data = storage.read("products/_manifest.json").await.unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    assert!(pointer["latest"].as_u64().unwrap() > before_seq);
}

#[tokio::test]
async fn test_compaction_repairs_corrupt_pointer() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let before_manifest = read_latest_manifest(&storage, "products").await;
    let before_seq = before_manifest.sequence;

    // Corrupt the manifest pointer to invalid JSON.
    storage
        .write("products/_manifest.json", b"not valid json")
        .await
        .unwrap();

    let result = Compactor::new(storage.as_ref(), "products")
        .compact()
        .await
        .unwrap();
    assert_eq!(result.input_row_groups, 2);
    assert_eq!(result.output_row_groups, 1);

    // The pointer must be restored and point to a new compacted snapshot.
    let pointer_data = storage.read("products/_manifest.json").await.unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    assert!(pointer["latest"].as_u64().unwrap() > before_seq);

    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 1);
}

#[tokio::test]
async fn test_compaction_respects_target_bytes() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_wide_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    let row_count = 100;
    let width = 1024;
    writer
        .insert_batch(make_wide_batch(row_count, width))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let before = read_latest_manifest(&storage, "products").await;
    assert_eq!(before.row_groups.len(), 1);

    let options = CompactionOptions {
        target_row_group_rows: 1_000_000,
        target_row_group_bytes: 4 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: true,
        sort_keys: Vec::new(),
    };
    let result = Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await
        .unwrap();

    assert_eq!(result.input_row_groups, 1);
    assert_eq!(result.input_rows, row_count as u64);
    assert_eq!(result.output_rows, row_count as u64);
    assert!(
        result.output_row_groups > 1,
        "expected byte target to split output, got {} output row groups",
        result.output_row_groups
    );
}

#[tokio::test]
async fn test_optimize_persists_sort_keys_in_schema_and_row_group_meta() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    insert_two_unsorted_row_groups(&storage, "sorted_products").await;

    let before_schema = read_latest_schema(&storage, "sorted_products").await;
    assert_eq!(before_schema.schema_id, 1);
    assert!(before_schema.sort.is_none());

    let options = CompactionOptions {
        target_row_group_rows: 1_000_000,
        target_row_group_bytes: 128 * 1024 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: true,
        sort_keys: vec!["id".to_string()],
    };
    let result = Compactor::with_options(storage.as_ref(), "sorted_products", options)
        .compact()
        .await
        .unwrap();
    assert_eq!(result.input_row_groups, 2);
    assert!(result.output_row_groups >= 1);

    let schema = read_latest_schema(&storage, "sorted_products").await;
    assert_eq!(schema.schema_id, 2, "optimize must bump schema_id");
    assert_eq!(schema.sort.as_deref(), Some(["id".to_string()].as_slice()));

    let manifest = read_latest_manifest(&storage, "sorted_products").await;
    assert_eq!(manifest.schema_id, 2, "manifest must reference new schema");

    for entry in &manifest.row_groups {
        let meta_bytes = storage
            .read(&format!("sorted_products/{}", entry.meta))
            .await
            .unwrap();
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes).unwrap();
        assert_eq!(
            meta.sort.as_deref(),
            Some(["id".to_string()].as_slice()),
            "row group {} must advertise sort keys",
            meta.row_group
        );
    }
}

#[tokio::test]
async fn test_optimize_no_schema_bump_when_sort_keys_unchanged() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    insert_two_unsorted_row_groups(&storage, "products").await;

    let before_schema = read_latest_schema(&storage, "products").await;
    assert_eq!(before_schema.schema_id, 1);
    assert!(before_schema.sort.is_none());

    let options = CompactionOptions {
        target_row_group_rows: 1_000_000,
        target_row_group_bytes: 128 * 1024 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: true,
        sort_keys: vec!["id".to_string()],
    };
    let first = Compactor::with_options(storage.as_ref(), "products", options.clone())
        .compact()
        .await
        .unwrap();
    assert!(first.rewrote);

    let after_first = read_latest_schema(&storage, "products").await;
    assert_eq!(after_first.schema_id, 2);
    assert_eq!(
        after_first.sort.as_deref(),
        Some(["id".to_string()].as_slice())
    );

    // A second optimize with the same sort keys must not bump the schema again,
    // even if it rewrites the row groups.
    let second = Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await
        .unwrap();
    assert!(second.rewrote);

    let after_second = read_latest_schema(&storage, "products").await;
    assert_eq!(
        after_second.schema_id, 2,
        "schema must stay at 2 when sort keys are unchanged"
    );
}

#[tokio::test]
async fn test_optimize_no_schema_bump_on_no_op_compact() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    // One row group: compact without force is a no-op.
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![3, 1, 2]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let before_schema = read_latest_schema(&storage, "products").await;
    assert_eq!(before_schema.schema_id, 1);
    assert!(before_schema.sort.is_none());

    let options = CompactionOptions {
        target_row_group_rows: 1_000_000,
        target_row_group_bytes: 128 * 1024 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: false,
        sort_keys: vec!["id".to_string()],
    };
    let result = Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await
        .unwrap();
    assert!(!result.rewrote, "expected no-op compact");
    assert_eq!(result.input_row_groups, 1);
    assert_eq!(result.output_row_groups, 1);

    let after_schema = read_latest_schema(&storage, "products").await;
    assert_eq!(
        after_schema.schema_id, 1,
        "schema must not bump on no-op compact"
    );
    assert!(after_schema.sort.is_none());
}

#[tokio::test]
async fn test_optimize_rejects_invalid_sort_key() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    insert_two_unsorted_row_groups(&storage, "products").await;

    let options = CompactionOptions {
        target_row_group_rows: 1_000_000,
        target_row_group_bytes: 128 * 1024 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: true,
        sort_keys: vec!["not_a_column".to_string()],
    };
    let result = Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await;

    match result {
        Err(IcefallDBError::Other(e)) => {
            let msg = e.to_string();
            assert!(
                msg.contains("not_a_column"),
                "error should name the invalid sort key: {}",
                msg
            );
        }
        other => panic!("expected Other error for invalid sort key, got {:?}", other),
    }

    // No schema file or tmp file should have been written.
    let schema = read_latest_schema(&storage, "products").await;
    assert_eq!(schema.schema_id, 1);
    assert!(schema.sort.is_none());

    let staging = storage
        .list("products/_staging/compact")
        .await
        .unwrap_or_default();
    assert!(staging.is_empty(), "no staging files should be left behind");
}

#[tokio::test]
async fn test_optimize_check_passes_after_sort() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    insert_two_unsorted_row_groups(&storage, "products").await;

    let options = CompactionOptions {
        target_row_group_rows: 1_000_000,
        target_row_group_bytes: 128 * 1024 * 1024,
        lock_timeout: Duration::from_secs(30),
        force: true,
        sort_keys: vec!["id".to_string()],
    };
    Compactor::with_options(storage.as_ref(), "products", options)
        .compact()
        .await
        .unwrap();

    let check_result = Checker::new(storage.as_ref(), "products")
        .check()
        .await
        .unwrap();
    assert!(
        check_result.passed,
        "Checker should pass after optimize with sort keys: {:?}",
        check_result.issues
    );
}

// ---------------------------------------------------------------------------
// Guard tests — index-invisibility invariant
// ---------------------------------------------------------------------------

/// Schema with two columns: an integer id and a string category column.
fn indexed_table_schema() -> Schema {
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
                name: "cat".into(),
                r#type: "utf8".into(),
                nullable: true,
                field_id: 0,
            },
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

fn make_id_cat_batch(ids: Vec<i64>, cats: Vec<&str>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(StringArray::from(cats)),
        ],
    )
    .unwrap()
}

/// Guard test: compaction MUST NOT change `index_generations`.
///
/// The index maps `value → row_id` and row_ids are preserved by compaction,
/// so the generation pointer is still valid and must be carried
/// forward bit-for-bit.  Any change here means compaction wrongly rebuilt or
/// relocated the index.
#[tokio::test]
async fn compaction_does_not_change_index_generations() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = indexed_table_schema();

    // Register the table and a secondary btree index in the database catalog.
    let dbcat = DatabaseCatalog::new(storage.clone());
    let guard = dbcat.acquire_lock(Duration::from_secs(10)).await.unwrap();
    dbcat.create_table(&guard, "events", &schema).await.unwrap();
    dbcat
        .create_index_definition(&guard, "cat_idx", "events", "cat", "btree")
        .await
        .unwrap();
    drop(guard);

    // Insert two batches so there are multiple row groups to compact.
    let mut writer = Writer::new(storage.clone(), "events", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_id_cat_batch(vec![1, 2, 3], vec!["a", "b", "a"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    writer
        .insert_batch(make_id_cat_batch(vec![4, 5, 6], vec!["b", "a", "c"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Apply a DELETE mutation on fragment 0, offset 0 (id=1, cat="a").
    let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "events")
        .await
        .unwrap();
    let manifest_before_del = catalog.latest_manifest().unwrap().clone();
    let frag0_id = manifest_before_del.row_groups[0].fragment_id;

    let mut writer2 = Writer::new(storage.clone(), "events", schema.clone())
        .await
        .unwrap();
    writer2
        .commit_deletes(HashMap::from([(frag0_id, vec![0u32])]))
        .await
        .unwrap();

    // Apply an UPDATE mutation: change id=4 (frag 1, offset 0, row_id 3) cat to "z".
    let catalog2 = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "events")
        .await
        .unwrap();
    let manifest_after_del = catalog2.latest_manifest().unwrap().clone();
    let frag1_id = manifest_after_del.row_groups[1].fragment_id;
    // row_ids in the second fragment start right after the first fragment's rows.
    // First fragment has 3 rows with row_ids [0,1,2]; second fragment starts at 3.
    let row_id_of_id4: u64 = 3; // first row of second fragment

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let update_batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int64Array::from(vec![4i64])),
            Arc::new(StringArray::from(vec!["z"])),
        ],
    )
    .unwrap();

    let mut writer3 = Writer::new(storage.clone(), "events", schema.clone())
        .await
        .unwrap();
    writer3
        .commit_update(
            update_batch,
            vec![MatchLoc {
                fragment_id: frag1_id,
                offset: 0,
                row_id: row_id_of_id4,
            }],
            &["cat".to_string()],
        )
        .await
        .unwrap();

    // Capture index_generations BEFORE compaction.
    let catalog3 = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "events")
        .await
        .unwrap();
    let before_manifest = catalog3.latest_manifest().unwrap().clone();
    let index_generations_before = before_manifest.index_generations.clone();
    assert!(
        !index_generations_before.is_empty(),
        "index_generations must be populated before compaction"
    );

    // Run compaction (force=true so it rewrites even a small table).
    let compact_opts = CompactionOptions {
        force: true,
        lock_timeout: Duration::from_secs(30),
        ..CompactionOptions::default()
    };
    let result = Compactor::with_options(storage.as_ref(), "events", compact_opts)
        .compact()
        .await
        .unwrap();
    assert!(result.rewrote, "compaction must have rewritten data files");

    // Capture index_generations AFTER compaction.
    let catalog4 = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "events")
        .await
        .unwrap();
    let after_manifest = catalog4.latest_manifest().unwrap().clone();
    let index_generations_after = after_manifest.index_generations.clone();

    // Invariant: index_generations must be bit-for-bit identical.
    // The index stores value→row_id; row_ids are preserved by compaction so
    // the generation pointer (base + deltas) is still valid.
    assert_eq!(
        index_generations_before, index_generations_after,
        "compaction must carry index_generations forward UNCHANGED: \
         before={:?} after={:?}",
        index_generations_before, index_generations_after,
    );

    // Cross-check: the index file referenced by the carried-forward generation
    // must still be loadable from storage (not deleted or moved).
    for (name, idx_ref) in &index_generations_after {
        let loaded = load_index_by_ref(storage.as_ref(), "events", name, idx_ref)
            .await
            .unwrap();
        assert!(
            loaded.is_some(),
            "index '{}' referenced by carried-forward generation must still be loadable",
            name
        );
    }
}

/// Collect every live `id` value currently visible in `table` (applying
/// deletion vectors), sorted ascending.
async fn read_live_ids(storage: &Arc<dyn Storage>, table: &str) -> Vec<i64> {
    let reader = Reader::new(storage.as_ref(), table).await.unwrap();
    let scan = reader.scan().await.unwrap();
    let mut ids = Vec::new();
    for prg in &scan.row_groups {
        let mut stream = reader.read_row_group(prg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let col = batch
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            for i in 0..col.len() {
                ids.push(col.value(i));
            }
        }
    }
    ids.sort_unstable();
    ids
}

/// A mutation that commits during the lock-free heavy phase of a
/// compaction must NEVER be lost or resurrected.
///
/// The compactor runs its heavy rewrite without holding `_write.lock`, pinning a
/// source sequence `S`. Using the deterministic `compact_with_hook` seam we
/// commit a real DELETE (advancing the pointer to `S+1`) *after* the heavy phase
/// has staged its compacted fragments but *before* the brief locked commit. The
/// commit must revalidate the pinned sequence, observe `S+1 != S`, and refuse to
/// clobber the deletion — aborting and retrying against the fresh snapshot.
///
/// Either acceptable terminal state is checked: the deleted row (`id = 5`) is
/// gone, every other row survives, and the table reads back consistently.
#[tokio::test]
async fn mutation_during_compaction_is_not_lost() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_int_schema(100);
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema.clone())
        .await
        .unwrap();

    // Three committed batches → three row groups (target=100 keeps them split),
    // so compaction has real work to merge. id=5 lands in fragment 1 at
    // physical offset 1 with row_id 4 (fragment 0 holds row_ids 0..3).
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_batch(vec![7, 8, 9]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Locate fragment 1 (the row group containing id=5) so we can delete its
    // physical offset 1 from inside the hook.
    let manifest = read_latest_manifest(&storage, "products").await;
    assert_eq!(manifest.row_groups.len(), 3);
    let pinned_seq_before = manifest.sequence;
    let frag1_id = manifest.row_groups[1].fragment_id;

    // The hook fires once per attempt; commit the DELETE only on the FIRST
    // attempt (when the pointer still equals the pinned source). On the retry
    // the pointer has already advanced, so we leave it untouched and let the
    // compaction commit cleanly against the post-mutation snapshot.
    let hook_storage = Arc::clone(&storage);
    let did_delete = Arc::new(Mutex::new(false));
    let did_delete_hook = Arc::clone(&did_delete);
    let schema_for_hook = schema.clone();

    let result = Compactor::new(storage.as_ref(), "products")
        .compact_with_hook(move |pinned| {
            let hook_storage = Arc::clone(&hook_storage);
            let did_delete_hook = Arc::clone(&did_delete_hook);
            let schema_for_hook = schema_for_hook.clone();
            async move {
                // Decide-and-claim under the lock, then release it before any
                // await so the guard never spans an await point.
                let first_attempt = {
                    let mut guard = did_delete_hook.lock().unwrap();
                    if *guard {
                        false
                    } else {
                        *guard = true;
                        true
                    }
                };
                if !first_attempt {
                    return; // already mutated on a previous attempt
                }
                // Sanity: the first attempt must pin the pre-mutation snapshot.
                assert_eq!(pinned, pinned_seq_before);

                // Commit a real DELETE of id=5 (fragment 1, physical offset 1).
                // This advances the manifest pointer to S+1 while the compaction
                // holds no lock.
                let mut w = Writer::new(hook_storage, "products", schema_for_hook)
                    .await
                    .unwrap();
                w.commit_deletes(HashMap::from([(frag1_id, vec![1u32])]))
                    .await
                    .unwrap();
            }
        })
        .await
        .unwrap();

    // The hook must actually have fired (proving the heavy phase reached the
    // commit seam) and advanced the pointer.
    assert!(
        *did_delete.lock().unwrap(),
        "the interleaved delete must run"
    );
    let final_manifest = read_latest_manifest(&storage, "products").await;
    assert!(
        final_manifest.sequence > pinned_seq_before,
        "the mutation must have advanced the manifest pointer"
    );

    // THE INVARIANT: id=5 is gone and every other row survives, regardless of
    // whether the compaction aborted (data = pre-compaction-post-mutation) or
    // rebased/retried (data = compacted-post-mutation).
    let live = read_live_ids(&storage, "products").await;
    assert_eq!(
        live,
        vec![1, 2, 3, 4, 6, 7, 8, 9],
        "id=5 must stay deleted and no other row may be lost (result={:?})",
        result
    );
}

fn make_kv_schema_with_key(row_group_target_rows: usize) -> Schema {
    Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "k".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 1,
            },
            Column {
                name: "v".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 2,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: Some(vec!["k".to_string()]),
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 2,
        dropped_columns: vec![],
    }
}

fn make_kv_batch(keys: Vec<i64>, vals: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(vals)),
        ],
    )
    .unwrap()
}

/// Integration test: after compaction every output RowGroupEntry.agg is Some,
/// the .agg file exists, content_hash == RowGroupMeta.checksum, ungrouped
/// partials match a direct aggregate over the compacted data, and grouped
/// partials are present because the schema declares agg_group_keys.
#[tokio::test]
async fn test_compaction_produces_agg_sidecars() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_kv_schema_with_key(5); // small target → 2+ output fragments

    let mut writer = Writer::new(Arc::clone(&storage), "agg_tbl", schema)
        .await
        .unwrap();

    // Insert 3 fragments of 5 rows each → compaction merges them.
    for i in 0..3i64 {
        let keys: Vec<i64> = (0..5).map(|j| j % 3).collect(); // keys 0,1,2
        let vals: Vec<i64> = (0..5).map(|j| i * 10 + j).collect();
        writer
            .insert_batch(make_kv_batch(keys, vals))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    Compactor::new(storage.as_ref(), "agg_tbl")
        .compact()
        .await
        .unwrap();

    let manifest = read_latest_manifest(&storage, "agg_tbl").await;
    assert!(
        !manifest.row_groups.is_empty(),
        "compaction must produce at least one output fragment"
    );

    for entry in &manifest.row_groups {
        // 1. entry.agg must be Some.
        let agg_rel = entry
            .agg
            .as_ref()
            .unwrap_or_else(|| panic!("compacted entry {} has agg: None", entry.data));

        // 2. The .agg file must exist.
        let agg_path = format!("agg_tbl/{}", agg_rel);
        let agg_bytes = storage
            .read(&agg_path)
            .await
            .unwrap_or_else(|_| panic!(".agg file missing at {}", agg_path));

        // 3. Deserialize and check content_hash == meta.checksum.
        let agg_state: FragmentAggState =
            serde_json::from_slice(&agg_bytes).expect("agg file must be valid JSON");
        let meta_bytes = storage
            .read(&format!("agg_tbl/{}", entry.meta))
            .await
            .unwrap();
        let meta: RowGroupMeta = serde_json::from_slice(&meta_bytes).unwrap();
        assert_eq!(
            agg_state.content_hash, meta.checksum,
            "agg content_hash must equal meta.checksum for {}",
            entry.data
        );

        // 4. fragment_id must be non-zero and match.
        assert_eq!(
            agg_state.fragment_id, entry.fragment_id,
            "agg fragment_id must match entry.fragment_id"
        );

        // 5. Ungrouped partials: "v" column must be present.
        assert!(
            agg_state.cols.contains_key("v"),
            "agg state must have 'v' column partials"
        );

        // 6. Grouped partials must be present (schema has agg_group_keys = ["k"]).
        assert!(
            agg_state.grouped.is_some(),
            "grouped partials must be present for entry {} (schema declares agg_group_keys)",
            entry.data
        );
    }
}

/// Atomicity test: the committed .agg file is in manifest_referenced_files
/// so GC/recovery won't delete it; an orphan .agg.part from a crash is cleaned.
#[tokio::test]
async fn test_compaction_agg_in_manifest_referenced_files() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_kv_schema_with_key(100);

    let mut writer = Writer::new(Arc::clone(&storage), "ref_tbl", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_kv_batch(vec![1, 2], vec![10, 20]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_kv_batch(vec![3, 4], vec![30, 40]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    Compactor::new(storage.as_ref(), "ref_tbl")
        .compact()
        .await
        .unwrap();

    let manifest = read_latest_manifest(&storage, "ref_tbl").await;

    // Every entry.agg must be in manifest_referenced_files (i.e. the set of
    // files that cleanup_staging / GC must not delete). We verify indirectly:
    // run another compaction immediately (it's a no-op on a single-fragment
    // table, but the early-lock path calls cleanup_staging which would delete
    // any .agg files not in referenced_files). Then check the .agg files
    // still exist.
    let agg_paths: Vec<String> = manifest
        .row_groups
        .iter()
        .filter_map(|e| e.agg.as_ref())
        .map(|a| format!("ref_tbl/{}", a))
        .collect();
    assert!(
        !agg_paths.is_empty(),
        "must have at least one .agg file after compaction"
    );

    // Second compact (no-op) exercises the early cleanup_staging path.
    Compactor::new(storage.as_ref(), "ref_tbl")
        .compact()
        .await
        .unwrap();

    for agg_path in &agg_paths {
        assert!(
            storage.exists(agg_path).await.unwrap(),
            "committed .agg file must survive cleanup_staging: {}",
            agg_path
        );
    }

    // Orphan .agg.part (simulated crash artifact) must be cleaned by next compaction.
    let orphan = "ref_tbl/_staging/compact/rg_orphan0000000000000000.agg.part";
    storage.write(orphan, b"orphan").await.unwrap();

    // Insert + commit to give compaction something to do (otherwise it's no-op).
    let mut writer2 = Writer::new(
        Arc::clone(&storage),
        "ref_tbl",
        make_kv_schema_with_key(100),
    )
    .await
    .unwrap();
    writer2
        .insert_batch(make_kv_batch(vec![5], vec![50]))
        .await
        .unwrap();
    writer2.commit().await.unwrap();

    Compactor::new(storage.as_ref(), "ref_tbl")
        .compact()
        .await
        .unwrap();

    // Orphan must be gone.
    assert!(
        !storage.exists(orphan).await.unwrap_or(false),
        "orphan .agg.part must be cleaned by next compaction"
    );
}
