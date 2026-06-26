use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use futures::StreamExt;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::{memory::MemoryStorage, Storage};
use icefalldb_core::{Reader, Writer};

fn make_test_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".to_string(),
            r#type: "int64".to_string(),
            nullable: false,
            field_id: 0,
        }],
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_int_batch(values: Vec<i64>) -> RecordBatch {
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(values)) as ArrayRef],
    )
    .unwrap()
}

fn extract_ids(batches: &[RecordBatch]) -> Vec<i64> {
    let mut ids = Vec::new();
    for batch in batches {
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for i in 0..batch.num_rows() {
            ids.push(col.value(i));
        }
    }
    ids
}

async fn read_table(storage: &MemoryStorage, table: &str) -> Vec<RecordBatch> {
    let reader = match Reader::new(storage, table).await {
        Ok(reader) => reader,
        Err(icefalldb_core::IcefallDBError::EmptyTable(_)) => return Vec::new(),
        Err(e) => panic!("unexpected reader error: {e:?}"),
    };
    let plan = reader.scan().await.unwrap();
    let mut batches = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            batches.push(batch.unwrap());
        }
    }
    batches
}

async fn new_writer(storage: &Arc<MemoryStorage>, schema: &Schema) -> Writer {
    Writer::new(
        Arc::clone(storage) as Arc<dyn Storage>,
        "products",
        schema.clone(),
    )
    .await
    .unwrap()
}

async fn assert_matches_model(storage: &Arc<MemoryStorage>, model: &ReferenceModel) {
    assert_batches_match(
        &read_table(storage.as_ref(), "products").await,
        &model.read_all(),
    );
}

#[derive(Debug, Default)]
struct ReferenceModel {
    committed: Vec<Vec<RecordBatch>>,
    buffered: Vec<RecordBatch>,
}

impl ReferenceModel {
    fn insert_batch(&mut self, batch: RecordBatch) {
        self.buffered.push(batch);
    }

    fn commit(&mut self) {
        if !self.buffered.is_empty() {
            self.committed.push(std::mem::take(&mut self.buffered));
        }
    }

    fn crash(&mut self) {
        self.buffered.clear();
    }

    fn read_all(&self) -> Vec<RecordBatch> {
        self.committed
            .iter()
            .flat_map(|v| v.iter().cloned())
            .collect()
    }
}

fn assert_batches_match(real: &[RecordBatch], model: &[RecordBatch]) {
    let real_ids = extract_ids(real);
    let model_ids = extract_ids(model);
    assert_eq!(
        real_ids, model_ids,
        "real table IDs {:?} do not match model IDs {:?}",
        real_ids, model_ids
    );
}

#[tokio::test]
async fn test_model_insert_commit_crash_sequence() {
    let storage = Arc::new(MemoryStorage::new());
    let schema = make_test_schema();
    let mut model = ReferenceModel::default();

    // 1. Insert and commit batch A.
    let mut writer = new_writer(&storage, &schema).await;
    let batch_a = make_int_batch(vec![1, 2, 3]);
    writer.insert_batch(batch_a.clone()).await.unwrap();
    writer.commit().await.unwrap();
    model.insert_batch(batch_a);
    model.commit();
    assert_matches_model(&storage, &model).await;

    // 2. Insert batch B but crash (drop writer) before commit.
    let mut writer = new_writer(&storage, &schema).await;
    let batch_b = make_int_batch(vec![4, 5]);
    writer.insert_batch(batch_b).await.unwrap();
    drop(writer);
    model.crash();
    assert_matches_model(&storage, &model).await;

    // 3. Insert and commit batch C.
    let mut writer = new_writer(&storage, &schema).await;
    let batch_c = make_int_batch(vec![6]);
    writer.insert_batch(batch_c.clone()).await.unwrap();
    writer.commit().await.unwrap();
    model.insert_batch(batch_c);
    model.commit();
    assert_matches_model(&storage, &model).await;
}

#[tokio::test]
async fn test_model_multiple_commits() {
    let storage = Arc::new(MemoryStorage::new());
    let schema = make_test_schema();
    let mut model = ReferenceModel::default();

    let values = vec![vec![1, 2, 3], vec![4, 5], vec![6, 7, 8, 9]];
    for chunk in &values {
        let mut writer = new_writer(&storage, &schema).await;
        let batch = make_int_batch(chunk.clone());
        writer.insert_batch(batch.clone()).await.unwrap();
        writer.commit().await.unwrap();
        model.insert_batch(batch);
        model.commit();
        assert_matches_model(&storage, &model).await;
    }
}

#[tokio::test]
async fn test_model_empty_commit_is_no_op() {
    let storage = Arc::new(MemoryStorage::new());
    let schema = make_test_schema();
    let mut model = ReferenceModel::default();

    // Commit with no buffered data must not advance sequence.
    let mut writer = new_writer(&storage, &schema).await;
    writer.commit().await.unwrap();
    model.commit();
    assert_matches_model(&storage, &model).await;

    // Now commit a real batch.
    let batch = make_int_batch(vec![1, 2]);
    writer.insert_batch(batch.clone()).await.unwrap();
    writer.commit().await.unwrap();
    model.insert_batch(batch);
    model.commit();
    assert_matches_model(&storage, &model).await;
}
