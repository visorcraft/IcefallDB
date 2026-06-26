use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::commit_delta::CommitKind;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{MatchLoc, Writer};
use std::sync::Arc;

fn int_schema() -> Schema {
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
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn int_batch(vals: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals))]).unwrap()
}

#[tokio::test]
async fn append_commit_returns_delta_with_added_row_group() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "t", int_schema())
        .await
        .unwrap();
    writer.insert_batch(int_batch(vec![1, 2, 3])).await.unwrap();
    let delta = writer.commit().await.unwrap();

    assert_eq!(delta.kind, CommitKind::Append);
    assert!(!delta.is_noop());
    assert_eq!(delta.new_sequence, delta.previous_sequence + 1);
    assert_eq!(delta.added_row_groups().len(), 1);
    assert!(delta.removed_fragment_ids().is_empty());
    assert!(delta.updated_fragments().is_empty());
}

#[tokio::test]
async fn empty_commit_returns_noop_delta() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "t", int_schema())
        .await
        .unwrap();
    writer.insert_batch(int_batch(vec![1])).await.unwrap();
    writer.commit().await.unwrap();

    let delta = writer.commit().await.unwrap();
    assert!(delta.is_noop());
    assert_eq!(delta.kind, CommitKind::Noop);
    assert_eq!(delta.previous_sequence, delta.new_sequence);
}

#[tokio::test]
async fn replace_commit_returns_delta_with_replacement() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "t", int_schema())
        .await
        .unwrap();
    writer.insert_batch(int_batch(vec![1, 2])).await.unwrap();
    writer.commit().await.unwrap();

    writer.insert_batch(int_batch(vec![3, 4])).await.unwrap();
    let delta = writer.replace().await.unwrap();

    assert_eq!(delta.kind, CommitKind::Replace);
    assert!(!delta.is_noop());
    assert_eq!(delta.added_row_groups().len(), 1);
    assert_eq!(delta.removed_fragment_ids().len(), 1);
}

#[tokio::test]
async fn delete_commit_returns_delta_with_updated_fragment() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "t", int_schema())
        .await
        .unwrap();
    writer.insert_batch(int_batch(vec![1, 2, 3])).await.unwrap();
    writer.commit().await.unwrap();

    let frag_id = {
        let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "t")
            .await
            .unwrap();
        catalog.latest_manifest().unwrap().row_groups[0].fragment_id
    };

    let delta = writer
        .commit_deletes(std::collections::HashMap::from([(frag_id, vec![0u32])]))
        .await
        .unwrap();

    assert_eq!(delta.kind, CommitKind::Delete);
    assert!(!delta.is_noop());
    assert_eq!(delta.updated_fragments().len(), 1);
    assert_eq!(delta.updated_fragments()[0].fragment_id, frag_id);
    assert_eq!(delta.updated_fragments()[0].new_deleted_count, 1);
    assert!(delta.added_row_groups().is_empty());
    assert!(delta.removed_fragment_ids().is_empty());
}

#[tokio::test]
async fn update_commit_returns_delta_with_patch_and_tombstone() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "t", int_schema())
        .await
        .unwrap();
    writer
        .insert_batch(int_batch(vec![10, 20, 30]))
        .await
        .unwrap();
    let delta = writer.commit().await.unwrap();
    let orig_fragment_id = delta.added_row_groups()[0].fragment_id;

    // Update the second row (offset 1, row_id 1).
    let update_delta = writer
        .commit_update(
            int_batch(vec![25]),
            vec![MatchLoc {
                fragment_id: orig_fragment_id,
                offset: 1,
                row_id: 1,
            }],
            &[],
        )
        .await
        .unwrap();

    assert_eq!(update_delta.kind, CommitKind::Update);
    assert!(!update_delta.is_noop());
    // The patch fragment is added; the original fragment is tombstoned.
    assert_eq!(update_delta.added_row_groups().len(), 1);
    assert_eq!(update_delta.updated_fragments().len(), 1);
    assert_eq!(
        update_delta.updated_fragments()[0].fragment_id,
        orig_fragment_id
    );
    assert_eq!(update_delta.updated_fragments()[0].new_deleted_count, 1);
}

#[tokio::test]
async fn merge_commit_returns_delta_with_patch_insert_and_tombstone() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "t", int_schema())
        .await
        .unwrap();
    writer
        .insert_batch(int_batch(vec![10, 20, 30]))
        .await
        .unwrap();
    let delta = writer.commit().await.unwrap();
    let orig_fragment_id = delta.added_row_groups()[0].fragment_id;

    // Update row 1 and insert a new row in one atomic MERGE.
    let merge_delta = writer
        .commit_merge(
            int_batch(vec![25]),
            vec![MatchLoc {
                fragment_id: orig_fragment_id,
                offset: 1,
                row_id: 1,
            }],
            &[],
            int_batch(vec![40]),
        )
        .await
        .unwrap();

    assert_eq!(merge_delta.kind, CommitKind::Merge);
    assert!(!merge_delta.is_noop());
    // One patch fragment + one insert fragment are added.
    assert_eq!(merge_delta.added_row_groups().len(), 2);
    assert_eq!(merge_delta.updated_fragments().len(), 1);
    assert_eq!(
        merge_delta.updated_fragments()[0].fragment_id,
        orig_fragment_id
    );
}
