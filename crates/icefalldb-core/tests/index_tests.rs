use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::catalog::Catalog;
use icefalldb_core::index::{build_btree_index, load_index, load_index_by_ref, IndexDefinition};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::Writer;
use std::sync::Arc;

fn event_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column::new("id", "int64", false),
            Column::new("cat", "utf8", true),
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

#[tokio::test]
async fn test_btree_index_lookup_prunes_row_groups() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let mut schema = event_schema();
    schema.row_group_target_rows = 2;
    let mut writer = Writer::create(storage.clone(), "events", schema)
        .await
        .unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec![
                Some("a"),
                Some("b"),
                Some("a"),
                Some("b"),
            ])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let catalog = Catalog::load(&*storage, "events").await.unwrap();
    let manifest = catalog.latest_manifest().unwrap();
    let definition = IndexDefinition {
        name: "events_cat_idx".into(),
        table: "events".into(),
        column: "cat".into(),
        unique: false,
    };
    let index = build_btree_index(&*storage, &definition, manifest)
        .await
        .unwrap();
    assert_eq!(index.lookup("a").len(), 2);
    assert!(index.lookup("c").is_empty());

    index.save(&*storage).await.unwrap();
    let loaded = load_index(&*storage, "events", "events_cat_idx")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(loaded.lookup("b").len(), 2);
}

#[tokio::test]
async fn test_index_rebuilt_after_commit() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = event_schema();

    let catalog = icefalldb_core::DatabaseCatalog::new(storage.clone());
    let guard = catalog
        .acquire_lock(std::time::Duration::from_secs(5))
        .await
        .unwrap();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();
    catalog
        .create_index_definition(&guard, "events_cat_idx", "events", "cat", "btree")
        .await
        .unwrap();

    let mut writer = Writer::new(storage.clone(), "events", schema)
        .await
        .unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(StringArray::from(vec![Some("x")])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    // The commit must have rebuilt the index INSIDE the atomic commit and
    // recorded its generation in the committed manifest's `index_generations`.
    // Load it through that generation (the real snapshot-scoped path) rather
    // than the removed legacy unversioned file.
    let core_catalog = Catalog::load(&*storage, "events").await.unwrap();
    let manifest = core_catalog.latest_manifest().unwrap();
    let index_ref = manifest
        .index_generations
        .get("events_cat_idx")
        .expect("commit must record the index generation in the manifest");
    assert!(
        index_ref.base.is_some(),
        "index generation must reference a versioned base file"
    );
    let index = load_index_by_ref(&*storage, "events", "events_cat_idx", index_ref)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(index.lookup("x").len(), 1);

    // Sanity: the manifest is self-consistent (checksum already covers the
    // populated `index_generations`, proving it was written exactly once).
    assert!(manifest.verify_checksum().unwrap());
}
