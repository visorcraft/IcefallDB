use arrow::array::{AsArray, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::compaction::{CompactionOptions, Compactor};
use icefalldb_core::database_catalog::DatabaseCatalog;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::{MatchLoc, Writer};
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

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
        row_group_target_rows: 2,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

#[tokio::test]
async fn test_index_accelerates_equality_query() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let schema = event_schema();

    let catalog = DatabaseCatalog::new(storage.clone());
    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();
    catalog
        .create_index_definition(&guard, "cat_idx", "events", "cat", "btree")
        .await
        .unwrap();

    // Insert data across two row groups.
    let mut writer = Writer::new(storage.clone(), "events", schema)
        .await
        .unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![
            Arc::new(Int64Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let ctx = icefalldb_session(2, 8192);
    let provider = IcefallDBTableProvider::new(storage, "events", ProviderConfig::default())
        .await
        .unwrap();
    ctx.register_table("events", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT id FROM events WHERE cat = 'a'")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let mut ids: Vec<i64> = Vec::new();
    for batch in &batches {
        let array = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..array.len() {
            ids.push(array.value(i));
        }
    }
    ids.sort();
    assert_eq!(ids, vec![1, 3]);
}

/// Regression: a lazily-opened provider whose manifest is advanced by an
/// EXTERNAL writer (not via `apply_committed_delta`) must re-pin its
/// indexes/statistics/sequence on the next query, so an indexed-equality lookup
/// finds rows committed after open instead of dropping the new fragment.
#[tokio::test]
async fn test_lazy_open_repins_on_external_manifest_advance() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let schema = event_schema();

    let catalog = DatabaseCatalog::new(storage.clone());
    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();
    catalog
        .create_index_definition(&guard, "cat_idx", "events", "cat", "btree")
        .await
        .unwrap();
    drop(guard);

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let make_batch = |ids: Vec<i64>, cats: Vec<&'static str>| {
        RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(StringArray::from(
                    cats.into_iter().map(Some).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap()
    };

    // Commit 1 (sequence S): ids 1,2 → cats a,b. One fragment.
    let mut writer = Writer::new(storage.clone(), "events", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2], vec!["a", "b"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Open the provider LAZILY at S, before the external advance. Force the
    // custom-scan secondary-index path (no tiny-table cache) so the index is
    // actually consulted — that is where stale indexes would drop the new
    // fragment.
    let config = ProviderConfig {
        native_parquet_threshold: usize::MAX,
        tiny_table_cache_threshold_rows: 0,
        tiny_table_cache_threshold_bytes: 0,
        ..ProviderConfig::default()
    };
    let provider = Arc::new(
        IcefallDBTableProvider::new(storage.clone(), "events", config)
            .await
            .unwrap(),
    );
    let ctx = icefalldb_session(2, 8192);
    ctx.register_table("events", provider).unwrap();

    // EXTERNAL writer advances the manifest to S+1 with a new cat='a' row (id 3),
    // WITHOUT going through the provider's apply_committed_delta.
    let mut writer = Writer::new(storage.clone(), "events", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![3, 4], vec!["a", "b"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // The provider's first indexed-equality query must re-pin (indexes@S+1) and
    // surface id=3. Without re-pin it would use indexes@S and drop the new
    // fragment, returning only id=1.
    let df = ctx
        .sql("SELECT id FROM events WHERE cat = 'a' ORDER BY id")
        .await
        .unwrap();
    let mut ids: Vec<i64> = Vec::new();
    for batch in df.collect().await.unwrap() {
        let a = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..a.len() {
            ids.push(a.value(i));
        }
    }
    assert_eq!(
        ids,
        vec![1, 3],
        "re-pin must surface the externally-added cat='a' row"
    );
}

/// `WHERE _rowid IN (...)` returns exactly the rows with those stable row-ids,
/// across fragments, exercising the `_rowid` RowSelection pushdown. Correctness
/// here is the same with or without the pushdown (the pushed filter is retained
/// as a post-scan guard); this pins the result so a pushdown bug can't slip by.
#[tokio::test]
async fn test_rowid_in_pushdown_returns_correct_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let schema = event_schema();

    let catalog = DatabaseCatalog::new(storage.clone());
    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();
    drop(guard);

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    // Fragment 0: ids 10,20 → row_ids 0,1. Fragment 1: ids 30,40 → row_ids 2,3.
    let mut writer = Writer::new(storage.clone(), "events", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(
            RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![
                    Arc::new(Int64Array::from(vec![10, 20])),
                    Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    writer.commit().await.unwrap();
    let mut writer = Writer::new(storage.clone(), "events", schema)
        .await
        .unwrap();
    writer
        .insert_batch(
            RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![
                    Arc::new(Int64Array::from(vec![30, 40])),
                    Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let ctx = icefalldb_session(2, 8192);
    let provider = IcefallDBTableProvider::new(storage, "events", ProviderConfig::default())
        .await
        .unwrap();
    ctx.register_table("events", Arc::new(provider)).unwrap();

    // row_id 1 (id 20, frag 0) + row_id 2 (id 30, frag 1).
    let df = ctx
        .sql("SELECT id FROM events WHERE _rowid IN (1, 2) ORDER BY id")
        .await
        .unwrap();
    let mut ids: Vec<i64> = Vec::new();
    for batch in df.collect().await.unwrap() {
        let array = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..array.len() {
            ids.push(array.value(i));
        }
    }
    assert_eq!(
        ids,
        vec![20, 30],
        "_rowid IN (1,2) must return ids 20 and 30"
    );

    // A target that maps to one fragment plus an absent row-id: only the present
    // one comes back (the absent id selects nothing in any fragment).
    let rows = ctx
        .sql("SELECT id FROM events WHERE _rowid IN (3, 999) ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut got: Vec<i64> = Vec::new();
    for b in &rows {
        let a = b
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..a.len() {
            got.push(a.value(i));
        }
    }
    assert_eq!(
        got,
        vec![40],
        "_rowid IN (3, 999) must return exactly id 40"
    );
}

#[tokio::test]
async fn test_index_accelerates_in_list_query() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let schema = event_schema();

    let catalog = DatabaseCatalog::new(storage.clone());
    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();
    catalog
        .create_index_definition(&guard, "cat_idx", "events", "cat", "btree")
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
        Arc::clone(&arrow_schema),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let batch = RecordBatch::try_new(
        Arc::clone(&arrow_schema),
        vec![
            Arc::new(Int64Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec![Some("a"), Some("b")])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let ctx = icefalldb_session(2, 8192);
    let provider = IcefallDBTableProvider::new(storage, "events", ProviderConfig::default())
        .await
        .unwrap();
    ctx.register_table("events", Arc::new(provider)).unwrap();

    let df = ctx
        .sql("SELECT id FROM events WHERE cat IN ('a')")
        .await
        .unwrap();
    let batches = df.collect().await.unwrap();

    let mut ids: Vec<i64> = Vec::new();
    for batch in &batches {
        let array = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..array.len() {
            ids.push(array.value(i));
        }
    }
    ids.sort();
    assert_eq!(ids, vec![1, 3]);
}

/// End-to-end: a table with a secondary index and a fragment carrying a
/// deletion vector. Running an equality query on the indexed column through the
/// real provider must make the scan result set, the COUNT, and the index-pruned
/// rows all agree with the deletion vector:
///   * the deleted row is absent from the result,
///   * the COUNT equals the number of live matching rows, and
///   * the index resolves the indexed value to the live row_id only.
///
/// This proves scan + count + index agree under deletions through the full
/// query path (`IcefallDBTableProvider` + `SessionContext`).
#[tokio::test]
async fn e2e_index_count_and_scan_agree_under_deletion() {
    use icefalldb_core::catalog::Catalog;
    use icefalldb_core::index::{build_btree_index, IndexDefinition};
    use icefalldb_core::metadata::Manifest;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::DeletionVector;

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    // Single row group: id=[1,2,3,4], cat=["a","b","a","b"].
    // Large target rows so all four rows land in ONE fragment (one DV applies).
    let mut schema = event_schema();
    schema.row_group_target_rows = 1000;

    let dbcat = DatabaseCatalog::new(storage.clone());
    let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
    dbcat.create_table(&guard, "events", &schema).await.unwrap();
    dbcat
        .create_index_definition(&guard, "cat_idx", "events", "cat", "btree")
        .await
        .unwrap();
    drop(guard);

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let mut writer = Writer::new(storage.clone(), "events", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(
            RecordBatch::try_new(
                Arc::clone(&arrow_schema),
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
            .unwrap(),
        )
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // ── Inject a deletion vector deleting physical offset 2 (id=3, cat="a") ──
    // and rebuild the index manually to simulate a post-deletion snapshot.
    // NOTE: this does NOT replicate the real mutation/commit path (Writer::delete
    // + a proper commit) — that lands in P1.  Instead the test injects the
    // deletion vector directly into a new manifest and rebuilds the index by
    // hand, recording the new generation into the SAME manifest, so the provider
    // sees a consistent (scan plan, index) pair at sequence next_seq.
    let pointer_bytes = storage.read("events/_manifest.json").await.unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
    let seq = pointer["latest"].as_u64().unwrap();
    let manifest_path = format!("events/{}", Manifest::filename(seq));
    let manifest_bytes = storage.read(&manifest_path).await.unwrap();
    let mut manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();

    let del_rel = "_deletions/rg0__v1.del";
    let mut dv = DeletionVector::default();
    dv.union_offsets([2u32]); // physical offset 2 → id=3, cat="a"
    storage
        .write(&format!("events/{}", del_rel), &dv.serialize())
        .await
        .unwrap();

    assert!(!manifest.row_groups.is_empty());
    manifest.row_groups[0].deletes = Some(del_rel.to_string());
    manifest.row_groups[0].deleted_count = dv.cardinality();

    // Bump to the next sequence and rebuild the index honoring the DV.
    let next_seq = seq + 1;
    manifest.sequence = next_seq;

    let definition = IndexDefinition {
        name: "cat_idx".into(),
        table: "events".into(),
        column: "cat".into(),
        unique: false,
    };
    let rebuilt = build_btree_index(storage.as_ref(), &definition, &manifest)
        .await
        .unwrap();
    let rel_path = rebuilt.save_versioned(storage.as_ref()).await.unwrap();
    manifest.index_generations.insert(
        "cat_idx".into(),
        icefalldb_core::metadata::manifest::IndexRef {
            base: Some(rel_path),
            deltas: vec![],
        },
    );

    // The index built over the DV-bearing snapshot must already exclude the
    // deleted row: "a" resolves to exactly ONE live row_id.
    assert_eq!(
        rebuilt.lookup("a").len(),
        1,
        "index for the post-deletion snapshot must resolve 'a' to the single live row"
    );

    manifest.checksum = String::new();
    manifest.checksum = manifest.compute_checksum().unwrap();
    let new_manifest_path = format!("events/{}", Manifest::filename(next_seq));
    storage
        .write(&new_manifest_path, &serde_json::to_vec(&manifest).unwrap())
        .await
        .unwrap();
    storage
        .write(
            "events/_manifest.json",
            &serde_json::to_vec(&serde_json::json!({"latest": next_seq})).unwrap(),
        )
        .await
        .unwrap();

    // ── Confirm the provider's pinned snapshot index generation reflects the
    // deletion (load_index_by_ref on the manifest's index_generations).
    let pinned = {
        let cat = Catalog::load(storage.as_ref(), "events").await.unwrap();
        let m = cat.latest_manifest().unwrap();
        assert_eq!(m.sequence, next_seq);
        let r = m.index_generations.get("cat_idx").unwrap();
        icefalldb_core::index::load_index_by_ref(storage.as_ref(), "events", "cat_idx", r)
            .await
            .unwrap()
            .unwrap()
    };
    assert_eq!(
        pinned.lookup("a").len(),
        1,
        "provider's pinned index generation must reflect the deletion"
    );

    // ── Run the equality query through the full provider + SessionContext ──
    // Force the custom scan path so deletion vectors are honored during decode.
    let config = ProviderConfig {
        native_parquet_threshold: usize::MAX,
        tiny_table_cache_threshold_rows: 0,
        tiny_table_cache_threshold_bytes: 0,
        wal_mode: true,
        ..ProviderConfig::default()
    };
    let provider = IcefallDBTableProvider::new(storage.clone(), "events", config)
        .await
        .unwrap();
    let ctx = icefalldb_session(2, 8192);
    ctx.register_table("events", Arc::new(provider)).unwrap();

    // (1) Result set: only id=1 survives for cat='a'.
    let batches = ctx
        .sql("SELECT id FROM events WHERE cat = 'a'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut ids: Vec<i64> = Vec::new();
    for batch in &batches {
        let array = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..array.len() {
            ids.push(array.value(i));
        }
    }
    ids.sort();
    assert_eq!(
        ids,
        vec![1],
        "scan: deleted id=3 (cat='a') must be absent; only live id=1 remains"
    );

    // (2) COUNT must agree with the live-row scan and the index.
    let count_batches = ctx
        .sql("SELECT COUNT(*) AS c FROM events WHERE cat = 'a'")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = count_batches[0]
        .column_by_name("c")
        .unwrap()
        .as_primitive::<arrow::datatypes::Int64Type>()
        .value(0);
    assert_eq!(
        count, 1,
        "COUNT(cat='a') must equal the single live matching row"
    );
    assert_eq!(
        count as usize,
        pinned.lookup("a").len(),
        "COUNT, scan, and index-pruned row count must all agree under deletion"
    );

    // (3) A query on the non-deleted value is unaffected: cat='b' → id=2,4.
    let b_batches = ctx
        .sql("SELECT id FROM events WHERE cat = 'b' ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut b_ids: Vec<i64> = Vec::new();
    for batch in &b_batches {
        let array = batch
            .column_by_name("id")
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..array.len() {
            b_ids.push(array.value(i));
        }
    }
    assert_eq!(
        b_ids,
        vec![2, 4],
        "cat='b' rows are untouched by the deletion"
    );
}

/// Verify that `IcefallDBTableProvider` reads the manifest pointer exactly once
/// and uses the SAME manifest for both the scan plan and the secondary indexes.
///
/// `pinned_sequence()` is set from the manifest that was passed to
/// `load_snapshot_indexes`, which is the same object the scan plan was built
/// from.  For a non-empty table the manifest sequence must equal the `snapshot`
/// field embedded in every `PlannedRowGroup` (which the `Reader` stamped in
/// during `scan_internal`).
///
/// This is the focused regression test for the TOCTOU issue (I-A): if a second
/// catalog read were performed inside `load_snapshot_indexes` and observed a
/// newer manifest S+1, `pinned_sequence` would disagree with the row-group
/// `snapshot` values.
#[tokio::test]
async fn index_is_snapshot_scoped() {
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::Writer;
    use std::time::Duration;

    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let mut schema = event_schema();
    schema.row_group_target_rows = 1000;

    let dbcat = icefalldb_core::database_catalog::DatabaseCatalog::new(storage.clone());
    let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
    dbcat.create_table(&guard, "events", &schema).await.unwrap();
    dbcat
        .create_index_definition(&guard, "cat_idx", "events", "cat", "btree")
        .await
        .unwrap();
    drop(guard);

    let arrow_schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int64, false),
        arrow::datatypes::Field::new("cat", arrow::datatypes::DataType::Utf8, true),
    ]));
    let mut writer = Writer::new(storage.clone(), "events", schema)
        .await
        .unwrap();
    writer
        .insert_batch(
            RecordBatch::try_new(
                Arc::clone(&arrow_schema),
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
            .unwrap(),
        )
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let provider =
        icefalldb_query::IcefallDBTableProvider::new(storage, "events", ProviderConfig::default())
            .await
            .unwrap();

    let pinned = provider.pinned_sequence();
    assert!(
        pinned > 0,
        "a committed table must have a non-zero sequence"
    );

    // Every row group in the scan plan must carry the same snapshot sequence as
    // the one the indexes were pinned to.  If the provider had performed a
    // second manifest-pointer read for the index load and observed a newer
    // manifest, this assertion would catch the split.
    let row_group_sequences: Vec<u64> = provider
        .scan_plan()
        .await
        .unwrap()
        .row_groups
        .iter()
        .map(|rg| rg.snapshot)
        .collect();
    assert!(
        !row_group_sequences.is_empty(),
        "scan plan must have at least one row group"
    );
    for &seq in &row_group_sequences {
        assert_eq!(
            seq, pinned,
            "row-group snapshot {seq} must equal pinned_sequence {pinned}: \
             scan plan and index set must come from the same manifest read"
        );
    }
}

// ---------------------------------------------------------------------------
// Guard: indexed equality query returns correct results after compaction
// ---------------------------------------------------------------------------

/// Collect all values from the `col` column across a set of DataFusion batches.
fn collect_int64_col(batches: &[RecordBatch], col: &str) -> Vec<i64> {
    let mut out = Vec::new();
    for batch in batches {
        let arr = batch
            .column_by_name(col)
            .unwrap()
            .as_primitive::<arrow::datatypes::Int64Type>();
        for i in 0..arr.len() {
            out.push(arr.value(i));
        }
    }
    out
}

/// Build a fresh `IcefallDBTableProvider` pinned to the current manifest, so
/// the post-compaction query sees the compacted snapshot.
async fn fresh_provider(
    storage: Arc<dyn icefalldb_core::storage::Storage>,
    table: &str,
) -> IcefallDBTableProvider {
    // Disable native Parquet (use the custom scan that honours deletion vectors)
    // and the tiny-table cache so we exercise the real scan path.
    let config = ProviderConfig {
        native_parquet_threshold: usize::MAX,
        tiny_table_cache_threshold_rows: 0,
        tiny_table_cache_threshold_bytes: 0,
        wal_mode: true,
        ..ProviderConfig::default()
    };
    IcefallDBTableProvider::new(storage, table, config)
        .await
        .unwrap()
}

/// Correctness proof that the index carried forward by compaction is valid:
///
/// 1. Build a table with a btree index on `cat`.
/// 2. Apply a DELETE (removes row id=1, cat="a") and an UPDATE (changes id=4
///    cat from "b" to "z").
/// 3. Run `WHERE cat = 'a'` BEFORE compaction — must return id=3 and id=5 only.
/// 4. Run compaction (force=true to collapse multiple fragments).
/// 5. Re-register a fresh provider pinned to the compacted snapshot.
/// 6. Run the SAME query AFTER compaction — must return the same result.
/// 7. Run `WHERE cat = 'b'` — must NOT return id=4 (was updated to "z").
/// 8. Run `WHERE cat = 'z'` — must return id=4 (the updated value).
/// 9. Run `WHERE cat = 'a'` for the deleted value id=1 — must NOT appear.
#[tokio::test]
async fn indexed_query_correct_after_compaction() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn icefalldb_core::storage::Storage> =
        Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column::new("id", "int64", false),
            Column::new("cat", "utf8", true),
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None, // Small target so the two inserts land in separate row groups.
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };

    let dbcat = DatabaseCatalog::new(storage.clone());
    let guard = dbcat.acquire_lock(Duration::from_secs(10)).await.unwrap();
    dbcat.create_table(&guard, "evt", &schema).await.unwrap();
    dbcat
        .create_index_definition(&guard, "cat_idx", "evt", "cat", "btree")
        .await
        .unwrap();
    drop(guard);

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));

    // Insert first fragment: id=[1,2,3], cat=["a","b","a"]
    let mut writer = Writer::new(storage.clone(), "evt", schema.clone())
        .await
        .unwrap();
    writer
        .insert_batch(
            RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![
                    Arc::new(Int64Array::from(vec![1i64, 2, 3])),
                    Arc::new(StringArray::from(vec![Some("a"), Some("b"), Some("a")])),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Insert second fragment: id=[4,5,6], cat=["b","a","c"]
    writer
        .insert_batch(
            RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![
                    Arc::new(Int64Array::from(vec![4i64, 5, 6])),
                    Arc::new(StringArray::from(vec![Some("b"), Some("a"), Some("c")])),
                ],
            )
            .unwrap(),
        )
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Retrieve fragment ids so we can build MatchLoc for the UPDATE.
    let cat_before = {
        let cat = icefalldb_core::catalog::Catalog::load(storage.as_ref(), "evt")
            .await
            .unwrap();
        cat.latest_manifest().unwrap().clone()
    };
    let frag0 = cat_before.row_groups[0].fragment_id;
    let frag1 = cat_before.row_groups[1].fragment_id;

    // DELETE id=1 (frag 0, offset 0, row_id 0).
    let mut wdel = Writer::new(storage.clone(), "evt", schema.clone())
        .await
        .unwrap();
    wdel.commit_deletes(HashMap::from([(frag0, vec![0u32])]))
        .await
        .unwrap();

    // UPDATE id=4 (frag 1, offset 0, row_id 3): cat "b" → "z".
    // row_ids in frag 1 start at 3 (frag 0 had rows 0,1,2).
    let mut wupd = Writer::new(storage.clone(), "evt", schema.clone())
        .await
        .unwrap();
    wupd.commit_update(
        RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![4i64])),
                Arc::new(StringArray::from(vec![Some("z")])),
            ],
        )
        .unwrap(),
        vec![MatchLoc {
            fragment_id: frag1,
            offset: 0,
            row_id: 3,
        }],
        &["cat".to_string()],
    )
    .await
    .unwrap();

    // ── BEFORE compaction: run indexed equality queries ─────────────────────
    let ctx_before = icefalldb_session(2, 8192);
    let provider_before = fresh_provider(storage.clone(), "evt").await;
    ctx_before
        .register_table("evt", Arc::new(provider_before))
        .unwrap();

    let mut ids_a_before = collect_int64_col(
        &ctx_before
            .sql("SELECT id FROM evt WHERE cat = 'a'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
        "id",
    );
    ids_a_before.sort();
    // id=1 is deleted (cat "a"), id=4 is updated to "z" — so only id=3 and id=5.
    assert_eq!(
        ids_a_before,
        vec![3, 5],
        "BEFORE compaction: cat='a' must return id=3 and id=5 only"
    );

    // ── Compact ────────────────────────────────────────────────────────────
    let compact_opts = CompactionOptions {
        force: true,
        lock_timeout: Duration::from_secs(30),
        ..CompactionOptions::default()
    };
    let result = Compactor::with_options(storage.as_ref(), "evt", compact_opts)
        .compact()
        .await
        .unwrap();
    assert!(result.rewrote, "compaction must have rewritten data files");

    // ── AFTER compaction: fresh provider pinned to compacted snapshot ───────
    let ctx_after = icefalldb_session(2, 8192);
    let provider_after = fresh_provider(storage.clone(), "evt").await;
    ctx_after
        .register_table("evt", Arc::new(provider_after))
        .unwrap();

    // cat='a': same two surviving rows.
    let mut ids_a_after = collect_int64_col(
        &ctx_after
            .sql("SELECT id FROM evt WHERE cat = 'a'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
        "id",
    );
    ids_a_after.sort();
    assert_eq!(
        ids_a_after, ids_a_before,
        "AFTER compaction: cat='a' must return the same rows as before"
    );

    // cat='b': id=4 was updated to cat="z", so only id=2 remains.
    let mut ids_b_after = collect_int64_col(
        &ctx_after
            .sql("SELECT id FROM evt WHERE cat = 'b'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
        "id",
    );
    ids_b_after.sort();
    assert_eq!(
        ids_b_after,
        vec![2],
        "AFTER compaction: cat='b' must NOT include id=4 (updated to 'z')"
    );

    // cat='z': must find id=4 in the new location (post-compaction fragment).
    let mut ids_z_after = collect_int64_col(
        &ctx_after
            .sql("SELECT id FROM evt WHERE cat = 'z'")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
        "id",
    );
    ids_z_after.sort();
    assert_eq!(
        ids_z_after,
        vec![4],
        "AFTER compaction: cat='z' must find id=4 in its post-compaction location"
    );

    // Deleted row id=1 must not appear for cat='a'.
    assert!(
        !ids_a_after.contains(&1),
        "AFTER compaction: deleted id=1 must not appear for cat='a'"
    );
}
