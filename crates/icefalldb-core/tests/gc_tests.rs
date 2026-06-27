use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::compaction::Compactor;
use icefalldb_core::metadata::{Column, Manifest, RowGroupEntry, Schema, SnapshotCheckpoint};
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::Writer;
use icefalldb_core::{
    dv_density, should_recompute, GarbageCollector, IcefallDBError, RECOMPUTE_DENSITY,
};
use std::sync::Arc;

fn make_schema() -> Schema {
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
        row_group_target_rows: 10,
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

async fn count_manifests(storage: &dyn Storage, table: &str) -> usize {
    storage
        .list(&format!("{}/_manifests", table))
        .await
        .unwrap_or_default()
        .len()
}

async fn list_row_group_files(storage: &dyn Storage, table: &str) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list(table)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|p| {
            let name = std::path::Path::new(p)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            name.starts_with("rg_") && (name.ends_with(".parquet") || name.ends_with(".meta"))
        })
        .collect();
    files.sort();
    files
}

#[tokio::test]
async fn test_gc_removes_orphan_row_groups_after_compaction() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
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

    let before_row_groups = list_row_group_files(storage.as_ref(), "products").await;
    assert_eq!(
        before_row_groups.len(),
        8,
        "expected four row groups (two old + two compacted) before gc"
    );

    let result = GarbageCollector::new(storage.as_ref(), "products", 1)
        .run()
        .await
        .unwrap();

    assert_eq!(result.retained_snapshots, vec![3]);

    let after_row_groups = list_row_group_files(storage.as_ref(), "products").await;
    assert_eq!(
        after_row_groups.len(),
        4,
        "only the compacted row groups should remain"
    );

    assert_eq!(
        count_manifests(storage.as_ref(), "products").await,
        1,
        "only the latest manifest should remain"
    );

    assert!(
        result.deleted.len() >= 6,
        "expected old row groups and manifests to be deleted, got {:?}",
        result.deleted
    );
}

#[tokio::test]
async fn test_gc_retains_referenced_files() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..3 {
        let start = i * 10 + 1;
        writer
            .insert_batch(make_batch((start..start + 10).collect()))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    let result = GarbageCollector::new(storage.as_ref(), "products", 2)
        .run()
        .await
        .unwrap();

    assert_eq!(result.retained_snapshots, vec![3, 2]);

    let latest = read_manifest(storage.as_ref(), "products", 3).await;
    let referenced: std::collections::HashSet<String> = latest
        .row_groups
        .iter()
        .flat_map(|e| [e.data.clone(), e.meta.clone()])
        .collect();

    for file in &referenced {
        assert!(
            storage.exists(&format!("products/{}", file)).await.unwrap(),
            "referenced file {} should still exist",
            file
        );
    }

    assert!(
        !storage
            .exists(&format!("products/{}", Manifest::filename(1)))
            .await
            .unwrap(),
        "oldest manifest should be deleted"
    );
}

#[tokio::test]
async fn test_gc_retains_configured_snapshots() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..4 {
        let start = i * 10 + 1;
        writer
            .insert_batch(make_batch((start..start + 10).collect()))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    let result = GarbageCollector::new(storage.as_ref(), "products", 2)
        .run()
        .await
        .unwrap();

    // cutoff = 4 - (2 - 1) = 3, so snapshots 3 and 4 are retained.
    assert_eq!(result.retained_snapshots, vec![4, 3]);
    assert_eq!(count_manifests(storage.as_ref(), "products").await, 2);
    assert!(
        !storage
            .exists(&format!("products/{}", Manifest::filename(1)))
            .await
            .unwrap(),
        "snapshot 1 should be deleted"
    );
    assert!(
        !storage
            .exists(&format!("products/{}", Manifest::filename(2)))
            .await
            .unwrap(),
        "snapshot 2 should be deleted"
    );
}

#[tokio::test]
async fn test_gc_retention_uses_sequence_cutoff() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..4 {
        let start = i * 10 + 1;
        writer
            .insert_batch(make_batch((start..start + 10).collect()))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    // Corrupt snapshot 3 so it is invalid; the catalog still points to sequence 4.
    storage
        .write(&format!("products/{}", Manifest::filename(3)), b"corrupted")
        .await
        .unwrap();

    let result = GarbageCollector::new(storage.as_ref(), "products", 2)
        .run()
        .await
        .unwrap();

    // cutoff = 4 - (2 - 1) = 3. Snapshot 3 is invalid but its file is retained
    // because its sequence number is above the cutoff; snapshot 2 is deleted.
    assert_eq!(result.retained_snapshots, vec![4]);
    assert!(!storage
        .exists(&format!("products/{}", Manifest::filename(1)))
        .await
        .unwrap());
    assert!(!storage
        .exists(&format!("products/{}", Manifest::filename(2)))
        .await
        .unwrap());
    assert!(storage
        .exists(&format!("products/{}", Manifest::filename(3)))
        .await
        .unwrap());
    assert!(storage
        .exists(&format!("products/{}", Manifest::filename(4)))
        .await
        .unwrap());
}

#[tokio::test]
async fn test_gc_removes_stale_intents_and_parts() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    writer
        .insert_batch(make_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Add stale staging artifacts.
    storage
        .write(
            "products/_staging/intents/stale.json",
            serde_json::json!({
                "txn_id": "txn_stale",
                "files": ["rg_stale.parquet", "rg_stale.meta"],
            })
            .to_string()
            .as_bytes(),
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_staging/incoming/rg_stale.parquet.part",
            b"stale-part",
        )
        .await
        .unwrap();
    storage
        .write(
            "products/_staging/compact/rg_stale.meta.part",
            b"stale-part",
        )
        .await
        .unwrap();

    let result = GarbageCollector::new(storage.as_ref(), "products", 3)
        .run()
        .await
        .unwrap();

    assert!(!storage
        .exists("products/_staging/intents/stale.json")
        .await
        .unwrap());
    assert!(!storage
        .exists("products/_staging/incoming/rg_stale.parquet.part")
        .await
        .unwrap());
    assert!(!storage
        .exists("products/_staging/compact/rg_stale.meta.part")
        .await
        .unwrap());

    let deleted_set: std::collections::HashSet<String> = result.deleted.into_iter().collect();
    assert!(deleted_set.contains("products/_staging/intents/stale.json"));
    assert!(deleted_set.contains("products/_staging/incoming/rg_stale.parquet.part"));
    assert!(deleted_set.contains("products/_staging/compact/rg_stale.meta.part"));

    // The committed row group and manifest must survive.
    let latest = read_manifest(storage.as_ref(), "products", 1).await;
    assert!(storage
        .exists(&format!("products/{}", latest.row_groups[0].data))
        .await
        .unwrap());
}

#[tokio::test]
async fn test_gc_no_op_on_empty_table() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    Writer::create(Arc::clone(&storage), "empty", make_schema())
        .await
        .unwrap();

    let result = GarbageCollector::new(storage.as_ref(), "empty", 3)
        .run()
        .await
        .unwrap();

    assert!(result.deleted.is_empty());
    assert!(result.retained_snapshots.is_empty());
}

#[tokio::test]
async fn test_gc_fails_on_missing_table() {
    let storage = MemoryStorage::new();
    let err = GarbageCollector::new(&storage, "missing", 3)
        .run()
        .await
        .unwrap_err();

    assert!(matches!(err, IcefallDBError::TableNotFound(_)), "{err}");
}

async fn read_manifest(storage: &dyn Storage, table: &str, seq: u64) -> Manifest {
    let data = storage
        .read(&format!("{}/{}", table, Manifest::filename(seq)))
        .await
        .unwrap();
    serde_json::from_slice(&data).unwrap()
}

#[tokio::test]
async fn test_gc_repairs_missing_pointer() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..3 {
        let start = i * 10 + 1;
        writer
            .insert_batch(make_batch((start..start + 10).collect()))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    // Delete the manifest pointer but leave the snapshots intact. The table
    // still exists, so GC must fall back to the highest valid snapshot and
    // repair the pointer (the same recovery as a corrupt pointer), not report
    // the table as missing.
    storage.delete("products/_manifest.json").await.unwrap();

    let result = GarbageCollector::new(storage.as_ref(), "products", 1)
        .run()
        .await
        .unwrap();

    assert_eq!(result.retained_snapshots, vec![3]);

    let pointer_data = storage.read("products/_manifest.json").await.unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    assert_eq!(pointer["latest"], 3);
}

#[tokio::test]
async fn test_gc_repairs_corrupt_pointer_to_missing_snapshot() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..3 {
        let start = i * 10 + 1;
        writer
            .insert_batch(make_batch((start..start + 10).collect()))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    // Point the manifest to a snapshot that does not exist.
    storage
        .write(
            "products/_manifest.json",
            serde_json::json!({ "latest": 99 }).to_string().as_bytes(),
        )
        .await
        .unwrap();

    let result = GarbageCollector::new(storage.as_ref(), "products", 1)
        .run()
        .await
        .unwrap();

    // GC falls back to the highest valid sequence and repairs the pointer.
    assert_eq!(result.retained_snapshots, vec![3]);

    let pointer_data = storage.read("products/_manifest.json").await.unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    assert_eq!(pointer["latest"], 3);

    assert!(!storage
        .exists(&format!("products/{}", Manifest::filename(1)))
        .await
        .unwrap());
    assert!(!storage
        .exists(&format!("products/{}", Manifest::filename(2)))
        .await
        .unwrap());
    assert!(storage
        .exists(&format!("products/{}", Manifest::filename(3)))
        .await
        .unwrap());
}

#[tokio::test]
async fn test_gc_retain_zero_keeps_all_snapshots() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), "products", schema)
        .await
        .unwrap();

    for i in 0..3 {
        let start = i * 10 + 1;
        writer
            .insert_batch(make_batch((start..start + 10).collect()))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    let result = GarbageCollector::new(storage.as_ref(), "products", 0)
        .run()
        .await
        .unwrap();

    assert_eq!(result.retained_snapshots, vec![3, 2, 1]);
    assert_eq!(count_manifests(storage.as_ref(), "products").await, 3);

    for seq in 1..=3 {
        assert!(storage
            .exists(&format!("products/{}", Manifest::filename(seq)))
            .await
            .unwrap());
    }
}

#[tokio::test]
async fn test_gc_removes_root_manifest_tmp_file() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let mut writer = Writer::new(Arc::clone(&storage), "products", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Simulate a leaked pointer temp file from a crash.
    storage
        .write("products/_manifest.json.tmp", b"{}")
        .await
        .unwrap();

    let result = GarbageCollector::new(storage.as_ref(), "products", 1)
        .run()
        .await
        .unwrap();

    assert!(
        result
            .deleted
            .contains(&"products/_manifest.json.tmp".to_string()),
        "expected _manifest.json.tmp to be deleted, got {:?}",
        result.deleted
    );
    assert!(!storage.exists("products/_manifest.json.tmp").await.unwrap());
}

// ── Helpers for .agg / dv_density tests ─────────────────────────────────────

/// Schema with a numeric `v` column so .agg sidecars are written by the
/// compactor.
fn make_v_schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "v".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_v_batch(vals: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("v", DataType::Int64, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int64Array::from(vals))]).unwrap()
}

async fn read_latest_manifest_gc(storage: &dyn Storage, table: &str) -> Manifest {
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

/// Collect all .agg file paths under table root.
async fn list_agg_files(storage: &dyn Storage, table: &str) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list(table)
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|p| {
            std::path::Path::new(p)
                .file_name()
                .and_then(|s| s.to_str())
                .map(|f| f.starts_with("rg_") && f.ends_with(".agg"))
                .unwrap_or(false)
        })
        .collect();
    files.sort();
    files
}

/// Collect all .del files under _deletions/.
async fn list_del_files(storage: &dyn Storage, table: &str) -> Vec<String> {
    let mut files: Vec<String> = storage
        .list(&format!("{}/_deletions", table))
        .await
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.ends_with(".del"))
        .collect();
    files.sort();
    files
}

// ── Part A: orphan .agg GC ───────────────────────────────────────────────────

/// After compaction supersedes pre-compaction fragments, a GC run that drops
/// the old snapshot must delete the pre-compaction .agg sidecars while leaving
/// the post-compaction .agg sidecars intact.
#[tokio::test]
async fn test_gc_collects_orphan_agg_files_after_compaction() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    // Two inserts → two fragments each with a .parquet/.meta/.agg triple.
    let mut writer = Writer::new(Arc::clone(&storage), "gc_agg", make_v_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_v_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_v_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Compact: produces new fragments (with new .agg files).
    Compactor::new(storage.as_ref(), "gc_agg")
        .compact()
        .await
        .unwrap();

    // Capture all .agg paths present after compaction (old + new).
    let all_agg_before = list_agg_files(storage.as_ref(), "gc_agg").await;

    // The new (post-compact) manifest's .agg paths.
    let manifest_after = read_latest_manifest_gc(storage.as_ref(), "gc_agg").await;
    let referenced_agg: std::collections::HashSet<String> = manifest_after
        .row_groups
        .iter()
        .filter_map(|e| e.agg.as_ref())
        .map(|a| format!("gc_agg/{}", a))
        .collect();

    // Sanity: there must be .agg files referenced by the post-compact manifest.
    assert!(
        !referenced_agg.is_empty(),
        "compaction must produce .agg files for output fragments"
    );

    // Before GC there should be more .agg files than referenced (old + new).
    // (If compaction happened to reuse the same UUIDs this assertion would be
    // vacuous, but in practice new fragments get new UUIDs.)
    let orphan_count = all_agg_before
        .iter()
        .filter(|p| !referenced_agg.contains(*p))
        .count();
    // There may be 0 orphan .agg files at this point if the writer didn't produce
    // them for the initial inserts — that's fine; the important assertion is
    // post-GC referenced files survive.

    // Run GC dropping the pre-compaction snapshots (retain_snapshots = 1).
    let gc_result = GarbageCollector::new(storage.as_ref(), "gc_agg", 1)
        .run()
        .await
        .unwrap();

    // (a) Every orphan .agg that existed before GC is now gone.
    let all_agg_after = list_agg_files(storage.as_ref(), "gc_agg").await;
    for orphan_path in all_agg_before
        .iter()
        .filter(|p| !referenced_agg.contains(*p))
    {
        assert!(
            !storage.exists(orphan_path).await.unwrap(),
            "orphan .agg {} must be deleted by GC",
            orphan_path
        );
    }
    // Also verify via deleted set if there were orphans.
    if orphan_count > 0 {
        for deleted_path in &gc_result.deleted {
            // All deleted paths that are .agg files must NOT be in referenced set.
            if deleted_path.ends_with(".agg") {
                assert!(
                    !referenced_agg.contains(deleted_path),
                    "GC must not delete referenced .agg file {}",
                    deleted_path
                );
            }
        }
    }

    // (b) Every .agg referenced by the current manifest still exists.
    for agg_path in &referenced_agg {
        assert!(
            storage.exists(agg_path).await.unwrap(),
            "referenced .agg {} must survive GC",
            agg_path
        );
    }

    // (c) After GC the remaining .agg files are exactly the referenced ones.
    let remaining_agg_set: std::collections::HashSet<String> = all_agg_after.into_iter().collect();
    assert_eq!(
        remaining_agg_set, referenced_agg,
        "post-GC .agg files must be exactly the referenced ones"
    );

    let _ = orphan_count; // suppress unused if 0
}

// ── Part A: safety — referenced .agg is NEVER deleted ───────────────────────

/// The safety gate: GC must not delete any .agg file that is referenced by
/// the current (or any retained) manifest, even when the GC is configured to
/// retain only one snapshot.  This is the data-loss guard.
#[tokio::test]
async fn test_gc_never_deletes_referenced_agg() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    let mut writer = Writer::new(Arc::clone(&storage), "safe_agg", make_v_schema())
        .await
        .unwrap();
    // One insert + compact so the manifest has .agg files.
    writer
        .insert_batch(make_v_batch((1..=10).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    writer
        .insert_batch(make_v_batch((11..=20).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    Compactor::new(storage.as_ref(), "safe_agg")
        .compact()
        .await
        .unwrap();

    let manifest = read_latest_manifest_gc(storage.as_ref(), "safe_agg").await;
    let referenced_agg: Vec<String> = manifest
        .row_groups
        .iter()
        .filter_map(|e| e.agg.as_ref())
        .map(|a| format!("safe_agg/{}", a))
        .collect();
    assert!(
        !referenced_agg.is_empty(),
        "test setup: manifest must have .agg files"
    );

    // Run GC with retain_snapshots=1 (drops all but the current).
    let gc_result = GarbageCollector::new(storage.as_ref(), "safe_agg", 1)
        .run()
        .await
        .unwrap();

    // Safety assertion: every .agg in the current manifest must survive.
    for agg_path in &referenced_agg {
        assert!(
            storage.exists(agg_path).await.unwrap(),
            "SAFETY: referenced .agg {} must not be deleted by GC",
            agg_path
        );
        assert!(
            !gc_result.deleted.contains(agg_path),
            "SAFETY: GC deleted list must not contain referenced .agg {}",
            agg_path
        );
    }
}

/// Orphan .del files in _deletions/ are collected by GC when the snapshot that
/// referenced them has been dropped, and live .del files are never deleted.
#[tokio::test]
async fn test_gc_collects_orphan_del_files() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    // Manually inject an orphan .del file (simulating a post-delete snapshot
    // whose manifest was later dropped by retention).
    let orphan_del = "del_tbl/_deletions/rg_0000000000000001__v1.del";
    storage
        .write(
            "del_tbl/_manifests/placeholder",
            b"", // ensure dir exists for listing
        )
        .await
        .unwrap_or(());

    // Create a real table with one insert so GC has a valid snapshot.
    let mut writer = Writer::new(Arc::clone(&storage), "del_tbl", make_v_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_v_batch((1..=5).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Now inject the orphan .del under _deletions/ — it's not referenced by
    // the current manifest, so GC must collect it.
    storage
        .write(orphan_del, b"orphan-del-bytes")
        .await
        .unwrap();

    let gc_result = GarbageCollector::new(storage.as_ref(), "del_tbl", 1)
        .run()
        .await
        .unwrap();

    assert!(
        !storage.exists(orphan_del).await.unwrap(),
        "orphan .del {} must be deleted by GC",
        orphan_del
    );
    assert!(
        gc_result.deleted.contains(&orphan_del.to_string()),
        "GC deleted list must contain the orphan .del; got {:?}",
        gc_result.deleted
    );

    // No .del files remain (there were no real deletions in this table).
    let remaining_dels = list_del_files(storage.as_ref(), "del_tbl").await;
    assert!(
        remaining_dels.is_empty(),
        "no .del files should remain after GC removes orphan; got {:?}",
        remaining_dels
    );

    // Referenced files survive.
    let manifest = read_latest_manifest_gc(storage.as_ref(), "del_tbl").await;
    for entry in &manifest.row_groups {
        assert!(storage
            .exists(&format!("del_tbl/{}", entry.data))
            .await
            .unwrap());
        assert!(storage
            .exists(&format!("del_tbl/{}", entry.meta))
            .await
            .unwrap());
    }
}

/// Versioned/generated metadata artifacts should be collected when no retained
/// manifest references them. Legacy top-level `_indexes/<name>.*` files are
/// intentionally preserved because old manifests do not record them.
#[tokio::test]
async fn test_gc_collects_orphan_generated_metadata_artifacts() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());

    let mut writer = Writer::new(Arc::clone(&storage), "meta_gc", make_v_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_v_batch((1..=5).collect()))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let manifest = read_latest_manifest_gc(storage.as_ref(), "meta_gc").await;
    let live_checkpoint = manifest
        .checkpoint
        .as_ref()
        .expect("writer emits a snapshot checkpoint")
        .clone();
    let live_checkpoint_abs = format!("meta_gc/{live_checkpoint}");
    let live_archive_abs = format!(
        "meta_gc/{}",
        SnapshotCheckpoint::archive_filename(manifest.sequence)
    );
    storage
        .write(&live_archive_abs, b"live archive")
        .await
        .unwrap();

    let live_rowindex_files: Vec<String> = manifest
        .rowindex_generation
        .as_ref()
        .into_iter()
        .flat_map(|gen| gen.base.iter().chain(gen.deltas.iter()))
        .map(|p| format!("meta_gc/{p}"))
        .collect();

    let orphans = [
        "meta_gc/_checkpoints/000000000.json",
        "meta_gc/_checkpoints/000000000.rkyv",
        "meta_gc/_rowindex/orphan.idx",
        "meta_gc/_indexes/id_idx/base__v000000000.json",
        "meta_gc/_indexes/id_idx/base__v000000000.idx",
        "meta_gc/_indexes/id_idx/base__v000000000.model",
        "meta_gc/_indexes/id_idx/delta__v000000000.json",
    ];
    for path in orphans {
        storage.write(path, b"orphan").await.unwrap();
    }

    let legacy_index_files = [
        "meta_gc/_indexes/id_idx.json",
        "meta_gc/_indexes/id_idx.idx",
        "meta_gc/_indexes/id_idx.model",
    ];
    for path in legacy_index_files {
        storage.write(path, b"legacy").await.unwrap();
    }

    let gc_result = GarbageCollector::new(storage.as_ref(), "meta_gc", 1)
        .run()
        .await
        .unwrap();

    for path in orphans {
        assert!(
            !storage.exists(path).await.unwrap(),
            "orphan generated artifact {path} must be deleted"
        );
        assert!(
            gc_result.deleted.contains(&path.to_string()),
            "deleted list must contain orphan {path}; got {:?}",
            gc_result.deleted
        );
    }

    assert!(storage.exists(&live_checkpoint_abs).await.unwrap());
    assert!(storage.exists(&live_archive_abs).await.unwrap());
    for path in live_rowindex_files {
        assert!(
            storage.exists(&path).await.unwrap(),
            "live row-index artifact {path} must survive"
        );
    }
    for path in legacy_index_files {
        assert!(
            storage.exists(path).await.unwrap(),
            "legacy top-level index file {path} must not be swept"
        );
    }
}

// ── Part B: dv_density / should_recompute unit tests ────────────────────────

fn entry_with_deleted(deleted_count: u64) -> RowGroupEntry {
    RowGroupEntry {
        data: "rg_test.parquet".into(),
        meta: "rg_test.meta".into(),
        deleted_count,
        ..Default::default()
    }
}

#[test]
fn test_dv_density_basic() {
    let entry = entry_with_deleted(5);
    let density = dv_density(&entry, 20);
    assert!(
        (density - 0.25).abs() < 1e-12,
        "5/20 = 0.25, got {}",
        density
    );
}

#[test]
fn test_dv_density_zero_deleted() {
    let entry = entry_with_deleted(0);
    assert_eq!(dv_density(&entry, 100), 0.0);
}

#[test]
fn test_dv_density_all_deleted() {
    let entry = entry_with_deleted(10);
    assert!(
        (dv_density(&entry, 10) - 1.0).abs() < 1e-12,
        "all rows deleted → density 1.0"
    );
}

#[test]
fn test_dv_density_zero_total_rows_guard() {
    // When total_rows is 0 the formula uses max(1) to avoid division by zero.
    let entry = entry_with_deleted(0);
    assert_eq!(dv_density(&entry, 0), 0.0, "0/max(0,1)=0/1=0.0");
}

#[test]
fn test_should_recompute_below_threshold() {
    // 9 of 100 deleted → 0.09 < 0.10 → should_recompute = false
    let entry = entry_with_deleted(9);
    assert!(!should_recompute(&entry, 100));
}

#[test]
fn test_should_recompute_at_threshold() {
    // Exactly at RECOMPUTE_DENSITY should return true (>=).
    // 10 of 100 deleted → 0.10 >= 0.10 → true.
    let entry = entry_with_deleted(10);
    assert!(should_recompute(&entry, 100));
}

#[test]
fn test_should_recompute_above_threshold() {
    // 50 of 100 deleted → 0.50 ≥ 0.10 → true
    let entry = entry_with_deleted(50);
    assert!(should_recompute(&entry, 100));
}

#[test]
fn test_recompute_density_constant_value() {
    // Regression guard: the constant must be 0.10 (measured crossover ≈ 0.103,
    // bench 2026-06-23, 1 M-row Int64 fragment, MemoryStorage).
    assert!(
        (RECOMPUTE_DENSITY - 0.10).abs() < f64::EPSILON,
        "RECOMPUTE_DENSITY must be 0.10, got {}",
        RECOMPUTE_DENSITY
    );
}
