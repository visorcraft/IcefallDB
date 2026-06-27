use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::metadata::{Column, Manifest, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::{Writer, WriterOptions};
use icefalldb_core::{ActionKind, DiagnosisKind, Doctor, IcefallDBError};
use std::sync::Arc;
use std::time::Duration;

fn make_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

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

async fn setup_committed_table() -> (Arc<dyn Storage>, String) {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "products";
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), table, schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    (storage, table.to_string())
}

async fn read_latest_sequence(storage: &dyn Storage, table: &str) -> u64 {
    let pointer_data = storage
        .read(&format!("{}/_manifest.json", table))
        .await
        .unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&pointer_data).unwrap();
    pointer["latest"].as_u64().unwrap()
}

async fn read_latest_manifest(storage: &dyn Storage, table: &str) -> Manifest {
    let seq = read_latest_sequence(storage, table).await;
    let data = storage
        .read(&format!("{}/{}", table, Manifest::filename(seq)))
        .await
        .unwrap();
    serde_json::from_slice(&data).unwrap()
}

#[tokio::test]
async fn test_doctor_no_op_on_healthy_table() {
    let (storage, table) = setup_committed_table().await;

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert_eq!(result.table, table);
    assert!(!result.repaired);
    assert!(result.actions.is_empty());
}

#[tokio::test]
async fn test_doctor_repairs_missing_pointer() {
    let (storage, table) = setup_committed_table().await;
    let original_seq = read_latest_sequence(storage.as_ref(), &table).await;

    storage
        .delete(&format!("{}/_manifest.json", table))
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    let pointer_action = result
        .actions
        .iter()
        .find(|a| a.kind == ActionKind::PointerUpdated)
        .expect("expected PointerUpdated action");
    assert_eq!(pointer_action.path, "_manifest.json");

    let restored_seq = read_latest_sequence(storage.as_ref(), &table).await;
    assert_eq!(restored_seq, original_seq);
}

#[tokio::test]
async fn test_doctor_rolls_back_stale_intent() {
    let (storage, table) = setup_committed_table().await;

    // Create a stale intent referencing a staged .part file and a final file.
    let intent = serde_json::json!({
        "txn_id": "txn_stale",
        "files": [
            "_staging/incoming/rg_stale.parquet.part",
            "rg_stale.parquet"
        ],
    });
    storage
        .write(
            &format!("{}/_staging/intents/txn_stale.json", table),
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();
    storage
        .write(&format!("{}/rg_stale.parquet", table), b"stale-data")
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/_staging/incoming/rg_stale.parquet.part", table),
            b"stale-part",
        )
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    assert!(result
        .actions
        .iter()
        .any(|a| a.kind == ActionKind::RolledBack));

    assert!(!storage
        .exists(&format!("{}/_staging/intents/txn_stale.json", table))
        .await
        .unwrap());
    assert!(!storage
        .exists(&format!("{}/rg_stale.parquet", table))
        .await
        .unwrap());
    assert!(!storage
        .exists(&format!(
            "{}/_staging/incoming/rg_stale.parquet.part",
            table
        ))
        .await
        .unwrap());
}

#[tokio::test]
async fn test_doctor_removes_unreferenced_row_group() {
    let (storage, table) = setup_committed_table().await;

    storage
        .write(&format!("{}/rg_orphan.parquet", table), b"orphan-data")
        .await
        .unwrap();
    storage
        .write(&format!("{}/rg_orphan.meta", table), b"orphan-meta")
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    assert!(result
        .actions
        .iter()
        .any(|a| a.kind == ActionKind::OrphanRemoved && a.path == "rg_orphan.parquet"));
    assert!(result
        .actions
        .iter()
        .any(|a| a.kind == ActionKind::OrphanRemoved && a.path == "rg_orphan.meta"));

    assert!(!storage
        .exists(&format!("{}/rg_orphan.parquet", table))
        .await
        .unwrap());
    assert!(!storage
        .exists(&format!("{}/rg_orphan.meta", table))
        .await
        .unwrap());
}

#[tokio::test]
async fn test_doctor_removes_orphan_manifest_snapshot() {
    let (storage, table) = setup_committed_table().await;
    let latest = read_latest_sequence(storage.as_ref(), &table).await;

    // Add a newer corrupt manifest snapshot.
    let corrupt_seq = latest + 1;
    storage
        .write(
            &format!("{}/{}", table, Manifest::filename(corrupt_seq)),
            b"not valid json",
        )
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    assert!(!storage
        .exists(&format!("{}/{}", table, Manifest::filename(corrupt_seq)))
        .await
        .unwrap());
}

#[tokio::test]
async fn test_doctor_lock_timeout_when_writer_active() {
    let (storage, table) = setup_committed_table().await;

    let lock_path = format!("{}/_write.lock", table);
    let _guard = storage
        .lock_exclusive(&lock_path, Duration::from_secs(10))
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table).with_timeout(Duration::from_millis(50));
    let result = doctor.repair().await;

    assert!(
        matches!(result, Err(IcefallDBError::LockTimeout(_))),
        "expected LockTimeout, got {:?}",
        result
    );
}

#[tokio::test]
async fn test_doctor_does_not_delete_referenced_files() {
    let (storage, table) = setup_committed_table().await;

    let _manifest = {
        let seq = read_latest_sequence(storage.as_ref(), &table).await;
        let data = storage
            .read(&format!("{}/{}", table, Manifest::filename(seq)))
            .await
            .unwrap();
        let mut manifest: Manifest = serde_json::from_slice(&data).unwrap();
        // Add a fake unreferenced entry that the doctor should remove.
        manifest
            .row_groups
            .push(icefalldb_core::metadata::RowGroupEntry {
                data: "rg_real.parquet".into(),
                meta: "rg_real.meta".into(),
                ..Default::default()
            });
        // Recompute checksum so the manifest remains valid.
        manifest.checksum = manifest.compute_checksum().unwrap();
        storage
            .write(
                &format!("{}/{}", table, Manifest::filename(manifest.sequence)),
                serde_json::to_vec_pretty(&manifest).unwrap().as_slice(),
            )
            .await
            .unwrap();
        manifest
    };

    // Create the referenced files and an unrelated orphan.
    storage
        .write(&format!("{}/rg_real.parquet", table), b"real-data")
        .await
        .unwrap();
    storage
        .write(&format!("{}/rg_real.meta", table), b"real-meta")
        .await
        .unwrap();
    storage
        .write(&format!("{}/rg_orphan.parquet", table), b"orphan-data")
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    assert!(storage
        .exists(&format!("{}/rg_real.parquet", table))
        .await
        .unwrap());
    assert!(storage
        .exists(&format!("{}/rg_real.meta", table))
        .await
        .unwrap());
    assert!(!storage
        .exists(&format!("{}/rg_orphan.parquet", table))
        .await
        .unwrap());

    // The referenced fake entry should remain; the orphan should be reported.
    assert!(result
        .actions
        .iter()
        .any(|a| a.kind == ActionKind::OrphanRemoved && a.path == "rg_orphan.parquet"));
    assert!(!result
        .actions
        .iter()
        .any(|a| a.kind == ActionKind::OrphanRemoved
            && (a.path == "rg_real.parquet" || a.path == "rg_real.meta")));
}

#[tokio::test]
async fn test_doctor_diagnostic_only_no_changes() {
    let (storage, table) = setup_committed_table().await;

    // Create corrupting state that would normally be repaired.
    storage
        .write(&format!("{}/rg_orphan.parquet", table), b"orphan-data")
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/_staging/incoming/rg_orphan.parquet.part", table),
            b"orphan-part",
        )
        .await
        .unwrap();
    let intent = serde_json::json!({
        "txn_id": "txn_stale",
        "files": ["rg_stale.parquet"],
    });
    storage
        .write(
            &format!("{}/_staging/intents/txn_stale.json", table),
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();

    // Capture the set of files before diagnosis.
    let before_files = list_all_files(storage.as_ref(), &table).await;

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.diagnose().await.unwrap();

    assert!(!result.healthy);
    assert!(result
        .issues
        .iter()
        .any(|i| i.kind == DiagnosisKind::OrphanRowGroup));
    assert!(result
        .issues
        .iter()
        .any(|i| i.kind == DiagnosisKind::OrphanStagedPart));
    assert!(result
        .issues
        .iter()
        .any(|i| i.kind == DiagnosisKind::StaleIntent));

    // No files should have been created, deleted, or modified.
    let after_files = list_all_files(storage.as_ref(), &table).await;
    assert_eq!(before_files, after_files);
}

#[tokio::test]
async fn test_doctor_chooses_highest_valid_sequence() {
    let (storage, table) = setup_committed_table().await;

    // Commit a second sequence so both seq=1 and seq=2 are valid.
    let schema = make_schema();
    let mut writer = Writer::new(Arc::clone(&storage), &table, schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Point the manifest pointer back to seq=1.
    let pointer = serde_json::json!({ "latest": 1 });
    storage
        .write(
            &format!("{}/_manifest.json", table),
            serde_json::to_vec_pretty(&pointer).unwrap().as_slice(),
        )
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    assert!(result
        .actions
        .iter()
        .any(|a| a.kind == ActionKind::PointerUpdated));

    let restored_seq = read_latest_sequence(storage.as_ref(), &table).await;
    assert_eq!(restored_seq, 2);
}

#[tokio::test]
async fn test_doctor_does_not_delete_committed_files_after_crash() {
    let (storage, table) = setup_committed_table().await;

    // Read the committed manifest to discover the files that belong to the
    // current snapshot.
    let seq = read_latest_sequence(storage.as_ref(), &table).await;
    let manifest_data = storage
        .read(&format!("{}/{}", table, Manifest::filename(seq)))
        .await
        .unwrap();
    let manifest: Manifest = serde_json::from_slice(&manifest_data).unwrap();

    // Simulate a crash after the manifest pointer was updated but before the
    // intent file was deleted: leave an intent that lists the committed files.
    let committed_files: Vec<String> = manifest
        .row_groups
        .iter()
        .flat_map(|rg| [rg.data.clone(), rg.meta.clone()])
        .collect();
    let intent = serde_json::json!({
        "txn_id": "txn_crash",
        "files": committed_files,
    });
    storage
        .write(
            &format!("{}/_staging/intents/txn_crash.json", table),
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    // Committed files must survive the repair.
    for rg in &manifest.row_groups {
        assert!(
            storage
                .exists(&format!("{}/{}", table, rg.data))
                .await
                .unwrap(),
            "committed data file {} was deleted",
            rg.data
        );
        assert!(
            storage
                .exists(&format!("{}/{}", table, rg.meta))
                .await
                .unwrap(),
            "committed meta file {} was deleted",
            rg.meta
        );
    }

    // The stale intent itself must be removed.
    assert!(!storage
        .exists(&format!("{}/_staging/intents/txn_crash.json", table))
        .await
        .unwrap());

    // The rollback should report the committed files as skipped, not deleted.
    let skipped: Vec<_> = result
        .actions
        .iter()
        .filter(|a| {
            a.kind == ActionKind::Skipped && a.detail == "referenced by a retained valid snapshot"
        })
        .collect();
    assert!(
        !skipped.is_empty(),
        "expected Skipped actions for committed files"
    );
    for rg in &manifest.row_groups {
        assert!(
            skipped.iter().any(|a| a.path == rg.data),
            "missing Skipped action for {}",
            rg.data
        );
        assert!(
            skipped.iter().any(|a| a.path == rg.meta),
            "missing Skipped action for {}",
            rg.meta
        );
    }
}

#[tokio::test]
async fn test_doctor_diagnose_empty_table() {
    let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
    let table = "empty_table";

    let doctor = Doctor::new(storage.as_ref(), table);
    let result = doctor.diagnose().await.unwrap();

    assert_eq!(result.table, table);
    assert!(result.healthy);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].kind, DiagnosisKind::Info);
    assert_eq!(result.issues[0].path, "_manifest.json");
    assert_eq!(result.issues[0].detail, "empty table");
}

#[tokio::test]
async fn test_doctor_removes_manifest_tmp_files() {
    let (storage, table) = setup_committed_table().await;

    storage
        .write(
            &format!("{}/_manifests/999.json.tmp", table),
            b"incomplete-write",
        )
        .await
        .unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired);
    assert!(result
        .actions
        .iter()
        .any(|a| { a.kind == ActionKind::OrphanRemoved && a.path == "_manifests/999.json.tmp" }));
    assert!(!storage
        .exists(&format!("{}/_manifests/999.json.tmp", table))
        .await
        .unwrap());
}

async fn list_all_files(storage: &dyn Storage, table: &str) -> Vec<String> {
    let mut files = Vec::new();
    let mut stack = vec![table.to_string()];

    while let Some(prefix) = stack.pop() {
        let entries = match storage.list(&prefix).await {
            Ok(e) => e,
            Err(icefalldb_core::IcefallDBError::NotFound(_)) => continue,
            Err(_) => break,
        };
        for entry in entries {
            if storage.list(&entry).await.is_ok() {
                stack.push(entry);
            } else {
                files.push(entry);
            }
        }
    }

    files.sort();
    files
}

#[tokio::test]
async fn test_doctor_preserves_files_referenced_by_older_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let table = "products";

    // Write an initial schema and row group under schema_id 1.
    let schema_v1 = Schema {
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
    let mut writer = Writer::new(Arc::new(storage.clone()), table, schema_v1.clone())
        .await
        .unwrap();
    let batch = {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )
        .unwrap()
    };
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    // Advance to schema_id 2 and commit a second row group.
    let schema_v2 = Schema {
        schema_id: 2,
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
    storage
        .write(
            &format!("{}/_schemas/000002.json", table),
            &serde_json::to_vec_pretty(&schema_v2).unwrap(),
        )
        .await
        .unwrap();
    storage
        .write(
            &format!("{}/_schema.json", table),
            &serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
        )
        .await
        .unwrap();
    let mut writer = Writer::new_with_options(
        Arc::new(storage.clone()),
        table,
        schema_v2,
        WriterOptions::default(),
    )
    .await
    .unwrap();
    let batch = {
        let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from(vec![4, 5, 6]))],
        )
        .unwrap()
    };
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    // The manifest now points to sequence 2 (schema v2). Find the files from
    // sequence 1, which are only referenced by the older retained snapshot.
    let manifest_v1: Manifest = serde_json::from_slice(
        &storage
            .read(&format!("{}/{}", table, Manifest::filename(1)))
            .await
            .unwrap(),
    )
    .unwrap();
    let older_files: Vec<String> = manifest_v1
        .row_groups
        .iter()
        .flat_map(|rg| [rg.data.clone(), rg.meta.clone()])
        .collect();

    // Create a stale intent that lists those older-only files.
    let intent = serde_json::json!({
        "txn_id": "txn_old",
        "files": older_files,
    });
    storage
        .write(
            &format!("{}/_staging/intents/txn_old.json", table),
            serde_json::to_vec_pretty(&intent).unwrap().as_slice(),
        )
        .await
        .unwrap();

    let doctor = Doctor::new(&storage, table);
    let result = doctor.repair().await.unwrap();

    // Older files must survive because they are referenced by a retained valid snapshot.
    for rg in &manifest_v1.row_groups {
        assert!(
            storage
                .exists(&format!("{}/{}", table, rg.data))
                .await
                .unwrap(),
            "older data file {} was deleted",
            rg.data
        );
        assert!(
            storage
                .exists(&format!("{}/{}", table, rg.meta))
                .await
                .unwrap(),
            "older meta file {} was deleted",
            rg.meta
        );
    }

    assert!(!storage
        .exists(&format!("{}/_staging/intents/txn_old.json", table))
        .await
        .unwrap());

    let skipped: Vec<_> = result
        .actions
        .iter()
        .filter(|a| {
            a.kind == ActionKind::Skipped && a.detail == "referenced by a retained valid snapshot"
        })
        .collect();
    assert!(
        skipped
            .iter()
            .any(|a| a.path == manifest_v1.row_groups[0].data),
        "expected Skipped action for older data file"
    );
}

#[tokio::test]
async fn test_doctor_repair_regenerates_missing_row_group_meta() {
    let (storage, table) = setup_committed_table().await;
    let manifest = read_latest_manifest(storage.as_ref(), &table).await;
    let meta_path = format!("{}/{}", table, manifest.row_groups[0].meta);

    storage.delete(&meta_path).await.unwrap();

    // `check` should report the missing sidecar before repair.
    let check_before = icefalldb_core::Checker::new(storage.as_ref(), &table)
        .check()
        .await
        .unwrap();
    assert!(!check_before.passed);
    assert!(check_before
        .issues
        .iter()
        .any(|i| i.code == "MISSING_ROW_GROUP_META"));

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(result.repaired, "expected a repair action");
    assert!(result.healthy, "repair should leave the table healthy");
    assert!(
        result
            .actions
            .iter()
            .any(|a| a.kind == ActionKind::Regenerated && a.path == manifest.row_groups[0].meta),
        "expected Regenerated action for missing meta"
    );
    assert!(
        storage.exists(&meta_path).await.unwrap(),
        "regenerated meta file should exist"
    );

    let check_after = icefalldb_core::Checker::new(storage.as_ref(), &table)
        .check()
        .await
        .unwrap();
    assert!(check_after.passed, "issues: {:?}", check_after.issues);
}

#[tokio::test]
async fn test_doctor_diagnose_reports_missing_row_group_meta() {
    let (storage, table) = setup_committed_table().await;
    let manifest = read_latest_manifest(storage.as_ref(), &table).await;
    let meta_path = format!("{}/{}", table, manifest.row_groups[0].meta);

    storage.delete(&meta_path).await.unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.diagnose().await.unwrap();

    assert!(!result.healthy);
    assert!(result
        .issues
        .iter()
        .any(|i| i.kind == DiagnosisKind::MissingRowGroupMeta));
}

#[tokio::test]
async fn test_doctor_repair_missing_schema_is_unrepairable() {
    let (storage, table) = setup_committed_table().await;
    let manifest = read_latest_manifest(storage.as_ref(), &table).await;
    let schema_path = format!("{}/{}", table, Schema::filename(manifest.schema_id));
    let meta_path = format!("{}/{}", table, manifest.row_groups[0].meta);

    // Delete the schema file (but leave the schema pointer so repair proceeds
    // past the existence check). `load_schema` will now return None.
    storage.delete(&schema_path).await.unwrap();
    // Also delete the meta sidecar so there is something to repair.
    storage.delete(&meta_path).await.unwrap();

    let doctor = Doctor::new(storage.as_ref(), &table);
    let result = doctor.repair().await.unwrap();

    assert!(
        !result.healthy,
        "repair should be non-healthy when the schema is missing"
    );
    assert!(
        result
            .actions
            .iter()
            .any(|a| a.kind == ActionKind::Unrepairable && a.path == manifest.row_groups[0].meta),
        "expected Unrepairable action for missing meta when schema is missing: {:?}",
        result.actions
    );
}
