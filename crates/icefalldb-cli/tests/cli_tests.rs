use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::ipc::writer::FileWriter;
use futures::stream::StreamExt;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::writer::Writer;
use icefalldb_core::{DatabaseCatalog, Reader};
use parquet::arrow::arrow_writer::ArrowWriter;
use std::process::Command;
use std::sync::Arc;

fn icefalldb_bin() -> std::path::PathBuf {
    std::env::var_os("CARGO_BIN_EXE_icefalldb")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let mut path = std::env::current_exe().unwrap();
            path.pop(); // deps
            path.pop(); // debug or release
            path.push("icefalldb");
            path
        })
}

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

/// Schema matching the default used by `icefalldb create` without `--schema`.
fn make_cli_default_schema() -> Schema {
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
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 134_217_728,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

async fn setup_table(root: &std::path::Path) -> std::path::PathBuf {
    let table_path = root.join("products");
    let storage = LocalStorage::new(root).unwrap();
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
    table_path
}

#[tokio::test]
async fn test_cli_check_passes_on_valid_table() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    let output = Command::new(icefalldb_bin())
        .arg("check")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("check passed"));
}

#[tokio::test]
async fn test_cli_check_fails_on_corrupt_table() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Corrupt the row group meta checksum so the checker reports an error.
    let table_path = tmp.path().join("products");
    let mut meta_path = None;
    for entry in std::fs::read_dir(&table_path).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("rg_") && name.ends_with(".meta") {
            meta_path = Some(entry.path());
            break;
        }
    }
    let meta_path = meta_path.expect("row group meta file should exist");
    let mut meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();
    meta.as_object_mut()
        .unwrap()
        .insert("meta_checksum".into(), serde_json::json!("deadbeef"));
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("check")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(1));
}

#[tokio::test]
async fn test_cli_compact_reduces_row_groups() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Commit a second small row group so compaction has something to rewrite.
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("compact")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("compacted"));
    assert!(stdout.contains("2 input row groups -> 1 output row groups"));
    assert!(stdout.contains("6 rows -> 6 rows"));

    let pointer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(table_path(tmp.path()).join("_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(pointer["latest"].as_u64().unwrap(), 3);
}

#[tokio::test]
async fn test_cli_optimize_reduces_files() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Commit a second small row group so optimize has something to rewrite.
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("optimize")
        .arg(tmp.path())
        .arg("products")
        .arg("--retain-snapshots")
        .arg("1")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("optimized"));
    assert!(stdout.contains("2 input row groups -> 1 output row groups"));
    assert!(stdout.contains("6 rows -> 6 rows"));
    assert!(stdout.contains("deleted"));
    assert!(stdout.contains("retained snapshots"));

    let parquet_count = std::fs::read_dir(table_path(tmp.path()))
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .map(|ext| ext == "parquet")
                .unwrap_or(false)
        })
        .count();
    assert_eq!(parquet_count, 1, "expected one parquet file after optimize");
}

#[tokio::test]
async fn test_cli_doctor_no_op_on_healthy_table() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("table is healthy"));
}

#[tokio::test]
async fn test_cli_doctor_repair_flag_repairs_missing_pointer() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Delete the pointer to force a repair.
    std::fs::remove_file(table_path(tmp.path()).join("_manifest.json")).unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg("--repair")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PointerUpdated"));
    assert!(stdout.contains("repaired"));

    // Pointer should be restored.
    let pointer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(table_path(tmp.path()).join("_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(pointer["latest"].as_u64().unwrap(), 1);
}

#[tokio::test]
async fn test_cli_doctor_repair_regenerates_missing_meta() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Locate and delete the row group metadata sidecar.
    let mut meta_path = None;
    for entry in std::fs::read_dir(table_path(tmp.path())).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with("rg_") && name.ends_with(".meta") {
            meta_path = Some(entry.path());
            break;
        }
    }
    let meta_path = meta_path.expect("row group meta file should exist");
    std::fs::remove_file(&meta_path).unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg("--repair")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Regenerated"), "stdout: {stdout}");
    assert!(stdout.contains("repaired"), "stdout: {stdout}");

    // The sidecar should be back and `check` should pass.
    assert!(meta_path.exists());
    let check = Command::new(icefalldb_bin())
        .arg("check")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(String::from_utf8_lossy(&check.stdout).contains("check passed"));
}

#[tokio::test]
async fn test_cli_doctor_diagnostic_only_no_changes() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Add an orphan row group file.
    std::fs::write(
        table_path(tmp.path()).join("rg_orphan.parquet"),
        b"orphan-data",
    )
    .unwrap();

    // Capture directory state before running doctor without --repair.
    let before = list_files(&table_path(tmp.path()));

    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("OrphanRowGroup"));
    assert!(stdout.contains("issues found"));

    let after = list_files(&table_path(tmp.path()));
    assert_eq!(before, after);
}

#[tokio::test]
async fn test_cli_doctor_chooses_highest_valid_sequence() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Commit a second sequence.
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Point back to sequence 1.
    let pointer = serde_json::json!({ "latest": 1 });
    std::fs::write(
        table_path(tmp.path()).join("_manifest.json"),
        serde_json::to_vec_pretty(&pointer).unwrap(),
    )
    .unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg("--repair")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let pointer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(table_path(tmp.path()).join("_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(pointer["latest"].as_u64().unwrap(), 2);
}

#[tokio::test]
async fn test_cli_doctor_error_returns_exit_code_one() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Remove all manifest snapshots so doctor has no valid snapshots.
    let manifests_dir = table_path(tmp.path()).join("_manifests");
    for entry in std::fs::read_dir(&manifests_dir).unwrap() {
        let entry = entry.unwrap();
        std::fs::remove_file(entry.path()).unwrap();
    }

    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg("--repair")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(1));
}

#[test]
fn test_cli_parse_error_returns_exit_code_two() {
    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg("--unknown-flag")
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn test_cli_help_returns_exit_code_zero() {
    let output = Command::new(icefalldb_bin())
        .arg("--help")
        .output()
        .unwrap();

    assert!(output.status.success());
    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"));
}

#[tokio::test]
async fn test_cli_iceberg_export_produces_metadata() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    let output = tempfile::tempdir().unwrap();
    let output = output.path();

    let output = Command::new(icefalldb_bin())
        .arg("iceberg-export")
        .arg(tmp.path())
        .arg("products")
        .arg(output)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata_path_str = String::from_utf8_lossy(&output.stdout)
        .trim_end_matches('\n')
        .to_string();
    let metadata_path = std::path::Path::new(&metadata_path_str);
    assert!(metadata_path.exists());
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(metadata_path).unwrap()).unwrap();
    assert_eq!(metadata["format-version"], 2);
    assert!(metadata["current-snapshot-id"].as_i64().unwrap() > 0);
}

#[tokio::test]
async fn test_cli_gc_removes_old_row_groups() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Commit a second small row group so compaction rewrites older data.
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = make_schema();
    let mut writer = Writer::new(Arc::new(storage), "products", schema)
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let compact_output = Command::new(icefalldb_bin())
        .arg("compact")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(
        compact_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&compact_output.stderr)
    );

    let gc_output = Command::new(icefalldb_bin())
        .arg("gc")
        .arg(tmp.path())
        .arg("products")
        .arg("--retain-snapshots")
        .arg("1")
        .output()
        .unwrap();

    assert!(
        gc_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&gc_output.stderr)
    );
    let stdout = String::from_utf8_lossy(&gc_output.stdout);
    assert!(stdout.contains("gc completed: deleted"));
    assert!(stdout.contains("retained snapshots [3]"));

    let pointer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(table_path(tmp.path()).join("_manifest.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(pointer["latest"].as_u64().unwrap(), 3);

    let manifests_dir = table_path(tmp.path()).join("_manifests");
    let manifest_count = std::fs::read_dir(&manifests_dir)
        .unwrap()
        .filter(|e| {
            let path = e.as_ref().unwrap().path();
            path.extension().map(|e| e == "json").unwrap_or(false)
        })
        .count();
    assert_eq!(manifest_count, 1);
}

fn table_path(root: &std::path::Path) -> std::path::PathBuf {
    root.join("products")
}

fn list_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    collect_files(dir, &mut files);
    files.sort();
    files
}

fn collect_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out);
        } else {
            out.push(path);
        }
    }
}

fn write_arrow_file(path: &std::path::Path, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
}

fn write_parquet_file(path: &std::path::Path, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

async fn read_all_ids(storage: &LocalStorage, table: &str) -> Vec<i64> {
    let reader = Reader::new(storage, table).await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut ids = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let col = batch.column_by_name("id").unwrap();
            let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..arr.len() {
                ids.push(arr.value(i));
            }
        }
    }
    ids
}

#[tokio::test]
async fn test_cli_create_table() {
    let tmp = tempfile::tempdir().unwrap();
    let table = tmp.path().join("products");

    let output = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("created table products"));

    assert!(table.join("_schema.json").exists());
    assert!(table.join("_schemas").join("000001.json").exists());
    assert!(table.join("_manifest.json").exists());
    // A newly created table has no committed manifests yet.
    assert!(!table.join("_manifests").join("000000001.json").exists());

    let schema_pointer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(table.join("_schema.json")).unwrap()).unwrap();
    assert_eq!(schema_pointer["latest"].as_u64().unwrap(), 1);

    let manifest_pointer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(table.join("_manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest_pointer["latest"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn test_cli_create_with_schema() {
    let tmp = tempfile::tempdir().unwrap();
    let table = tmp.path().join("products");

    let custom_schema = serde_json::json!({
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": false, "field_id": 1},
            {"name": "name", "type": "utf8", "nullable": true, "field_id": 2}
        ],
        "partition_by": null,
        "sort": null,
        "row_group_target_rows": 10,
        "row_group_target_bytes": 1024,
        "dropped_columns": []
    });
    let schema_path = tmp.path().join("schema.json");
    std::fs::write(
        &schema_path,
        serde_json::to_vec_pretty(&custom_schema).unwrap(),
    )
    .unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("create")
        .arg("--schema")
        .arg(&schema_path)
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("created table products"));

    let expected: Schema = serde_json::from_value(custom_schema).unwrap();
    let written: Schema =
        serde_json::from_slice(&std::fs::read(table.join("_schemas").join("000001.json")).unwrap())
            .unwrap();

    // The writer assigns stable field IDs and updates max_field_id on creation;
    // mirror that in the expected value before comparing.
    let mut expected = expected;
    expected.columns[0].field_id = 1;
    expected.columns[1].field_id = 2;
    expected.max_field_id = 2;
    assert_eq!(written, expected);
}

/// `icefalldb create` must register the table in `_catalog.json` so the daemon
/// can serve it via `list_tables` (which reads the catalog's `tables` map).
#[tokio::test]
async fn test_cli_create_registers_in_catalog() {
    let tmp = tempfile::tempdir().unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The central catalog must exist and must contain the table.
    assert!(
        tmp.path().join("_catalog.json").exists(),
        "_catalog.json was not created by `icefalldb create`"
    );
    let storage: Arc<dyn icefalldb_core::storage::Storage> =
        Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let catalog = DatabaseCatalog::new(storage);
    let data = catalog.load().await.unwrap();
    assert!(
        data.tables.contains_key("products"),
        "catalog tables map should contain 'products' after `icefalldb create`, got: {:?}",
        data.tables.keys().collect::<Vec<_>>()
    );
}

/// A second `icefalldb create` on the same table must still fail (table already
/// exists on disk), but the catalog entry written by the first run must survive.
#[tokio::test]
async fn test_cli_create_catalog_idempotent_on_existing_entry() {
    let tmp = tempfile::tempdir().unwrap();

    // First create — succeeds and registers.
    let first = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(first.status.success());

    // Verify that the catalog entry is present after the first successful create.
    let storage: Arc<dyn icefalldb_core::storage::Storage> =
        Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let catalog = DatabaseCatalog::new(storage);
    let data = catalog.load().await.unwrap();
    assert!(
        data.tables.contains_key("products"),
        "catalog entry missing after first create"
    );
}

#[tokio::test]
async fn test_cli_create_refuses_overwrite() {
    let tmp = tempfile::tempdir().unwrap();

    let first = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(
        first.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.stderr)
    );

    let second = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();

    assert!(!second.status.success());
    assert_eq!(second.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(stderr.contains("already exists"));
}

#[tokio::test]
async fn test_cli_create_race_is_safe() {
    let tmp = tempfile::tempdir().unwrap();
    let table = tmp.path().join("products");

    // Spawn two concurrent `icefalldb create` processes against the same table.
    let mut child1 = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .spawn()
        .unwrap();
    let mut child2 = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .spawn()
        .unwrap();

    let status1 = child1.wait().unwrap();
    let status2 = child2.wait().unwrap();

    let successes = [status1.success(), status2.success()];
    assert_eq!(
        successes.iter().filter(|&&s| s).count(),
        1,
        "exactly one concurrent create should succeed: {:?} {:?}",
        status1,
        status2
    );

    // The manifest pointer must exist and must not regress below the empty-table
    // sentinel.
    assert!(table.join("_manifest.json").exists());
    let manifest_pointer: serde_json::Value =
        serde_json::from_slice(&std::fs::read(table.join("_manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest_pointer["latest"].as_u64().unwrap(), 0);

    // Verify the table is writable and readable, proving no data was lost by a
    // potential pointer regression or partial initialization.
    let input = tmp.path().join("data.arrow");
    write_arrow_file(&input, &make_batch(vec![7, 8, 9]));
    let insert_output = Command::new(icefalldb_bin())
        .arg("insert")
        .arg(tmp.path())
        .arg("products")
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        insert_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&insert_output.stderr)
    );

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let ids = read_all_ids(&storage, "products").await;
    assert_eq!(ids, vec![7, 8, 9]);
}

#[tokio::test]
async fn test_cli_create_races_with_writer_new() {
    let tmp = tempfile::tempdir().unwrap();
    let table = tmp.path().join("products");
    let db_path = tmp.path().to_path_buf();

    // Start a `icefalldb create` process and a programmatic `Writer::new`
    // concurrently. They contend for the same writer lock, so table
    // initialization is serialized deterministically.
    let cli_future = tokio::task::spawn_blocking(move || {
        Command::new(icefalldb_bin())
            .arg("create")
            .arg(&db_path)
            .arg("products")
            .output()
    });

    let writer_tmp_path = tmp.path().to_path_buf();
    let writer_future = tokio::spawn(async move {
        let storage = LocalStorage::new(&writer_tmp_path).unwrap();
        // Use the same default schema as the CLI so the writer can open the
        // table regardless of which side wins the race.
        Writer::new(Arc::new(storage), "products", make_cli_default_schema()).await
    });

    let cli_result = cli_future.await.unwrap().unwrap();
    let writer_result = writer_future.await.unwrap();

    // The table must exist and be valid regardless of which call won.
    assert!(table.join("_schema.json").exists());
    assert!(table.join("_schemas").join("000001.json").exists());
    assert!(table.join("_manifest.json").exists());

    // The CLI either won the race (success) or lost it (table already exists).
    if !cli_result.status.success() {
        let stderr = String::from_utf8_lossy(&cli_result.stderr);
        assert!(stderr.contains("already exists"));
    }

    // The writer always succeeds: it creates the table if needed, otherwise it
    // opens the existing table created by the CLI.
    assert!(
        writer_result.is_ok(),
        "writer should open or create the table: {:?}",
        writer_result.as_ref().map(|_| ())
    );
}

#[tokio::test]
async fn test_partial_create_state_is_existing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let table = tmp.path().join("products");
    std::fs::create_dir_all(table.join("_schemas")).unwrap();

    // Simulate a partial create where `_schema.json` and the schema file exist
    // but `_manifest.json` does not. `_schema.json` is the authoritative table
    // marker, so this must be treated as an existing table.
    let schema = make_schema();
    std::fs::write(
        table.join("_schemas").join("000001.json"),
        serde_json::to_vec_pretty(&schema).unwrap(),
    )
    .unwrap();
    std::fs::write(
        table.join("_schema.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"latest": 1})).unwrap(),
    )
    .unwrap();

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let writer = Writer::new(Arc::new(storage), "products", make_schema()).await;
    assert!(
        writer.is_ok(),
        "writer should treat partial state as existing table: {:?}",
        writer.as_ref().map(|_| ())
    );

    let output = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("already exists"));
}

#[tokio::test]
async fn test_cli_insert_arrow_file() {
    let tmp = tempfile::tempdir().unwrap();

    let create_output = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(
        create_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&create_output.stderr)
    );

    let input = tmp.path().join("data.arrow");
    write_arrow_file(&input, &make_batch(vec![10, 20, 30]));

    let output = Command::new(icefalldb_bin())
        .arg("insert")
        .arg(tmp.path())
        .arg("products")
        .arg(&input)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("inserted 3 rows into products"));

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let ids = read_all_ids(&storage, "products").await;
    assert_eq!(ids, vec![10, 20, 30]);
}

#[tokio::test]
async fn test_cli_insert_parquet_file() {
    let tmp = tempfile::tempdir().unwrap();

    let create_output = Command::new(icefalldb_bin())
        .arg("create")
        .arg(tmp.path())
        .arg("products")
        .output()
        .unwrap();
    assert!(
        create_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&create_output.stderr)
    );

    let input = tmp.path().join("data.parquet");
    write_parquet_file(&input, &make_batch(vec![100, 200, 300]));

    let output = Command::new(icefalldb_bin())
        .arg("insert")
        .arg(tmp.path())
        .arg("products")
        .arg(&input)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("inserted 3 rows into products"));

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let ids = read_all_ids(&storage, "products").await;
    assert_eq!(ids, vec![100, 200, 300]);
}

fn duckdb_available() -> bool {
    Command::new("duckdb")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn make_products_batch(ids: Vec<i64>, values: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]);
    let id_array = Int64Array::from(ids);
    let value_array = Int64Array::from(values);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(id_array), Arc::new(value_array)],
    )
    .unwrap()
}

fn make_products_schema() -> Schema {
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
                name: "value".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
        ],
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

#[tokio::test]
async fn test_cli_create_view_writes_definition() {
    let tmp = tempfile::tempdir().unwrap();

    let query_file = tmp.path().join("top_products.sql");
    std::fs::write(
        &query_file,
        "SELECT id, value FROM products WHERE value > 10",
    )
    .unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("top_products")
        .arg(&query_file)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("created view top_products"));

    let view_file = tmp.path().join("views").join("top_products.sql");
    assert!(view_file.exists());
    assert_eq!(
        std::fs::read_to_string(&view_file).unwrap(),
        "SELECT id, value FROM products WHERE value > 10"
    );
}

#[tokio::test]
async fn test_cli_create_view_rejects_non_select() {
    let tmp = tempfile::tempdir().unwrap();

    let query_file = tmp.path().join("bad.sql");
    std::fs::write(&query_file, "DROP TABLE products").unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("bad")
        .arg(&query_file)
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("query must be a single SELECT statement"));
}

#[tokio::test]
async fn test_cli_refresh_view_materializes_query() {
    if !duckdb_available() {
        eprintln!("skipping refresh-view test: duckdb CLI not available");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();

    // Create source table with id and value columns.
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "products", make_products_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_products_batch(vec![1, 2, 3, 4], vec![5, 15, 25, 35]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let query_file = tmp.path().join("top_products.sql");
    std::fs::write(
        &query_file,
        "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
    )
    .unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("top_products")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(
        create_view.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&create_view.stderr)
    );

    let refresh = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("top_products")
        .output()
        .unwrap();
    assert!(
        refresh.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&refresh.stderr)
    );
    let stdout = String::from_utf8_lossy(&refresh.stdout);
    assert!(stdout.contains("refreshed view top_products"));

    // The derived table should exist and contain the expected rows.
    let view_table = tmp.path().join("views").join("top_products");
    assert!(view_table.join("_schema.json").exists());
    assert!(view_table.join("_manifest.json").exists());

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let reader = Reader::new(&storage, "views/top_products").await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut rows = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let id_col = batch.column_by_name("id").unwrap();
            let value_col = batch.column_by_name("value").unwrap();
            let ids = id_col.as_any().downcast_ref::<Int64Array>().unwrap();
            let values = value_col.as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..ids.len() {
                rows.push((ids.value(i), values.value(i)));
            }
        }
    }
    assert_eq!(rows, vec![(2, 15), (3, 25), (4, 35)]);
}

#[tokio::test]
async fn test_cli_refresh_view_is_idempotent() {
    if !duckdb_available() {
        eprintln!("skipping refresh-view idempotency test: duckdb CLI not available");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "products", make_products_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_products_batch(vec![1, 2, 3, 4], vec![5, 15, 25, 35]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let query_file = tmp.path().join("top_products.sql");
    std::fs::write(
        &query_file,
        "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
    )
    .unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("top_products")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(create_view.status.success());

    for _ in 0..2 {
        let refresh = Command::new(icefalldb_bin())
            .arg("refresh-view")
            .arg(tmp.path())
            .arg("top_products")
            .output()
            .unwrap();
        assert!(
            refresh.status.success(),
            "stderr: {}",
            String::from_utf8_lossy(&refresh.stderr)
        );
    }

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let reader = Reader::new(&storage, "views/top_products").await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut rows = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let id_col = batch.column_by_name("id").unwrap();
            let value_col = batch.column_by_name("value").unwrap();
            let ids = id_col.as_any().downcast_ref::<Int64Array>().unwrap();
            let values = value_col.as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..ids.len() {
                rows.push((ids.value(i), values.value(i)));
            }
        }
    }
    assert_eq!(rows, vec![(2, 15), (3, 25), (4, 35)]);

    // A second refresh must not accumulate rows: the latest manifest should
    // contain only the row groups written by the most recent refresh.
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            tmp.path()
                .join("views")
                .join("top_products")
                .join("_manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(manifest["latest"].as_u64().unwrap(), 2);
}

#[tokio::test]
async fn test_cli_refresh_view_concurrent_refreshes_are_safe() {
    if !duckdb_available() {
        eprintln!("skipping concurrent refresh-view test: duckdb CLI not available");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "products", make_products_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_products_batch(vec![1, 2, 3, 4], vec![5, 15, 25, 35]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let query_file = tmp.path().join("top_products.sql");
    std::fs::write(
        &query_file,
        "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
    )
    .unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("top_products")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(create_view.status.success());

    // Spawn two concurrent refreshes for the same view. The CLI holds the
    // view-table writer lock for the entire refresh, so they must serialize
    // rather than corrupt or duplicate the derived table.
    let mut child1 = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("top_products")
        .spawn()
        .unwrap();
    let mut child2 = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("top_products")
        .spawn()
        .unwrap();

    let status1 = child1.wait().unwrap();
    let status2 = child2.wait().unwrap();

    assert!(
        status1.success() || status2.success(),
        "at least one concurrent refresh should succeed: {:?} {:?}",
        status1,
        status2
    );

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let reader = Reader::new(&storage, "views/top_products").await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut rows = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let id_col = batch.column_by_name("id").unwrap();
            let value_col = batch.column_by_name("value").unwrap();
            let ids = id_col.as_any().downcast_ref::<Int64Array>().unwrap();
            let values = value_col.as_any().downcast_ref::<Int64Array>().unwrap();
            for i in 0..ids.len() {
                rows.push((ids.value(i), values.value(i)));
            }
        }
    }
    assert_eq!(rows, vec![(2, 15), (3, 25), (4, 35)]);

    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            tmp.path()
                .join("views")
                .join("top_products")
                .join("_manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    // Both refreshes should complete and advance the manifest exactly twice.
    assert_eq!(manifest["latest"].as_u64().unwrap(), 2);
}

#[tokio::test]
async fn test_cli_refresh_view_empty_source_table() {
    if !duckdb_available() {
        eprintln!("skipping refresh-view empty-source test: duckdb CLI not available");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let _writer = Writer::new(Arc::new(storage), "products", make_products_schema())
        .await
        .unwrap();

    let query_file = tmp.path().join("empty_view.sql");
    std::fs::write(
        &query_file,
        "SELECT id, value FROM products WHERE value > 10 ORDER BY id",
    )
    .unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("empty_view")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(create_view.status.success());

    let refresh = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("empty_view")
        .output()
        .unwrap();
    assert!(
        refresh.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&refresh.stderr)
    );

    let view_table = tmp.path().join("views").join("empty_view");
    assert!(view_table.join("_schema.json").exists());
    assert!(view_table.join("_manifest.json").exists());

    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(view_table.join("_manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["latest"].as_u64().unwrap(), 1);

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let reader = Reader::new(&storage, "views/empty_view").await.unwrap();
    let plan = reader.scan().await.unwrap();
    assert!(plan.row_groups.is_empty());
}

#[tokio::test]
async fn test_cli_refresh_view_timestamp_roundtrip() {
    if !duckdb_available() {
        eprintln!("skipping refresh-view timestamp test: duckdb CLI not available");
        return;
    }

    use arrow::array::TimestampMicrosecondArray;

    let tmp = tempfile::tempdir().unwrap();

    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 0,
            },
            Column {
                name: "ts".into(),
                r#type: "timestamp[us]".into(),
                nullable: false,
                field_id: 0,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
            false,
        ),
    ]));
    let ids = Int64Array::from(vec![1, 2]);
    let ts_values = TimestampMicrosecondArray::from(vec![1_000_000, 2_000_000]);
    let batch =
        RecordBatch::try_new(arrow_schema, vec![Arc::new(ids), Arc::new(ts_values)]).unwrap();

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "events", schema)
        .await
        .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let query_file = tmp.path().join("ts_view.sql");
    std::fs::write(&query_file, "SELECT id, ts FROM events ORDER BY id").unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("ts_view")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(create_view.status.success());

    let refresh = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("ts_view")
        .output()
        .unwrap();
    assert!(
        refresh.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&refresh.stderr)
    );

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let reader = Reader::new(&storage, "views/ts_view").await.unwrap();
    let plan = reader.scan().await.unwrap();
    let mut rows = Vec::new();
    for rg in &plan.row_groups {
        let mut stream = reader.read_row_group(rg).await.unwrap();
        while let Some(batch) = stream.next().await {
            let batch = batch.unwrap();
            let id_col = batch.column_by_name("id").unwrap();
            let ts_col = batch.column_by_name("ts").unwrap();
            let ids = id_col.as_any().downcast_ref::<Int64Array>().unwrap();
            let ts = ts_col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            for i in 0..ids.len() {
                rows.push((ids.value(i), ts.value(i)));
            }
        }
    }
    assert_eq!(rows, vec![(1, 1_000_000), (2, 2_000_000)]);
}

#[tokio::test]
async fn test_cli_create_view_accepts_semicolon_in_string_literal() {
    let tmp = tempfile::tempdir().unwrap();

    let query_file = tmp.path().join("semi.sql");
    std::fs::write(&query_file, "SELECT id, ';' AS semi FROM products").unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("semi_view")
        .arg(&query_file)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let view_file = tmp.path().join("views").join("semi_view.sql");
    assert_eq!(
        std::fs::read_to_string(&view_file).unwrap(),
        "SELECT id, ';' AS semi FROM products"
    );
}

#[tokio::test]
async fn test_cli_refresh_view_missing_definition_fails() {
    let tmp = tempfile::tempdir().unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("does_not_exist")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("does not exist"));
}

#[tokio::test]
async fn test_cli_refresh_view_missing_duckdb_cli_fails() {
    let tmp = tempfile::tempdir().unwrap();

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "products", make_products_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_products_batch(vec![1], vec![1]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let query_file = tmp.path().join("v.sql");
    std::fs::write(&query_file, "SELECT * FROM products").unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("v")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(create_view.status.success());

    let output = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("v")
        .env_remove("PATH")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("duckdb CLI not found"));
}

#[tokio::test]
async fn test_cli_refresh_view_source_schema_change_fails() {
    if !duckdb_available() {
        eprintln!("skipping refresh-view schema-change test: duckdb CLI not available");
        return;
    }

    let tmp = tempfile::tempdir().unwrap();

    // Start with a single-column source table and a matching view.
    let storage = LocalStorage::new(tmp.path()).unwrap();
    let schema = Schema {
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
    let mut writer = Writer::new(Arc::new(storage), "products", schema.clone())
        .await
        .unwrap();
    writer.insert_batch(make_batch(vec![1, 2])).await.unwrap();
    writer.commit().await.unwrap();

    let query_file = tmp.path().join("narrow.sql");
    std::fs::write(&query_file, "SELECT id FROM products").unwrap();

    let create_view = Command::new(icefalldb_bin())
        .arg("create-view")
        .arg(tmp.path())
        .arg("narrow")
        .arg(&query_file)
        .output()
        .unwrap();
    assert!(create_view.status.success());

    let refresh = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("narrow")
        .output()
        .unwrap();
    assert!(refresh.status.success());

    // Evolve the source table schema by adding a column and committing a new
    // row group that contains both columns. Then update the view query to
    // select the new column. The existing derived table has only `id`, so the
    // refresh must fail with a schema mismatch.
    let new_schema = Schema {
        schema_id: 2,
        columns: vec![
            Column {
                name: "id".into(),
                r#type: "int64".into(),
                nullable: false,
                field_id: 1,
            },
            Column {
                name: "value".into(),
                r#type: "int64".into(),
                nullable: true,
                field_id: 2,
            },
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 2,
        dropped_columns: vec![],
    };
    std::fs::write(
        tmp.path()
            .join("products")
            .join("_schemas")
            .join("000002.json"),
        serde_json::to_vec_pretty(&new_schema).unwrap(),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("products").join("_schema.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
    )
    .unwrap();

    // Write a new row group that actually contains both columns.
    let table_dir = tmp.path().join("products");
    let evolved_batch = make_products_batch(vec![1, 2], vec![10, 20]);
    let data_path = table_dir.join("rg_evolved.parquet");
    let meta_path = table_dir.join("rg_evolved.meta");
    write_parquet_file(&data_path, &evolved_batch);
    let parquet_bytes = std::fs::read(&data_path).unwrap();
    let checksum = icefalldb_core::metadata::checksum_bytes(&parquet_bytes);
    let meta = serde_json::json!({
        "row_group": "rg_evolved",
        "schema_id": 2,
        "rows": 2,
        "columns": {
            "id": {"min": 1, "max": 2, "nulls": 0},
            "value": {"min": 10, "max": 20, "nulls": 0},
        },
        "checksum": checksum,
        "meta_checksum": "",
    });
    let meta_checksum = icefalldb_core::metadata::checksum_json(&meta);
    let mut meta = meta;
    meta["meta_checksum"] = serde_json::Value::String(meta_checksum);
    std::fs::write(&meta_path, serde_json::to_vec_pretty(&meta).unwrap()).unwrap();

    let mut new_manifest = serde_json::json!({
        "format_version": 1,
        "sequence": 2,
        "schema_id": 2,
        "row_groups": [
            {"data": "rg_evolved.parquet", "meta": "rg_evolved.meta"}
        ],
        "checksum": "",
    });
    let checksum = icefalldb_core::metadata::checksum_json(&new_manifest);
    new_manifest["checksum"] = serde_json::Value::String(checksum);
    std::fs::write(
        tmp.path()
            .join("products")
            .join("_manifests")
            .join("000000002.json"),
        serde_json::to_vec_pretty(&new_manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(
        tmp.path().join("products").join("_manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"latest": 2})).unwrap(),
    )
    .unwrap();

    // Update the view definition to select the new column.
    std::fs::write(
        tmp.path().join("views").join("narrow.sql"),
        "SELECT id, value FROM products",
    )
    .unwrap();

    let refresh = Command::new(icefalldb_bin())
        .arg("refresh-view")
        .arg(tmp.path())
        .arg("narrow")
        .output()
        .unwrap();

    assert!(!refresh.status.success());
    let stderr = String::from_utf8_lossy(&refresh.stderr);
    assert!(stderr.contains("SchemaMismatch") || stderr.contains("schema"));
}

#[tokio::test]
async fn test_cli_tsv_import_creates_table() {
    let tmp = tempfile::tempdir().unwrap();
    let input = tmp.path().join("data.tsv");
    std::fs::write(&input, "id\tname\n1\talice\n2\tbob\n").unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("import")
        .arg(tmp.path())
        .arg("products")
        .arg(&input)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("imported 2 rows into products"));

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let ids = read_all_ids(&storage, "products").await;
    assert_eq!(ids, vec![1, 2]);
}

#[tokio::test]
async fn test_cli_tsv_export_round_trip() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    let exported = tmp.path().join("products.tsv");
    let output = Command::new(icefalldb_bin())
        .arg("export")
        .arg(tmp.path())
        .arg("products")
        .arg(&exported)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("exported 3 rows from products"));
    assert!(exported.exists());

    let import_output = Command::new(icefalldb_bin())
        .arg("import")
        .arg(tmp.path())
        .arg("products_copy")
        .arg(&exported)
        .output()
        .unwrap();
    assert!(
        import_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&import_output.stderr)
    );

    let storage = LocalStorage::new(tmp.path()).unwrap();
    let ids = read_all_ids(&storage, "products_copy").await;
    assert_eq!(ids, vec![1, 2, 3]);
}

#[tokio::test]
async fn test_cli_create_index_builds_btree() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    let schema = serde_json::json!({
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": false, "field_id": 1},
            {"name": "cat", "type": "utf8", "nullable": true, "field_id": 2}
        ],
        "partition_by": null,
        "sort": null,
        "row_group_target_rows": 10,
        "row_group_target_bytes": 1024 * 1024,
        "dropped_columns": []
    });
    let schema_path = db.join("schema.json");
    std::fs::write(&schema_path, serde_json::to_vec_pretty(&schema).unwrap()).unwrap();

    let create_output = Command::new(icefalldb_bin())
        .arg("create")
        .arg("--schema")
        .arg(&schema_path)
        .arg(db)
        .arg("items")
        .output()
        .unwrap();
    assert!(
        create_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&create_output.stderr)
    );

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]));
    let ids = Int64Array::from(vec![1, 2]);
    let cats = StringArray::from(vec!["a", "b"]);
    let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(ids), Arc::new(cats)]).unwrap();

    let parquet_path = db.join("items.parquet");
    write_parquet_file(&parquet_path, &batch);

    let insert_output = Command::new(icefalldb_bin())
        .arg("insert")
        .arg(db)
        .arg("items")
        .arg(&parquet_path)
        .output()
        .unwrap();
    assert!(
        insert_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&insert_output.stderr)
    );

    let index_output = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(db)
        .arg("items")
        .arg("cat")
        .output()
        .unwrap();
    assert!(
        index_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&index_output.stderr)
    );
    let stdout = String::from_utf8_lossy(&index_output.stdout);
    assert!(stdout.contains("created index items_cat_idx on items.cat"));

    assert!(db.join("items/_indexes/items_cat_idx.json").exists());
}

fn make_items_schema() -> Schema {
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
        row_group_target_rows: 10,
        row_group_target_bytes: 1024 * 1024,
        max_field_id: 0,
        dropped_columns: vec![],
    };
    schema.assign_field_ids(None);
    schema
}

fn make_items_batch(ids: Vec<i64>, cats: Vec<&str>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("cat", DataType::Utf8, true),
    ]);
    let id_array = Int64Array::from(ids);
    let cat_array = StringArray::from(cats);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(id_array), Arc::new(cat_array)],
    )
    .unwrap()
}

async fn setup_items_table_with_data(root: &std::path::Path) {
    let storage = LocalStorage::new(root).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "items", make_items_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_items_batch(vec![1, 2], vec!["a", "b"]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
}

#[tokio::test]
async fn test_cli_create_index_rejects_duplicate_name() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_items_table_with_data(db).await;

    let first = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(db)
        .arg("items")
        .arg("cat")
        .output()
        .unwrap();
    assert!(
        first.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&first.stderr)
    );

    let second = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(db)
        .arg("items")
        .arg("cat")
        .output()
        .unwrap();
    assert!(!second.status.success());
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(stderr.contains("already exists"), "stderr: {}", stderr);
}

#[tokio::test]
async fn test_cli_create_index_rejects_missing_column() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_items_table_with_data(db).await;

    let output = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(db)
        .arg("items")
        .arg("does_not_exist")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("column") || stderr.contains("not found"),
        "stderr: {}",
        stderr
    );
}

#[tokio::test]
async fn test_cli_create_index_rejects_empty_table() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    let schema = serde_json::json!({
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": false, "field_id": 1},
            {"name": "cat", "type": "utf8", "nullable": true, "field_id": 2}
        ],
        "partition_by": null,
        "sort": null,
        "row_group_target_rows": 10,
        "row_group_target_bytes": 1024 * 1024,
        "dropped_columns": []
    });
    let schema_path = db.join("schema.json");
    std::fs::write(&schema_path, serde_json::to_vec_pretty(&schema).unwrap()).unwrap();

    let create_output = Command::new(icefalldb_bin())
        .arg("create")
        .arg("--schema")
        .arg(&schema_path)
        .arg(db)
        .arg("items")
        .output()
        .unwrap();
    assert!(
        create_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&create_output.stderr)
    );

    let output = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(db)
        .arg("items")
        .arg("cat")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("created index items_cat_idx on items.cat"));

    // No manifest means no index file, but the catalog definition must exist.
    assert!(!db.join("items/_indexes/items_cat_idx.json").exists());
    let catalog: serde_json::Value =
        serde_json::from_slice(&std::fs::read(db.join("_catalog.json")).unwrap()).unwrap();
    assert!(catalog["indexes"].get("items_cat_idx").is_some());
}

#[tokio::test]
async fn test_cli_create_index_uses_custom_name() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_items_table_with_data(db).await;

    let output = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(db)
        .arg("items")
        .arg("cat")
        .arg("--name")
        .arg("my_idx")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("created index my_idx on items.cat"));

    assert!(db.join("items/_indexes/my_idx.json").exists());
}

// ── query command: mutation routing ──────────────────────────────────────────

/// Query all (id, value) pairs from the `products` table via `icefalldb query`
/// SELECT, which applies deletion vectors correctly (unlike the raw Reader).
/// Returns rows sorted by id.
fn select_products_id_value(table_path: &std::path::Path) -> Vec<(i64, i64)> {
    let output = Command::new(icefalldb_bin())
        .arg("query")
        .arg(table_path)
        .arg("SELECT id, value FROM products ORDER BY id")
        .arg("--format")
        .arg("csv")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "SELECT stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // CSV format: header line + data lines. Skip the header.
    stdout
        .lines()
        .skip(1)
        .filter(|l| !l.is_empty())
        .map(|line| {
            let mut parts = line.splitn(2, ',');
            let id: i64 = parts.next().unwrap().trim().parse().unwrap();
            let value: i64 = parts.next().unwrap().trim().parse().unwrap();
            (id, value)
        })
        .collect()
}

/// Set up a `products` table (id, value columns) with three rows.
async fn setup_products_with_values(root: &std::path::Path) {
    let storage = LocalStorage::new(root).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "products", make_products_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_products_batch(vec![1, 2, 3], vec![10, 20, 30]))
        .await
        .unwrap();
    writer.commit().await.unwrap();
}

#[tokio::test]
async fn test_cli_query_update_mutates_row_and_select_reflects_change() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_products_with_values(db).await;

    // The path argument is the table directory when it contains _manifest.json.
    let table_path = db.join("products");

    let update_output = Command::new(icefalldb_bin())
        .arg("query")
        .arg(&table_path)
        .arg("UPDATE products SET value = 99 WHERE id = 2")
        .output()
        .unwrap();

    assert!(
        update_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&update_output.stderr)
    );
    let stderr = String::from_utf8_lossy(&update_output.stderr);
    assert!(
        stderr.contains("row(s) updated"),
        "expected 'row(s) updated' in stderr, got: {stderr}"
    );

    // Verify the mutation is durable by querying back via the CLI SELECT path
    // (which applies deletion vectors and correctly reflects the post-UPDATE state).
    let rows = select_products_id_value(&table_path);
    assert_eq!(
        rows,
        vec![(1, 10), (2, 99), (3, 30)],
        "row with id=2 should have value=99 after UPDATE"
    );
}

#[tokio::test]
async fn test_cli_query_update_no_match_reports_zero_affected() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_products_with_values(db).await;

    let table_path = db.join("products");

    let update_output = Command::new(icefalldb_bin())
        .arg("query")
        .arg(&table_path)
        .arg("UPDATE products SET value = 0 WHERE id = 999")
        .output()
        .unwrap();

    assert!(
        update_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&update_output.stderr)
    );
    let stderr = String::from_utf8_lossy(&update_output.stderr);
    // 0 rows affected should still report the 'row(s) updated' line.
    assert!(
        stderr.contains("row(s) updated"),
        "expected 'row(s) updated' in stderr, got: {stderr}"
    );

    // Data must be unchanged.
    let rows = select_products_id_value(&table_path);
    assert_eq!(rows, vec![(1, 10), (2, 20), (3, 30)]);
}

#[tokio::test]
async fn test_cli_query_delete_still_works_after_routing_change() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_products_with_values(db).await;

    let table_path = db.join("products");

    let delete_output = Command::new(icefalldb_bin())
        .arg("query")
        .arg(&table_path)
        .arg("DELETE FROM products WHERE id = 1")
        .output()
        .unwrap();

    assert!(
        delete_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&delete_output.stderr)
    );
    let stderr = String::from_utf8_lossy(&delete_output.stderr);
    assert!(
        stderr.contains("row(s) deleted"),
        "expected 'row(s) deleted' in stderr, got: {stderr}"
    );

    let rows = select_products_id_value(&table_path);
    assert_eq!(
        rows,
        vec![(2, 20), (3, 30)],
        "id=1 should be gone after DELETE"
    );
}

#[tokio::test]
async fn test_cli_query_merge_upserts_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    setup_products_with_values(db).await;

    let table_path = db.join("products");

    // Create a unique index on `id` — required by MERGE.
    let index_output = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg("--unique")
        .arg(db)
        .arg("products")
        .arg("id")
        .output()
        .unwrap();
    assert!(
        index_output.status.success(),
        "create-index stderr: {}",
        String::from_utf8_lossy(&index_output.stderr)
    );

    // MERGE: update id=2 (value 20→77), insert id=4 (new row with value=40).
    let merge_sql = "MERGE INTO products USING \
        (VALUES (2, 77), (4, 40)) AS src(id, value) \
        ON products.id = src.id \
        WHEN MATCHED THEN UPDATE SET id = src.id, value = src.value \
        WHEN NOT MATCHED THEN INSERT (id, value) VALUES (src.id, src.value)";

    let merge_output = Command::new(icefalldb_bin())
        .arg("query")
        .arg(&table_path)
        .arg(merge_sql)
        .output()
        .unwrap();

    assert!(
        merge_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&merge_output.stderr)
    );
    let stderr = String::from_utf8_lossy(&merge_output.stderr);
    assert!(
        stderr.contains("row(s) affected"),
        "expected 'row(s) affected' in stderr, got: {stderr}"
    );

    let rows = select_products_id_value(&table_path);
    // id=1 unchanged (value=10), id=2 updated (value=77), id=3 unchanged (value=30), id=4 inserted (value=40).
    assert_eq!(
        rows,
        vec![(1, 10), (2, 77), (3, 30), (4, 40)],
        "MERGE should have updated id=2 and inserted id=4"
    );
}

/// A writer-only command runs on the lean current-thread runtime and
/// builds NO DataFusion query session, while a query does. Proven via the CLI's
/// `ICEFALLDB_REPORT_SESSIONS` instrumentation (`sessions=<n>` on stderr).
#[tokio::test]
async fn test_cli_write_path_builds_no_query_session() {
    let tmp = tempfile::tempdir().unwrap();
    setup_table(tmp.path()).await;

    // Writer-only command (create-index): must build zero query sessions.
    let out = Command::new(icefalldb_bin())
        .arg("create-index")
        .arg(tmp.path())
        .arg("products")
        .arg("id")
        .env("ICEFALLDB_REPORT_SESSIONS", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "create-index stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("sessions=0"),
        "writer-only command must build no query session; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A query builds the query session.
    let out = Command::new(icefalldb_bin())
        .arg("query")
        .arg(tmp.path().join("products"))
        .arg("SELECT COUNT(*) FROM products")
        .env("ICEFALLDB_REPORT_SESSIONS", "1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "query stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("sessions=1"),
        "query must build exactly one query session; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
