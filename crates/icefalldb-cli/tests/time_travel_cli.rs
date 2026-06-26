use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::writer::Writer;
use std::process::Command;
use std::sync::Arc;

/// Parse `[{"COUNT(*)": N}]` (or any single-key integer JSON array) and return
/// the integer value, panicking with the raw output on failure.
fn parse_count(json: &str) -> i64 {
    let arr: serde_json::Value = serde_json::from_str(json.trim())
        .unwrap_or_else(|e| panic!("failed to parse CLI JSON output: {e}\noutput: {json:?}"));
    let obj = arr
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_object())
        .unwrap_or_else(|| panic!("expected JSON array with one object, got: {json:?}"));
    // The key may be "COUNT(*)" or "count(*)" depending on the SQL dialect.
    let val = obj
        .values()
        .next()
        .unwrap_or_else(|| panic!("empty object in JSON output: {json:?}"));
    val.as_i64()
        .unwrap_or_else(|| panic!("expected integer value, got: {val:?}\noutput: {json:?}"))
}

/// Run icefalldb CLI, expecting a **non-zero** exit code.  Returns (stdout, stderr).
fn run_cli_expect_failure(args: &[&str]) -> (String, String) {
    let output = Command::new(icefalldb_bin())
        .args(args)
        .output()
        .expect("failed to run icefalldb");
    assert!(
        !output.status.success(),
        "expected non-zero exit for icefalldb {:?} but it succeeded\nstdout: {}",
        args,
        String::from_utf8_lossy(&output.stdout)
    );
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

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

fn make_schema() -> Schema {
    Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 1,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1_000_000,
        row_group_target_bytes: 134_217_728,
        max_field_id: 1,
        dropped_columns: vec![],
    }
}

fn make_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

/// Run icefalldb CLI with the given args and return stdout as a string.
/// Panics with stderr on failure.
fn run_cli(args: &[&str]) -> String {
    let output = Command::new(icefalldb_bin())
        .args(args)
        .output()
        .expect("failed to run icefalldb");
    assert!(
        output.status.success(),
        "icefalldb {:?} failed:\nstderr: {}",
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Build a table with one insert (seq 1), then delete one row (seq 2).
/// Returns (db_path_string, table_path_string).
async fn setup_table_with_delete() -> (tempfile::TempDir, String, String) {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    let storage = LocalStorage::new(db).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "bench", make_schema())
        .await
        .unwrap();
    // Insert 3 rows — this becomes snapshot sequence 1.
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let table_path = db.join("bench");
    let table_path_str = table_path.to_str().unwrap().to_string();

    // Delete one row via the CLI DELETE path — this becomes snapshot sequence 2.
    let output = Command::new(icefalldb_bin())
        .args(["query", &table_path_str, "DELETE FROM bench WHERE id = 1"])
        .output()
        .expect("failed to run icefalldb DELETE");
    assert!(
        output.status.success(),
        "DELETE failed:\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let db_str = db.to_str().unwrap().to_string();
    (tmp, db_str, table_path_str)
}

#[tokio::test]
async fn query_as_of_sees_pre_delete_rows() {
    let (_tmp, _db_str, table_path) = setup_table_with_delete().await;

    // Latest snapshot sees 2 rows (one was deleted).
    let latest = run_cli(&["query", &table_path, "SELECT COUNT(*) FROM bench"]);

    // As-of snapshot 1 sees 3 rows (before the DELETE).
    let asof = run_cli(&[
        "query",
        &table_path,
        "SELECT COUNT(*) FROM bench",
        "--snapshot",
        "1",
    ]);

    assert_ne!(
        latest, asof,
        "as-of snapshot 1 must include the later-deleted row: latest={latest:?} asof={asof:?}"
    );

    // Verify concrete counts by parsing the JSON integer directly.
    let latest_count = parse_count(&latest);
    let asof_count = parse_count(&asof);
    assert_eq!(
        latest_count, 2,
        "latest query should report 2 rows, got: {latest}"
    );
    assert_eq!(
        asof_count, 3,
        "as-of query should report 3 rows, got: {asof}"
    );
}

#[tokio::test]
async fn query_snapshot_not_found_is_clean_error() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    let storage = LocalStorage::new(db).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "bench", make_schema())
        .await
        .unwrap();
    writer.insert_batch(make_batch(vec![10, 20])).await.unwrap();
    writer.commit().await.unwrap();

    let table_path = db.join("bench");
    let table_path_str = table_path.to_str().unwrap();

    // Request a snapshot that does not exist.
    let output = Command::new(icefalldb_bin())
        .args([
            "query",
            table_path_str,
            "SELECT COUNT(*) FROM bench",
            "--snapshot",
            "999",
        ])
        .output()
        .expect("failed to run icefalldb");

    assert!(
        !output.status.success(),
        "expected failure for missing snapshot but got success"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("999") || stderr.contains("snapshot"),
        "expected informative error mentioning snapshot 999, got: {stderr}"
    );
}

/// --snapshot and --server are mutually exclusive; CLI must reject the combination
/// with a non-zero exit and a message containing both flag names.
#[tokio::test]
async fn snapshot_and_server_flags_are_incompatible() {
    let (_tmp, _db_str, table_path) = setup_table_with_delete().await;

    let (_stdout, stderr) = run_cli_expect_failure(&[
        "query",
        &table_path,
        "SELECT COUNT(*) FROM bench",
        "--snapshot",
        "1",
        "--server",
        "http://127.0.0.1:19999", // nothing listening; error fires before network call
    ]);

    assert!(
        stderr.contains("--snapshot") && stderr.contains("--server"),
        "error message should mention both --snapshot and --server, got: {stderr}"
    );
}

/// DELETE with --snapshot must fail with a clear read-only error, not a
/// confusing DataFusion/engine error.
#[tokio::test]
async fn snapshot_with_mutation_is_rejected() {
    let (_tmp, _db_str, table_path) = setup_table_with_delete().await;

    let (_stdout, stderr) = run_cli_expect_failure(&[
        "query",
        &table_path,
        "DELETE FROM bench WHERE id = 2",
        "--snapshot",
        "1",
    ]);

    assert!(
        stderr.contains("read-only") || stderr.contains("--snapshot"),
        "error message should mention read-only or --snapshot, got: {stderr}"
    );
}
