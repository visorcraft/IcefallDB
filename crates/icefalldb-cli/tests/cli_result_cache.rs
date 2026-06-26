use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::writer::Writer;
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

#[tokio::test]
async fn query_caches_eligible_select() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    // Build a small table programmatically (same pattern as cli_tests.rs).
    let storage = LocalStorage::new(db).unwrap();
    let mut writer = Writer::new(Arc::new(storage), "t", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let table_path = db.join("t");

    // Run the same SELECT twice via the CLI — both should succeed with
    // identical output (second call is a cache hit).
    let out1 = run_cli(&[
        "query",
        table_path.to_str().unwrap(),
        "SELECT COUNT(*) FROM t",
    ]);
    let out2 = run_cli(&[
        "query",
        table_path.to_str().unwrap(),
        "SELECT COUNT(*) FROM t",
    ]);

    assert_eq!(out1, out2, "cached and live results should be identical");

    // The cache dir lives at <db_root>/_query_cache (matches Python adapter).
    let cache_dir = db.join("_query_cache");
    let has_arrow_entry = std::fs::read_dir(&cache_dir)
        .map(|d| {
            d.filter_map(|e| e.ok())
                .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("arrow"))
        })
        .unwrap_or(false);
    assert!(
        has_arrow_entry,
        "expected a *.arrow cache entry under {cache_dir:?} after two identical queries"
    );
}
