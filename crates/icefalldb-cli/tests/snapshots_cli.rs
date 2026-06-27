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

/// Return the latest snapshot *data* line (leading token parses as a u64
/// sequence number), skipping the header and the WAL-fold note line.
fn latest_snapshot_data_line(out: &str) -> &str {
    out.lines()
        .find(|l| {
            l.split_whitespace()
                .next()
                .is_some_and(|t| t.parse::<u64>().is_ok())
        })
        .expect("expected a snapshot data line")
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

/// Build a table with two inserts (seq 1 and seq 2).
/// Returns (TempDir, db_path_string).
async fn setup_cli_table_two_inserts() -> (tempfile::TempDir, String) {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    let storage = LocalStorage::new(db).unwrap();

    // First insert — snapshot sequence 1.
    let mut writer = Writer::new(Arc::new(storage.clone()), "bench", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Second insert — snapshot sequence 2.
    let mut writer2 = Writer::new(Arc::new(storage), "bench", make_schema())
        .await
        .unwrap();
    writer2
        .insert_batch(make_batch(vec![4, 5, 6]))
        .await
        .unwrap();
    writer2.commit().await.unwrap();

    let db_str = db.to_str().unwrap().to_string();
    (tmp, db_str)
}

#[tokio::test]
async fn snapshots_lists_history() {
    let (_tmp, db) = setup_cli_table_two_inserts().await;
    let out = run_cli(&["snapshots", &db, "bench"]);
    // Must contain the header row and at least 2 data rows.
    assert!(
        out.contains("sequence"),
        "expected header with 'sequence', got:\n{out}"
    );
    let data_lines: Vec<&str> = out
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains("sequence"))
        .collect();
    assert!(
        data_lines.len() >= 2,
        "expected >=2 snapshot rows, got {}:\n{out}",
        data_lines.len()
    );
}

#[tokio::test]
async fn snapshots_empty_table_shows_no_snapshots() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();

    // Create table with no inserts.
    let storage = LocalStorage::new(db).unwrap();
    Writer::create(Arc::new(storage), "empty_table", make_schema())
        .await
        .unwrap();

    let db_str = db.to_str().unwrap().to_string();
    let out = run_cli(&["snapshots", &db_str, "empty_table"]);
    // A created-but-never-inserted table has no committed snapshots: the command
    // must take the "(no snapshots)" branch, not merely print the always-present
    // header (the old `contains("sequence")` check was vacuously true).
    assert!(
        out.contains("(no snapshots)"),
        "expected the (no snapshots) branch for a never-inserted table, got:\n{out}"
    );
    // And there must be zero snapshot data rows (only the header + the note).
    let data_lines: Vec<&str> = out
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.contains("sequence") && !l.contains("no snapshots"))
        .collect();
    assert!(
        data_lines.is_empty(),
        "expected zero snapshot rows, got {}:\n{out}",
        data_lines.len()
    );
}

/// After a DELETE, `snapshots` must report live rows (physical minus deleted),
/// matching the live query count. Regression test for M07.
#[tokio::test]
async fn snapshots_shows_live_rows_after_delete() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    let storage = LocalStorage::new(db).unwrap();

    // Insert 3 rows in a single row group.
    let mut writer = Writer::new(Arc::new(storage.clone()), "bench", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    // Delete row with id=1 (physical offset 0 in fragment 0).
    let mut deleter = Writer::new(Arc::new(storage.clone()), "bench", make_schema())
        .await
        .unwrap();
    let mut deletes = std::collections::HashMap::new();
    deletes.insert(0, vec![0u32]);
    deleter.commit_deletes(deletes).await.unwrap();

    let db_str = db.to_str().unwrap().to_string();

    // Live query count should be 2.
    let query_out = run_cli(&[
        "query",
        &format!("{}/bench", db_str),
        "SELECT COUNT(*) FROM bench",
    ]);
    assert!(
        query_out.contains("2"),
        "live query should return 2, got:\n{query_out}"
    );

    // Snapshot display should also show 2 rows for the latest snapshot.
    let out = run_cli(&["snapshots", &db_str, "bench"]);
    let data_lines: Vec<&str> = out
        .lines()
        .filter(|l| {
            !l.trim().is_empty() && !l.contains("sequence") && !l.contains("pending mutation")
        })
        .collect();
    assert!(
        !data_lines.is_empty(),
        "expected at least one snapshot row, got:\n{out}"
    );
    let latest_line = data_lines.last().unwrap();
    assert!(
        latest_line.contains(" 2 "),
        "latest snapshot should show 2 live rows, got:\n{out}"
    );
}

/// After a WAL-mode DELETE is folded into the checkpoint by `gc`, the folded
/// manifest is published without the denormalized `row_counts`. `snapshots` must
/// still report the live row count (from the canonical `.meta` sidecars), not 0.
/// Regression test for M07 — the post-fold case the earlier test did not cover.
#[tokio::test]
async fn snapshots_live_rows_after_wal_fold() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    let storage = LocalStorage::new(db).unwrap();

    let mut writer = Writer::new(Arc::new(storage.clone()), "bench", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let db_str = db.to_str().unwrap().to_string();
    let table_arg = format!("{}/bench", db_str);

    // DELETE through the CLI uses WAL fast-commit by default.
    run_cli(&["query", &table_arg, "DELETE FROM bench WHERE id = 1"]);
    // Fold the WAL into a fresh checkpoint; this republishes the manifest with
    // `row_counts: None`, which used to make `snapshots` display 0 live rows.
    run_cli(&["gc", &db_str, "bench"]);

    let query_out = run_cli(&["query", &table_arg, "SELECT COUNT(*) FROM bench"]);
    assert!(
        query_out.contains("2"),
        "live query should return 2 after fold, got:\n{query_out}"
    );

    let out = run_cli(&["snapshots", &db_str, "bench"]);
    let data_lines: Vec<&str> = out
        .lines()
        .filter(|l| {
            !l.trim().is_empty() && !l.contains("sequence") && !l.contains("pending mutation")
        })
        .collect();
    let latest_line = data_lines.last().expect("expected a snapshot row");
    assert!(
        latest_line.contains(" 2 "),
        "latest snapshot should show 2 live rows after WAL fold, got:\n{out}"
    );
}

/// UPDATE goes through the WAL fast-commit path and writes a patch fragment
/// (delete the original row + add a replacement row). The folded live-row count
/// must account for that patch fragment (matching `SELECT COUNT(*)`), not
/// undercount it. Regression test for the M07 patch-fragment folding path that
/// the DELETE-only tests did not cover — the count stays at 3 because UPDATE
/// deletes-then-inserts one row, but a buggy fold that ignored the patch
/// fragment's addition would report 2.
#[tokio::test]
async fn snapshots_live_rows_after_update() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    let storage = LocalStorage::new(db).unwrap();

    let mut writer = Writer::new(Arc::new(storage.clone()), "bench", make_schema())
        .await
        .unwrap();
    writer
        .insert_batch(make_batch(vec![1, 2, 3]))
        .await
        .unwrap();
    writer.commit().await.unwrap();

    let db_str = db.to_str().unwrap().to_string();
    let table_arg = format!("{}/bench", db_str);

    // UPDATE writes a patch fragment via WAL fast-commit; row count stays 3
    // (one row deleted from the original fragment, one added in the patch).
    run_cli(&["query", &table_arg, "UPDATE bench SET id = 10 WHERE id = 1"]);

    // Live query count must be 3 (UPDATE does not change the row count).
    let query_out = run_cli(&["query", &table_arg, "SELECT COUNT(*) FROM bench"]);
    assert!(
        query_out.contains("3"),
        "live query should return 3, got:\n{query_out}"
    );

    // Helper: return the latest snapshot *data* line (leading token is a u64
    // sequence number), skipping the header and the WAL-fold note line.
    // `snapshots` must show 3 for the latest (WAL-folded) snapshot. A fold that
    // ignored the patch fragment's added row would report 2.
    let out = run_cli(&["snapshots", &db_str, "bench"]);
    let latest_line = latest_snapshot_data_line(&out);
    let rows = latest_line
        .split_whitespace()
        .nth(2)
        .expect("rows column present");
    assert_eq!(
        rows, "3",
        "latest snapshot should show 3 live rows after UPDATE, got:\n{out}"
    );

    // Fold the WAL into a fresh checkpoint (republishes manifest with
    // `row_counts: None`) and re-check: the count must still be 3, proving the
    // `.meta`-based fallback counts the patch fragment's rows too.
    run_cli(&["gc", &db_str, "bench"]);
    let out = run_cli(&["snapshots", &db_str, "bench"]);
    let latest_line = latest_snapshot_data_line(&out);
    let rows = latest_line
        .split_whitespace()
        .nth(2)
        .expect("rows column present");
    assert_eq!(
        rows, "3",
        "latest snapshot should show 3 live rows after WAL fold, got:\n{out}"
    );
}
