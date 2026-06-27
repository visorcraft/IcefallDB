use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::ipc::writer::FileWriter;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

fn icefalldb_bin() -> PathBuf {
    std::env::var_os("CARGO_BIN_EXE_icefalldb")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let mut path = std::env::current_exe().unwrap();
            path.pop(); // deps
            path.pop(); // debug or release
            path.push("icefalldb");
            path
        })
}

fn make_id_batch(ids: Vec<i64>) -> RecordBatch {
    let schema = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    let array = Int64Array::from(ids);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(array)]).unwrap()
}

fn write_arrow_file(path: &Path, batch: &RecordBatch) {
    let file = std::fs::File::create(path).unwrap();
    let mut writer = FileWriter::try_new(file, batch.schema().as_ref()).unwrap();
    writer.write(batch).unwrap();
    writer.finish().unwrap();
}

fn create_id_table(db: &Path, table: &str) {
    let schema = serde_json::json!({
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": false, "field_id": 1}
        ],
        "partition_by": null,
        "sort": null,
        "row_group_target_rows": 1000,
        "row_group_target_bytes": 134_217_728,
        "dropped_columns": []
    });
    let schema_path = db.join("schema.json");
    std::fs::write(&schema_path, serde_json::to_vec_pretty(&schema).unwrap()).unwrap();

    let output = Command::new(icefalldb_bin())
        .arg("create")
        .arg("--schema")
        .arg(&schema_path)
        .arg(db)
        .arg(table)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "create failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn query_count(db: &Path, table: &str) -> i64 {
    let output = Command::new(icefalldb_bin())
        .arg("query")
        .arg(db.join(table))
        .arg(format!("SELECT COUNT(*) AS c FROM {}", table))
        .arg("--format")
        .arg("csv")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "count query failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let data_line = stdout.lines().nth(1).expect("CSV header + data line");
    data_line.trim().parse().expect("count value")
}

fn latest_manifest_sequence(db: &Path, table: &str) -> u64 {
    let pointer_path = db.join(table).join("_manifest.json");
    let bytes = std::fs::read(&pointer_path).unwrap();
    let pointer: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    pointer["latest"]
        .as_u64()
        .expect("manifest latest sequence")
}

/// Two concurrent `icefalldb insert` processes into the same table with disjoint
/// integer keys must not lose rows and must advance the manifest sequence.
///
/// The `_write.lock` must serialize commits so the two processes observe a
/// consistent manifest and the final table contains the union of both batches.
#[tokio::test]
async fn test_cross_process_insert_disjoint_keys() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path();
    let table = "items";
    create_id_table(db, table);

    let range_a: Vec<i64> = (0..500).collect();
    let range_b: Vec<i64> = (1000..1500).collect();

    let file_a = tmp.path().join("batch_a.arrow");
    let file_b = tmp.path().join("batch_b.arrow");
    write_arrow_file(&file_a, &make_id_batch(range_a.clone()));
    write_arrow_file(&file_b, &make_id_batch(range_b.clone()));

    let bin = icefalldb_bin();
    let child_a = Command::new(&bin)
        .arg("insert")
        .arg(db)
        .arg(table)
        .arg(&file_a)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let child_b = Command::new(&bin)
        .arg("insert")
        .arg(db)
        .arg(table)
        .arg(&file_b)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    let (result_a, result_b) = tokio::join!(
        tokio::time::timeout(
            Duration::from_secs(60),
            tokio::task::spawn_blocking(move || child_a.wait_with_output())
        ),
        tokio::time::timeout(
            Duration::from_secs(60),
            tokio::task::spawn_blocking(move || child_b.wait_with_output())
        ),
    );

    let output_a = result_a
        .expect("insert A should finish before timeout")
        .expect("spawn_blocking task A should not panic")
        .expect("child A output should be readable");
    let output_b = result_b
        .expect("insert B should finish before timeout")
        .expect("spawn_blocking task B should not panic")
        .expect("child B output should be readable");

    assert!(
        output_a.status.success(),
        "insert A failed: {}",
        String::from_utf8_lossy(&output_a.stderr)
    );
    assert!(
        output_b.status.success(),
        "insert B failed: {}",
        String::from_utf8_lossy(&output_b.stderr)
    );

    // Both batches must be present: no lost rows.
    let expected_count = (range_a.len() + range_b.len()) as i64;
    let actual_count = query_count(db, table);
    assert_eq!(
        actual_count, expected_count,
        "expected {} rows from two disjoint inserts, got {}",
        expected_count, actual_count
    );

    // The manifest must have advanced past the initial empty pointer (latest=0).
    let seq = latest_manifest_sequence(db, table);
    assert!(
        seq >= 2,
        "manifest sequence should advance after two commits, got {}",
        seq
    );
}
