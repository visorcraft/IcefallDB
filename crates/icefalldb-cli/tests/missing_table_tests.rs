use std::process::Command;

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

fn assert_missing_table_error(output: &std::process::Output, command: &str) {
    assert!(
        !output.status.success(),
        "{} should fail for a missing table",
        command
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("table 'missing' not found"),
        "{} stderr did not report missing table: {}",
        command,
        stderr
    );
}

#[tokio::test]
async fn test_compact_fails_on_missing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(icefalldb_bin())
        .arg("compact")
        .arg(tmp.path())
        .arg("missing")
        .output()
        .unwrap();
    assert_missing_table_error(&output, "compact");
}

#[tokio::test]
async fn test_optimize_fails_on_missing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(icefalldb_bin())
        .arg("optimize")
        .arg(tmp.path())
        .arg("missing")
        .output()
        .unwrap();
    assert_missing_table_error(&output, "optimize");
}

#[tokio::test]
async fn test_gc_fails_on_missing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(icefalldb_bin())
        .arg("gc")
        .arg(tmp.path())
        .arg("missing")
        .output()
        .unwrap();
    assert_missing_table_error(&output, "gc");
}

#[tokio::test]
async fn test_doctor_diagnose_fails_on_missing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg(tmp.path())
        .arg("missing")
        .output()
        .unwrap();
    assert_missing_table_error(&output, "doctor");
}

#[tokio::test]
async fn test_doctor_repair_fails_on_missing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(icefalldb_bin())
        .arg("doctor")
        .arg("--repair")
        .arg(tmp.path())
        .arg("missing")
        .output()
        .unwrap();
    assert_missing_table_error(&output, "doctor --repair");
}

#[tokio::test]
async fn test_snapshots_fails_on_missing_table() {
    let tmp = tempfile::tempdir().unwrap();
    let output = Command::new(icefalldb_bin())
        .arg("snapshots")
        .arg(tmp.path())
        .arg("missing")
        .output()
        .unwrap();
    assert_missing_table_error(&output, "snapshots");
}
