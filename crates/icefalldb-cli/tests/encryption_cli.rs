//! End-to-end CLI test: create an encrypted table with `import --encrypt`,
//! confirm the data is encrypted at rest, and read it back with `query`.
//!
//! Only built when the `encryption` feature is on (the default).
#![cfg(feature = "encryption")]

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_icefalldb")
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[test]
fn encrypted_import_and_query_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("orders.tsv");
    std::fs::write(
        &tsv,
        "order_id\tcategory\tamount\n1\tbooks\t9.99\n2\tgames\t39.50\n3\tbooks\t14.00\n",
    )
    .unwrap();

    // 16-byte (AES-128) footer key. Key id for table `orders` at schema_id 1 is
    // `orders-v1`, so the env var is `ICEFALLDB_KEY_ORDERS_V1`.
    let key = "000102030405060708090a0b0c0d0e0f";

    // create + load encrypted
    let out = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "orders",
            tsv.to_str().unwrap(),
            "--encrypt",
        ])
        .env("ICEFALLDB_KEY_ORDERS_V1", key)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "import failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // the marker exists and the data is genuinely encrypted at rest
    assert!(db.join("orders/_encryption.json").exists());
    let rg = std::fs::read_dir(db.join("orders"))
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().ends_with(".parquet"))
        .expect("a parquet row group");
    let raw = std::fs::read(rg.path()).unwrap();
    assert!(
        !contains(&raw, b"books"),
        "plaintext value leaked into the encrypted parquet file"
    );

    let table_dir = db.join("orders");

    // read it back with the key
    let out = Command::new(bin())
        .args([
            "query",
            table_dir.to_str().unwrap(),
            "SELECT category, SUM(amount) AS rev FROM orders GROUP BY category ORDER BY category",
            "--format",
            "csv",
        ])
        .env("ICEFALLDB_KEY_ORDERS_V1", key)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("books"), "missing books row: {stdout}");
    assert!(stdout.contains("games"), "missing games row: {stdout}");

    // without the key the read fails (no silent garbage)
    let out = Command::new(bin())
        .args([
            "query",
            table_dir.to_str().unwrap(),
            "SELECT category FROM orders",
        ])
        .env_remove("ICEFALLDB_KEY_ORDERS_V1")
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "query without the key unexpectedly succeeded"
    );
    // The failure must be a clear missing-key error, not an opaque Parquet
    // decrypt failure: a uniform-encrypted table reading a data column requires
    // the footer key (M10 lazy-resolution must not take the no-key path here).
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("encryption key not found"),
        "expected a clear missing-key error, got: {stderr}"
    );
}

#[test]
fn encrypted_stats_not_leaked_in_plaintext_sidecars() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    // Distinctive values that should never appear in a plaintext sidecar.
    std::fs::write(&tsv, "id\tsecret\n1\t987654321\n2\t123456789\n").unwrap();
    let key = "000102030405060708090a0b0c0d0e0f";

    let out = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "import: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Recursively scan EVERY file under the table directory (data, .meta, .agg,
    // _checkpoints/, _manifests/, _manifest.json, ...). The only file allowed to
    // hold the encrypted column's bytes is the encrypted `.parquet` itself; no
    // plaintext metadata sidecar may contain the values, and no `.agg` may exist.
    let mut stack = vec![db.join("t")];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path.to_string_lossy().to_string();
            assert!(
                !name.ends_with(".agg"),
                "encrypted table wrote an .agg sidecar: {name}"
            );
            if name.ends_with(".parquet") {
                continue; // the encrypted data file is allowed to hold the bytes
            }
            let bytes = std::fs::read(&path).unwrap_or_default();
            assert!(
                !contains(&bytes, b"987654321") && !contains(&bytes, b"123456789"),
                "encrypted column value leaked into plaintext metadata file {name}"
            );
        }
    }
}

#[test]
fn rejects_compaction_of_encrypted_table() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    std::fs::write(&tsv, "id\tsecret\n1\t42\n").unwrap();
    let key = "000102030405060708090a0b0c0d0e0f";
    let imp = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(imp.status.success());

    for cmd in [["compact"], ["optimize"]] {
        let out = Command::new(bin())
            .args([cmd[0], db.to_str().unwrap(), "t"])
            .env("ICEFALLDB_KEY_T_V1", key)
            .output()
            .unwrap();
        assert!(
            !out.status.success(),
            "{} of an encrypted table should be rejected",
            cmd[0]
        );
        assert!(String::from_utf8_lossy(&out.stderr).contains("not supported"));
    }
}

#[test]
fn rejects_nonexistent_encrypt_column() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    std::fs::write(&tsv, "id\tname\n1\ta\n").unwrap();
    let out = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt-column",
            "nope",
        ])
        .env("ICEFALLDB_KEY_T_V1", "000102030405060708090a0b0c0d0e0f")
        .env(
            "ICEFALLDB_KEY_T_V1_NOPE",
            "000102030405060708090a0b0c0d0e0f",
        )
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "a mistyped column should be rejected"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("is not a column"));
}

#[test]
fn encrypted_check_with_key_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    std::fs::write(&tsv, "id\tsecret\n1\t42\n").unwrap();
    let key = "000102030405060708090a0b0c0d0e0f";

    let imp = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(
        imp.status.success(),
        "import: {}",
        String::from_utf8_lossy(&imp.stderr)
    );

    let out = Command::new(bin())
        .args(["check", db.to_str().unwrap(), "t"])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "check failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("check passed"), "stdout: {stdout}");
    assert!(
        !stdout.contains("ENCRYPTION_KEY_UNAVAILABLE"),
        "should not skip data-page validation with keys: {stdout}"
    );
}

#[test]
fn encrypted_check_without_key_reports_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    std::fs::write(&tsv, "id\tsecret\n1\t42\n").unwrap();
    let key = "000102030405060708090a0b0c0d0e0f";

    let imp = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(
        imp.status.success(),
        "import: {}",
        String::from_utf8_lossy(&imp.stderr)
    );

    let out = Command::new(bin())
        .args(["check", db.to_str().unwrap(), "t"])
        .env_remove("ICEFALLDB_KEY_T_V1")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "check failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("check passed"), "stdout: {stdout}");
    assert!(
        stdout.contains("ENCRYPTION_KEY_UNAVAILABLE"),
        "expected encryption-specific message: {stdout}"
    );
}

#[test]
fn rejects_encrypt_and_encrypt_column_together() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    std::fs::write(&tsv, "id\tname\n1\ta\n").unwrap();
    let out = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt",
            "--encrypt-column",
            "name",
        ])
        .env("ICEFALLDB_KEY_T_V1", "000102030405060708090a0b0c0d0e0f")
        .env(
            "ICEFALLDB_KEY_T_V1_NAME",
            "000102030405060708090a0b0c0d0e0f",
        )
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "ambiguous flag combo should be rejected"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("not both"));
}

#[test]
fn per_column_encryption_plaintext_column_query_needs_no_column_key() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv = tmp.path().join("t.tsv");
    std::fs::write(&tsv, "id\tsecret\n1\talpha\n2\tbeta\n").unwrap();
    let key = "000102030405060708090a0b0c0d0e0f";

    // Import with per-column encryption on `secret`; both keys present.
    let imp = Command::new(bin())
        .args([
            "import",
            db.to_str().unwrap(),
            "t",
            tsv.to_str().unwrap(),
            "--encrypt-column",
            "secret",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .env("ICEFALLDB_KEY_T_V1_SECRET", key)
        .output()
        .unwrap();
    assert!(
        imp.status.success(),
        "import: {}",
        String::from_utf8_lossy(&imp.stderr)
    );

    let table_dir = db.join("t");

    // Query only the plaintext `id` column with only the footer key.
    let out = Command::new(bin())
        .args([
            "query",
            table_dir.to_str().unwrap(),
            "SELECT id FROM t ORDER BY id",
            "--format",
            "csv",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .env_remove("ICEFALLDB_KEY_T_V1_SECRET")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "plaintext-column query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("1"), "missing row 1: {stdout}");
    assert!(stdout.contains("2"), "missing row 2: {stdout}");

    // Querying the encrypted column without its key must fail.
    let out = Command::new(bin())
        .args(["query", table_dir.to_str().unwrap(), "SELECT secret FROM t"])
        .env("ICEFALLDB_KEY_T_V1", key)
        .env_remove("ICEFALLDB_KEY_T_V1_SECRET")
        .output()
        .unwrap();
    assert!(
        !out.status.success(),
        "encrypted-column query without key unexpectedly succeeded"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("secret")
            || stderr.contains("EncryptionKeyNotFound")
            || stderr.contains("No column decryption key"),
        "expected missing-key error, got: {stderr}"
    );
}

#[test]
fn encrypted_table_snapshot_time_travel_is_read_only() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().join("db");
    let tsv1 = tmp.path().join("t1.tsv");
    let tsv2 = tmp.path().join("t2.tsv");
    std::fs::write(&tsv1, "id\tname\n1\ta\n").unwrap();
    std::fs::write(&tsv2, "id\tname\n2\tb\n").unwrap();
    let key = "000102030405060708090a0b0c0d0e0f";

    // Two imports produce snapshot 1 (one row) and snapshot 2 (two rows).
    for tsv in [&tsv1, &tsv2] {
        let imp = Command::new(bin())
            .args([
                "import",
                db.to_str().unwrap(),
                "t",
                tsv.to_str().unwrap(),
                "--encrypt",
            ])
            .env("ICEFALLDB_KEY_T_V1", key)
            .output()
            .unwrap();
        assert!(
            imp.status.success(),
            "import: {}",
            String::from_utf8_lossy(&imp.stderr)
        );
    }

    let table_dir = db.join("t");

    let latest = Command::new(bin())
        .args([
            "query",
            table_dir.to_str().unwrap(),
            "SELECT COUNT(*) FROM t",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(
        latest.status.success(),
        "latest query: {}",
        String::from_utf8_lossy(&latest.stderr)
    );

    let asof = Command::new(bin())
        .args([
            "query",
            table_dir.to_str().unwrap(),
            "SELECT COUNT(*) FROM t",
            "--snapshot",
            "1",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(
        asof.status.success(),
        "snapshot query failed: {}",
        String::from_utf8_lossy(&asof.stderr)
    );
    assert_ne!(
        String::from_utf8_lossy(&latest.stdout).trim(),
        String::from_utf8_lossy(&asof.stdout).trim(),
        "snapshot 1 must differ from latest"
    );

    // Mutations on a snapshot-pinned encrypted table are rejected.
    let del = Command::new(bin())
        .args([
            "query",
            table_dir.to_str().unwrap(),
            "DELETE FROM t WHERE id = 1",
            "--snapshot",
            "1",
        ])
        .env("ICEFALLDB_KEY_T_V1", key)
        .output()
        .unwrap();
    assert!(!del.status.success(), "DELETE on snapshot should fail");
    let stderr = String::from_utf8_lossy(&del.stderr);
    assert!(
        stderr.contains("read-only") || stderr.contains("--snapshot"),
        "expected read-only error, got: {stderr}"
    );
}
