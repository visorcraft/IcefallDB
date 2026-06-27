use std::process::Command;
use std::time::Duration;

use icefalldb_server::Server;

/// Path to the built `icefalldb-server` binary. Cargo exposes this as an
/// environment variable for integration tests so we don't need to hard-code
/// `target/debug` paths.
fn server_bin() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_BIN_EXE_icefalldb-server"))
}

/// `--help` must exit successfully and not panic on conflicting short flags.
#[test]
fn help_exits_zero() {
    let output = Command::new(server_bin())
        .arg("--help")
        .output()
        .expect("failed to spawn icefalldb-server --help");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "icefalldb-server --help exited with {:#?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    // Spot-check that the help text mentions the host argument (the fixed one).
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("--host") || combined.contains("-H"),
        "help output should mention the host flag:\n{combined}"
    );
}

/// Starting the server on an ephemeral port succeeds and the `/tables` endpoint
/// responds, proving the argument parsing and binding path work end-to-end.
#[tokio::test]
async fn smoke_startup_binds() {
    let tmp = tempfile::tempdir().expect("failed to create temp directory");

    let server = Server::new_with_cache_mb(tmp.path(), 0)
        .await
        .expect("failed to create server");
    let (base_url, handle) = server
        .start_for_test()
        .await
        .expect("failed to start server on ephemeral port");

    // Poll `/tables` until the server is actually serving, up to 5 seconds.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    let mut last_status = None;
    let resp = loop {
        match reqwest::get(format!("{base_url}/tables")).await {
            Ok(resp) if resp.status().is_success() => break resp,
            Ok(resp) => {
                last_status = Some(resp.status());
                if tokio::time::Instant::now() > deadline {
                    panic!(
                        "GET /tables did not return success within 5s; last status: {:?}",
                        last_status
                    );
                }
            }
            Err(_) => {
                if tokio::time::Instant::now() > deadline {
                    panic!(
                        "GET /tables did not return success within 5s; last status: {:?}",
                        last_status
                    );
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    };

    assert!(
        resp.status().is_success(),
        "GET /tables returned {}",
        resp.status()
    );

    handle.abort();
    let _ = handle.await;
}
