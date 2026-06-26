use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::Writer;
use icefalldb_server::Server;
use std::sync::Arc;

/// `POST /sql` with an optional `snapshot` field performs time-travel reads.
///
/// Scenario:
///   - Seed 3 rows and commit (snapshot 1).
///   - DELETE one row via `/mutate` (snapshot 2 / latest).
///   - `POST /sql {sql, snapshot:1}` → pre-delete count = 3.
///   - `POST /sql {sql}` (no snapshot) → post-delete count = 2.
///   - `POST /sql {sql, snapshot:999}` → HTTP 404.
#[tokio::test]
async fn test_time_travel_snapshot_read() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();

    // Seed three rows and commit → snapshot 1.
    let storage = Arc::new(LocalStorage::new(&db).unwrap());
    let schema = Schema {
        schema_id: 1,
        columns: vec![Column::new("id", "int64", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    let mut writer = Writer::create(storage, "events", schema).await.unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(vec![1i64, 2, 3]))],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap(); // → snapshot 1

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();
    let sql = "SELECT COUNT(*) FROM events";

    // Delete one row via /mutate → snapshot 2 (latest).
    let mutate_resp = client
        .post(format!("{}/mutate", url))
        .json(&serde_json::json!({ "sql": "DELETE FROM events WHERE id = 1" }))
        .send()
        .await
        .unwrap();
    assert!(
        mutate_resp.status().is_success(),
        "DELETE must succeed: {}",
        mutate_resp.status()
    );
    let body: serde_json::Value = mutate_resp.json().await.unwrap();
    assert_eq!(body["affected"], 1, "DELETE must affect exactly one row");

    // snapshot=1 → pre-delete state: count must be 3.
    let resp_snap1: serde_json::Value = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({ "sql": sql, "snapshot": 1u64 }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp_snap1["data"][0]["count(*)"].as_i64().unwrap(),
        3,
        "snapshot=1 must see pre-delete count of 3"
    );

    // No snapshot → latest: count must be 2.
    let resp_latest: serde_json::Value = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({ "sql": sql }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp_latest["data"][0]["count(*)"].as_i64().unwrap(),
        2,
        "no snapshot (latest) must see post-delete count of 2"
    );

    // snapshot=999 → does not exist → HTTP 404.
    let resp_404 = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({ "sql": sql, "snapshot": 999u64 }))
        .send()
        .await
        .unwrap();
    assert_eq!(
        resp_404.status(),
        reqwest::StatusCode::NOT_FOUND,
        "non-existent snapshot must return 404"
    );
}
