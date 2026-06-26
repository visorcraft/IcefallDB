use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::Writer;
use icefalldb_server::Server;
use std::sync::Arc;

#[tokio::test]
async fn test_sql_selects_inserted_rows() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();

    let storage = Arc::new(LocalStorage::new(&db).unwrap());
    let schema = Schema {
        schema_id: 1,
        columns: vec![
            Column::new("id", "int64", false),
            Column::new("name", "utf8", true),
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    let mut writer = Writer::create(storage, "users", schema).await.unwrap();

    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("alice"), Some("bob")])),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({"sql": "SELECT * FROM users ORDER BY id"}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();

    let rows = resp.get("data").unwrap().as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], 1);
    assert_eq!(rows[0]["name"], "alice");
}

#[tokio::test]
async fn test_tables_endpoint_lists_discovered_tables() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();

    let storage = Arc::new(LocalStorage::new(&db).unwrap());
    let schema = Schema {
        schema_id: 1,
        columns: vec![Column::new("id", "int64", false)],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 1000,
        row_group_target_bytes: 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    let mut writer = Writer::create(storage, "events", schema).await.unwrap();
    writer.commit().await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/tables", url))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();

    let tables = resp.get("tables").unwrap().as_array().unwrap();
    assert_eq!(tables, &["events"]);
}

/// `POST /mutate` runs a DELETE through the daemon's writer and
/// incrementally refreshes the registered provider, so a follow-up `/sql` on the
/// SAME server instance reflects it WITHOUT a restart — and a fresh server (full
/// reload) sees the identical committed state.
#[tokio::test]
async fn test_mutate_endpoint_refreshes_without_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();

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
    let mut writer = Writer::create(storage, "users", schema).await.unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();

    async fn count(client: &reqwest::Client, url: &str, sql: &str) -> i64 {
        let resp = client
            .post(format!("{}/sql", url))
            .json(&serde_json::json!({ "sql": sql }))
            .send()
            .await
            .unwrap()
            .json::<serde_json::Value>()
            .await
            .unwrap();
        resp["data"][0]["count(*)"].as_i64().unwrap()
    }

    // Mutate via /mutate.
    let resp = client
        .post(format!("{}/mutate", url))
        .json(&serde_json::json!({"sql": "DELETE FROM users WHERE id = 2"}))
        .send()
        .await
        .unwrap();
    assert!(resp.status().is_success(), "status: {}", resp.status());
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["affected"], 1, "DELETE must affect one row");

    // Visible on the SAME server without restart (incremental refresh).
    assert_eq!(count(&client, &url, "SELECT COUNT(*) FROM users").await, 2);
    assert_eq!(
        count(&client, &url, "SELECT COUNT(*) FROM users WHERE id = 2").await,
        0,
        "the deleted row must be gone"
    );

    // A fresh server (full reload) sees the identical committed state.
    let server2 = Server::new(&db).await.unwrap();
    let (url2, _h2) = server2.start_for_test().await.unwrap();
    assert_eq!(count(&client, &url2, "SELECT COUNT(*) FROM users").await, 2);
    assert_eq!(
        count(&client, &url2, "SELECT COUNT(*) FROM users WHERE id = 2").await,
        0,
    );
}

/// Hardening: concurrent `/mutate` requests against the
/// same table must all succeed and apply — the daemon serializes the
/// locate→commit→refresh so none fails the stale-provider apply check or loses an
/// update. (Without the mutate lock, racing requests read a stale snapshot.)
#[tokio::test]
async fn test_mutate_concurrent_requests_serialize() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();

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
    let mut writer = Writer::create(storage, "users", schema).await.unwrap();
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from((0..10i64).collect::<Vec<_>>()))],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();

    // Fire 6 concurrent DELETEs of distinct rows.
    let mut handles = Vec::new();
    for id in 0..6i64 {
        let c = client.clone();
        let u = url.clone();
        handles.push(tokio::spawn(async move {
            c.post(format!("{u}/mutate"))
                .json(&serde_json::json!({"sql": format!("DELETE FROM users WHERE id = {id}")}))
                .send()
                .await
                .unwrap()
        }));
    }
    for h in handles {
        let resp = h.await.unwrap();
        assert!(
            resp.status().is_success(),
            "concurrent /mutate must not 500"
        );
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["affected"], 1);
    }

    // All 6 distinct deletes applied → 4 rows remain.
    let resp = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({"sql": "SELECT COUNT(*) FROM users"}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(resp["data"][0]["count(*)"].as_i64().unwrap(), 4);
}
