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

/// Regression (M01 server follow-up): the HTTP `/index` endpoint must honor the
/// `UNIQUE` keyword. A `CREATE UNIQUE INDEX` over duplicate live keys must fail,
/// and the catalog must be left unchanged (build-before-catalog ordering — no
/// dangling definition on failure). Previously the handler hardcoded
/// `unique: false` and wrote the catalog entry *before* building the index.
#[tokio::test]
async fn test_create_unique_index_rejects_duplicates_and_leaves_catalog_clean() {
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
    let mut writer = Writer::create(storage, "dupes", schema).await.unwrap();
    // Two rows share id=1 → a unique index on `id` must be rejected.
    let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "id",
        DataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int64Array::from(vec![1, 1, 2]))],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();

    let resp = client
        .post(format!("{}/index", url))
        .json(&serde_json::json!({
            "sql": "CREATE UNIQUE INDEX dup_id_idx ON dupes (id)"
        }))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body: serde_json::Value = resp.json().await.unwrap();
    // The build must fail with a uniqueness violation (mapped to 500 Internal).
    assert!(
        !status.is_success(),
        "CREATE UNIQUE INDEX over duplicate keys should fail, got status {status}: {body}"
    );
    let msg = body
        .get("error")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_lowercase();
    assert!(
        msg.contains("unique") || msg.contains("duplicate"),
        "expected a uniqueness/duplicate error, got: {body:?}"
    );

    // Build-before-catalog: the failed build must not have left a catalog entry.
    let cat = icefalldb_core::database_catalog::DatabaseCatalog::new(Arc::new(
        LocalStorage::new(&db).unwrap(),
    ));
    let loaded = cat.load().await.unwrap();
    assert!(
        !loaded.indexes.contains_key("dup_id_idx"),
        "failed unique-index build must not leave a dangling catalog definition: {loaded:?}"
    );

    // A plain (non-unique) index on the same column must succeed, proving the
    // endpoint still works and that the rejection above was due to uniqueness.
    let resp = client
        .post(format!("{}/index", url))
        .json(&serde_json::json!({
            "sql": "CREATE INDEX dup_id_plain ON dupes (id)"
        }))
        .send()
        .await
        .unwrap();
    assert!(
        resp.status().is_success(),
        "non-unique CREATE INDEX should succeed, got {}: {}",
        resp.status(),
        resp.text().await.unwrap()
    );
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "created");
}

#[tokio::test]
async fn test_sql_unknown_table_is_client_error_not_500() {
    // A query naming a table that does not exist is a *client* error (bad SQL),
    // not a server fault: it must map to 4xx, never 500. Regression guard for the
    // daemon surfacing DataFusion planning errors as HTTP 500.
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
    Writer::create(storage, "real", schema).await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({"sql": "SELECT COUNT(*) FROM does_not_exist"}))
        .send()
        .await
        .unwrap();
    let status = resp.status();
    let body = resp.text().await.unwrap();
    assert!(
        status.is_client_error(),
        "unknown-table query must be a 4xx client error, got {status}: {body}"
    );
    assert!(
        body.to_lowercase().contains("not found") || body.contains("does_not_exist"),
        "error body should explain the missing table: {body}"
    );
}
