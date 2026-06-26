use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::Writer;
use icefalldb_server::Server;
use std::sync::Arc;

/// `/sql` caches eligible SELECTs in `<db>/_query_cache`; `/mutate`
/// clears the cache so the next SELECT sees the committed state rather than a
/// stale hit.
#[tokio::test]
async fn test_result_cache_hit_and_mutate_invalidation() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();

    // Seed three rows.
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
        vec![Arc::new(Int64Array::from(vec![1i64, 2, 3]))],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();

    let sql = "SELECT COUNT(*) FROM users";

    // First request — populates the cache.
    let resp1: serde_json::Value = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({ "sql": sql }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(resp1["data"][0]["count(*)"].as_i64().unwrap(), 3);

    // A cache entry must exist in <db>/_query_cache/ after the first SELECT.
    let cache_dir = db.join("_query_cache");
    let arrow_files: Vec<_> = std::fs::read_dir(&cache_dir)
        .expect("_query_cache dir must exist")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("arrow"))
        .collect();
    assert!(
        !arrow_files.is_empty(),
        "a .arrow cache entry must exist after the first SELECT"
    );

    // Second request — same SQL and same snapshot → must be served from cache.
    let resp2: serde_json::Value = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({ "sql": sql }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp2["data"][0]["count(*)"].as_i64().unwrap(),
        3,
        "second request must return the cached count"
    );

    // Mutate: delete one row → cache must be invalidated.
    let mutate_resp = client
        .post(format!("{}/mutate", url))
        .json(&serde_json::json!({ "sql": "DELETE FROM users WHERE id = 1" }))
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

    // Re-query: must reflect the deletion, not a stale cached 3.
    let resp3: serde_json::Value = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({ "sql": sql }))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        resp3["data"][0]["count(*)"].as_i64().unwrap(),
        2,
        "post-mutate SELECT must return 2, not stale 3"
    );
}
