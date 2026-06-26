use icefalldb_core::database_catalog::DatabaseCatalog;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::schema_util::{DEFAULT_ROW_GROUP_TARGET_BYTES, DEFAULT_ROW_GROUP_TARGET_ROWS};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_server::Server;
use std::sync::Arc;
use std::time::Duration;

fn test_schema(columns: Vec<Column>) -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: DEFAULT_ROW_GROUP_TARGET_ROWS,
        row_group_target_bytes: DEFAULT_ROW_GROUP_TARGET_BYTES,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

#[tokio::test]
async fn test_multi_table_transaction_commits() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();
    let storage = Arc::new(LocalStorage::new(&db).unwrap());

    let catalog = DatabaseCatalog::new(storage);
    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    let schema = test_schema(vec![
        Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        },
        Column {
            name: "name".into(),
            r#type: "utf8".into(),
            nullable: true,
            field_id: 0,
        },
    ]);
    catalog
        .create_table(&guard, "users", &schema)
        .await
        .unwrap();
    catalog
        .create_table(&guard, "orders", &schema)
        .await
        .unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();

    let begin = client
        .post(format!("{}/tx/begin", url))
        .send()
        .await
        .unwrap();
    let tx_id = begin.json::<serde_json::Value>().await.unwrap()["tx_id"]
        .as_str()
        .unwrap()
        .to_string();

    client
        .post(format!("{}/tx/sql", url))
        .json(&serde_json::json!({"tx_id": tx_id, "sql": "INSERT INTO users VALUES (1, 'alice')"}))
        .send()
        .await
        .unwrap();
    client
        .post(format!("{}/tx/sql", url))
        .json(
            &serde_json::json!({"tx_id": tx_id, "sql": "INSERT INTO orders VALUES (1, 'order-1')"}),
        )
        .send()
        .await
        .unwrap();

    let commit = client
        .post(format!("{}/tx/commit", url))
        .json(&serde_json::json!({"tx_id": tx_id}))
        .send()
        .await
        .unwrap();
    assert!(commit.status().is_success());

    let users = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({"sql": "SELECT * FROM users"}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(users["data"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn test_rollback_discards_inserts() {
    let tmp = tempfile::tempdir().unwrap();
    let db = tmp.path().to_path_buf();
    let storage = Arc::new(LocalStorage::new(&db).unwrap());

    let catalog = DatabaseCatalog::new(storage);
    let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
    let schema = test_schema(vec![Column {
        name: "id".into(),
        r#type: "int64".into(),
        nullable: false,
        field_id: 0,
    }]);
    catalog
        .create_table(&guard, "events", &schema)
        .await
        .unwrap();

    let server = Server::new(&db).await.unwrap();
    let (url, _handle) = server.start_for_test().await.unwrap();
    let client = reqwest::Client::new();

    let begin = client
        .post(format!("{}/tx/begin", url))
        .send()
        .await
        .unwrap();
    let tx_id = begin.json::<serde_json::Value>().await.unwrap()["tx_id"]
        .as_str()
        .unwrap()
        .to_string();

    client
        .post(format!("{}/tx/sql", url))
        .json(&serde_json::json!({"tx_id": tx_id, "sql": "INSERT INTO events VALUES (1)"}))
        .send()
        .await
        .unwrap();
    client
        .post(format!("{}/tx/rollback", url))
        .json(&serde_json::json!({"tx_id": tx_id}))
        .send()
        .await
        .unwrap();

    let events = client
        .post(format!("{}/sql", url))
        .json(&serde_json::json!({"sql": "SELECT * FROM events"}))
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await
        .unwrap();
    assert_eq!(events["data"].as_array().unwrap().len(), 0);
}
