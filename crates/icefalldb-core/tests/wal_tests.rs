use icefalldb_core::recovery::recover;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::wal::{LogEntryBody, Wal, WalReader};
use std::sync::Arc;

#[tokio::test]
async fn test_wal_append_and_read() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let wal = Wal::open(Arc::clone(&storage)).await.unwrap();

    wal.append(LogEntryBody::Begin {
        tx_id: "tx-1".into(),
    })
    .await
    .unwrap();
    wal.append(LogEntryBody::Commit {
        tx_id: "tx-1".into(),
    })
    .await
    .unwrap();

    let reader = WalReader::new(&*storage);
    let entries = reader.entries().await.unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].tx_id(), "tx-1");
    assert_eq!(entries[1].tx_id(), "tx-1");
}

#[tokio::test]
async fn test_wal_lsn_assignment_is_monotonic() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let wal = Wal::open(storage).await.unwrap();

    let e1 = wal
        .append(LogEntryBody::Begin { tx_id: "a".into() })
        .await
        .unwrap();
    let e2 = wal
        .append(LogEntryBody::Begin { tx_id: "b".into() })
        .await
        .unwrap();
    assert!(e2.lsn > e1.lsn);
}

#[tokio::test]
async fn test_recovery_classifies_transactions() {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let wal = Wal::open(Arc::clone(&storage)).await.unwrap();

    wal.append(LogEntryBody::Begin {
        tx_id: "committed".into(),
    })
    .await
    .unwrap();
    wal.append(LogEntryBody::Commit {
        tx_id: "committed".into(),
    })
    .await
    .unwrap();
    wal.append(LogEntryBody::Begin {
        tx_id: "rolled".into(),
    })
    .await
    .unwrap();
    wal.append(LogEntryBody::Rollback {
        tx_id: "rolled".into(),
    })
    .await
    .unwrap();
    wal.append(LogEntryBody::Begin {
        tx_id: "undecided".into(),
    })
    .await
    .unwrap();

    let state = recover(&*storage).await.unwrap();
    assert_eq!(state.committed, vec!["committed"]);
    assert_eq!(state.aborted, vec!["rolled"]);
    assert_eq!(state.undecided, vec!["undecided"]);
}
