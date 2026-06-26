use icefalldb_core::{
    storage::{local::LocalStorage, memory::MemoryStorage, Storage},
    IcefallDBError,
};

#[tokio::test]
async fn test_local_storage_read_write() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("hello.txt", b"world").await.unwrap();
    let data = storage.read("hello.txt").await.unwrap();
    assert_eq!(data, b"world");
}

#[tokio::test]
async fn test_memory_storage_read_write() {
    let storage = MemoryStorage::new();
    storage.write("hello.txt", b"world").await.unwrap();
    let data = storage.read("hello.txt").await.unwrap();
    assert_eq!(data, b"world");
}

#[tokio::test]
async fn test_local_storage_size() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("hello.txt", b"world").await.unwrap();
    assert_eq!(storage.size("hello.txt").await.unwrap(), 5);
}

#[tokio::test]
async fn test_memory_storage_size() {
    let storage = MemoryStorage::new();
    storage.write("hello.txt", b"world").await.unwrap();
    assert_eq!(storage.size("hello.txt").await.unwrap(), 5);
}

#[tokio::test]
async fn test_local_storage_size_missing() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    let err = storage.size("missing.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_memory_storage_size_missing() {
    let storage = MemoryStorage::new();
    let err = storage.size("missing.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_local_storage_delete() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("delete-me.txt", b"data").await.unwrap();
    assert!(storage.exists("delete-me.txt").await.unwrap());
    storage.delete("delete-me.txt").await.unwrap();
    assert!(!storage.exists("delete-me.txt").await.unwrap());
}

#[tokio::test]
async fn test_memory_storage_delete() {
    let storage = MemoryStorage::new();
    storage.write("delete-me.txt", b"data").await.unwrap();
    assert!(storage.exists("delete-me.txt").await.unwrap());
    storage.delete("delete-me.txt").await.unwrap();
    assert!(!storage.exists("delete-me.txt").await.unwrap());
}

#[tokio::test]
async fn test_local_storage_rename() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("old.txt", b"content").await.unwrap();
    storage.rename("old.txt", "new.txt").await.unwrap();
    assert!(!storage.exists("old.txt").await.unwrap());
    assert!(storage.exists("new.txt").await.unwrap());
    assert_eq!(storage.read("new.txt").await.unwrap(), b"content");
}

#[tokio::test]
async fn test_memory_storage_rename() {
    let storage = MemoryStorage::new();
    storage.write("old.txt", b"content").await.unwrap();
    storage.rename("old.txt", "new.txt").await.unwrap();
    assert!(!storage.exists("old.txt").await.unwrap());
    assert!(storage.exists("new.txt").await.unwrap());
    assert_eq!(storage.read("new.txt").await.unwrap(), b"content");
}

#[tokio::test]
async fn test_local_storage_rename_missing_key() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    let err = storage.rename("missing.txt", "new.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_memory_storage_rename_missing_key() {
    let storage = MemoryStorage::new();
    let err = storage.rename("missing.txt", "new.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_local_storage_list() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("a/1.txt", b"1").await.unwrap();
    storage.write("a/2.txt", b"2").await.unwrap();
    storage.write("a/b/3.txt", b"3").await.unwrap();
    storage.write("c.txt", b"c").await.unwrap();

    let mut entries = storage.list("a").await.unwrap();
    entries.sort();
    assert_eq!(entries, vec!["a/1.txt", "a/2.txt", "a/b"]);

    let top = storage.list("").await.unwrap();
    assert_eq!(top, vec!["a", "c.txt"]);
}

#[tokio::test]
async fn test_memory_storage_list() {
    let storage = MemoryStorage::new();
    storage.write("a/1.txt", b"1").await.unwrap();
    storage.write("a/2.txt", b"2").await.unwrap();
    storage.write("a/b/3.txt", b"3").await.unwrap();
    storage.write("c.txt", b"c").await.unwrap();

    let mut entries = storage.list("a").await.unwrap();
    entries.sort();
    assert_eq!(entries, vec!["a/1.txt", "a/2.txt", "a/b"]);

    let top = storage.list("").await.unwrap();
    assert_eq!(top, vec!["a", "c.txt"]);
}

#[tokio::test]
async fn test_local_storage_exists() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    assert!(!storage.exists("nope.txt").await.unwrap());
    storage.write("yep.txt", b"yep").await.unwrap();
    assert!(storage.exists("yep.txt").await.unwrap());
}

#[tokio::test]
async fn test_memory_storage_exists() {
    let storage = MemoryStorage::new();
    assert!(!storage.exists("nope.txt").await.unwrap());
    storage.write("yep.txt", b"yep").await.unwrap();
    assert!(storage.exists("yep.txt").await.unwrap());
}

#[tokio::test]
async fn test_local_storage_directory_auto_creation() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("nested/dir/file.txt", b"data").await.unwrap();
    assert_eq!(storage.read("nested/dir/file.txt").await.unwrap(), b"data");
}

#[tokio::test]
async fn test_local_storage_missing_key_read() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    let err = storage.read("missing.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_memory_storage_missing_key_read() {
    let storage = MemoryStorage::new();
    let err = storage.read("missing.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_local_storage_path_traversal_absolute() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("safe.txt", b"safe").await.unwrap();

    let err = storage.read("/etc/passwd").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::InvalidPath(_)));
}

#[tokio::test]
async fn test_local_storage_path_traversal_dotdot() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("safe.txt", b"safe").await.unwrap();

    let err = storage.read("../outside.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::InvalidPath(_)));
}

#[tokio::test]
async fn test_local_storage_path_traversal_embedded() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();
    storage.write("safe.txt", b"safe").await.unwrap();

    let err = storage.read("foo/../../outside.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::InvalidPath(_)));
}

#[tokio::test]
async fn test_memory_storage_path_traversal() {
    let storage = MemoryStorage::new();
    storage.write("safe.txt", b"safe").await.unwrap();

    let err = storage.read("../outside.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::InvalidPath(_)));

    let err = storage.read("foo/../../outside.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::InvalidPath(_)));
}

#[tokio::test]
async fn test_memory_storage_absolute_path_rejected() {
    let storage = MemoryStorage::new();

    let err = storage.read("/etc/passwd").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::InvalidPath(_)));
}

#[tokio::test]
async fn test_storage_delete_missing_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalStorage::new(dir.path()).unwrap();
    let err = local.delete("missing.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));

    let memory = MemoryStorage::new();
    let err = memory.delete("missing.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_storage_list_nonexistent_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalStorage::new(dir.path()).unwrap();
    let err = local.list("nonexistent").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));

    let memory = MemoryStorage::new();
    let err = memory.list("nonexistent").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_storage_list_file_like_prefix_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let local = LocalStorage::new(dir.path()).unwrap();
    local.write("some.txt", b"data").await.unwrap();
    let err = local.list("some.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));

    let memory = MemoryStorage::new();
    memory.write("some.txt", b"data").await.unwrap();
    let err = memory.list("some.txt").await.unwrap_err();
    assert!(matches!(err, IcefallDBError::NotFound(_)));
}

#[tokio::test]
async fn test_memory_storage_concurrent_access() {
    use std::sync::Arc;

    let storage = Arc::new(MemoryStorage::new());
    let mut handles = vec![];

    for i in 0..10 {
        let s = storage.clone();
        handles.push(tokio::spawn(async move {
            s.write(&format!("file{}.txt", i), b"data").await.unwrap();
            s.read(&format!("file{}.txt", i)).await.unwrap();
            s.exists(&format!("file{}.txt", i)).await.unwrap();
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    let list = storage.list("").await.unwrap();
    assert_eq!(list.len(), 10);
}

async fn assert_invalid_path<S: Storage>(storage: &S, path: &str) {
    for res in [
        storage.read(path).await.map(|_| ()),
        storage.write(path, b"x").await.map(|_| ()),
        storage.delete(path).await.map(|_| ()),
        storage.rename(path, "dest").await.map(|_| ()),
        storage.rename("src", path).await.map(|_| ()),
        storage.exists(path).await.map(|_| ()),
    ] {
        assert!(matches!(res.unwrap_err(), IcefallDBError::InvalidPath(_)));
    }
    // An empty prefix is valid for `list` (it means the storage root), so only
    // exercise `list` for non-empty invalid paths.
    if !path.is_empty() {
        assert!(matches!(
            storage.list(path).await.unwrap_err(),
            IcefallDBError::InvalidPath(_)
        ));
    }
}

#[tokio::test]
async fn test_local_storage_invalid_paths_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();

    assert_invalid_path(&storage, "").await;
    assert_invalid_path(&storage, "/absolute").await;
    assert_invalid_path(&storage, "../outside").await;
    assert_invalid_path(&storage, "foo/../../outside").await;
    assert_invalid_path(&storage, "./../outside").await;
}

#[tokio::test]
async fn test_memory_storage_invalid_paths_rejected() {
    let storage = MemoryStorage::new();

    assert_invalid_path(&storage, "").await;
    assert_invalid_path(&storage, "/absolute").await;
    assert_invalid_path(&storage, "../outside").await;
    assert_invalid_path(&storage, "foo/../../outside").await;
    assert_invalid_path(&storage, "./../outside").await;
}

#[tokio::test]
async fn test_local_storage_lock_and_sync() {
    let dir = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(dir.path()).unwrap();

    let guard = storage
        .lock_exclusive("locks/write.lock", std::time::Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(guard.path(), "locks/write.lock");
    drop(guard);

    storage.sync("locks/write.lock").await.unwrap();
    storage.sync("locks").await.unwrap();
}

#[tokio::test]
async fn test_local_storage_sync_root_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();
    storage.sync_root().await.unwrap();
}

#[tokio::test]
async fn test_local_storage_append_creates_file() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = LocalStorage::new(tmp.path()).unwrap();
    storage.append("wal/000.log", b"line1\n").await.unwrap();
    storage.append("wal/000.log", b"line2\n").await.unwrap();
    let data = storage.read("wal/000.log").await.unwrap();
    assert_eq!(data, b"line1\nline2\n");
}

#[tokio::test]
async fn test_memory_storage_lock_and_sync_are_no_ops() {
    let storage = MemoryStorage::new();

    let guard = storage
        .lock_exclusive("locks/write.lock", std::time::Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(guard.path(), "locks/write.lock");
    drop(guard);

    storage.sync("locks/write.lock").await.unwrap();
    storage.sync("locks").await.unwrap();
}
