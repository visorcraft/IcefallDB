use super::{path::validate_path, LockGuard, Storage};
use crate::{IcefallDBError, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Process-global count of durability barriers (`fsync`/`fdatasync`) issued by
/// every [`LocalStorage`]. Used to measure the per-commit fsync floor (the
/// `commit_floor` benchmark) and to gate fsync-coalescing changes. Monotonic;
/// read [`global_fsync_count`] deltas around an operation.
static FSYNC_COUNT: AtomicU64 = AtomicU64::new(0);

/// Total durability barriers issued since process start (see [`FSYNC_COUNT`]).
pub fn global_fsync_count() -> u64 {
    FSYNC_COUNT.load(Ordering::Relaxed)
}

#[derive(Debug, Clone)]
pub struct LocalStorage {
    root: PathBuf,
    writer_semaphores: Arc<Mutex<HashMap<String, Weak<Semaphore>>>>,
    lock_call_count: Arc<AtomicUsize>,
}

impl LocalStorage {
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = std::path::absolute(root.as_ref())?;
        Ok(Self {
            root,
            writer_semaphores: Arc::new(Mutex::new(HashMap::new())),
            lock_call_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Return the absolute local filesystem root.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Resolve a user-supplied relative path into an absolute path under
    /// `self.root` after validation.
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let cleaned = validate_path(path)?;
        Ok(self.root.join(&cleaned))
    }
}

fn map_io_err(err: std::io::Error, path: &str) -> IcefallDBError {
    match err.kind() {
        std::io::ErrorKind::NotFound => IcefallDBError::NotFound(path.to_string()),
        _ => IcefallDBError::Io(err),
    }
}

fn map_list_io_err(err: std::io::Error, prefix: &str) -> IcefallDBError {
    match err.kind() {
        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory => {
            IcefallDBError::NotFound(prefix.to_string())
        }
        _ => IcefallDBError::Io(err),
    }
}

/// Advisory lock guard for [`LocalStorage`].
///
/// Holds an open file with an exclusive `flock()` and an owned permit from the
/// in-process per-path semaphore. Both locks are released when the guard is
/// dropped, providing cross-process exclusion (via `flock()`) and in-process
/// exclusion (via the semaphore).
pub struct LocalLockGuard {
    file: File,
    path: String,
    _permit: OwnedSemaphorePermit,
}

impl LockGuard for LocalLockGuard {
    fn path(&self) -> &str {
        &self.path
    }
}

impl Drop for LocalLockGuard {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

#[async_trait]
impl Storage for LocalStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn local_root(&self) -> Option<&std::path::Path> {
        Some(&self.root)
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        let full = self.resolve_path(path)?;
        tokio::fs::read(&full)
            .await
            .map_err(|e| map_io_err(e, path))
    }

    async fn size(&self, path: &str) -> Result<u64> {
        let full = self.resolve_path(path)?;
        let metadata = tokio::fs::metadata(&full)
            .await
            .map_err(|e| map_io_err(e, path))?;
        Ok(metadata.len())
    }

    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        use tokio::io::{AsyncReadExt, AsyncSeekExt};
        let full = self.resolve_path(path)?;
        let mut file = tokio::fs::File::open(&full)
            .await
            .map_err(|e| map_io_err(e, path))?;
        file.seek(std::io::SeekFrom::Start(offset))
            .await
            .map_err(|e| map_io_err(e, path))?;
        let mut buf = vec![0u8; len as usize];
        file.read_exact(&mut buf)
            .await
            .map_err(|e| map_io_err(e, path))?;
        Ok(buf)
    }

    async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        let full = self.resolve_path(path)?;
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&full, data).await.map_err(Into::into)
    }

    async fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        use tokio::fs::OpenOptions;
        use tokio::io::AsyncWriteExt;

        let full = self.resolve_path(path)?;
        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(full)
            .await?;
        file.write_all(data).await?;
        Ok(())
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let full = self.resolve_path(path)?;
        tokio::fs::remove_file(&full)
            .await
            .map_err(|e| map_io_err(e, path))
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full = if prefix.is_empty() {
            self.root.clone()
        } else {
            self.root.join(validate_path(prefix)?)
        };

        let mut entries = vec![];
        let mut reader = tokio::fs::read_dir(&full)
            .await
            .map_err(|e| map_list_io_err(e, prefix))?;

        while let Some(entry) = reader.next_entry().await? {
            if let Ok(p) = entry.path().strip_prefix(&self.root) {
                entries.push(p.to_string_lossy().replace('\\', "/"));
            }
        }
        entries.sort();
        Ok(entries)
    }

    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        let from_full = self.resolve_path(from)?;
        let to_full = self.resolve_path(to)?;
        if let Some(parent) = to_full.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::rename(&from_full, &to_full)
            .await
            .map_err(|e| map_io_err(e, from))
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let full = self.resolve_path(path)?;
        tokio::fs::try_exists(&full).await.map_err(Into::into)
    }

    /// Acquire an exclusive advisory lock on the file at `path`.
    ///
    /// The returned guard provides both in-process and cross-process exclusion.
    /// An owned permit from a per-path `tokio::sync::Semaphore` serializes
    /// writers within the same process, and `fs2::try_lock_exclusive` provides
    /// non-blocking `flock()`-based exclusion across processes. The flock call
    /// is retried with an asynchronous sleep between attempts until the lock is
    /// acquired or `timeout` elapses.
    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>> {
        let path = validate_path(path)?;
        let path_owned = path.to_string();
        let deadline = tokio::time::Instant::now() + timeout;
        let call_count = self.lock_call_count.fetch_add(1, Ordering::Relaxed);
        let semaphore = {
            let mut map = self.writer_semaphores.lock().unwrap();
            let should_prune = call_count.is_multiple_of(100) || map.len() > 1000;
            match map.get(&path_owned).and_then(Weak::upgrade) {
                Some(semaphore) => {
                    if should_prune {
                        map.retain(|_, weak| weak.upgrade().is_some());
                    }
                    semaphore
                }
                None => {
                    if should_prune {
                        map.retain(|_, weak| weak.upgrade().is_some());
                    }
                    let semaphore = Arc::new(Semaphore::new(1));
                    map.insert(path_owned.clone(), Arc::downgrade(&semaphore));
                    semaphore
                }
            }
        };
        let semaphore_remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        let permit =
            match tokio::time::timeout(semaphore_remaining, semaphore.acquire_owned()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(e)) => return Err(IcefallDBError::Other(Box::new(e))),
                Err(_) => return Err(IcefallDBError::LockTimeout(path_owned.clone())),
            };

        let full = self.resolve_path(&path_owned)?;

        let mut file = tokio::task::spawn_blocking({
            let path_for_err = path_owned.clone();
            move || -> Result<std::fs::File> {
                if let Some(parent) = full.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&full)
                    .map_err(|e| map_io_err(e, &path_for_err))?;
                Ok(file)
            }
        })
        .await
        .map_err(|e| IcefallDBError::Other(Box::new(e)))??;

        loop {
            let attempt_file = file.try_clone().map_err(IcefallDBError::Io)?;
            let lock_result = tokio::task::spawn_blocking(move || -> Result<std::fs::File> {
                fs2::FileExt::try_lock_exclusive(&attempt_file)?;
                Ok(attempt_file)
            })
            .await;

            match lock_result {
                Ok(Ok(locked_file)) => {
                    file = locked_file;
                    break;
                }
                Ok(Err(IcefallDBError::Io(e))) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    if remaining.is_zero() {
                        return Err(IcefallDBError::LockTimeout(path_owned));
                    }
                    let sleep_for = Duration::from_millis(50).min(remaining);
                    if sleep_for.is_zero() {
                        return Err(IcefallDBError::LockTimeout(path_owned));
                    }
                    tokio::time::sleep(sleep_for).await;
                }
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(IcefallDBError::Other(Box::new(e))),
            }
        }

        Ok(Box::new(LocalLockGuard {
            file,
            path: path_owned,
            _permit: permit,
        }))
    }

    async fn sync(&self, path: &str) -> Result<()> {
        let full = self.resolve_path(path)?;
        let metadata = tokio::fs::metadata(&full)
            .await
            .map_err(|e| map_io_err(e, path))?;

        if metadata.is_file() {
            let file = tokio::fs::File::open(&full)
                .await
                .map_err(|e| map_io_err(e, path))?;
            file.sync_all().await.map_err(|e| map_io_err(e, path))?;
            FSYNC_COUNT.fetch_add(1, Ordering::Relaxed);
        } else if metadata.is_dir() {
            let file = tokio::fs::OpenOptions::new()
                .read(true)
                .open(&full)
                .await
                .map_err(|e| map_io_err(e, path))?;
            file.sync_all().await.map_err(|e| map_io_err(e, path))?;
            FSYNC_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn sync_data(&self, path: &str) -> Result<()> {
        let full = self.resolve_path(path)?;
        let metadata = tokio::fs::metadata(&full)
            .await
            .map_err(|e| map_io_err(e, path))?;

        if metadata.is_file() {
            let file = tokio::fs::File::open(&full)
                .await
                .map_err(|e| map_io_err(e, path))?;
            file.sync_data().await.map_err(|e| map_io_err(e, path))?;
            FSYNC_COUNT.fetch_add(1, Ordering::Relaxed);
        } else if metadata.is_dir() {
            // fdatasync on a directory does not guarantee directory-entry
            // durability on all systems; fall back to a full sync.
            let file = tokio::fs::OpenOptions::new()
                .read(true)
                .open(&full)
                .await
                .map_err(|e| map_io_err(e, path))?;
            file.sync_all().await.map_err(|e| map_io_err(e, path))?;
            FSYNC_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn sync_root(&self) -> Result<()> {
        let file = tokio::fs::File::open(&self.root).await?;
        file.sync_all().await?;
        FSYNC_COUNT.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_lock_timeout_honored_for_flock() {
        let dir = tempfile::tempdir().unwrap();
        let storage1 = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let _guard = storage1
            .lock_exclusive("products/_write.lock", Duration::from_secs(10))
            .await
            .unwrap();

        let storage2 = LocalStorage::new(dir.path()).unwrap();
        let timeout = Duration::from_millis(100);
        let start = tokio::time::Instant::now();
        let result = storage2
            .lock_exclusive("products/_write.lock", timeout)
            .await;
        let elapsed = start.elapsed();

        assert!(
            matches!(result, Err(IcefallDBError::LockTimeout(_))),
            "expected LockTimeout, got {:?}",
            result.map(|_| ())
        );
        assert!(
            elapsed <= timeout + Duration::from_millis(50),
            "lock wait exceeded timeout margin: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_lock_timeout_honored_for_semaphore() {
        let dir = tempfile::tempdir().unwrap();
        let storage = Arc::new(LocalStorage::new(dir.path()).unwrap());
        let _guard = storage
            .lock_exclusive("products/_write.lock", Duration::from_secs(10))
            .await
            .unwrap();

        let storage2 = Arc::clone(&storage);
        let timeout = Duration::from_millis(100);
        let start = tokio::time::Instant::now();
        let result = storage2
            .lock_exclusive("products/_write.lock", timeout)
            .await;
        let elapsed = start.elapsed();

        assert!(
            matches!(result, Err(IcefallDBError::LockTimeout(_))),
            "expected LockTimeout, got {:?}",
            result.map(|_| ())
        );
        assert!(
            elapsed <= timeout + Duration::from_millis(50),
            "lock wait exceeded timeout margin: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_read_range() {
        let dir = tempfile::tempdir().unwrap();
        let storage = LocalStorage::new(dir.path()).unwrap();
        storage.write("data.bin", b"hello world").await.unwrap();

        let bytes = storage.read_range("data.bin", 0, 5).await.unwrap();
        assert_eq!(bytes, b"hello");

        let bytes = storage.read_range("data.bin", 6, 5).await.unwrap();
        assert_eq!(bytes, b"world");

        assert!(storage.read_range("data.bin", 6, 10).await.is_err());
    }
}
