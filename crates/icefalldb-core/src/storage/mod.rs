use crate::Result;
use async_trait::async_trait;
use std::any::Any;
use std::time::Duration;

pub mod local;
pub mod memory;
pub mod path;

#[cfg(feature = "s3")]
pub mod s3;

/// A guard returned by [`Storage::lock_exclusive`].
///
/// The lock is released when the guard is dropped.
pub trait LockGuard: Send {
    /// Returns the locked path, relative to the storage root.
    fn path(&self) -> &str;
}

/// Abstract storage backend for IcefallDB.
///
/// All paths are relative to the storage root. Implementations must reject
/// absolute paths and path traversal attempts (`..`).
#[async_trait]
pub trait Storage: Send + Sync {
    /// Return this storage implementation as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Return the local filesystem root for storage implementations that are
    /// backed by a directory on disk.  Non-local implementations return `None`.
    fn local_root(&self) -> Option<&std::path::Path> {
        None
    }

    /// Read the full contents of the object at `path`.
    async fn read(&self, path: &str) -> Result<Vec<u8>>;

    /// Return the size in bytes of the object at `path`.
    async fn size(&self, path: &str) -> Result<u64>;

    /// Read a contiguous byte range from the object at `path`.
    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>>;

    /// Write `data` to `path`, creating any missing parent directories.
    async fn write(&self, path: &str, data: &[u8]) -> Result<()>;

    /// Delete the object at `path`.
    async fn delete(&self, path: &str) -> Result<()>;

    /// Rename the object at `from` to `to`, creating any missing parent
    /// directories under `to`.
    async fn rename(&self, from: &str, to: &str) -> Result<()>;

    /// List the direct children of the directory identified by `prefix`.
    ///
    /// `prefix` is treated as a directory path. The returned strings are
    /// paths relative to the storage root for each direct child (file or
    /// subdirectory). Results are sorted lexicographically.
    ///
    /// Returns `NotFound` if `prefix` does not exist as a directory.
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Return `true` if an object exists at `path`.
    async fn exists(&self, path: &str) -> Result<bool>;

    /// Acquire an exclusive advisory lock on the file at `path`.
    ///
    /// The file is created if it does not already exist. The lock is held
    /// until the returned [`LockGuard`] is dropped.
    ///
    /// The returned guard provides in-process exclusion across all writers
    /// created from the same storage instance. For [`LocalStorage`], it also
    /// wraps a POSIX `flock()` via `fs2::try_lock_exclusive`, which provides
    /// cross-process exclusion as well.
    ///
    /// Implementations must retry acquisition until the lock is obtained or
    /// `timeout` elapses. If the timeout is reached, a [`IcefallDBError::LockTimeout`]
    /// must be returned.
    ///
    /// [`LocalStorage`]: crate::storage::local::LocalStorage
    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>>;

    /// Synchronize the file or directory at `path` to durable storage.
    async fn sync(&self, path: &str) -> Result<()>;

    /// Synchronize only the file data (not metadata) at `path` to durable storage.
    ///
    /// This is cheaper than [`sync`] and is appropriate for regular files whose
    /// metadata timestamps do not need to be durable. The default implementation
    /// forwards to [`sync`].
    async fn sync_data(&self, path: &str) -> Result<()> {
        self.sync(path).await
    }

    /// Synchronize the storage root directory to durable storage.
    ///
    /// This is used after atomic metadata renames so the directory entry is
    /// durable. The default implementation is a no-op; local storage overrides
    /// this with an `fsync` on the root directory.
    async fn sync_root(&self) -> Result<()> {
        // Default: no-op. Local storage overrides this.
        Ok(())
    }

    /// Append `data` to the object at `path`, creating it if it does not exist.
    async fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        let mut existing = match self.read(path).await {
            Ok(data) => data,
            Err(e) if crate::is_not_found(&e) => Vec::new(),
            Err(e) => return Err(e),
        };
        existing.extend_from_slice(data);
        self.write(path, &existing).await
    }
}
