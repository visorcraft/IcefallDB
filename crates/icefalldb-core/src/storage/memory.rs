use super::{path::validate_path, LockGuard, Storage};
use crate::{IcefallDBError, Result};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug)]
pub struct MemoryStorage {
    data: tokio::sync::Mutex<HashMap<String, Vec<u8>>>,
    writer_semaphores: Mutex<HashMap<String, Weak<Semaphore>>>,
    lock_call_count: AtomicUsize,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            data: tokio::sync::Mutex::new(HashMap::new()),
            writer_semaphores: Mutex::new(HashMap::new()),
            lock_call_count: AtomicUsize::new(0),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// In-process exclusive lock guard for [`MemoryStorage`].
///
/// Holds an owned permit from the per-path semaphore. The permit is released
/// when the guard is dropped, allowing the next waiter on the same path to
/// acquire the lock.
pub struct MemoryLockGuard {
    path: String,
    _permit: OwnedSemaphorePermit,
}

impl LockGuard for MemoryLockGuard {
    fn path(&self) -> &str {
        &self.path
    }
}

#[async_trait]
impl Storage for MemoryStorage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        let path = validate_path(path)?;
        let data = self.data.lock().await;
        data.get(&path)
            .cloned()
            .ok_or_else(|| IcefallDBError::NotFound(path))
    }

    async fn size(&self, path: &str) -> Result<u64> {
        let path = validate_path(path)?;
        let data = self.data.lock().await;
        data.get(&path)
            .map(|bytes| bytes.len() as u64)
            .ok_or_else(|| IcefallDBError::NotFound(path))
    }

    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let path = validate_path(path)?;
        let data = self.data.lock().await;
        let bytes = data
            .get(&path)
            .ok_or_else(|| IcefallDBError::NotFound(path.clone()))?;
        let start = offset as usize;
        let end =
            start
                .checked_add(len as usize)
                .ok_or_else(|| IcefallDBError::RangeReadError {
                    path: path.clone(),
                    reason: "range length overflow".into(),
                })?;
        if end > bytes.len() {
            return Err(IcefallDBError::RangeReadError {
                path,
                reason: format!(
                    "range {}..{} exceeds object size {}",
                    start,
                    end,
                    bytes.len()
                ),
            });
        }
        Ok(bytes[start..end].to_vec())
    }

    async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        let path = validate_path(path)?;
        self.data.lock().await.insert(path, data.to_vec());
        Ok(())
    }

    async fn append(&self, path: &str, data: &[u8]) -> Result<()> {
        let path = validate_path(path)?;
        let mut map = self.data.lock().await;
        let entry = map.entry(path).or_default();
        entry.extend_from_slice(data);
        Ok(())
    }

    async fn delete(&self, path: &str) -> Result<()> {
        let path = validate_path(path)?;
        let removed = self.data.lock().await.remove(&path);
        if removed.is_none() {
            return Err(IcefallDBError::NotFound(path));
        }
        Ok(())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let normalized = if prefix.is_empty() {
            String::new()
        } else {
            let cleaned = validate_path(prefix)?;
            if cleaned.ends_with('/') {
                cleaned
            } else {
                format!("{cleaned}/")
            }
        };

        let data = self.data.lock().await;
        let mut children = HashSet::new();

        for key in data.keys() {
            if let Some(rest) = key.strip_prefix(&normalized) {
                let child = if let Some((component, _)) = rest.split_once('/') {
                    format!("{normalized}{component}")
                } else {
                    format!("{normalized}{rest}")
                };
                children.insert(child);
            }
        }

        if children.is_empty() && !prefix.is_empty() {
            return Err(IcefallDBError::NotFound(prefix.to_string()));
        }

        let mut result: Vec<_> = children.into_iter().collect();
        result.sort();
        Ok(result)
    }

    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        let from = validate_path(from)?;
        let to = validate_path(to)?;
        let mut data = self.data.lock().await;
        let value = data
            .remove(&from)
            .ok_or_else(|| IcefallDBError::NotFound(from))?;
        data.insert(to, value);
        Ok(())
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let path = validate_path(path)?;
        Ok(self.data.lock().await.contains_key(&path))
    }

    /// Acquire an in-process exclusive advisory lock on the file at `path`.
    ///
    /// The returned guard holds an owned permit from a per-path semaphore,
    /// which serializes multiple writers from the same process. There is no
    /// cross-process exclusion because `MemoryStorage` is in-memory only.
    ///
    /// Acquisition honors `timeout`; if the permit cannot be obtained before the
    /// timeout elapses, a [`IcefallDBError::LockTimeout`] is returned.
    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>> {
        let path = validate_path(path)?;
        let call_count = self.lock_call_count.fetch_add(1, Ordering::Relaxed);
        let semaphore = {
            let mut map = self.writer_semaphores.lock().unwrap();
            let should_prune = call_count.is_multiple_of(100) || map.len() > 1000;
            match map.get(&path).and_then(Weak::upgrade) {
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
                    map.insert(path.to_string(), Arc::downgrade(&semaphore));
                    semaphore
                }
            }
        };
        let permit = match tokio::time::timeout(timeout, semaphore.acquire_owned()).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(e)) => return Err(IcefallDBError::Other(Box::new(e))),
            Err(_) => return Err(IcefallDBError::LockTimeout(path)),
        };
        Ok(Box::new(MemoryLockGuard {
            path: path.to_string(),
            _permit: permit,
        }))
    }

    async fn sync(&self, _path: &str) -> Result<()> {
        Ok(())
    }

    async fn sync_data(&self, _path: &str) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_range() {
        let storage = MemoryStorage::new();
        storage.write("data.bin", b"hello world").await.unwrap();

        let bytes = storage.read_range("data.bin", 0, 5).await.unwrap();
        assert_eq!(bytes, b"hello");

        let bytes = storage.read_range("data.bin", 6, 5).await.unwrap();
        assert_eq!(bytes, b"world");

        assert!(matches!(
            storage.read_range("data.bin", 6, 10).await,
            Err(IcefallDBError::RangeReadError { .. })
        ));
    }
}
