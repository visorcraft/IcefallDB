#![cfg(feature = "s3")]

use async_trait::async_trait;
use object_store::path::Path;
use object_store::{Error as ObjectStoreError, GetOptions, GetRange, ObjectStore, ObjectStoreExt};
use std::sync::Arc;
use std::time::Duration;

use super::{path::validate_path, LockGuard, Storage};
use crate::{IcefallDBError, Result};

/// Read-only S3-compatible storage backend.
///
/// Wraps a generic [`ObjectStore`] so it can be tested with
/// [`object_store::memory::InMemory`] and used in production with
/// [`object_store::aws::AmazonS3`].
#[derive(Debug, Clone)]
pub struct S3Storage {
    store: Arc<dyn ObjectStore>,
    prefix: String,
}

impl S3Storage {
    /// Create a new `S3Storage` backed by `store`.
    ///
    /// `prefix` is the S3 key prefix that represents the storage root.
    /// All `Storage` paths are interpreted relative to this prefix.
    pub fn new(store: Arc<dyn ObjectStore>, prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        let prefix = prefix.trim_end_matches('/').to_string();
        Self { store, prefix }
    }

    /// Build an `S3Storage` from environment variables.
    ///
    /// Required:
    /// - `ICEFALLDB_S3_BUCKET`
    ///
    /// Optional:
    /// - `ICEFALLDB_S3_REGION` (default: `us-east-1`)
    /// - `ICEFALLDB_S3_ENDPOINT`
    /// - `ICEFALLDB_S3_ACCESS_KEY_ID`
    /// - `ICEFALLDB_S3_SECRET_ACCESS_KEY`
    /// - `ICEFALLDB_S3_ALLOW_HTTP` (default: `false`; set to `true` to override)
    pub fn from_env(prefix: impl Into<String>) -> Result<Self> {
        use object_store::aws::AmazonS3Builder;

        let bucket = std::env::var("ICEFALLDB_S3_BUCKET")
            .map_err(|_| IcefallDBError::Other("ICEFALLDB_S3_BUCKET not set".into()))?;
        let region = std::env::var("ICEFALLDB_S3_REGION").unwrap_or_else(|_| "us-east-1".into());
        let endpoint = std::env::var("ICEFALLDB_S3_ENDPOINT").ok();
        let access_key = std::env::var("ICEFALLDB_S3_ACCESS_KEY_ID").ok();
        let secret_key = std::env::var("ICEFALLDB_S3_SECRET_ACCESS_KEY").ok();
        let allow_http_env = std::env::var("ICEFALLDB_S3_ALLOW_HTTP").ok();

        let mut builder = AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_region(region);

        if let Some(endpoint) = endpoint {
            let allow_http = endpoint.to_ascii_lowercase().starts_with("http://")
                || allow_http_env.as_deref() == Some("true");
            builder = builder.with_endpoint(endpoint).with_allow_http(allow_http);
        }

        match (access_key, secret_key) {
            (Some(key), Some(secret)) => {
                builder = builder
                    .with_access_key_id(key)
                    .with_secret_access_key(secret);
            }
            (None, None) => {}
            _ => {
                return Err(IcefallDBError::Other(
                    "both ICEFALLDB_S3_ACCESS_KEY_ID and ICEFALLDB_S3_SECRET_ACCESS_KEY must be set"
                        .into(),
                ));
            }
        }

        let store = builder
            .build()
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;

        Ok(Self::new(Arc::new(store), prefix))
    }

    fn s3_path(&self, path: &str) -> Result<Path> {
        let cleaned = validate_path(path)?;
        if self.prefix.is_empty() {
            Ok(Path::from(cleaned))
        } else {
            Ok(Path::from(format!("{}/{}", self.prefix, cleaned)))
        }
    }

    fn storage_path(&self, s3_key: &str) -> Option<String> {
        if self.prefix.is_empty() {
            Some(s3_key.to_string())
        } else {
            s3_key
                .strip_prefix(&self.prefix)
                .and_then(|s| s.strip_prefix('/'))
                .map(|s| s.to_string())
        }
    }
}

fn map_error(path: &str, err: ObjectStoreError) -> IcefallDBError {
    match err {
        ObjectStoreError::NotFound { .. } => IcefallDBError::NotFound(path.to_string()),
        _ => IcefallDBError::Other(Box::new(err)),
    }
}

#[async_trait]
impl Storage for S3Storage {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        let location = self.s3_path(path)?;
        let result = self
            .store
            .get(&location)
            .await
            .map_err(|e| map_error(path, e))?;
        let bytes = result.bytes().await.map_err(|e| map_error(path, e))?;
        Ok(bytes.to_vec())
    }

    async fn size(&self, path: &str) -> Result<u64> {
        let location = self.s3_path(path)?;
        let meta = self
            .store
            .head(&location)
            .await
            .map_err(|e| map_error(path, e))?;
        Ok(meta.size as u64)
    }

    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        let location = self.s3_path(path)?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| IcefallDBError::RangeReadError {
                path: path.to_string(),
                reason: "range length overflow".into(),
            })?;
        let opts = GetOptions {
            range: Some(GetRange::Bounded(offset..end)),
            ..Default::default()
        };
        let result = self
            .store
            .get_opts(&location, opts)
            .await
            .map_err(|e| map_error(path, e))?;
        let bytes = result.bytes().await.map_err(|e| map_error(path, e))?;
        Ok(bytes.to_vec())
    }

    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let location: Option<Path> = if prefix.is_empty() {
            if self.prefix.is_empty() {
                None
            } else {
                Some(Path::from(self.prefix.as_str()))
            }
        } else {
            Some(self.s3_path(prefix)?)
        };

        let list_result = self
            .store
            .list_with_delimiter(location.as_ref())
            .await
            .map_err(|e| map_error(prefix, e))?;

        if list_result.objects.is_empty() && list_result.common_prefixes.is_empty() {
            return Err(IcefallDBError::NotFound(prefix.to_string()));
        }

        let mut entries = Vec::new();
        for obj in list_result.objects {
            if let Some(relative) = self.storage_path(obj.location.as_ref()) {
                entries.push(relative);
            }
        }
        for prefix_path in list_result.common_prefixes {
            let key = prefix_path.to_string();
            let key = key.trim_end_matches('/');
            if let Some(relative) = self.storage_path(key) {
                entries.push(relative);
            }
        }

        entries.sort();
        Ok(entries)
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let location = self.s3_path(path)?;
        match self.store.head(&location).await {
            Ok(_) => Ok(true),
            Err(ObjectStoreError::NotFound { .. }) => Ok(false),
            Err(e) => Err(map_error(path, e)),
        }
    }

    async fn write(&self, _path: &str, _data: &[u8]) -> Result<()> {
        Err(IcefallDBError::Other(
            "S3 storage is read-only in v1".into(),
        ))
    }

    async fn delete(&self, _path: &str) -> Result<()> {
        Err(IcefallDBError::Other(
            "S3 storage is read-only in v1".into(),
        ))
    }

    async fn rename(&self, _from: &str, _to: &str) -> Result<()> {
        Err(IcefallDBError::Other(
            "S3 storage is read-only in v1".into(),
        ))
    }

    async fn lock_exclusive(&self, _path: &str, _timeout: Duration) -> Result<Box<dyn LockGuard>> {
        Err(IcefallDBError::Other(
            "S3 storage is read-only in v1".into(),
        ))
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
    use object_store::memory::InMemory;

    #[tokio::test]
    async fn test_list_empty_prefix() {
        let store = Arc::new(InMemory::new());
        let storage = S3Storage::new(store.clone(), "mydb");

        store
            .put(&Path::from("mydb/products/_manifest.json"), "{}".into())
            .await
            .unwrap();
        store
            .put(&Path::from("mydb/_schemas/000001.json"), "{}".into())
            .await
            .unwrap();

        let entries = storage.list("").await.unwrap();
        assert_eq!(entries, vec!["_schemas", "products"]);
    }

    #[tokio::test]
    async fn test_read_range() {
        let store = Arc::new(InMemory::new());
        let storage = S3Storage::new(store.clone(), "mydb");

        store
            .put(&Path::from("mydb/data.bin"), "hello world".into())
            .await
            .unwrap();

        let bytes = storage.read_range("data.bin", 0, 5).await.unwrap();
        assert_eq!(bytes, b"hello");

        let bytes = storage.read_range("data.bin", 6, 5).await.unwrap();
        assert_eq!(bytes, b"world");
    }
}
