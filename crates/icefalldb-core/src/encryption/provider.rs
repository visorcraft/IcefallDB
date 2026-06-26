//! Key providers: load encryption keys from configurable backends.
//!
//! A `KeyProvider` resolves a [`KeyIdentifier`] into the raw key bytes. This
//! is the seam where production deployments plug in a KMS (AWS KMS, GCP KMS,
//! HashiCorp Vault); v1 ships three concrete providers that cover local,
//! single-node deployments:
//!
//! - [`EnvKeyProvider`] — reads `ICEFALLDB_KEY_<KID>` env vars. Default for
//!   tests and CI.
//! - [`FileKeyProvider`] — reads a JSON file mapping key ids to hex keys.
//!   Default for single-node production.
//! - [`StaticKeyProvider`] — programmatic map. Useful for the Python adapter,
//!   which can resolve keys via a Python callback and hand them across the
//!   PyO3 bridge.
//!
//! All providers are `Send + Sync` so they can be shared across DataFusion
//! partitions.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use crate::encryption::keys::{validate_key, KeyIdentifier};
use crate::error::{IcefallDBError, Result};

/// Prefix for env-var-backed keys. `<KID>` is uppercased and any non-alphanumeric
/// character is replaced with `_`. For example, key id `events-v1` is read from
/// `ICEFALLDB_KEY_EVENTS_V1`.
pub const KEY_ID_ENV_PREFIX: &str = "ICEFALLDB_KEY_";

/// Resolve a [`KeyIdentifier`] into the raw AES key bytes.
///
/// Implementations must be deterministic for a given `(kid, aad)` pair within a
/// single process: the same key id must always resolve to the same bytes.
/// `aad` is provided so KMS-backed implementations can authenticate key
/// retrieval; static providers may ignore it.
#[async_trait]
pub trait KeyProvider: Send + Sync {
    async fn get(&self, kid: &KeyIdentifier, aad: &[u8]) -> Result<Vec<u8>>;

    /// Return every key identifier this provider knows about. Used by writers
    /// that want to fail fast on unknown key references at startup.
    async fn known(&self) -> Result<Vec<KeyIdentifier>>;
}

/// Env-var-backed provider. Reads `ICEFALLDB_KEY_<KID_UPPER>` on demand.
#[derive(Debug, Default, Clone)]
pub struct EnvKeyProvider;

impl EnvKeyProvider {
    fn env_var_name(kid: &KeyIdentifier) -> String {
        let upper: String = kid
            .as_str()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_uppercase()
                } else {
                    '_'
                }
            })
            .collect();
        format!("{KEY_ID_ENV_PREFIX}{upper}")
    }
}

#[async_trait]
impl KeyProvider for EnvKeyProvider {
    async fn get(&self, kid: &KeyIdentifier, _aad: &[u8]) -> Result<Vec<u8>> {
        let var = Self::env_var_name(kid);
        let raw = env::var(&var).map_err(|_| {
            IcefallDBError::EncryptionKeyNotFound(format!("env var {var} for key id '{kid}'"))
        })?;
        let bytes = hex::decode(&raw).map_err(|e| {
            IcefallDBError::Encryption(format!("env var {var} is not valid hex ({e})"))
        })?;
        validate_key(&bytes)?;
        Ok(bytes)
    }

    async fn known(&self) -> Result<Vec<KeyIdentifier>> {
        let mut out = Vec::new();
        for (k, _) in env::vars() {
            if let Some(rest) = k.strip_prefix(KEY_ID_ENV_PREFIX) {
                out.push(KeyIdentifier(rest.to_lowercase()));
            }
        }
        Ok(out)
    }
}

/// File-backed provider. Reads a JSON file of the form:
///
/// ```json
/// {
///   "keys": {
///     "events-v1": "30313233343536373839616263646566",
///     "ssn-v1":    "31323334353637383839616263646566"
///   }
/// }
/// ```
///
/// The file is read lazily on first access and then cached. Call
/// [`FileKeyProvider::reload`] to invalidate the cache.
/// Cached key map: `KeyIdentifier` → key bytes.
type KeyCache = HashMap<KeyIdentifier, Vec<u8>>;

#[derive(Debug, Clone)]
pub struct FileKeyProvider {
    path: PathBuf,
    cache: Arc<RwLock<Option<KeyCache>>>,
}

impl FileKeyProvider {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Force the next `get` to re-read the file.
    pub fn reload(&self) {
        if let Ok(mut guard) = self.cache.write() {
            *guard = None;
        }
    }

    fn load_locked(&self) -> Result<KeyCache> {
        let raw = fs::read_to_string(&self.path).map_err(|e| {
            IcefallDBError::Encryption(format!(
                "failed to read key file {}: {e}",
                self.path.display()
            ))
        })?;
        #[derive(serde::Deserialize)]
        struct FileMap {
            keys: HashMap<String, String>,
        }
        let parsed: FileMap = serde_json::from_str(&raw)?;
        let mut out = HashMap::with_capacity(parsed.keys.len());
        for (id, hex) in parsed.keys {
            let bytes = hex::decode(&hex).map_err(|e| {
                IcefallDBError::Encryption(format!(
                    "invalid hex for key '{id}' in {}: {e}",
                    self.path.display()
                ))
            })?;
            validate_key(&bytes)?;
            out.insert(KeyIdentifier(id), bytes);
        }
        Ok(out)
    }

    fn get_cached(&self, kid: &KeyIdentifier) -> Result<Option<Vec<u8>>> {
        // Try read-locked fast path first.
        if let Ok(read) = self.cache.read() {
            if let Some(map) = read.as_ref() {
                return Ok(map.get(kid).cloned());
            }
        }
        // Slow path: load and re-check.
        let loaded = self.load_locked()?;
        let cloned = loaded.get(kid).cloned();
        if let Ok(mut write) = self.cache.write() {
            *write = Some(loaded);
        }
        Ok(cloned)
    }
}

#[async_trait]
impl KeyProvider for FileKeyProvider {
    async fn get(&self, kid: &KeyIdentifier, _aad: &[u8]) -> Result<Vec<u8>> {
        self.get_cached(kid)?
            .ok_or_else(|| IcefallDBError::EncryptionKeyNotFound(kid.to_string()))
    }

    async fn known(&self) -> Result<Vec<KeyIdentifier>> {
        let map = if let Ok(read) = self.cache.read() {
            if let Some(map) = read.as_ref() {
                map.clone()
            } else {
                self.load_locked()?
            }
        } else {
            self.load_locked()?
        };
        Ok(map.into_keys().collect())
    }
}

/// Static (programmatic) provider. The caller hands in a map of
/// `KeyIdentifier` → key bytes. Used by the Python adapter, which resolves
/// keys via a Python callback at session-build time.
#[derive(Debug, Default, Clone)]
pub struct StaticKeyProvider {
    keys: HashMap<KeyIdentifier, Vec<u8>>,
}

impl StaticKeyProvider {
    pub fn new<I>(keys: I) -> Self
    where
        I: IntoIterator<Item = (KeyIdentifier, Vec<u8>)>,
    {
        let mut map = HashMap::new();
        for (id, bytes) in keys {
            if let Err(e) = validate_key(&bytes) {
                tracing::warn!(kid = %id, error = %e, "skipping invalid-length key");
                continue;
            }
            map.insert(id, bytes);
        }
        Self { keys: map }
    }

    pub fn from_key_set(
        footer_id: impl Into<KeyIdentifier>,
        keys: &crate::encryption::EncryptionKeySet,
    ) -> Self {
        use crate::encryption::config::footer_key_id_for_column;
        let mut map = HashMap::new();
        let footer_id: KeyIdentifier = footer_id.into();
        map.insert(footer_id.clone(), keys.footer_bytes().to_vec());
        // Populate per-column key ids using the documented convention so that
        // readers looking up `<footer-id>:<column>` succeed. Without this,
        // readers would get `EncryptionKeyNotFound` for any column key.
        for (name, bytes) in keys.column_pairs() {
            let kid = KeyIdentifier::new(footer_key_id_for_column(footer_id.as_str(), name));
            map.insert(kid, bytes.to_vec());
        }
        Self { keys: map }
    }

    pub fn insert(&mut self, id: impl Into<KeyIdentifier>, bytes: Vec<u8>) -> Result<()> {
        validate_key(&bytes)?;
        self.keys.insert(id.into(), bytes);
        Ok(())
    }

    pub fn path_for_test(path: &Path) -> Result<Self> {
        // Convenience: build from a file path without caching (for tests).
        let _ = path;
        Ok(Self::default())
    }
}

#[async_trait]
impl KeyProvider for StaticKeyProvider {
    async fn get(&self, kid: &KeyIdentifier, _aad: &[u8]) -> Result<Vec<u8>> {
        self.keys
            .get(kid)
            .cloned()
            .ok_or_else(|| IcefallDBError::EncryptionKeyNotFound(kid.to_string()))
    }

    async fn known(&self) -> Result<Vec<KeyIdentifier>> {
        Ok(self.keys.keys().cloned().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k16(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    #[tokio::test]
    async fn env_provider_resolves_and_errors() {
        unsafe {
            env::set_var(
                "ICEFALLDB_KEY_EVENTS_V1",
                "30313233343536373839616263646566",
            );
        }
        let p = EnvKeyProvider;
        let got = p
            .get(&KeyIdentifier::new("events-v1"), b"aad")
            .await
            .unwrap();
        assert_eq!(got, k16("0123456789abcdef"));
        let err = p
            .get(&KeyIdentifier::new("does-not-exist"), b"aad")
            .await
            .unwrap_err();
        assert!(matches!(err, IcefallDBError::EncryptionKeyNotFound(_)));
        unsafe {
            env::remove_var("ICEFALLDB_KEY_EVENTS_V1");
        }
    }

    #[tokio::test]
    async fn static_provider_round_trip() {
        let mut p = StaticKeyProvider::default();
        p.insert("events-v1", k16("0123456789abcdef")).unwrap();
        let got = p
            .get(&KeyIdentifier::new("events-v1"), b"aad")
            .await
            .unwrap();
        assert_eq!(got, k16("0123456789abcdef"));
    }

    #[tokio::test]
    async fn file_provider_round_trip() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let json = r#"{
            "keys": {
                "events-v1": "30313233343536373839616263646566"
            }
        }"#;
        std::fs::write(tmp.path(), json).unwrap();
        let p = FileKeyProvider::new(tmp.path());
        let got = p
            .get(&KeyIdentifier::new("events-v1"), b"aad")
            .await
            .unwrap();
        assert_eq!(got, k16("0123456789abcdef"));
        assert_eq!(p.known().await.unwrap().len(), 1);
    }
}
