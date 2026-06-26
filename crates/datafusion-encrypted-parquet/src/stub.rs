//! Stub implementation used when the `encryption` feature is off.

use std::sync::Arc;

use datafusion::execution::context::SessionContext;
use datafusion_common::{DataFusionError, Result};

/// Factory id used when registering on the `RuntimeEnv`.
pub const FACTORY_ID: &str = "encrypted-parquet";

/// Source of AES keys. Stub when the `encryption` feature is off.
pub trait KeySource: Send + Sync {
    fn get(&self, _kid: &str) -> Result<Vec<u8>>;
}

/// Returns an error: callers must enable the `encryption` feature.
pub fn register_encryption_factory(_ctx: &SessionContext, _keys: Arc<dyn KeySource>) -> Result<()> {
    Err(DataFusionError::Configuration(
        "datafusion-encrypted-parquet's `encryption` feature is not enabled; \
         decryption is unavailable"
            .into(),
    ))
}
