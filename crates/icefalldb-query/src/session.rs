//! DataFusion session factory tuned for IcefallDB.
//!
//! The produced session disables Parquet view-type forcing so that Arrow schemas
//! line up exactly with the types stored by `icefalldb-core`, and registers the
//! IcefallDB physical optimizer rules.

use datafusion::common::config::ConfigExtension;
use datafusion::common::extensions_options;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::SessionContext;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::execution::SessionState;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Process-global count of full IcefallDB query `SessionState`s built. Used to
/// prove the write path never constructs the query stack: a
/// writer-only command (`icefalldb insert`) must leave this at 0.
static SESSION_BUILD_COUNT: AtomicU64 = AtomicU64::new(0);

/// Number of IcefallDB query sessions built since process start.
pub fn session_build_count() -> u64 {
    SESSION_BUILD_COUNT.load(Ordering::Relaxed)
}

extensions_options! {
    /// IcefallDB query-engine configuration options.
    pub struct IcefallDBConfig {
        /// Enable the metadata aggregate physical optimizer rule.
        pub metadata_aggregate: bool, default = true

        /// Enable the tiny-build-side join specialization rule.
        pub tiny_build_join: bool, default = true

        /// Enable dynamic filter pushdown from build side to probe side scan.
        pub dynamic_filter_pushdown: bool, default = true

        /// Skip the native Parquet row-filter (selective-decode path) and bulk
        /// decode + `FilterExec` instead when the query's projection is a subset
        /// of its filter columns (nothing to late-materialize). Faster on
        /// non-prunable filtered scans; statistics/page-index pruning is kept.
        pub native_bulk_decode: bool, default = true

        /// Enable the sorted group-by physical optimizer rule.
        pub sorted_group_by: bool, default = true

        /// Enable the low-cardinality group-by physical optimizer rule.
        pub low_cardinality_group_by: bool, default = true

        /// Maximum estimated group count for the `LowCardinalityGroupBy` rule to fire.
        pub low_cardinality_group_by_threshold: usize, default = 4096

        /// Enable the lookup join physical optimizer rule.
        pub lookup_join: bool, default = false

        /// Maximum build-side row count for the `TinyBuildJoin` rule to fire.
        pub tiny_build_join_threshold: usize, default = 4096

        /// Maximum build-side row count for the `LookupJoin` rule to fire.
        pub lookup_join_threshold: usize, default = 4096
    }
}

impl ConfigExtension for IcefallDBConfig {
    const PREFIX: &'static str = "icefalldb";
}

/// Build a DataFusion `SessionConfig` with IcefallDB defaults.
///
/// `target_partitions` controls the default partition count for scans; `batch_size`
/// sets the record-batch size used by most operators.
pub fn icefalldb_session_config(target_partitions: usize, batch_size: usize) -> SessionConfig {
    let mut cfg = SessionConfig::new()
        .with_target_partitions(target_partitions)
        .with_batch_size(batch_size)
        .with_option_extension(IcefallDBConfig::default());
    // Disable view-type forcing so schemas match icefalldb-core exactly.
    cfg.options_mut().execution.parquet.schema_force_view_types = false;
    // Disable DataFusion's automatic file-scan repartitioning.  IcefallDB builds
    // explicit byte-range file groups in `build_native_parquet_exec` based on
    // file size, so automatic splitting would undo that sizing and force tiny
    // files (e.g. 10 MB event tables) onto 16 partitions.
    cfg.options_mut().optimizer.repartition_file_scans = false;
    cfg
}

/// Build a DataFusion `SessionState` from [`icefalldb_session_config`] with IcefallDB
/// physical optimizer rules registered.
pub fn icefalldb_session_state(target_partitions: usize, batch_size: usize) -> SessionState {
    icefalldb_session_state_from_config(icefalldb_session_config(target_partitions, batch_size))
}

/// Build a DataFusion `SessionState` from an existing [`SessionConfig`] with
/// IcefallDB physical optimizer rules registered.
pub fn icefalldb_session_state_from_config(config: SessionConfig) -> SessionState {
    SESSION_BUILD_COUNT.fetch_add(1, Ordering::Relaxed);
    let state = SessionStateBuilder::new_with_default_features()
        .with_config(config)
        .with_optimizer_rule(Arc::new(crate::rules::SimplifyCastPredicates::new()))
        .build();
    crate::rules::register_icefalldb_rules(state)
}

/// Build a DataFusion `SessionContext` from [`icefalldb_session_state`].
pub fn icefalldb_session(target_partitions: usize, batch_size: usize) -> SessionContext {
    SessionContext::new_with_state(icefalldb_session_state(target_partitions, batch_size))
}

#[cfg(feature = "encryption")]
pub use encryption_session::{icefalldb_encrypted_session, icefalldb_encrypted_session_state};

#[cfg(feature = "encryption")]
mod encryption_session {
    use std::sync::Arc;

    use datafusion::execution::context::SessionContext;
    use datafusion::execution::session_state::SessionStateBuilder;
    use datafusion::execution::SessionState;
    use icefalldb_core::encryption::provider::KeyProvider;

    use crate::encryption::IcefallDBEncryptionFactory;

    /// Build a `SessionState` with a [`IcefallDBEncryptionFactory`] registered
    /// on the `RuntimeEnv` under [`IcefallDBEncryptionFactory::FACTORY_ID`].
    /// Tables opt into encryption by setting
    /// `format.crypto.factory_id = "icefalldb"` in their session config.
    pub fn icefalldb_encrypted_session_state(
        target_partitions: usize,
        batch_size: usize,
        provider: Arc<dyn KeyProvider>,
    ) -> SessionState {
        let config = crate::session::icefalldb_session_config(target_partitions, batch_size);
        let state = SessionStateBuilder::new_with_default_features()
            .with_config(config)
            .with_optimizer_rule(Arc::new(crate::rules::SimplifyCastPredicates::new()))
            .build();
        let _previous = state.runtime_env().register_parquet_encryption_factory(
            IcefallDBEncryptionFactory::FACTORY_ID,
            Arc::new(IcefallDBEncryptionFactory::new(provider)),
        );
        crate::rules::register_icefalldb_rules(state)
    }

    /// Build a `SessionContext` with encryption support. Equivalent to
    /// [`icefalldb_session`] but with a registered encryption factory.
    pub fn icefalldb_encrypted_session(
        target_partitions: usize,
        batch_size: usize,
        provider: Arc<dyn KeyProvider>,
    ) -> SessionContext {
        SessionContext::new_with_state(icefalldb_encrypted_session_state(
            target_partitions,
            batch_size,
            provider,
        ))
    }
}
