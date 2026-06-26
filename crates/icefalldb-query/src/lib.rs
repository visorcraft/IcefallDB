pub mod bench;
pub mod catalog;
pub mod coalesce;
#[cfg(feature = "encryption")]
pub mod encryption;
pub mod error;
pub mod execution;
pub mod index_selector;
pub(crate) mod metadata_cache;
pub mod mutate;
pub mod parquet_exec;
pub mod predicate;
pub mod provider;
pub mod result_cache;
pub mod rules;
pub mod scalar_codec;
pub mod scan;
pub mod session;
pub mod stats;

pub use catalog::IcefallDBCatalog;
pub use error::{QueryError, Result};
pub use execution::{AggType, LookupJoinExec, StreamingGroupByExec};
pub use icefalldb_core::MatchLoc;
pub use mutate::{
    execute_sql, execute_sql_batch, locate_matches, mutation_target_table, require_unique_key_index,
};
pub use provider::{IcefallDBTableProvider, ProviderConfig};
pub use rules::{MetadataAggregate, SimplifyCastPredicates};
pub use session::{
    icefalldb_session, icefalldb_session_config, icefalldb_session_state,
    icefalldb_session_state_from_config, session_build_count, IcefallDBConfig,
};

#[cfg(feature = "encryption")]
pub use session::{icefalldb_encrypted_session, icefalldb_encrypted_session_state};
