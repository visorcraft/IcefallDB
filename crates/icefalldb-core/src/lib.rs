pub mod agg_cache;
pub mod catalog;
pub mod check;
pub mod commit_delta;
pub mod compaction;
pub mod database_catalog;
pub mod deletion;
pub mod doctor;
#[cfg(feature = "encryption")]
pub mod encryption;
mod error;
pub mod gc;
#[cfg(feature = "iceberg")]
pub mod iceberg;
pub mod index;
pub mod meta_cache;
pub mod metadata;
pub mod mutation_wal;
mod reader;
pub mod recovery;
pub mod rowid;
pub mod rowindex;
pub mod schema_util;
pub mod storage;
pub mod tsv;
pub mod wal;
pub mod writer;

pub use agg_cache::{
    compute_agg_state_with_key, dv_density, merge_grouped, retract_grouped, should_recompute,
    AggScalar, AggStateCache, ColAgg, FragmentAggState, GroupedPartials, MAX_DECLARED_GROUPS,
    RECOMPUTE_DENSITY,
};
pub use check::{CheckIssue, CheckResult, Checker, Severity};
pub use commit_delta::{CommitDelta, CommitKind, FragmentDelta};
pub use compaction::{CompactionOptions, CompactionResult, Compactor};
pub use database_catalog::{DatabaseCatalog, DatabaseCatalogData, IndexEntry, TableEntry};
pub use deletion::DeletionVector;
pub use doctor::{
    verify_history, ActionKind, ChainBreak, DiagnosisIssue, DiagnosisKind, DiagnosisResult, Doctor,
    HistoryReport, RepairAction, RepairResult,
};
pub(crate) use error::is_not_found;
pub use error::{IcefallDBError, Result};
pub use gc::{GarbageCollector, GcResult};
pub use index::{
    append_tombstones, build_btree_index, list_index_names, load_index, load_index_by_ref,
    resolve_live_addresses, resolve_live_addresses_storage, BTreeIndex, IndexDefinition,
    IndexMaintainer, TombstoneDelta,
};
pub use reader::predicate_eval;
pub use reader::{
    build_scan_plan_at, list_snapshots, require_table_exists, Literal, PlannedRowGroup, Predicate,
    Reader, RowGroupStream, ScanPlan, SnapshotInfo,
};
pub use recovery::{apply_committed_transactions, recover, RecoveryState};
pub use rowid::{allocate_range, segment_ids, RowIdSegment};
pub use rowindex::{decode_idx, derive_base, encode_idx, AddrSegment, AddressMap, DecodedIdx};
pub use schema_util::{
    arrow_field_to_column, arrow_schema_to_icefalldb, arrow_type_to_icefalldb,
    DEFAULT_ROW_GROUP_TARGET_BYTES, DEFAULT_ROW_GROUP_TARGET_ROWS,
};
pub use tsv::{split_tsv_line, TsvDecoder, TsvEncoder};
pub use wal::{LogEntry, LogEntryBody, Wal, WalReader};
pub use writer::{InsertParquetOutcome, MatchLoc, Writer, WriterOptions, WriterOptionsFull};
