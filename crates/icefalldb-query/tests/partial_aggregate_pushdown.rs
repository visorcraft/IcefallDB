//! Correctness tests for partial-aggregate pushdown over range-filtered
//! SUM / COUNT.
//!
//! A range/equality-filtered `SUM`/`COUNT` over a clustered (time-series) table
//! composes the cached partials of FULLY-COVERED fragments (zero Parquet I/O)
//! and scans only the BOUNDARY fragments that straddle the filter edge.
//!
//! The absolute gate is correctness: every fast-path result must be BYTE-EQUAL
//! to the same query with the rule disabled (a full filtered scan).  These tests
//! assert byte-equality AND that only the boundary fragments' Parquet data is
//! read (via a counting storage wrapper), plus the documented fallback cases.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use async_trait::async_trait;
use datafusion::execution::context::SessionContext;
use icefalldb_core::agg_cache::AggStateCache;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::{arrow_schema_to_icefalldb, Result, Writer};
use icefalldb_query::{
    execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    IcefallDBTableProvider, ProviderConfig,
};

// ── Counting storage: tallies Parquet data-file reads ─────────────────────────

/// Wraps any `Storage` and counts reads (`read` + `read_range`) whose path ends
/// with `.parquet`.  The partial-aggregate fast path must read ONLY boundary
/// fragments, so this counter distinguishes "scanned N fragments" from "scanned
/// every overlapping fragment".
struct ParquetReadCounter {
    inner: Arc<dyn Storage>,
    parquet_reads: Arc<AtomicUsize>,
}

impl ParquetReadCounter {
    fn new(inner: Arc<dyn Storage>) -> (Self, Arc<AtomicUsize>) {
        let counter = Arc::new(AtomicUsize::new(0));
        (
            Self {
                inner,
                parquet_reads: Arc::clone(&counter),
            },
            counter,
        )
    }
}

#[async_trait]
impl Storage for ParquetReadCounter {
    fn as_any(&self) -> &dyn Any {
        self
    }
    async fn read(&self, path: &str) -> Result<Vec<u8>> {
        if path.ends_with(".parquet") {
            self.parquet_reads.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.read(path).await
    }
    async fn size(&self, path: &str) -> Result<u64> {
        self.inner.size(path).await
    }
    async fn read_range(&self, path: &str, offset: u64, len: u64) -> Result<Vec<u8>> {
        if path.ends_with(".parquet") {
            self.parquet_reads.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.read_range(path, offset, len).await
    }
    async fn write(&self, path: &str, data: &[u8]) -> Result<()> {
        self.inner.write(path, data).await
    }
    async fn delete(&self, path: &str) -> Result<()> {
        self.inner.delete(path).await
    }
    async fn rename(&self, from: &str, to: &str) -> Result<()> {
        self.inner.rename(from, to).await
    }
    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        self.inner.list(prefix).await
    }
    async fn exists(&self, path: &str) -> Result<bool> {
        self.inner.exists(path).await
    }
    async fn lock_exclusive(&self, path: &str, timeout: Duration) -> Result<Box<dyn LockGuard>> {
        self.inner.lock_exclusive(path, timeout).await
    }
    async fn sync(&self, path: &str) -> Result<()> {
        self.inner.sync(path).await
    }
    async fn sync_data(&self, path: &str) -> Result<()> {
        self.inner.sync_data(path).await
    }
    async fn sync_root(&self) -> Result<()> {
        self.inner.sync_root().await
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("day", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]))
}

/// Write a clustered table where each fragment spans a contiguous block of days
/// (`blocks[i]` lists the days held by fragment i, one row per day).  Each
/// fragment is a separate commit so it gets its own `.agg` sidecar with a tight
/// `day` min/max.  `v = day * 100`.  Returns `(counting_storage, counter, tmp)`.
async fn write_clustered_blocks(
    blocks: &[Vec<i64>],
    table: &str,
) -> (Arc<dyn Storage>, Arc<AtomicUsize>, tempfile::TempDir) {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let local: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let arrow = schema();
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow));
    // Large target so each block insert+commit is a single row group / fragment.
    mdb_schema.row_group_target_rows = 1_000_000;

    let mut writer = Writer::create(Arc::clone(&local), table, mdb_schema)
        .await
        .unwrap();
    for block in blocks {
        let day_col: Vec<i64> = block.clone();
        let v_col: Vec<i64> = block.iter().map(|d| d * 100).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow),
            vec![
                Arc::new(Int64Array::from(day_col)),
                Arc::new(Int64Array::from(v_col)),
            ],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();
    }

    let (counting, counter) = ParquetReadCounter::new(local);
    (Arc::new(counting), counter, tmp)
}

/// Day-clustered table with 4 fragments spanning 5 contiguous days each:
/// frag0=[10..14], frag1=[15..19], frag2=[20..24], frag3=[25..29].
fn four_blocks() -> Vec<Vec<i64>> {
    vec![
        (10..=14).collect(),
        (15..=19).collect(),
        (20..=24).collect(),
        (25..=29).collect(),
    ]
}

async fn make_ctx(storage: Arc<dyn Storage>, table: &str, rule_enabled: bool) -> SessionContext {
    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        table,
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 4,
            io_coalesce_window: 0,
            io_concurrency: 1,
            // Force the IcefallDBScanExec path (never the native parquet exec) so
            // the rule sees a IcefallDBScanExec with pushed physical filters.
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let mut cfg = icefalldb_session_config(4, 1024);
    if let Some(c) = cfg
        .options_mut()
        .extensions
        .get_mut::<icefalldb_query::IcefallDBConfig>()
    {
        c.metadata_aggregate = rule_enabled;
    }
    let state = icefalldb_session_state_from_config(cfg);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table(table, Arc::new(provider)).unwrap();
    ctx
}

async fn run_single_i64(ctx: &SessionContext, sql: &str) -> i64 {
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    assert_eq!(batches.len(), 1, "expected exactly one batch for `{sql}`");
    assert_eq!(
        batches[0].num_rows(),
        1,
        "expected exactly one row for `{sql}`"
    );
    let arr = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    // A NULL result is its own bug: `.value(0)` silently reads the backing
    // buffer (which a `null + const` BinaryExpr leaves set to `const`), so we
    // must reject nulls explicitly to keep the byte-equal assertions honest.
    assert!(!arr.is_null(0), "result of `{sql}` must not be NULL");
    arr.value(0)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Covered + boundary: `day BETWEEN 12 AND 27` over 4 fragments spanning
/// frag0=[10..14], frag1=[15..19], frag2=[20..24], frag3=[25..29].  frag0
/// straddles the lower edge (12) and frag3 straddles the upper edge (27) →
/// BOUNDARY; frag1,frag2 are fully inside → COVERED.  The fast-path result must
/// be byte-equal to the rule-disabled full filtered scan AND only the two
/// boundary fragments' Parquet must be read (the covered ones are composed from
/// `.agg` sidecars with zero I/O).
#[tokio::test]
async fn sum_covered_plus_boundary_byte_equal_and_reads_only_boundary() {
    let (storage, counter, _tmp) = write_clustered_blocks(&four_blocks(), "t_sum").await;
    let sql = "SELECT SUM(v) FROM t_sum WHERE day BETWEEN 12 AND 27";

    // Reference: rule disabled → full filtered scan over all 4 overlapping frags.
    let ctx_ref = make_ctx(Arc::clone(&storage), "t_sum", false).await;
    counter.store(0, Ordering::Relaxed);
    let expected = run_single_i64(&ctx_ref, sql).await;
    let ref_reads = counter.load(Ordering::Relaxed);

    // Fast path: rule enabled.
    counter.store(0, Ordering::Relaxed);
    let ctx_fast = make_ctx(Arc::clone(&storage), "t_sum", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    let fast_reads = counter.load(Ordering::Relaxed);

    assert_eq!(
        got, expected,
        "SUM fast path must be byte-equal to full filtered scan"
    );
    assert!(
        fast_reads > 0,
        "the two boundary fragments must actually be scanned (got {fast_reads})"
    );
    // Per-fragment sparse reads are a small constant; the fast path reads only 2
    // boundary fragments while the full scan reads all 4 overlapping fragments,
    // so the fast path must read strictly fewer parquet bytes/calls.
    assert!(
        fast_reads < ref_reads,
        "fast path must read fewer fragments than the full filtered scan \
         (fast={fast_reads}, full={ref_reads})"
    );
    assert!(
        fast_reads <= ref_reads / 2,
        "fast path must read only the 2 of 4 overlapping fragments \
         (fast={fast_reads}, full={ref_reads})"
    );
}

/// Same shape, COUNT(*).
#[tokio::test]
async fn count_star_covered_plus_boundary_byte_equal() {
    let (storage, counter, _tmp) = write_clustered_blocks(&four_blocks(), "t_cs").await;
    let sql = "SELECT COUNT(*) FROM t_cs WHERE day BETWEEN 12 AND 27";

    let ctx_ref = make_ctx(Arc::clone(&storage), "t_cs", false).await;
    counter.store(0, Ordering::Relaxed);
    let expected = run_single_i64(&ctx_ref, sql).await;
    let ref_reads = counter.load(Ordering::Relaxed);
    assert_eq!(expected, 16, "days 12..=27 inclusive = 16 rows");

    counter.store(0, Ordering::Relaxed);
    let ctx_fast = make_ctx(Arc::clone(&storage), "t_cs", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    let fast_reads = counter.load(Ordering::Relaxed);
    assert_eq!(
        got, expected,
        "COUNT(*) fast path must be byte-equal to full scan"
    );
    assert!(
        fast_reads <= ref_reads / 2,
        "COUNT(*) fast path must read only the 2 boundary fragments \
         (fast={fast_reads}, full={ref_reads})"
    );
}

/// Same shape, COUNT(v).
#[tokio::test]
async fn count_col_covered_plus_boundary_byte_equal() {
    let (storage, counter, _tmp) = write_clustered_blocks(&four_blocks(), "t_cc").await;
    let sql = "SELECT COUNT(v) FROM t_cc WHERE day BETWEEN 12 AND 27";

    let ctx_ref = make_ctx(Arc::clone(&storage), "t_cc", false).await;
    counter.store(0, Ordering::Relaxed);
    let expected = run_single_i64(&ctx_ref, sql).await;
    let ref_reads = counter.load(Ordering::Relaxed);

    counter.store(0, Ordering::Relaxed);
    let ctx_fast = make_ctx(Arc::clone(&storage), "t_cc", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    let fast_reads = counter.load(Ordering::Relaxed);
    assert_eq!(
        got, expected,
        "COUNT(v) fast path must be byte-equal to full scan"
    );
    assert!(
        fast_reads <= ref_reads / 2,
        "COUNT(v) fast path must read only the 2 boundary fragments \
         (fast={fast_reads}, full={ref_reads})"
    );
}

/// Exact cover (no boundary): `day BETWEEN 15 AND 24` covers whole fragments
/// frag1=[15..19] and frag2=[20..24] exactly (frag0/frag3 are disjoint).  Result
/// is the composed constant, byte-equal, with ZERO Parquet data reads.
#[tokio::test]
async fn exact_cover_no_boundary_zero_io() {
    let (storage, counter, _tmp) = write_clustered_blocks(&four_blocks(), "t_exact").await;
    let sql = "SELECT SUM(v) FROM t_exact WHERE day BETWEEN 15 AND 24";

    let ctx_ref = make_ctx(Arc::clone(&storage), "t_exact", false).await;
    let expected = run_single_i64(&ctx_ref, sql).await;

    counter.store(0, Ordering::Relaxed);
    let ctx_fast = make_ctx(Arc::clone(&storage), "t_exact", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    assert_eq!(got, expected, "exact-cover SUM must be byte-equal");

    let reads = counter.load(Ordering::Relaxed);
    assert_eq!(
        reads, 0,
        "exact-cover (no boundary) must read ZERO Parquet data (got {reads})"
    );
}

/// Boundary fragment with ZERO passing rows: a fragment whose `day` min/max
/// STRADDLES the filter edges (so it is classified BOUNDARY) but whose actual
/// rows all fall OUTSIDE the range, so the boundary `SUM(v)` is NULL.  The
/// covered contribution must NOT be discarded by `NULL + const` — the result
/// must be byte-equal to the full filtered scan (i.e. the covered sum).
#[tokio::test]
async fn boundary_with_no_passing_rows_keeps_covered_sum() {
    // frag0 holds days {10, 50} (min=10,max=50 → straddles both edges of
    // [12,27], but neither row passes); frag1=[15..19] fully covered.
    let blocks = vec![vec![10i64, 50], (15..=19).collect()];
    let (storage, _counter, _tmp) = write_clustered_blocks(&blocks, "t_bnull").await;
    let sql = "SELECT SUM(v) FROM t_bnull WHERE day BETWEEN 12 AND 27";

    let ctx_ref = make_ctx(Arc::clone(&storage), "t_bnull", false).await;
    let expected = run_single_i64(&ctx_ref, sql).await;
    // Only frag1 days 15..=19 pass: v = 1500+1600+1700+1800+1900 = 8500.
    assert_eq!(
        expected, 8500,
        "only the covered fragment's rows pass the filter"
    );

    let ctx_fast = make_ctx(Arc::clone(&storage), "t_bnull", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    assert_eq!(
        got, expected,
        "a NULL boundary SUM must not discard the covered contribution"
    );
}

/// Equality filter that exactly covers a single-day fragment.  Build a table
/// where one fragment holds only day=12 (frag covers `day = 12` exactly) → the
/// answer is the composed constant with zero Parquet I/O.
#[tokio::test]
async fn equality_exact_cover_zero_io() {
    // frag0=[12] (single day), frag1=[13..17], frag2=[18..22].
    let blocks = vec![vec![12i64], (13..=17).collect(), (18..=22).collect()];
    let (storage, counter, _tmp) = write_clustered_blocks(&blocks, "t_eq").await;
    let sql = "SELECT SUM(v) FROM t_eq WHERE day = 12";

    let ctx_ref = make_ctx(Arc::clone(&storage), "t_eq", false).await;
    let expected = run_single_i64(&ctx_ref, sql).await;

    counter.store(0, Ordering::Relaxed);
    let ctx_fast = make_ctx(Arc::clone(&storage), "t_eq", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    assert_eq!(got, expected, "day = 12 SUM must be byte-equal");

    let reads = counter.load(Ordering::Relaxed);
    assert_eq!(
        reads, 0,
        "day = 12 exactly covers the single-day fragment → zero Parquet I/O"
    );
}

/// Non-clustered table: every fragment holds the full day range, so the filter
/// straddles every fragment → no covered fragment → falls back to a full
/// filtered scan.  Result must still be correct (byte-equal).
#[tokio::test]
async fn non_clustered_falls_back_still_correct() {
    AggStateCache::global().clear();
    let tmp = tempfile::tempdir().unwrap();
    let local: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
    let arrow = schema();
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow));
    mdb_schema.row_group_target_rows = 1_000_000;

    // Two fragments, each spanning days 10..=15 (NOT clustered).
    let mut writer = Writer::create(Arc::clone(&local), "t_nc", mdb_schema)
        .await
        .unwrap();
    for _frag in 0..2 {
        let day_col: Vec<i64> = (10..=15).collect();
        let v_col: Vec<i64> = (10..=15).map(|d| d * 100).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow),
            vec![
                Arc::new(Int64Array::from(day_col)),
                Arc::new(Int64Array::from(v_col)),
            ],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();
    }
    let storage: Arc<dyn Storage> = local;

    let sql = "SELECT SUM(v) FROM t_nc WHERE day BETWEEN 11 AND 14";
    let ctx_ref = make_ctx(Arc::clone(&storage), "t_nc", false).await;
    let expected = run_single_i64(&ctx_ref, sql).await;

    let ctx_fast = make_ctx(Arc::clone(&storage), "t_nc", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;
    assert_eq!(
        got, expected,
        "non-clustered table must fall back to a correct full filtered scan"
    );
}

/// AVG with a range filter falls back (additive merge of AVG isn't a simple `+`).
/// The result must still be correct.
#[tokio::test]
async fn avg_with_filter_falls_back_still_correct() {
    let (storage, _counter, _tmp) = write_clustered_blocks(&four_blocks(), "t_avg").await;

    let sql = "SELECT AVG(v) FROM t_avg WHERE day BETWEEN 12 AND 27";

    let ctx_ref = make_ctx(Arc::clone(&storage), "t_avg", false).await;
    let df_ref = ctx_ref.sql(sql).await.unwrap();
    let ref_batches = df_ref.collect().await.unwrap();
    let expected = ref_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap()
        .value(0);

    let ctx_fast = make_ctx(Arc::clone(&storage), "t_avg", true).await;
    let df_fast = ctx_fast.sql(sql).await.unwrap();
    let fast_batches = df_fast.collect().await.unwrap();
    let got = fast_batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .unwrap()
        .value(0);

    assert!(
        (got - expected).abs() < 1e-9,
        "AVG with filter must fall back to a correct scan (got {got}, expected {expected})"
    );
}

/// A dirty covered fragment (a row deleted from a fully-covered day) must NOT be
/// composed from its sidecar partial — the rule falls back so the deletion is
/// honoured.  Result must be byte-equal to the rule-disabled scan.
#[tokio::test]
async fn dirty_covered_fragment_falls_back_still_correct() {
    let (storage, _counter, _tmp) = write_clustered_blocks(&four_blocks(), "t_dirty").await;

    // Delete one row from a FULLY-COVERED interior fragment (frag1=[15..19], day=17,
    // v=1700) so its sidecar partial no longer matches the on-disk survivors.
    // execute_sql writes a real deletion vector and bumps the manifest.
    let ctx_setup = make_ctx(Arc::clone(&storage), "t_dirty", true).await;
    execute_sql(
        &ctx_setup,
        Arc::clone(&storage),
        "t_dirty",
        "DELETE FROM t_dirty WHERE day = 17 AND v = 1700",
    )
    .await
    .unwrap();
    AggStateCache::global().clear();

    let sql = "SELECT SUM(v) FROM t_dirty WHERE day BETWEEN 12 AND 27";

    // Fresh providers so the post-delete manifest (with the deletion vector) is
    // loaded into both the reference and fast-path sessions.
    let ctx_ref = make_ctx(Arc::clone(&storage), "t_dirty", false).await;
    let expected = run_single_i64(&ctx_ref, sql).await;

    let ctx_fast = make_ctx(Arc::clone(&storage), "t_dirty", true).await;
    let got = run_single_i64(&ctx_fast, sql).await;

    assert_eq!(
        got, expected,
        "a dirty covered fragment must fall back so the deletion is honoured"
    );
}
