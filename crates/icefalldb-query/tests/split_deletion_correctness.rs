//! REGRESSION: scans over a table with deletions whose row group is large
//! enough to be split for intra-row-group parallelism must NOT duplicate
//! surviving rows.
//!
//! A single fragment of N >= `MIN_ROWS_PER_PARTITION * 2` (= 20 000) rows is
//! split into several parallel scan partitions.  Each partition's
//! `RowSelection` was built as `[skip(offset), select(chunk)]`, covering only
//! `offset + chunk` rows.  When the fragment also carried a deletion vector,
//! the scan intersected that (shorter) selection with the full-length
//! deletion-vector selection; `RowSelection::intersection` appends the longer
//! selection's trailing selectors verbatim, so every partition ended up
//! reading a suffix to the END of the row group.  The union over partitions
//! produced a triangular ~`N_partitions * live / 2` duplication: a 25k-row
//! table with 5k deleted returned COUNT(*) = 169 981 instead of 20 000, broke
//! GROUP BY counts, and made compaction fail with "row_id appears more than
//! once".
//!
//! These end-to-end tests build a >=20k-row single-fragment table, apply a real
//! DELETE (and an UPDATE), then query through a FRESH provider configured with
//! multiple target partitions (so the split path runs) and assert the live
//! counts are correct and free of physical duplicates, and that compaction
//! succeeds after the mutation.

use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::execution::context::SessionContext;
use std::time::Duration;

use icefalldb_core::compaction::{CompactionOptions, Compactor};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
use icefalldb_query::{
    execute_sql, icefalldb_session_config, icefalldb_session_state_from_config,
    IcefallDBTableProvider, ProviderConfig,
};

const N: usize = 25_000;

fn schema() -> Arc<ArrowSchema> {
    Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("grp", DataType::Int64, false),
    ]))
}

/// Write an N-row table as a SINGLE fragment / row group (`row_group_target_rows`
/// is set huge so the whole insert stays in one row group).  `id` = row index,
/// `grp` = id % 10.  Returns the storage and the temp dir (kept alive by caller).
async fn write_single_fragment(table: &str) -> (Arc<dyn Storage>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

    let arrow = schema();
    let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow));
    // Force the entire insert into a single row group so the scan must SPLIT it
    // (the bug only manifests through `split_row_groups`).
    mdb_schema.row_group_target_rows = 10_000_000;
    mdb_schema.row_group_target_bytes = usize::MAX;

    let mut writer = Writer::create(Arc::clone(&storage), table, mdb_schema)
        .await
        .unwrap();

    let id_col: Vec<i64> = (0..N as i64).collect();
    let grp_col: Vec<i64> = (0..N as i64).map(|i| i % 10).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&arrow),
        vec![
            Arc::new(Int64Array::from(id_col)),
            Arc::new(Int64Array::from(grp_col)),
        ],
    )
    .unwrap();
    writer.insert_batch(batch).await.unwrap();
    writer.commit().await.unwrap();

    (storage, tmp)
}

/// A FRESH query context whose provider uses `target_partitions > 1` and forces
/// the custom IcefallDBScanExec path (`native_parquet_threshold = MAX`), so a
/// single large row group with a deletion vector goes through `split_row_groups`.
async fn fresh_split_ctx(storage: Arc<dyn Storage>, table: &str) -> SessionContext {
    let provider = IcefallDBTableProvider::new(
        Arc::clone(&storage),
        table,
        ProviderConfig {
            batch_size: 1024,
            // > 1 and > number of row groups (1) so the single fragment splits.
            target_partitions: 8,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        },
    )
    .await
    .unwrap();

    let cfg = icefalldb_session_config(8, 1024);
    let state = icefalldb_session_state_from_config(cfg);
    let ctx = SessionContext::new_with_state(state);
    ctx.register_table(table, Arc::new(provider)).unwrap();
    ctx
}

async fn run_single_i64(ctx: &SessionContext, sql: &str) -> i64 {
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 1, "expected exactly one row for `{sql}`");
    let batch = batches.iter().find(|b| b.num_rows() == 1).unwrap();
    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert!(!arr.is_null(0), "result of `{sql}` must not be NULL");
    arr.value(0)
}

/// Collect a GROUP BY result into a sorted `(key, count)` vector.
async fn run_group_counts(ctx: &SessionContext, sql: &str) -> Vec<(i64, i64)> {
    let df = ctx.sql(sql).await.unwrap();
    let batches = df.collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let keys = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let counts = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            out.push((keys.value(i), counts.value(i)));
        }
    }
    out.sort_unstable();
    out
}

/// DELETE 5 000 rows (id % 5 == 0) from a 25k-row single fragment, then assert —
/// through the split scan path — that COUNT(*), a filtered COUNT(*), and a
/// GROUP BY ... COUNT(*) all return the correct (non-inflated) values and that
/// COUNT(DISTINCT id) == COUNT(*) (no physical duplicates).
#[tokio::test]
async fn split_scan_after_delete_counts_live_rows_exactly_once() {
    let (storage, _tmp) = write_single_fragment("t_del").await;

    // Sanity: before the delete the split scan must see all N rows.
    {
        let ctx = fresh_split_ctx(Arc::clone(&storage), "t_del").await;
        assert_eq!(
            run_single_i64(&ctx, "SELECT COUNT(*) FROM t_del").await,
            N as i64,
            "pre-delete COUNT(*) must equal N"
        );
    }

    // DELETE every 5th row -> 5 000 deletions, 20 000 survivors.
    {
        let mut_ctx = fresh_split_ctx(Arc::clone(&storage), "t_del").await;
        let deleted = execute_sql(
            &mut_ctx,
            Arc::clone(&storage),
            "t_del",
            "DELETE FROM t_del WHERE id % 5 = 0",
        )
        .await
        .unwrap();
        assert_eq!(
            deleted,
            (N / 5) as u64,
            "DELETE must remove exactly N/5 rows"
        );
    }

    let live = (N - N / 5) as i64; // 20 000

    // Re-open a FRESH provider (target_partitions = 8) so the now-deletion-bearing
    // single row group goes through `split_row_groups`.
    let ctx = fresh_split_ctx(Arc::clone(&storage), "t_del").await;

    let count_star = run_single_i64(&ctx, "SELECT COUNT(*) FROM t_del").await;
    assert_eq!(
        count_star, live,
        "COUNT(*) after delete must be the live count {live}, not an inflated value"
    );

    // No physical duplicates: distinct primary keys must equal the live count.
    let distinct = run_single_i64(&ctx, "SELECT COUNT(DISTINCT id) FROM t_del").await;
    assert_eq!(
        distinct, live,
        "COUNT(DISTINCT id) must equal COUNT(*) (each live row read exactly once)"
    );
    assert_eq!(distinct, count_star, "no physical row duplication");

    // Filtered COUNT(*): of ids in [0, 10000), the survivors are those not
    // divisible by 5 -> 10000 - 2000 = 8000.
    let filtered = run_single_i64(&ctx, "SELECT COUNT(*) FROM t_del WHERE id < 10000").await;
    assert_eq!(
        filtered, 8_000,
        "filtered COUNT(*) (id < 10000) must be 8000 survivors"
    );

    // GROUP BY grp COUNT(*): grp = id % 10.  For each grp g, the rows are
    // ids {g, g+10, g+20, ...} (2500 of them); a row is deleted iff id % 5 == 0,
    // i.e. iff g % 5 == 0.  So groups 0 and 5 lose all 2500 of their rows
    // (deleted), and every other group keeps all 2500.
    let groups = run_group_counts(
        &ctx,
        "SELECT grp, COUNT(*) FROM t_del GROUP BY grp ORDER BY grp",
    )
    .await;
    let expected: Vec<(i64, i64)> = (0..10)
        .map(|g| (g, if g % 5 == 0 { 0 } else { 2_500 }))
        .filter(|&(_, c)| c > 0) // fully-deleted groups produce no rows
        .collect();
    assert_eq!(
        groups, expected,
        "GROUP BY grp COUNT(*) must reflect the true live distribution"
    );
    let grouped_total: i64 = groups.iter().map(|&(_, c)| c).sum();
    assert_eq!(
        grouped_total, live,
        "the grouped counts must sum to the live total {live}"
    );

    // Compaction must succeed post-delete (the bug made `derive_base` fail with
    // "row_id appears more than once") and produce exactly the live row count.
    // `force: true` ensures the deletion-bearing fragment is actually rewritten
    // (the path that builds the `_rowindex` base and surfaced the duplication).
    let opts = CompactionOptions {
        force: true,
        lock_timeout: Duration::from_secs(30),
        ..CompactionOptions::default()
    };
    let result = Compactor::with_options(storage.as_ref(), "t_del", opts)
        .compact()
        .await
        .expect("compaction must succeed after a delete on a split-eligible fragment");
    assert!(
        result.rewrote,
        "compaction must rewrite the deletion-bearing fragment"
    );
    assert_eq!(
        result.output_rows, live as u64,
        "compacted output must contain exactly the live rows"
    );

    // After compaction the count must still be exactly the live set.
    let ctx2 = fresh_split_ctx(Arc::clone(&storage), "t_del").await;
    assert_eq!(
        run_single_i64(&ctx2, "SELECT COUNT(*) FROM t_del").await,
        live,
        "post-compaction COUNT(*) must remain the live count"
    );
}

/// An UPDATE on a split-eligible fragment writes a tombstone for the updated
/// rows plus a new fragment with the post-image; the original fragment is now
/// deletion-bearing AND split-eligible.  Assert the row count stays correct and
/// the updated value is visible exactly once.
#[tokio::test]
async fn split_scan_after_update_is_consistent() {
    let (storage, _tmp) = write_single_fragment("t_upd").await;

    // UPDATE 5 000 rows (id % 5 == 0): set grp = 99.
    {
        let mut_ctx = fresh_split_ctx(Arc::clone(&storage), "t_upd").await;
        let updated = execute_sql(
            &mut_ctx,
            Arc::clone(&storage),
            "t_upd",
            "UPDATE t_upd SET grp = 99 WHERE id % 5 = 0",
        )
        .await
        .unwrap();
        assert_eq!(
            updated,
            (N / 5) as u64,
            "UPDATE must touch exactly N/5 rows"
        );
    }

    let ctx = fresh_split_ctx(Arc::clone(&storage), "t_upd").await;

    // Total row count is preserved (UPDATE never changes cardinality).
    assert_eq!(
        run_single_i64(&ctx, "SELECT COUNT(*) FROM t_upd").await,
        N as i64,
        "UPDATE must preserve the total row count (no duplication from the split)"
    );
    assert_eq!(
        run_single_i64(&ctx, "SELECT COUNT(DISTINCT id) FROM t_upd").await,
        N as i64,
        "every id must appear exactly once after the UPDATE"
    );

    // Exactly the updated rows carry grp = 99.
    assert_eq!(
        run_single_i64(&ctx, "SELECT COUNT(*) FROM t_upd WHERE grp = 99").await,
        (N / 5) as i64,
        "exactly N/5 rows must carry the updated value, read once each"
    );

    // Compaction must succeed and preserve the full cardinality.
    let opts = CompactionOptions {
        force: true,
        lock_timeout: Duration::from_secs(30),
        ..CompactionOptions::default()
    };
    let result = Compactor::with_options(storage.as_ref(), "t_upd", opts)
        .compact()
        .await
        .expect("compaction must succeed after an update on a split-eligible fragment");
    assert_eq!(
        result.output_rows, N as u64,
        "compacted output must contain exactly N live rows"
    );
}
