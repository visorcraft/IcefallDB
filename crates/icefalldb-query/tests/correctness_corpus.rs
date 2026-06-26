//! Correctness corpus: sidecar statistics are an optimization, never a source of
//! truth for row data.
//!
//! We build a temp IcefallDB table, record query results as the truth, delete (or
//! corrupt) every `.meta` sidecar, and assert that the same queries return
//! identical answers. Missing sidecars must only degrade performance.

use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use datafusion::prelude::SessionContext;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn corpus_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("int_col", DataType::Int64, false),
        Field::new("float_col", DataType::Float64, true),
        Field::new("utf8_col", DataType::Utf8, true),
        Field::new("nullable_col", DataType::Int64, true),
    ]));

    let id = Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
    let int_col = Int64Array::from(vec![10, 20, 30, 40, 50, 60, 70, 80]);
    let float_col = Float64Array::from(vec![
        Some(1.5),
        Some(2.5),
        None,
        Some(4.5),
        Some(5.5),
        Some(6.5),
        Some(7.5),
        Some(8.5),
    ]);
    let utf8_col = StringArray::from(vec![
        Some("a"),
        Some("b"),
        Some("c"),
        Some("x"),
        Some("x"),
        Some("d"),
        None,
        Some("e"),
    ]);
    let nullable_col = Int64Array::from(vec![
        Some(100),
        None,
        Some(200),
        None,
        Some(300),
        Some(400),
        None,
        Some(500),
    ]);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id),
            Arc::new(int_col),
            Arc::new(float_col),
            Arc::new(utf8_col),
            Arc::new(nullable_col),
        ],
    )
    .unwrap()
}

async fn create_corpus_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let mut schema = arrow_schema_to_icefalldb(corpus_batch().schema());
    // Force several row groups so that metadata shortcuts have to combine stats.
    schema.row_group_target_rows = 3;
    let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
        .await
        .unwrap();
    writer.insert_batch(corpus_batch()).await.unwrap();
    writer.commit().await.unwrap();
    storage
}

fn remove_meta_files(dir: &std::path::Path) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            remove_meta_files(&path);
        } else if path.extension().and_then(|e| e.to_str()) == Some("meta") {
            std::fs::remove_file(&path).unwrap();
        }
    }
}

fn corrupt_meta_file(dir: &std::path::Path) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            corrupt_meta_file(&path);
        } else if path.extension().and_then(|e| e.to_str()) == Some("meta") {
            std::fs::write(&path, b"{}").unwrap();
            return;
        }
    }
}

async fn query_ctx_for(storage: Arc<dyn Storage>) -> SessionContext {
    let provider = IcefallDBTableProvider::new(
        storage,
        "t",
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 256,
            tiny_table_cache_threshold_rows: 65_536,
            tiny_table_cache_threshold_bytes: 8 * 1024 * 1024,
            wal_mode: true,
        },
    )
    .await
    .unwrap();
    let ctx = icefalldb_session(1, 1024);
    ctx.register_table("t", Arc::new(provider)).unwrap();
    ctx
}

async fn run_query(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

fn extract_i64(batches: &[RecordBatch]) -> i64 {
    assert!(!batches.is_empty(), "expected at least one batch");
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

fn extract_i64_pair(batches: &[RecordBatch]) -> (i64, i64) {
    assert!(!batches.is_empty());
    let min = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    let max = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    (min, max)
}

fn extract_f64_pair(batches: &[RecordBatch]) -> (f64, f64) {
    assert!(!batches.is_empty());
    let min = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    let max = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    (min, max)
}

fn extract_null_rows(batches: &[RecordBatch]) -> Vec<(i64, i64, Option<f64>, Option<String>)> {
    let mut rows = Vec::new();
    for batch in batches {
        let id = batch
            .column_by_name("id")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let int_col = batch
            .column_by_name("int_col")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let float_col = batch
            .column_by_name("float_col")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let utf8_col = batch
            .column_by_name("utf8_col")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            rows.push((
                id.value(i),
                int_col.value(i),
                if float_col.is_null(i) {
                    None
                } else {
                    Some(float_col.value(i))
                },
                if utf8_col.is_null(i) {
                    None
                } else {
                    Some(utf8_col.value(i).to_string())
                },
            ));
        }
    }
    rows
}

fn f64_eq(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    a.total_cmp(&b).is_eq()
}

#[derive(Debug, Clone)]
struct CorpusTruth {
    count_star: i64,
    count_col: i64,
    min_int: i64,
    max_int: i64,
    min_float: f64,
    max_float: f64,
    count_gt2: i64,
    count_eq_x: i64,
    null_rows: Vec<(i64, i64, Option<f64>, Option<String>)>,
}

async fn collect_truth(ctx: &SessionContext) -> CorpusTruth {
    CorpusTruth {
        count_star: extract_i64(&run_query(ctx, "SELECT COUNT(*) FROM t").await),
        count_col: extract_i64(&run_query(ctx, "SELECT COUNT(nullable_col) FROM t").await),
        min_int: extract_i64_pair(
            &run_query(ctx, "SELECT MIN(int_col), MAX(int_col) FROM t").await,
        )
        .0,
        max_int: extract_i64_pair(
            &run_query(ctx, "SELECT MIN(int_col), MAX(int_col) FROM t").await,
        )
        .1,
        min_float: extract_f64_pair(
            &run_query(ctx, "SELECT MIN(float_col), MAX(float_col) FROM t").await,
        )
        .0,
        max_float: extract_f64_pair(
            &run_query(ctx, "SELECT MIN(float_col), MAX(float_col) FROM t").await,
        )
        .1,
        count_gt2: extract_i64(&run_query(ctx, "SELECT COUNT(*) FROM t WHERE int_col > 2").await),
        count_eq_x: extract_i64(
            &run_query(ctx, "SELECT COUNT(*) FROM t WHERE utf8_col = 'x'").await,
        ),
        null_rows: extract_null_rows(
            &run_query(
                ctx,
                "SELECT * FROM t WHERE nullable_col IS NULL ORDER BY id",
            )
            .await,
        ),
    }
}

fn assert_truth_matches(actual: &CorpusTruth, expected: &CorpusTruth) {
    assert_eq!(actual.count_star, expected.count_star, "COUNT(*) mismatch");
    assert_eq!(
        actual.count_col, expected.count_col,
        "COUNT(nullable_col) mismatch"
    );
    assert_eq!(
        (actual.min_int, actual.max_int),
        (expected.min_int, expected.max_int),
        "MIN/MAX(int_col) mismatch"
    );
    assert!(
        f64_eq(actual.min_float, expected.min_float),
        "MIN(float_col) mismatch: {} != {}",
        actual.min_float,
        expected.min_float
    );
    assert!(
        f64_eq(actual.max_float, expected.max_float),
        "MAX(float_col) mismatch: {} != {}",
        actual.max_float,
        expected.max_float
    );
    assert_eq!(
        actual.count_gt2, expected.count_gt2,
        "COUNT(*) WHERE int_col > 2 mismatch"
    );
    assert_eq!(
        actual.count_eq_x, expected.count_eq_x,
        "COUNT(*) WHERE utf8_col = 'x' mismatch"
    );
    assert_eq!(
        actual.null_rows, expected.null_rows,
        "IS NULL rows mismatch"
    );
}

#[tokio::test]
async fn test_stats_degradation_when_meta_removed() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_corpus_table(tmp.path()).await;

    let ctx = query_ctx_for(Arc::clone(&storage)).await;
    let truth = collect_truth(&ctx).await;

    remove_meta_files(&tmp.path().join("t"));

    let ctx_after = query_ctx_for(storage).await;
    let after_truth = collect_truth(&ctx_after).await;

    assert_truth_matches(&after_truth, &truth);
}

#[tokio::test]
async fn test_stats_degradation_when_meta_corrupted() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_corpus_table(tmp.path()).await;

    let ctx = query_ctx_for(Arc::clone(&storage)).await;
    let truth = collect_truth(&ctx).await;

    corrupt_meta_file(&tmp.path().join("t"));

    let ctx_after = query_ctx_for(storage).await;
    let after_truth = collect_truth(&ctx_after).await;

    assert_truth_matches(&after_truth, &truth);
}
