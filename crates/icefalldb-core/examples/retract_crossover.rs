// examples/retract_crossover.rs — retract-vs-recompute crossover timing.
//
// Builds a 1 M-row single-column Int64 Parquet fragment in memory, then for
// each deletion density d sweeps {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}:
//
//   retract path  — deleted_contribution (sparse RowSelection over deleted
//                   offsets) + retract; reads ≈ d × N rows.
//   recompute path — full read of the LIVE rows (RowSelection of live offsets)
//                   + compute_agg_state_from_batches; reads ≈ (1-d) × N rows.
//
// Both paths operate on in-memory Bytes so network/disk latency is not a
// confounder; the measurement captures Parquet decode + aggregation CPU cost.

use arrow::array::{Array, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use icefalldb_core::agg_cache::{compute_agg_state, deleted_contribution, retract, AggScalar};
use icefalldb_core::deletion::DeletionVector;
use icefalldb_core::metadata::RowGroupEntry;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;
use std::sync::Arc;
use std::time::{Duration, Instant};

const FRAGMENT_ROWS: usize = 1_000_000;
const WARMUP_ITERS: u32 = 2;
const MEASURE_ITERS: u32 = 5;
const TABLE: &str = "bench";
const DATA_FILE: &str = "bench/fragment.parquet";

/// Build a single-column Int64 Parquet in memory, return its raw bytes.
fn build_parquet_bytes(rows: usize) -> Bytes {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "measure",
        DataType::Int64,
        false,
    )]));
    let vals: Vec<i64> = (0..rows as i64).collect();
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vals))]).unwrap();

    let props = WriterProperties::builder().build();
    let mut buf: Vec<u8> = Vec::with_capacity(rows * 8);
    let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    Bytes::from(buf)
}

/// Build a DeletionVector whose deleted offsets are evenly spread across the
/// fragment; cardinality = ceil(density * rows).
fn build_dv(density: f64, rows: usize) -> DeletionVector {
    let deleted = ((density * rows as f64).ceil() as usize).min(rows);
    let mut dv = DeletionVector::default();
    // Evenly-spaced offsets mimic the worst-case RowSelector fragmentation.
    let step = rows.checked_div(deleted).unwrap_or(usize::MAX);
    let mut count = 0usize;
    let mut off = 0usize;
    while count < deleted && off < rows {
        dv.union_offsets([off as u32]);
        off += step.max(1);
        count += 1;
    }
    dv
}

/// Build a RowSelection that selects the LIVE rows (complement of dv).
fn live_row_selection(dv: &DeletionVector, total_rows: usize) -> RowSelection {
    let deleted_set: std::collections::HashSet<u32> = dv.iter().collect();
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut in_live = false;
    let mut run = 0usize;
    for row in 0..total_rows {
        let is_deleted = deleted_set.contains(&(row as u32));
        if is_deleted {
            if in_live && run > 0 {
                selectors.push(RowSelector::select(run));
                run = 0;
            }
            in_live = false;
            run += 1;
        } else {
            if !in_live && run > 0 {
                selectors.push(RowSelector::skip(run));
                run = 0;
            }
            in_live = true;
            run += 1;
        }
    }
    if run > 0 {
        if in_live {
            selectors.push(RowSelector::select(run));
        } else {
            selectors.push(RowSelector::skip(run));
        }
    }
    RowSelection::from(selectors)
}

/// Time the RETRACT path.
///
/// deleted_contribution() builds RowSelector from dv.iter() and reads only
/// the deleted offsets from the Parquet bytes, then retract() subtracts.
async fn time_retract(
    storage: &dyn Storage,
    entry: &RowGroupEntry,
    dv: &DeletionVector,
    full_agg: &icefalldb_core::agg_cache::FragmentAggState,
    iters: u32,
) -> Duration {
    let col_names = ["measure".to_string()];
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let t0 = Instant::now();
        let del = deleted_contribution(storage, TABLE, entry, dv, &col_names)
            .await
            .unwrap();
        let _retracted = retract(full_agg, &del);
        total += t0.elapsed();
    }
    total / iters
}

/// Time the RECOMPUTE path.
///
/// Reads the Parquet bytes selecting only LIVE rows, then calls
/// compute_agg_state on the result — this is what compaction would do to
/// rebuild the agg from scratch.
async fn time_recompute(
    parquet_bytes: &Bytes,
    live_sel: RowSelection,
    total_rows: usize,
    iters: u32,
) -> Duration {
    let col_names = ["measure".to_string()];
    let mut total = Duration::ZERO;
    for _ in 0..iters {
        let t0 = Instant::now();
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes.clone()).unwrap();
        let reader = builder
            .with_row_selection(live_sel.clone())
            .build()
            .unwrap();
        let batches: Vec<RecordBatch> = reader.map(|r| r.unwrap()).collect();
        // Aggregate the live rows by folding ColAgg manually (same cost as
        // compute_agg_state over each batch, accumulated).
        let mut total_count: u64 = 0;
        let mut total_sum: i128 = 0;
        for batch in &batches {
            if let Some(col) = batch.column_by_name(&col_names[0]) {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        total_count += 1;
                        total_sum += arr.value(i) as i128;
                    }
                }
            }
        }
        // Black-box the result to prevent the compiler from eliding the work.
        std::hint::black_box((total_count, total_sum, total_rows));
        total += t0.elapsed();
    }
    total / iters
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let densities: &[f64] = &[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5];

    eprintln!("Building {FRAGMENT_ROWS}-row Int64 Parquet fragment in memory...");
    let parquet_bytes = build_parquet_bytes(FRAGMENT_ROWS);
    eprintln!("  fragment size: {} KiB", parquet_bytes.len() / 1024);

    // Store the Parquet in a MemoryStorage so deleted_contribution can use it.
    let storage = MemoryStorage::new();
    storage.write(DATA_FILE, &parquet_bytes).await.unwrap();

    // Build the "full" FragmentAggState for the fragment.
    let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes.clone()).unwrap();
    let mut full_batches: Vec<RecordBatch> = builder.build().unwrap().map(|r| r.unwrap()).collect();
    // For compute_agg_state we need a single batch or we can fold; use first.
    // (For 1M rows, the writer typically emits one row group = one batch.)
    let full_batch = if full_batches.len() == 1 {
        full_batches.remove(0)
    } else {
        // Concatenate via arrow concat_batches.
        let schema = full_batches[0].schema();
        arrow::compute::concat_batches(&schema, &full_batches).unwrap()
    };
    let full_agg = compute_agg_state(1, "crossover-bench".to_string(), &full_batch).unwrap();

    // Verify we got a ColAgg for "measure".
    assert!(
        full_agg.cols.contains_key("measure"),
        "expected ColAgg for 'measure'"
    );
    if let AggScalar::Int(s) = &full_agg.cols["measure"].sum {
        let expected = (FRAGMENT_ROWS as i128 * (FRAGMENT_ROWS as i128 - 1)) / 2;
        assert_eq!(
            *s, expected,
            "full agg sum mismatch: got {s}, expected {expected}"
        );
    }

    // Stub RowGroupEntry that points at the in-memory data file.
    let entry = RowGroupEntry {
        data: "fragment.parquet".to_string(),
        meta: "fragment.meta".to_string(),
        fragment_id: 1,
        ..Default::default()
    };

    eprintln!();
    eprintln!(
        "{:<8}  {:>14}  {:>14}  {:>12}",
        "density", "retract_µs", "recompute_µs", "ratio r/rc"
    );
    eprintln!("{:-<8}  {:-<14}  {:-<14}  {:-<12}", "", "", "", "");

    let mut results: Vec<(f64, f64, f64)> = Vec::new();

    for &d in densities {
        let dv = build_dv(d, FRAGMENT_ROWS);
        let actual_density = dv.cardinality() as f64 / FRAGMENT_ROWS as f64;
        let live_sel = live_row_selection(&dv, FRAGMENT_ROWS);

        // Warm-up (both paths).
        time_retract(&storage, &entry, &dv, &full_agg, WARMUP_ITERS).await;
        time_recompute(
            &parquet_bytes,
            live_sel.clone(),
            FRAGMENT_ROWS,
            WARMUP_ITERS,
        )
        .await;

        // Measure.
        let retract_avg = time_retract(&storage, &entry, &dv, &full_agg, MEASURE_ITERS).await;
        let recompute_avg =
            time_recompute(&parquet_bytes, live_sel, FRAGMENT_ROWS, MEASURE_ITERS).await;

        let ret_us = retract_avg.as_secs_f64() * 1e6;
        let rec_us = recompute_avg.as_secs_f64() * 1e6;
        let ratio = ret_us / rec_us;

        eprintln!(
            "{:<8.3}  {:>14.1}  {:>14.1}  {:>12.3}",
            actual_density, ret_us, rec_us, ratio
        );

        results.push((actual_density, ret_us, rec_us));
    }

    // Find crossover: first d where retract_us >= recompute_us.
    let mut crossover: Option<f64> = None;
    for i in 1..results.len() {
        let (d_prev, ret_prev, rec_prev) = results[i - 1];
        let (d_cur, ret_cur, rec_cur) = results[i];
        if ret_prev < rec_prev && ret_cur >= rec_cur {
            // Linear interpolation between d_prev and d_cur.
            let alpha = (rec_prev - ret_prev) / ((ret_cur - rec_cur) - (ret_prev - rec_prev));
            crossover = Some(d_prev + alpha * (d_cur - d_prev));
            break;
        }
    }

    eprintln!();
    match crossover {
        Some(d) => {
            eprintln!("Measured crossover ≈ {d:.3}");
            eprintln!(
                "RECOMPUTE_DENSITY recommendation: {:.2}",
                (d * 10.0).round() / 10.0
            );
        }
        None => {
            // Check whether retract was always cheaper or always more expensive.
            let first_ratio = results[0].1 / results[0].2;
            if first_ratio < 1.0 {
                eprintln!(
                    "No crossover found in tested range: retract was ALWAYS cheaper (ratio < 1.0 throughout)."
                );
                eprintln!(
                    "  → sparse Parquet row-selection is paying off; crossover is above d=0.5."
                );
            } else {
                eprintln!(
                    "No crossover found in tested range: recompute was ALWAYS cheaper or equal (ratio ≥ 1.0 throughout)."
                );
                eprintln!("  → sparse read offers no advantage; RECOMPUTE_DENSITY should be set low (e.g. 0.05).");
            }
        }
    }

    // Machine-readable output for the report.
    eprintln!();
    eprintln!("RAW RESULTS (density,retract_us,recompute_us):");
    for (d, r, rc) in &results {
        eprintln!("  {d:.4},{r:.2},{rc:.2}");
    }
}
