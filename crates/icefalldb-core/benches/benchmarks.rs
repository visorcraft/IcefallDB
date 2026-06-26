use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use bytes::Bytes;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use futures::StreamExt;
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::memory::MemoryStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{
    Checker, CompactionOptions, Compactor, GarbageCollector, Reader, TsvDecoder, TsvEncoder, Writer,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

const TABLE: &str = "bench";

fn make_icefalldb_schema(row_group_target_rows: usize) -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![
            Column::new("id", "int64", false),
            Column::new("value", "float64", false),
            Column::new("name", "utf8", false),
        ],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

fn make_arrow_schema() -> ArrowSchema {
    ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
        Field::new("name", DataType::Utf8, false),
    ])
}

/// Generate a deterministic record batch of `rows` rows starting at `offset`.
fn make_batch(rows: usize, offset: i64) -> RecordBatch {
    let schema = Arc::new(make_arrow_schema());
    let ids: Vec<i64> = (0..rows).map(|i| offset + i as i64).collect();
    // Use whole-number floats to avoid a serde_json float-roundtrip issue in
    // IcefallDB's row-group metadata checksum (non-terminating decimal floats can
    // serialize differently on write vs. canonicalization).
    let values: Vec<f64> = ids.iter().map(|&i| i as f64).collect();
    let names: Vec<String> = ids.iter().map(|&i| format!("name-{i}")).collect();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Float64Array::from(values)),
            Arc::new(StringArray::from(names)),
        ],
    )
    .unwrap()
}

async fn make_memory_table(row_groups: usize, rows_per_group: usize) -> Arc<MemoryStorage> {
    let storage: Arc<MemoryStorage> = Arc::new(MemoryStorage::new());
    let schema = make_icefalldb_schema(rows_per_group);
    let mut writer = Writer::create(Arc::clone(&storage) as Arc<dyn Storage>, TABLE, schema)
        .await
        .unwrap();

    for g in 0..row_groups {
        let offset = (g * rows_per_group) as i64;
        writer
            .insert_batch(make_batch(rows_per_group, offset))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    storage
}

async fn make_local_table(
    root: &std::path::Path,
    row_groups: usize,
    rows_per_group: usize,
) -> Arc<LocalStorage> {
    let storage: Arc<LocalStorage> = Arc::new(LocalStorage::new(root).unwrap());
    let schema = make_icefalldb_schema(rows_per_group);
    let mut writer = Writer::create(Arc::clone(&storage) as Arc<dyn Storage>, TABLE, schema)
        .await
        .unwrap();

    for g in 0..row_groups {
        let offset = (g * rows_per_group) as i64;
        writer
            .insert_batch(make_batch(rows_per_group, offset))
            .await
            .unwrap();
        writer.commit().await.unwrap();
    }

    storage
}

fn ingest_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let total_rows = 100_000usize;
    let batch_sizes = vec![1_000, 10_000, 50_000, 100_000];

    let mut group = c.benchmark_group("ingest_throughput");
    for batch_size in batch_sizes {
        let num_batches = total_rows / batch_size;
        group.throughput(Throughput::Elements(total_rows as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                b.to_async(&rt).iter_custom(|iters| async move {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let storage: Arc<dyn Storage> = Arc::new(MemoryStorage::new());
                        let schema = make_icefalldb_schema(batch_size);
                        let mut writer = Writer::create(Arc::clone(&storage), TABLE, schema)
                            .await
                            .unwrap();

                        let start = Instant::now();
                        for i in 0..num_batches {
                            let offset = (i * batch_size) as i64;
                            writer
                                .insert_batch(make_batch(batch_size, offset))
                                .await
                                .unwrap();
                            writer.commit().await.unwrap();
                        }
                        total += start.elapsed();
                    }
                    total
                })
            },
        );
    }
    group.finish();
}

fn local_ingest_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("local_ingest_throughput");
    group.measurement_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(10_000));
    group.bench_function("10k_rows", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(dir.path()).unwrap());
                let schema = make_icefalldb_schema(10_000);
                let mut writer = Writer::create(Arc::clone(&storage), TABLE, schema)
                    .await
                    .unwrap();

                let start = Instant::now();
                writer.insert_batch(make_batch(10_000, 0)).await.unwrap();
                writer.commit().await.unwrap();
                total += start.elapsed();
            }
            total
        })
    });
    group.finish();
}

fn query_latency_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("query_latency");

    group.bench_function("cold_scan", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let storage = make_memory_table(10, 10_000).await;

                let start = Instant::now();
                let reader = Reader::new(&*storage, TABLE).await.unwrap();
                let plan = reader.scan().await.unwrap();
                black_box(plan.row_groups.len());
                total += start.elapsed();
            }
            total
        })
    });

    group.bench_function("warm_scan", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let storage = make_memory_table(10, 10_000).await;
                let mut reader = Reader::new(&*storage, TABLE).await.unwrap();
                // Warm-up: load the catalog snapshot once before measuring.
                let _ = reader.scan().await.unwrap();

                let start = Instant::now();
                reader.refresh().await.unwrap();
                let plan = reader.scan().await.unwrap();
                black_box(plan.row_groups.len());
                total += start.elapsed();
            }
            total
        })
    });

    group.bench_function("full_read", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let storage = make_memory_table(10, 10_000).await;
                let reader = Reader::new(&*storage, TABLE).await.unwrap();

                let start = Instant::now();
                let plan = reader.scan().await.unwrap();
                let mut rows = 0usize;
                for rg in &plan.row_groups {
                    let mut stream = reader.read_row_group(rg).await.unwrap();
                    while let Some(batch) = stream.next().await {
                        rows += batch.unwrap().num_rows();
                    }
                }
                black_box(rows);
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn predicate_pruning_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("predicate_pruning");
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("prune_by_stats", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let storage = make_memory_table(100, 1_000).await;
                let reader = Reader::new(&*storage, TABLE).await.unwrap();
                let plan = reader.scan().await.unwrap();

                // id ranges are contiguous per row group, so this predicate
                // should prune roughly half of the 100 row groups.
                let predicate = icefalldb_core::Predicate::Gte {
                    column: "id".into(),
                    value: 50_000i64.into(),
                };

                let start = Instant::now();
                let pruned = plan.prune(&[predicate]).unwrap();
                black_box(pruned.row_groups.len());
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn compaction_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("compaction_impact");
    // Keep the filesystem benchmark short: each iteration creates 50 small row
    // groups with fsyncs.
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(500));

    group.bench_function("before", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let storage = make_local_table(dir.path(), 50, 100).await;
                let reader = Reader::new(&*storage, TABLE).await.unwrap();

                let start = Instant::now();
                let plan = reader.scan().await.unwrap();
                let mut rows = 0usize;
                for rg in &plan.row_groups {
                    let mut stream = reader.read_row_group(rg).await.unwrap();
                    while let Some(batch) = stream.next().await {
                        rows += batch.unwrap().num_rows();
                    }
                }
                black_box(rows);
                total += start.elapsed();
            }
            total
        })
    });

    group.bench_function("after", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let storage = make_local_table(dir.path(), 50, 100).await;
                let compactor = Compactor::with_options(
                    &*storage,
                    TABLE,
                    CompactionOptions {
                        target_row_group_rows: 5_000,
                        target_row_group_bytes: 1024 * 1024,
                        ..Default::default()
                    },
                );
                compactor.compact().await.unwrap();
                let reader = Reader::new(&*storage, TABLE).await.unwrap();

                let start = Instant::now();
                let plan = reader.scan().await.unwrap();
                let mut rows = 0usize;
                for rg in &plan.row_groups {
                    let mut stream = reader.read_row_group(rg).await.unwrap();
                    while let Some(batch) = stream.next().await {
                        rows += batch.unwrap().num_rows();
                    }
                }
                black_box(rows);
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn compaction_time_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("compaction_time");
    group.measurement_time(Duration::from_secs(1));
    group.warm_up_time(Duration::from_millis(500));

    group.bench_function("compact_50_small_groups", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let storage = make_local_table(dir.path(), 50, 100).await;
                let compactor = Compactor::with_options(
                    &*storage,
                    TABLE,
                    CompactionOptions {
                        target_row_group_rows: 5_000,
                        target_row_group_bytes: 1024 * 1024,
                        ..Default::default()
                    },
                );

                let start = Instant::now();
                compactor.compact().await.unwrap();
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn metadata_overhead_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("metadata_overhead");

    group.bench_function("icefalldb_full_scan", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let storage = make_memory_table(20, 10_000).await;
                let reader = Reader::new(&*storage, TABLE).await.unwrap();

                let start = Instant::now();
                let plan = reader.scan().await.unwrap();
                let mut rows = 0usize;
                for rg in &plan.row_groups {
                    let mut stream = reader.read_row_group(rg).await.unwrap();
                    while let Some(batch) = stream.next().await {
                        rows += batch.unwrap().num_rows();
                    }
                }
                black_box(rows);
                total += start.elapsed();
            }
            total
        })
    });

    group.bench_function("raw_parquet_full_read", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let storage = make_memory_table(20, 10_000).await;
                let entries = storage.list(TABLE).await.unwrap();
                let parquet_paths: Vec<String> = entries
                    .into_iter()
                    .filter(|p| p.ends_with(".parquet") && !p.contains("_staging"))
                    .collect();

                let start = Instant::now();
                let mut rows = 0usize;
                for path in parquet_paths {
                    let bytes = storage.read(&path).await.unwrap();
                    let builder =
                        ParquetRecordBatchReaderBuilder::try_new(Bytes::from(bytes)).unwrap();
                    let reader = builder.build().unwrap();
                    for batch in reader {
                        rows += batch.unwrap().num_rows();
                    }
                }
                black_box(rows);
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn tsv_roundtrip_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("tsv_roundtrip");
    group.throughput(Throughput::Elements(10_000));
    group.bench_function("encode_decode_10k", |b| {
        let batch = make_batch(10_000, 0);
        let schema = make_icefalldb_schema(10_000);
        b.iter(|| {
            let encoded = TsvEncoder::encode(&batch);
            let decoded = TsvDecoder::decode(&encoded, &schema).unwrap();
            black_box(decoded);
        })
    });
    group.finish();
}

fn check_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("check_latency");
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("check_100_row_groups", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let storage = make_local_table(dir.path(), 100, 1_000).await;
                let checker = Checker::new(&*storage, TABLE);

                let start = Instant::now();
                let result = checker.check().await.unwrap();
                black_box(result.passed);
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn gc_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("gc_latency");
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("gc_retain_1_after_compaction", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let storage = make_local_table(dir.path(), 50, 100).await;
                let compactor = Compactor::with_options(
                    &*storage,
                    TABLE,
                    CompactionOptions {
                        target_row_group_rows: 5_000,
                        target_row_group_bytes: 1024 * 1024,
                        ..Default::default()
                    },
                );
                compactor.compact().await.unwrap();
                let gc = GarbageCollector::new(&*storage, TABLE, 1);

                let start = Instant::now();
                let result = gc.run().await.unwrap();
                black_box(result.deleted.len());
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

fn iceberg_export_benchmark(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("iceberg_export");
    group.measurement_time(Duration::from_secs(1));

    group.bench_function("export_10_row_groups", |b| {
        b.to_async(&rt).iter_custom(|iters| async move {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let dir = tempfile::tempdir().unwrap();
                let out = dir.path().join("iceberg");
                let storage = make_local_table(dir.path(), 10, 10_000).await;
                let table_root_uri = format!(
                    "file://{}",
                    dir.path().join(TABLE).to_string_lossy().replace('\\', "/")
                );

                let start = Instant::now();
                let metadata_path = icefalldb_core::iceberg::export_table(
                    &*storage,
                    TABLE,
                    &out,
                    None,
                    &table_root_uri,
                )
                .await
                .unwrap();
                black_box(metadata_path);
                total += start.elapsed();
            }
            total
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    ingest_benchmark,
    local_ingest_benchmark,
    query_latency_benchmark,
    predicate_pruning_benchmark,
    compaction_benchmark,
    compaction_time_benchmark,
    metadata_overhead_benchmark,
    tsv_roundtrip_benchmark,
    check_benchmark,
    gc_benchmark,
    iceberg_export_benchmark
);
criterion_main!(benches);
