//! Min/max codec fidelity integration test.
//!
//! Extreme values for each supported type are written to a temp IcefallDB table and
//! then queried through DataFusion. The DataFusion result is compared against
//! a direct full-scan (Arrow compute) oracle.

use std::sync::Arc;

use arrow::array::{
    BinaryArray, Decimal128Array, Float64Array, Int64Array, RecordBatch, StringArray,
    TimestampMicrosecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, TimeUnit};
use datafusion::common::ScalarValue;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{arrow_schema_to_icefalldb, Writer};
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

fn fidelity_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("i64_col", DataType::Int64, true),
        Field::new("u64_col", DataType::UInt64, true),
        Field::new("f64_col", DataType::Float64, true),
        Field::new("decimal_col", DataType::Decimal128(10, 2), true),
        Field::new(
            "timestamp_col",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("binary_col", DataType::Binary, true),
        Field::new("utf8_col", DataType::Utf8, true),
    ]));

    let i64_col = Int64Array::from(vec![
        Some(i64::MIN),
        Some(-5),
        Some(0),
        Some(5),
        Some(i64::MAX),
        None,
    ]);
    let u64_col = UInt64Array::from(vec![
        Some(0),
        Some(1),
        Some((1u64 << 60) + 123),
        Some(1u64 << 63),
        Some(u64::MAX),
        None,
    ]);
    let f64_col = Float64Array::from(vec![
        Some(f64::NEG_INFINITY),
        Some(-1.0),
        Some(f64::NAN),
        Some(1.0),
        Some(f64::INFINITY),
        None,
    ]);
    let decimal_col = Decimal128Array::from(vec![
        Some(-99_999_999i128),
        Some(-123i128),
        Some(0i128),
        Some(123i128),
        Some(99_999_999i128),
        None,
    ])
    .with_precision_and_scale(10, 2)
    .unwrap();
    let timestamp_col = TimestampMicrosecondArray::from(vec![
        Some(-1_700_000_000_000_000i64),
        Some(-100_000_000_000_000i64),
        Some(0),
        Some(100_000_000_000_000i64),
        Some(1_700_000_000_000_000i64),
        None,
    ]);
    let binary_col = BinaryArray::from_opt_vec(vec![
        Some(b"\x00"),
        Some(b"\x7f"),
        Some(b"\xff\xfe"),
        Some(b"a"),
        Some(b"zzz"),
        None,
    ]);
    let utf8_col = StringArray::from(vec![
        Some(""),
        Some("a"),
        Some("Z"),
        Some("é"),
        Some("zzz"),
        None,
    ]);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(i64_col),
            Arc::new(u64_col),
            Arc::new(f64_col),
            Arc::new(decimal_col),
            Arc::new(timestamp_col),
            Arc::new(binary_col),
            Arc::new(utf8_col),
        ],
    )
    .unwrap()
}

async fn create_fidelity_table(root: &std::path::Path) -> Arc<dyn Storage> {
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    let mut schema = arrow_schema_to_icefalldb(fidelity_batch().schema());
    schema.row_group_target_rows = 1000;
    let mut writer = Writer::create(Arc::clone(&storage), "t", schema)
        .await
        .unwrap();
    writer.insert_batch(fidelity_batch()).await.unwrap();
    writer.commit().await.unwrap();
    storage
}

fn expected_min_max(
    array: &arrow::array::ArrayRef,
    data_type: &DataType,
) -> (Option<ScalarValue>, Option<ScalarValue>) {
    use arrow::compute;

    match data_type {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
            (
                compute::min(arr).map(|v| ScalarValue::Int64(Some(v))),
                compute::max(arr).map(|v| ScalarValue::Int64(Some(v))),
            )
        }
        DataType::UInt64 => {
            let arr = array.as_any().downcast_ref::<UInt64Array>().unwrap();
            (
                compute::min(arr).map(|v| ScalarValue::UInt64(Some(v))),
                compute::max(arr).map(|v| ScalarValue::UInt64(Some(v))),
            )
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            (
                compute::min(arr).map(|v| ScalarValue::Float64(Some(v))),
                compute::max(arr).map(|v| ScalarValue::Float64(Some(v))),
            )
        }
        DataType::Decimal128(p, s) => {
            let arr = array.as_any().downcast_ref::<Decimal128Array>().unwrap();
            (
                compute::min(arr).map(|v| ScalarValue::Decimal128(Some(v), *p, *s)),
                compute::max(arr).map(|v| ScalarValue::Decimal128(Some(v), *p, *s)),
            )
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            let arr = array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            (
                compute::min(arr).map(|v| ScalarValue::TimestampMicrosecond(Some(v), tz.clone())),
                compute::max(arr).map(|v| ScalarValue::TimestampMicrosecond(Some(v), tz.clone())),
            )
        }
        DataType::Binary => {
            let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            (
                compute::min_binary(arr).map(|b| ScalarValue::Binary(Some(b.to_vec()))),
                compute::max_binary(arr).map(|b| ScalarValue::Binary(Some(b.to_vec()))),
            )
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
            (
                compute::min_string(arr).map(|s| ScalarValue::Utf8(Some(s.to_string()))),
                compute::max_string(arr).map(|s| ScalarValue::Utf8(Some(s.to_string()))),
            )
        }
        other => panic!("unsupported fidelity type: {other:?}"),
    }
}

fn extract_scalar(batch: &RecordBatch, col: usize) -> ScalarValue {
    let array = batch.column(col);
    match array.data_type() {
        DataType::Int64 => ScalarValue::Int64(
            array
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .next()
                .flatten(),
        ),
        DataType::UInt64 => ScalarValue::UInt64(
            array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .iter()
                .next()
                .flatten(),
        ),
        DataType::Float64 => ScalarValue::Float64(
            array
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .next()
                .flatten(),
        ),
        DataType::Decimal128(p, s) => ScalarValue::Decimal128(
            array
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .unwrap()
                .iter()
                .next()
                .flatten(),
            *p,
            *s,
        ),
        DataType::Timestamp(TimeUnit::Microsecond, tz) => ScalarValue::TimestampMicrosecond(
            array
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .iter()
                .next()
                .flatten(),
            tz.clone(),
        ),
        DataType::Binary => ScalarValue::Binary(
            array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .unwrap()
                .iter()
                .next()
                .flatten()
                .map(|b| b.to_vec()),
        ),
        DataType::Utf8 => ScalarValue::Utf8(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .next()
                .flatten()
                .map(|s| s.to_string()),
        ),
        other => panic!("unexpected result type: {other:?}"),
    }
}

fn scalar_eq(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Float64(Some(a)), ScalarValue::Float64(Some(b))) => {
            if a.is_nan() && b.is_nan() {
                return true;
            }
            a.total_cmp(b).is_eq()
        }
        (ScalarValue::Float32(Some(a)), ScalarValue::Float32(Some(b))) => {
            if a.is_nan() && b.is_nan() {
                return true;
            }
            a.total_cmp(b).is_eq()
        }
        _ => a == b,
    }
}

#[tokio::test]
async fn test_min_max_codec_fidelity() {
    let tmp = tempfile::tempdir().unwrap();
    let storage = create_fidelity_table(tmp.path()).await;

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

    let batch = fidelity_batch();
    let columns = vec![
        ("i64_col", DataType::Int64),
        ("u64_col", DataType::UInt64),
        ("f64_col", DataType::Float64),
        ("decimal_col", DataType::Decimal128(10, 2)),
        (
            "timestamp_col",
            DataType::Timestamp(TimeUnit::Microsecond, None),
        ),
        ("binary_col", DataType::Binary),
        ("utf8_col", DataType::Utf8),
    ];

    for (name, data_type) in &columns {
        let sql = format!("SELECT MIN({name}), MAX({name}) FROM t");
        let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
        assert!(
            !batches.is_empty() && batches[0].num_rows() == 1,
            "{sql} should return one row"
        );

        let actual_min = extract_scalar(&batches[0], 0);
        let actual_max = extract_scalar(&batches[0], 1);

        let array = batch.column_by_name(name).unwrap();
        let (expected_min, expected_max) = expected_min_max(array, data_type);

        assert!(
            scalar_eq(&actual_min, expected_min.as_ref().unwrap()),
            "MIN({name}) mismatch: {actual_min:?} != {expected_min:?}"
        );
        assert!(
            scalar_eq(&actual_max, expected_max.as_ref().unwrap()),
            "MAX({name}) mismatch: {actual_max:?} != {expected_max:?}"
        );
    }
}
