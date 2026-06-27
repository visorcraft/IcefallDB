/// Per-fragment additive aggregate primitives sidecar (`.agg`).
///
/// # Float exactness policy
///
/// - Integer / Decimal columns: `sum` and `sumsq` are EXACT. Integer sums are
///   accumulated into `i128` to avoid i64 overflow across many rows.  On i128
///   overflow (astronomically large inputs) the affected column's [`ColAgg`] is
///   omitted from `cols` so the metadata-aggregate rule falls back to a full scan for that
///   column rather than producing a silently wrong result.
///
/// - Float32 / Float64 columns: floating-point addition is non-associative, so
///   a composed partial-sum is exact only up to reassociation — DataFusion's
///   own multi-partition aggregate exhibits the same behaviour.  Sums are
///   stored as `f64`.  The correctness test for float columns asserts equality
///   WITHIN a tight relative tolerance (`|a - b| <= 1e-9 * max(1, |expected|)`)
///   rather than bit-for-bit equality.
///
/// # Content-hash keying
///
/// `FragmentAggState.content_hash` equals `RowGroupMeta.checksum` for the same
/// fragment.  Because fragments are write-once, a `.agg` sidecar is
/// permanently valid for any fragment whose current checksum matches.
///
/// # What is NOT stored
///
/// `min` and `max` already live in `RowGroupMeta.columns` (`ColumnStats`).
/// This module stores ONLY the new additive primitives `{count_non_null, sum,
/// sumsq}`.  The metadata-aggregate rule composes SUM/AVG/VAR/STDDEV from these and
/// MIN/MAX from `ColumnStats`.
use arrow::array::{
    Array, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array, RecordBatch,
    UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use bytes::Bytes;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector};
use parquet::arrow::ProjectionMask;

use crate::deletion::DeletionVector;
use crate::metadata::RowGroupEntry;
use crate::storage::Storage;
use crate::{IcefallDBError, Result};

/// Additive scalar value stored for each numeric column.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "t", content = "v")]
pub enum AggScalar {
    /// Exact integer value (i128 covers all integer column widths without
    /// overflow for realistic fragment sizes).
    Int(i128),
    /// f64 approximation for Float32 / Float64 columns.
    //
    // JSON cannot represent NaN/±Inf as bare numbers, so the f64 content is
    // serialized through `f64_serde` as the strings `"NaN"`, `"+Inf"`, `"-Inf"`
    // or a regular number. Deserialization also accepts the legacy `null`
    // representation (produced by older sidecars for NaN) and maps it back to
    // `f64::NAN` so existing tables keep working.
    Float(#[serde(with = "f64_serde")] f64),
    /// The column contained only null values; no numeric contribution.
    Null,
}

mod f64_serde {
    //! Custom serialization for `f64` aggregate partials so that NaN and
    //! infinities survive a JSON round-trip. Older `.agg` sidecars may contain
    //! `null` for NaN (the default serde_json behaviour); deserialization
    //! accepts that as a backward-compatible alias for `f64::NAN`.

    use serde::{Deserializer, Serializer};

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        if v.is_nan() {
            s.serialize_str("NaN")
        } else if v.is_infinite() {
            s.serialize_str(if v.is_sign_positive() { "+Inf" } else { "-Inf" })
        } else {
            s.serialize_f64(*v)
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = f64;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a float number or one of 'NaN', '+Inf', '-Inf'")
            }

            fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
                Ok(v)
            }

            fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
                Ok(v as f64)
            }

            fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
                Ok(v as f64)
            }

            fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
                match v {
                    "NaN" => Ok(f64::NAN),
                    "+Inf" => Ok(f64::INFINITY),
                    "-Inf" => Ok(f64::NEG_INFINITY),
                    _ => Err(E::custom(format!("unknown float token: {v}"))),
                }
            }

            fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(f64::NAN)
            }

            fn visit_none<E: serde::de::Error>(self) -> Result<Self::Value, E> {
                Ok(f64::NAN)
            }
        }

        d.deserialize_any(V)
    }
}

/// Per-column additive aggregate primitives.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColAgg {
    /// Number of non-null rows in this column.
    pub count_non_null: u64,
    /// Sum of non-null values.
    pub sum: AggScalar,
    /// Sum of squares of non-null values.
    pub sumsq: AggScalar,
    /// Physical row offset (0-based within the Parquet file) of the row holding
    /// the column's minimum value, computed at write time.  `None`
    /// on legacy fragments or when all values are null.  Not tracked for float
    /// columns (float guard is unconditional).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_off: Option<u32>,
    /// Physical row offset of the row holding the column's maximum value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_off: Option<u32>,
    /// Resolved live minimum after accounting for the current deletion vector.
    /// Set by `scan_internal` when preloading the dirty-fragment partial.
    /// `None` when unresolved (old .agg file, float column, or read error).
    /// Never serialized — computed at scan time.
    #[serde(skip)]
    pub live_min_json: Option<serde_json::Value>,
    /// Resolved live maximum.  Same rules as `live_min_json`.
    #[serde(skip)]
    pub live_max_json: Option<serde_json::Value>,
}

/// Aggregate state for one fragment, keyed by the fragment's content hash.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FragmentAggState {
    /// Stable fragment identifier (matches `RowGroupEntry.fragment_id`).
    pub fragment_id: u64,
    /// Content hash of the Parquet data file (`RowGroupMeta.checksum`).
    /// Must equal the fragment's `RowGroupMeta.checksum` for this `.agg` to
    /// be trusted by the metadata-aggregate rule.
    pub content_hash: String,
    /// Per-column additive primitives.  Only numeric columns appear here; the
    /// The metadata-aggregate rule falls back to a full scan for any column absent from this map.
    /// BTreeMap gives stable serialization order for deterministic JSON output.
    pub cols: BTreeMap<String, ColAgg>,
    /// Per-group additive primitives, populated when the table declares
    /// `agg_group_keys`.  `None` for tables that do not declare
    /// grouping keys, when the cardinality guard tripped (> [`MAX_DECLARED_GROUPS`]
    /// distinct values), or for legacy fragments.
    /// Omitted from JSON when absent so existing `.agg` sidecars round-trip
    /// unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grouped: Option<GroupedPartials>,
    /// Per-column serialized CPC sketch bytes for distinct-count estimation.
    ///
    /// Populated only when the `sketches` feature is enabled at write time.
    /// The field is ALWAYS present in both builds (not `#[cfg]`-gated) so that
    /// `.agg` sidecars written with the feature can be read by a non-feature
    /// build without error (the bytes are ignored).  Omitted from JSON when
    /// absent so existing sidecars round-trip unchanged.
    ///
    /// Keyed by column name → raw CPC sketch bytes (`CpcSketch::serialize`).
    /// Deletion-aware A-not-B (Theta) support is not yet implemented; for now dirty
    /// fragments fall back to a full scan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub distinct: Option<std::collections::BTreeMap<String, Vec<u8>>>,
    /// Per-column serialized T-Digest sketch bytes for approximate quantile estimation.
    ///
    /// Populated only when the `sketches` feature is enabled at write time.
    /// The field is ALWAYS present in both builds (not `#[cfg]`-gated) so that
    /// `.agg` sidecars written with the feature can be read by a non-feature
    /// build without error (the bytes are ignored).  Omitted from JSON when
    /// absent so existing sidecars round-trip unchanged.
    ///
    /// Keyed by column name → raw T-Digest bytes (`TDigestMut::serialize`).
    /// T-Digest k=200 (default): rank error ~1-2% for smooth distributions;
    /// we assert within 3%.  Only numeric (Float32/Float64/Int*) columns are
    /// sketched.  Dirty fragments fall back to a full scan (deferred).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantile: Option<std::collections::BTreeMap<String, Vec<u8>>>,
}

/// Compute a [`FragmentAggState`] from an Arrow [`RecordBatch`].
///
/// Only numeric columns (Int8/16/32/64, UInt8/16/32/64, Float32, Float64)
/// produce a [`ColAgg`].  Non-numeric columns are silently skipped; the metadata-aggregate
/// rule falls back for those columns.
///
/// Integer sums use `i128` with checked arithmetic.  On overflow the column is
/// omitted (the metadata-aggregate rule falls back) rather than silently wrapping.
///
/// `content_hash` must be the `RowGroupMeta.checksum` computed for the same
/// Parquet bytes that were derived from `batch`.
pub fn compute_agg_state(
    fragment_id: u64,
    content_hash: String,
    batch: &RecordBatch,
) -> Result<FragmentAggState> {
    let schema = batch.schema();
    let mut cols = BTreeMap::new();

    for (field, column) in schema.fields().iter().zip(batch.columns().iter()) {
        let col_name = field.name().clone();
        let maybe_col_agg = match field.data_type() {
            DataType::Int8 => {
                let arr = column.as_any().downcast_ref::<Int8Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    arr.is_null(i)
                        .then_some(())
                        .map_or(Some(arr.value(i) as i128), |_| None)
                })
            }
            DataType::Int16 => {
                let arr = column.as_any().downcast_ref::<Int16Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::Int32 => {
                let arr = column.as_any().downcast_ref::<Int32Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::Int64 => {
                let arr = column.as_any().downcast_ref::<Int64Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::UInt8 => {
                let arr = column.as_any().downcast_ref::<UInt8Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::UInt16 => {
                let arr = column.as_any().downcast_ref::<UInt16Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::UInt32 => {
                let arr = column.as_any().downcast_ref::<UInt32Array>().unwrap();
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::UInt64 => {
                let arr = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                // u64 fits in i128 (max u64 = 1.8e19, max i128 = 1.7e38).
                compute_signed_int_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as i128)
                    }
                })
            }
            DataType::Float32 => {
                let arr = column.as_any().downcast_ref::<Float32Array>().unwrap();
                Some(compute_float_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i) as f64)
                    }
                }))
            }
            DataType::Float64 => {
                let arr = column.as_any().downcast_ref::<Float64Array>().unwrap();
                Some(compute_float_col(arr.len(), arr.null_count(), |i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                }))
            }
            // Non-numeric types (Utf8, Bool, Timestamp, …) are skipped.
            _ => None,
        };

        if let Some(col_agg) = maybe_col_agg {
            cols.insert(col_name, col_agg);
        }
    }

    #[cfg(feature = "sketches")]
    let distinct = {
        use datasketches::cpc::CpcSketch;
        use std::collections::BTreeMap;

        // CPC sketch with lg_k=12 → nominal error ~1.04% at 50k cardinality.
        const LG_K: u8 = 12;
        let mut sketch_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        for (field, column) in schema.fields().iter().zip(batch.columns().iter()) {
            let col_name = field.name().clone();
            let len = column.len();
            let null_count = column.null_count();
            if null_count == len {
                // All-null column: skip.
                continue;
            }
            let mut sketch = CpcSketch::new(LG_K);
            use arrow::datatypes::DataType as DT;
            match field.data_type() {
                DT::Int8 => {
                    let arr = column.as_any().downcast_ref::<Int8Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as i64);
                        }
                    }
                }
                DT::Int16 => {
                    let arr = column.as_any().downcast_ref::<Int16Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as i64);
                        }
                    }
                }
                DT::Int32 => {
                    let arr = column.as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as i64);
                        }
                    }
                }
                DT::Int64 => {
                    let arr = column.as_any().downcast_ref::<Int64Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i));
                        }
                    }
                }
                DT::UInt8 => {
                    let arr = column.as_any().downcast_ref::<UInt8Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as u64);
                        }
                    }
                }
                DT::UInt16 => {
                    let arr = column.as_any().downcast_ref::<UInt16Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as u64);
                        }
                    }
                }
                DT::UInt32 => {
                    let arr = column.as_any().downcast_ref::<UInt32Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as u64);
                        }
                    }
                }
                DT::UInt64 => {
                    let arr = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i));
                        }
                    }
                }
                DT::Float32 => {
                    let arr = column.as_any().downcast_ref::<Float32Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            // Use bit-cast to u32 for hashing — NaN-safe distinct counting.
                            sketch.update(arr.value(i).to_bits() as u64);
                        }
                    }
                }
                DT::Float64 => {
                    let arr = column.as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i).to_bits());
                        }
                    }
                }
                DT::Utf8 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<arrow::array::StringArray>()
                        .unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i));
                        }
                    }
                }
                DT::LargeUtf8 => {
                    let arr = column
                        .as_any()
                        .downcast_ref::<arrow::array::LargeStringArray>()
                        .unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i));
                        }
                    }
                }
                _ => continue, // unsupported type → no sketch
            }
            sketch_map.insert(col_name, sketch.serialize());
        }
        if sketch_map.is_empty() {
            None
        } else {
            Some(sketch_map)
        }
    };

    #[cfg(not(feature = "sketches"))]
    let distinct: Option<std::collections::BTreeMap<String, Vec<u8>>> = None;

    // ── per-column T-Digest sketch for approx_percentile_cont ──────────
    #[cfg(feature = "sketches")]
    let quantile = {
        use datasketches::tdigest::TDigestMut;
        use std::collections::BTreeMap;

        // T-Digest k=200 (default): rank error ~1-2% for smooth distributions.
        let mut sketch_map: BTreeMap<String, Vec<u8>> = BTreeMap::new();

        for (field, column) in schema.fields().iter().zip(batch.columns().iter()) {
            let col_name = field.name().clone();
            let len = column.len();
            let null_count = column.null_count();
            if null_count == len {
                // All-null column: skip.
                continue;
            }
            let mut sketch = TDigestMut::default(); // k=200
            use arrow::datatypes::DataType as DT;
            match field.data_type() {
                DT::Int8 => {
                    let arr = column.as_any().downcast_ref::<Int8Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::Int16 => {
                    let arr = column.as_any().downcast_ref::<Int16Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::Int32 => {
                    let arr = column.as_any().downcast_ref::<Int32Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::Int64 => {
                    let arr = column.as_any().downcast_ref::<Int64Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::UInt8 => {
                    let arr = column.as_any().downcast_ref::<UInt8Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::UInt16 => {
                    let arr = column.as_any().downcast_ref::<UInt16Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::UInt32 => {
                    let arr = column.as_any().downcast_ref::<UInt32Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::UInt64 => {
                    let arr = column.as_any().downcast_ref::<UInt64Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i) as f64);
                        }
                    }
                }
                DT::Float32 => {
                    let arr = column.as_any().downcast_ref::<Float32Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            let v = arr.value(i) as f64;
                            // T-Digest silently drops NaN/Inf.
                            sketch.update(v);
                        }
                    }
                }
                DT::Float64 => {
                    let arr = column.as_any().downcast_ref::<Float64Array>().unwrap();
                    for i in 0..len {
                        if !arr.is_null(i) {
                            sketch.update(arr.value(i));
                        }
                    }
                }
                _ => continue, // non-numeric type → no quantile sketch
            }
            if !sketch.is_empty() {
                sketch_map.insert(col_name, sketch.serialize());
            }
        }
        if sketch_map.is_empty() {
            None
        } else {
            Some(sketch_map)
        }
    };

    #[cfg(not(feature = "sketches"))]
    let quantile: Option<std::collections::BTreeMap<String, Vec<u8>>> = None;

    Ok(FragmentAggState {
        fragment_id,
        content_hash,
        cols,
        grouped: None,
        distinct,
        quantile,
    })
}

/// Serialize a [`FragmentAggState`] to JSON bytes suitable for writing to a
/// `.agg` sidecar file.
pub fn serialize_agg_state(state: &FragmentAggState) -> Result<Vec<u8>> {
    serde_json::to_vec(state).map_err(IcefallDBError::Serialization)
}

/// Deserialize a [`FragmentAggState`] from JSON bytes read from a `.agg`
/// sidecar file.
pub fn deserialize_agg_state(bytes: &[u8]) -> Result<FragmentAggState> {
    serde_json::from_slice(bytes).map_err(IcefallDBError::Serialization)
}

// ── Deletion retraction ──────────────────────────────────────────────────────

/// Per-column aggregate contribution of the DELETED rows (used for retraction).
///
/// Same layout as `FragmentAggState.cols` but represents only the deleted rows.
#[derive(Debug, Clone, PartialEq)]
pub struct DeletedContribution {
    pub cols: BTreeMap<String, ColAgg>,
}

/// Compute the aggregate contribution of only the DELETED rows in a fragment.
///
/// Reads the fragment's Parquet file selecting ONLY the deleted offsets via
/// [`RowSelection`] built from `dv.iter()`, then folds each numeric column into
/// a [`ColAgg`].  Uses checked i128 arithmetic for integers; on overflow the
/// column is omitted so the caller can fall back.
///
/// `numeric_cols` is the list of column names to compute (typically the keys of
/// the full [`FragmentAggState::cols`]).  Non-numeric or absent columns are
/// silently skipped.
pub async fn deleted_contribution(
    storage: &dyn Storage,
    table: &str,
    entry: &RowGroupEntry,
    dv: &DeletionVector,
    numeric_cols: &[String],
) -> Result<DeletedContribution> {
    // Build RowSelection selecting only the deleted row offsets.
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut pos: usize = 0;
    for offset in dv.iter() {
        let off = offset as usize;
        if off > pos {
            selectors.push(RowSelector::skip(off - pos));
        }
        selectors.push(RowSelector::select(1));
        pos = off + 1;
    }
    if selectors.is_empty() {
        return Ok(DeletedContribution {
            cols: BTreeMap::new(),
        });
    }
    let row_selection = RowSelection::from(selectors);

    // Read the Parquet data file.
    let data_path = format!("{}/{}", table, entry.data);
    let data = storage.read(&data_path).await?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data))
        .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;

    let arrow_schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();

    // Build ProjectionMask for only the requested numeric columns.
    let mut col_indices: Vec<usize> = Vec::new();
    let mut col_names_present: Vec<String> = Vec::new();
    for col_name in numeric_cols {
        if let Ok(idx) = arrow_schema.index_of(col_name) {
            col_indices.push(idx);
            col_names_present.push(col_name.clone());
        }
    }
    if col_indices.is_empty() {
        return Ok(DeletedContribution {
            cols: BTreeMap::new(),
        });
    }
    let mask = ProjectionMask::roots(&parquet_schema, col_indices);

    let reader = builder
        .with_projection(mask)
        .with_row_selection(row_selection)
        .build()
        .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;

    // Collect all record batches (only the deleted rows, projected columns).
    let batches: Vec<RecordBatch> = reader
        .map(|r| r.map_err(|e| IcefallDBError::ParquetDecode(e.to_string())))
        .collect::<Result<Vec<_>>>()?;

    if batches.is_empty() {
        return Ok(DeletedContribution {
            cols: BTreeMap::new(),
        });
    }

    // The projected batches contain only col_names_present, look them up by name.
    let projected_schema = batches[0].schema();
    let mut cols = BTreeMap::new();

    for col_name in &col_names_present {
        let field_idx = match projected_schema.index_of(col_name) {
            Ok(i) => i,
            Err(_) => continue,
        };
        let field = projected_schema.field(field_idx);

        let maybe_col_agg: Option<ColAgg> = match field.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => {
                let mut count: u64 = 0;
                let mut sum: i128 = 0;
                let mut sumsq: i128 = 0;
                let mut overflow = false;
                let mut all_null = true;

                'int_batches: for batch in &batches {
                    let col = batch.column(field_idx);
                    let len = col.len();
                    if col.null_count() < len {
                        all_null = false;
                    }
                    macro_rules! fold_int_arr {
                        ($ArrayType:ty) => {{
                            use arrow::array::Array as _;
                            let arr = col.as_any().downcast_ref::<$ArrayType>().unwrap();
                            for i in 0..len {
                                if arr.is_null(i) {
                                    continue;
                                }
                                let v = arr.value(i) as i128;
                                count += 1;
                                match sum.checked_add(v) {
                                    Some(s) => sum = s,
                                    None => {
                                        overflow = true;
                                        break 'int_batches;
                                    }
                                }
                                match v.checked_mul(v).and_then(|sq| sumsq.checked_add(sq)) {
                                    Some(sq) => sumsq = sq,
                                    None => {
                                        overflow = true;
                                        break 'int_batches;
                                    }
                                }
                            }
                        }};
                    }
                    match field.data_type() {
                        DataType::Int8 => fold_int_arr!(arrow::array::Int8Array),
                        DataType::Int16 => fold_int_arr!(arrow::array::Int16Array),
                        DataType::Int32 => fold_int_arr!(arrow::array::Int32Array),
                        DataType::Int64 => fold_int_arr!(arrow::array::Int64Array),
                        DataType::UInt8 => fold_int_arr!(arrow::array::UInt8Array),
                        DataType::UInt16 => fold_int_arr!(arrow::array::UInt16Array),
                        DataType::UInt32 => fold_int_arr!(arrow::array::UInt32Array),
                        DataType::UInt64 => fold_int_arr!(arrow::array::UInt64Array),
                        _ => {}
                    }
                }

                if overflow {
                    // Fail closed: return Err so scan_internal sets agg_state = None
                    // and the rule falls back to a real scan instead of silently
                    // retaining the over-counted full value.
                    return Err(IcefallDBError::ParquetDecode(
                        "i128 overflow in deleted_contribution".to_string(),
                    ));
                } else if all_null || count == 0 {
                    Some(ColAgg {
                        count_non_null: 0,
                        sum: AggScalar::Null,
                        sumsq: AggScalar::Null,
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    })
                } else {
                    Some(ColAgg {
                        count_non_null: count,
                        sum: AggScalar::Int(sum),
                        sumsq: AggScalar::Int(sumsq),
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    })
                }
            }
            DataType::Float32 | DataType::Float64 => {
                let mut count: u64 = 0;
                let mut sum: f64 = 0.0;
                let mut sumsq: f64 = 0.0;
                let mut all_null = true;

                for batch in &batches {
                    let col = batch.column(field_idx);
                    let len = col.len();
                    if col.null_count() < len {
                        all_null = false;
                    }
                    match field.data_type() {
                        DataType::Float32 => {
                            let arr = col.as_any().downcast_ref::<Float32Array>().unwrap();
                            for i in 0..len {
                                if arr.is_null(i) {
                                    continue;
                                }
                                let v = arr.value(i) as f64;
                                count += 1;
                                sum += v;
                                sumsq += v * v;
                            }
                        }
                        DataType::Float64 => {
                            let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                            for i in 0..len {
                                if arr.is_null(i) {
                                    continue;
                                }
                                let v = arr.value(i);
                                count += 1;
                                sum += v;
                                sumsq += v * v;
                            }
                        }
                        _ => {}
                    }
                }

                if all_null || count == 0 {
                    Some(ColAgg {
                        count_non_null: 0,
                        sum: AggScalar::Null,
                        sumsq: AggScalar::Null,
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    })
                } else {
                    Some(ColAgg {
                        count_non_null: count,
                        sum: AggScalar::Float(sum),
                        sumsq: AggScalar::Float(sumsq),
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    })
                }
            }
            _ => None,
        };

        if let Some(agg) = maybe_col_agg {
            cols.insert(col_name.clone(), agg);
        }
    }

    Ok(DeletedContribution { cols })
}

/// Subtract the contribution of deleted rows from a full fragment partial.
///
/// Returns a new [`FragmentAggState`] with the deletion applied.  For each
/// column present in BOTH `full` and `del`:
/// - `count_non_null` is decremented by `del.count_non_null` (saturating).
/// - `sum` and `sumsq` are subtracted (Int i128 checked sub, Float f64 sub).
///
/// If the arithmetic cannot be done exactly (variant mismatch or integer
/// underflow), the column is REMOVED from the result so the rule falls back for
/// that column.
pub fn retract(full: &FragmentAggState, del: &DeletedContribution) -> FragmentAggState {
    let mut cols = full.cols.clone();
    let mut to_remove: Vec<String> = Vec::new();

    for (col_name, del_agg) in &del.cols {
        let Some(full_agg) = cols.get_mut(col_name) else {
            continue;
        };

        let new_count = full_agg
            .count_non_null
            .saturating_sub(del_agg.count_non_null);

        let new_sum = match (&full_agg.sum, &del_agg.sum) {
            (AggScalar::Int(fs), AggScalar::Int(ds)) => match fs.checked_sub(*ds) {
                Some(v) => AggScalar::Int(v),
                None => {
                    to_remove.push(col_name.clone());
                    continue;
                }
            },
            (AggScalar::Float(fs), AggScalar::Float(ds)) => AggScalar::Float(fs - ds),
            (AggScalar::Null, AggScalar::Null) => AggScalar::Null,
            // Deleted rows were all null; full partial is unchanged.
            (full_sum, AggScalar::Null) => full_sum.clone(),
            _ => {
                // Variant mismatch: drop column for safety.
                to_remove.push(col_name.clone());
                continue;
            }
        };

        let new_sumsq = match (&full_agg.sumsq, &del_agg.sumsq) {
            (AggScalar::Int(fq), AggScalar::Int(dq)) => match fq.checked_sub(*dq) {
                Some(v) => AggScalar::Int(v),
                None => {
                    to_remove.push(col_name.clone());
                    continue;
                }
            },
            (AggScalar::Float(fq), AggScalar::Float(dq)) => AggScalar::Float(fq - dq),
            (AggScalar::Null, AggScalar::Null) => AggScalar::Null,
            (full_sumsq, AggScalar::Null) => full_sumsq.clone(),
            _ => {
                to_remove.push(col_name.clone());
                continue;
            }
        };

        let saved_min_off = full_agg.min_off;
        let saved_max_off = full_agg.max_off;
        *full_agg = ColAgg {
            count_non_null: new_count,
            sum: new_sum,
            sumsq: new_sumsq,
            min_off: saved_min_off,
            max_off: saved_max_off,
            live_min_json: None,
            live_max_json: None,
        };
    }

    for col_name in to_remove {
        cols.remove(&col_name);
    }

    FragmentAggState {
        fragment_id: full.fragment_id,
        content_hash: full.content_hash.clone(),
        cols,
        // grouped partials are retracted separately via retract_grouped.
        grouped: full.grouped.clone(),
        // Distinct sketches are NOT retracted for dirty fragments.
        // A dirty fragment → the rule falls back to an exact scan.
        // Deletion-aware sketch retraction (Theta A-not-B) is not yet implemented.
        distinct: None,
        // Quantile sketches are NOT retracted for dirty fragments.
        // A dirty fragment → the rule falls back to an exact scan.
        // Deletion-aware quantile sketch update is not yet implemented.
        quantile: None,
    }
}

// ── extremum tracking under deletions ───────────────────────────────────

/// Which extremum (minimum or maximum) to test or recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtremumKind {
    Min,
    Max,
}

/// Returns `true` when the row holding the cached extremum is in the deletion
/// vector — meaning the extremal row was deleted and the cached value is no
/// longer valid.
///
/// Returns `false` when:
/// - The extremal offset is NOT in the DV (survivor-extremum is valid).
/// - `min_off`/`max_off` is `None` (graceful fallback for old `.agg` files).
pub fn extremum_deleted(col_agg: &ColAgg, dv: &DeletionVector, kind: ExtremumKind) -> bool {
    let off = match kind {
        ExtremumKind::Min => col_agg.min_off,
        ExtremumKind::Max => col_agg.max_off,
    };
    match off {
        Some(offset) => dv.contains(offset),
        None => false, // unknown offset → not deleted (graceful fallback)
    }
}

/// Read the surviving rows (complement of `dv`) for `col_name` in the fragment
/// referenced by `entry`, then return the min or max as a [`serde_json::Value`].
///
/// Uses a [`RowSelection`] that selects LIVE rows (skips deleted ones).
///
/// Returns `Ok(None)` when there are no surviving non-null values.
/// Returns `Err(_)` on I/O or parse failure (caller should fall back).
pub async fn scoped_recompute_extremum(
    storage: &dyn Storage,
    table: &str,
    entry: &RowGroupEntry,
    dv: &DeletionVector,
    total_rows: usize,
    col_name: &str,
    kind: ExtremumKind,
) -> Result<Option<serde_json::Value>> {
    // Build a RowSelection that selects LIVE rows (complement of DV).
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut pos: usize = 0;
    for offset in dv.iter() {
        let off = offset as usize;
        if off > pos {
            selectors.push(RowSelector::select(off - pos));
        }
        selectors.push(RowSelector::skip(1));
        pos = off + 1;
    }
    if pos < total_rows {
        selectors.push(RowSelector::select(total_rows - pos));
    }
    if selectors.is_empty() {
        return Ok(None);
    }
    let row_selection = RowSelection::from(selectors);

    // Read the Parquet data file.
    let data_path = format!("{}/{}", table, entry.data);
    let data = storage.read(&data_path).await?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data))
        .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;

    let arrow_schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();

    let col_idx = arrow_schema
        .index_of(col_name)
        .map_err(|_| IcefallDBError::ParquetDecode(format!("column {col_name} not found")))?;

    let field = arrow_schema.field(col_idx).clone();

    // Only handle integer columns — float guard is unconditional.
    let is_integer = matches!(
        field.data_type(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    );
    if !is_integer {
        return Ok(None);
    }

    let mask = ProjectionMask::roots(&parquet_schema, vec![col_idx]);
    let reader = builder
        .with_projection(mask)
        .with_row_selection(row_selection)
        .build()
        .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;

    let batches: Vec<RecordBatch> = reader
        .map(|r| r.map_err(|e| IcefallDBError::ParquetDecode(e.to_string())))
        .collect::<Result<Vec<_>>>()?;

    if batches.is_empty() {
        return Ok(None);
    }

    // Find min or max across all survivor batches.
    let proj_schema = batches[0].schema();
    let field_idx = proj_schema.index_of(col_name).map_err(|_| {
        IcefallDBError::ParquetDecode(format!("column {col_name} not in projected schema"))
    })?;
    let proj_field = proj_schema.field(field_idx).clone();

    let mut current: Option<i128> = None;

    for batch in &batches {
        let col = batch.column(field_idx);
        let len = col.len();

        macro_rules! fold_batch {
            ($ArrayType:ty) => {{
                let arr = col.as_any().downcast_ref::<$ArrayType>().unwrap();
                for i in 0..len {
                    if arr.is_null(i) {
                        continue;
                    }
                    let v = arr.value(i) as i128;
                    current = Some(match current {
                        None => v,
                        Some(c) => match kind {
                            ExtremumKind::Min => c.min(v),
                            ExtremumKind::Max => c.max(v),
                        },
                    });
                }
            }};
        }

        match proj_field.data_type() {
            DataType::Int8 => fold_batch!(Int8Array),
            DataType::Int16 => fold_batch!(Int16Array),
            DataType::Int32 => fold_batch!(Int32Array),
            DataType::Int64 => fold_batch!(Int64Array),
            DataType::UInt8 => fold_batch!(UInt8Array),
            DataType::UInt16 => fold_batch!(UInt16Array),
            DataType::UInt32 => fold_batch!(UInt32Array),
            DataType::UInt64 => fold_batch!(UInt64Array),
            _ => return Ok(None),
        }
    }

    let Some(val) = current else {
        return Ok(None);
    };

    // Convert i128 back to a JSON number.  Use an unsigned representation when
    // the value exceeds i64::MAX (only reachable for UInt64 survivor extrema).
    // Writing `val as i64` for those values wraps negative, causing
    // `json_to_scalar_value` / `to_u64` to reject the number and silently
    // disable the fast path.
    let json_val = if val >= 0 && val <= i64::MAX as i128 {
        serde_json::Value::Number(serde_json::Number::from(val as i64))
    } else if val > i64::MAX as i128 && val <= u64::MAX as i128 {
        serde_json::Value::Number(serde_json::Number::from(val as u64))
    } else {
        // Out of range for any supported integer JSON encoding — should not
        // happen in practice; fall back gracefully.
        return Ok(None);
    };
    Ok(Some(json_val))
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Compute a [`ColAgg`] for a column whose values are accessed as `i128`.
///
/// `get_value(i)` returns `None` for null rows, `Some(v)` for non-null rows.
///
/// Returns `None` on i128 overflow so the caller can omit the column and let
/// the metadata-aggregate rule fall back to a full scan.
fn compute_signed_int_col(
    len: usize,
    null_count: usize,
    get_value: impl Fn(usize) -> Option<i128>,
) -> Option<ColAgg> {
    if null_count == len {
        // All nulls — no arithmetic needed.
        return Some(ColAgg {
            count_non_null: 0,
            sum: AggScalar::Null,
            sumsq: AggScalar::Null,
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        });
    }

    let mut count_non_null: u64 = 0;
    let mut sum: i128 = 0;
    let mut sumsq: i128 = 0;
    let mut min_val: Option<i128> = None;
    let mut max_val: Option<i128> = None;
    let mut min_off_res: Option<u32> = None;
    let mut max_off_res: Option<u32> = None;

    for i in 0..len {
        if let Some(v) = get_value(i) {
            count_non_null += 1;
            sum = sum.checked_add(v)?;
            let sq = v.checked_mul(v)?;
            sumsq = sumsq.checked_add(sq)?;

            match min_val {
                None => {
                    min_val = Some(v);
                    min_off_res = Some(i as u32);
                }
                Some(m) if v < m => {
                    min_val = Some(v);
                    min_off_res = Some(i as u32);
                }
                _ => {}
            }
            match max_val {
                None => {
                    max_val = Some(v);
                    max_off_res = Some(i as u32);
                }
                Some(m) if v > m => {
                    max_val = Some(v);
                    max_off_res = Some(i as u32);
                }
                _ => {}
            }
        }
    }

    Some(ColAgg {
        count_non_null,
        sum: AggScalar::Int(sum),
        sumsq: AggScalar::Int(sumsq),
        min_off: min_off_res,
        max_off: max_off_res,
        live_min_json: None,
        live_max_json: None,
    })
}

/// Compute a [`ColAgg`] for a float column.  Values are always `Some` when
/// the row is non-null; `None` signals a null row.
fn compute_float_col(
    len: usize,
    null_count: usize,
    get_value: impl Fn(usize) -> Option<f64>,
) -> ColAgg {
    if null_count == len {
        return ColAgg {
            count_non_null: 0,
            sum: AggScalar::Null,
            sumsq: AggScalar::Null,
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        };
    }

    let mut count_non_null: u64 = 0;
    let mut sum: f64 = 0.0;
    let mut sumsq: f64 = 0.0;

    for i in 0..len {
        if let Some(v) = get_value(i) {
            count_non_null += 1;
            sum += v;
            sumsq += v * v;
        }
    }

    ColAgg {
        count_non_null,
        sum: AggScalar::Float(sum),
        sumsq: AggScalar::Float(sumsq),
        min_off: None,
        max_off: None,
        live_min_json: None,
        live_max_json: None,
    }
}

// ── per-group partials (GroupedPartials) ──────────────────────────────

/// Maximum number of distinct key values allowed per fragment before the
/// grouped partials are discarded.  When a fragment has more distinct keys
/// than this cap the `grouped` field is set to `None` and the rule falls back
/// to a full scan — this prevents unbounded maps for mis-declared high-
/// cardinality columns.  4096 was chosen to comfortably cover typical
/// low-cardinality dimensions (country, status, category, …) while bounding
/// memory: 4096 groups × a few measure columns × ~80 bytes/ColAgg ≈ a few MB.
pub const MAX_DECLARED_GROUPS: usize = 4096;

/// Stable, serialisable, hashable group-key value.
///
/// # Representation choice
///
/// We store the group key as its **canonical JSON string** (produced by
/// `serde_json::to_string`).  This gives us:
/// - A single, simple type for the `BTreeMap` key (no complex enum matching).
/// - Natural JSON round-tripping: `null`, `"text"`, `42`, `true` each
///   serialise to their JSON form and deserialise back with
///   `serde_json::from_str`.
/// - Stable ordering: BTreeMap keys are compared lexicographically; the
///   canonical JSON ordering (null < numbers < strings) is acceptable for
///   sidecar determinism.
///
/// Supported Arrow key column types:
/// - Any integer (Int8/16/32/64, UInt8/16/32/64) → JSON number.
/// - Utf8 / LargeUtf8 → JSON string (quoted).
/// - Boolean → JSON `true`/`false`.
/// - Null cell → JSON `null` (all NULL-key rows land in one group — SQL
///   `GROUP BY` semantics).
fn arrow_scalar_to_group_key_string(col: &dyn arrow::array::Array, row: usize) -> Option<String> {
    use arrow::array::{
        BooleanArray, Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, StringArray,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow::datatypes::DataType as DT;
    let dt = col.data_type().clone();
    if col.is_null(row) {
        return Some("null".to_string());
    }
    let json_val: serde_json::Value = match dt {
        DT::Int8 => {
            let v = col.as_any().downcast_ref::<Int8Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::Int16 => {
            let v = col.as_any().downcast_ref::<Int16Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::Int32 => {
            let v = col.as_any().downcast_ref::<Int32Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::Int64 => {
            let v = col.as_any().downcast_ref::<Int64Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::UInt8 => {
            let v = col.as_any().downcast_ref::<UInt8Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::UInt16 => {
            let v = col.as_any().downcast_ref::<UInt16Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::UInt32 => {
            let v = col.as_any().downcast_ref::<UInt32Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::UInt64 => {
            let v = col.as_any().downcast_ref::<UInt64Array>()?.value(row);
            serde_json::Value::Number(v.into())
        }
        DT::Utf8 => {
            let v = col.as_any().downcast_ref::<StringArray>()?.value(row);
            serde_json::Value::String(v.to_string())
        }
        DT::LargeUtf8 => {
            let v = col.as_any().downcast_ref::<LargeStringArray>()?.value(row);
            serde_json::Value::String(v.to_string())
        }
        DT::Boolean => {
            let v = col.as_any().downcast_ref::<BooleanArray>()?.value(row);
            serde_json::Value::Bool(v)
        }
        _ => return None, // unsupported key type → skip grouped for this batch
    };
    Some(serde_json::to_string(&json_val).unwrap_or_else(|_| "null".to_string()))
}

/// Per-group additive aggregate primitives for one fragment.
///
/// `groups` maps a canonical JSON-string key (see [`arrow_scalar_to_group_key_string`])
/// to a per-group map of column-name → [`ColAgg`].  Using a `BTreeMap` gives
/// deterministic JSON serialisation order.
///
/// Only numeric measure columns appear in each group's map; the warm-GROUP-BY rule
/// falls back to a full scan for any measure column absent here.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GroupedPartials {
    /// Name of the declared grouping-key column.
    pub key_col: String,
    /// Map from canonical-JSON key string → (measure column name → ColAgg).
    pub groups: BTreeMap<String, BTreeMap<String, ColAgg>>,
}

impl FragmentAggState {
    /// Return the new field — kept as a method to avoid
    /// touching the public struct literal syntax in callers.
    pub fn grouped(&self) -> Option<&GroupedPartials> {
        self.grouped.as_ref()
    }
}

/// Compute a [`FragmentAggState`] from an Arrow [`RecordBatch`], optionally
/// building per-group partials when `key_col` is provided.
///
/// `key_col` should be `schema.agg_group_keys.first()` (or `None`).  When the
/// key column is absent from the batch, or has > [`MAX_DECLARED_GROUPS`]
/// distinct values, `grouped` is set to `None`.
pub fn compute_agg_state_with_key(
    fragment_id: u64,
    content_hash: String,
    batch: &RecordBatch,
    key_col: Option<&str>,
) -> Result<FragmentAggState> {
    // Compute the ungrouped cols as before.
    let mut base = compute_agg_state(fragment_id, content_hash, batch)?;

    let Some(key_name) = key_col else {
        return Ok(base);
    };

    // Locate the key column in the batch.
    let arrow_schema = batch.schema();
    let key_idx = match arrow_schema.index_of(key_name) {
        Ok(i) => i,
        Err(_) => {
            // Key column absent — leave grouped = None.
            return Ok(base);
        }
    };
    let key_col_arr = batch.column(key_idx);

    // Identify numeric measure columns (those present in base.cols).
    let measure_names: Vec<String> = base.cols.keys().cloned().collect();

    let num_rows = batch.num_rows();

    // Bucket row indices by group key.
    let mut group_rows: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for row in 0..num_rows {
        let Some(key_str) = arrow_scalar_to_group_key_string(key_col_arr.as_ref(), row) else {
            // Unsupported key type — disable grouped for this batch.
            return Ok(base);
        };
        group_rows.entry(key_str).or_default().push(row);
    }

    // Cardinality guard.
    if group_rows.len() > MAX_DECLARED_GROUPS {
        // grouped stays None; ungrouped cols are preserved.
        return Ok(base);
    }

    // For each group, fold the measure columns over the bucket's row indices.
    let mut groups: BTreeMap<String, BTreeMap<String, ColAgg>> = BTreeMap::new();

    for (key_str, row_indices) in &group_rows {
        let mut group_cols: BTreeMap<String, ColAgg> = BTreeMap::new();

        for measure in &measure_names {
            let col_idx = match arrow_schema.index_of(measure) {
                Ok(i) => i,
                Err(_) => continue,
            };
            let col = batch.column(col_idx);
            let field = arrow_schema.field(col_idx);

            let maybe_col_agg: Option<ColAgg> = match field.data_type() {
                DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64 => fold_int_rows_to_col_agg(col, field.data_type(), row_indices),
                DataType::Float32 | DataType::Float64 => Some(fold_float_rows_to_col_agg(
                    col,
                    field.data_type(),
                    row_indices,
                )),
                _ => None,
            };

            if let Some(agg) = maybe_col_agg {
                group_cols.insert(measure.clone(), agg);
            }
        }

        groups.insert(key_str.clone(), group_cols);
    }

    base.grouped = Some(GroupedPartials {
        key_col: key_name.to_string(),
        groups,
    });
    Ok(base)
}

/// Fold integer column rows at the given indices into a [`ColAgg`].
/// Returns `None` on i128 overflow (caller drops the column for that group).
fn fold_int_rows_to_col_agg(
    col: &dyn arrow::array::Array,
    dt: &DataType,
    row_indices: &[usize],
) -> Option<ColAgg> {
    use arrow::array::{
        Int16Array, Int32Array, Int64Array, Int8Array, UInt16Array, UInt32Array, UInt64Array,
        UInt8Array,
    };

    let mut count: u64 = 0;
    let mut sum: i128 = 0;
    let mut sumsq: i128 = 0;
    let mut all_null = true;

    macro_rules! fold_typed {
        ($ArrType:ty) => {{
            let arr = col.as_any().downcast_ref::<$ArrType>()?;
            for &i in row_indices {
                if arr.is_null(i) {
                    continue;
                }
                all_null = false;
                let v = arr.value(i) as i128;
                count += 1;
                sum = sum.checked_add(v)?;
                let sq = v.checked_mul(v)?;
                sumsq = sumsq.checked_add(sq)?;
            }
        }};
    }

    match dt {
        DataType::Int8 => fold_typed!(Int8Array),
        DataType::Int16 => fold_typed!(Int16Array),
        DataType::Int32 => fold_typed!(Int32Array),
        DataType::Int64 => fold_typed!(Int64Array),
        DataType::UInt8 => fold_typed!(UInt8Array),
        DataType::UInt16 => fold_typed!(UInt16Array),
        DataType::UInt32 => fold_typed!(UInt32Array),
        DataType::UInt64 => fold_typed!(UInt64Array),
        _ => return None,
    }

    if all_null || count == 0 {
        Some(ColAgg {
            count_non_null: 0,
            sum: AggScalar::Null,
            sumsq: AggScalar::Null,
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        })
    } else {
        Some(ColAgg {
            count_non_null: count,
            sum: AggScalar::Int(sum),
            sumsq: AggScalar::Int(sumsq),
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        })
    }
}

/// Fold float column rows at the given indices into a [`ColAgg`].
fn fold_float_rows_to_col_agg(
    col: &dyn arrow::array::Array,
    dt: &DataType,
    row_indices: &[usize],
) -> ColAgg {
    let mut count: u64 = 0;
    let mut sum: f64 = 0.0;
    let mut sumsq: f64 = 0.0;
    let mut all_null = true;

    match dt {
        DataType::Float32 => {
            if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
                for &i in row_indices {
                    if arr.is_null(i) {
                        continue;
                    }
                    all_null = false;
                    let v = arr.value(i) as f64;
                    count += 1;
                    sum += v;
                    sumsq += v * v;
                }
            }
        }
        DataType::Float64 => {
            if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
                for &i in row_indices {
                    if arr.is_null(i) {
                        continue;
                    }
                    all_null = false;
                    let v = arr.value(i);
                    count += 1;
                    sum += v;
                    sumsq += v * v;
                }
            }
        }
        _ => {}
    }

    if all_null || count == 0 {
        ColAgg {
            count_non_null: 0,
            sum: AggScalar::Null,
            sumsq: AggScalar::Null,
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        }
    } else {
        ColAgg {
            count_non_null: count,
            sum: AggScalar::Float(sum),
            sumsq: AggScalar::Float(sumsq),
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        }
    }
}

/// Merge per-group partials from multiple fragments into a single map.
///
/// All `parts` must share the same `key_col` (asserted).  For keys present in
/// multiple fragments, each measure column's `ColAgg` is merged additively
/// (count += , sum += , sumsq += with the same checked-i128 / f64 semantics).
/// On integer overflow a measure column is dropped for that group so the rule
/// falls back.
///
/// Returns a `BTreeMap<group_key_str, BTreeMap<col_name, ColAgg>>`.
pub fn merge_grouped(parts: &[&GroupedPartials]) -> BTreeMap<String, BTreeMap<String, ColAgg>> {
    if parts.is_empty() {
        return BTreeMap::new();
    }

    // Assert all share the same key_col.
    let key_col = &parts[0].key_col;
    for p in parts.iter().skip(1) {
        assert_eq!(
            &p.key_col, key_col,
            "merge_grouped: all fragments must share the same key_col"
        );
    }

    let mut merged: BTreeMap<String, BTreeMap<String, ColAgg>> = BTreeMap::new();

    for part in parts {
        for (group_key, cols) in &part.groups {
            let entry = merged.entry(group_key.clone()).or_default();
            for (col_name, incoming) in cols {
                if !entry.contains_key(col_name) {
                    entry.insert(col_name.clone(), incoming.clone());
                    continue;
                }
                let existing = entry.get(col_name).unwrap().clone();
                let merged_agg = merge_two_col_aggs(&existing, incoming);
                match merged_agg {
                    Some(agg) => {
                        entry.insert(col_name.clone(), agg);
                    }
                    None => {
                        entry.remove(col_name);
                    }
                }
            }
        }
    }

    merged
}

/// Merge two [`ColAgg`] values additively.  Returns `None` on integer overflow
/// (the measure column should be dropped for that group).
fn merge_two_col_aggs(a: &ColAgg, b: &ColAgg) -> Option<ColAgg> {
    let count = a.count_non_null + b.count_non_null;

    let sum = match (&a.sum, &b.sum) {
        (AggScalar::Int(av), AggScalar::Int(bv)) => AggScalar::Int(av.checked_add(*bv)?),
        (AggScalar::Float(av), AggScalar::Float(bv)) => AggScalar::Float(av + bv),
        (AggScalar::Null, other) | (other, AggScalar::Null) => other.clone(),
        _ => return None, // variant mismatch
    };

    let sumsq = match (&a.sumsq, &b.sumsq) {
        (AggScalar::Int(av), AggScalar::Int(bv)) => AggScalar::Int(av.checked_add(*bv)?),
        (AggScalar::Float(av), AggScalar::Float(bv)) => AggScalar::Float(av + bv),
        (AggScalar::Null, other) | (other, AggScalar::Null) => other.clone(),
        _ => return None,
    };

    Some(ColAgg {
        count_non_null: count,
        sum,
        sumsq,
        min_off: None,
        max_off: None,
        live_min_json: None,
        live_max_json: None,
    })
}

/// Compute live per-group partials for a dirty fragment by reading ONLY the
/// deleted rows (via [`DeletionVector`]) for the key column and the requested
/// measure columns, routing each deleted row to its group, then subtracting
/// its contribution from the full per-group partial.
///
/// Returns a `BTreeMap<group_key_str, BTreeMap<col_name, ColAgg>>` covering
/// only the LIVE rows.  A group whose total count reaches 0 after retraction
/// is still present in the map (the warm-GROUP-BY rule discards zero-count groups).
///
/// NULL measure values are skipped (SUM/COUNT ignore NULLs).
/// A deleted row with an unsupported key type causes that group to be dropped.
pub async fn retract_grouped(
    storage: &dyn Storage,
    table: &str,
    entry: &RowGroupEntry,
    gp: &GroupedPartials,
    dv: &DeletionVector,
    measure_cols: &[String],
) -> Result<BTreeMap<String, BTreeMap<String, ColAgg>>> {
    // Start with the full grouped partials.
    let mut live: BTreeMap<String, BTreeMap<String, ColAgg>> = gp.groups.clone();

    if dv.cardinality() == 0 {
        return Ok(live);
    }

    // Build RowSelection selecting only the deleted row offsets.
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut pos: usize = 0;
    for offset in dv.iter() {
        let off = offset as usize;
        if off > pos {
            selectors.push(RowSelector::skip(off - pos));
        }
        selectors.push(RowSelector::select(1));
        pos = off + 1;
    }
    let row_selection = RowSelection::from(selectors);

    // Read the Parquet data file.
    let data_path = format!("{}/{}", table, entry.data);
    let data = storage.read(&data_path).await?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(Bytes::from(data))
        .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;
    let arrow_schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();

    // Build projection: key_col + requested measure columns.
    let key_arrow_idx = arrow_schema.index_of(&gp.key_col).map_err(|_| {
        IcefallDBError::ParquetDecode(format!("key column {} not found", gp.key_col))
    })?;
    let mut col_indices: Vec<usize> = vec![key_arrow_idx];
    for col_name in measure_cols {
        if let Ok(idx) = arrow_schema.index_of(col_name) {
            if !col_indices.contains(&idx) {
                col_indices.push(idx);
            }
        }
    }
    let mask = ProjectionMask::roots(&parquet_schema, col_indices.clone());
    let reader = builder
        .with_projection(mask)
        .with_row_selection(row_selection)
        .build()
        .map_err(|e| IcefallDBError::ParquetDecode(e.to_string()))?;

    let batches: Vec<RecordBatch> = reader
        .map(|r| r.map_err(|e| IcefallDBError::ParquetDecode(e.to_string())))
        .collect::<Result<Vec<_>>>()?;

    if batches.is_empty() {
        return Ok(live);
    }

    // Determine projected schema column indices.
    let proj_schema = batches[0].schema();
    let key_idx_in_proj = match proj_schema.index_of(&gp.key_col) {
        Ok(i) => i,
        Err(_) => return Ok(live),
    };

    // For each deleted batch row: determine group key, compute per-measure
    // contribution, subtract from live.
    for batch in &batches {
        let key_col_arr = batch.column(key_idx_in_proj);
        let num_rows = batch.num_rows();

        for row in 0..num_rows {
            let Some(key_str) = arrow_scalar_to_group_key_string(key_col_arr.as_ref(), row) else {
                continue;
            };

            let Some(group_cols) = live.get_mut(&key_str) else {
                // Deleted row's group wasn't in the full partial; skip.
                continue;
            };

            for col_name in measure_cols {
                let col_proj_idx = match proj_schema.index_of(col_name) {
                    Ok(i) => i,
                    Err(_) => continue,
                };
                let meas_col = batch.column(col_proj_idx);
                let meas_field = proj_schema.field(col_proj_idx);

                // Build a single-row "deleted contribution" ColAgg for this row.
                let del_agg: Option<ColAgg> = match meas_field.data_type() {
                    DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64 => {
                        fold_int_rows_to_col_agg(meas_col.as_ref(), meas_field.data_type(), &[row])
                    }
                    DataType::Float32 | DataType::Float64 => Some(fold_float_rows_to_col_agg(
                        meas_col.as_ref(),
                        meas_field.data_type(),
                        &[row],
                    )),
                    _ => None,
                };

                let Some(del) = del_agg else {
                    continue;
                };

                // Skip all-null deleted rows (no contribution to count/sum/sumsq).
                if del.count_non_null == 0 {
                    continue;
                }

                let Some(full_agg) = group_cols.get(col_name).cloned() else {
                    continue;
                };

                // Subtract: reuse retract logic.
                let new_count = full_agg.count_non_null.saturating_sub(del.count_non_null);
                let new_sum = match (&full_agg.sum, &del.sum) {
                    (AggScalar::Int(fs), AggScalar::Int(ds)) => match fs.checked_sub(*ds) {
                        Some(v) => AggScalar::Int(v),
                        None => {
                            group_cols.remove(col_name);
                            continue;
                        }
                    },
                    (AggScalar::Float(fs), AggScalar::Float(ds)) => AggScalar::Float(fs - ds),
                    (AggScalar::Null, AggScalar::Null) => AggScalar::Null,
                    (full_sum, AggScalar::Null) => full_sum.clone(),
                    _ => {
                        group_cols.remove(col_name);
                        continue;
                    }
                };
                let new_sumsq = match (&full_agg.sumsq, &del.sumsq) {
                    (AggScalar::Int(fq), AggScalar::Int(dq)) => match fq.checked_sub(*dq) {
                        Some(v) => AggScalar::Int(v),
                        None => {
                            group_cols.remove(col_name);
                            continue;
                        }
                    },
                    (AggScalar::Float(fq), AggScalar::Float(dq)) => AggScalar::Float(fq - dq),
                    (AggScalar::Null, AggScalar::Null) => AggScalar::Null,
                    (full_sq, AggScalar::Null) => full_sq.clone(),
                    _ => {
                        group_cols.remove(col_name);
                        continue;
                    }
                };

                group_cols.insert(
                    col_name.clone(),
                    ColAgg {
                        count_non_null: new_count,
                        sum: new_sum,
                        sumsq: new_sumsq,
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    },
                );
            }
        }
    }

    Ok(live)
}

// ── DV-density recompute policy helper ──────────────────────────────────────

/// Deletion-vector density threshold above which recomputing a fragment's `.agg`
/// via a full compaction pass is more economical than incremental retraction.
///
/// # Measured crossover (bench, 2026-06-23)
///
/// Benchmark: 1 M-row single-column Int64 Parquet fragment held in
/// `MemoryStorage` (eliminates disk I/O variance); deletion offsets evenly
/// spaced across the fragment; 5 measured iterations per density point.
///
/// | density | retract µs | recompute µs | ratio |
/// |---------|------------|--------------|-------|
/// | 0.01    | 1665       | 2014         | 0.83  |
/// | 0.02    | 1587       | 2163         | 0.73  |
/// | 0.05    | 1856       | 2242         | 0.83  |
/// | 0.10    | 2495       | 2516         | 0.99  |
/// | 0.20    | 4158       | 3331         | 1.25  |
/// | 0.30    | 5031       | 3459         | 1.45  |
/// | 0.50    | 7662       | 4366         | 1.76  |
///
/// Interpolated crossover ≈ 0.103.  Rounded to one decimal place: **0.10**.
/// Previous default was 0.2, which sat materially above the crossover (ratio
/// 1.25 at d=0.2 — retract was already 25% slower than recompute).
///
/// `run: cargo run --example retract_crossover -p icefalldb-core --release`
///
/// Callers that need a different threshold can compare
/// `dv_density(entry, total_rows)` against their own constant instead.
pub const RECOMPUTE_DENSITY: f64 = 0.10;

/// Compute the deletion-vector density for a fragment: the fraction of its
/// physical rows that have been logically deleted.
///
/// `total_rows` is the physical row count of the fragment (from `RowGroupMeta.rows`).
/// When `total_rows` is 0 the result is 0.0 (no rows → no deletion pressure).
///
/// The computation is zero-I/O: both inputs come from the in-memory manifest
/// entry and its companion `RowGroupMeta`.
///
/// # Examples
///
/// ```ignore
/// // 5 of 20 rows deleted → 25% density
/// assert_eq!(dv_density_with(5, 20), 0.25);
/// ```
pub fn dv_density(entry: &crate::metadata::RowGroupEntry, total_rows: u64) -> f64 {
    entry.deleted_count as f64 / total_rows.max(1) as f64
}

/// Returns `true` when the fragment's deletion density meets or exceeds
/// [`RECOMPUTE_DENSITY`], indicating that a compaction pass to rewrite the
/// fragment (and produce a fresh exact `.agg` with `deleted_count = 0`) is
/// preferred over incremental retraction.
///
/// Callers / compaction drivers can use this predicate to gate compaction
/// on deletion pressure rather than compacting the whole table unconditionally.
pub fn should_recompute(entry: &crate::metadata::RowGroupEntry, total_rows: u64) -> bool {
    dv_density(entry, total_rows) >= RECOMPUTE_DENSITY
}

// ── AggStateCache ────────────────────────────────────────────────────────────

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Default maximum number of `.agg` sidecar entries retained process-wide.
///
/// `.agg` files are immutable (write-once, UUID-named), so a path-keyed cache
/// is permanently valid for the lifetime of the process.
pub const DEFAULT_AGG_CACHE_CAPACITY: usize = 65_536;

/// Shared statistics counters for [`AggStateCache`].
///
/// Placed behind an `Arc` so that clones of `AggStateCache` (e.g. from
/// `AggStateCache::global()`) all update and read the same counters.
/// Atomics with `Relaxed` ordering — suitable for non-critical counters.
#[derive(Debug, Default)]
struct AggCacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
}

/// Bounded, thread-safe LRU cache mapping a storage path to its decoded
/// [`FragmentAggState`].
///
/// Mirrors [`crate::meta_cache::MetaCache`].  Capacity `0` disables the cache.
///
/// Every call to [`Self::get`] increments either the `hits` or `misses` counter
/// (shared across all clones of the same cache instance).  Use
/// [`Self::stats`] to read and [`Self::reset_stats`] to zero the counters —
/// the latter is intended for tests that need a clean baseline.
#[derive(Debug, Clone)]
pub struct AggStateCache {
    inner: std::sync::Arc<std::sync::RwLock<AggCacheInner>>,
    /// Shared hit/miss counters — `Arc` ensures all clones see the same values.
    stats: std::sync::Arc<AggCacheStats>,
}

#[derive(Debug)]
struct AggCacheInner {
    capacity: usize,
    entries: std::collections::HashMap<String, std::sync::Arc<FragmentAggState>>,
    /// LRU order: front = least-recently used, back = most-recently used.
    order: std::collections::VecDeque<String>,
}

static GLOBAL_AGG_CACHE: OnceLock<AggStateCache> = OnceLock::new();

impl AggStateCache {
    /// Create a new cache with the given entry capacity.
    ///
    /// Capacity `0` disables the cache; all `get`/`put` calls become no-ops.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: std::sync::Arc::new(std::sync::RwLock::new(AggCacheInner {
                capacity,
                entries: std::collections::HashMap::with_capacity(capacity.max(1)),
                order: std::collections::VecDeque::with_capacity(capacity.max(1)),
            })),
            stats: std::sync::Arc::new(AggCacheStats::default()),
        }
    }

    /// Return the process-wide singleton cache, lazily initialised with
    /// [`DEFAULT_AGG_CACHE_CAPACITY`].
    pub fn global() -> Self {
        GLOBAL_AGG_CACHE
            .get_or_init(|| AggStateCache::new(DEFAULT_AGG_CACHE_CAPACITY))
            .clone()
    }

    /// Return the process-wide singleton, seeding it with `capacity` if it has
    /// not been created yet.
    pub fn global_with_capacity(capacity: usize) -> Self {
        GLOBAL_AGG_CACHE
            .get_or_init(|| AggStateCache::new(capacity))
            .clone()
    }

    /// Look up an entry by its storage path.
    ///
    /// Returns `Some(Arc<FragmentAggState>)` on a hit and promotes to MRU.
    /// Returns `None` on a miss or when the cache is disabled (capacity 0).
    ///
    /// Increments the shared `hits` counter on a cache hit, `misses` on a miss.
    /// Uses `Relaxed` ordering — appropriate for non-critical stat counters.
    pub fn get(&self, path: &str) -> Option<std::sync::Arc<FragmentAggState>> {
        let value = {
            let guard = self.inner.read().ok()?;
            guard.entries.get(path).cloned()
        };
        if value.is_some() {
            self.stats.hits.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut guard) = self.inner.write() {
                if guard.entries.contains_key(path) {
                    guard.order.retain(|k| k != path);
                    guard.order.push_back(path.to_string());
                }
            }
        } else {
            self.stats.misses.fetch_add(1, Ordering::Relaxed);
        }
        value
    }

    /// Insert an entry.
    ///
    /// When capacity is `0`, this is a no-op.  When at capacity, the
    /// least-recently-used entry is evicted.
    pub fn put(&self, path: String, state: std::sync::Arc<FragmentAggState>) {
        let Ok(mut guard) = self.inner.write() else {
            return;
        };
        if guard.capacity == 0 {
            return;
        }
        let existed = guard.entries.contains_key(&path);
        if existed {
            guard.order.retain(|k| k != &path);
        }
        guard.order.push_back(path.clone());
        guard.entries.insert(path, state);
        if !existed && guard.order.len() > guard.capacity {
            if let Some(evicted) = guard.order.pop_front() {
                guard.entries.remove(&evicted);
            }
        }
    }

    /// Remove all entries from the cache.
    pub fn clear(&self) {
        let Ok(mut guard) = self.inner.write() else {
            return;
        };
        guard.entries.clear();
        guard.order.clear();
    }

    /// Return the number of entries currently held.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.entries.len())
            .unwrap_or_default()
    }

    /// Returns `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the configured capacity.
    pub fn capacity(&self) -> usize {
        self.inner
            .read()
            .map(|g| g.capacity)
            .unwrap_or(DEFAULT_AGG_CACHE_CAPACITY)
    }

    /// Return `(hits, misses)` accumulated since the last [`Self::reset_stats`]
    /// call (or since the cache was created).  Uses `Relaxed` ordering.
    pub fn stats(&self) -> (u64, u64) {
        (
            self.stats.hits.load(Ordering::Relaxed),
            self.stats.misses.load(Ordering::Relaxed),
        )
    }

    /// Reset both hit and miss counters to zero.  Intended for test isolation.
    pub fn reset_stats(&self) {
        self.stats.hits.store(0, Ordering::Relaxed);
        self.stats.misses.store(0, Ordering::Relaxed);
    }
}

/// Merge a collection of serialized per-column CPC sketch maps and return the
/// combined estimate for a named column.
///
/// `sketch_maps` is one entry per fragment.  Each entry maps column name → raw
/// CPC serialized bytes produced by `CpcSketch::serialize`.
///
/// Returns `None` if any fragment is missing the column's sketch (caller falls
/// back to an exact scan).
///
/// CPC at lg_k=12 has a nominal relative error ≈ 1.04% (1 std-dev) for
/// cardinalities above ~1000.  The test asserts ≤ 3% at 50k distinct values.
#[cfg(feature = "sketches")]
pub fn cpc_distinct_estimate(
    sketch_maps: &[&std::collections::BTreeMap<String, Vec<u8>>],
    col_name: &str,
) -> Option<f64> {
    use datasketches::cpc::{CpcSketch, CpcUnion};
    const LG_K: u8 = 12;
    let mut union = CpcUnion::new(LG_K);
    for map in sketch_maps {
        let bytes = map.get(col_name)?;
        let sketch = CpcSketch::deserialize(bytes).ok()?;
        union.update(&sketch);
    }
    Some(union.to_sketch().estimate())
}

// ── T-Digest merge + quantile estimate ─────────────────────────────────

/// Merge T-Digest sketches from multiple fragment maps and query the merged
/// sketch at quantile `q` (in [0, 1]).
///
/// Each entry in `sketch_maps` is the `quantile` field of a `FragmentAggState`.
/// Returns `None` when any map is missing the column or deserialization fails.
///
/// Sketch choice: T-Digest k=200 (datasketches default).
/// Error bound: rank error ~1-2% for smooth distributions; we assert ≤3% in
/// tests.  Note T-Digest is biased near extremes (underestimates low ranks,
/// overestimates high ranks) — the rank-error assertion in tests uses a
/// conservative 3% bound.
///
/// Dirty fragments are handled by the caller — they must NOT pass dirty-fragment
/// maps here.  The compose rule already falls back for dirty fragments.
#[cfg(feature = "sketches")]
pub fn tdigest_quantile_estimate(
    sketch_maps: &[&std::collections::BTreeMap<String, Vec<u8>>],
    col_name: &str,
    q: f64,
) -> Option<f64> {
    use datasketches::tdigest::TDigestMut;

    if sketch_maps.is_empty() {
        return None;
    }

    // Deserialize the first sketch as the merge target.
    let first_bytes = sketch_maps[0].get(col_name)?;
    let mut merged = TDigestMut::deserialize(first_bytes, false).ok()?;

    // Merge the remaining sketches.
    for map in &sketch_maps[1..] {
        let bytes = map.get(col_name)?;
        let other = TDigestMut::deserialize(bytes, false).ok()?;
        merged.merge(&other);
    }

    merged.freeze().quantile(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
    use std::sync::Arc;

    // ── retract unit tests ────────────────────────────────────────────────────

    #[test]
    fn retract_is_exact_inverse() {
        // Full fragment: values [10, 20, 30]
        // count=3, sum=60, sumsq=100+400+900=1400
        let full = FragmentAggState {
            fragment_id: 1,
            content_hash: "h1".into(),
            cols: {
                let mut m = BTreeMap::new();
                m.insert(
                    "v".to_string(),
                    ColAgg {
                        count_non_null: 3,
                        sum: AggScalar::Int(60),
                        sumsq: AggScalar::Int(1400),
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    },
                );
                m
            },
            grouped: None,
            distinct: None,
            quantile: None,
        };

        // Deleted contribution: the row with value 20
        // count=1, sum=20, sumsq=400
        let del = DeletedContribution {
            cols: {
                let mut m = BTreeMap::new();
                m.insert(
                    "v".to_string(),
                    ColAgg {
                        count_non_null: 1,
                        sum: AggScalar::Int(20),
                        sumsq: AggScalar::Int(400),
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    },
                );
                m
            },
        };

        let live = retract(&full, &del);
        let v = live.cols.get("v").expect("v column present");
        // Live: [10, 30] → count=2, sum=40, sumsq=100+900=1000
        assert_eq!(v.count_non_null, 2);
        assert_eq!(v.sum, AggScalar::Int(40));
        assert_eq!(v.sumsq, AggScalar::Int(1000));
    }

    #[test]
    fn retract_drops_column_on_variant_mismatch() {
        let full = FragmentAggState {
            fragment_id: 1,
            content_hash: "h".into(),
            cols: {
                let mut m = BTreeMap::new();
                m.insert(
                    "v".to_string(),
                    ColAgg {
                        count_non_null: 3,
                        sum: AggScalar::Int(60),
                        sumsq: AggScalar::Int(1400),
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    },
                );
                m
            },
            grouped: None,
            distinct: None,
            quantile: None,
        };
        let del = DeletedContribution {
            cols: {
                let mut m = BTreeMap::new();
                m.insert(
                    "v".to_string(),
                    ColAgg {
                        count_non_null: 1,
                        sum: AggScalar::Float(20.0), // mismatch
                        sumsq: AggScalar::Float(400.0),
                        min_off: None,
                        max_off: None,
                        live_min_json: None,
                        live_max_json: None,
                    },
                );
                m
            },
        };
        let live = retract(&full, &del);
        assert!(
            !live.cols.contains_key("v"),
            "mismatched column must be dropped"
        );
    }

    // ── deleted_contribution async test ──────────────────────────────────────

    #[tokio::test]
    async fn deleted_contribution_selects_only_deleted_rows() {
        use crate::deletion::DeletionVector;
        use crate::metadata::RowGroupEntry;
        use crate::storage::memory::MemoryStorage;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use parquet::arrow::ArrowWriter;

        // Build a small fragment with values [10, 20, 30, 40, 50] for column "v".
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));
        let values: Int64Array = vec![10i64, 20, 30, 40, 50].into_iter().map(Some).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let storage = MemoryStorage::new();
        storage.write("t/rg_test.parquet", &buf).await.unwrap();

        let entry = RowGroupEntry {
            data: "rg_test.parquet".into(),
            meta: "rg_test.meta".into(),
            ..Default::default()
        };

        // Delete rows at offsets 1 (value=20) and 3 (value=40).
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32, 3]);

        let contrib = deleted_contribution(&storage, "t", &entry, &dv, &["v".to_string()])
            .await
            .unwrap();

        let v = contrib.cols.get("v").expect("v column in contribution");
        // Deleted rows: 20, 40 → count=2, sum=60, sumsq=400+1600=2000
        assert_eq!(v.count_non_null, 2);
        assert_eq!(v.sum, AggScalar::Int(60));
        assert_eq!(v.sumsq, AggScalar::Int(2000));
    }

    /// Float tolerance: relative ULP bound.
    fn float_approx_eq(a: f64, b: f64) -> bool {
        let tolerance = 1e-9 * b.abs().max(1.0);
        (a - b).abs() <= tolerance
    }

    fn extract_int(scalar: &AggScalar) -> i128 {
        match scalar {
            AggScalar::Int(v) => *v,
            other => panic!("expected AggScalar::Int, got {other:?}"),
        }
    }

    fn extract_float(scalar: &AggScalar) -> f64 {
        match scalar {
            AggScalar::Float(v) => *v,
            other => panic!("expected AggScalar::Float, got {other:?}"),
        }
    }

    /// Build a RecordBatch with one Int64 and one Float64 column, each
    /// containing some nulls.
    fn make_mixed_batch() -> RecordBatch {
        // Int64 column: [10, null, 20, 30]
        // count_non_null = 3, sum = 60, sumsq = 100+400+900 = 1400
        let ints: Int64Array = [Some(10i64), None, Some(20), Some(30)]
            .into_iter()
            .collect();

        // Float64 column: [1.5, 2.5, null, 4.0]
        // count_non_null = 3, sum = 8.0, sumsq = 2.25 + 6.25 + 16.0 = 24.5
        let floats: Float64Array = [Some(1.5_f64), Some(2.5), None, Some(4.0)]
            .into_iter()
            .collect();

        let schema = ArrowSchema::new(vec![
            Field::new("qty", DataType::Int64, true),
            Field::new("price", DataType::Float64, true),
        ]);

        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ints), Arc::new(floats)]).unwrap()
    }

    #[test]
    fn compute_int_column_exact_values_and_count() {
        let batch = make_mixed_batch();
        let state = compute_agg_state(42, "sha256:abc".into(), &batch).unwrap();

        let qty = state.cols.get("qty").expect("qty column present");
        assert_eq!(qty.count_non_null, 3);
        assert_eq!(extract_int(&qty.sum), 60);
        assert_eq!(extract_int(&qty.sumsq), 1400);
    }

    #[test]
    fn compute_float_column_within_tolerance() {
        let batch = make_mixed_batch();
        let state = compute_agg_state(42, "sha256:abc".into(), &batch).unwrap();

        let price = state.cols.get("price").expect("price column present");
        assert_eq!(price.count_non_null, 3);

        let sum = extract_float(&price.sum);
        let expected_sum = 8.0_f64;
        assert!(
            float_approx_eq(sum, expected_sum),
            "sum {sum} not within tolerance of {expected_sum}"
        );

        let sumsq = extract_float(&price.sumsq);
        let expected_sumsq = 24.5_f64;
        assert!(
            float_approx_eq(sumsq, expected_sumsq),
            "sumsq {sumsq} not within tolerance of {expected_sumsq}"
        );
    }

    #[test]
    fn all_null_column_produces_null_scalar() {
        let all_null: Int64Array = [None, None, None].into_iter().collect();
        let schema = ArrowSchema::new(vec![Field::new("x", DataType::Int64, true)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(all_null)]).unwrap();

        let state = compute_agg_state(1, "sha256:x".into(), &batch).unwrap();
        let x = state.cols.get("x").expect("x column present");
        assert_eq!(x.count_non_null, 0);
        assert_eq!(x.sum, AggScalar::Null);
        assert_eq!(x.sumsq, AggScalar::Null);
    }

    #[test]
    fn non_numeric_columns_absent_from_cols() {
        let strings: StringArray = [Some("hello"), Some("world")].into_iter().collect();
        let schema = ArrowSchema::new(vec![Field::new("label", DataType::Utf8, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(strings)]).unwrap();

        let state = compute_agg_state(7, "sha256:y".into(), &batch).unwrap();
        assert!(
            state.cols.is_empty(),
            "non-numeric columns must be absent from cols"
        );
    }

    #[test]
    fn fragment_id_and_content_hash_preserved() {
        let batch = make_mixed_batch();
        let hash = "sha256:deadbeef01234567".to_string();
        let state = compute_agg_state(99, hash.clone(), &batch).unwrap();
        assert_eq!(state.fragment_id, 99);
        assert_eq!(state.content_hash, hash);
    }

    #[test]
    fn round_trip_serialize_deserialize() {
        let batch = make_mixed_batch();
        let original = compute_agg_state(5, "sha256:roundtrip".into(), &batch).unwrap();
        let bytes = serialize_agg_state(&original).unwrap();
        let restored = deserialize_agg_state(&bytes).unwrap();
        assert_eq!(original, restored, "round-trip must be lossless");
    }

    #[test]
    fn agg_cache_put_and_get() {
        let cache = AggStateCache::new(4);
        let state = Arc::new(FragmentAggState {
            fragment_id: 1,
            content_hash: "sha256:abc".into(),
            cols: BTreeMap::new(),
            grouped: None,
            distinct: None,
            quantile: None,
        });
        assert!(cache.get("table/rg_a.agg").is_none());
        cache.put("table/rg_a.agg".to_string(), Arc::clone(&state));
        let got = cache.get("table/rg_a.agg").expect("cached state");
        assert!(Arc::ptr_eq(&got, &state));
    }

    #[test]
    fn agg_cache_capacity_zero_disables() {
        let cache = AggStateCache::new(0);
        let state = Arc::new(FragmentAggState {
            fragment_id: 2,
            content_hash: "sha256:zero".into(),
            cols: BTreeMap::new(),
            grouped: None,
            distinct: None,
            quantile: None,
        });
        cache.put("table/rg_z.agg".to_string(), Arc::clone(&state));
        assert!(cache.get("table/rg_z.agg").is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn agg_cache_lru_eviction() {
        let cache = AggStateCache::new(2);
        let s1 = Arc::new(FragmentAggState {
            fragment_id: 1,
            content_hash: "h1".into(),
            cols: BTreeMap::new(),
            grouped: None,
            distinct: None,
            quantile: None,
        });
        let s2 = Arc::new(FragmentAggState {
            fragment_id: 2,
            content_hash: "h2".into(),
            cols: BTreeMap::new(),
            grouped: None,
            distinct: None,
            quantile: None,
        });
        let s3 = Arc::new(FragmentAggState {
            fragment_id: 3,
            content_hash: "h3".into(),
            cols: BTreeMap::new(),
            grouped: None,
            distinct: None,
            quantile: None,
        });
        cache.put("t/rg_1.agg".to_string(), Arc::clone(&s1));
        cache.put("t/rg_2.agg".to_string(), Arc::clone(&s2));
        // Access rg_1 → it becomes MRU.
        assert!(cache.get("t/rg_1.agg").is_some());
        // Insert rg_3 → rg_2 (LRU) should be evicted.
        cache.put("t/rg_3.agg".to_string(), Arc::clone(&s3));
        assert!(cache.get("t/rg_1.agg").is_some(), "rg_1 must survive");
        assert!(cache.get("t/rg_2.agg").is_none(), "rg_2 must be evicted");
        assert!(cache.get("t/rg_3.agg").is_some(), "rg_3 must be present");
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn agg_cache_clear() {
        let cache = AggStateCache::new(4);
        cache.put(
            "t/rg_a.agg".to_string(),
            Arc::new(FragmentAggState {
                fragment_id: 1,
                content_hash: "h".into(),
                cols: BTreeMap::new(),
                grouped: None,
                distinct: None,
                quantile: None,
            }),
        );
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    /// deleted_contribution with NULLs in the deleted rows: null values must be
    /// excluded from count_non_null, sum, and sumsq — so retract is the exact
    /// inverse of the NULL-skipping full partial.
    #[tokio::test]
    async fn deleted_contribution_excludes_null_rows() {
        use crate::deletion::DeletionVector;
        use crate::metadata::RowGroupEntry;
        use crate::storage::memory::MemoryStorage;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use parquet::arrow::ArrowWriter;

        // Fragment: column "v" = [10, null, 30, 40, null]
        // Full partial would count non-nulls: indices 0,2,3 → count=3, sum=80, sumsq=100+900+1600=2600
        // Delete offsets 1 (null) and 3 (value=40).
        // Deleted contribution should exclude the null at index 1:
        //   count=1 (only value=40), sum=40, sumsq=1600
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            true,
        )]));
        let values: Int64Array = [Some(10i64), None, Some(30), Some(40), None]
            .into_iter()
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let storage = MemoryStorage::new();
        storage.write("t/rg_null.parquet", &buf).await.unwrap();

        let entry = RowGroupEntry {
            data: "rg_null.parquet".into(),
            meta: "rg_null.meta".into(),
            ..Default::default()
        };

        // Delete offsets 1 (null) and 3 (value=40).
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32, 3]);

        let contrib = deleted_contribution(&storage, "t", &entry, &dv, &["v".to_string()])
            .await
            .unwrap();

        let v = contrib.cols.get("v").expect("v column in contribution");
        // Only non-null deleted row is value=40.
        assert_eq!(v.count_non_null, 1, "null deleted rows must not count");
        assert_eq!(
            v.sum,
            AggScalar::Int(40),
            "null deleted rows excluded from sum"
        );
        assert_eq!(
            v.sumsq,
            AggScalar::Int(1600),
            "null deleted rows excluded from sumsq"
        );
    }

    // ── extremum_deleted and scoped_recompute_extremum ─────────────────

    /// extremum_deleted returns false for a non-extremal delete,
    /// true when the extremal row itself is deleted.
    #[test]
    fn extremum_deleted_true_only_when_extremal_row_is_deleted() {
        use crate::deletion::DeletionVector;

        let col_agg = ColAgg {
            count_non_null: 3,
            sum: AggScalar::Int(60),
            sumsq: AggScalar::Int(1400),
            min_off: Some(0),
            max_off: Some(2),
            live_min_json: None,
            live_max_json: None,
        };

        // Delete offset 1 (non-extremal) → both false.
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32]);
        assert!(
            !extremum_deleted(&col_agg, &dv, ExtremumKind::Min),
            "non-extremal delete must not mark min as deleted"
        );
        assert!(
            !extremum_deleted(&col_agg, &dv, ExtremumKind::Max),
            "non-extremal delete must not mark max as deleted"
        );

        // Delete offset 0 (the MIN row) → min is deleted, max is not.
        let mut dv2 = DeletionVector::default();
        dv2.union_offsets([0u32]);
        assert!(
            extremum_deleted(&col_agg, &dv2, ExtremumKind::Min),
            "deleting the min row must return true for Min"
        );
        assert!(
            !extremum_deleted(&col_agg, &dv2, ExtremumKind::Max),
            "deleting offset 0 must not affect Max (max_off=2)"
        );
    }

    /// extremum_deleted returns false when min_off/max_off is None.
    #[test]
    fn extremum_deleted_false_when_offset_unknown() {
        use crate::deletion::DeletionVector;

        let col_agg = ColAgg {
            count_non_null: 3,
            sum: AggScalar::Int(60),
            sumsq: AggScalar::Int(1400),
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        };

        let mut dv = DeletionVector::default();
        dv.union_offsets([0u32, 1, 2]);
        assert!(
            !extremum_deleted(&col_agg, &dv, ExtremumKind::Min),
            "None min_off must return false (graceful fallback)"
        );
        assert!(
            !extremum_deleted(&col_agg, &dv, ExtremumKind::Max),
            "None max_off must return false (graceful fallback)"
        );
    }

    // ── grouped partials TDD tests ────────────────────────────────────

    /// Helper: build a ColAgg for integer values from a slice (no overflow possible
    /// for test data).
    fn int_col_agg(vals: &[i64]) -> ColAgg {
        let count = vals.len() as u64;
        let sum: i128 = vals.iter().map(|&v| v as i128).sum();
        let sumsq: i128 = vals.iter().map(|&v| (v as i128) * (v as i128)).sum();
        ColAgg {
            count_non_null: count,
            sum: AggScalar::Int(sum),
            sumsq: AggScalar::Int(sumsq),
            min_off: None,
            max_off: None,
            live_min_json: None,
            live_max_json: None,
        }
    }

    /// Build a minimal GroupedPartials for tests (bypasses compute_agg_state).
    fn make_gp(key_col: &str, groups: &[(&str, &[i64])]) -> GroupedPartials {
        let mut map: BTreeMap<String, BTreeMap<String, ColAgg>> = BTreeMap::new();
        for (key_str, vals) in groups {
            let mut cols = BTreeMap::new();
            cols.insert("v".to_string(), int_col_agg(vals));
            map.insert(key_str.to_string(), cols);
        }
        GroupedPartials {
            key_col: key_col.to_string(),
            groups: map,
        }
    }

    // ── Test 1: merge_grouped then retract per group ─────────────────────────

    /// Fragment A {red: sum 10, blue: sum 20}, fragment B {red: sum 5}.
    /// merge_grouped ⇒ red 15, blue 20.
    /// Then subtract a deleted row with value 5 from red ⇒ red 10.
    #[test]
    fn grouped_merge_then_subtract_per_group() {
        // Fragment A: red=[10], blue=[20]
        let gp_a = make_gp("k", &[("red", &[10]), ("blue", &[20])]);
        // Fragment B: red=[5]
        let gp_b = make_gp("k", &[("red", &[5])]);

        let merged = merge_grouped(&[&gp_a, &gp_b]);

        // red: count=2, sum=15, sumsq=100+25=125
        let red = merged.get("red").expect("red group present");
        let red_v = red.get("v").expect("v col in red");
        assert_eq!(red_v.count_non_null, 2);
        assert_eq!(red_v.sum, AggScalar::Int(15));
        assert_eq!(red_v.sumsq, AggScalar::Int(125));

        // blue: count=1, sum=20, sumsq=400
        let blue = merged.get("blue").expect("blue group present");
        let blue_v = blue.get("v").expect("v col in blue");
        assert_eq!(blue_v.count_non_null, 1);
        assert_eq!(blue_v.sum, AggScalar::Int(20));
        assert_eq!(blue_v.sumsq, AggScalar::Int(400));

        // Subtract a deleted row with value 5 from red (simulating retract logic).
        // Start from merged red ColAgg, subtract the deleted row contribution.
        let del_agg = int_col_agg(&[5]);
        let new_count = red_v.count_non_null - del_agg.count_non_null;
        let new_sum = match (&red_v.sum, &del_agg.sum) {
            (AggScalar::Int(fs), AggScalar::Int(ds)) => AggScalar::Int(fs - ds),
            _ => panic!("expected Int"),
        };
        let new_sumsq = match (&red_v.sumsq, &del_agg.sumsq) {
            (AggScalar::Int(fq), AggScalar::Int(dq)) => AggScalar::Int(fq - dq),
            _ => panic!("expected Int"),
        };
        // After subtracting 5: sum = 10, sumsq = 125-25=100, count=1
        assert_eq!(new_count, 1);
        assert_eq!(new_sum, AggScalar::Int(10));
        assert_eq!(new_sumsq, AggScalar::Int(100));
    }

    // ── Test 2: write-time grouped partials match direct per-group aggregates ─

    /// Build a batch with declared key `k` (3 groups) and measure `v`;
    /// compute FragmentAggState; assert each group's count/sum/sumsq equals a
    /// direct per-group aggregate.  Round-trip the sidecar.
    #[test]
    fn write_time_grouped_partials_match_direct_group_by() {
        // Batch: k = [A, A, B, B, C], v = [1, 2, 10, 20, 100]
        // Group A: count=2, sum=3, sumsq=1+4=5
        // Group B: count=2, sum=30, sumsq=100+400=500
        // Group C: count=1, sum=100, sumsq=10000
        let k_col: StringArray = vec![Some("A"), Some("A"), Some("B"), Some("B"), Some("C")]
            .into_iter()
            .collect();
        let v_col: Int64Array = vec![Some(1i64), Some(2), Some(10), Some(20), Some(100)]
            .into_iter()
            .collect();
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(k_col), Arc::new(v_col)],
        )
        .unwrap();

        let state =
            compute_agg_state_with_key(42, "sha256:test".into(), &batch, Some("k")).unwrap();

        let gp = state
            .grouped
            .as_ref()
            .expect("grouped partials must be present");
        assert_eq!(gp.key_col, "k");

        // Group A
        let a = gp.groups.get("\"A\"").expect("group A");
        let a_v = a.get("v").expect("v in group A");
        assert_eq!(a_v.count_non_null, 2);
        assert_eq!(a_v.sum, AggScalar::Int(3));
        assert_eq!(a_v.sumsq, AggScalar::Int(5));

        // Group B
        let b = gp.groups.get("\"B\"").expect("group B");
        let b_v = b.get("v").expect("v in group B");
        assert_eq!(b_v.count_non_null, 2);
        assert_eq!(b_v.sum, AggScalar::Int(30));
        assert_eq!(b_v.sumsq, AggScalar::Int(500));

        // Group C
        let c = gp.groups.get("\"C\"").expect("group C");
        let c_v = c.get("v").expect("v in group C");
        assert_eq!(c_v.count_non_null, 1);
        assert_eq!(c_v.sum, AggScalar::Int(100));
        assert_eq!(c_v.sumsq, AggScalar::Int(10000));

        // Round-trip
        let bytes = serialize_agg_state(&state).unwrap();
        let restored = deserialize_agg_state(&bytes).unwrap();
        assert_eq!(state, restored, "sidecar round-trip must be lossless");
    }

    // ── Test 3: cardinality guard disables grouped ────────────────────────────

    /// A batch whose key has > MAX_DECLARED_GROUPS distinct values
    /// must produce grouped == None.
    #[test]
    fn cardinality_guard_disables_grouped() {
        // Build a batch with MAX_DECLARED_GROUPS+1 distinct key values.
        let n = MAX_DECLARED_GROUPS + 1;
        let keys: StringArray = (0..n).map(|i| Some(format!("k{i}"))).collect();
        let vals: Int64Array = (0..n as i64).map(Some).collect();
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]);
        let batch =
            RecordBatch::try_new(Arc::new(arrow_schema), vec![Arc::new(keys), Arc::new(vals)])
                .unwrap();

        let state = compute_agg_state_with_key(1, "sha256:card".into(), &batch, Some("k")).unwrap();
        assert!(
            state.grouped.is_none(),
            "cardinality guard: grouped must be None when distinct keys > MAX_DECLARED_GROUPS"
        );
        // Ungrouped cols must still be present.
        assert!(
            state.cols.contains_key("v"),
            "ungrouped cols must survive cardinality guard"
        );
    }

    // ── Test 4: retract_grouped reads only deleted and routes by key ──────────

    /// Write a fragment, delete 2 rows in known groups, assert the
    /// returned live per-group partials equal the full per-group minus those rows.
    #[tokio::test]
    async fn retract_grouped_reads_only_deleted_and_routes_by_key() {
        use crate::deletion::DeletionVector;
        use crate::metadata::RowGroupEntry;
        use crate::storage::memory::MemoryStorage;
        use parquet::arrow::ArrowWriter;

        // Fragment: k=[A, A, B, B], v=[1, 2, 10, 20]
        // Written at offsets 0..3.
        // Delete offsets 1 (k=A, v=2) and 3 (k=B, v=20).
        // After retraction:
        //   Group A live: sum=1, count=1
        //   Group B live: sum=10, count=1
        let k_col: StringArray = vec![Some("A"), Some("A"), Some("B"), Some("B")]
            .into_iter()
            .collect();
        let v_col: Int64Array = vec![Some(1i64), Some(2), Some(10), Some(20)]
            .into_iter()
            .collect();
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(k_col), Arc::new(v_col)],
        )
        .unwrap();

        // Write the full agg state.
        let full_state =
            compute_agg_state_with_key(1, "sha256:rg4".into(), &batch, Some("k")).unwrap();
        let gp = full_state.grouped.as_ref().expect("grouped present");

        // Persist to MemoryStorage.
        let mut buf = Vec::new();
        {
            let mut writer =
                ArrowWriter::try_new(&mut buf, Arc::clone(&arrow_schema), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
        let storage = MemoryStorage::new();
        storage.write("t/rg_r4.parquet", &buf).await.unwrap();

        let entry = RowGroupEntry {
            data: "rg_r4.parquet".into(),
            meta: "rg_r4.meta".into(),
            ..Default::default()
        };

        // Delete offsets 1 (A/v=2) and 3 (B/v=20).
        let mut dv = DeletionVector::default();
        dv.union_offsets([1u32, 3]);

        let live = retract_grouped(&storage, "t", &entry, gp, &dv, &["v".to_string()])
            .await
            .unwrap();

        // Group A live: v=1 only → count=1, sum=1
        let a = live.get("\"A\"").expect("group A in live");
        let a_v = a.get("v").expect("v in A");
        assert_eq!(a_v.count_non_null, 1, "group A: count after retract");
        assert_eq!(a_v.sum, AggScalar::Int(1), "group A: sum after retract");

        // Group B live: v=10 only → count=1, sum=10
        let b = live.get("\"B\"").expect("group B in live");
        let b_v = b.get("v").expect("v in B");
        assert_eq!(b_v.count_non_null, 1, "group B: count after retract");
        assert_eq!(b_v.sum, AggScalar::Int(10), "group B: sum after retract");
    }

    // ── Test 5: NULL-key grouping ─────────────────────────────────────────────

    /// Rows with NULL key form exactly one group (SQL GROUP BY NULLs
    /// together).  Verify write-time bucketing and that a round-tripped sidecar
    /// contains the NULL group.
    #[test]
    fn null_key_grouping_consistent() {
        // Batch: k=[A, null, null, B], v=[1, 2, 3, 4]
        // Group "A": sum=1
        // Group NULL: sum=5 (rows 1+2)
        // Group "B": sum=4
        let k_col: StringArray = vec![Some("A"), None, None, Some("B")].into_iter().collect();
        let v_col: Int64Array = vec![Some(1i64), Some(2), Some(3), Some(4)]
            .into_iter()
            .collect();
        let arrow_schema = ArrowSchema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("v", DataType::Int64, false),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![Arc::new(k_col), Arc::new(v_col)],
        )
        .unwrap();

        let state = compute_agg_state_with_key(7, "sha256:null".into(), &batch, Some("k")).unwrap();
        let gp = state.grouped.as_ref().expect("grouped present");

        // NULL group key is represented as "null" (JSON canonical form).
        let null_grp = gp.groups.get("null").expect("null group present");
        let null_v = null_grp.get("v").expect("v in null group");
        assert_eq!(null_v.count_non_null, 2);
        assert_eq!(null_v.sum, AggScalar::Int(5));

        // Round-trip preserves the null group.
        let bytes = serialize_agg_state(&state).unwrap();
        let restored = deserialize_agg_state(&bytes).unwrap();
        let gp2 = restored.grouped.as_ref().expect("grouped after round-trip");
        assert!(
            gp2.groups.contains_key("null"),
            "null group survives round-trip"
        );
    }

    // ── schema agg_group_keys absent ⇒ stable checksum ────────────────

    /// The new `agg_group_keys` field uses `#[serde(skip_serializing_if = "Option::is_none", default)]`
    /// so a Schema without it serializes to the same bytes as before.
    #[test]
    fn schema_absent_agg_group_keys_stable_bytes() {
        use crate::metadata::schema::{Column, Schema};

        let schema_without = Schema {
            schema_id: 1,
            columns: vec![Column::new("v", "int64", false)],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1_000_000,
            dropped_columns: vec![],
            max_field_id: 1,
        };

        let json = serde_json::to_string(&schema_without).unwrap();
        // The absent field must not appear in the JSON.
        assert!(
            !json.contains("agg_group_keys"),
            "absent agg_group_keys must not appear in serialized JSON: {json}"
        );
    }

    /// scoped_recompute_extremum over survivors equals direct survivor-min.
    #[tokio::test]
    async fn scoped_recompute_extremum_matches_direct_survivor_min() {
        use crate::deletion::DeletionVector;
        use crate::metadata::RowGroupEntry;
        use crate::storage::memory::MemoryStorage;
        use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
        use parquet::arrow::ArrowWriter;

        // Fragment: column "v" = [1, 5, 2, 8, 3] at offsets 0..4.
        // min = 1 at offset 0. Delete offset 0 → survivor-min = 2 (offset 2).
        let schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            false,
        )]));
        let values: Int64Array = vec![1i64, 5, 2, 8, 3].into_iter().map(Some).collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)]).unwrap();

        let mut buf = Vec::new();
        {
            let mut writer = ArrowWriter::try_new(&mut buf, Arc::clone(&schema), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }

        let storage = MemoryStorage::new();
        storage.write("t/rg_r22.parquet", &buf).await.unwrap();

        let entry = RowGroupEntry {
            data: "rg_r22.parquet".into(),
            meta: "rg_r22.meta".into(),
            ..Default::default()
        };

        // Delete offset 0 (value=1, the minimum).
        let mut dv = DeletionVector::default();
        dv.union_offsets([0u32]);

        let result =
            scoped_recompute_extremum(&storage, "t", &entry, &dv, 5, "v", ExtremumKind::Min)
                .await
                .unwrap();

        // Direct survivor min: [5, 2, 8, 3] → 2
        match result {
            Some(v) => {
                let num = v.as_i64().expect("expected integer JSON value");
                assert_eq!(num, 2i64, "survivor-min must be 2 after deleting value=1");
            }
            None => panic!("expected Some(2), got None"),
        }
    }

    // ── CPC sketch tests ────────────────────────────────────────────────

    #[cfg(feature = "sketches")]
    #[test]
    fn sketch_round_trip_and_estimate_within_error() {
        // Build a batch with ~500 distinct values for column "v" (Int64).
        // CPC lg_k=12 nominal error ≈ 1.04% (1σ); we assert within 5% here.
        let n = 500usize;
        let vals: Int64Array = (0..n as i64).map(Some).collect();
        let schema = ArrowSchema::new(vec![Field::new("v", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(vals)]).unwrap();

        let state = compute_agg_state(1, "sha256:sketch".into(), &batch).unwrap();
        let distinct_map = state
            .distinct
            .as_ref()
            .expect("distinct must be present with sketches feature");
        let bytes = distinct_map.get("v").expect("sketch for column v");

        use datasketches::cpc::CpcSketch;
        let sketch = CpcSketch::deserialize(bytes).unwrap();
        let estimate = sketch.estimate();
        let error = (estimate - n as f64).abs() / n as f64;
        assert!(
            error < 0.05,
            "CPC estimate {estimate:.1} for true cardinality {n}: relative error {error:.4} must be < 5%"
        );
    }

    #[cfg(feature = "sketches")]
    #[test]
    fn sketch_absent_without_feature_compile_check() {
        // With the feature enabled, distinct must be populated after compute_agg_state.
        let vals: Int64Array = (0..10i64).map(Some).collect();
        let schema = ArrowSchema::new(vec![Field::new("v", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(vals)]).unwrap();
        let state = compute_agg_state(1, "sha256:x".into(), &batch).unwrap();
        assert!(
            state.distinct.is_some(),
            "distinct must be populated with sketches feature"
        );
    }

    #[cfg(feature = "sketches")]
    #[test]
    fn cpc_merge_estimate_three_fragments() {
        use super::cpc_distinct_estimate;
        use datasketches::cpc::CpcSketch;
        use std::collections::BTreeMap;

        // Three non-overlapping sets of 1000 distinct values each → total distinct = 3000.
        let mut maps = vec![];
        for frag in 0..3u64 {
            let mut sketch = CpcSketch::new(12);
            for i in 0..1000u64 {
                sketch.update(frag * 1000 + i);
            }
            let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            map.insert("v".to_string(), sketch.serialize());
            maps.push(map);
        }
        let refs: Vec<&BTreeMap<String, Vec<u8>>> = maps.iter().collect();
        let estimate = cpc_distinct_estimate(&refs, "v").unwrap();
        let true_val = 3000.0f64;
        let error = (estimate - true_val).abs() / true_val;
        assert!(
            error < 0.05,
            "merged CPC estimate {estimate:.1} for {true_val}: relative error {error:.4} must be < 5%"
        );
    }

    // ── T-Digest sketch tests ───────────────────────────────────────────

    /// Unit test: T-Digest sketch round-trip and quantile within error bound.
    ///
    /// Chosen sketch: T-Digest k=200 (datasketches default).
    /// Error bound: rank error ≤3% (conservative over T-Digest's typical ~1-2%).
    /// We assert `|rank(returned_p95) - 0.95| ≤ 0.03` (rank-error assertion),
    /// which is tighter than byte-equality and correct per the brief.
    #[cfg(feature = "sketches")]
    #[test]
    fn tdigest_round_trip_and_quantile_within_error() {
        use crate::agg_cache::tdigest_quantile_estimate;

        // Build a batch with 10_000 Float64 values in [0, 10_000).
        // True p95 ≈ 9500.
        let n = 10_000usize;
        let vals: Float64Array = (0..n as i64).map(|i| Some(i as f64)).collect();
        let schema = ArrowSchema::new(vec![Field::new("v", DataType::Float64, false)]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(vals.clone())]).unwrap();

        let state = compute_agg_state(1, "sha256:tdigest-rt".into(), &batch).unwrap();
        let quantile_map = state
            .quantile
            .as_ref()
            .expect("quantile must be present with sketches feature");
        assert!(
            quantile_map.contains_key("v"),
            "sketch for column v must be present"
        );

        // Deserialize and query at p95.
        let maps = vec![quantile_map];
        let p95_est =
            tdigest_quantile_estimate(&maps, "v", 0.95).expect("quantile estimate must succeed");

        // Compute rank of the returned value over the original sorted data.
        let mut sorted: Vec<f64> = (0..n as i64).map(|i| i as f64).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rank_pos = sorted.partition_point(|&x| x <= p95_est);
        let actual_rank = rank_pos as f64 / n as f64;

        let rank_err = (actual_rank - 0.95f64).abs();
        assert!(
            rank_err <= 0.03,
            "T-Digest p95 estimate {p95_est:.1}: rank {actual_rank:.4} vs target 0.95, \
             rank error {rank_err:.4} must be ≤ 0.03 (T-Digest k=200 bound)"
        );
    }

    /// Merge test: three fragments with skewed data; merged T-Digest p95 within bound.
    #[cfg(feature = "sketches")]
    #[test]
    fn tdigest_merge_three_fragments_within_error() {
        use crate::agg_cache::tdigest_quantile_estimate;
        use datasketches::tdigest::TDigestMut;
        use std::collections::BTreeMap;

        // Fragment 0: values [0, 333_333)  — lower third
        // Fragment 1: values [333_333, 666_666)  — middle third
        // Fragment 2: values [666_666, 1_000_000)  — upper third (skewed toward high)
        // Total ~1M rows (split into 3 fragments for performance in unit tests: use 100k each)
        const N: usize = 100_000;
        let mut all_values: Vec<f64> = Vec::with_capacity(3 * N);
        let mut maps: Vec<BTreeMap<String, Vec<u8>>> = Vec::new();

        for frag in 0..3usize {
            let base = frag * N;
            let mut sketch = TDigestMut::default(); // k=200
            for i in 0..N {
                let v = (base + i) as f64;
                sketch.update(v);
                all_values.push(v);
            }
            let mut map: BTreeMap<String, Vec<u8>> = BTreeMap::new();
            map.insert("v".to_string(), sketch.serialize());
            maps.push(map);
        }

        let refs: Vec<&BTreeMap<String, Vec<u8>>> = maps.iter().collect();
        let p95_est = tdigest_quantile_estimate(&refs, "v", 0.95)
            .expect("merged quantile estimate must succeed");

        // Compute rank of the returned value over the combined data.
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let rank_pos = all_values.partition_point(|&x| x <= p95_est);
        let actual_rank = rank_pos as f64 / all_values.len() as f64;

        let rank_err = (actual_rank - 0.95f64).abs();
        assert!(
            rank_err <= 0.03,
            "Merged T-Digest p95 estimate {p95_est:.1}: rank {actual_rank:.4} vs target 0.95, \
             rank error {rank_err:.4} must be ≤ 0.03 (T-Digest k=200 bound)"
        );
    }
}
