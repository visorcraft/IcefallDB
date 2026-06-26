//! Single-pass streaming group-by aggregate for pre-sorted inputs.
//!
//! `StreamingGroupByExec` assumes its input is ordered by the group key(s).  It
//! evaluates the group expressions on each incoming batch, maintains one running
//! aggregate state per distinct group, and emits a batch of completed groups
//! whenever the group key changes between consecutive batches.  At end-of-stream
//! the remaining groups are flushed.
//!
//! Supported aggregates: `COUNT(*)`, `COUNT(col)`, `SUM(col)`, `AVG(col)`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    RecordBatch, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::row::{RowConverter, SortField};
use datafusion::common::Result as DFResult;
use datafusion::common::ScalarValue;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::ColumnarValue;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use futures::TryStreamExt;

use crate::Result;

/// Aggregate types supported by the streaming group-by operator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AggType {
    /// `COUNT(*)` or `COUNT(col)`.
    Count,
    /// `SUM(col)`.
    Sum,
    /// `AVG(col)`.
    Avg,
}

/// Exact typed accumulator for a single aggregate expression.
///
/// Signed integers are accumulated in `i128`, unsigned integers in `u128`, and
/// floats in `f64`.  The final output array is built in the original schema
/// type.
#[derive(Clone, Debug)]
enum Accumulator {
    /// No numeric accumulator (used for `Count`).
    Count,
    /// Signed integer accumulator.
    I128(Option<i128>),
    /// Unsigned integer accumulator.
    U128(Option<u128>),
    /// Floating-point accumulator.
    F64(Option<f64>),
}

impl Accumulator {
    /// Create an accumulator matching the aggregate input type.
    fn new(input_type: &DataType, agg_type: AggType) -> Result<Self> {
        match agg_type {
            AggType::Count => Ok(Accumulator::Count),
            AggType::Sum | AggType::Avg => match input_type {
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                    Ok(Accumulator::I128(None))
                }
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                    Ok(Accumulator::U128(None))
                }
                DataType::Float32 | DataType::Float64 => Ok(Accumulator::F64(None)),
                other => Err(crate::QueryError::Other(format!(
                    "unsupported numeric type for streaming aggregation: {other}"
                ))),
            },
        }
    }
}

/// Group-by aggregate for inputs that are already sorted by the group key(s).
#[derive(Clone)]
pub struct StreamingGroupByExec {
    input: Arc<dyn ExecutionPlan>,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    aggr_exprs: Vec<Arc<dyn PhysicalExpr>>,
    aggr_types: Vec<AggType>,
    /// Data type of each aggregate input expression.
    aggr_input_types: Vec<DataType>,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

impl StreamingGroupByExec {
    /// Create a new `StreamingGroupByExec`.
    ///
    /// `group_exprs` produce the group key columns.  `aggr_exprs` produce the
    /// values fed into the aggregate; for `COUNT(*)` this should be a literal
    /// `1`, for columnar aggregates it should be a column expression.
    /// `schema` is the output schema (group columns followed by aggregate
    /// columns).
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        group_exprs: Vec<Arc<dyn PhysicalExpr>>,
        aggr_exprs: Vec<(AggType, Arc<dyn PhysicalExpr>)>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let (types, exprs): (Vec<_>, Vec<_>) = aggr_exprs.into_iter().unzip();
        let input_schema = input.schema();
        let aggr_input_types = exprs
            .iter()
            .map(|expr| {
                expr.data_type(&input_schema)
                    .map_err(crate::QueryError::DataFusion)
            })
            .collect::<Result<Vec<_>>>()?;

        // Fail fast on unsupported SUM/AVG input types.
        for (input_type, agg_type) in aggr_input_types.iter().zip(&types) {
            let _ = Accumulator::new(input_type, *agg_type)?;
        }

        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&schema)),
            input.properties().partitioning.clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            input,
            group_exprs,
            aggr_exprs: exprs,
            aggr_types: types,
            aggr_input_types,
            schema,
            properties: Arc::new(properties),
        })
    }

    /// Return the output schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    /// Return the group expressions.
    pub fn group_exprs(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.group_exprs
    }
}

impl fmt::Debug for StreamingGroupByExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamingGroupByExec")
            .field("group_exprs", &self.group_exprs.len())
            .field("aggr_exprs", &self.aggr_exprs.len())
            .field("schema", &self.schema)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for StreamingGroupByExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StreamingGroupByExec: group_exprs={}, aggr_exprs={}",
            self.group_exprs.len(),
            self.aggr_exprs.len()
        )
    }
}

impl ExecutionPlan for StreamingGroupByExec {
    fn name(&self) -> &str {
        "StreamingGroupByExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let mut me = (*self).clone();
        me.input = Arc::clone(&children[0]);
        Ok(Arc::new(me))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let partition_count = self.properties.partitioning.partition_count();
        if partition >= partition_count {
            return Err(datafusion::error::DataFusionError::Internal(format!(
                "Invalid partition index {partition}, partition count {partition_count}"
            )));
        }

        let input_stream = self.input.execute(partition, context)?;
        let schema = Arc::clone(&self.schema);
        let group_exprs = self.group_exprs.clone();
        let aggr_exprs = self.aggr_exprs.clone();
        let aggr_types = self.aggr_types.clone();
        let aggr_input_types = self.aggr_input_types.clone();
        let input_schema = self.input.schema();

        let sort_fields: Vec<SortField> = group_exprs
            .iter()
            .map(|e| {
                let dtype = e.data_type(&input_schema)?;
                Ok(SortField::new(dtype))
            })
            .collect::<DFResult<Vec<_>>>()?;
        let row_converter = RowConverter::new(sort_fields).map_err(crate::QueryError::Arrow)?;

        let initial_state = StreamingState {
            groups: BTreeMap::new(),
            drained_up_to: None,
            row_converter,
            pending: Vec::new(),
            done_reading: false,
        };

        // `SendableRecordBatchStream` is not `Clone`, so share it with an async
        // mutex.  The streaming group-by operates on a single partition, so the
        // mutex is uncontended except across await points.
        let input_stream = Arc::new(tokio::sync::Mutex::new(input_stream));

        let stream = futures::stream::try_unfold(initial_state, move |mut state| {
            let schema = Arc::clone(&schema);
            let group_exprs = group_exprs.clone();
            let aggr_exprs = aggr_exprs.clone();
            let aggr_types = aggr_types.clone();
            let aggr_input_types = aggr_input_types.clone();
            let input_stream = Arc::clone(&input_stream);

            async move {
                loop {
                    if let Some(batch) = state.pending.pop() {
                        return Ok(Some((batch, state)));
                    }
                    if state.done_reading {
                        return Ok(None);
                    }

                    let mut stream_guard = input_stream.lock().await;
                    let next = stream_guard.try_next().await?;
                    drop(stream_guard);

                    match next {
                        Some(batch) => {
                            process_batch(
                                &batch,
                                &group_exprs,
                                &aggr_exprs,
                                &aggr_types,
                                &aggr_input_types,
                                &schema,
                                &mut state,
                            )?;
                        }
                        None => {
                            flush_remaining(&aggr_types, &aggr_input_types, &schema, &mut state)?;
                            state.done_reading = true;
                        }
                    }
                }
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.schema),
            stream,
        )))
    }
}

/// Running aggregate state for a single aggregate expression within one group.
#[derive(Clone, Debug)]
struct PerAggState {
    /// Count of rows (valid for `Count`) or non-null rows (valid for `Avg`).
    count: i64,
    /// Typed running sum (valid for `Sum` and `Avg`).
    accumulator: Accumulator,
}

impl PerAggState {
    fn new(input_type: &DataType, agg_type: AggType) -> Result<Self> {
        Ok(Self {
            count: 0,
            accumulator: Accumulator::new(input_type, agg_type)?,
        })
    }
}

/// Running state for a single group.
#[derive(Clone, Debug)]
struct GroupState {
    per_agg: Vec<PerAggState>,
    /// Scalar values of the group key, used to rebuild group columns on flush.
    key_scalars: Vec<ScalarValue>,
}

/// Internal state kept across `try_unfold` iterations.
struct StreamingState {
    /// Groups stored in sorted key order so draining completed keys is a single
    /// `BTreeMap::split_off`.
    groups: BTreeMap<Vec<u8>, GroupState>,
    /// Exclusive upper bound: all groups with keys strictly less than this have
    /// already been emitted.
    drained_up_to: Option<Vec<u8>>,
    row_converter: RowConverter,
    pending: Vec<RecordBatch>,
    done_reading: bool,
}

/// Evaluate the group and aggregate expressions for `batch`, update running
/// state, and emit any groups that are known to be complete.
fn process_batch(
    batch: &RecordBatch,
    group_exprs: &[Arc<dyn PhysicalExpr>],
    aggr_exprs: &[Arc<dyn PhysicalExpr>],
    aggr_types: &[AggType],
    aggr_input_types: &[DataType],
    output_schema: &SchemaRef,
    state: &mut StreamingState,
) -> Result<()> {
    let group_arrays = evaluate_exprs(batch, group_exprs)?;
    let aggr_arrays = evaluate_exprs(batch, aggr_exprs)?;

    let rows = state.row_converter.convert_columns(&group_arrays)?;
    if rows.num_rows() == 0 {
        return Ok(());
    }
    let first_key = rows.row(0).as_ref().to_vec();

    // Verify sortedness and drain completed groups.
    if let Some(ref drained) = state.drained_up_to {
        match first_key.as_slice().cmp(drained.as_slice()) {
            std::cmp::Ordering::Less => {
                return Err(crate::QueryError::Other(
                    "StreamingGroupByExec input is not sorted by group key".into(),
                ));
            }
            std::cmp::Ordering::Greater => {
                drain_groups_lt(
                    &first_key,
                    aggr_types,
                    aggr_input_types,
                    output_schema,
                    state,
                )?;
                state.drained_up_to = Some(first_key.clone());
            }
            std::cmp::Ordering::Equal => {}
        }
    } else {
        state.drained_up_to = Some(first_key.clone());
    }

    // Update running aggregates.
    for row_idx in 0..rows.num_rows() {
        let key = rows.row(row_idx).as_ref().to_vec();
        if let std::collections::btree_map::Entry::Vacant(vacant) = state.groups.entry(key.clone())
        {
            let key_scalars = group_arrays
                .iter()
                .map(|arr| {
                    ScalarValue::try_from_array(arr, row_idx).map_err(crate::QueryError::DataFusion)
                })
                .collect::<Result<Vec<_>>>()?;
            let per_agg = aggr_input_types
                .iter()
                .zip(aggr_types.iter())
                .map(|(input_type, agg_type)| PerAggState::new(input_type, *agg_type))
                .collect::<Result<Vec<_>>>()?;
            vacant.insert(GroupState {
                per_agg,
                key_scalars,
            });
        }
        let entry = state.groups.get_mut(&key).ok_or_else(|| {
            crate::QueryError::Other(
                "streaming group-by internal error: group state missing after insertion".into(),
            )
        })?;

        for (agg_idx, typ) in aggr_types.iter().enumerate() {
            update_per_agg_state(
                &mut entry.per_agg[agg_idx],
                &aggr_arrays[agg_idx],
                row_idx,
                *typ,
            )?;
        }
    }

    Ok(())
}

/// Evaluate a slice of physical expressions against a batch and return the
/// resulting arrays.
fn evaluate_exprs(batch: &RecordBatch, exprs: &[Arc<dyn PhysicalExpr>]) -> Result<Vec<ArrayRef>> {
    exprs
        .iter()
        .map(|expr| {
            Ok(
                match expr
                    .evaluate(batch)
                    .map_err(crate::QueryError::DataFusion)?
                {
                    ColumnarValue::Array(arr) => arr,
                    ColumnarValue::Scalar(scalar) => scalar
                        .to_array_of_size(batch.num_rows())
                        .map_err(crate::QueryError::DataFusion)?,
                },
            )
        })
        .collect()
}

/// Update a single per-aggregate state with the value at `row_idx`.
fn update_per_agg_state(
    state: &mut PerAggState,
    array: &ArrayRef,
    row_idx: usize,
    typ: AggType,
) -> Result<()> {
    match typ {
        AggType::Count => {
            if array.is_valid(row_idx) {
                state.count += 1;
            }
        }
        AggType::Sum | AggType::Avg => {
            if array.is_valid(row_idx) {
                match &mut state.accumulator {
                    Accumulator::I128(acc) => {
                        let v = extract_i128(array, row_idx)?;
                        *acc = Some(acc.unwrap_or(0) + v);
                    }
                    Accumulator::U128(acc) => {
                        let v = extract_u128(array, row_idx)?;
                        *acc = Some(acc.unwrap_or(0) + v);
                    }
                    Accumulator::F64(acc) => {
                        let v = extract_f64(array, row_idx)?;
                        *acc = Some(acc.unwrap_or(0.0) + v);
                    }
                    Accumulator::Count => {
                        return Err(crate::QueryError::Other(
                            "streaming group-by internal error: Count accumulator used for Sum/Avg"
                                .into(),
                        ));
                    }
                }
                if typ == AggType::Avg {
                    state.count += 1;
                }
            }
        }
    }
    Ok(())
}

/// Extract a signed integer value as `i128` from a supported array type.
fn extract_i128(array: &ArrayRef, row_idx: usize) -> Result<i128> {
    let err = |msg: &str| crate::QueryError::Other(msg.to_string());
    match array.data_type() {
        DataType::Int8 => Ok(array
            .as_any()
            .downcast_ref::<Int8Array>()
            .ok_or_else(|| err("failed to downcast Int8 array"))?
            .value(row_idx)
            .into()),
        DataType::Int16 => Ok(array
            .as_any()
            .downcast_ref::<Int16Array>()
            .ok_or_else(|| err("failed to downcast Int16 array"))?
            .value(row_idx)
            .into()),
        DataType::Int32 => Ok(array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| err("failed to downcast Int32 array"))?
            .value(row_idx)
            .into()),
        DataType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| err("failed to downcast Int64 array"))?
            .value(row_idx)
            .into()),
        other => Err(crate::QueryError::Other(format!(
            "expected signed integer array, got {other}"
        ))),
    }
}

/// Extract an unsigned integer value as `u128` from a supported array type.
fn extract_u128(array: &ArrayRef, row_idx: usize) -> Result<u128> {
    let err = |msg: &str| crate::QueryError::Other(msg.to_string());
    match array.data_type() {
        DataType::UInt8 => Ok(array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| err("failed to downcast UInt8 array"))?
            .value(row_idx)
            .into()),
        DataType::UInt16 => Ok(array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .ok_or_else(|| err("failed to downcast UInt16 array"))?
            .value(row_idx)
            .into()),
        DataType::UInt32 => Ok(array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| err("failed to downcast UInt32 array"))?
            .value(row_idx)
            .into()),
        DataType::UInt64 => Ok(array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| err("failed to downcast UInt64 array"))?
            .value(row_idx)
            .into()),
        other => Err(crate::QueryError::Other(format!(
            "expected unsigned integer array, got {other}"
        ))),
    }
}

/// Extract a floating-point value as `f64` from a supported array type.
fn extract_f64(array: &ArrayRef, row_idx: usize) -> Result<f64> {
    let err = |msg: &str| crate::QueryError::Other(msg.to_string());
    match array.data_type() {
        DataType::Int8 => Ok(array
            .as_any()
            .downcast_ref::<Int8Array>()
            .ok_or_else(|| err("failed to downcast Int8 array"))?
            .value(row_idx) as f64),
        DataType::Int16 => Ok(array
            .as_any()
            .downcast_ref::<Int16Array>()
            .ok_or_else(|| err("failed to downcast Int16 array"))?
            .value(row_idx) as f64),
        DataType::Int32 => Ok(array
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| err("failed to downcast Int32 array"))?
            .value(row_idx) as f64),
        DataType::Int64 => Ok(array
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| err("failed to downcast Int64 array"))?
            .value(row_idx) as f64),
        DataType::UInt8 => Ok(array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| err("failed to downcast UInt8 array"))?
            .value(row_idx) as f64),
        DataType::UInt16 => Ok(array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .ok_or_else(|| err("failed to downcast UInt16 array"))?
            .value(row_idx) as f64),
        DataType::UInt32 => Ok(array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| err("failed to downcast UInt32 array"))?
            .value(row_idx) as f64),
        DataType::UInt64 => Ok(array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| err("failed to downcast UInt64 array"))?
            .value(row_idx) as f64),
        DataType::Float32 => Ok(array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| err("failed to downcast Float32 array"))?
            .value(row_idx) as f64),
        DataType::Float64 => Ok(array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| err("failed to downcast Float64 array"))?
            .value(row_idx)),
        other => Err(crate::QueryError::Other(format!(
            "unsupported numeric type for streaming aggregation: {other}"
        ))),
    }
}

/// Drain all groups whose key is strictly less than `threshold` and append an
/// output batch to `state.pending`.
///
/// `BTreeMap::split_off(&threshold)` returns the keys `>= threshold`; the
/// original map keeps keys `< threshold`.  We swap the two halves so that the
/// live state keeps the not-yet-drained keys.
fn drain_groups_lt(
    threshold: &[u8],
    aggr_types: &[AggType],
    aggr_input_types: &[DataType],
    output_schema: &SchemaRef,
    state: &mut StreamingState,
) -> Result<()> {
    if state.groups.is_empty() {
        return Ok(());
    }

    let remaining = state.groups.split_off(threshold);
    let completed = std::mem::replace(&mut state.groups, remaining);
    if completed.is_empty() {
        return Ok(());
    }

    let to_emit: Vec<(Vec<u8>, GroupState)> = completed.into_iter().collect();
    state.pending.push(build_output_batch(
        to_emit,
        aggr_types,
        aggr_input_types,
        output_schema,
    )?);
    Ok(())
}

/// Flush all remaining groups at end-of-stream.
fn flush_remaining(
    aggr_types: &[AggType],
    aggr_input_types: &[DataType],
    output_schema: &SchemaRef,
    state: &mut StreamingState,
) -> Result<()> {
    if state.groups.is_empty() {
        return Ok(());
    }
    let to_emit: Vec<(Vec<u8>, GroupState)> =
        std::mem::take(&mut state.groups).into_iter().collect();
    state.pending.push(build_output_batch(
        to_emit,
        aggr_types,
        aggr_input_types,
        output_schema,
    )?);
    Ok(())
}

/// Build an output batch from a collection of completed groups.
///
/// `groups` is assumed to be sorted by key, which is true for batches drained
/// from a `BTreeMap`.
fn build_output_batch(
    groups: Vec<(Vec<u8>, GroupState)>,
    aggr_types: &[AggType],
    aggr_input_types: &[DataType],
    output_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let num_group_cols = groups
        .first()
        .map(|(_, g)| g.key_scalars.len())
        .unwrap_or(0);

    // Build group key columns.
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col_idx in 0..num_group_cols {
        let scalars = groups
            .iter()
            .map(|(_, g)| g.key_scalars[col_idx].clone())
            .collect::<Vec<_>>();
        let array = ScalarValue::iter_to_array(scalars).map_err(crate::QueryError::DataFusion)?;
        columns.push(array);
    }

    // Build aggregate columns.
    for (agg_idx, typ) in aggr_types.iter().enumerate() {
        let output_field = output_schema.field(num_group_cols + agg_idx);
        let array = build_agg_array(
            &groups,
            agg_idx,
            *typ,
            &aggr_input_types[agg_idx],
            output_field.data_type(),
        )?;
        columns.push(array);
    }

    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(crate::QueryError::Arrow)
}

/// Build one aggregate output column.
fn build_agg_array(
    groups: &[(Vec<u8>, GroupState)],
    agg_idx: usize,
    typ: AggType,
    input_type: &DataType,
    output_type: &DataType,
) -> Result<ArrayRef> {
    match typ {
        AggType::Count => {
            let values: Vec<Option<i64>> = groups
                .iter()
                .map(|(_, g)| Some(g.per_agg[agg_idx].count))
                .collect();
            Ok(Arc::new(Int64Array::from(values)) as ArrayRef)
        }
        AggType::Sum => build_sum_array(groups, agg_idx, input_type, output_type),
        AggType::Avg => build_avg_array(groups, agg_idx, input_type, output_type),
    }
}

/// Build a `SUM` output column from typed accumulators.
fn build_sum_array(
    groups: &[(Vec<u8>, GroupState)],
    agg_idx: usize,
    input_type: &DataType,
    output_type: &DataType,
) -> Result<ArrayRef> {
    let err = |msg: &str| crate::QueryError::Other(msg.to_string());

    match input_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            let values: Vec<Option<i128>> = groups
                .iter()
                .map(|(_, g)| match &g.per_agg[agg_idx].accumulator {
                    Accumulator::I128(v) => Ok(*v),
                    other => Err(err(&format!(
                        "streaming group-by internal error: expected I128 accumulator, got {other:?}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            build_i128_output(values, output_type)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            let values: Vec<Option<u128>> = groups
                .iter()
                .map(|(_, g)| match &g.per_agg[agg_idx].accumulator {
                    Accumulator::U128(v) => Ok(*v),
                    other => Err(err(&format!(
                        "streaming group-by internal error: expected U128 accumulator, got {other:?}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            build_u128_output(values, output_type)
        }
        DataType::Float32 | DataType::Float64 => {
            let values: Vec<Option<f64>> = groups
                .iter()
                .map(|(_, g)| match &g.per_agg[agg_idx].accumulator {
                    Accumulator::F64(v) => Ok(*v),
                    other => Err(err(&format!(
                        "streaming group-by internal error: expected F64 accumulator, got {other:?}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            build_f64_output(values, output_type)
        }
        other => Err(crate::QueryError::Other(format!(
            "unsupported SUM input type: {other}"
        ))),
    }
}

/// Build an `AVG` output column from typed accumulators.
fn build_avg_array(
    groups: &[(Vec<u8>, GroupState)],
    agg_idx: usize,
    _input_type: &DataType,
    output_type: &DataType,
) -> Result<ArrayRef> {
    let err = |msg: &str| crate::QueryError::Other(msg.to_string());

    let mut values = Vec::with_capacity(groups.len());
    for (_, group) in groups {
        let state = &group.per_agg[agg_idx];
        if state.count > 0 {
            let sum = match &state.accumulator {
                Accumulator::I128(Some(v)) => *v as f64,
                Accumulator::U128(Some(v)) => *v as f64,
                Accumulator::F64(Some(v)) => *v,
                Accumulator::I128(None) | Accumulator::U128(None) | Accumulator::F64(None) => {
                    return Err(err(
                        "streaming group-by internal error: empty accumulator with non-zero count",
                    ))
                }
                Accumulator::Count => {
                    return Err(err(
                        "streaming group-by internal error: Count accumulator used for Avg",
                    ))
                }
            };
            values.push(Some(sum / state.count as f64));
        } else {
            values.push(None);
        }
    }

    match output_type {
        DataType::Float64 => Ok(Arc::new(Float64Array::from(values)) as ArrayRef),
        DataType::Float32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as f32))
                .collect::<Float32Array>(),
        ) as ArrayRef),
        other => Err(crate::QueryError::Other(format!(
            "unsupported AVG output type: {other}"
        ))),
    }
}

/// Convert signed integer accumulator values to the requested output type.
fn build_i128_output(values: Vec<Option<i128>>, output_type: &DataType) -> Result<ArrayRef> {
    match output_type {
        DataType::Int8 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i8))
                .collect::<Int8Array>(),
        ) as ArrayRef),
        DataType::Int16 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i16))
                .collect::<Int16Array>(),
        ) as ArrayRef),
        DataType::Int32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i32))
                .collect::<Int32Array>(),
        ) as ArrayRef),
        DataType::Int64 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i64))
                .collect::<Int64Array>(),
        ) as ArrayRef),
        DataType::Float32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as f32))
                .collect::<Float32Array>(),
        ) as ArrayRef),
        DataType::Float64 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as f64))
                .collect::<Float64Array>(),
        ) as ArrayRef),
        other => Err(crate::QueryError::Other(format!(
            "unsupported SUM output type: {other}"
        ))),
    }
}

/// Convert unsigned integer accumulator values to the requested output type.
fn build_u128_output(values: Vec<Option<u128>>, output_type: &DataType) -> Result<ArrayRef> {
    match output_type {
        DataType::UInt8 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u8))
                .collect::<UInt8Array>(),
        ) as ArrayRef),
        DataType::UInt16 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u16))
                .collect::<UInt16Array>(),
        ) as ArrayRef),
        DataType::UInt32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u32))
                .collect::<UInt32Array>(),
        ) as ArrayRef),
        DataType::UInt64 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u64))
                .collect::<UInt64Array>(),
        ) as ArrayRef),
        DataType::Float32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as f32))
                .collect::<Float32Array>(),
        ) as ArrayRef),
        DataType::Float64 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as f64))
                .collect::<Float64Array>(),
        ) as ArrayRef),
        other => Err(crate::QueryError::Other(format!(
            "unsupported SUM output type: {other}"
        ))),
    }
}

/// Convert floating-point accumulator values to the requested output type.
fn build_f64_output(values: Vec<Option<f64>>, output_type: &DataType) -> Result<ArrayRef> {
    match output_type {
        DataType::Float64 => Ok(Arc::new(Float64Array::from(values)) as ArrayRef),
        DataType::Float32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as f32))
                .collect::<Float32Array>(),
        ) as ArrayRef),
        DataType::Int64 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i64))
                .collect::<Int64Array>(),
        ) as ArrayRef),
        DataType::Int32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i32))
                .collect::<Int32Array>(),
        ) as ArrayRef),
        DataType::Int16 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i16))
                .collect::<Int16Array>(),
        ) as ArrayRef),
        DataType::Int8 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as i8))
                .collect::<Int8Array>(),
        ) as ArrayRef),
        DataType::UInt64 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u64))
                .collect::<UInt64Array>(),
        ) as ArrayRef),
        DataType::UInt32 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u32))
                .collect::<UInt32Array>(),
        ) as ArrayRef),
        DataType::UInt16 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u16))
                .collect::<UInt16Array>(),
        ) as ArrayRef),
        DataType::UInt8 => Ok(Arc::new(
            values
                .into_iter()
                .map(|o| o.map(|v| v as u8))
                .collect::<UInt8Array>(),
        ) as ArrayRef),
        other => Err(crate::QueryError::Other(format!(
            "unsupported SUM output type: {other}"
        ))),
    }
}
