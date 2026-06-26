//! Custom lookup join execution plan for tiny build sides.
//!
//! `LookupJoinExec` materializes the build side into a `HashMap` keyed by join-key
//! `ScalarValue` and streams the probe side one partition at a time.  For every probe
//! row it evaluates the probe-key expression, looks up the list of matching build-row
//! indices, and emits the combined row.  Only inner equi-join semantics are supported:
//! rows without a match are dropped, and duplicate build keys produce a Cartesian
//! product for matching probe rows.
//!
//! The optimizer rule only creates a `LookupJoinExec` when the original
//! `HashJoinExec` uses `NullEquality::NullEqualsNothing`.  This implementation
//! does not implement null-matching semantics: null build keys are omitted when
//! constructing the lookup map and null probe keys never match.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use arrow::array::{new_empty_array, ArrayRef, RecordBatch, UInt32Array};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::SchemaRef;
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

/// Materialized build side shared across probe partitions.
struct BuildIndex {
    /// Concatenated build rows, used to project build columns into the output.
    batch: RecordBatch,
    /// Map from join-key value to the row indices in `batch` that have that key.
    map: HashMap<ScalarValue, Vec<u32>>,
}

/// Inner equi-join execution plan with a tiny, materialized build side.
///
/// The output schema is expected to be the build-side columns followed by the
/// probe-side columns, matching [`HashJoinExec`]'s inner-join output ordering.
#[derive(Clone)]
pub struct LookupJoinExec {
    probe: Arc<dyn ExecutionPlan>,
    probe_key_expr: Arc<dyn PhysicalExpr>,
    output_schema: SchemaRef,
    build_index: Arc<BuildIndex>,
    properties: Arc<PlanProperties>,
}

impl LookupJoinExec {
    /// Create a new [`LookupJoinExec`].
    ///
    /// `build_keys` must contain one scalar join-key value per build row, in the
    /// same order as the rows across `build_batches`.  Null keys are ignored.
    ///
    /// `output_schema` must contain the build-side columns followed by the
    /// probe-side columns.
    pub fn try_new(
        probe: Arc<dyn ExecutionPlan>,
        probe_key_expr: Arc<dyn PhysicalExpr>,
        build_keys: Vec<ScalarValue>,
        build_schema: SchemaRef,
        build_batches: Vec<RecordBatch>,
        output_schema: SchemaRef,
    ) -> Result<Self> {
        let build_index = Arc::new(build_index(build_keys, build_schema, build_batches)?);

        let probe_schema = probe.schema();
        if output_schema.fields().len()
            != build_index.batch.schema().fields().len() + probe_schema.fields().len()
        {
            return Err(crate::QueryError::Other(
                "LookupJoinExec output schema does not match build + probe columns".into(),
            ));
        }

        let properties = PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&output_schema)),
            probe.properties().partitioning.clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            probe,
            probe_key_expr,
            output_schema,
            build_index,
            properties: Arc::new(properties),
        })
    }

    /// Return the output schema.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }
}

impl fmt::Debug for LookupJoinExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupJoinExec")
            .field("build_rows", &self.build_index.batch.num_rows())
            .field("output_schema", &self.output_schema)
            .finish_non_exhaustive()
    }
}

impl DisplayAs for LookupJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "LookupJoinExec: build_rows={}",
            self.build_index.batch.num_rows()
        )
    }
}

impl ExecutionPlan for LookupJoinExec {
    fn name(&self) -> &str {
        "LookupJoinExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.probe]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DFResult<Arc<dyn ExecutionPlan>> {
        let mut me = (*self).clone();
        me.probe = Arc::clone(&children[0]);
        me.properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(Arc::clone(&me.output_schema)),
            me.probe.properties().partitioning.clone(),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
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

        // Empty build side can never produce output rows.
        if self.build_index.map.is_empty() {
            return Ok(Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.output_schema),
                futures::stream::empty(),
            )));
        }

        let input_stream = self.probe.execute(partition, context)?;
        let output_schema = Arc::clone(&self.output_schema);
        let probe_key_expr = Arc::clone(&self.probe_key_expr);
        let build_index = Arc::clone(&self.build_index);

        let stream = input_stream.try_filter_map(move |batch| {
            let output_schema = Arc::clone(&output_schema);
            let probe_key_expr = Arc::clone(&probe_key_expr);
            let build_index = Arc::clone(&build_index);

            async move {
                let output =
                    build_output_batch(&batch, &probe_key_expr, &build_index, &output_schema)
                        .map_err(datafusion::error::DataFusionError::from)?;
                if output.num_rows() == 0 {
                    Ok(None)
                } else {
                    Ok(Some(output))
                }
            }
        });

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.output_schema),
            stream,
        )))
    }
}

/// Materialize the build side into a single concatenated batch and a key-to-indices map.
fn build_index(
    build_keys: Vec<ScalarValue>,
    build_schema: SchemaRef,
    build_batches: Vec<RecordBatch>,
) -> Result<BuildIndex> {
    let batch = if build_batches.is_empty() {
        let columns: Vec<ArrayRef> = build_schema
            .fields()
            .iter()
            .map(|f| new_empty_array(f.data_type()))
            .collect();
        RecordBatch::try_new(build_schema, columns).map_err(crate::QueryError::Arrow)?
    } else {
        concat_batches(&build_schema, &build_batches).map_err(crate::QueryError::Arrow)?
    };

    if build_keys.len() != batch.num_rows() {
        return Err(crate::QueryError::Other(format!(
            "LookupJoinExec build_keys length ({}) does not match build rows ({})",
            build_keys.len(),
            batch.num_rows()
        )));
    }

    let mut map: HashMap<ScalarValue, Vec<u32>> = HashMap::new();
    for (idx, key) in build_keys.iter().enumerate() {
        if !key.is_null() {
            map.entry(key.clone()).or_default().push(idx as u32);
        }
    }

    Ok(BuildIndex { batch, map })
}

/// Build one output batch from a probe batch and the materialized build side.
fn build_output_batch(
    probe_batch: &RecordBatch,
    probe_key_expr: &Arc<dyn PhysicalExpr>,
    build_index: &BuildIndex,
    output_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let key_value = probe_key_expr
        .evaluate(probe_batch)
        .map_err(crate::QueryError::DataFusion)?;
    let key_array = match key_value {
        ColumnarValue::Array(arr) => arr,
        ColumnarValue::Scalar(scalar) => scalar
            .to_array_of_size(probe_batch.num_rows())
            .map_err(crate::QueryError::DataFusion)?,
    };

    let build_col_count = build_index.batch.num_columns();

    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for probe_row in 0..probe_batch.num_rows() {
        let key = ScalarValue::try_from_array(&key_array, probe_row)
            .map_err(crate::QueryError::DataFusion)?;
        if key.is_null() {
            continue;
        }
        if let Some(build_rows) = build_index.map.get(&key) {
            for &build_row in build_rows {
                pairs.push((probe_row as u32, build_row));
            }
        }
    }

    if pairs.is_empty() {
        let empty_columns: Vec<ArrayRef> = output_schema
            .fields()
            .iter()
            .map(|f| new_empty_array(f.data_type()))
            .collect();
        return RecordBatch::try_new(Arc::clone(output_schema), empty_columns)
            .map_err(crate::QueryError::Arrow);
    }

    let probe_indices = UInt32Array::from(pairs.iter().map(|(p, _)| *p).collect::<Vec<_>>());
    let build_indices = UInt32Array::from(pairs.iter().map(|(_, b)| *b).collect::<Vec<_>>());

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for out_idx in 0..output_schema.fields().len() {
        let source_array: &ArrayRef = if out_idx < build_col_count {
            build_index.batch.column(out_idx)
        } else {
            probe_batch.column(out_idx - build_col_count)
        };
        let indices: &UInt32Array = if out_idx < build_col_count {
            &build_indices
        } else {
            &probe_indices
        };
        let taken = take(source_array.as_ref(), indices, None).map_err(crate::QueryError::Arrow)?;
        columns.push(taken);
    }

    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(crate::QueryError::Arrow)
}
