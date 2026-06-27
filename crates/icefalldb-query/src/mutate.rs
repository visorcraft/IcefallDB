//! Row-mutation helpers for IcefallDB.
//!
//! The first step of any DELETE or UPDATE is **locating** the rows that match a
//! predicate.  [`locate_matches`] does this by running a
//! `SELECT _rowid, _rowaddr FROM t WHERE predicate` query through the existing
//! DataFusion engine, which means pruning, index acceleration, and
//! deletion-vector filtering are all applied automatically.  Only live rows
//! that satisfy the predicate are returned.
//!
//! [`execute_sql`] is the SQL-level entry point: it parses a `DELETE FROM t
//! WHERE p` or `UPDATE t SET col = expr WHERE p` statement, converts the
//! predicate to a DataFusion [`Expr`], and routes to the appropriate commit
//! path (`Writer::commit_deletes` for DELETE, `Writer::commit_update` for
//! UPDATE), then refreshes the table registration so subsequent queries see
//! the new snapshot, and returns the affected-row count.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, Int64Array, RecordBatch};
use arrow::datatypes::{DataType, UInt64Type};
use datafusion::common::{DFSchema, ScalarValue};
use datafusion::logical_expr::{col, Expr, Operator};
use datafusion::prelude::SessionContext;
use datafusion::sql::parser::DFParser;
use datafusion::sql::sqlparser::ast::{
    AssignmentTarget, BinaryOperator, Expr as SqlExpr, FromTable, MergeAction, MergeClauseKind,
    Statement as SQLStatement, TableFactor,
};
use icefalldb_core::database_catalog::DatabaseCatalog;
use icefalldb_core::storage::Storage;
use icefalldb_core::{CommitDelta, MatchLoc, Writer};

use crate::{IcefallDBTableProvider, QueryError, Result};
use icefalldb_core::IcefallDBError;

// ── Source dedup + cardinality check ───────────────────────────────────

/// How to handle a MERGE SOURCE that contains two rows with the same key value.
///
/// SQL-standard MERGE is undefined when the source relation contains duplicate
/// match-key values (a single target row cannot be updated by two source rows).
/// `DupPolicy` lets the caller opt-in to a lenient last-writer-wins mode for
/// use cases where the source is known to be a "last event wins" changelog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DupPolicy {
    /// Return `Err` on the second occurrence of a key — SQL-standard behaviour.
    Error,
    /// Overwrite the earlier row with the later one; the last row in source
    /// order for a given key survives.
    LastWriterWins,
}

/// A materialized source row: the full column values extracted from a
/// `RecordBatch` at a specific row index.
///
/// Storing the per-column `ScalarValue`s (rather than a `(Arc<RecordBatch>,
/// usize)` reference) avoids lifetime issues and keeps the deduped map
/// self-contained, which is convenient for the matched/unmatched routing
/// step that will consume it.
#[derive(Debug, Clone)]
pub struct RecordBatchRow {
    /// Column names in schema order (same as the source `RecordBatch`).
    pub columns: Vec<String>,
    /// Scalar value for each column, aligned with `columns`.
    pub values: Vec<ScalarValue>,
}

impl RecordBatchRow {
    /// Extract row `row_idx` from `batch`.
    pub fn from_batch(batch: &RecordBatch, row_idx: usize) -> crate::Result<Self> {
        let schema = batch.schema();
        let mut columns = Vec::with_capacity(batch.num_columns());
        let mut values = Vec::with_capacity(batch.num_columns());
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let sv = ScalarValue::try_from_array(batch.column(col_idx), row_idx)
                .map_err(|e| QueryError::Other(format!("ScalarValue extraction error: {e}")))?;
            columns.push(field.name().clone());
            values.push(sv);
        }
        Ok(Self { columns, values })
    }

    /// Return the value of a named column as `i64`, panicking if the column is
    /// absent or holds a non-`Int64` value.  Used in unit tests.
    #[cfg(test)]
    pub fn get_i64(&self, col_name: &str) -> i64 {
        let pos = self
            .columns
            .iter()
            .position(|c| c == col_name)
            .unwrap_or_else(|| panic!("column '{col_name}' not found"));
        match &self.values[pos] {
            ScalarValue::Int64(Some(v)) => *v,
            other => panic!("expected Int64, got {other:?}"),
        }
    }
}

/// An insertion-ordered, deduplicated view of a MERGE source `RecordBatch`,
/// keyed by the merge-key column value.
///
/// # Data structure
/// `indexmap` is only a transitive dependency of this crate (pulled in by
/// DataFusion/Arrow), so to avoid adding a new *direct* Cargo dependency the
/// map is represented as a `Vec<(ScalarValue, RecordBatchRow)>` (preserving
/// insertion order) plus a `HashMap<ScalarValue, usize>` index that maps each
/// key to its slot in the vec for O(1) lookup.
pub struct DedupedSource {
    /// Insertion-ordered key → row pairs.  Once an entry is inserted, its
    /// position in the vec never changes (LastWriterWins overwrites the row
    /// value in-place), so downstream code can iterate in stable source order.
    pub by_key: Vec<(ScalarValue, RecordBatchRow)>,
    /// Maps each key to its index in `by_key` for O(1) lookup.
    key_index: HashMap<ScalarValue, usize>,
}

impl DedupedSource {
    fn new() -> Self {
        Self {
            by_key: Vec::new(),
            key_index: HashMap::new(),
        }
    }

    /// Look up the row for `key`, if present.
    pub fn get(&self, key: &ScalarValue) -> Option<&RecordBatchRow> {
        self.key_index.get(key).map(|&i| &self.by_key[i].1)
    }

    /// Number of distinct keys.
    pub fn len(&self) -> usize {
        self.by_key.len()
    }

    /// True when no source rows were inserted.
    pub fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }
}

/// Fold `src` into a deduplicated, insertion-ordered map keyed by `key_col`.
///
/// # Null keys
/// A null value in `key_col` is rejected with `Err` regardless of `policy`.
/// A null merge key is semantically undefined (NULL ≠ NULL in SQL), so
/// allowing it would silently mis-route rows.
///
/// # Errors
/// - `key_col` is not found in `src`'s schema.
/// - Any row has a null value in `key_col`.
/// - `policy == DupPolicy::Error` and two rows share the same key value.
pub fn dedup_source(
    src: &RecordBatch,
    key_col: &str,
    policy: DupPolicy,
) -> crate::Result<DedupedSource> {
    // Locate the key column index.
    let schema = src.schema();
    let key_col_idx = schema.index_of(key_col).map_err(|_| {
        QueryError::Other(format!("MERGE key column '{key_col}' not found in source"))
    })?;

    let n_rows = src.num_rows();
    let mut out = DedupedSource::new();

    for row_idx in 0..n_rows {
        let key_sv = normalize_merge_key(
            ScalarValue::try_from_array(src.column(key_col_idx), row_idx).map_err(|e| {
                QueryError::Other(format!("ScalarValue extraction for key col: {e}"))
            })?,
        );

        // Null keys are always rejected.
        if key_sv.is_null() {
            return Err(QueryError::Other(format!(
                "MERGE source contains a NULL value in key column '{key_col}' at row {row_idx}; \
                 null merge keys are not permitted"
            )));
        }

        let row = RecordBatchRow::from_batch(src, row_idx)?;

        if let Some(&slot) = out.key_index.get(&key_sv) {
            // Key seen before.
            match policy {
                DupPolicy::Error => {
                    return Err(QueryError::Other(format!(
                        "MERGE source contains duplicate key {key_sv} (rows {slot} and {row_idx}); \
                         use DupPolicy::LastWriterWins to allow this"
                    )));
                }
                DupPolicy::LastWriterWins => {
                    // Overwrite the existing slot in-place (preserves insertion order).
                    out.by_key[slot].1 = row;
                }
            }
        } else {
            let slot = out.by_key.len();
            out.by_key.push((key_sv.clone(), row));
            out.key_index.insert(key_sv, slot);
        }
    }

    Ok(out)
}

/// Return the name of the unique btree index on `key` in `table`, or an error.
///
/// Scans the database-wide catalog for a btree index whose `table` and
/// `column` match the arguments **and** whose `unique` flag is `true`.
/// Returns the first such index name, or a descriptive `Err` if none exists.
///
/// This is the contract-check that MERGE requires: a MERGE INTO keyed on
/// `key` must have a unique index so probing for an existing row is cheap and
/// unambiguous.
pub async fn require_unique_key_index(
    storage: Arc<dyn Storage>,
    table: &str,
    key: &str,
) -> Result<String> {
    let catalog = DatabaseCatalog::new(Arc::clone(&storage));
    let data = catalog.load().await.map_err(QueryError::Core)?;

    for (name, entry) in &data.indexes {
        if entry.table == table
            && entry.column == key
            && entry.index_type == "btree"
            && entry.unique
        {
            return Ok(name.clone());
        }
    }

    Err(QueryError::Other(format!(
        "MERGE INTO {table} requires a UNIQUE index on {key}"
    )))
}

/// Result of committing a single mutation without refreshing the registered
/// provider. The optional `delta` is `None` when the statement was a no-op.
#[derive(Debug)]
struct MutationDelta {
    count: u64,
    delta: Option<CommitDelta>,
    table: String,
}

/// Result of committing a MERGE without refreshing the registered provider.
#[derive(Debug)]
struct MergeDelta {
    stats: MergeStats,
    delta: Option<CommitDelta>,
    table: String,
}

/// Apply a committed delta to the registered `IcefallDBTableProvider` for
/// `table_name`.
async fn apply_delta_to_provider(
    ctx: &SessionContext,
    table_name: &str,
    delta: &CommitDelta,
) -> Result<()> {
    let provider = ctx
        .table_provider(table_name)
        .await
        .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
    let provider = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<IcefallDBTableProvider>()
        .ok_or_else(|| {
            QueryError::Other("registered provider is not a IcefallDBTableProvider".into())
        })?;
    provider.apply_committed_delta(delta).await
}

/// Locate all live rows in `table` that satisfy `predicate`.
///
/// Internally this runs:
/// ```sql
/// SELECT _rowid, _rowaddr FROM <table> WHERE <predicate>
/// ```
/// through the existing DataFusion session.  The scan respects deletion vectors
/// (rows already deleted are not returned) and benefits from any index or
/// statistics-based pruning the engine applies.
///
/// Returns one [`MatchLoc`] per matching live row.
pub async fn locate_matches(
    ctx: &SessionContext,
    table: &str,
    predicate: Expr,
) -> Result<Vec<MatchLoc>> {
    // Fast path: an indexed point / `IN` predicate resolves rows directly from
    // the in-memory secondary index + row-id segments, with no DataFusion plan
    // or scan. This is the dominant cost for point DELETEs on a UNIQUE-indexed
    // column (a full scan otherwise).
    // Any miss or error falls back to the scan path below; the fast path is a
    // pure accelerator and must never be why a DELETE breaks.
    if let Ok(provider) = ctx.table_provider(table).await {
        if let Some(p) =
            (provider.as_ref() as &dyn std::any::Any).downcast_ref::<IcefallDBTableProvider>()
        {
            if let Ok(Some(locs)) = p.try_locate_by_index(&predicate).await {
                return Ok(locs);
            }
        }
    }

    let df = ctx
        .table(table)
        .await?
        .filter(predicate)?
        .select(vec![col("_rowid"), col("_rowaddr")])?;

    let mut out = Vec::new();
    for batch in df.collect().await? {
        let ids = batch.column(0).as_primitive::<UInt64Type>();
        let addr = batch.column(1).as_primitive::<UInt64Type>();
        for i in 0..batch.num_rows() {
            let a = addr.value(i);
            out.push(MatchLoc {
                fragment_id: a >> 32,
                offset: (a & 0xFFFF_FFFF) as u32,
                row_id: ids.value(i),
            });
        }
    }
    Ok(out)
}

/// The post-image rows for an UPDATE, together with their physical locations.
///
/// `rows` holds the updated column values for every row that matched the
/// UPDATE predicate, using the table's full data schema (same field order as
/// the table, post-`SET`).  `locs[k]` is the physical location of the row
/// stored at index `k` in `rows`.
#[derive(Debug)]
pub struct UpdateBatch {
    /// Full-schema RecordBatch containing the post-SET column values.
    pub rows: RecordBatch,
    /// Physical location of each row in `rows`, aligned by index.
    pub locs: Vec<MatchLoc>,
}

/// Compute the post-`SET` rows and their physical locations for an UPDATE.
///
/// For every data column in `table`, the projection substitutes the matching
/// `SET` expression (aliased back to the column's own name) or passes the
/// column through unchanged.  The filter runs **before** the projection, so
/// a self-referential `SET v = v + 1` reads the pre-image value of `v`.
///
/// # Arguments
/// * `ctx`       – DataFusion session with `table` registered.
/// * `table`     – Name of the registered table.
/// * `sets`      – `(column_name, new_value_expr)` pairs; columns not listed
///   are projected unchanged.
/// * `predicate` – Row filter (equivalent to `WHERE` clause).
///
/// # Returns
/// An [`UpdateBatch`] whose `rows` are the post-SET values and whose `locs`
/// identify where each row lives on disk, in aligned order.
/// Reduce an expression to a constant [`ScalarValue`] if it is a literal (or a
/// cast/alias of one). Returns `None` for anything that references a column —
/// i.e. a non-self-referential ("blind") SET value is exactly the `Some` case.
fn expr_as_literal(e: &Expr) -> Option<ScalarValue> {
    match e {
        Expr::Literal(s, _) => Some(s.clone()),
        Expr::Alias(a) => expr_as_literal(&a.expr),
        // Deliberately do NOT descend `Expr::Cast`: an explicit `CAST(x AS T)`
        // with T ≠ the column type could store a different value than the read
        // path (which evaluates the whole cast expression). Casted SETs fall back
        // to the proven read path; only bare literals elide (byte-equal by
        // construction — the caller re-casts to the column type, exactly as the
        // read path's projection coerces a literal to the column type).
        _ => None,
    }
}

/// Collect `column = <literal>` bindings from a predicate (descending through
/// `AND`). Only single-valued equalities pin a column; `IN`/ranges do not.
fn collect_eq_pins(predicate: &Expr, pins: &mut std::collections::HashMap<String, ScalarValue>) {
    if let Expr::BinaryExpr(b) = predicate {
        match b.op {
            Operator::And => {
                collect_eq_pins(&b.left, pins);
                collect_eq_pins(&b.right, pins);
            }
            Operator::Eq => {
                if let (Expr::Column(c), Some(s)) = (b.left.as_ref(), expr_as_literal(&b.right)) {
                    pins.insert(c.name().to_string(), s);
                } else if let (Some(s), Expr::Column(c)) =
                    (expr_as_literal(&b.left), b.right.as_ref())
                {
                    pins.insert(c.name().to_string(), s);
                }
            }
            _ => {}
        }
    }
}

/// Blind-write read elision: when every `SET` value is a literal
/// (non-self-referential) and every other data column is pinned to a single
/// literal by an equality predicate, the post-image is fully determined without
/// reading the pre-image. Build the patch (one identical row per located match)
/// directly. Returns `None` — caller falls back to the proven read path — if any
/// column is undetermined or a value cannot be coerced to the column type.
fn try_blind_determined_update(
    full_schema: &arrow::datatypes::SchemaRef,
    n_data: usize,
    set_map: &std::collections::HashMap<&str, &Expr>,
    predicate: &Expr,
    locs: &[MatchLoc],
) -> Option<UpdateBatch> {
    // Every SET value must be a blind literal; a single column ref aborts elision.
    let mut set_scalars: std::collections::HashMap<&str, ScalarValue> =
        std::collections::HashMap::with_capacity(set_map.len());
    for (col_name, expr) in set_map {
        set_scalars.insert(col_name, expr_as_literal(expr)?);
    }
    let mut pins = std::collections::HashMap::new();
    collect_eq_pins(predicate, &mut pins);

    let n = locs.len();
    let mut data_fields: Vec<arrow::datatypes::FieldRef> = Vec::with_capacity(n_data);
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(n_data);
    for i in 0..n_data {
        let field = full_schema.field(i);
        let scalar = match set_scalars.get(field.name().as_str()) {
            Some(s) => s.clone(),
            None => pins.get(field.name()).cloned()?, // undetermined → fall back
        };
        let scalar = scalar.cast_to(field.data_type()).ok()?;
        arrays.push(scalar.to_array_of_size(n).ok()?);
        data_fields.push(full_schema.field(i).clone().into());
    }
    let data_schema = Arc::new(arrow::datatypes::Schema::new(data_fields));
    let rows = RecordBatch::try_new(data_schema, arrays).ok()?;
    Some(UpdateBatch {
        rows,
        locs: locs.to_vec(),
    })
}

pub async fn plan_update(
    ctx: &SessionContext,
    table: &str,
    sets: &[(String, Expr)],
    predicate: Expr,
) -> Result<UpdateBatch> {
    // Resolve the registered provider to get the full schema (data cols +
    // _rowid/_rowaddr pseudo-columns appended last).
    let provider = ctx
        .table_provider(table)
        .await
        .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
    let full_schema = provider.schema();

    // The full schema has N_data data columns + 2 pseudo-columns (_rowid,
    // _rowaddr).  Determine where the pseudo-columns start.
    let n_fields = full_schema.fields().len();
    // Guard: at minimum the two pseudo-columns must be present.
    if n_fields < 2 {
        return Err(QueryError::Other(format!(
            "plan_update: schema for '{table}' has fewer than 2 fields (no pseudo-columns?)"
        )));
    }
    let n_data = n_fields - 2; // number of real data columns

    // Build a lookup map for the SET expressions.
    let set_map: std::collections::HashMap<&str, &Expr> = sets
        .iter()
        .map(|(name, expr)| (name.as_str(), expr))
        .collect();

    // Build the projection: data columns (SET or passthrough) then pseudo-cols.
    let mut projection: Vec<Expr> = Vec::with_capacity(n_data + 2);
    for i in 0..n_data {
        let field_name = full_schema.field(i).name();
        let expr = if let Some(set_expr) = set_map.get(field_name.as_str()) {
            // Apply SET expression, aliased back to the column's own name so the
            // output schema matches the table's data schema.
            (*set_expr).clone().alias(field_name)
        } else {
            col(field_name)
        };
        projection.push(expr);
    }
    projection.push(col("_rowid"));
    projection.push(col("_rowaddr"));

    // Fast path: when `predicate` is answerable from a secondary index (a point /
    // `IN` equality on an indexed column — the same locate DELETE uses), read only
    // the matched rows via the `_rowid IN (..)` pushdown instead of scanning the
    // whole stats-pruned fragment. A point UPDATE on a 1 M-row table otherwise
    // full-scans the surviving row group (~1.2s vs DELETE's ~260ms). Any miss or
    // error falls through to the proven scan path below — this is a pure
    // accelerator and must never change UPDATE results.
    let fast_locs: Option<Vec<MatchLoc>> = match ctx.table_provider(table).await {
        Ok(p) => {
            match (p.as_ref() as &dyn std::any::Any).downcast_ref::<IcefallDBTableProvider>() {
                Some(prov) => match prov.try_locate_by_index(&predicate).await {
                    Ok(Some(locs)) => Some(locs),
                    _ => None,
                },
                None => None,
            }
        }
        Err(_) => None,
    };

    // Blind-write read elision: when the index located the matches and the
    // post-image is fully determined by literal SETs + equality-pinned columns,
    // write the patch without reading the pre-image. Falls back to the read
    // path (below) for self-referential SETs or undetermined columns.
    if let Some(locs) = &fast_locs {
        if !locs.is_empty() {
            if let Some(ub) =
                try_blind_determined_update(&full_schema, n_data, &set_map, &predicate, locs)
            {
                return Ok(ub);
            }
        }
    }

    let fast_row_ids: Option<Vec<u64>> =
        fast_locs.map(|locs| locs.into_iter().map(|l| l.row_id).collect());

    let batches = match fast_row_ids {
        // Index proved no live row matches — nothing to read; the empty-batch
        // handler below produces an empty UpdateBatch.
        Some(ids) if ids.is_empty() => Vec::new(),
        Some(ids) => {
            // The `_rowid IN (..)` filter is unknown to filter-pushdown (pseudo-
            // columns are absent from the data schema), so it stays a FilterExec
            // over the scan and `RowIdSelectionPushdown` rewrites the scan to
            // decode only these rows. `SELECT *` already yields `_rowid`/`_rowaddr`
            // (full schema), and `select(projection)` applies SET over the
            // pre-image — identical batch shape to the fallback path.
            let id_list = ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT * FROM \"{}\" WHERE _rowid IN ({id_list})",
                table.replace('"', "\"\"")
            );
            ctx.sql(&sql).await?.select(projection)?.collect().await?
        }
        // Fallback: filter first (preserving pre-image values), then project.
        None => {
            ctx.table(table)
                .await?
                .filter(predicate)?
                .select(projection)?
                .collect()
                .await?
        }
    };

    if batches.is_empty() {
        // No matching rows — return an empty UpdateBatch with the data schema.
        let data_fields: arrow::datatypes::Fields =
            full_schema.fields().iter().take(n_data).cloned().collect();
        let data_schema = Arc::new(arrow::datatypes::Schema::new(data_fields));
        let empty_cols: Vec<Arc<dyn arrow::array::Array>> = data_schema
            .fields()
            .iter()
            .map(|f| arrow::array::new_empty_array(f.data_type()))
            .collect();
        let empty_batch = RecordBatch::try_new(Arc::clone(&data_schema), empty_cols)
            .map_err(QueryError::Arrow)?;
        return Ok(UpdateBatch {
            rows: empty_batch,
            locs: Vec::new(),
        });
    }

    // Each collected batch has columns: [data_0, ..., data_{n_data-1}, _rowid, _rowaddr].
    // Split into data and loc parts and accumulate across batches.
    let mut row_batches: Vec<RecordBatch> = Vec::with_capacity(batches.len());
    let mut locs: Vec<MatchLoc> = Vec::new();

    // The data schema is reconstructed from the first batch's schema (first n_data cols).
    let first_batch_schema = batches[0].schema();
    let data_fields: arrow::datatypes::Fields = first_batch_schema
        .fields()
        .iter()
        .take(n_data)
        .cloned()
        .collect();
    let data_schema = Arc::new(arrow::datatypes::Schema::new(data_fields));

    for batch in &batches {
        let n_rows = batch.num_rows();

        // ── Data columns ────────────────────────────────────────────────────
        let data_cols: Vec<Arc<dyn arrow::array::Array>> =
            (0..n_data).map(|i| Arc::clone(batch.column(i))).collect();
        let data_batch =
            RecordBatch::try_new(Arc::clone(&data_schema), data_cols).map_err(QueryError::Arrow)?;
        row_batches.push(data_batch);

        // ── Pseudo-columns → MatchLoc ────────────────────────────────────
        let ids = batch.column(n_data).as_primitive::<UInt64Type>();
        let addr = batch.column(n_data + 1).as_primitive::<UInt64Type>();
        for i in 0..n_rows {
            let a = addr.value(i);
            locs.push(MatchLoc {
                fragment_id: a >> 32,
                offset: (a & 0xFFFF_FFFF) as u32,
                row_id: ids.value(i),
            });
        }
    }

    // Concatenate all data batches into a single RecordBatch.
    let rows =
        arrow::compute::concat_batches(&data_schema, &row_batches).map_err(QueryError::Arrow)?;

    // Enforce non-nullable columns at runtime: a SET expression that is not a
    // bare NULL literal (e.g. `SET non_null = CASE ... END`, or a nullable
    // source column) can still evaluate to NULL on the matched rows. Reject it
    // before the writer sees the batch — `Writer::commit_update` only compares
    // names/types and the Parquet encoder accepts a validity buffer on a
    // non-nullable field, so without this check the schema invariant would be
    // silently corrupted. (See M02 follow-up.)
    let target_data_fields = full_schema
        .fields()
        .iter()
        .take(n_data)
        .cloned()
        .collect::<Vec<_>>();
    enforce_non_nullable_post_image(table, &target_data_fields, &rows)?;

    Ok(UpdateBatch { rows, locs })
}

// ── MERGE action types and execute_merge ───────────────────────────────

/// What to do when the MERGE key matches a live row in the target table.
#[derive(Debug, Clone)]
pub enum MatchedAction {
    /// Update the target row's columns from the source row.
    ///
    /// All source columns are written to the target row, preserving the source
    /// schema order.
    UpdateAll,
    /// Update only the named target columns.  The RHS expression (as a SQL
    /// string) may reference source columns via the source alias and target
    /// columns via the target table name; unassigned columns are preserved from
    /// the target pre-image.
    UpdateAssignments(Vec<(String, String)>),
}

/// What to do when the MERGE key has no live match in the target table (the
/// key is absent or the only existing entry is dead/tombstoned).
#[derive(Debug, Clone)]
pub enum NotMatchedAction {
    /// Insert the source row as a new row with a fresh row_id.
    Insert,
}

/// Statistics returned by [`execute_merge`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MergeStats {
    /// Number of target rows updated (MATCHED path).
    pub updated: u64,
    /// Number of source rows inserted as new rows (NOT MATCHED path).
    pub inserted: u64,
}

/// Core MERGE INTO implementation.
///
/// For each row in the deduped source:
/// - **MATCHED** (live row_id found in the unique index for the key value):
///   route to `commit_update` — overwrites the row in place (preserving its
///   stable `row_id`).
/// - **NOT MATCHED** (no live row_id; key is absent or the only entry is
///   tombstoned/deleted): route to `insert_batch` + `commit` — allocates a
///   fresh `row_id`.  The commit's `IndexMaintainer::maintain` call writes the
///   new `key → fresh_row_id` entry into the unique index, **replacing** any
///   stale dead entry (it rebuilds the full index from live rows, so dead
///   entries simply don't appear in the new base).
///
/// **Batched probe (single scan):** rather than probing the index once per key
/// (O(N) DataFusion plan+execute cycles), this function issues a single
/// `SELECT key, _rowid, _rowaddr FROM table WHERE key IN (k1, …, kN)` scan
/// covering all deduped source keys at once and builds a `HashMap<key →
/// MatchLoc>` from the result.  Each source row is then classified by an O(1)
/// map lookup.  For very large sources the IN-list is chunked into groups of
/// 1 000 keys to avoid planner limits.
///
/// Both sides are batched: matched rows are collected and written in one
/// `commit_update` call; unmatched rows are collected and written in one
/// `insert_batch` + `commit` call.  The registration is refreshed after so
/// subsequent queries see the new snapshot.
#[allow(clippy::too_many_arguments)]
pub async fn execute_merge(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    table: &str,
    key: &str,
    src: RecordBatch,
    matched: MatchedAction,
    not_matched: NotMatchedAction,
) -> crate::Result<MergeStats> {
    let merge_delta = execute_merge_to_delta(
        ctx,
        storage,
        table_root,
        table,
        key,
        src,
        "",
        matched,
        not_matched,
    )
    .await?;
    if let Some(delta) = &merge_delta.delta {
        apply_delta_to_provider(ctx, &merge_delta.table, delta).await?;
    }
    Ok(merge_delta.stats)
}

/// Core MERGE INTO implementation that commits the mutation but does **not**
/// refresh the registered provider. Returns the MERGE statistics and the delta
/// so callers can batch several mutations and refresh exactly once.
#[allow(clippy::too_many_arguments)]
async fn execute_merge_to_delta(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    table: &str,
    key: &str,
    src: RecordBatch,
    source_alias: &str,
    matched: MatchedAction,
    not_matched: NotMatchedAction,
) -> crate::Result<MergeDelta> {
    use icefalldb_core::catalog::Catalog;

    // ── Step 1: verify a unique index on `key` exists ───────────────────────
    let unique_index_name = require_unique_key_index(Arc::clone(&storage), table, key).await?;

    // ── Step 2: dedup source (Error on duplicates — SQL-standard) ───────────
    let deduped = dedup_source(&src, key, DupPolicy::Error)?;
    if deduped.is_empty() {
        return Ok(MergeDelta {
            stats: MergeStats::default(),
            delta: None,
            table: table.to_string(),
        });
    }

    // ── Step 3: load schema (needed to open a Writer for updates/inserts) ───
    let catalog = Catalog::load(storage.as_ref(), table_root)
        .await
        .map_err(QueryError::Core)?;
    let schema = catalog
        .latest_schema()
        .cloned()
        .ok_or_else(|| QueryError::Other(format!("no schema found for table '{table_root}'")))?;

    // ── Step 4: classify each deduped source row with ONE batched index probe ──
    //
    // Instead of calling locate_matches() once per key (N separate DataFusion
    // plan+execute cycles → O(N) scans), we build a single IN-list predicate
    // covering all N deduped keys and run exactly ONE scan.  The scan applies
    // deletion vectors automatically, so only LIVE rows are returned.
    //
    //   key PRESENT in scan result → MATCHED  (live row exists → update)
    //   key ABSENT  from result    → NOT MATCHED (absent or tombstoned → insert)
    //
    // For very large source batches we chunk the IN-list into groups of 1000
    // keys to avoid any practical DataFusion planner limit on list size.
    const IN_LIST_CHUNK: usize = 1000;

    // Build the key → MatchLoc map by issuing one batched probe per chunk.
    let mut live_map: HashMap<ScalarValue, MatchLoc> = HashMap::new();
    let keys: Vec<&ScalarValue> = deduped.by_key.iter().map(|(k, _)| k).collect();

    for chunk in keys.chunks(IN_LIST_CHUNK) {
        // Build the IN-list literals for this chunk.
        let list: Vec<Expr> = chunk
            .iter()
            .map(|sv| scalar_value_to_datafusion_lit(sv))
            .collect::<crate::Result<Vec<_>>>()?;

        // Build the predicate: `key_col IN (k1, k2, …, kN)`.
        let predicate = col(key).in_list(list, false);

        // Fast path: resolve this chunk's live rows from the unique key index
        // (require_unique_key_index guaranteed one exists), then read only those
        // rows by `_rowid IN (..)` instead of scanning the stats-pruned fragment
        // for `key_col IN (...)`. A point MERGE otherwise full-scans the surviving
        // row group, the same cost UPDATE paid. Any miss/error falls back to the
        // proven scan; classification (key→MatchLoc) is identical either way.
        let fast_row_ids: Option<Vec<u64>> = match ctx.table_provider(table).await {
            Ok(p) => {
                match (p.as_ref() as &dyn std::any::Any).downcast_ref::<IcefallDBTableProvider>() {
                    Some(prov) => match prov.try_locate_by_index(&predicate).await {
                        Ok(Some(locs)) => Some(locs.into_iter().map(|l| l.row_id).collect()),
                        _ => None,
                    },
                    None => None,
                }
            }
            Err(_) => None,
        };

        // Run ONE scan: SELECT key_col, _rowid, _rowaddr restricted to this
        // chunk's matched rows. Only live (non-tombstoned) rows are returned.
        let batches = match fast_row_ids {
            Some(ids) if ids.is_empty() => continue, // no live match in this chunk
            Some(ids) => {
                let id_list = ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "SELECT \"{}\", _rowid, _rowaddr FROM \"{}\" WHERE _rowid IN ({id_list})",
                    key.replace('"', "\"\""),
                    table.replace('"', "\"\"")
                );
                ctx.sql(&sql).await?.collect().await?
            }
            None => {
                ctx.table(table)
                    .await?
                    .filter(predicate)?
                    .select(vec![col(key), col("_rowid"), col("_rowaddr")])?
                    .collect()
                    .await?
            }
        };

        for batch in batches {
            let n = batch.num_rows();
            let ids = batch.column(1).as_primitive::<UInt64Type>();
            let addr = batch.column(2).as_primitive::<UInt64Type>();
            for i in 0..n {
                let key_sv =
                    normalize_merge_key(ScalarValue::try_from_array(batch.column(0), i).map_err(
                        |e| QueryError::Other(format!("MERGE probe: key extraction error: {e}")),
                    )?);
                let a = addr.value(i);
                let loc = MatchLoc {
                    fragment_id: a >> 32,
                    offset: (a & 0xFFFF_FFFF) as u32,
                    row_id: ids.value(i),
                };
                // In the unique-index contract at most one live row exists per key.
                // A second live entry means the contract is already broken; refuse the
                // MERGE rather than silently masking the corruption.
                if live_map.contains_key(&key_sv) {
                    return Err(QueryError::Core(IcefallDBError::UniqueKeyViolation {
                        table: table.to_string(),
                        index: unique_index_name.clone(),
                        key: key_sv.to_string(),
                    }));
                }
                live_map.insert(key_sv, loc);
            }
        }
    }

    // Route each deduped source row via O(1) map lookup.
    let mut matched_rows: Vec<RecordBatchRow> = Vec::new();
    let mut matched_locs: Vec<MatchLoc> = Vec::new();
    let mut unmatched_rows: Vec<RecordBatchRow> = Vec::new();

    for (key_sv, src_row) in &deduped.by_key {
        if let Some(loc) = live_map.remove(key_sv) {
            matched_rows.push(src_row.clone());
            matched_locs.push(loc);
        } else {
            unmatched_rows.push(src_row.clone());
        }
    }

    // ── Step 5: commit BOTH sides in ONE atomic manifest ─────────────────────
    //
    // Previously the matched updates and the unmatched inserts were committed as
    // two separate Writer commits, advancing the manifest sequence twice and
    // leaving a window where a crash could land a partial MERGE (matched updated
    // but inserts missing).  `commit_merge` writes a single new manifest covering
    // the patch fragment, the insert fragment, the matched relocation delta, the
    // tombstones, and a full index rebuild — one atomic pointer swap.
    if matched_rows.is_empty() && unmatched_rows.is_empty() {
        return Ok(MergeDelta {
            stats: MergeStats::default(),
            delta: None,
            table: table.to_string(),
        });
    }

    let NotMatchedAction::Insert = not_matched;

    // Post-image batch for matched rows (empty batch when none matched).
    let (matched_batch, set_columns) = match matched {
        MatchedAction::UpdateAll => {
            let batch = record_batch_from_rows(&matched_rows, src.schema())?;
            let set_cols: Vec<String> = src
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            (batch, set_cols)
        }
        MatchedAction::UpdateAssignments(ref assignments) => {
            if source_alias.is_empty() {
                return Err(QueryError::Other(
                    "MERGE UPDATE SET requires the USING source to have an alias".into(),
                ));
            }
            let batch = build_matched_batch_with_assignments(
                ctx,
                table,
                key,
                source_alias,
                &src.schema(),
                &matched_rows,
                &matched_locs,
                assignments,
            )
            .await?;
            let set_cols: Vec<String> = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            (batch, set_cols)
        }
    };
    // Insert batch for unmatched rows (empty batch when none unmatched).
    let insert_batch = record_batch_from_rows(&unmatched_rows, src.schema())?;

    let wal_mode = resolve_wal_mode(ctx, table).await;
    let writer = Writer::new(Arc::clone(&storage), table_root, schema)
        .await
        .map_err(QueryError::Core)?;
    let mut writer = writer.with_wal_mode(wal_mode);
    let delta = writer
        .commit_merge(matched_batch, matched_locs, &set_columns, insert_batch)
        .await
        .map_err(QueryError::Core)?;

    Ok(MergeDelta {
        stats: MergeStats {
            updated: matched_rows.len() as u64,
            inserted: unmatched_rows.len() as u64,
        },
        delta: Some(delta),
        table: table.to_string(),
    })
}

/// Normalise a merge-key [`ScalarValue`] to a canonical representation so that
/// values extracted from the source batch and from the target scan result compare
/// equal under [`PartialEq`]/[`Hash`] regardless of string-width variant.
///
/// Mappings applied (all others pass through unchanged):
/// - `LargeUtf8(x)`  → `Utf8(x)`
/// - `Utf8View(x)`   → `Utf8(x)`
///
/// This eliminates the silent-duplicate hazard where the source column is typed
/// as `Utf8` but the stored Parquet column is read back as `LargeUtf8` (or vice
/// versa): without normalisation the `HashMap<ScalarValue, MatchLoc>` lookup
/// misses and the live key is incorrectly treated as NOT MATCHED → duplicate row.
fn normalize_merge_key(sv: ScalarValue) -> ScalarValue {
    match sv {
        ScalarValue::LargeUtf8(x) => ScalarValue::Utf8(x),
        ScalarValue::Utf8View(x) => ScalarValue::Utf8(x),
        other => other,
    }
}

/// Convert a [`ScalarValue`] to a DataFusion [`Expr`] literal suitable for use
/// as the RHS of a filter predicate (`key_col = <lit>`).
fn scalar_value_to_datafusion_lit(sv: &datafusion::common::ScalarValue) -> crate::Result<Expr> {
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::lit;
    match sv {
        ScalarValue::Int64(Some(v)) => Ok(lit(*v)),
        ScalarValue::Int32(Some(v)) => Ok(lit(*v)),
        ScalarValue::Int16(Some(v)) => Ok(lit(*v)),
        ScalarValue::Int8(Some(v)) => Ok(lit(*v)),
        ScalarValue::UInt64(Some(v)) => Ok(lit(*v)),
        ScalarValue::UInt32(Some(v)) => Ok(lit(*v)),
        ScalarValue::UInt16(Some(v)) => Ok(lit(*v)),
        ScalarValue::UInt8(Some(v)) => Ok(lit(*v)),
        ScalarValue::Utf8(Some(s)) | ScalarValue::LargeUtf8(Some(s)) => Ok(lit(s.as_str())),
        other => Err(QueryError::Other(format!(
            "MERGE key type not supported for live-row probe: {other:?}"
        ))),
    }
}

/// Materialise a `Vec<RecordBatchRow>` back into a `RecordBatch` using
/// the provided `schema` as the column layout.
fn record_batch_from_rows(
    rows: &[RecordBatchRow],
    schema: arrow::datatypes::SchemaRef,
) -> crate::Result<RecordBatch> {
    use datafusion::common::ScalarValue;

    let n_cols = schema.fields().len();
    let n_rows = rows.len();

    // Empty input: `ScalarValue::iter_to_array` rejects an empty iterator (it
    // cannot infer the array type), so build a correctly-typed empty batch
    // directly from the schema's field data types.
    if n_rows == 0 {
        let empty_cols: Vec<arrow::array::ArrayRef> = schema
            .fields()
            .iter()
            .map(|f| arrow::array::new_empty_array(f.data_type()))
            .collect();
        return RecordBatch::try_new(schema, empty_cols).map_err(QueryError::Arrow);
    }

    // Build one `ScalarValue` builder per column by collecting all values then
    // using `ScalarValue::iter_to_array`.
    let mut col_values: Vec<Vec<ScalarValue>> = vec![Vec::with_capacity(n_rows); n_cols];

    for row in rows {
        // Map by column name since the row was built from a (potentially different-ordered) schema.
        for (col_idx, field) in schema.fields().iter().enumerate() {
            let pos = row.columns.iter().position(|c| c == field.name());
            let sv = match pos {
                Some(p) => row.values[p].clone(),
                None => ScalarValue::try_from(field.data_type()).map_err(|e| {
                    QueryError::Other(format!("cannot build null ScalarValue: {e}"))
                })?,
            };
            col_values[col_idx].push(sv);
        }
    }

    let arrays: Vec<arrow::array::ArrayRef> = col_values
        .into_iter()
        .map(|vals| {
            ScalarValue::iter_to_array(vals)
                .map_err(|e| QueryError::Other(format!("array build error: {e}")))
        })
        .collect::<crate::Result<Vec<_>>>()?;

    RecordBatch::try_new(schema, arrays).map_err(QueryError::Arrow)
}

/// Return the set of real (non-pseudo) column names for a provider schema.
///
/// IcefallDB providers append `_rowid` and `_rowaddr` pseudo-columns after the
/// data columns; this helper strips them.
fn data_column_names(schema: &arrow::datatypes::SchemaRef) -> HashSet<String> {
    let n_data = schema.fields().len().saturating_sub(2);
    schema
        .fields()
        .iter()
        .take(n_data)
        .map(|f| f.name().clone())
        .collect()
}

/// Normalize a SQL identifier to match DataFusion's resolution rules.
///
/// Unquoted identifiers are folded to lowercase; quoted identifiers are kept
/// verbatim so that uppercase or mixed-case names written with quotes still
/// resolve to the column that was declared with that exact casing.
fn normalize_sql_identifier(ident: &datafusion::sql::sqlparser::ast::Ident) -> String {
    if ident.quote_style.is_none() {
        ident.value.to_lowercase()
    } else {
        ident.value.clone()
    }
}

/// Validate that every UPDATE SET target exists in the table's data schema.
fn validate_update_targets(
    table_name: &str,
    schema: &arrow::datatypes::SchemaRef,
    sets: &[(String, Expr)],
) -> Result<()> {
    let data_cols = data_column_names(schema);
    for (name, _) in sets {
        if !data_cols.contains(name) {
            return Err(QueryError::Other(format!(
                "UPDATE SET target column '{name}' does not exist in table '{table_name}'"
            )));
        }
    }
    Ok(())
}

/// Replace untyped `NULL` literals in UPDATE SET assignments with typed nulls
/// of the target column's Arrow data type.
///
/// A bare SQL `NULL` parses into `ScalarValue::Null`, which DataFusion projects
/// as a `NullArray` (type `DataType::Null`). Building a patch fragment with that
/// array fails when the target column has a concrete type such as `Int64`. This
/// helper rewrites the literal to a typed null of the column's type so the
/// projection produces a matching array.
///
/// Non-nullable targets are rejected with a schema error before any rows are
/// read or written.
fn coerce_update_null_literals(
    table_name: &str,
    schema: &arrow::datatypes::SchemaRef,
    sets: &mut [(String, Expr)],
) -> Result<()> {
    use datafusion::common::ScalarValue;
    for (col_name, expr) in sets.iter_mut() {
        let is_untyped_null = matches!(expr, Expr::Literal(ScalarValue::Null, _));
        if !is_untyped_null {
            continue;
        }
        let field = schema.field_with_name(col_name).map_err(|_| {
            QueryError::Other(format!(
                "UPDATE SET target column '{col_name}' does not exist in table '{table_name}'"
            ))
        })?;
        if !field.is_nullable() {
            return Err(QueryError::Core(IcefallDBError::SchemaMismatch {
                column: col_name.clone(),
                expected: "non-nullable column cannot be assigned NULL".into(),
                path: table_name.to_string(),
            }));
        }
        let typed_null = ScalarValue::try_from(field.data_type()).map_err(|e| {
            QueryError::Other(format!(
                "UPDATE SET NULL cannot create typed null for column '{col_name}' \
                 in table '{table_name}': {e}"
            ))
        })?;
        *expr = Expr::Literal(typed_null, None);
    }
    Ok(())
}

/// Enforce that a post-image batch does not write NULLs into a non-nullable
/// target column.
///
/// `coerce_update_null_literals` and the MERGE `DataType::Null` cast only catch
/// a *bare* `NULL` literal. A non-literal expression that *evaluates* to null
/// at runtime — e.g. `SET non_null_col = CASE WHEN ... THEN NULL ELSE 0 END`
/// or `SET non_null_col = other_nullable_col` — is typed by DataFusion as the
/// concrete type with a validity bitmap and would otherwise slip through,
/// silently corrupting the schema invariant (Parquet encoding of a non-nullable
/// field accepts a validity buffer, and `Writer::commit_update` compares names
/// and types only). This check walks the non-nullable target data fields and
/// rejects any column whose post-image array has a non-zero null count.
fn enforce_non_nullable_post_image(
    table_name: &str,
    target_data_fields: &[arrow::datatypes::FieldRef],
    rows: &RecordBatch,
) -> Result<()> {
    for (i, field) in target_data_fields.iter().enumerate() {
        if !field.is_nullable() {
            let nulls = rows.column(i).null_count();
            if nulls > 0 {
                return Err(QueryError::Core(IcefallDBError::SchemaMismatch {
                    column: field.name().to_string(),
                    expected: format!(
                        "non-nullable column received {nulls} NULL value(s) from SET expression"
                    ),
                    path: table_name.to_string(),
                }));
            }
        }
    }
    Ok(())
}

/// clause.
///
/// Returns a vector of `(target_column, rhs_sql_string)` pairs.  The RHS string
/// is the original SQL expression and may reference source columns through the
/// source alias and target columns through the target table name.
fn parse_merge_update_assignments(
    table_name: &str,
    schema: &arrow::datatypes::SchemaRef,
    assignments: &[datafusion::sql::sqlparser::ast::Assignment],
) -> Result<Vec<(String, String)>> {
    let data_cols = data_column_names(schema);
    let mut out = Vec::with_capacity(assignments.len());
    let mut seen = HashSet::new();

    for assignment in assignments {
        let target = match &assignment.target {
            AssignmentTarget::ColumnName(obj_name) => obj_name
                .0
                .last()
                .and_then(|p| p.as_ident())
                .map(normalize_sql_identifier)
                .ok_or_else(|| {
                    QueryError::Other(
                        "MERGE UPDATE SET target column name is not a plain identifier".into(),
                    )
                })?,
            AssignmentTarget::Tuple(_) => {
                return Err(QueryError::Other(
                    "MERGE UPDATE SET tuple assignment targets are not supported".into(),
                ));
            }
        };

        if !data_cols.contains(&target) {
            return Err(QueryError::Other(format!(
                "MERGE UPDATE SET target column '{target}' does not exist in table '{table_name}'"
            )));
        }

        if !seen.insert(target.clone()) {
            return Err(QueryError::Other(format!(
                "MERGE UPDATE SET target column '{target}' specified more than once"
            )));
        }

        out.push((target, assignment.value.to_string()));
    }

    Ok(out)
}

/// Extract the alias (or name) of a MERGE USING source.
fn extract_source_alias(source: &TableFactor) -> Option<String> {
    match source {
        TableFactor::Derived { alias, .. } => alias.as_ref().map(|a| a.name.value.clone()),
        TableFactor::Table { name, alias, .. } => {
            alias.as_ref().map(|a| a.name.value.clone()).or_else(|| {
                name.0
                    .last()
                    .and_then(|p| p.as_ident())
                    .map(|i| i.value.clone())
            })
        }
        _ => None,
    }
}

/// Quote an identifier for use in a SQL string.
fn quote_sql_identifier(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

/// RAII guard that deregisters a temporary DataFusion table on drop.
///
/// Used for MERGE's transient `__icefall_merge_src` registration so the table
/// is cleaned up even if the join/evaluate query fails.
struct DeregisterTableOnDrop<'a> {
    ctx: &'a SessionContext,
    name: &'a str,
}

impl<'a> DeregisterTableOnDrop<'a> {
    fn new(ctx: &'a SessionContext, name: &'a str) -> Self {
        Self { ctx, name }
    }
}

impl Drop for DeregisterTableOnDrop<'_> {
    fn drop(&mut self) {
        let _ = self.ctx.deregister_table(self.name);
    }
}

/// Build the post-image RecordBatch for a MERGE ... UPDATE SET with explicit
/// assignments.
///
/// A temporary table containing the matched source rows (plus an ordering
/// column) is registered and joined to the target table on the merge key.  The
/// SELECT list evaluates each assignment expression over the joined row and
/// preserves unassigned target columns from the target pre-image.  The result
/// is ordered by the synthetic ordering column so it remains aligned with
/// `matched_locs` before `commit_merge` sorts both by row_id.
#[allow(clippy::too_many_arguments)]
async fn build_matched_batch_with_assignments(
    ctx: &SessionContext,
    table: &str,
    key_col: &str,
    source_alias: &str,
    src_schema: &arrow::datatypes::SchemaRef,
    matched_rows: &[RecordBatchRow],
    _matched_locs: &[MatchLoc],
    assignments: &[(String, String)],
) -> Result<RecordBatch> {
    let n_rows = matched_rows.len();

    // Target data schema (pseudo-columns excluded).
    let provider = ctx
        .table_provider(table)
        .await
        .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
    let full_schema = provider.schema();
    let n_data = full_schema.fields().len().saturating_sub(2);
    let target_data_fields: Vec<Arc<arrow::datatypes::Field>> =
        full_schema.fields().iter().take(n_data).cloned().collect();
    let target_data_schema = Arc::new(arrow::datatypes::Schema::new(target_data_fields.clone()));

    if n_rows == 0 {
        let empty_cols: Vec<ArrayRef> = target_data_schema
            .fields()
            .iter()
            .map(|f| arrow::array::new_empty_array(f.data_type()))
            .collect();
        return RecordBatch::try_new(target_data_schema, empty_cols).map_err(QueryError::Arrow);
    }

    // Build a source batch with a synthetic ordering/index column.
    let mut src_fields: Vec<Arc<arrow::datatypes::Field>> =
        src_schema.fields().iter().cloned().collect();
    src_fields.push(Arc::new(arrow::datatypes::Field::new(
        "__merge_idx__",
        DataType::Int64,
        false,
    )));
    let mut src_arrays: Vec<ArrayRef> = Vec::with_capacity(src_fields.len());

    for field in src_schema.fields().iter() {
        let mut values = Vec::with_capacity(n_rows);
        for row in matched_rows {
            let pos = row
                .columns
                .iter()
                .position(|c| c == field.name())
                .ok_or_else(|| {
                    QueryError::Other(format!(
                        "MERGE: matched source row missing column '{}'",
                        field.name()
                    ))
                })?;
            values.push(row.values[pos].clone());
        }
        src_arrays.push(
            ScalarValue::iter_to_array(values)
                .map_err(|e| QueryError::Other(format!("array build error: {e}")))?,
        );
    }
    src_arrays.push(Arc::new(Int64Array::from(
        (0..n_rows as i64).collect::<Vec<_>>(),
    )));

    let src_batch = RecordBatch::try_new(
        Arc::new(arrow::datatypes::Schema::new(src_fields)),
        src_arrays,
    )
    .map_err(QueryError::Arrow)?;

    // Register the matched source rows under an internal name; the SQL alias
    // used in the query is the original source alias so assignment RHS refs
    // resolve correctly.  The guard ensures the temp table is deregistered on
    // every exit path, including errors from the join/evaluate query.
    let internal_src_name = "__icefall_merge_src";
    let _ = ctx.deregister_table(internal_src_name);
    ctx.register_batch(internal_src_name, src_batch)
        .map_err(|e| QueryError::Other(format!("MERGE: register source batch: {e}")))?;
    let _src_guard = DeregisterTableOnDrop::new(ctx, internal_src_name);

    let table_q = quote_sql_identifier(table);
    let src_q = quote_sql_identifier(source_alias);
    let internal_q = quote_sql_identifier(internal_src_name);
    let key_q = quote_sql_identifier(key_col);
    let idx_q = quote_sql_identifier("__merge_idx__");

    let assignment_map: HashMap<&str, &str> = assignments
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();

    let mut select_items: Vec<String> = Vec::with_capacity(n_data + 1);
    for field in &target_data_fields {
        let col_name = field.name();
        let col_q = quote_sql_identifier(col_name);
        let item = if let Some(rhs) = assignment_map.get(col_name.as_str()) {
            format!("({rhs}) AS {col_q}")
        } else {
            format!("{table_q}.{col_q} AS {col_q}")
        };
        select_items.push(item);
    }
    select_items.push(format!("{src_q}.{idx_q}"));

    let sql = format!(
        "SELECT {} FROM {} JOIN {} AS {} ON {}.{} = {}.{} ORDER BY {}.{}",
        select_items.join(", "),
        table_q,
        internal_q,
        src_q,
        table_q,
        key_q,
        src_q,
        key_q,
        src_q,
        idx_q,
    );

    let batches = ctx
        .sql(&sql)
        .await
        .map_err(|e| QueryError::Other(format!("MERGE assignment evaluation error: {e}")))?
        .collect()
        .await
        .map_err(|e| QueryError::Other(format!("MERGE assignment evaluation error: {e}")))?;

    if batches.is_empty() {
        return Err(QueryError::Other(
            "MERGE assignment evaluation produced no rows".into(),
        ));
    }

    let result = arrow::compute::concat_batches(&batches[0].schema(), &batches)
        .map_err(QueryError::Arrow)?;
    if result.num_rows() != n_rows {
        return Err(QueryError::Other(format!(
            "MERGE assignment evaluation produced {} rows, expected {}",
            result.num_rows(),
            n_rows
        )));
    }

    // A bare `NULL` (or any all-NULL expression) is projected by DataFusion as a
    // `DataType::Null` column, which would not match the target field's concrete
    // type in `try_new`. Cast such columns to the target type so `SET col = NULL`
    // works in MERGE exactly as it does in UPDATE (M02). A NULL assigned to a
    // non-nullable column is rejected here, before any write — `commit_merge`
    // does not re-check nullability.
    let data_arrays: Vec<ArrayRef> = (0..n_data)
        .map(|i| {
            let col = result.column(i);
            let target_field = &target_data_fields[i];
            if col.data_type() == &DataType::Null && target_field.data_type() != &DataType::Null {
                if !target_field.is_nullable() {
                    return Err(QueryError::Other(format!(
                        "MERGE UPDATE SET assigns NULL to non-nullable column '{}'",
                        target_field.name()
                    )));
                }
                arrow::compute::cast(col, target_field.data_type()).map_err(QueryError::Arrow)
            } else {
                Ok(Arc::clone(col))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let out = RecordBatch::try_new(target_data_schema, data_arrays).map_err(QueryError::Arrow)?;
    // Enforce non-nullable columns at runtime for the same reason `plan_update`
    // does: a non-bare-NULL expression (e.g. `SET non_null = CASE ... END` or a
    // nullable source column) evaluates to a typed-but-nullable array that the
    // cast branch above does not inspect. (M02 follow-up.)
    enforce_non_nullable_post_image(table, &target_data_fields, &out)?;
    Ok(out)
}

/// Parse and execute a SQL statement against a IcefallDB table.
///
/// For `DELETE FROM t WHERE p` statements this function:
/// 1. Parses the SQL and extracts the target table name and predicate.
/// 2. Converts the predicate to a DataFusion [`Expr`] using the registered
///    table's schema.
/// 3. Calls [`locate_matches`] to find all live matching rows.
/// 4. Groups the matches by `fragment_id` and calls
///    `Writer::commit_deletes` to write deletion vectors and advance the
///    manifest.
/// 5. Deregisters and re-registers the table in `ctx` so that subsequent
///    queries see the new snapshot.
/// 6. Returns the number of rows deleted.
///
/// For `UPDATE t SET col = expr [WHERE p]` statements this function:
/// 1. Parses the SQL and extracts the target table name, SET assignments,
///    and optional WHERE predicate.
/// 2. Converts each assignment value AST and the selection to DataFusion
///    [`Expr`]s using the registered table's schema.
/// 3. Calls [`plan_update`] to compute the post-SET rows and their physical
///    locations.
/// 4. Opens a [`Writer`] and calls `Writer::commit_update` to write the
///    patch fragment, tombstone the original rows, update the row-index
///    delta, and maintain secondary indexes incrementally.
/// 5. Deregisters and re-registers the table in `ctx` so that subsequent
///    queries see the new snapshot.
/// 6. Returns the number of rows updated.
///
/// Any other statement is delegated to `ctx.sql(sql)`, executed, and
/// `0` is returned.  This preserves full `SELECT`/DDL support through the
/// same entry point.
///
/// # Arguments
/// * `ctx`         – The DataFusion [`SessionContext`] with the table registered.
/// * `storage`     – The [`Storage`] backend backing the table.
/// * `table_root`  – The root path of the table (used to open the `Writer`).
/// * `sql`         – The SQL statement to execute.
///
/// # Errors
/// Returns an error if parsing fails, the table is not registered, the
/// predicate cannot be resolved, or the deletion commit fails.
pub async fn execute_sql(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    sql: &str,
) -> Result<u64> {
    // Parse the SQL to detect whether it is a DELETE statement.
    let mut statements =
        DFParser::parse_sql(sql).map_err(|e| QueryError::Other(format!("SQL parse error: {e}")))?;

    if statements.len() != 1 {
        return Err(QueryError::Other(format!(
            "execute_sql: expected exactly one statement, got {}",
            statements.len()
        )));
    }

    let stmt = statements.pop_front().unwrap();

    // Unwrap the DataFusion Statement wrapper to get the inner sqlparser
    // Statement.
    let sql_stmt = match stmt {
        datafusion::sql::parser::Statement::Statement(inner) => *inner,
        other => {
            // DataFusion-specific statement (COPY, CREATE EXTERNAL TABLE, …):
            // delegate to ctx.sql to execute and return 0.
            drop(other);
            ctx.sql(sql)
                .await
                .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?
                .collect()
                .await
                .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?;
            return Ok(0);
        }
    };

    match sql_stmt {
        SQLStatement::Delete(delete) => execute_delete(ctx, storage, table_root, delete).await,
        SQLStatement::Update(update) => execute_update(ctx, storage, table_root, update).await,
        SQLStatement::Merge(merge) => {
            let stats = execute_merge_sql(ctx, storage, table_root, merge).await?;
            Ok(stats.updated + stats.inserted)
        }
        _ => {
            // Non-mutation statement: delegate to ctx.sql.
            ctx.sql(sql)
                .await
                .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?
                .collect()
                .await
                .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?;
            Ok(0)
        }
    }
}

/// Parse and execute a batch of SQL mutation statements, refreshing the
/// registered provider exactly once at the end.
///
/// Each statement is parsed individually. `DELETE`, `UPDATE`, and `MERGE`
/// statements are committed to storage using the same paths as [`execute_sql`].
/// A write-side context maintains providers that are refreshed incrementally
/// after each statement so later statements see rows touched by earlier ones.
/// After the loop, the final snapshot of each mutated table's write-side
/// provider is copied to the corresponding public provider in `ctx`, producing
/// exactly one public refresh for the whole batch.
///
/// # Arguments
/// * `ctx`         – The DataFusion [`SessionContext`] with the table registered.
/// * `storage`     – The [`Storage`] backend backing the table.
/// * `table_root`  – The root path of the table (used to open the `Writer`).
/// * `sqls`        – The SQL statements to execute.
///
/// # Errors
/// Returns an error if any statement fails to parse or commit. Note that
/// earlier statements in the batch may already be persisted if a later one
/// fails; callers who need cross-statement atomicity should use a single
/// `MERGE` or wrap the batch in their own transaction mechanism.
/// Extract the target table name from a DML statement.
///
/// Returns `None` for statements that do not have a single target table
/// (e.g. `COPY`).
fn statement_target_table(sql_stmt: &SQLStatement) -> Option<String> {
    match sql_stmt {
        SQLStatement::Delete(delete) => {
            let tables = match &delete.from {
                FromTable::WithFromKeyword(ts) | FromTable::WithoutKeyword(ts) => ts,
            };
            if tables.len() != 1 {
                return None;
            }
            match &tables[0].relation {
                TableFactor::Table { name, .. } => Some(name.to_string()),
                _ => None,
            }
        }
        SQLStatement::Update(update) => match &update.table.relation {
            TableFactor::Table { name, .. } => Some(name.to_string()),
            _ => None,
        },
        SQLStatement::Merge(merge) => match &merge.table {
            TableFactor::Table { name, .. } => Some(name.to_string()),
            _ => None,
        },
        _ => None,
    }
}

/// The **bare** target table name of a single-statement DML mutation
/// (`DELETE`/`UPDATE`/`MERGE`), or `None`. Quotes are stripped so the result
/// matches the registered table name / on-disk table directory — used by the
/// server's `/mutate` endpoint to resolve the table root from the SQL.
pub fn mutation_target_table(sql: &str) -> Option<String> {
    let mut statements = DFParser::parse_sql(sql).ok()?;
    if statements.len() != 1 {
        return None;
    }
    let datafusion::sql::parser::Statement::Statement(stmt) = statements.pop_front()? else {
        return None;
    };
    let relation = match &*stmt {
        SQLStatement::Delete(d) => {
            let tables = match &d.from {
                FromTable::WithFromKeyword(ts) | FromTable::WithoutKeyword(ts) => ts,
            };
            if tables.len() != 1 {
                return None;
            }
            &tables[0].relation
        }
        SQLStatement::Update(u) => &u.table.relation,
        SQLStatement::Merge(m) => &m.table,
        _ => return None,
    };
    match relation {
        TableFactor::Table { name, .. } => name
            .0
            .last()
            .and_then(|p| p.as_ident())
            .map(|i| i.value.clone()),
        _ => None,
    }
}

/// Downcast a DataFusion `TableProvider` reference to [`IcefallDBTableProvider`].
fn downcast_icefalldb_provider(
    provider: &Arc<dyn datafusion::catalog::TableProvider>,
) -> Result<&IcefallDBTableProvider> {
    (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<IcefallDBTableProvider>()
        .ok_or_else(|| {
            QueryError::Other("registered provider is not a IcefallDBTableProvider".into())
        })
}

pub async fn execute_sql_batch(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    sqls: &[String],
) -> Result<Vec<u64>> {
    if sqls.is_empty() {
        return Ok(Vec::new());
    }

    let mut counts = Vec::with_capacity(sqls.len());

    // Maintain a write-side context whose registered providers are refreshed
    // incrementally after each statement. This keeps later statements in the
    // batch from reading a stale snapshot when they target rows touched by an
    // earlier statement, while the public `ctx` provider is refreshed exactly
    // once at the end of the batch.
    let write_ctx = SessionContext::new();
    let mut write_providers: HashMap<String, Arc<IcefallDBTableProvider>> = HashMap::new();
    let mut mutated_tables: HashSet<String> = HashSet::new();

    for sql in sqls {
        let mut statements = DFParser::parse_sql(sql)
            .map_err(|e| QueryError::Other(format!("SQL parse error: {e}")))?;

        if statements.len() != 1 {
            return Err(QueryError::Other(format!(
                "execute_sql_batch: expected exactly one statement, got {}",
                statements.len()
            )));
        }

        let stmt = statements.pop_front().unwrap();
        let sql_stmt = match stmt {
            datafusion::sql::parser::Statement::Statement(inner) => *inner,
            other => {
                drop(other);
                ctx.sql(sql)
                    .await
                    .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?
                    .collect()
                    .await
                    .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?;
                counts.push(0);
                continue;
            }
        };

        // Ensure the write-side context has an up-to-date provider for the
        // statement's target table before we plan the mutation.
        let table_name = statement_target_table(&sql_stmt);
        if let Some(table) = table_name.as_ref() {
            if !write_providers.contains_key(table) {
                let original = ctx
                    .table_provider(table)
                    .await
                    .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
                let original = downcast_icefalldb_provider(&original)?;
                let snapshot = original.snapshot();
                let pinned = original.pinned_sequence();
                // Materialize the public provider's pinned plan (open is lazy, so
                // its in-memory snapshot may still be the empty placeholder) and
                // seed the write-side cache with it so the first mutation read and
                // `apply_committed_delta` see the correct base fragments.
                let base_plan = original.scan_plan().await?;
                // Use the provider's canonical table name rather than the raw SQL
                // identifier: a quoted identifier like `"trips"` stringifies with
                // its quote characters, and `apply_committed_delta` interpolates
                // `self.table` into the deletion-vector / patch-fragment storage
                // paths (`{table}/_deletions/...`). The quoted form yields a path
                // that no writer ever creates, so a later same-fragment statement
                // fails to read it. The public provider (single-statement path) is
                // immune because it is always registered under the canonical name.
                let canonical = original.table().to_string();
                let mut write_provider = IcefallDBTableProvider::from_snapshot(
                    Arc::clone(&storage),
                    canonical,
                    *original.config(),
                    snapshot,
                    Some((pinned, base_plan)),
                );
                #[cfg(feature = "encryption")]
                if let Some(resolver) = original.encryption_resolver() {
                    write_provider = write_provider.with_encryption_resolver(resolver);
                }
                let provider = Arc::new(write_provider);
                write_ctx
                    .register_table(table.as_str(), provider.clone())
                    .map_err(|e| QueryError::Other(format!("register write provider: {e}")))?;
                write_providers.insert(table.clone(), provider);
            }
        }

        let md = match sql_stmt {
            SQLStatement::Delete(delete) => {
                execute_delete_to_delta(&write_ctx, Arc::clone(&storage), table_root, delete)
                    .await?
            }
            SQLStatement::Update(update) => {
                execute_update_to_delta(&write_ctx, Arc::clone(&storage), table_root, update)
                    .await?
            }
            SQLStatement::Merge(merge) => {
                let merge_delta =
                    execute_merge_sql_to_delta(&write_ctx, Arc::clone(&storage), table_root, merge)
                        .await?;
                MutationDelta {
                    count: merge_delta.stats.updated + merge_delta.stats.inserted,
                    delta: merge_delta.delta,
                    table: merge_delta.table,
                }
            }
            _ => {
                ctx.sql(sql)
                    .await
                    .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?
                    .collect()
                    .await
                    .map_err(|e| QueryError::Other(format!("SQL execute error: {e}")))?;
                counts.push(0);
                continue;
            }
        };

        counts.push(md.count);

        if let Some(delta) = &md.delta {
            // Keep the write-side provider pinned to the latest committed
            // snapshot so subsequent statements see their predecessors' changes.
            if let Some(write_provider) = write_providers.get(md.table.as_str()) {
                write_provider.apply_committed_delta(delta).await?;
            }
            mutated_tables.insert(md.table.clone());
        }
    }

    // Copy the final write-side snapshot to each public provider once.
    for table_name in &mutated_tables {
        let write_provider = write_providers.get(table_name).ok_or_else(|| {
            QueryError::Other(format!("write provider missing for table '{table_name}'"))
        })?;
        let public_provider = ctx
            .table_provider(table_name)
            .await
            .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
        let public_provider = downcast_icefalldb_provider(&public_provider)?;
        public_provider
            .replace_snapshot_from(write_provider)
            .await?;
    }

    Ok(counts)
}

/// Build the DataFusion [`Expr`] for a SQL WHERE clause given a table name
/// and the raw sqlparser AST expression.  Returns `lit(true)` when
/// `selection` is `None` (no WHERE clause → every row matches).
async fn build_predicate(
    ctx: &SessionContext,
    table_name: &str,
    selection: &Option<datafusion::sql::sqlparser::ast::Expr>,
) -> Result<Expr> {
    match selection {
        None => Ok(datafusion::logical_expr::lit(true)),
        Some(sel) => {
            let provider = ctx
                .table_provider(table_name)
                .await
                .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
            let arrow_schema = provider.schema();
            let df_schema = DFSchema::try_from(arrow_schema.as_ref().clone())
                .map_err(|e| QueryError::Other(format!("DFSchema error: {e}")))?;
            ctx.parse_sql_expr(&sel.to_string(), &df_schema)
                .map_err(|e| QueryError::Other(format!("predicate parse error: {e}")))
        }
    }
}

/// Inner implementation for `DELETE FROM t WHERE p`.
async fn execute_delete(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    delete: datafusion::sql::sqlparser::ast::Delete,
) -> Result<u64> {
    let md = execute_delete_to_delta(ctx, storage, table_root, delete).await?;
    if let Some(delta) = &md.delta {
        apply_delta_to_provider(ctx, &md.table, delta).await?;
    }
    Ok(md.count)
}

/// Commit a `DELETE` and return the affected-row count and delta without
/// refreshing the registered provider.
/// Resolve deferred-commit (mutation WAL) mode for a mutation on `table`.
///
/// Source of truth is the registered provider's [`ProviderConfig::wal_mode`]
/// (default `true`). The `ICEFALLDB_WAL` env var, when set, overrides it
/// (`1`/`0`) — intended for A/B benchmarking only.
async fn resolve_wal_mode(ctx: &SessionContext, table: &str) -> bool {
    if let Ok(v) = std::env::var("ICEFALLDB_WAL") {
        return v == "1";
    }
    if let Ok(provider) = ctx.table_provider(table).await {
        if let Some(mp) =
            (provider.as_ref() as &dyn std::any::Any).downcast_ref::<IcefallDBTableProvider>()
        {
            return mp.config().wal_mode;
        }
    }
    true
}

async fn execute_delete_to_delta(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    delete: datafusion::sql::sqlparser::ast::Delete,
) -> Result<MutationDelta> {
    // Extract the table name from the FROM clause.
    let tables = match &delete.from {
        FromTable::WithFromKeyword(ts) | FromTable::WithoutKeyword(ts) => ts,
    };
    if tables.len() != 1 {
        return Err(QueryError::Other(
            "execute_sql: DELETE with multiple or zero FROM tables is not supported".into(),
        ));
    }
    let table_name = match &tables[0].relation {
        TableFactor::Table { name, .. } => name.to_string(),
        _ => {
            return Err(QueryError::Other(
                "execute_sql: DELETE FROM target must be a plain table name".into(),
            ));
        }
    };

    let predicate = build_predicate(ctx, &table_name, &delete.selection).await?;

    // Locate all live rows matching the predicate.
    let matches = locate_matches(ctx, &table_name, predicate).await?;
    let match_count = matches.len() as u64;

    if match_count == 0 {
        return Ok(MutationDelta {
            count: 0,
            delta: None,
            table: table_name,
        });
    }

    // Group offsets by fragment_id.
    let mut by_fragment: HashMap<u64, Vec<u32>> = HashMap::new();
    for m in &matches {
        by_fragment.entry(m.fragment_id).or_default().push(m.offset);
    }

    // Load the table schema from storage so we can open a Writer.
    let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), table_root)
        .await
        .map_err(QueryError::Core)?;
    let schema = catalog
        .latest_schema()
        .cloned()
        .ok_or_else(|| QueryError::Other(format!("no schema found for table '{table_root}'")))?;

    // Commit the deletion vectors and advance the manifest.
    let wal_mode = resolve_wal_mode(ctx, &table_name).await;
    let writer = Writer::new(Arc::clone(&storage), table_root, schema)
        .await
        .map_err(QueryError::Core)?;
    let mut writer = writer.with_wal_mode(wal_mode);
    let delta = writer
        .commit_deletes(by_fragment)
        .await
        .map_err(QueryError::Core)?;

    Ok(MutationDelta {
        count: match_count,
        delta: Some(delta),
        table: table_name,
    })
}

/// Inner implementation for `UPDATE t SET col = expr [WHERE p]`.
async fn execute_update(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    update: datafusion::sql::sqlparser::ast::Update,
) -> Result<u64> {
    let md = execute_update_to_delta(ctx, storage, table_root, update).await?;
    if let Some(delta) = &md.delta {
        apply_delta_to_provider(ctx, &md.table, delta).await?;
    }
    Ok(md.count)
}

/// Commit an `UPDATE` and return the affected-row count and delta without
/// refreshing the registered provider.
async fn execute_update_to_delta(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    update: datafusion::sql::sqlparser::ast::Update,
) -> Result<MutationDelta> {
    // Extract the target table name from the TableWithJoins. Take the bare
    // identifier value (last, non-schema part) rather than `name.to_string()`:
    // the latter re-emits a quoted identifier verbatim (`"t"` keeps its quotes),
    // which the index fast path would then double-escape into an unresolvable
    // `"""t"""`. This mirrors the MERGE target-name extraction.
    let table_name = match &update.table.relation {
        TableFactor::Table { name, .. } => name
            .0
            .last()
            .and_then(|part| part.as_ident())
            .map(|ident| ident.value.clone())
            .ok_or_else(|| {
                QueryError::Other("execute_sql: UPDATE target must be a plain table name".into())
            })?,
        _ => {
            return Err(QueryError::Other(
                "execute_sql: UPDATE target must be a plain table name".into(),
            ));
        }
    };

    // Retrieve the table schema so we can resolve column references inside
    // SET expressions.
    let provider = ctx
        .table_provider(table_name.as_str())
        .await
        .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
    let arrow_schema = provider.schema();
    let df_schema = DFSchema::try_from(arrow_schema.as_ref().clone())
        .map_err(|e| QueryError::Other(format!("DFSchema error: {e}")))?;

    // Convert each assignment `col = ast_expr` into `(column_name, DataFusion Expr)`.
    let mut sets: Vec<(String, Expr)> = Vec::with_capacity(update.assignments.len());
    let mut set_columns: Vec<String> = Vec::with_capacity(update.assignments.len());
    for assignment in &update.assignments {
        // Only support simple column-name targets (not tuple assignments).
        let col_name = match &assignment.target {
            AssignmentTarget::ColumnName(obj_name) => {
                // An ObjectName is a sequence of ObjectNamePart; take the last
                // part (the column name itself, ignoring any schema/table prefix)
                // and extract its Ident value, normalizing unquoted identifiers
                // to lowercase to match DataFusion's identifier resolution.
                normalize_sql_identifier(
                    obj_name
                        .0
                        .last()
                        .ok_or_else(|| {
                            QueryError::Other("UPDATE SET target column name is empty".into())
                        })?
                        .as_ident()
                        .ok_or_else(|| {
                            QueryError::Other(
                                "UPDATE SET target column name is not a plain identifier".into(),
                            )
                        })?,
                )
            }
            AssignmentTarget::Tuple(_) => {
                return Err(QueryError::Other(
                    "execute_sql: tuple assignment targets in UPDATE are not supported".into(),
                ));
            }
        };
        // Convert the RHS expression AST to a DataFusion Expr using the table
        // schema so that column references inside the expression are resolved.
        let value_expr = ctx
            .parse_sql_expr(&assignment.value.to_string(), &df_schema)
            .map_err(|e| {
                QueryError::Other(format!(
                    "UPDATE SET expression parse error for column '{col_name}': {e}"
                ))
            })?;
        set_columns.push(col_name.clone());
        sets.push((col_name, value_expr));
    }

    if sets.is_empty() {
        return Err(QueryError::Other(
            "execute_sql: UPDATE with no SET assignments".into(),
        ));
    }

    // Validate all SET targets against the target schema before planning the
    // predicate or touching any rows.
    validate_update_targets(&table_name, &arrow_schema, &sets)?;
    // Rewrite bare `NULL` literals to typed nulls of the target column's type;
    // reject NULL assignments to non-nullable columns before reading/writing.
    coerce_update_null_literals(&table_name, &arrow_schema, &mut sets)?;

    // Build the WHERE predicate.
    let predicate = build_predicate(ctx, &table_name, &update.selection).await?;

    // Compute the post-SET rows and their physical locations.
    let ub = plan_update(ctx, &table_name, &sets, predicate).await?;

    if ub.locs.is_empty() {
        // No rows matched the predicate — nothing to write.
        return Ok(MutationDelta {
            count: 0,
            delta: None,
            table: table_name,
        });
    }

    let row_count = ub.locs.len() as u64;

    // Load the table schema from storage so we can open a Writer.
    let catalog = icefalldb_core::catalog::Catalog::load(storage.as_ref(), table_root)
        .await
        .map_err(QueryError::Core)?;
    let schema = catalog
        .latest_schema()
        .cloned()
        .ok_or_else(|| QueryError::Other(format!("no schema found for table '{table_root}'")))?;

    // Commit the update: write the patch fragment, tombstone the originals,
    // advance the row-index delta, and incrementally maintain indexes for the
    // SET columns.
    let wal_mode = resolve_wal_mode(ctx, &table_name).await;
    let writer = Writer::new(Arc::clone(&storage), table_root, schema)
        .await
        .map_err(QueryError::Core)?;
    let mut writer = writer.with_wal_mode(wal_mode);
    let delta = writer
        .commit_update(ub.rows, ub.locs, &set_columns)
        .await
        .map_err(QueryError::Core)?;

    Ok(MutationDelta {
        count: row_count,
        delta: Some(delta),
        table: table_name,
    })
}

// ── MERGE INTO … SQL surface ──────────────────────────────────────────

/// Parse and execute a `MERGE INTO t USING … ON … WHEN … THEN …` statement.
///
/// This is the SQL-level entry point for CDC/upsert workloads.  It:
///
/// 1. Extracts the **target table name** from `merge.table`.
/// 2. **Materializes the source** into an Arrow `RecordBatch` by running a
///    `SELECT * FROM (<source>)` query through DataFusion.  The source alias
///    (the name after `AS` on the USING clause) determines the registered name
///    used for that temporary table.
/// 3. **Extracts the merge key** from the `ON <target>.<key> = <source>.<key>`
///    predicate (canonical equality form).
/// 4. **Maps the WHEN clauses** to [`MatchedAction`]/[`NotMatchedAction`].
/// 5. Delegates to [`execute_merge`] which enforces uniqueness, deduplicates
///    the source, classifies rows, and commits updates/inserts.
///
/// # Supported forms (canonical upsert)
/// - `WHEN MATCHED THEN UPDATE SET …` → [`MatchedAction::UpdateAssignments`]
///   (or [`MatchedAction::UpdateAll`] when the SET list is empty)
/// - `WHEN NOT MATCHED THEN INSERT (cols) VALUES (vals)` → [`NotMatchedAction::Insert`]
///
/// # Unsupported forms (return `Err` with a clear message)
/// - `WHEN NOT MATCHED BY SOURCE THEN …` (BigQuery/T-SQL extension)
/// - `WHEN NOT MATCHED BY TARGET THEN …` (BigQuery extension)
/// - `WHEN MATCHED THEN DELETE`
/// - Conditional clauses (`WHEN MATCHED AND <predicate>`)
/// - Multiple `WHEN MATCHED` clauses
pub async fn execute_merge_sql(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    merge: datafusion::sql::sqlparser::ast::Merge,
) -> crate::Result<MergeStats> {
    let merge_delta = execute_merge_sql_to_delta(ctx, storage, table_root, merge).await?;
    if let Some(delta) = &merge_delta.delta {
        apply_delta_to_provider(ctx, &merge_delta.table, delta).await?;
    }
    Ok(merge_delta.stats)
}

/// Parse a `MERGE INTO …` statement, commit it to storage, and return the
/// statistics and delta without refreshing the registered provider.
async fn execute_merge_sql_to_delta(
    ctx: &SessionContext,
    storage: Arc<dyn Storage>,
    table_root: &str,
    merge: datafusion::sql::sqlparser::ast::Merge,
) -> crate::Result<MergeDelta> {
    // ── Step 1: Extract target table name ────────────────────────────────────
    let table_name = match &merge.table {
        TableFactor::Table { name, .. } => {
            // ObjectName → last non-schema part is the table name.
            name.0
                .last()
                .and_then(|part| part.as_ident())
                .map(|ident| ident.value.clone())
                .ok_or_else(|| {
                    QueryError::Other("MERGE: cannot extract target table name".into())
                })?
        }
        other => {
            return Err(QueryError::Other(format!(
                "MERGE INTO target must be a plain table name, got: {other}"
            )));
        }
    };

    // ── Step 2: Materialize source into a RecordBatch ────────────────────────
    //
    // The source is a `TableFactor`.  Two canonical forms are supported:
    //
    //   (a) `Derived { subquery, alias }` — a parenthesized subquery.
    //       We run `SELECT * FROM (<subquery_sql>) AS __merge_src__` to
    //       materialize it and collect the results.
    //
    //   (b) `Table { name, alias }` — a named table already registered in the
    //       DataFusion session.  We run `SELECT * FROM <name>` to collect it.
    //
    // The source alias is used inside the WHEN clause expressions to reference
    // source columns (e.g. `cdc.v`), but we strip that qualification during
    // INSERT/UPDATE since `execute_merge` works on scalar values by column name.
    let src_batch_raw = materialize_merge_source(ctx, &merge.source).await?;

    // ── Step 2b: Coerce source schema to match the target table schema ────────
    //
    // DataFusion's VALUES materializer produces nullable columns regardless of
    // the declared nullability, but `Writer::insert_batch` / `commit_update`
    // enforce exact schema equality (including `is_nullable`).  We retrieve
    // the registered provider schema for the target table and rebuild the
    // source batch with the matching nullability flags (safe because the
    // underlying data is unchanged; only the field metadata differs).
    let src_batch = coerce_batch_schema_to_table(ctx, &table_name, src_batch_raw).await?;

    // ── Step 3: Extract key column from ON predicate ──────────────────────────
    //
    // Canonical form: `t.id = src.id`  (or `src.id = t.id` — both handled).
    // We look for a top-level `BinaryOp { left, op: Eq, right }` where one
    // side is `CompoundIdentifier([target_table, col])` and the other is
    // `CompoundIdentifier([src_alias, col])`.
    let key_col = extract_merge_key(&merge.on, &table_name)?;

    // Resolve the target schema so we can validate SET targets before any write.
    let provider = ctx
        .table_provider(&table_name)
        .await
        .map_err(|e| QueryError::Other(format!("table not registered: {e}")))?;
    let arrow_schema = provider.schema();

    // ── Step 4: Map WHEN clauses to action types ──────────────────────────────
    let mut matched_action: Option<MatchedAction> = None;
    let mut found_not_matched: bool = false;

    for clause in &merge.clauses {
        // Conditional clauses are not supported.
        if clause.predicate.is_some() {
            return Err(QueryError::Other(
                "MERGE: conditional WHEN clauses (WHEN … AND <predicate>) are not supported; \
                 only unconditional canonical upsert is supported"
                    .into(),
            ));
        }
        match clause.clause_kind {
            MergeClauseKind::Matched => match &clause.action {
                MergeAction::Update(update_expr) => {
                    if matched_action.is_some() {
                        return Err(QueryError::Other(
                            "MERGE: multiple WHEN MATCHED THEN UPDATE clauses are not supported"
                                .into(),
                        ));
                    }
                    let assignments = parse_merge_update_assignments(
                        &table_name,
                        &arrow_schema,
                        &update_expr.assignments,
                    )?;
                    matched_action = if assignments.is_empty() {
                        Some(MatchedAction::UpdateAll)
                    } else {
                        Some(MatchedAction::UpdateAssignments(assignments))
                    };
                }
                MergeAction::Delete { .. } => {
                    return Err(QueryError::Other(
                        "MERGE: WHEN MATCHED THEN DELETE is not supported".into(),
                    ));
                }
                MergeAction::Insert(_) => {
                    return Err(QueryError::Other(
                        "MERGE: WHEN MATCHED THEN INSERT is not a valid MERGE clause".into(),
                    ));
                }
            },
            MergeClauseKind::NotMatched => match &clause.action {
                MergeAction::Insert(_) => {
                    found_not_matched = true;
                }
                MergeAction::Update(_) => {
                    return Err(QueryError::Other(
                        "MERGE: WHEN NOT MATCHED THEN UPDATE is not a valid MERGE clause".into(),
                    ));
                }
                MergeAction::Delete { .. } => {
                    return Err(QueryError::Other(
                        "MERGE: WHEN NOT MATCHED THEN DELETE is not a valid MERGE clause".into(),
                    ));
                }
            },
            MergeClauseKind::NotMatchedByTarget => {
                return Err(QueryError::Other(
                    "MERGE: WHEN NOT MATCHED BY TARGET is not supported; \
                     use WHEN NOT MATCHED instead"
                        .into(),
                ));
            }
            MergeClauseKind::NotMatchedBySource => {
                return Err(QueryError::Other(
                    "MERGE: WHEN NOT MATCHED BY SOURCE is not supported".into(),
                ));
            }
        }
    }

    if matched_action.is_none() && !found_not_matched {
        return Err(QueryError::Other(
            "MERGE: must have at least one WHEN MATCHED or WHEN NOT MATCHED clause".into(),
        ));
    }

    // ── Step 5: Delegate to execute_merge_to_delta ───────────────────────────
    // `found_not_matched` is guaranteed true unless only a WHEN MATCHED clause
    // is present; execute_merge handles the matched-only case correctly.
    let matched_action = matched_action.unwrap_or(MatchedAction::UpdateAll);
    let not_matched_action = if found_not_matched {
        NotMatchedAction::Insert
    } else {
        return Err(QueryError::Other(
            "MERGE: a WHEN NOT MATCHED THEN INSERT clause is required for canonical upsert; \
             MERGE without NOT MATCHED is not supported"
                .into(),
        ));
    };

    let source_alias = extract_source_alias(&merge.source).unwrap_or_default();
    if matches!(matched_action, MatchedAction::UpdateAssignments(_)) && source_alias.is_empty() {
        return Err(QueryError::Other(
            "MERGE UPDATE SET requires the USING source to have an alias".into(),
        ));
    }

    execute_merge_to_delta(
        ctx,
        storage,
        table_root,
        &table_name,
        &key_col,
        src_batch,
        &source_alias,
        matched_action,
        not_matched_action,
    )
    .await
}

/// Materialize a MERGE `USING <source>` clause into a `RecordBatch`.
///
/// Supports:
/// - `Derived { subquery, alias }` — runs the subquery SQL through DataFusion.
///   When `alias` carries column aliases (e.g. `AS src(id, v)`), wraps the
///   inner query in an outer `SELECT col0, col1, ... FROM (...) AS __inner__`
///   to guarantee the output columns bear the alias names.
/// - `Table { name }` — runs `SELECT * FROM <name>`.
async fn materialize_merge_source(
    ctx: &SessionContext,
    source: &TableFactor,
) -> crate::Result<RecordBatch> {
    let query_sql: String = match source {
        TableFactor::Derived {
            subquery, alias, ..
        } => {
            // If the alias carries column names (e.g. `AS src(id, v)`), build
            // an outer SELECT that renames `column1`, `column2`, … to the
            // declared names.  This is necessary because DataFusion's VALUES
            // materializer names output columns `column1`, `column2`, etc.
            if let Some(a) = alias {
                if !a.columns.is_empty() {
                    let col_aliases: Vec<String> = a
                        .columns
                        .iter()
                        .enumerate()
                        .map(|(i, col_def)| {
                            // DataFusion names VALUES output cols as `column1`, `column2`, …
                            // (1-indexed).
                            format!("column{} AS {}", i + 1, col_def.name.value)
                        })
                        .collect();
                    let select_list = col_aliases.join(", ");
                    format!("SELECT {select_list} FROM ({subquery}) AS __mgr_inner__")
                } else {
                    // Alias with no column aliases — just select all from the subquery.
                    format!("SELECT * FROM ({subquery}) AS {}", a.name.value)
                }
            } else {
                format!("SELECT * FROM ({subquery}) AS __mgr_src__")
            }
        }
        TableFactor::Table { name, alias, .. } => {
            let tbl = name
                .0
                .last()
                .and_then(|p| p.as_ident())
                .map(|i| i.value.clone())
                .unwrap_or_else(|| name.to_string());
            let alias_part = alias
                .as_ref()
                .map(|a| format!(" AS {}", a.name.value))
                .unwrap_or_default();
            format!("SELECT * FROM {tbl}{alias_part}")
        }
        other => {
            return Err(QueryError::Other(format!(
                "MERGE USING source must be a subquery or named table, got: {other}"
            )));
        }
    };
    materialize_sql(ctx, &query_sql).await
}

/// Coerce the schema of `batch` to match the registered target table's schema.
///
/// `Writer::insert_batch` requires the source batch schema to exactly match
/// the table schema, including the `is_nullable` flag on each field.
/// `Writer::commit_update` compares names and types only, but MERGE coerces
/// the matched source batch as well so the patch fragment schema is tidy.
/// DataFusion's VALUES materializer always produces nullable columns.
/// This function:
///
/// 1. Retrieves the target table's Arrow schema from the registered provider
///    (excluding the pseudo-columns `_rowid` and `_rowaddr`).
/// 2. For each column present in both `batch` and the table schema, rewraps the
///    column array with the table's field (which has the correct nullability).
/// 3. Returns a `RecordBatch` whose schema matches the table's data schema.
async fn coerce_batch_schema_to_table(
    ctx: &SessionContext,
    table_name: &str,
    batch: RecordBatch,
) -> crate::Result<RecordBatch> {
    let provider = ctx
        .table_provider(table_name)
        .await
        .map_err(|e| QueryError::Other(format!("table not registered for schema coercion: {e}")))?;
    let full_schema = provider.schema();

    // The full schema includes pseudo-columns (_rowid, _rowaddr) at the end;
    // strip them by keeping only fields whose names match the source batch.
    let batch_schema = batch.schema();
    let src_field_names: std::collections::HashSet<&str> = batch_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();

    // Build the coerced schema using table field definitions for matching columns.
    let mut coerced_fields: Vec<Arc<arrow::datatypes::Field>> = Vec::new();
    let mut coerced_cols: Vec<arrow::array::ArrayRef> = Vec::new();

    for table_field in full_schema.fields().iter() {
        if !src_field_names.contains(table_field.name().as_str()) {
            continue;
        }
        let src_col_idx = batch_schema.index_of(table_field.name()).map_err(|e| {
            QueryError::Other(format!(
                "coerce: column '{}' not found: {e}",
                table_field.name()
            ))
        })?;

        coerced_fields.push(Arc::clone(table_field));
        coerced_cols.push(Arc::clone(batch.column(src_col_idx)));
    }

    let coerced_schema = Arc::new(arrow::datatypes::Schema::new(coerced_fields));
    RecordBatch::try_new(coerced_schema, coerced_cols).map_err(QueryError::Arrow)
}

/// Run `query_sql` through DataFusion and collect all result batches into one.
async fn materialize_sql(ctx: &SessionContext, query_sql: &str) -> crate::Result<RecordBatch> {
    let batches = ctx
        .sql(query_sql)
        .await
        .map_err(|e| QueryError::Other(format!("MERGE source materialization error: {e}")))?
        .collect()
        .await
        .map_err(|e| QueryError::Other(format!("MERGE source collection error: {e}")))?;

    if batches.is_empty() {
        return Err(QueryError::Other(
            "MERGE source produced no batches; cannot infer schema".into(),
        ));
    }

    let schema = batches[0].schema();
    arrow::compute::concat_batches(&schema, &batches).map_err(QueryError::Arrow)
}

/// Extract the target-table key column name from a MERGE ON predicate.
///
/// Canonical form: `target_table.col = source_alias.col`
/// (or reversed).  We accept any `BinaryOp { Eq }` where one side is a
/// `CompoundIdentifier([table_qualifier, col])` referencing `target_table`.
///
/// Returns the column name (without the table qualifier).
fn extract_merge_key(on: &SqlExpr, target_table: &str) -> crate::Result<String> {
    let target_lower = target_table.to_lowercase();

    match on {
        SqlExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
        } => {
            // Helper: if `expr` is a CompoundIdentifier whose qualifier matches
            // the target table, return the column name.  Plain identifiers are
            // not accepted here — they are ambiguous (either side could be the
            // target or the source).  The column name is normalized (unquoted →
            // lowercase, quoted preserved) so an ON clause like `t.Id = src.Id`
            // resolves to the declared `id` column instead of the case-sensitive
            // quoted `"Id"` failing the JOIN.
            let try_extract_compound = |expr: &SqlExpr| -> Option<String> {
                match expr {
                    SqlExpr::CompoundIdentifier(parts) if parts.len() == 2 => {
                        let qualifier = parts[0].value.to_lowercase();
                        if qualifier == target_lower {
                            Some(normalize_sql_identifier(&parts[1]))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            };

            // Prefer the side whose qualifier matches the target table.
            if let Some(col) = try_extract_compound(left) {
                return Ok(col);
            }
            if let Some(col) = try_extract_compound(right) {
                return Ok(col);
            }

            // Neither side is a CompoundIdentifier qualified by the target table.
            // Plain identifiers are ambiguous — we cannot safely determine which
            // side is the target column vs the source column.
            Err(QueryError::Other(format!(
                "MERGE ON predicate must reference the target table explicitly, e.g. \
                 `{target_table}.key = source.key`; cannot disambiguate plain \
                 identifiers in: {on}"
            )))
        }
        other => Err(QueryError::Other(format!(
            "MERGE ON predicate must be a simple equality (`=`); got: {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema as ArrowSchema};
    use datafusion::logical_expr::lit;
    use icefalldb_core::arrow_schema_to_icefalldb;
    use icefalldb_core::metadata::Manifest;
    use icefalldb_core::storage::local::LocalStorage;
    use icefalldb_core::storage::Storage;
    use icefalldb_core::{DeletionVector, Writer};

    use crate::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};

    /// Build a ProviderConfig suitable for mutation tests: no native Parquet
    /// (forces the custom scan that synthesizes pseudo-columns), no tiny-table
    /// cache (so deletion vectors are always applied live), single partition.
    fn mutate_provider_config() -> ProviderConfig {
        ProviderConfig {
            batch_size: 1024,
            target_partitions: 1,
            io_coalesce_window: 0,
            io_concurrency: 1,
            // Force the custom IcefallDBScanExec path, which synthesises
            // _rowid / _rowaddr pseudo-columns.
            native_parquet_threshold: usize::MAX,
            parquet_metadata_cache_capacity: 0,
            // Bypass the tiny-table cache so deletion vectors are applied every
            // time (the cache only materialises live rows, but we want to prove
            // the scan path itself works correctly for the deletion test).
            tiny_table_cache_threshold_rows: 0,
            tiny_table_cache_threshold_bytes: 0,
            wal_mode: true,
        }
    }

    /// Build a two-fragment table registered as "t" in a fresh session.
    ///
    /// Fragment 0 (fragment_id = 0): 5 rows, row_ids 0..=4, col v = row_id * 10
    ///                               (v = 0, 10, 20, 30, 40)
    /// Fragment 1 (fragment_id = 1): 5 rows, row_ids 5..=9, no `v` column
    ///                               (written without the v column to keep the
    ///                                schema stable; the column is still present
    ///                                in the schema with nulls for those rows)
    ///
    /// Returns (SessionContext, tempdir) — the tempdir must be kept alive for
    /// the duration of the test.
    async fn test_table_two_fragments() -> (SessionContext, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        // Arrow schema: one Int64 column "v".
        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            true,
        )]));

        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        // Each batch is its own row group / fragment.
        mdb_schema.row_group_target_rows = 5;

        // ── Fragment 0: rows 0..5, v = row_id * 10 ──────────────────────────
        let mut writer = Writer::create(Arc::clone(&storage), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch0 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![0i64, 10, 20, 30, 40]))],
        )
        .unwrap();
        writer.insert_batch(batch0).await.unwrap();
        writer.commit().await.unwrap();

        // ── Fragment 1: rows 5..10, v = null (or any value; we don't filter on
        //    them in the locate test) ─────────────────────────────────────────
        let mut writer = Writer::new(Arc::clone(&storage), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch1 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![50i64, 60, 70, 80, 90]))],
        )
        .unwrap();
        writer.insert_batch(batch1).await.unwrap();
        writer.commit().await.unwrap();

        // Register the table.
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", mutate_provider_config())
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t", provider).unwrap();

        (ctx, tmp)
    }

    // ── Acceptance test 1: locate returns the correct MatchLoc ───────────────

    /// `locate_matches` returns exactly one result for `v = 30`.
    ///
    /// Table layout:
    ///   Fragment 0 (fragment_id = 0), row_ids 0..=4, offsets 0..=4:
    ///     offset 0 → row_id 0, v = 0
    ///     offset 1 → row_id 1, v = 10
    ///     offset 2 → row_id 2, v = 20
    ///     offset 3 → row_id 3, v = 30   ← this is the one we locate
    ///     offset 4 → row_id 4, v = 40
    ///
    /// Note: the brief describes these as "fragment_id 1" using 1-indexed
    /// notation; the implementation assigns fragment_id = 0 to the first
    /// committed fragment (the Writer allocates IDs from `next_fragment_id`
    /// which starts at 0).
    #[tokio::test]
    async fn locate_returns_matching_offsets_per_fragment() {
        let (ctx, _tmp) = test_table_two_fragments().await;

        let locs = locate_matches(&ctx, "t", col("v").eq(lit(30i64)))
            .await
            .unwrap();

        assert_eq!(locs.len(), 1, "expected exactly one matching row");
        assert_eq!(locs[0].row_id, 3, "row_id for v=30 must be 3");
        assert_eq!(
            (locs[0].fragment_id, locs[0].offset),
            (0, 3),
            "v=30 is in fragment_id=0 at physical offset 3"
        );
    }

    // ── Micro-bench: index fast-path vs DataFusion scan for `locate` ─────────
    //
    // Isolates the row-location cost (the dominant WARM mutation cost) from the
    // fsync-bound commit. Run manually:
    //   cargo test -p icefalldb-query --release locate_fast_path_vs_scan_bench \
    //     -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "timing micro-bench; run manually with --release --ignored --nocapture"]
    async fn locate_fast_path_vs_scan_bench() {
        use std::time::Instant;

        const N_ROWS: i64 = 100_000;
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ])));
        mdb_schema.row_group_target_rows = N_ROWS as usize; // single fragment

        let catalog = icefalldb_core::database_catalog::DatabaseCatalog::new(storage.clone());
        let guard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_table(&guard, "t", &mdb_schema)
            .await
            .unwrap();
        catalog
            .create_index_definition(&guard, "id_idx", "t", "id", "btree")
            .await
            .unwrap();
        drop(guard);

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from((0..N_ROWS).collect::<Vec<_>>())),
                Arc::new(Int64Array::from(
                    (0..N_ROWS).map(|x| x * 7).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let mut writer = Writer::new(storage.clone(), "t", mdb_schema).await.unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let provider = Arc::new(
            IcefallDBTableProvider::new(storage.clone(), "t", ProviderConfig::default())
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(4, 8192);
        ctx.register_table("t", provider.clone()).unwrap();

        let key = N_ROWS / 2; // a middle id; worst case for an ordered scan
        let pred = || col("id").eq(lit(key));

        // Correctness: the fast path and the DataFusion scan must agree exactly.
        let fast_once = provider.try_locate_by_index(&pred()).await.unwrap();
        assert!(
            fast_once.is_some(),
            "fast path must engage for indexed id = k"
        );
        let fast_once = fast_once.unwrap();
        let scan_once = {
            let df = ctx
                .table("t")
                .await
                .unwrap()
                .filter(pred())
                .unwrap()
                .select(vec![col("_rowid"), col("_rowaddr")])
                .unwrap();
            let mut out = Vec::new();
            for b in df.collect().await.unwrap() {
                let ids = b.column(0).as_primitive::<UInt64Type>();
                let addr = b.column(1).as_primitive::<UInt64Type>();
                for i in 0..b.num_rows() {
                    let a = addr.value(i);
                    out.push(MatchLoc {
                        fragment_id: a >> 32,
                        offset: (a & 0xFFFF_FFFF) as u32,
                        row_id: ids.value(i),
                    });
                }
            }
            out
        };
        assert_eq!(
            fast_once, scan_once,
            "fast path and scan must return the same rows"
        );
        assert_eq!(fast_once.len(), 1);

        let iters = 50u32;
        let med = |mut v: Vec<f64>| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };

        let mut fast_ms = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let _ = provider.try_locate_by_index(&pred()).await.unwrap();
            fast_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let mut scan_ms = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let df = ctx
                .table("t")
                .await
                .unwrap()
                .filter(pred())
                .unwrap()
                .select(vec![col("_rowid"), col("_rowaddr")])
                .unwrap();
            let _ = df.collect().await.unwrap();
            scan_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }

        let (f, s) = (med(fast_ms), med(scan_ms));
        println!(
            "\nlocate @ {N_ROWS} rows, 1 fragment, indexed id = {key}:\n  \
             DataFusion scan : {s:8.3} ms (median of {iters})\n  \
             index fast path : {f:8.3} ms (median of {iters})\n  \
             speedup         : {:8.1}x\n",
            s / f
        );
    }

    // ── Acceptance test 2: deleted rows are not located ──────────────────────

    /// A row whose deletion vector bit is set must NOT appear in `locate_matches`
    /// output, even when the predicate would otherwise match it.
    ///
    /// We write a single-fragment table (5 rows, v = 0,10,20,30,40), inject a
    /// deletion vector that marks offset 3 (v=30) as deleted, then verify that
    /// `locate_matches` with `v = 30` returns an empty result.
    #[tokio::test]
    async fn locate_skips_deleted_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "v",
            DataType::Int64,
            true,
        )]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 5;

        // Write 5 rows: v = 0, 10, 20, 30, 40.
        let mut writer = Writer::create(Arc::clone(&storage), "t", mdb_schema.clone())
            .await
            .unwrap();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(vec![0i64, 10, 20, 30, 40]))],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        // Read the manifest and inject a deletion vector that marks offset 3 (v=30).
        let pointer_bytes = storage.read("t/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();

        let manifest_path = format!("t/{}", Manifest::filename(seq));
        let manifest_bytes = storage.read(&manifest_path).await.unwrap();
        let mut manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();

        // Delete physical offset 3 (the row with v=30).
        let del_rel_path = "_deletions/rg0__v1.del";
        let del_storage_path = format!("t/{}", del_rel_path);
        let mut dv = DeletionVector::default();
        dv.union_offsets([3u32]);
        let del_bytes = dv.serialize();
        storage.write(&del_storage_path, &del_bytes).await.unwrap();

        assert!(
            !manifest.row_groups.is_empty(),
            "expected at least one row group"
        );
        manifest.row_groups[0].deletes = Some(del_rel_path.to_string());
        manifest.row_groups[0].deleted_count = 1;

        let next_seq = seq + 1;
        manifest.sequence = next_seq;
        manifest.checksum = String::new();
        manifest.checksum = manifest.compute_checksum().unwrap();

        let new_manifest_path = format!("t/{}", Manifest::filename(next_seq));
        storage
            .write(&new_manifest_path, &serde_json::to_vec(&manifest).unwrap())
            .await
            .unwrap();
        storage
            .write(
                "t/_manifest.json",
                &serde_json::to_vec(&serde_json::json!({"latest": next_seq})).unwrap(),
            )
            .await
            .unwrap();

        // Register the provider and run locate_matches.
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", mutate_provider_config())
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t", provider).unwrap();

        let locs = locate_matches(&ctx, "t", col("v").eq(lit(30i64)))
            .await
            .unwrap();

        assert!(
            locs.is_empty(),
            "deleted row (v=30, offset=3) must not be located; got: {locs:?}"
        );
    }

    // ── End-to-end execute_sql tests ─────────────────────────────────────────

    /// Build a table named `table_name` with `rows` rows in a fresh tempdir.
    ///
    /// The table has a single `id` column (Int64) where `id = 0, 1, 2, …,
    /// rows-1`.  All rows are written in a single fragment so that the row-id
    /// equals the row offset.  Returns `(SessionContext, Arc<dyn Storage>,
    /// table_root_string, TempDir)`.
    async fn registered_table(
        table_name: &str,
        rows: usize,
    ) -> (SessionContext, Arc<dyn Storage>, String, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = rows.max(1);

        let ids: Vec<i64> = (0..rows as i64).collect();
        let batch = arrow::array::RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from(ids))],
        )
        .unwrap();

        let mut writer = Writer::create(Arc::clone(&storage), table_name, mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), table_name, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(table_name, provider).unwrap();

        let root = table_name.to_string();
        (ctx, storage, root, tmp)
    }

    /// Execute `sql` against `ctx` and return the first column of the first
    /// row as an i64.
    async fn scalar_i64(ctx: &SessionContext, sql: &str) -> i64 {
        use arrow::array::AsArray;
        use arrow::datatypes::Int64Type;

        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let batch = batches.first().expect("no batches returned");
        batch.column(0).as_primitive::<Int64Type>().value(0)
    }

    /// `execute_sql` deletes exactly one row matching `WHERE id = 5` from a
    /// 100-row table and the subsequent COUNT(*) reflects the deletion.
    #[tokio::test]
    async fn end_to_end_sql_delete_single_row() {
        let (ctx, storage, root, _tmp) = registered_table("t_e2e_single", 100).await;

        let affected = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "DELETE FROM t_e2e_single WHERE id = 5",
        )
        .await
        .unwrap();
        assert_eq!(affected, 1, "expected exactly 1 row deleted");

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_single").await,
            99,
            "table should have 99 rows after deleting id=5"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_single WHERE id = 5").await,
            0,
            "id=5 must not be visible after deletion"
        );
    }

    /// `execute_sql` deletes multiple rows matching a range predicate and the
    /// subsequent COUNT(*) reflects the deletion.
    #[tokio::test]
    async fn end_to_end_sql_delete_range() {
        // 100 rows: id = 0..99.  Delete id >= 90  → 10 rows deleted → 90 remain.
        let (ctx, storage, root, _tmp) = registered_table("t_e2e_range", 100).await;

        let affected = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "DELETE FROM t_e2e_range WHERE id >= 90",
        )
        .await
        .unwrap();
        assert_eq!(affected, 10, "expected 10 rows deleted");

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_range").await,
            90,
            "table should have 90 rows after deleting id>=90"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_range WHERE id >= 90").await,
            0,
            "rows with id>=90 must not be visible after deletion"
        );
    }

    // ── plan_update ────────────────────────────────────────────────────

    /// `plan_update` applies `SET v = v + 1` against the pre-image value.
    ///
    /// The table fixture (`test_table_two_fragments`) has row_id 3 with v = 30.
    /// After `plan_update` with `WHERE v = 30` and `SET v = v + 1`:
    ///   - Exactly one `MatchLoc` should be returned with `row_id == 3`.
    ///   - The `rows` RecordBatch should contain v = 31 (pre-image 30 + 1).
    #[tokio::test]
    async fn plan_update_applies_set_over_preimage() {
        let (ctx, _tmp) = test_table_two_fragments().await;
        let ub = plan_update(
            &ctx,
            "t",
            &[("v".into(), col("v") + lit(1i64))],
            col("v").eq(lit(30i64)),
        )
        .await
        .unwrap();

        assert_eq!(ub.locs.len(), 1, "expected exactly one matching row");
        assert_eq!(ub.locs[0].row_id, 3, "row_id for v=30 must be 3");
        let v = ub
            .rows
            .column_by_name("v")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(
            v.value(0),
            31,
            "SET v = v + 1 must yield 31 from pre-image 30"
        );
    }

    /// plan_update fast path: when the predicate is answerable from a secondary
    /// index, the read is restricted to the matched rows via the `_rowid IN`
    /// pushdown (the same locate DELETE uses) instead of scanning the whole
    /// fragment.  The post-image must be identical to the scan path: the located
    /// row and the self-referential `SET v = v + 1` over its pre-image.
    #[tokio::test]
    async fn plan_update_index_fast_path_matches_scan() {
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;

        let (_ctx0, storage, _root, _tmp) = registered_table_two_cols("t_fast_upd", 50).await;

        // Index the PREDICATE column `id` so try_locate_by_index answers `id = 5`.
        let catalog = DatabaseCatalog::new(Arc::clone(&storage));
        let lock: CatalogLockGuard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_index_definition(&lock, "idx_id", "t_fast_upd", "id", "btree")
            .await
            .unwrap();
        drop(lock);

        let pointer_bytes = storage.read("t_fast_upd/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest_path = format!("t_fast_upd/{}", Manifest::filename(seq));
        let manifest_bytes = storage.read(&manifest_path).await.unwrap();
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
        let definition = IndexDefinition {
            name: "idx_id".to_string(),
            table: "t_fast_upd".to_string(),
            column: "id".to_string(),
            unique: false,
        };
        build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap()
            .save(storage.as_ref())
            .await
            .unwrap();

        // Fresh provider + session so the snapshot loads the just-saved index,
        // then plan the UPDATE through the index fast path.
        let provider = Arc::new(
            IcefallDBTableProvider::new(
                Arc::clone(&storage),
                "t_fast_upd",
                mutate_provider_config(),
            )
            .await
            .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t_fast_upd", provider).unwrap();

        let ub = plan_update(
            &ctx,
            "t_fast_upd",
            &[("v".into(), col("v") + lit(1i64))],
            col("id").eq(lit(5i64)),
        )
        .await
        .unwrap();

        assert_eq!(
            ub.locs.len(),
            1,
            "index fast path must locate exactly one row"
        );
        assert_eq!(ub.locs[0].row_id, 5, "row_id for id=5 must be 5");
        let id = ub
            .rows
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(id.value(0), 5, "passthrough id must remain 5");
        let v = ub
            .rows
            .column_by_name("v")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(
            v.value(0),
            51,
            "SET v = v + 1 over pre-image 50 must yield 51"
        );
    }

    /// A blind UPDATE (`SET v = <literal>`) on an `id`-indexed table
    /// whose every column is determined by the SET literal + the `id = k`
    /// equality is planned WITHOUT reading the pre-image. Proven by deleting the
    /// fragment Parquet first: the plan still produces the correct, byte-equal
    /// post-image. (The self-referential `SET v = v + 1` case still reads — see
    /// `plan_update_index_fast_path_matches_scan`.)
    #[tokio::test]
    async fn plan_update_blind_elides_read() {
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;

        let (_ctx0, storage, _root, _tmp) = registered_table_two_cols("t_blind_upd", 50).await;

        let catalog = DatabaseCatalog::new(Arc::clone(&storage));
        let lock: CatalogLockGuard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_index_definition(&lock, "idx_id", "t_blind_upd", "id", "btree")
            .await
            .unwrap();
        drop(lock);

        let pointer: serde_json::Value =
            serde_json::from_slice(&storage.read("t_blind_upd/_manifest.json").await.unwrap())
                .unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("t_blind_upd/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        let definition = IndexDefinition {
            name: "idx_id".to_string(),
            table: "t_blind_upd".to_string(),
            column: "id".to_string(),
            unique: false,
        };
        build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap()
            .save(storage.as_ref())
            .await
            .unwrap();
        // Path of the only fragment's Parquet, so we can delete it below.
        let frag_data_path = format!("t_blind_upd/{}", manifest.row_groups[0].data);

        let provider = Arc::new(
            IcefallDBTableProvider::new(
                Arc::clone(&storage),
                "t_blind_upd",
                mutate_provider_config(),
            )
            .await
            .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t_blind_upd", provider).unwrap();

        // Sanity: a blind determined UPDATE produces the right post-image.
        let ub = plan_update(
            &ctx,
            "t_blind_upd",
            &[("v".into(), lit(999i64))],
            col("id").eq(lit(5i64)),
        )
        .await
        .unwrap();
        assert_eq!(ub.locs.len(), 1);
        assert_eq!(ub.locs[0].row_id, 5);
        let id = ub
            .rows
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        let v = ub
            .rows
            .column_by_name("v")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(id.value(0), 5, "id pinned by the predicate");
        assert_eq!(v.value(0), 999, "v set to the literal");

        // No-read proof: delete the fragment Parquet, then run another blind
        // determined UPDATE. It must still succeed — the post-image was built from
        // the located row-id + literals, never reading the (now-absent) data.
        storage.delete(&frag_data_path).await.unwrap();
        assert!(!storage.exists(&frag_data_path).await.unwrap());

        let ub2 = plan_update(
            &ctx,
            "t_blind_upd",
            &[("v".into(), lit(123i64))],
            col("id").eq(lit(7i64)),
        )
        .await
        .unwrap();
        assert_eq!(ub2.locs.len(), 1);
        assert_eq!(ub2.locs[0].row_id, 7);
        let id2 = ub2
            .rows
            .column_by_name("id")
            .unwrap()
            .as_primitive::<Int64Type>();
        let v2 = ub2
            .rows
            .column_by_name("v")
            .unwrap()
            .as_primitive::<Int64Type>();
        assert_eq!(id2.value(0), 7, "id pinned without reading the fragment");
        assert_eq!(v2.value(0), 123, "v set without reading the fragment");
    }

    /// Byte-equality vs the slow path: a blind `SET v = <literal>`
    /// committed on an `id`-indexed table (which elides the read) must produce
    /// the same on-disk result as the identical UPDATE on a non-indexed table
    /// (which reads the pre-image).
    #[tokio::test]
    async fn blind_update_elided_matches_read_path_e2e() {
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;

        // Non-indexed → read path.
        let (ctx_r, sr, root_r, _tr) = registered_table_two_cols("t_blind_read", 40).await;
        // Indexed on id → elision path.
        let (_c0, si, root_i, _ti) = registered_table_two_cols("t_blind_elide", 40).await;
        let catalog = DatabaseCatalog::new(Arc::clone(&si));
        let lock: CatalogLockGuard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_index_definition(&lock, "idx_id", "t_blind_elide", "id", "btree")
            .await
            .unwrap();
        drop(lock);
        let ptr: serde_json::Value =
            serde_json::from_slice(&si.read("t_blind_elide/_manifest.json").await.unwrap())
                .unwrap();
        let seq = ptr["latest"].as_u64().unwrap();
        let manifest: Manifest = serde_json::from_slice(
            &si.read(&format!("t_blind_elide/{}", Manifest::filename(seq)))
                .await
                .unwrap(),
        )
        .unwrap();
        let def = IndexDefinition {
            name: "idx_id".into(),
            table: "t_blind_elide".into(),
            column: "id".into(),
            unique: false,
        };
        build_btree_index(si.as_ref(), &def, &manifest)
            .await
            .unwrap()
            .save(si.as_ref())
            .await
            .unwrap();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&si), "t_blind_elide", mutate_provider_config())
                .await
                .unwrap(),
        );
        let ctx_i = icefalldb_session(1, 1024);
        ctx_i.register_table("t_blind_elide", provider).unwrap();

        // Same blind UPDATE on both paths.
        assert_eq!(
            execute_sql(
                &ctx_r,
                Arc::clone(&sr),
                &root_r,
                "UPDATE t_blind_read SET v = 777 WHERE id = 7"
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx_i,
                Arc::clone(&si),
                &root_i,
                "UPDATE t_blind_elide SET v = 777 WHERE id = 7"
            )
            .await
            .unwrap(),
            1
        );

        // Identical committed result on both paths.
        assert_eq!(
            scalar_i64(&ctx_r, "SELECT v FROM t_blind_read WHERE id = 7").await,
            777
        );
        assert_eq!(
            scalar_i64(&ctx_i, "SELECT v FROM t_blind_elide WHERE id = 7").await,
            777,
            "elided blind UPDATE must match the read path"
        );
        assert_eq!(
            scalar_i64(&ctx_i, "SELECT id FROM t_blind_elide WHERE id = 7").await,
            7,
            "the elided row preserves its key"
        );
        assert_eq!(
            scalar_i64(&ctx_r, "SELECT COUNT(*) FROM t_blind_read").await,
            scalar_i64(&ctx_i, "SELECT COUNT(*) FROM t_blind_elide").await
        );
        assert_eq!(
            scalar_i64(&ctx_r, "SELECT v FROM t_blind_read WHERE id = 20").await,
            scalar_i64(&ctx_i, "SELECT v FROM t_blind_elide WHERE id = 20").await,
            "untouched rows identical across paths"
        );
    }

    // ── end-to-end UPDATE SQL surface ──────────────────────────────────

    /// Build a table with `rows` rows and two columns: `id` (Int64, 0..rows-1)
    /// and `v` (Int64, id * 10).  All rows are written in a single fragment so
    /// that row_id equals physical offset.
    ///
    /// Returns `(SessionContext, Arc<dyn Storage>, table_root_string, TempDir)`.
    async fn registered_table_two_cols(
        table_name: &str,
        rows: usize,
    ) -> (SessionContext, Arc<dyn Storage>, String, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = rows.max(1);

        let ids: Vec<i64> = (0..rows as i64).collect();
        let vs: Vec<i64> = ids.iter().map(|i| i * 10).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(vs)),
            ],
        )
        .unwrap();

        let mut writer = Writer::create(Arc::clone(&storage), table_name, mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), table_name, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(table_name, provider).unwrap();

        let root = table_name.to_string();
        (ctx, storage, root, tmp)
    }

    /// Build a table with mixed nullable types for UPDATE NULL tests.
    ///
    /// Columns: `id` (Int64, non-nullable), `score` (Int64, nullable),
    /// `name` (Utf8, nullable), `weight` (Float64, nullable),
    /// `active` (Boolean, nullable). All rows are written in a single fragment.
    ///
    /// Returns `(SessionContext, Arc<dyn Storage>, table_root_string, TempDir)`.
    async fn registered_table_nullable_types(
        table_name: &str,
        rows: usize,
    ) -> (SessionContext, Arc<dyn Storage>, String, tempfile::TempDir) {
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("score", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
            Field::new("weight", DataType::Float64, true),
            Field::new("active", DataType::Boolean, true),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = rows.max(1);

        let ids: Vec<i64> = (0..rows as i64).collect();
        let scores: Vec<Option<i64>> = ids.iter().map(|i| Some(i * 10)).collect();
        let names: Vec<Option<String>> = ids.iter().map(|i| Some(format!("row-{i}"))).collect();
        let weights: Vec<Option<f64>> = ids.iter().map(|i| Some(*i as f64 * 1.5)).collect();
        let actives: Vec<Option<bool>> = ids.iter().map(|i| Some(i % 2 == 0)).collect();

        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(scores)),
                Arc::new(StringArray::from(names)),
                Arc::new(Float64Array::from(weights)),
                Arc::new(BooleanArray::from(actives)),
            ],
        )
        .unwrap();

        let mut writer = Writer::create(Arc::clone(&storage), table_name, mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), table_name, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(table_name, provider).unwrap();

        let root = table_name.to_string();
        (ctx, storage, root, tmp)
    }

    /// A table sorted on `id` needs **no secondary index** — a point
    /// UPDATE/DELETE locates via the proven, stats-pruned scan path and produces
    /// results byte-equal to the same ops on an `id`-indexed copy. (Per the design
    /// decision, a custom binary-search/page-index locate is deferred behind
    /// benchmarks; the scan fallback already locates correctly here, so sorted
    /// tables can skip the index and pay zero index load/maintenance.)
    #[tokio::test]
    async fn sorted_key_locate_without_index_matches_indexed_path() {
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;

        async fn add_id_index(storage: &Arc<dyn Storage>, table: &str) -> SessionContext {
            let catalog = DatabaseCatalog::new(Arc::clone(storage));
            let lock: CatalogLockGuard = catalog
                .acquire_lock(std::time::Duration::from_secs(5))
                .await
                .unwrap();
            catalog
                .create_index_definition(&lock, "idx_id", table, "id", "btree")
                .await
                .unwrap();
            drop(lock);
            let ptr: serde_json::Value = serde_json::from_slice(
                &storage
                    .read(&format!("{table}/_manifest.json"))
                    .await
                    .unwrap(),
            )
            .unwrap();
            let seq = ptr["latest"].as_u64().unwrap();
            let manifest: Manifest = serde_json::from_slice(
                &storage
                    .read(&format!("{table}/{}", Manifest::filename(seq)))
                    .await
                    .unwrap(),
            )
            .unwrap();
            let def = IndexDefinition {
                name: "idx_id".into(),
                table: table.into(),
                column: "id".into(),
                unique: false,
            };
            build_btree_index(storage.as_ref(), &def, &manifest)
                .await
                .unwrap()
                .save(storage.as_ref())
                .await
                .unwrap();
            let provider = Arc::new(
                IcefallDBTableProvider::new(Arc::clone(storage), table, mutate_provider_config())
                    .await
                    .unwrap(),
            );
            let ctx = icefalldb_session(1, 1024);
            ctx.register_table(table, provider).unwrap();
            ctx
        }

        let (ctx_a, sa, root_a, _ta) = registered_table_two_cols("t_sortnoidx", 50).await;
        let (_ctx_b0, sb, root_b, _tb) = registered_table_two_cols("t_sortidx", 50).await;
        let ctx_b = add_id_index(&sb, "t_sortidx").await;

        // Identical point ops on both: UPDATE id=7, DELETE id=13.
        assert_eq!(
            execute_sql(
                &ctx_a,
                Arc::clone(&sa),
                &root_a,
                "UPDATE t_sortnoidx SET v = v + 100 WHERE id = 7",
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx_a,
                Arc::clone(&sa),
                &root_a,
                "DELETE FROM t_sortnoidx WHERE id = 13",
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx_b,
                Arc::clone(&sb),
                &root_b,
                "UPDATE t_sortidx SET v = v + 100 WHERE id = 7",
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx_b,
                Arc::clone(&sb),
                &root_b,
                "DELETE FROM t_sortidx WHERE id = 13",
            )
            .await
            .unwrap(),
            1
        );

        // The no-index (sort-key) path matches the indexed path exactly.
        assert_eq!(
            scalar_i64(&ctx_a, "SELECT COUNT(*) FROM t_sortnoidx").await,
            49
        );
        assert_eq!(
            scalar_i64(&ctx_a, "SELECT COUNT(*) FROM t_sortnoidx").await,
            scalar_i64(&ctx_b, "SELECT COUNT(*) FROM t_sortidx").await
        );
        assert_eq!(
            scalar_i64(&ctx_a, "SELECT v FROM t_sortnoidx WHERE id = 7").await,
            170,
            "updated row value"
        );
        assert_eq!(
            scalar_i64(&ctx_a, "SELECT v FROM t_sortnoidx WHERE id = 7").await,
            scalar_i64(&ctx_b, "SELECT v FROM t_sortidx WHERE id = 7").await
        );
        assert_eq!(
            scalar_i64(&ctx_a, "SELECT COUNT(*) FROM t_sortnoidx WHERE id = 13").await,
            0,
            "deleted row gone"
        );
        assert_eq!(
            scalar_i64(&ctx_b, "SELECT COUNT(*) FROM t_sortidx WHERE id = 13").await,
            0
        );
        // An untouched row is identical across both paths.
        assert_eq!(
            scalar_i64(&ctx_a, "SELECT v FROM t_sortnoidx WHERE id = 20").await,
            scalar_i64(&ctx_b, "SELECT v FROM t_sortidx WHERE id = 20").await
        );
    }

    /// End-to-end UPDATE: `UPDATE t SET v = v + 1 WHERE id = 5`.
    ///
    /// Verifies:
    ///  1. Returns 1 (exactly one row updated).
    ///  2. The updated row is visible with the new value immediately after
    ///     (proves the table registration refresh works).
    ///  3. No duplicate row at id=5 (move-stable: original is tombstoned).
    ///  4. Total row count is unchanged.
    #[tokio::test]
    async fn end_to_end_sql_update() {
        let (ctx, storage, root, _tmp) = registered_table_two_cols("t_e2e_upd", 100).await;

        // Capture the pre-update value of v for id=5 (should be 50).
        let pre_v = scalar_i64(&ctx, "SELECT v FROM t_e2e_upd WHERE id = 5").await;
        assert_eq!(pre_v, 50, "pre-update v for id=5 must be 50");

        let n = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE t_e2e_upd SET v = v + 1 WHERE id = 5",
        )
        .await
        .unwrap();
        assert_eq!(n, 1, "UPDATE must report 1 row affected");

        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_e2e_upd WHERE id = 5").await,
            pre_v + 1,
            "post-update v for id=5 must be pre_v + 1"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_upd WHERE id = 5").await,
            1,
            "no duplicate row: id=5 must appear exactly once"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_upd").await,
            100,
            "total row count must remain 100"
        );
    }

    /// End-to-end UPDATE on an indexed column: after updating `v`, a
    /// point-lookup by the NEW value through the secondary index must find the
    /// row, and the OLD value must no longer appear in the index.
    ///
    /// This exercises the incremental index maintenance path
    /// (`Writer::commit_update` → `IndexMaintainer::maintain_on_update`)
    /// wired end-to-end through `execute_sql`.
    #[tokio::test]
    async fn end_to_end_sql_update_indexed_column() {
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;

        let (ctx, storage, root, _tmp) = registered_table_two_cols("t_idx_upd", 20).await;

        // Register a secondary btree index on column `v` via the DatabaseCatalog.
        let catalog = DatabaseCatalog::new(Arc::clone(&storage));
        let _lock: CatalogLockGuard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_index_definition(&_lock, "idx_v", "t_idx_upd", "v", "btree")
            .await
            .unwrap();
        drop(_lock);

        // Build the initial index from the current snapshot and save it so the
        // provider can load it on refresh.
        let pointer_bytes = storage.read("t_idx_upd/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest_path = format!("t_idx_upd/{}", Manifest::filename(seq));
        let manifest_bytes = storage.read(&manifest_path).await.unwrap();
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();

        let definition = IndexDefinition {
            name: "idx_v".to_string(),
            table: "t_idx_upd".to_string(),
            column: "v".to_string(),
            unique: false,
        };
        let base_index = build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap();
        // The index for v=50 (id=5) must point to row_id 5.
        assert!(
            base_index.lookup("50").contains(&5u64),
            "pre-update: index for v=50 must contain row_id 5"
        );
        base_index.save(storage.as_ref()).await.unwrap();

        // Execute the UPDATE via SQL.  This must trigger incremental index
        // maintenance: tombstone old v=50 → row_id=5 and add new v=51 → row_id=5.
        let n = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE t_idx_upd SET v = v + 1 WHERE id = 5",
        )
        .await
        .unwrap();
        assert_eq!(n, 1, "UPDATE must report 1 row affected");

        // Post-update data assertions.
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_idx_upd WHERE id = 5").await,
            51,
            "post-update v for id=5 must be 51"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_idx_upd WHERE id = 5").await,
            1,
            "no duplicate row after update"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_idx_upd").await,
            20,
            "total row count must remain 20"
        );

        // Load the updated index from storage and verify the new value is indexed.
        // After commit_update, the live manifest (pointer + WAL replay) records the
        // updated index generation.
        let new_manifest = latest_manifest(storage.as_ref(), "t_idx_upd").await;

        // Load the index via its versioned generation reference from the new manifest.
        let index_ref = new_manifest
            .index_generations
            .get("idx_v")
            .expect("new manifest must contain index_generations entry for idx_v");

        use icefalldb_core::index::load_index_by_ref;
        let updated_index = load_index_by_ref(storage.as_ref(), "t_idx_upd", "idx_v", index_ref)
            .await
            .unwrap()
            .expect("updated index must exist");

        // The new value v=51 must now be findable via the index.
        assert!(
            !updated_index.lookup("51").is_empty(),
            "post-update: index must contain an entry for the new value v=51"
        );
        // The old value v=50 must no longer point to row_id=5 (tombstoned).
        assert!(
            !updated_index.lookup("50").contains(&5u64),
            "post-update: index entry for old value v=50 must no longer map to row_id=5"
        );
    }

    /// End-to-end UPDATE: setting nullable int, string, float, and bool columns
    /// to bare SQL `NULL` succeeds and leaves the rows with NULL values.
    #[tokio::test]
    async fn end_to_end_sql_update_nullable_columns_to_null() {
        let (ctx, storage, root, _tmp) = registered_table_nullable_types("t_null_upd", 10).await;

        assert_eq!(
            execute_sql(
                &ctx,
                Arc::clone(&storage),
                &root,
                "UPDATE t_null_upd SET score = NULL WHERE id = 1",
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx,
                Arc::clone(&storage),
                &root,
                "UPDATE t_null_upd SET name = NULL WHERE id = 2",
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx,
                Arc::clone(&storage),
                &root,
                "UPDATE t_null_upd SET weight = NULL WHERE id = 3",
            )
            .await
            .unwrap(),
            1
        );
        assert_eq!(
            execute_sql(
                &ctx,
                Arc::clone(&storage),
                &root,
                "UPDATE t_null_upd SET active = NULL WHERE id = 4",
            )
            .await
            .unwrap(),
            1
        );

        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_null_upd WHERE id = 1 AND score IS NULL"
            )
            .await,
            1,
            "nullable Int64 column must be set to NULL"
        );
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_null_upd WHERE id = 2 AND name IS NULL"
            )
            .await,
            1,
            "nullable Utf8 column must be set to NULL"
        );
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_null_upd WHERE id = 3 AND weight IS NULL"
            )
            .await,
            1,
            "nullable Float64 column must be set to NULL"
        );
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_null_upd WHERE id = 4 AND active IS NULL"
            )
            .await,
            1,
            "nullable Boolean column must be set to NULL"
        );
    }

    /// Regression (M02): assigning a non-null literal to a nullable column must
    /// succeed even when DataFusion projects the assignment as non-nullable.
    ///
    /// The writer's defensive schema check for `commit_update` used to compare
    /// nullability as well as names and types, which rejected this valid update.
    #[tokio::test]
    async fn end_to_end_sql_update_nullable_column_to_non_null_literal() {
        let (ctx, storage, root, _tmp) =
            registered_table_nullable_types("t_null_literal_upd", 10).await;

        assert_eq!(
            execute_sql(
                &ctx,
                Arc::clone(&storage),
                &root,
                "UPDATE t_null_literal_upd SET score = 123 WHERE id = 1",
            )
            .await
            .unwrap(),
            1,
            "UPDATE must report 1 row affected"
        );

        assert_eq!(
            scalar_i64(&ctx, "SELECT score FROM t_null_literal_upd WHERE id = 1").await,
            123,
            "nullable Int64 column must hold the assigned non-null literal"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_null_literal_upd").await,
            10,
            "UPDATE must not change the row count"
        );
    }

    /// End-to-end UPDATE: assigning `NULL` to a non-nullable column is rejected
    /// before any rows are written, leaving the table unchanged.
    #[tokio::test]
    async fn end_to_end_sql_update_non_nullable_to_null_fails() {
        let (ctx, storage, root, _tmp) = registered_table_two_cols("t_non_null_upd", 10).await;

        let err = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE t_non_null_upd SET v = NULL WHERE id = 1",
        )
        .await
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("non-nullable"),
            "expected non-nullable column error, got: {msg}"
        );

        // Table must be unchanged: v for id=1 is still 10 and all rows remain.
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_non_null_upd WHERE id = 1").await,
            10,
            "rejected UPDATE must not modify the row"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_non_null_upd").await,
            10,
            "rejected UPDATE must not change the row count"
        );
    }

    /// Regression (M02 follow-up): a SET expression that is *not* a bare NULL
    /// literal but evaluates to NULL on the matched rows must also be rejected
    /// for a non-nullable target. `coerce_update_null_literals` only rewrites
    /// bare `Expr::Literal(Null)`; a CASE/nullable-column expression is typed by
    /// DataFusion as the concrete type with a validity bitmap and would slip
    /// through, silently corrupting the schema invariant. The post-image
    /// nullability check (`enforce_non_nullable_post_image`) must catch it.
    #[tokio::test]
    async fn update_non_nullable_via_null_expression_fails() {
        let (ctx, storage, root, _tmp) = registered_table_two_cols("t_non_null_expr", 10).await;

        // CASE WHEN id > 3 THEN NULL ELSE v END produces a typed Int64 array
        // with nulls on rows 4..10; v is non-nullable.
        let err = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE t_non_null_expr SET v = CASE WHEN id > 3 THEN NULL ELSE v END",
        )
        .await
        .unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("non-nullable"),
            "expected non-nullable column error from null-yielding expression, got: {msg}"
        );

        // Table must be unchanged.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_non_null_expr").await,
            10,
            "rejected UPDATE must not change the row count"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_non_null_expr WHERE id = 5").await,
            50,
            "rejected UPDATE must leave rows untouched"
        );
    }

    /// Regression (M04 follow-up): an unquoted mixed-case ON key column must
    /// resolve to the declared lowercase column. `extract_merge_key` used to
    /// return the raw identifier; quoting it then failed the join against the
    /// real `id` column.
    #[tokio::test]
    async fn merge_unquoted_uppercase_on_key_resolves() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_case", 5).await;

        // ON t.Id = src.Id — unquoted, must normalize to `id`.
        let n = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "MERGE INTO t_merge_case USING (VALUES (1, 999)) AS src(id, v) \
             ON t_merge_case.Id = src.Id \
             WHEN MATCHED THEN UPDATE SET v = src.v \
             WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)",
        )
        .await
        .unwrap();
        assert_eq!(n, 1, "MERGE should match the one row with id=1");

        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_case WHERE id = 1").await,
            999,
            "MERGE with unquoted mixed-case ON key must update the matched row"
        );
    }

    /// End-to-end UPDATE with a **quoted** target name on a table whose
    /// **predicate** column is indexed (so the index fast path fires).
    ///
    /// Regression: `execute_update_to_delta` extracted the target name with
    /// `name.to_string()`, which re-emits a quoted identifier verbatim
    /// (`"t"` keeps its quotes). The fast path then double-escaped that into
    /// `"""t"""` and `ctx.sql` could not resolve the table. The bare ident
    /// value must be used instead (matching the MERGE path), so a quoted
    /// UPDATE through the index fast path resolves the table and updates the row.
    #[tokio::test]
    async fn end_to_end_sql_update_quoted_name_index_fast_path() {
        use icefalldb_core::database_catalog::{CatalogLockGuard, DatabaseCatalog};
        use icefalldb_core::index::{build_btree_index, IndexDefinition};
        use icefalldb_core::metadata::Manifest;

        let (_ctx0, storage, root, _tmp) = registered_table_two_cols("t_quoted_upd", 20).await;

        // Index the PREDICATE column `id` so the fast path fires for `id = 5`.
        let catalog = DatabaseCatalog::new(Arc::clone(&storage));
        let lock: CatalogLockGuard = catalog
            .acquire_lock(std::time::Duration::from_secs(5))
            .await
            .unwrap();
        catalog
            .create_index_definition(&lock, "idx_id", "t_quoted_upd", "id", "btree")
            .await
            .unwrap();
        drop(lock);

        let pointer_bytes = storage.read("t_quoted_upd/_manifest.json").await.unwrap();
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_bytes).unwrap();
        let seq = pointer["latest"].as_u64().unwrap();
        let manifest_path = format!("t_quoted_upd/{}", Manifest::filename(seq));
        let manifest_bytes = storage.read(&manifest_path).await.unwrap();
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap();
        let definition = IndexDefinition {
            name: "idx_id".to_string(),
            table: "t_quoted_upd".to_string(),
            column: "id".to_string(),
            unique: false,
        };
        build_btree_index(storage.as_ref(), &definition, &manifest)
            .await
            .unwrap()
            .save(storage.as_ref())
            .await
            .unwrap();

        // Fresh provider + session so the snapshot loads the just-saved index
        // and `try_locate_by_index` answers `id = 5` (fast path actually fires).
        let provider = Arc::new(
            IcefallDBTableProvider::new(
                Arc::clone(&storage),
                "t_quoted_upd",
                mutate_provider_config(),
            )
            .await
            .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t_quoted_upd", provider).unwrap();

        // Quoted target name — the failing case before the fix.
        let n = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE \"t_quoted_upd\" SET v = v + 1 WHERE id = 5",
        )
        .await
        .unwrap();
        assert_eq!(n, 1, "quoted-name UPDATE must report 1 row affected");
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_quoted_upd WHERE id = 5").await,
            51,
            "post-update v for id=5 must be 51"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_quoted_upd").await,
            20,
            "total row count must remain 20"
        );
    }

    // ── dedup_source ───────────────────────────────────────────────────

    use arrow::array::{Int64Array as TestInt64Array, RecordBatch as TestRecordBatch};
    use arrow::datatypes::{DataType as TestDataType, Field as TestField, Schema as TestSchema};
    use datafusion::common::ScalarValue as TestScalarValue;
    use std::sync::Arc as TestArc;

    /// Build a simple `RecordBatch` from column name/value pairs (all Int64).
    ///
    /// `cols` is a slice of `(name, values)` where each values slice must have
    /// the same length.
    fn make_batch(cols: &[(&str, &[i64])]) -> TestRecordBatch {
        assert!(!cols.is_empty(), "need at least one column");
        let n = cols[0].1.len();
        for (name, vals) in cols.iter() {
            assert_eq!(vals.len(), n, "column '{}' has wrong length", name);
        }
        let fields: Vec<_> = cols
            .iter()
            .map(|(name, _)| TestField::new(*name, TestDataType::Int64, true))
            .collect();
        let schema = TestArc::new(TestSchema::new(fields));
        let arrays: Vec<_> = cols
            .iter()
            .map(|(_, vals)| {
                TestArc::new(TestInt64Array::from(vals.to_vec()))
                    as TestArc<dyn arrow::array::Array>
            })
            .collect();
        TestRecordBatch::try_new(schema, arrays).unwrap()
    }

    /// Duplicate keys with `DupPolicy::Error` must return `Err`; with
    /// `DupPolicy::LastWriterWins` must return a single entry holding the last row.
    #[test]
    fn duplicate_matched_keys_are_a_cardinality_violation() {
        let src = make_batch(&[("id", &[7, 7]), ("v", &[1, 2])]);

        // Error policy: duplicate key must be rejected.
        assert!(
            dedup_source(&src, "id", DupPolicy::Error).is_err(),
            "duplicate key must be an error under DupPolicy::Error"
        );

        // LastWriterWins: only one entry, value is from the last row (v == 2).
        let lww = dedup_source(&src, "id", DupPolicy::LastWriterWins).unwrap();
        assert_eq!(lww.len(), 1, "deduped map must have exactly one entry");
        let row = lww
            .get(&TestScalarValue::from(7i64))
            .expect("key 7 must be present");
        assert_eq!(
            row.get_i64("v"),
            2,
            "LastWriterWins must keep the last row (v=2)"
        );
    }

    /// Distinct keys all survive and the insertion order is preserved.
    #[test]
    fn distinct_keys_all_survive_in_insertion_order() {
        // Three rows with distinct keys: 10, 20, 30 in that order.
        let src = make_batch(&[("id", &[10, 20, 30]), ("v", &[100, 200, 300])]);

        for policy in [DupPolicy::Error, DupPolicy::LastWriterWins] {
            let ds = dedup_source(&src, "id", policy).unwrap();
            assert_eq!(ds.len(), 3, "all 3 distinct keys must survive");

            // Insertion order must be preserved (10 → 20 → 30).
            let keys: Vec<_> = ds.by_key.iter().map(|(k, _)| k.clone()).collect();
            assert_eq!(
                keys,
                vec![
                    TestScalarValue::from(10i64),
                    TestScalarValue::from(20i64),
                    TestScalarValue::from(30i64),
                ],
                "insertion order must be 10, 20, 30"
            );

            // Values are correct.
            assert_eq!(
                ds.get(&TestScalarValue::from(10i64)).unwrap().get_i64("v"),
                100
            );
            assert_eq!(
                ds.get(&TestScalarValue::from(20i64)).unwrap().get_i64("v"),
                200
            );
            assert_eq!(
                ds.get(&TestScalarValue::from(30i64)).unwrap().get_i64("v"),
                300
            );
        }
    }

    /// A null value in the key column is rejected regardless of `DupPolicy`.
    #[test]
    fn null_key_is_always_rejected() {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        let schema = TestArc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("v", DataType::Int64, true),
        ]));
        // Row 0: id=NULL, v=1.
        let id_arr = TestArc::new(Int64Array::from(vec![None, Some(42)]));
        let v_arr = TestArc::new(Int64Array::from(vec![Some(1), Some(2)]));
        let src = TestRecordBatch::try_new(schema, vec![id_arr, v_arr]).unwrap();

        assert!(
            dedup_source(&src, "id", DupPolicy::Error).is_err(),
            "null key must be rejected under DupPolicy::Error"
        );
        assert!(
            dedup_source(&src, "id", DupPolicy::LastWriterWins).is_err(),
            "null key must be rejected under DupPolicy::LastWriterWins"
        );
    }

    // ── require_unique_key_index ───────────────────────────────────────

    /// `require_unique_key_index` returns an `Err` whose message names the
    /// missing unique index when the table has no unique index on `id`.
    ///
    /// Two sub-cases are covered:
    ///   (a) No index at all on `id`.
    ///   (b) A non-unique index on `id` exists (unique == false).
    ///
    /// In both cases the error must contain "requires a UNIQUE index on id".
    #[tokio::test]
    async fn require_unique_key_index_errors_without_unique_index() {
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        // ── (a) No index at all ──────────────────────────────────────────────
        {
            let (_ctx, storage, _root, _tmp) = registered_table("t_no_idx", 5).await;

            let err = require_unique_key_index(Arc::clone(&storage), "t_no_idx", "id")
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("requires a UNIQUE index on id"),
                "error must mention missing unique index on id; got: {err}"
            );
        }

        // ── (b) Non-unique index exists, still no unique index ───────────────
        {
            let (_ctx, storage, _root, _tmp) = registered_table("t_nonuniq_idx", 5).await;

            // Register a non-unique index on `id`.
            let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
            let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
            dbcat
                .create_index_definition(&guard, "id_nonuniq", "t_nonuniq_idx", "id", "btree")
                .await
                .unwrap();
            drop(guard);

            // Still no unique index → must error.
            let err = require_unique_key_index(Arc::clone(&storage), "t_nonuniq_idx", "id")
                .await
                .unwrap_err();
            assert!(
                err.to_string().contains("requires a UNIQUE index on id"),
                "non-unique index must not satisfy the contract; got: {err}"
            );
        }
    }

    /// `require_unique_key_index` returns `Ok(index_name)` when a unique btree
    /// index on `id` exists in the catalog.
    #[tokio::test]
    async fn require_unique_key_index_finds_unique_index() {
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        let (_ctx, storage, _root, _tmp) = registered_table("t_uniq_idx", 5).await;

        // Register a UNIQUE index on `id`.
        let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard,
                "id_uniq",
                "t_uniq_idx",
                "id",
                "btree",
                true, // unique
            )
            .await
            .unwrap();
        drop(guard);

        let name = require_unique_key_index(Arc::clone(&storage), "t_uniq_idx", "id")
            .await
            .expect("unique index must be found");
        assert_eq!(
            name, "id_uniq",
            "returned name must match the registered index name"
        );
    }

    // ── execute_merge — matched/unmatched routing ──────────────────────

    /// Build a table with `rows` rows and columns `id` (Int64) and `v` (Int64,
    /// value = id * 10), plus a UNIQUE btree index on `id`.
    ///
    /// Returns `(SessionContext, Arc<dyn Storage>, table_root, TempDir)`.
    async fn registered_table_unique(
        table_name: &str,
        rows: usize,
    ) -> (SessionContext, Arc<dyn Storage>, String, tempfile::TempDir) {
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = rows.max(1);

        // Register a UNIQUE btree index on `id` before writing rows so the first
        // commit builds the index.
        let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard, "id_uniq", table_name, "id", "btree", true, // unique
            )
            .await
            .unwrap();
        drop(guard);

        let ids: Vec<i64> = (0..rows as i64).collect();
        let vs: Vec<i64> = ids.iter().map(|i| i * 10).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(vs)),
            ],
        )
        .unwrap();

        let mut writer = Writer::create(Arc::clone(&storage), table_name, mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), table_name, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(table_name, provider).unwrap();

        let root = table_name.to_string();
        (ctx, storage, root, tmp)
    }

    /// Build a table with `rows` rows and columns `id` (Int64), `v` (Int64,
    /// value = id * 10), and `w` (Int64, value = id * 100), plus a UNIQUE btree
    /// index on `id`.
    ///
    /// Returns `(SessionContext, Arc<dyn Storage>, table_root_string, TempDir)`.
    async fn registered_table_three_cols(
        table_name: &str,
        rows: usize,
    ) -> (SessionContext, Arc<dyn Storage>, String, tempfile::TempDir) {
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
            Field::new("w", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = rows.max(1);

        // Register a UNIQUE btree index on `id` before writing rows so the first
        // commit builds the index.
        let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard, "id_uniq", table_name, "id", "btree", true, // unique
            )
            .await
            .unwrap();
        drop(guard);

        let ids: Vec<i64> = (0..rows as i64).collect();
        let vs: Vec<i64> = ids.iter().map(|i| i * 10).collect();
        let ws: Vec<i64> = ids.iter().map(|i| i * 100).collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(vs)),
                Arc::new(Int64Array::from(ws)),
            ],
        )
        .unwrap();

        let mut writer = Writer::create(Arc::clone(&storage), table_name, mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), table_name, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(table_name, provider).unwrap();

        let root = table_name.to_string();
        (ctx, storage, root, tmp)
    }

    /// Build a two-column source `RecordBatch` with columns `id` and `v`.
    fn make_merge_src(id: i64, v: i64) -> RecordBatch {
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(vec![id])),
                Arc::new(Int64Array::from(vec![v])),
            ],
        )
        .unwrap()
    }

    /// Guard that MERGE classification actually uses the index fast path: the
    /// `key_col IN (..)` probe `execute_merge_to_delta` builds must resolve from
    /// the unique index in the provider snapshot, returning only LIVE rows.
    /// Without this guard the merge tests would still pass via the scan fallback
    /// while silently losing the per-MERGE locate speedup.
    #[tokio::test]
    async fn merge_key_probe_resolves_via_index_fast_path() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_probe_idx", 20).await;

        // Tombstone id=7 so the index path's liveness filtering is exercised.
        execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "DELETE FROM t_merge_probe_idx WHERE id = 7",
        )
        .await
        .unwrap();

        let provider = ctx.table_provider("t_merge_probe_idx").await.unwrap();
        let prov = (provider.as_ref() as &dyn std::any::Any)
            .downcast_ref::<IcefallDBTableProvider>()
            .expect("registered provider must be a IcefallDBTableProvider");

        // Same predicate shape the MERGE probe builds: `id IN (live, dead, absent)`.
        let probe = col("id").in_list(vec![lit(3i64), lit(7i64), lit(99i64)], false);
        let locs = prov
            .try_locate_by_index(&probe)
            .await
            .unwrap()
            .expect("MERGE key probe must resolve via the index, not the scan fallback");

        // Only id=3 is live: id=7 is tombstoned, id=99 was never inserted.
        assert_eq!(locs.len(), 1, "only the live key id=3 must locate");
        assert_eq!(locs[0].row_id, 3, "located row_id for id=3 must be 3");
    }

    /// ACCEPTANCE TEST: upsert over a previously-deleted key inserts
    /// a fresh row_id and leaves exactly ONE live row for that key.
    ///
    /// 1. Table `t` has rows id 0..10, each with a UNIQUE index on `id`.
    /// 2. `DELETE FROM t WHERE id = 4` → key 4 is dead (tombstoned in the index).
    /// 3. `execute_merge` with source `(id=4, v=999)`:
    ///    - The unique index still has `4 → row_id_4` but that row_id is dead
    ///      (the address map → DV check returns it as NOT live).
    ///    - Liveness-filtered probe: MISS → `not_matched=Insert`.
    ///    - `MergeStats { updated: 0, inserted: 1 }`.
    ///    - The new commit rebuilds the index: `4 → new_row_id` (the stale dead
    ///      entry is gone — only live rows are indexed).
    /// 4. `SELECT v FROM t WHERE id = 4` = 999; COUNT = 1.
    #[tokio::test]
    async fn upsert_over_deleted_key_inserts_fresh_id_single_live_entry() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_upsert_del", 10).await;

        // Delete key 4.
        execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "DELETE FROM t_merge_upsert_del WHERE id = 4",
        )
        .await
        .unwrap();

        // MERGE: dead key 4 is a MISS → Insert.
        let src = make_merge_src(4, 999);
        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_upsert_del",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!(
            (stats.updated, stats.inserted),
            (0, 1),
            "dead key must be treated as NOT MATCHED → inserted:1, updated:0"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_upsert_del WHERE id = 4").await,
            999,
            "inserted row must be visible with the new value"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_upsert_del WHERE id = 4").await,
            1,
            "exactly one live row for id=4 after upsert (no duplicate)"
        );
    }

    /// MERGE into a live key → MATCHED → `commit_update` → updated:1, inserted:0.
    ///
    /// Key 7 is live; source has (id=7, v=777).  After the merge:
    ///   - `MergeStats { updated: 1, inserted: 0 }`.
    ///   - `SELECT v FROM t WHERE id = 7` returns 777.
    ///   - COUNT(*) WHERE id = 7 is still 1 (no duplicate).
    #[tokio::test]
    async fn merge_live_key_routes_to_update() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_live_upd", 10).await;

        let src = make_merge_src(7, 777);
        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_live_upd",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!(
            (stats.updated, stats.inserted),
            (1, 0),
            "live key must route to UPDATE → updated:1, inserted:0"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_live_upd WHERE id = 7").await,
            777,
            "updated row must show the new value 777"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_live_upd WHERE id = 7").await,
            1,
            "no duplicate row after update"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_live_upd").await,
            10,
            "total row count must remain 10"
        );
    }

    /// MERGE with a brand-new key → NOT MATCHED → `insert_batch` + `commit`.
    ///
    /// Table has ids 0..10 (no id=99).  Source has (id=99, v=9900).
    ///   - `MergeStats { updated: 0, inserted: 1 }`.
    ///   - `SELECT v FROM t WHERE id = 99` returns 9900.
    ///   - `SELECT COUNT(*) FROM t` = 11.
    #[tokio::test]
    async fn merge_new_key_routes_to_insert() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_new_key", 10).await;

        let src = make_merge_src(99, 9900);
        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_new_key",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!(
            (stats.updated, stats.inserted),
            (0, 1),
            "new key must route to INSERT → inserted:1, updated:0"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_new_key WHERE id = 99").await,
            9900,
            "inserted row must be visible with value 9900"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_new_key").await,
            11,
            "total row count must increase to 11 after insert"
        );
    }

    /// MERGE with duplicate keys in the source relation must be rejected.
    ///
    /// SQL-standard MERGE is undefined when the source contains duplicate match
    /// keys; `DupPolicy::Error` catches this before any target mutation.
    #[tokio::test]
    async fn merge_rejects_duplicate_source_keys() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_dup_src", 10).await;

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let src = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![7i64, 7])),
                Arc::new(Int64Array::from(vec![700i64, 707])),
            ],
        )
        .unwrap();

        let err = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_dup_src",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap_err();

        let msg = format!("{err}");
        assert!(
            msg.contains("duplicate key") || msg.contains("duplicate"),
            "expected duplicate-source-key error, got: {msg}"
        );

        // No rows must have been changed.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_dup_src").await,
            10,
            "duplicate-source MERGE must not modify the table"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_dup_src WHERE id = 7").await,
            70,
            "row id=7 must be unchanged"
        );
    }

    /// A duplicate LIVE target row for a single merge key must surface as the
    /// structured `IcefallDBError::UniqueKeyViolation` variant, not a plain
    /// `QueryError::Other` string.
    #[tokio::test]
    async fn merge_duplicate_live_target_returns_structured_unique_key_violation() {
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 5;

        // Write duplicate live rows for id=7 without a unique index.
        let mut writer = Writer::create(Arc::clone(&storage), "t", mdb_schema)
            .await
            .unwrap();
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![7i64, 7])),
                Arc::new(Int64Array::from(vec![70i64, 700])),
            ],
        )
        .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        // Register a UNIQUE index definition (but do not build the index file),
        // so MERGE sees the unique-index contract while the data still contains
        // duplicate live target rows.
        let catalog = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = catalog.acquire_lock(Duration::from_secs(5)).await.unwrap();
        catalog
            .create_index_definition_with_options(
                &guard, "id_uniq", "t", "id", "btree", true, // unique
            )
            .await
            .unwrap();
        drop(guard);

        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t", mutate_provider_config())
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t", provider).unwrap();

        let src = make_merge_src(7, 999);
        let err = execute_merge(
            &ctx,
            Arc::clone(&storage),
            "t",
            "t",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap_err();

        match err {
            QueryError::Core(IcefallDBError::UniqueKeyViolation { table, index, key }) => {
                assert_eq!(table, "t", "violation must name the target table");
                assert_eq!(index, "id_uniq", "violation must name the unique index");
                assert_eq!(key, "7", "violation must include the duplicate key value");
            }
            other => {
                panic!("expected UniqueKeyViolation variant, got {other:?} (message: {other})")
            }
        }

        // The table must be left untouched.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t").await,
            2,
            "duplicate-live-target MERGE must not modify the table"
        );
    }

    // ── MERGE INTO … SQL surface ──────────────────────────────────────

    /// Build a MERGE SQL statement for a CDC batch against a table with columns
    /// `id` (Int64) and `v` (Int64), keyed on `id`.
    ///
    /// `updates` is a slice of `(id, new_v)` for MATCHED rows.
    /// `inserts` is a slice of `(id, v)` for NOT MATCHED rows.
    ///
    /// Produces:
    /// ```sql
    /// MERGE INTO <tbl> USING (VALUES (id0, v0), ...) AS src(id, v)
    /// ON <tbl>.id = src.id
    /// WHEN MATCHED THEN UPDATE SET v = src.v
    /// WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)
    /// ```
    fn build_cdc_merge_sql(tbl: &str, updates: &[(i64, i64)], inserts: &[(i64, i64)]) -> String {
        let mut all_rows: Vec<String> = Vec::new();
        for (id, v) in updates.iter().chain(inserts.iter()) {
            all_rows.push(format!("({id}, {v})"));
        }
        let values_clause = all_rows.join(", ");
        format!(
            "MERGE INTO {tbl} USING (VALUES {values_clause}) AS src(id, v) \
             ON {tbl}.id = src.id \
             WHEN MATCHED THEN UPDATE SET v = src.v \
             WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)"
        )
    }

    /// CDC capstone test: MERGE INTO driven through the full `execute_sql` router.
    ///
    /// Table has 100 initial rows (ids 0..100, v = id * 10).
    /// CDC batch: 8 updates (ids 0..8, new v = id * 10 + 1000) +
    ///            2 inserts (ids 100..102, v = id * 10 + 1000).
    ///
    /// Note: 100 rows / 8+2 batch preserves the 80:20 ratio from the canonical
    /// 1000-row spec while keeping the VALUES clause manageable in a unit test.
    ///
    /// This test routes through `execute_sql` (which returns `updated + inserted`)
    /// and proves correctness through observable end-state:
    ///   1. `affected` == 10 (8 updates + 2 inserts summed by execute_sql).
    ///   2. `SELECT COUNT(*) FROM t` == 102 (100 original + 2 inserts, no
    ///      spurious duplicates from the 8 updates — move-stable guarantee).
    ///   3. Spot-check an UPDATED id: `SELECT v FROM t WHERE id = 7` == 1070.
    ///   4. Spot-check an INSERTED id: `SELECT COUNT(*) FROM t WHERE id = 101` == 1.
    #[tokio::test]
    async fn merge_cdc_batch_yields_expected_state() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_cdc_merge", 100).await;

        // CDC batch: 8 updates (ids 0..8) + 2 inserts (ids 100..102).
        let updates: Vec<(i64, i64)> = (0..8i64).map(|id| (id, id * 10 + 1000)).collect();
        let inserts: Vec<(i64, i64)> = (100..102i64).map(|id| (id, id * 10 + 1000)).collect();

        let merge_sql = build_cdc_merge_sql("t_cdc_merge", &updates, &inserts);

        // Drive the MERGE through the full execute_sql router (the same path a
        // real caller would use), which returns updated + inserted as a single
        // affected-row count.
        let affected = execute_sql(&ctx, Arc::clone(&storage), &root, &merge_sql)
            .await
            .expect("execute_sql MERGE must succeed");

        // 1. Summed affected count: 8 updates + 2 inserts = 10.
        assert_eq!(
            affected, 10,
            "execute_sql must return updated+inserted = 10 (8 updates + 2 inserts)"
        );

        // 2. End-state row count: 100 original + 2 new inserts; 8 updates
        //    must NOT create duplicates (move-stable + single-live-entry guarantee).
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_cdc_merge").await,
            102,
            "total row count must be 102 after CDC merge (100 original + 2 inserted, no duplicates)"
        );

        // 3. Spot-check an UPDATED row: id=7, v was 70, must now be 7*10+1000=1070.
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_cdc_merge WHERE id = 7").await,
            1070,
            "updated id=7 must have new value v=1070"
        );

        // 4. Spot-check an INSERTED row: id=101 was absent, must now be present exactly once.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_cdc_merge WHERE id = 101").await,
            1,
            "inserted id=101 must appear exactly once"
        );
    }

    /// MERGE INTO via execute_sql (the full router path) also works end-to-end.
    ///
    /// Uses a small 5-row table with 2 updates + 1 insert to verify the
    /// `execute_sql` → `Statement::Merge` → `execute_merge_sql` path.
    #[tokio::test]
    async fn end_to_end_sql_merge_via_execute_sql() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_e2e_merge", 5).await;
        // ids 0..5, v = id * 10: (0,0), (1,10), (2,20), (3,30), (4,40)

        // Update ids 0 and 2; insert id 10.
        let merge_sql =
            "MERGE INTO t_e2e_merge USING (VALUES (0, 999), (2, 888), (10, 777)) AS src(id, v) \
             ON t_e2e_merge.id = src.id \
             WHEN MATCHED THEN UPDATE SET v = src.v \
             WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)";

        let affected = execute_sql(&ctx, Arc::clone(&storage), &root, merge_sql)
            .await
            .unwrap();
        assert_eq!(affected, 3, "2 updates + 1 insert = 3 affected rows");

        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_e2e_merge WHERE id = 0").await,
            999,
            "id=0 must be updated to 999"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_e2e_merge WHERE id = 2").await,
            888,
            "id=2 must be updated to 888"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_e2e_merge WHERE id = 10").await,
            777,
            "id=10 must be inserted with v=777"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_e2e_merge").await,
            6,
            "total row count must be 6 after merge (5 original + 1 inserted)"
        );
    }

    /// Regression test for the MERGE side of M02: `WHEN MATCHED THEN UPDATE SET
    /// <nullable> = NULL` must clear the column (just like UPDATE), and a NULL
    /// assigned to a non-nullable column must be rejected before any write.
    #[tokio::test]
    async fn merge_update_set_null() {
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("score", DataType::Int64, true), // nullable
            Field::new("tag", DataType::Int64, false),  // non-nullable
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 16;

        let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard,
                "id_uniq",
                "t_merge_null",
                "id",
                "btree",
                true,
            )
            .await
            .unwrap();
        drop(guard);

        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![0i64, 1, 2])),
                Arc::new(Int64Array::from(vec![100i64, 200, 300])),
                Arc::new(Int64Array::from(vec![7i64, 8, 9])),
            ],
        )
        .unwrap();
        let mut writer = Writer::create(Arc::clone(&storage), "t_merge_null", mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t_merge_null", config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t_merge_null", provider).unwrap();
        let root = "t_merge_null".to_string();

        // Success: clear `score` on matched row id=1 (NOT MATCHED never fires).
        let merge_ok = "MERGE INTO t_merge_null USING (VALUES (1, 999, 9)) AS src(id, score, tag) \
             ON t_merge_null.id = src.id \
             WHEN MATCHED THEN UPDATE SET score = NULL \
             WHEN NOT MATCHED THEN INSERT (id, score, tag) VALUES (src.id, src.score, src.tag)";
        execute_sql(&ctx, Arc::clone(&storage), &root, merge_ok)
            .await
            .unwrap();
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_merge_null WHERE id = 1 AND score IS NULL"
            )
            .await,
            1,
            "MERGE SET score = NULL must clear the column"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT score FROM t_merge_null WHERE id = 0").await,
            100,
            "non-matched rows must be untouched"
        );

        // Rejection: NULL into non-nullable `tag`, table unchanged.
        let merge_bad =
            "MERGE INTO t_merge_null USING (VALUES (2, 888, 8)) AS src(id, score, tag) \
             ON t_merge_null.id = src.id \
             WHEN MATCHED THEN UPDATE SET tag = NULL \
             WHEN NOT MATCHED THEN INSERT (id, score, tag) VALUES (src.id, src.score, src.tag)";
        let err = execute_sql(&ctx, Arc::clone(&storage), &root, merge_bad)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("non-nullable column 'tag'"),
            "expected non-nullable rejection, got: {err}"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT tag FROM t_merge_null WHERE id = 2").await,
            9,
            "rejected MERGE must leave the table unchanged"
        );
    }

    // ── M03/M04 data-integrity fixes ────────────────────────────────────

    /// UPDATE with an unknown SET column must fail and leave the table
    /// byte/logically unchanged.
    #[tokio::test]
    async fn update_unknown_set_column_fails_unchanged() {
        let (ctx, storage, root, _tmp) = registered_table_two_cols("t_upd_bad_col", 10).await;

        let count_before = scalar_i64(&ctx, "SELECT COUNT(*) FROM t_upd_bad_col").await;
        let v_before = scalar_i64(&ctx, "SELECT v FROM t_upd_bad_col WHERE id = 5").await;
        assert_eq!(v_before, 50, "pre-update v for id=5 must be 50");

        let err = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE t_upd_bad_col SET no_such_column = 999 WHERE id = 5",
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("UPDATE SET target column 'no_such_column' does not exist"),
            "expected schema error, got: {msg}"
        );

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_upd_bad_col").await,
            count_before,
            "table row count must be unchanged after failed UPDATE"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_upd_bad_col WHERE id = 5").await,
            v_before,
            "target row must be unchanged after failed UPDATE"
        );
    }

    /// MERGE with an unknown UPDATE SET target must fail and leave the table
    /// unchanged.
    #[tokio::test]
    async fn merge_unknown_update_target_fails_unchanged() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_bad_col", 10).await;

        let count_before = scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_bad_col").await;

        let merge_sql = "MERGE INTO t_merge_bad_col USING (VALUES (1, 99)) AS src(id, v) \
             ON t_merge_bad_col.id = src.id \
             WHEN MATCHED THEN UPDATE SET nonexistent = src.v \
             WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)";

        let err = execute_sql(&ctx, Arc::clone(&storage), &root, merge_sql)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MERGE UPDATE SET target column 'nonexistent' does not exist"),
            "expected schema error, got: {msg}"
        );

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_bad_col").await,
            count_before,
            "table row count must be unchanged after failed MERGE"
        );
    }

    /// UPDATE with an unquoted uppercase SET target must resolve to the lowercase
    /// column name, matching DataFusion's identifier normalization.
    #[tokio::test]
    async fn update_unquoted_uppercase_set_target_resolves() {
        let (ctx, storage, root, _tmp) = registered_table_two_cols("t_upd_upper", 10).await;

        let affected = execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "UPDATE t_upd_upper SET V = 999 WHERE id = 5",
        )
        .await
        .expect("UPDATE with unquoted uppercase target must succeed");
        assert_eq!(affected, 1, "one row should be updated");
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_upd_upper WHERE id = 5").await,
            999,
            "unquoted uppercase SET target must update the lowercase column"
        );
    }

    /// MERGE with an unquoted uppercase UPDATE SET target must resolve to the
    /// lowercase column name.
    #[tokio::test]
    async fn merge_unquoted_uppercase_update_target_resolves() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_upper", 5).await;

        let merge_sql = "MERGE INTO t_merge_upper USING (VALUES (0, 999)) AS src(id, v) \
             ON t_merge_upper.id = src.id \
             WHEN MATCHED THEN UPDATE SET V = src.v \
             WHEN NOT MATCHED THEN INSERT (id, v) VALUES (src.id, src.v)";

        let affected = execute_sql(&ctx, Arc::clone(&storage), &root, merge_sql)
            .await
            .expect("MERGE with unquoted uppercase target must succeed");
        assert_eq!(affected, 1, "one matched row should be updated");
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_upper WHERE id = 0").await,
            999,
            "unquoted uppercase MERGE UPDATE target must update the lowercase column"
        );
    }

    /// MERGE UPDATE SET temp-table registration must be cleaned up if the join
    /// query fails, leaving the session in a state where a retry can re-register
    /// the same internal name.
    #[tokio::test]
    async fn merge_temp_table_cleaned_up_on_assignment_error() {
        let (ctx, storage, root, _tmp) = registered_table_three_cols("t_merge_cleanup", 5).await;

        // Reference an unknown source column in the UPDATE SET RHS. This causes
        // ctx.sql to fail after the temp table has been registered.
        let merge_sql = "MERGE INTO t_merge_cleanup USING (VALUES (0, 1)) AS src(id, v) \
             ON t_merge_cleanup.id = src.id \
             WHEN MATCHED THEN UPDATE SET v = src.no_such_column \
             WHEN NOT MATCHED THEN INSERT (id, v, w) VALUES (src.id, src.v, 0)";

        let err = execute_sql(&ctx, Arc::clone(&storage), &root, merge_sql)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("no_such_column") || msg.contains("column") || msg.contains("field"),
            "expected column-not-found error, got: {msg}"
        );

        // A second attempt with the same internal temp-table name must be able
        // to register it again, proving the first registration was cleaned up.
        let merge_sql2 = "MERGE INTO t_merge_cleanup USING (VALUES (0, 999)) AS src(id, v) \
             ON t_merge_cleanup.id = src.id \
             WHEN MATCHED THEN UPDATE SET v = src.v \
             WHEN NOT MATCHED THEN INSERT (id, v, w) VALUES (src.id, src.v, 0)";

        let affected = execute_sql(&ctx, Arc::clone(&storage), &root, merge_sql2)
            .await
            .expect("retry after failed MERGE must succeed");
        assert_eq!(affected, 1, "one matched row should be updated on retry");
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_cleanup WHERE id = 0").await,
            999,
            "retry must apply the update"
        );
    }

    /// MERGE UPDATE SET targets are bound by name, not position: a reordered
    /// SET clause updates only the named columns and leaves unassigned columns
    /// unchanged.
    #[tokio::test]
    async fn merge_reordered_set_updates_named_columns_only() {
        let (ctx, storage, root, _tmp) = registered_table_three_cols("t_merge_reorder", 10).await;

        // Row id=5 has v=50, w=500. Source row id=5, v=999.
        let merge_sql = "MERGE INTO t_merge_reorder USING (VALUES (5, 999)) AS src(id, v) \
             ON t_merge_reorder.id = src.id \
             WHEN MATCHED THEN UPDATE SET id = src.v, v = src.id \
             WHEN NOT MATCHED THEN INSERT (id, v, w) VALUES (src.id, src.v, 0)";

        let affected = execute_sql(&ctx, Arc::clone(&storage), &root, merge_sql)
            .await
            .unwrap();
        assert_eq!(affected, 1, "one matched row should be updated");

        // The key was updated from 5 to 999, v was set to the original id (5),
        // and w was preserved from the target pre-image (500).
        assert_eq!(
            scalar_i64(&ctx, "SELECT id FROM t_merge_reorder WHERE w = 500").await,
            999,
            "id must be updated to src.v (999)"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_reorder WHERE w = 500").await,
            5,
            "v must be updated to src.id (5)"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_reorder WHERE id = 999").await,
            1,
            "updated row must appear exactly once"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_reorder").await,
            10,
            "total row count must be unchanged"
        );
    }

    // ── atomic single-manifest MERGE ──────────────────────────────────

    /// Read the current manifest `latest` sequence from `<root>/_manifest.json`.
    async fn current_manifest_seq(storage: &Arc<dyn Storage>, root: &str) -> u64 {
        // Live sequence: a deferred WAL mutation advances it logically without
        // moving the `_manifest.json` pointer (with WAL default-on).
        latest_manifest(storage.as_ref(), root).await.sequence
    }

    /// ACCEPTANCE TEST: a MERGE with BOTH matched updates AND
    /// unmatched inserts advances the manifest sequence by EXACTLY ONE — proving
    /// both sides land in a single atomic manifest (not two separate commits).
    ///
    /// Table has ids 0..10 (v = id*10). MERGE updates ids 1,2,3 and inserts
    /// ids 100,101. After the merge:
    ///   - manifest sequence advanced by exactly 1 (atomicity proof).
    ///   - updated rows carry their new values; inserts are present once each;
    ///     no duplicates; COUNT is 10 original + 2 inserted = 12.
    #[tokio::test]
    async fn merge_is_atomic_single_manifest_advance() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_atomic", 10).await;

        let seq_before = current_manifest_seq(&storage, &root).await;

        // Source: 3 updates (ids 1,2,3 → new v) + 2 inserts (ids 100,101).
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let ids = vec![1i64, 2, 3, 100, 101];
        let vs = vec![1001i64, 1002, 1003, 10000, 10100];
        let src = RecordBatch::try_new(
            arrow_schema,
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(vs)),
            ],
        )
        .unwrap();

        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_atomic",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!(
            (stats.updated, stats.inserted),
            (3, 2),
            "3 matched updates + 2 unmatched inserts"
        );

        let seq_after = current_manifest_seq(&storage, &root).await;
        assert_eq!(
            seq_after - seq_before,
            1,
            "MERGE with matched + inserts must advance the manifest sequence by EXACTLY ONE \
             (one atomic manifest, not two commits)"
        );

        // Final state correctness.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_atomic").await,
            12,
            "10 original + 2 inserted = 12 rows, no duplicates from the 3 updates"
        );
        for (id, v) in [(1i64, 1001i64), (2, 1002), (3, 1003)] {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT v FROM t_merge_atomic WHERE id = {id}")
                )
                .await,
                v,
                "updated id={id} must show its new value"
            );
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_merge_atomic WHERE id = {id}")
                )
                .await,
                1,
                "updated id={id} must remain a single live row"
            );
        }
        for (id, v) in [(100i64, 10000i64), (101, 10100)] {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT v FROM t_merge_atomic WHERE id = {id}")
                )
                .await,
                v,
                "inserted id={id} must be present with its value"
            );
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_merge_atomic WHERE id = {id}")
                )
                .await,
                1,
                "inserted id={id} must appear exactly once"
            );
        }
    }

    /// A MERGE with ONLY matched rows (no inserts) advances the sequence by one
    /// and updates in place.
    #[tokio::test]
    async fn merge_only_matched_single_advance() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_only_upd", 10).await;
        let seq_before = current_manifest_seq(&storage, &root).await;

        let src = make_merge_src(5, 5005);
        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_only_upd",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!((stats.updated, stats.inserted), (1, 0));
        assert_eq!(
            current_manifest_seq(&storage, &root).await - seq_before,
            1,
            "matched-only MERGE advances the sequence by exactly one"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_only_upd WHERE id = 5").await,
            5005
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_only_upd").await,
            10
        );
    }

    /// A MERGE with ONLY inserts (no matched rows) advances the sequence by one
    /// and inserts fresh rows.
    #[tokio::test]
    async fn merge_only_inserts_single_advance() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_merge_only_ins", 10).await;
        let seq_before = current_manifest_seq(&storage, &root).await;

        let src = make_merge_src(500, 5000);
        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_merge_only_ins",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!((stats.updated, stats.inserted), (0, 1));
        assert_eq!(
            current_manifest_seq(&storage, &root).await - seq_before,
            1,
            "insert-only MERGE advances the sequence by exactly one"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_merge_only_ins WHERE id = 500").await,
            5000
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_merge_only_ins").await,
            11
        );
    }

    // ── batched-probe correctness test ──────────────────────────────────

    /// Batched-probe MERGE with a mixed CDC batch: some keys matched (live),
    /// some unmatched (new), one key deleted (tombstoned → treated as NOT MATCHED).
    ///
    /// Table has ids 0..20 (v = id * 10), UNIQUE index on `id`.
    /// Batch: update ids 0,1,2 (live → MATCHED), insert ids 50,51 (new → NOT MATCHED),
    ///        upsert id 5 after deleting it (tombstoned → NOT MATCHED → insert fresh).
    ///
    /// All six source rows hit `execute_merge` in a single call.  The batched
    /// IN-list probe issues ONE scan to locate live rows for all six keys.
    ///
    /// Expected end state:
    ///   - stats.updated == 3 (ids 0,1,2)
    ///   - stats.inserted == 3 (ids 5,50,51)
    ///   - COUNT(*) == 20 - 1 (delete) + 3 (inserts) = 22
    ///   - id=0 → v=1000, id=1 → v=1001, id=2 → v=1002 (updated)
    ///   - id=5 → v=5999 (re-inserted fresh after delete)
    ///   - id=50 → v=5000, id=51 → v=5010 (new inserts)
    ///   - COUNT(*) WHERE id=5 == 1 (single live entry, not duplicate)
    #[tokio::test]
    async fn merge_batched_probe_mixed_batch_correctness() {
        let (ctx, storage, root, _tmp) = registered_table_unique("t_batch_probe_mixed", 20).await;

        // Delete id=5 so the tombstone path is exercised.
        execute_sql(
            &ctx,
            Arc::clone(&storage),
            &root,
            "DELETE FROM t_batch_probe_mixed WHERE id = 5",
        )
        .await
        .unwrap();

        // Build the mixed source batch (6 rows):
        //   ids 0,1,2  → live keys (MATCHED → update)
        //   id  5      → tombstoned key (NOT MATCHED → insert fresh)
        //   ids 50, 51 → new keys (NOT MATCHED → insert)
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, false),
        ]));
        let ids = vec![0i64, 1, 2, 5, 50, 51];
        let vs = vec![1000i64, 1001, 1002, 5999, 5000, 5010];
        let src = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(vs)),
            ],
        )
        .unwrap();

        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            &root,
            "t_batch_probe_mixed",
            "id",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!(
            (stats.updated, stats.inserted),
            (3, 3),
            "3 live-key updates + 3 inserts (2 new + 1 dead-key re-insert)"
        );

        // Total count: started with 20, deleted 1, inserted 3 net.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_batch_probe_mixed").await,
            22,
            "row count must be 22 (20 - 1 deleted + 3 inserted)"
        );

        // Updated rows carry new values.
        for (id, v) in [(0i64, 1000i64), (1, 1001), (2, 1002)] {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT v FROM t_batch_probe_mixed WHERE id = {id}")
                )
                .await,
                v,
                "updated id={id} must show v={v}"
            );
        }

        // Tombstoned key re-inserted as fresh row.
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_batch_probe_mixed WHERE id = 5").await,
            5999,
            "re-inserted id=5 must show v=5999"
        );
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_batch_probe_mixed WHERE id = 5"
            )
            .await,
            1,
            "id=5 must appear exactly once (single live entry)"
        );

        // New inserts are visible.
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_batch_probe_mixed WHERE id = 50").await,
            5000,
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT v FROM t_batch_probe_mixed WHERE id = 51").await,
            5010,
        );
    }

    // ── acceptance test ─────────────────────────────────────────────────

    /// Build a table with multiple fragments, some rows deleted and some updated
    /// (patch fragments), then compact.  Returns the context, storage, root, and
    /// tempdir.
    ///
    /// Table layout before mutations:
    ///   Frag 0: ids 0..10  (row_ids 0..10)
    ///   Frag 1: ids 10..20 (row_ids 10..20)
    ///
    /// Mutations applied:
    ///   DELETE WHERE id = 3  (tombstones offset 3 in frag 0)
    ///   DELETE WHERE id = 15 (tombstones offset 5 in frag 1)
    ///   UPDATE SET id = id + 100 WHERE id = 7  (patch fragment; row_id 7 relocated)
    async fn table_with_deletes_and_patches(
        tbl: &str,
    ) -> (Arc<dyn Storage>, String, tempfile::TempDir, SessionContext) {
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        // 10 rows per fragment so we get two fragments.
        mdb_schema.row_group_target_rows = 10;

        // Fragment 0: ids 0..10
        let batch0 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from((0i64..10).collect::<Vec<_>>()))],
        )
        .unwrap();
        let mut writer = Writer::create(Arc::clone(&storage), tbl, mdb_schema.clone())
            .await
            .unwrap();
        writer.insert_batch(batch0).await.unwrap();
        writer.commit().await.unwrap();

        // Fragment 1: ids 10..20
        let batch1 = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from((10i64..20).collect::<Vec<_>>()))],
        )
        .unwrap();
        let mut writer2 = Writer::new(Arc::clone(&storage), tbl, mdb_schema.clone())
            .await
            .unwrap();
        writer2.insert_batch(batch1).await.unwrap();
        writer2.commit().await.unwrap();

        // Register so execute_sql can run.
        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), tbl, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(tbl, provider).unwrap();

        // Apply mutations via SQL.
        execute_sql(
            &ctx,
            Arc::clone(&storage),
            tbl,
            &format!("DELETE FROM {tbl} WHERE id = 3"),
        )
        .await
        .unwrap();
        execute_sql(
            &ctx,
            Arc::clone(&storage),
            tbl,
            &format!("DELETE FROM {tbl} WHERE id = 15"),
        )
        .await
        .unwrap();
        execute_sql(
            &ctx,
            Arc::clone(&storage),
            tbl,
            &format!("UPDATE {tbl} SET id = id + 100 WHERE id = 7"),
        )
        .await
        .unwrap();

        (storage, tbl.to_string(), tmp, ctx)
    }

    /// Helper: execute `sql` and return all (id, _rowid) pairs from the result.
    ///
    /// Expects the query to return exactly two Int64 columns: `id` and `_rowid`.
    async fn rows_as_pairs(ctx: &SessionContext, sql: &str) -> Vec<(i64, i64)> {
        use arrow::datatypes::Int64Type;

        let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
        let mut pairs = Vec::new();
        for batch in &batches {
            let ids = batch.column(0).as_primitive::<Int64Type>();
            let rowids = batch.column(1).as_primitive::<Int64Type>();
            for i in 0..batch.num_rows() {
                pairs.push((ids.value(i), rowids.value(i)));
            }
        }
        pairs
    }

    /// Helper: load the RowGroupMeta for a RowGroupEntry from storage.
    async fn load_row_group_meta(
        storage: &dyn Storage,
        root: &str,
        entry: &icefalldb_core::metadata::RowGroupEntry,
    ) -> icefalldb_core::metadata::RowGroupMeta {
        let meta_bytes = storage
            .read(&format!("{}/{}", root, entry.meta))
            .await
            .unwrap();
        serde_json::from_slice(&meta_bytes).unwrap()
    }

    /// Helper: re-register the table with a fresh provider after storage changes.
    async fn refresh_registration(ctx: &SessionContext, storage: Arc<dyn Storage>, tbl: &str) {
        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), tbl, config)
                .await
                .unwrap(),
        );
        ctx.deregister_table(tbl).unwrap();
        ctx.register_table(tbl, provider).unwrap();
    }

    /// Helper: read the latest manifest from storage.
    async fn latest_manifest(
        storage: &dyn Storage,
        root: &str,
    ) -> icefalldb_core::metadata::Manifest {
        let ptr = storage
            .read(&format!("{root}/_manifest.json"))
            .await
            .unwrap();
        let ptr: serde_json::Value = serde_json::from_slice(&ptr).unwrap();
        let seq = ptr["latest"].as_u64().unwrap();
        let m_bytes = storage
            .read(&format!(
                "{root}/{}",
                icefalldb_core::metadata::Manifest::filename(seq)
            ))
            .await
            .unwrap();
        let base: icefalldb_core::metadata::Manifest = serde_json::from_slice(&m_bytes).unwrap();
        // Replay any deferred mutation WAL so the helper returns the live manifest
        // (with WAL default-on a mutation does not advance the pointer).
        icefalldb_core::mutation_wal::live_manifest(storage, root, base)
            .await
            .unwrap()
    }

    /// Compaction folds deletions/patches, preserves move-stable
    /// row IDs, and produces dense Range segments with deleted_count == 0.
    ///
    /// Steps:
    ///   1. Build a table with two fragments, delete two rows, update one row.
    ///   2. Capture (id, _rowid) pairs BEFORE compaction.
    ///   3. Compact with recluster=false (no sort_keys → preserve row-id order).
    ///   4. Refresh the DataFusion table registration.
    ///   5. Capture (id, _rowid) pairs AFTER compaction.
    ///   6. Assert: same pairs (data + move-stable row_ids).
    ///   7. Assert: every compacted fragment's row_ids are all Range segments.
    ///   8. Assert: every compacted fragment has deleted_count == 0.
    #[tokio::test]
    async fn compaction_preserves_results_and_row_ids() {
        use icefalldb_core::compaction::{CompactionOptions, Compactor};
        use icefalldb_core::RowIdSegment;

        let tbl = "t_compact_gate";
        let (storage, root, _tmp, ctx) = table_with_deletes_and_patches(tbl).await;

        // Capture (id, _rowid) pairs before compaction.
        let before: Vec<(i64, i64)> = rows_as_pairs(
            &ctx,
            &format!("SELECT id, CAST(_rowid AS BIGINT) FROM {tbl} ORDER BY id"),
        )
        .await;

        // Compact without reclustering (sort_keys empty → preserve row-id order).
        // Use `force: true` so the compactor rewrites even if only a few fragments.
        let result = Compactor::with_options(
            storage.as_ref(),
            &root,
            CompactionOptions {
                target_row_group_rows: 1_000,
                target_row_group_bytes: 128 * 1024 * 1024,
                lock_timeout: std::time::Duration::from_secs(30),
                force: true,
                sort_keys: vec![],
            },
        )
        .compact()
        .await
        .unwrap();

        assert!(result.rewrote, "compaction should have rewritten fragments");

        // Refresh the DataFusion registration so the new fragments are visible.
        refresh_registration(&ctx, Arc::clone(&storage), tbl).await;

        // Capture (id, _rowid) pairs after compaction.
        let after: Vec<(i64, i64)> = rows_as_pairs(
            &ctx,
            &format!("SELECT id, CAST(_rowid AS BIGINT) FROM {tbl} ORDER BY id"),
        )
        .await;

        // Gate (a): same (id, _rowid) pairs — data correct AND move-stable row_ids.
        assert_eq!(
            before, after,
            "compaction must preserve all (id, _rowid) pairs identically"
        );

        // Gate (b)+(c): every compacted fragment has only Range row_id segments
        // and deleted_count == 0.
        let manifest = latest_manifest(storage.as_ref(), &root).await;
        for entry in &manifest.row_groups {
            let meta = load_row_group_meta(storage.as_ref(), &root, entry).await;

            // Gate (b): dense Range.
            for seg in &meta.row_ids {
                assert!(
                    matches!(seg, RowIdSegment::Range { .. }),
                    "compacted fragment {} must use Range row_ids (no Sorted), got {:?}",
                    entry.data,
                    seg
                );
            }

            // Gate (c): deletions folded.
            assert_eq!(
                entry.deleted_count, 0,
                "compacted fragment {} must have deleted_count == 0",
                entry.data
            );
        }
    }

    // ── acceptance test ──────────────────────────────────────────────────

    /// Collect all live row_ids from a compacted manifest.
    ///
    /// Compacted fragments have no deletion vectors, so every row_id in every
    /// fragment's meta is live.
    async fn live_row_ids(storage: &dyn Storage, root: &str, manifest: &Manifest) -> Vec<u64> {
        use icefalldb_core::rowid::segment_ids;
        let mut ids = Vec::new();
        for entry in &manifest.row_groups {
            let meta = load_row_group_meta(storage, root, entry).await;
            for seg in &meta.row_ids {
                ids.extend(segment_ids(seg));
            }
        }
        ids.sort_unstable();
        ids
    }

    /// Compaction rebuilds the _rowindex base from the compacted
    /// fragments and drops all accumulated deltas (heals delta fragmentation).
    ///
    /// Steps:
    ///   1. Build a table with one fragment; apply 6 UPDATEs via execute_sql —
    ///      each commit_update appends one delta, so we accumulate > 5 deltas.
    ///   2. Assert pre-compaction delta count > 5.
    ///   3. Compact with recluster=false.
    ///   4. Assert post-compaction gen.deltas is EMPTY (healed) and base is Some.
    ///   5. Open AddressMap from the new generation and assert every live row_id
    ///      resolves to a valid (fragment_id, offset).
    #[tokio::test]
    async fn compaction_heals_rowindex_to_zero_deltas() {
        use icefalldb_core::compaction::{CompactionOptions, Compactor};
        use icefalldb_core::rowindex::AddressMap;

        let tbl = "t_heal_rowindex";
        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "val",
            DataType::Int64,
            false,
        )]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        // Keep all rows in one fragment so we can compact after updates.
        mdb_schema.row_group_target_rows = 1_000;

        // Insert 10 rows (val = 0..10).
        let batch = RecordBatch::try_new(
            Arc::clone(&arrow_schema),
            vec![Arc::new(Int64Array::from((0i64..10).collect::<Vec<_>>()))],
        )
        .unwrap();
        let mut writer = Writer::create(Arc::clone(&storage), tbl, mdb_schema.clone())
            .await
            .unwrap();
        writer.insert_batch(batch).await.unwrap();
        writer.commit().await.unwrap();

        // Register the table so execute_sql can plan UPDATEs.
        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), tbl, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(tbl, provider).unwrap();

        // Apply 6 UPDATEs — each one appends one delta to rowindex_generation.
        // Each round targets a distinct row (val = 0, 1, 2, 3, 4, 5) and sets
        // its val to val + 1000 so each UPDATE matches exactly one live row.
        // After the first UPDATE, val=0 no longer exists; subsequent rounds each
        // target a different original value so they always match exactly one row.
        for i in 0i64..6 {
            execute_sql(
                &ctx,
                Arc::clone(&storage),
                tbl,
                &format!("UPDATE {tbl} SET val = val + 1000 WHERE val = {i}"),
            )
            .await
            .unwrap();
            // Refresh registration so subsequent queries see the updated snapshot.
            refresh_registration(&ctx, Arc::clone(&storage), tbl).await;
        }

        // Assert: pre-compaction rowindex_generation has > 5 deltas.
        let pre_manifest = latest_manifest(storage.as_ref(), tbl).await;
        let pre_gen = pre_manifest
            .rowindex_generation
            .as_ref()
            .expect("rowindex_generation must be set after updates");
        assert!(
            pre_gen.deltas.len() > 5,
            "expected > 5 deltas before compaction, got {}",
            pre_gen.deltas.len()
        );

        // Compact with recluster=false, force=true so compaction always runs.
        let result = Compactor::with_options(
            storage.as_ref(),
            tbl,
            CompactionOptions {
                target_row_group_rows: 10_000,
                target_row_group_bytes: 128 * 1024 * 1024,
                lock_timeout: std::time::Duration::from_secs(30),
                force: true,
                sort_keys: vec![],
            },
        )
        .compact()
        .await
        .unwrap();
        assert!(result.rewrote, "compaction should rewrite fragments");

        // Assert: post-compaction rowindex_generation has NO deltas and a base.
        let post_manifest = latest_manifest(storage.as_ref(), tbl).await;
        let post_gen = post_manifest
            .rowindex_generation
            .as_ref()
            .expect("rowindex_generation must be set after compaction");
        assert!(
            post_gen.deltas.is_empty(),
            "compaction must produce zero deltas (healed), got {:?}",
            post_gen.deltas
        );
        assert!(
            post_gen.base.is_some(),
            "compaction must produce a base rowindex file"
        );

        // Assert: AddressMap resolves every live row_id in the compacted manifest.
        let am = AddressMap::open(storage.as_ref(), tbl, post_gen)
            .await
            .unwrap();
        let live_ids = live_row_ids(storage.as_ref(), tbl, &post_manifest).await;
        assert!(!live_ids.is_empty(), "compacted table must have live rows");
        for row_id in &live_ids {
            assert!(
                am.lookup(*row_id).is_some(),
                "AddressMap must resolve live row_id {row_id} after compaction"
            );
        }
    }

    // ── String merge-key normalization ───────────────────────────────────────

    /// MERGE with a STRING merge key correctly routes a live key to MATCHED
    /// (update) and a new key to NOT MATCHED (insert), even when the source and
    /// scan-result columns may differ in string width (`Utf8` vs `LargeUtf8`).
    ///
    /// Without `normalize_merge_key` the live string key would fail the
    /// `HashMap` lookup (different `ScalarValue` variant → different hash/eq)
    /// and would be silently routed to insert → duplicate row, violating the
    /// unique-key contract.
    ///
    /// Table: two columns — `email` (Utf8, NOT NULL, UNIQUE index) and `score`
    /// (Int64).  Initial rows:
    ///   ("qa-verify-alice@example.com", 10)
    ///   ("qa-verify-bob@example.com",   20)
    ///
    /// Source batch:
    ///   (a) live key  → ("qa-verify-alice@example.com", 99)  → MATCHED → update
    ///   (b) new key   → ("qa-verify-carol@example.com", 30)  → NOT MATCHED → insert
    ///
    /// Assertions:
    ///   - stats.updated == 1, stats.inserted == 1
    ///   - COUNT(*) WHERE email = 'qa-verify-alice@example.com' == 1  (no duplicate)
    ///   - score WHERE email = 'qa-verify-alice@example.com' == 99    (updated)
    ///   - COUNT(*) WHERE email = 'qa-verify-carol@example.com' == 1  (inserted)
    ///   - COUNT(*) == 3  (2 original + 1 inserted, no duplicates)
    #[tokio::test]
    async fn merge_string_key_routes_correctly() {
        use arrow::array::StringArray;
        use icefalldb_core::database_catalog::DatabaseCatalog;
        use std::time::Duration;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        // Schema: email (Utf8, not null), score (Int64, not null).
        let arrow_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("email", DataType::Utf8, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = 10;

        // Register a UNIQUE btree index on `email` before writing.
        let dbcat = DatabaseCatalog::new(Arc::clone(&storage));
        let guard = dbcat.acquire_lock(Duration::from_secs(5)).await.unwrap();
        dbcat
            .create_index_definition_with_options(
                &guard,
                "email_uniq",
                "t_str_merge",
                "email",
                "btree",
                true,
            )
            .await
            .unwrap();
        drop(guard);

        // Initial rows.
        let email_arr: Arc<dyn arrow::array::Array> = Arc::new(StringArray::from(vec![
            "qa-verify-alice@example.com",
            "qa-verify-bob@example.com",
        ]));
        let score_arr: Arc<dyn arrow::array::Array> = Arc::new(Int64Array::from(vec![10i64, 20]));
        let init_batch =
            RecordBatch::try_new(Arc::clone(&arrow_schema), vec![email_arr, score_arr]).unwrap();

        let mut writer = Writer::create(Arc::clone(&storage), "t_str_merge", mdb_schema)
            .await
            .unwrap();
        writer.insert_batch(init_batch).await.unwrap();
        writer.commit().await.unwrap();

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), "t_str_merge", config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table("t_str_merge", provider).unwrap();

        // Source batch: live key alice (update) + new key carol (insert).
        let src_email: Arc<dyn arrow::array::Array> = Arc::new(StringArray::from(vec![
            "qa-verify-alice@example.com",
            "qa-verify-carol@example.com",
        ]));
        let src_score: Arc<dyn arrow::array::Array> = Arc::new(Int64Array::from(vec![99i64, 30]));
        let src =
            RecordBatch::try_new(Arc::clone(&arrow_schema), vec![src_email, src_score]).unwrap();

        let stats = execute_merge(
            &ctx,
            Arc::clone(&storage),
            "t_str_merge",
            "t_str_merge",
            "email",
            src,
            MatchedAction::UpdateAll,
            NotMatchedAction::Insert,
        )
        .await
        .unwrap();

        assert_eq!(
            (stats.updated, stats.inserted),
            (1, 1),
            "live string key must route to UPDATE, new string key to INSERT"
        );

        // No duplicate for the live key.
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_str_merge WHERE email = 'qa-verify-alice@example.com'"
            )
            .await,
            1,
            "live string key must appear exactly once after update (no duplicate)"
        );

        // Value updated.
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT score FROM t_str_merge WHERE email = 'qa-verify-alice@example.com'"
            )
            .await,
            99,
            "live string key must have its score updated to 99"
        );

        // New key inserted once.
        assert_eq!(
            scalar_i64(
                &ctx,
                "SELECT COUNT(*) FROM t_str_merge WHERE email = 'qa-verify-carol@example.com'"
            )
            .await,
            1,
            "new string key must be inserted exactly once"
        );

        // Total row count: 2 original + 1 inserted.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_str_merge").await,
            3,
            "total row count must be 3 (2 original + 1 inserted, no duplicates from update)"
        );
    }

    // ── batched mutations with a single refresh ────────────────────────

    /// Build a multi-fragment table with `fragments` fragments of
    /// `rows_per_fragment` rows each, registered as `table_name`. Returns the
    /// context, storage, table root, tempdir, and the registered provider so
    /// tests can inspect the provider refresh counter.
    async fn registered_table_fragments(
        table_name: &str,
        rows_per_fragment: usize,
        fragments: usize,
    ) -> (
        SessionContext,
        Arc<dyn Storage>,
        String,
        tempfile::TempDir,
        Arc<IcefallDBTableProvider>,
    ) {
        use datafusion::catalog::TableProvider;

        let tmp = tempfile::tempdir().unwrap();
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(tmp.path()).unwrap());

        let arrow_schema = Arc::new(ArrowSchema::new(vec![Field::new(
            "id",
            DataType::Int64,
            false,
        )]));
        let mut mdb_schema = arrow_schema_to_icefalldb(Arc::clone(&arrow_schema));
        mdb_schema.row_group_target_rows = rows_per_fragment;

        for f in 0..fragments {
            let start = f * rows_per_fragment;
            let ids: Vec<i64> = (start as i64..(start + rows_per_fragment) as i64).collect();
            let batch = RecordBatch::try_new(
                Arc::clone(&arrow_schema),
                vec![Arc::new(Int64Array::from(ids))],
            )
            .unwrap();

            let mut writer = if f == 0 {
                Writer::create(Arc::clone(&storage), table_name, mdb_schema.clone())
                    .await
                    .unwrap()
            } else {
                Writer::new(Arc::clone(&storage), table_name, mdb_schema.clone())
                    .await
                    .unwrap()
            };
            writer.insert_batch(batch).await.unwrap();
            writer.commit().await.unwrap();
        }

        let config = mutate_provider_config();
        let provider = Arc::new(
            IcefallDBTableProvider::new(Arc::clone(&storage), table_name, config)
                .await
                .unwrap(),
        );
        let ctx = icefalldb_session(1, 1024);
        ctx.register_table(table_name, Arc::clone(&provider) as Arc<dyn TableProvider>)
            .unwrap();

        let root = table_name.to_string();
        (ctx, storage, root, tmp, provider)
    }

    /// `execute_sql_batch` applies 50 point DELETEs and refreshes the provider
    /// exactly once. The final row count equals the result of deleting the same
    /// 50 rows individually.
    #[tokio::test]
    async fn batch_delete_fifty_point_deletes_single_refresh() {
        let (ctx, storage, root, _tmp, provider) =
            registered_table_fragments("t_batch_del", 100, 3).await;

        let sqls: Vec<String> = (0..50i64)
            .map(|id| format!("DELETE FROM t_batch_del WHERE id = {id}"))
            .collect();

        let counts = execute_sql_batch(&ctx, Arc::clone(&storage), &root, &sqls)
            .await
            .unwrap();

        assert_eq!(counts.len(), 50, "must return one count per statement");
        assert_eq!(
            counts.iter().sum::<u64>(),
            50,
            "fifty point deletes must delete exactly 50 rows"
        );
        assert!(
            counts.iter().all(|&c| c == 1),
            "each delete must affect 1 row"
        );

        // Exactly one provider refresh despite 50 commits.
        assert_eq!(
            provider.apply_delta_count(),
            1,
            "batched deletes must refresh the provider exactly once"
        );

        // Final state: 300 - 50 = 250 rows.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_batch_del").await,
            250,
            "row count after batch deletes must be 250"
        );

        // Deleted ids are gone, surviving ids are present.
        for id in 0i64..50 {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_batch_del WHERE id = {id}")
                )
                .await,
                0,
                "deleted id={id} must not be visible"
            );
        }
        for id in 50i64..300 {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_batch_del WHERE id = {id}")
                )
                .await,
                1,
                "surviving id={id} must be visible"
            );
        }
    }

    /// Regression: a batch using a *quoted* table identifier must not build a
    /// quoted deletion-vector path. Two deletes hitting the same fragment in one
    /// batch previously failed with `not found: "trips"/_deletions/...` because
    /// the write-side provider was constructed with the raw (quoted) SQL
    /// identifier instead of the canonical table name, so `apply_committed_delta`
    /// formatted the `.del` path with embedded quotes.
    #[tokio::test]
    async fn batch_delete_quoted_identifier_same_fragment() {
        let (ctx, storage, root, _tmp, _provider) =
            registered_table_fragments("trips", 100, 1).await;

        let sqls = vec![
            "DELETE FROM \"trips\" WHERE id = 1".to_string(),
            "DELETE FROM \"trips\" WHERE id = 2".to_string(),
        ];

        let counts = execute_sql_batch(&ctx, Arc::clone(&storage), &root, &sqls)
            .await
            .expect("quoted-identifier batch must not error");

        assert_eq!(counts, vec![1, 1], "each delete affects exactly one row");
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM trips").await,
            98,
            "two rows deleted from a 100-row table",
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM trips WHERE id = 1 OR id = 2").await,
            0,
            "both deleted ids must be gone",
        );
    }

    /// The final table state after `execute_sql_batch` matches applying the
    /// same statements individually through `execute_sql`.
    #[tokio::test]
    async fn batch_delete_matches_individual_deletes() {
        let (batch_ctx, batch_storage, batch_root, _batch_tmp, _batch_provider) =
            registered_table_fragments("t_batch_match", 100, 3).await;
        let (single_ctx, single_storage, single_root, _single_tmp, _single_provider) =
            registered_table_fragments("t_single_match", 100, 3).await;

        let sqls: Vec<String> = (0..50i64)
            .map(|id| format!("DELETE FROM t_batch_match WHERE id = {id}"))
            .collect();
        let single_sqls: Vec<String> = (0..50i64)
            .map(|id| format!("DELETE FROM t_single_match WHERE id = {id}"))
            .collect();

        let batch_counts =
            execute_sql_batch(&batch_ctx, Arc::clone(&batch_storage), &batch_root, &sqls)
                .await
                .unwrap();

        for sql in &single_sqls {
            execute_sql(&single_ctx, Arc::clone(&single_storage), &single_root, sql)
                .await
                .unwrap();
        }

        assert_eq!(
            batch_counts.iter().sum::<u64>(),
            50,
            "batch must report 50 deleted rows"
        );
        assert_eq!(
            scalar_i64(&batch_ctx, "SELECT COUNT(*) FROM t_batch_match").await,
            scalar_i64(&single_ctx, "SELECT COUNT(*) FROM t_single_match").await,
            "batch and individual deletes must leave the same row count"
        );

        let batch_ids = {
            let batches = batch_ctx
                .sql("SELECT id FROM t_batch_match ORDER BY id")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut ids = Vec::new();
            for batch in &batches {
                let arr = batch.column(0).as_primitive::<Int64Type>();
                for i in 0..batch.num_rows() {
                    ids.push(arr.value(i));
                }
            }
            ids
        };
        let single_ids = {
            let batches = single_ctx
                .sql("SELECT id FROM t_single_match ORDER BY id")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut ids = Vec::new();
            for batch in &batches {
                let arr = batch.column(0).as_primitive::<Int64Type>();
                for i in 0..batch.num_rows() {
                    ids.push(arr.value(i));
                }
            }
            ids
        };

        assert_eq!(
            batch_ids, single_ids,
            "batch and individual deletes must leave identical ordered ids"
        );
    }

    /// An empty batch is a no-op and performs zero provider refreshes.
    #[tokio::test]
    async fn empty_batch_is_noop() {
        let (ctx, storage, root, _tmp, provider) =
            registered_table_fragments("t_empty_batch", 100, 1).await;

        let counts = execute_sql_batch(&ctx, Arc::clone(&storage), &root, &[])
            .await
            .unwrap();

        assert!(counts.is_empty(), "empty batch must return empty counts");
        assert_eq!(
            provider.apply_delta_count(),
            0,
            "empty batch must not refresh the provider"
        );
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_empty_batch").await,
            100,
            "empty batch must not change the table"
        );
    }

    /// A batch where one UPDATE's changes are visible to the next UPDATE.
    ///
    /// The first statement moves ids 0..10 to 1000..1009.  The second statement
    /// updates ids in the 1000..1010 range, so it must see the rows just moved
    /// by the first statement.  Without incremental refresh of the write-side
    /// provider the second statement would run against the original snapshot and
    /// affect zero rows.
    #[tokio::test]
    async fn batch_update_chain_sees_predecessor_changes() {
        let (ctx, storage, root, _tmp, provider) =
            registered_table_fragments("t_batch_upd_chain", 100, 3).await;

        let sqls = vec![
            "UPDATE t_batch_upd_chain SET id = id + 1000 WHERE id < 10".to_string(),
            "UPDATE t_batch_upd_chain SET id = id + 1 WHERE id >= 1000 AND id < 1010".to_string(),
        ];

        let counts = execute_sql_batch(&ctx, Arc::clone(&storage), &root, &sqls)
            .await
            .unwrap();

        assert_eq!(counts, vec![10, 10], "each UPDATE must affect 10 rows");
        assert_eq!(
            provider.apply_delta_count(),
            1,
            "batched updates must refresh the provider exactly once"
        );

        // Original ids 0..9 are now 1001..1010; original 10..299 unchanged.
        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_batch_upd_chain").await,
            300,
            "total row count must stay 300"
        );
        for id in 0i64..10 {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_batch_upd_chain WHERE id = {id}")
                )
                .await,
                0,
                "original id={id} must have been moved"
            );
        }
        for id in 1001i64..1011 {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_batch_upd_chain WHERE id = {id}")
                )
                .await,
                1,
                "moved id={id} must be present"
            );
        }
    }

    /// A batch where an UPDATE changes rows and a subsequent DELETE removes
    /// rows by their post-UPDATE values.  This guards dependent statements that
    /// read rows touched by an earlier statement.
    #[tokio::test]
    async fn batch_update_then_delete_by_new_values() {
        let (ctx, storage, root, _tmp, provider) =
            registered_table_fragments("t_batch_upd_del", 100, 2).await;

        let sqls = vec![
            "UPDATE t_batch_upd_del SET id = id + 1000 WHERE id < 5".to_string(),
            "DELETE FROM t_batch_upd_del WHERE id >= 1000 AND id < 1005".to_string(),
        ];

        let counts = execute_sql_batch(&ctx, Arc::clone(&storage), &root, &sqls)
            .await
            .unwrap();

        assert_eq!(counts, vec![5, 5], "UPDATE 5 rows then DELETE those 5 rows");
        assert_eq!(
            provider.apply_delta_count(),
            1,
            "batched statements must refresh the provider exactly once"
        );

        assert_eq!(
            scalar_i64(&ctx, "SELECT COUNT(*) FROM t_batch_upd_del").await,
            195,
            "200 - 5 deleted rows = 195"
        );
        for id in 0i64..5 {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_batch_upd_del WHERE id = {id}")
                )
                .await,
                0,
                "updated-then-deleted id={id} must be gone"
            );
        }
        for id in 5i64..200 {
            assert_eq!(
                scalar_i64(
                    &ctx,
                    &format!("SELECT COUNT(*) FROM t_batch_upd_del WHERE id = {id}")
                )
                .await,
                1,
                "untouched id={id} must be present"
            );
        }
    }

    /// Many point UPDATEs in one batch refresh the provider once and match
    /// applying the same statements individually.
    #[tokio::test]
    async fn batch_update_matches_individual_updates() {
        let (batch_ctx, batch_storage, batch_root, _batch_tmp, _batch_provider) =
            registered_table_fragments("t_batch_upd_match", 100, 2).await;
        let (single_ctx, single_storage, single_root, _single_tmp, _single_provider) =
            registered_table_fragments("t_single_upd_match", 100, 2).await;

        let sqls: Vec<String> = (0..50i64)
            .map(|id| format!("UPDATE t_batch_upd_match SET id = id + 1000 WHERE id = {id}"))
            .collect();
        let single_sqls: Vec<String> = (0..50i64)
            .map(|id| format!("UPDATE t_single_upd_match SET id = id + 1000 WHERE id = {id}"))
            .collect();

        let batch_counts =
            execute_sql_batch(&batch_ctx, Arc::clone(&batch_storage), &batch_root, &sqls)
                .await
                .unwrap();

        for sql in &single_sqls {
            execute_sql(&single_ctx, Arc::clone(&single_storage), &single_root, sql)
                .await
                .unwrap();
        }

        assert_eq!(
            batch_counts.iter().sum::<u64>(),
            50,
            "batch must report 50 updated rows"
        );
        assert_eq!(
            scalar_i64(&batch_ctx, "SELECT COUNT(*) FROM t_batch_upd_match").await,
            scalar_i64(&single_ctx, "SELECT COUNT(*) FROM t_single_upd_match").await,
            "batch and individual updates must leave the same row count"
        );

        let batch_ids = {
            let batches = batch_ctx
                .sql("SELECT id FROM t_batch_upd_match ORDER BY id")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut ids = Vec::new();
            for batch in &batches {
                let arr = batch.column(0).as_primitive::<Int64Type>();
                for i in 0..batch.num_rows() {
                    ids.push(arr.value(i));
                }
            }
            ids
        };
        let single_ids = {
            let batches = single_ctx
                .sql("SELECT id FROM t_single_upd_match ORDER BY id")
                .await
                .unwrap()
                .collect()
                .await
                .unwrap();
            let mut ids = Vec::new();
            for batch in &batches {
                let arr = batch.column(0).as_primitive::<Int64Type>();
                for i in 0..batch.num_rows() {
                    ids.push(arr.value(i));
                }
            }
            ids
        };

        assert_eq!(
            batch_ids, single_ids,
            "batch and individual updates must leave identical ordered ids"
        );
    }
}
