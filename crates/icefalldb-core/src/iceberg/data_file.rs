use crate::metadata::RowGroupMeta;
use crate::storage::Storage;
use crate::Result;
use serde_json::Value;
use std::collections::HashMap;

/// A data file entry for an Iceberg manifest.
#[derive(Debug, Clone)]
pub struct DataFile {
    pub content: i32,
    pub file_path: String,
    pub file_format: String,
    pub record_count: i64,
    pub file_size_in_bytes: i64,
    pub column_sizes: HashMap<i32, i64>,
    pub value_counts: HashMap<i32, i64>,
    pub null_value_counts: HashMap<i32, i64>,
    pub lower_bounds: HashMap<i32, Vec<u8>>,
    pub upper_bounds: HashMap<i32, Vec<u8>>,
}

impl DataFile {
    pub async fn from_icefalldb(
        storage: &dyn Storage,
        table: &str,
        data_path: &str,
        meta: &RowGroupMeta,
        field_ids: &HashMap<String, i32>,
    ) -> Result<Self> {
        let full_data_path = format!("{}/{}", table, data_path);
        let file_size = storage.size(&full_data_path).await?;

        let mut column_sizes = HashMap::new();
        let mut value_counts = HashMap::new();
        let mut null_value_counts = HashMap::new();
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();

        for (col_name, stats) in &meta.columns {
            let Some(&field_id) = field_ids.get(col_name) else {
                continue;
            };
            // Placeholder column size: we do not have per-column Parquet sizes
            // readily available, so record zero.
            column_sizes.insert(field_id, 0i64);
            value_counts.insert(field_id, meta.rows as i64);
            null_value_counts.insert(field_id, stats.nulls as i64);

            if let Some(min) = bound_bytes(&stats.min) {
                lower_bounds.insert(field_id, min);
            }
            if let Some(max) = bound_bytes(&stats.max) {
                upper_bounds.insert(field_id, max);
            }
        }

        Ok(Self {
            content: 0, // DATA
            file_path: full_data_path,
            file_format: "PARQUET".into(),
            record_count: meta.rows as i64,
            file_size_in_bytes: file_size as i64,
            column_sizes,
            value_counts,
            null_value_counts,
            lower_bounds,
            upper_bounds,
        })
    }
}

/// Serialize a JSON statistic value to Iceberg bound bytes.
///
/// Iceberg stores lower/upper bounds as opaque byte arrays whose interpretation
/// depends on the column type. For primitive numeric and string types we use a
/// simple deterministic encoding (decimal string for numbers, UTF-8 bytes for
/// strings). Complex types are skipped in v1.
fn bound_bytes(value: &Option<Value>) -> Option<Vec<u8>> {
    let value = value.as_ref()?;
    match value {
        Value::Bool(b) => Some(vec![if *b { 1 } else { 0 }]),
        Value::Number(n) => Some(n.to_string().into_bytes()),
        Value::String(s) => Some(s.clone().into_bytes()),
        _ => None,
    }
}
