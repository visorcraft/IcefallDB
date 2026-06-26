use arrow::datatypes::{DataType, Field, Fields, TimeUnit};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// A column in a table schema.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Logical type of the column (e.g. `int64`, `string`).
    pub r#type: String,
    /// Whether the column may contain null values.
    pub nullable: bool,
    /// Stable integer identifier for the column.
    ///
    /// Field IDs are positive, assigned monotonically per table. A value of `0`
    /// means the ID has not been assigned yet and should be repaired before the
    /// schema is persisted. Dropped column IDs are tracked via
    /// [`Schema::max_field_id`] and are never reused.
    #[serde(default)]
    pub field_id: i32,
}

impl Column {
    /// Create a new column with an unassigned (`0`) field ID.
    pub fn new(name: impl Into<String>, r#type: impl Into<String>, nullable: bool) -> Self {
        Self {
            name: name.into(),
            r#type: r#type.into(),
            nullable,
            field_id: 0,
        }
    }
}

impl Default for Column {
    fn default() -> Self {
        Self {
            name: String::new(),
            r#type: String::new(),
            nullable: true,
            field_id: 0,
        }
    }
}

/// A table schema.
///
/// Schemas are versioned and stored as JSON files under `_schemas/` within the
/// table directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Schema {
    /// Monotonically increasing schema version id.
    pub schema_id: u64,
    /// Columns defined in this schema.
    pub columns: Vec<Column>,
    /// Optional list of columns used to partition the table.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_by: Option<Vec<String>>,
    /// Optional sort order for row groups.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<Vec<String>>,
    /// Optional list of low-cardinality columns declared as GROUP BY keys for
    /// warm partial aggregation.  When present, `compute_agg_state`
    /// buckets each fragment's rows by these key columns and stores per-group
    /// `ColAgg` partials in the `.agg` sidecar.  Absent for tables that do not
    /// declare grouping keys; the field is omitted from JSON so existing schema
    /// checksums remain stable.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub agg_group_keys: Option<Vec<String>>,
    /// Target number of rows per row group.
    pub row_group_target_rows: usize,
    /// Target uncompressed byte size per row group.
    pub row_group_target_bytes: usize,
    /// Columns that have been dropped from the schema.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub dropped_columns: Vec<String>,
    /// Highest field ID ever assigned to this schema, including dropped columns.
    ///
    /// This is used to ensure dropped column IDs are never reused. It is
    /// updated by [`Schema::assign_field_ids`] and persisted in schema files.
    #[serde(default)]
    pub max_field_id: i32,
}

/// Map a IcefallDB type string to its Arrow `DataType`.
///
/// This is the canonical mapping shared by the writer and the checker. In
/// addition to the primitive types it supports `list<T>`, `struct<...>`,
/// `map<K,V>`, `decimal128(p,s)`, and `fixed_size_binary(n)`.
pub fn icefalldb_type_to_arrow(type_str: &str) -> Option<DataType> {
    TypeParser::new(type_str.trim()).parse_type().ok()
}

/// Recursive-descent parser for IcefallDB type strings.
struct TypeParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> TypeParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse_type(&mut self) -> Result<DataType, String> {
        self.skip_ws();
        let word = self.parse_word()?;
        self.skip_ws();
        match word.as_str() {
            "int8" => Ok(DataType::Int8),
            "int16" => Ok(DataType::Int16),
            "int32" => Ok(DataType::Int32),
            "int64" => Ok(DataType::Int64),
            "uint8" => Ok(DataType::UInt8),
            "uint16" => Ok(DataType::UInt16),
            "uint32" => Ok(DataType::UInt32),
            "uint64" => Ok(DataType::UInt64),
            "float32" => Ok(DataType::Float32),
            "float64" => Ok(DataType::Float64),
            "utf8" | "string" => Ok(DataType::Utf8),
            "large_utf8" => Ok(DataType::LargeUtf8),
            "binary" => Ok(DataType::Binary),
            "large_binary" => Ok(DataType::LargeBinary),
            "bool" => Ok(DataType::Boolean),
            "timestamp" | "timestamp[us]" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
            "list" => {
                self.expect('<')?;
                let item = self.parse_type()?;
                self.expect('>')?;
                Ok(DataType::List(Arc::new(Field::new("item", item, true))))
            }
            "large_list" => {
                self.expect('<')?;
                let item = self.parse_type()?;
                self.expect('>')?;
                Ok(DataType::LargeList(Arc::new(Field::new(
                    "item", item, true,
                ))))
            }
            "struct" => {
                self.expect('<')?;
                let fields = self.parse_struct_fields()?;
                self.expect('>')?;
                Ok(DataType::Struct(fields.into()))
            }
            "map" => {
                self.expect('<')?;
                let key = self.parse_type()?;
                self.expect(',')?;
                let value = self.parse_type()?;
                self.expect('>')?;
                let entry_fields = Fields::from(vec![
                    Field::new("key", key, false),
                    Field::new("value", value, true),
                ]);
                Ok(DataType::Map(
                    Arc::new(Field::new("entries", DataType::Struct(entry_fields), false)),
                    false,
                ))
            }
            "decimal128" => {
                self.expect('(')?;
                let precision = self.parse_usize()?;
                self.expect(',')?;
                let scale = self.parse_usize()?;
                self.expect(')')?;
                Ok(DataType::Decimal128(precision as u8, scale as i8))
            }
            "fixed_size_binary" => {
                self.expect('(')?;
                let n = self.parse_usize()?;
                self.expect(')')?;
                Ok(DataType::FixedSizeBinary(n as i32))
            }
            other => Err(format!("unsupported type: {}", other)),
        }
    }

    fn parse_struct_fields(&mut self) -> Result<Vec<Field>, String> {
        let mut fields = Vec::new();
        loop {
            self.skip_ws();
            if self.peek() == Some('>') {
                break;
            }
            let name = self.parse_field_name()?;
            self.skip_ws();
            self.expect(':')?;
            let data_type = self.parse_type()?;
            fields.push(Field::new(name, data_type, true));
            self.skip_ws();
            if self.peek() == Some(',') {
                self.advance();
                continue;
            }
            if self.peek() == Some('>') {
                break;
            }
            return Err(format!(
                "expected ',' or '>' in struct fields at position {}",
                self.pos
            ));
        }
        Ok(fields)
    }

    fn parse_word(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        let Some(c) = self.peek_char() else {
            return Err("unexpected end of type string".into());
        };
        if !c.is_alphabetic() && c != '_' {
            return Err(format!(
                "expected type name at position {}: {:?}",
                self.pos, c
            ));
        }
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }
        let word = &self.input[start..self.pos];
        if word == "timestamp" {
            // Support both `timestamp` and `timestamp[us]`.
            self.skip_ws();
            if self.peek() == Some('[') {
                self.advance();
                let inner_start = self.pos;
                while let Some(c) = self.peek_char() {
                    if c == ']' {
                        break;
                    }
                    self.advance();
                }
                if self.peek() != Some(']') {
                    return Err("unclosed '[' in timestamp type".into());
                }
                let inner = &self.input[inner_start..self.pos];
                if inner != "us" {
                    return Err(format!(
                        "unsupported timestamp resolution: {}; only [us] is supported",
                        inner
                    ));
                }
                self.advance(); // ']'
            }
        }
        Ok(word.to_string())
    }

    fn parse_field_name(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        let Some(c) = self.peek_char() else {
            return Err("unexpected end of type string".into());
        };
        if !c.is_alphabetic() && c != '_' {
            return Err(format!(
                "expected field name at position {}: {:?}",
                self.pos, c
            ));
        }
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_usize(&mut self) -> Result<usize, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(c) = self.peek_char() {
            if c.is_ascii_digit() {
                self.advance();
            } else {
                break;
            }
        }
        let num = &self.input[start..self.pos];
        num.parse::<usize>()
            .map_err(|e| format!("invalid integer '{}': {}", num, e))
    }

    fn expect(&mut self, expected: char) -> Result<(), String> {
        self.skip_ws();
        match self.peek() {
            Some(c) if c == expected => {
                self.advance();
                Ok(())
            }
            Some(c) => Err(format!(
                "expected '{}' but found '{}' at position {}",
                expected, c, self.pos
            )),
            None => Err(format!(
                "expected '{}' but reached end of input at position {}",
                expected, self.pos
            )),
        }
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn peek(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn peek_char(&self) -> Option<char> {
        self.peek()
    }

    fn advance(&mut self) -> Option<char> {
        let mut chars = self.input[self.pos..].chars();
        let c = chars.next()?;
        self.pos += c.len_utf8();
        Some(c)
    }
}

impl Schema {
    /// Build an Arrow schema from this IcefallDB schema.
    ///
    /// Returns `None` if any column declares an unsupported type.
    pub fn arrow_schema(&self) -> Option<arrow::datatypes::Schema> {
        let fields: Vec<_> = self
            .columns
            .iter()
            .map(|c| {
                icefalldb_type_to_arrow(&c.r#type)
                    .map(|dt| arrow::datatypes::Field::new(&c.name, dt, c.nullable))
            })
            .collect::<Option<_>>()?;
        Some(arrow::datatypes::Schema::new(fields))
    }

    /// Assign stable field IDs to all columns.
    ///
    /// - If `previous` is `None`, assigns IDs `1..=N` to all columns in order.
    /// - If `previous` is provided, columns that exist in the previous schema
    ///   keep their previous IDs, and new columns get monotonically increasing
    ///   IDs starting from `max(previous max_field_id, previous column IDs) + 1`.
    ///
    /// Dropped column IDs are captured by `previous.max_field_id` and are never
    /// reused. After assignment, `self.max_field_id` is updated to the maximum
    /// of the previous `max_field_id` and the highest ID assigned by this call
    /// so it never decreases.
    pub fn assign_field_ids(&mut self, previous: Option<&Schema>) {
        let dropped: std::collections::HashSet<&str> = previous
            .iter()
            .flat_map(|s| s.dropped_columns.iter().map(|n| n.as_str()))
            .collect();

        let previous_ids: HashMap<&str, i32> = previous
            .iter()
            .flat_map(|s| {
                s.columns
                    .iter()
                    .filter(|c| c.field_id > 0 && !dropped.contains(c.name.as_str()))
                    .map(|c| (c.name.as_str(), c.field_id))
            })
            .collect();

        let previous_max_field_id = previous.map(|s| s.max_field_id.max(0)).unwrap_or(0);
        let previous_highest_column_id = previous
            .map(|s| {
                s.columns
                    .iter()
                    .map(|c| c.field_id)
                    .filter(|&id| id > 0)
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        let mut next_id = previous_max_field_id
            .max(previous_highest_column_id)
            .max(self.max_field_id.max(0))
            + 1;

        for column in &mut self.columns {
            if let Some(&existing_id) = previous_ids.get(column.name.as_str()) {
                column.field_id = existing_id;
            } else {
                column.field_id = next_id;
                next_id += 1;
            }
        }

        let highest_newly_assigned = next_id - 1;
        self.max_field_id = self
            .max_field_id
            .max(0)
            .max(previous_max_field_id)
            .max(highest_newly_assigned);
    }

    /// Reassign all field IDs monotonically for legacy schemas that have no
    /// valid field IDs.
    ///
    /// Assignment starts from `max(1, max_field_id + 1)` so that IDs previously
    /// tracked as dropped (captured by `max_field_id`) are preserved and not
    /// reused. Callers should guard with [`Schema::has_field_ids`] to avoid
    /// rewriting IDs that are already valid.
    pub fn repair_field_ids(&mut self) {
        self.assign_field_ids(None);
    }

    /// Returns true if every column has a positive field ID.
    pub fn has_field_ids(&self) -> bool {
        self.columns.iter().all(|c| c.field_id > 0)
    }

    /// Returns the next field ID that would be assigned to a new column.
    ///
    /// This accounts for both existing column IDs and dropped columns tracked
    /// by `max_field_id`. It is `max(max_field_id, existing IDs) + 1`, or `1`
    /// if no IDs exist.
    pub fn next_field_id(&self) -> i32 {
        let max_existing_id = self
            .columns
            .iter()
            .map(|c| c.field_id)
            .filter(|&id| id > 0)
            .max()
            .unwrap_or(0);
        self.max_field_id.max(max_existing_id) + 1
    }

    /// Returns the relative path for a schema with the given id within the table
    /// directory.
    pub fn filename(id: u64) -> String {
        format!("_schemas/{:06}.json", id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icefalldb_type_to_arrow_timestamp() {
        assert_eq!(
            icefalldb_type_to_arrow("timestamp[us]"),
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        );
        assert_eq!(
            icefalldb_type_to_arrow("timestamp"),
            Some(DataType::Timestamp(TimeUnit::Microsecond, None))
        );
    }

    #[test]
    fn test_icefalldb_type_to_arrow_large_list() {
        assert_eq!(
            icefalldb_type_to_arrow("large_list<int64>"),
            Some(DataType::LargeList(Arc::new(Field::new(
                "item",
                DataType::Int64,
                true
            ))))
        );
        assert_eq!(
            icefalldb_type_to_arrow("large_list<utf8>"),
            Some(DataType::LargeList(Arc::new(Field::new(
                "item",
                DataType::Utf8,
                true
            ))))
        );
    }

    #[test]
    fn test_assign_field_ids_negative_max_field_id() {
        let mut schema = Schema {
            schema_id: 1,
            columns: vec![
                Column::new("a", "int64", false),
                Column::new("b", "utf8", true),
            ],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1,
            row_group_target_bytes: 1,
            dropped_columns: vec![],
            max_field_id: -5,
        };

        schema.assign_field_ids(None);

        assert_eq!(schema.columns[0].field_id, 1);
        assert_eq!(schema.columns[1].field_id, 2);
        assert_eq!(schema.max_field_id, 2);
    }
}
