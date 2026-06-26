//! TSV import/export for IcefallDB.
//!
//! Flat tables are encoded as a header row plus tab-separated values. NULL is an
//! empty field; tabs, newlines, carriage returns, and backslashes are escaped as
//! `\t`, `\n`, `\r`, and `\\`. Complex values (lists, structs, maps, decimals,
//! fixed-size binary, large binary) are serialized as JSON strings inside the
//! TSV cell and parsed back according to the declared schema type.

use crate::metadata::schema::icefalldb_type_to_arrow;
use crate::metadata::Schema;
use crate::{IcefallDBError, Result};
use arrow::array::builder::{
    ArrayBuilder, BinaryBuilder, BooleanBuilder, Decimal128Builder, Decimal256Builder,
    FixedSizeBinaryBuilder, Float32Builder, Float64Builder, Int16Builder, Int32Builder,
    Int64Builder, Int8Builder, LargeBinaryBuilder, LargeStringBuilder, ListBuilder, MapBuilder,
    StringBuilder, StructBuilder, TimestampMicrosecondBuilder, TimestampMillisecondBuilder,
    TimestampNanosecondBuilder, TimestampSecondBuilder, UInt16Builder, UInt32Builder,
    UInt64Builder, UInt8Builder,
};
use arrow::array::{Array, ArrayRef, AsArray, RecordBatch};
use arrow::datatypes::{DataType, TimeUnit};
use std::sync::Arc;

fn tsv_error(
    line: usize,
    column: usize,
    value: impl Into<String>,
    message: impl Into<String>,
) -> IcefallDBError {
    IcefallDBError::TsvError {
        line,
        column,
        value: value.into(),
        reason: message.into(),
    }
}

/// Escape a single TSV field.
///
/// Tabs, newlines, carriage returns, and backslashes are escaped. NULL is
/// represented by the empty string, so this function is never called for NULL
/// values.
pub fn escape_tsv_field(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Unescape a single TSV field.
///
/// Returns `None` when the input is empty (representing NULL) and `Some(...)`
/// otherwise. Invalid escape sequences produce an error.
pub fn unescape_tsv_field(line: usize, column: usize, value: &str) -> Result<Option<String>> {
    if value.is_empty() {
        return Ok(None);
    }
    let mut out = String::with_capacity(value.len());
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some(other) => {
                return Err(tsv_error(
                    line,
                    column,
                    value,
                    format!("invalid escape sequence: \\{}", other),
                ))
            }
            None => {
                return Err(tsv_error(line, column, value, "trailing backslash"));
            }
        }
    }
    Ok(Some(out))
}

/// Split a TSV line into fields, respecting escaped tab characters (`\t`).
///
/// The returned fields are byte slices into the original line and still contain
/// escape sequences for backslashes and newlines; callers should use
/// [`unescape_tsv_field`] to fully unescape each field.
pub fn split_tsv_line(line_text: &[u8]) -> Result<Vec<&[u8]>> {
    let line_text = if line_text.ends_with(b"\r") {
        &line_text[..line_text.len() - 1]
    } else {
        line_text
    };
    if line_text.is_empty() {
        return Ok(Vec::new());
    }
    let mut fields = Vec::new();
    let mut start = 0;
    let mut escaped = false;
    for (i, &b) in line_text.iter().enumerate() {
        if escaped {
            escaped = false;
            continue;
        }
        if b == b'\\' {
            escaped = true;
            continue;
        }
        if b == b'\t' {
            fields.push(&line_text[start..i]);
            start = i + 1;
        }
    }
    fields.push(&line_text[start..]);
    Ok(fields)
}

/// Join TSV fields into a single line.
pub fn join_tsv_fields(fields: &[String]) -> String {
    fields.join("\t")
}

/// Encode an Arrow [`RecordBatch`] as TSV bytes.
pub fn encode_tsv(batch: &RecordBatch) -> Result<Vec<u8>> {
    Ok(TsvEncoder::encode(batch))
}

/// Decode TSV bytes into Arrow [`RecordBatch`]es using the supplied IcefallDB
/// [`Schema`].
pub fn decode_tsv(data: &[u8], schema: &Schema) -> Result<Vec<RecordBatch>> {
    TsvDecoder::decode(data, schema)
}

/// TSV encoder for a fixed Arrow schema.
pub struct TsvEncoder;

impl TsvEncoder {
    /// Encode a record batch. The output begins with a header row of field names.
    pub fn encode(batch: &RecordBatch) -> Vec<u8> {
        let schema = batch.schema();
        let mut lines: Vec<String> = Vec::with_capacity(batch.num_rows() + 1);
        let header: Vec<String> = schema
            .fields()
            .iter()
            .map(|f| escape_tsv_field(f.name()))
            .collect();
        lines.push(join_tsv_fields(&header));

        for row in 0..batch.num_rows() {
            let mut fields = Vec::with_capacity(batch.num_columns());
            for col in 0..batch.num_columns() {
                let array = batch.column(col);
                let field = schema.field(col);
                let value = serialize_value(array, row, field.data_type())
                    .expect("TSV serialization should not fail for supported types");
                fields.push(value.unwrap_or_default());
            }
            lines.push(join_tsv_fields(&fields));
        }

        lines.join("\n").into_bytes()
    }
}

/// TSV decoder for a fixed IcefallDB schema.
pub struct TsvDecoder;

impl TsvDecoder {
    /// Decode TSV bytes into a single-element vector of record batches.
    ///
    /// The first non-empty line is the header.
    pub fn decode(data: &[u8], schema: &Schema) -> Result<Vec<RecordBatch>> {
        let arrow_schema =
            Arc::new(
                schema
                    .arrow_schema()
                    .ok_or_else(|| IcefallDBError::SchemaMismatch {
                        column: "schema".into(),
                        expected: "supported Arrow types".into(),
                        path: "tsv decoder".into(),
                    })?,
            );
        let text = std::str::from_utf8(data)
            .map_err(|e| tsv_error(0, 0, "", format!("invalid UTF-8: {}", e)))?;
        let mut lines = text.lines().enumerate();
        let (header_line_no, header_line) = lines
            .next()
            .ok_or_else(|| tsv_error(0, 0, "", "empty TSV input"))?;
        let header_line_no = header_line_no + 1;
        let header_fields = split_tsv_line(header_line.as_bytes())?;
        let header_fields: Vec<String> = header_fields
            .iter()
            .map(|f| String::from_utf8_lossy(f).to_string())
            .collect();

        // Map column name -> position in the file.
        let mut positions: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for (idx, name) in header_fields.iter().enumerate() {
            let unescaped = unescape_tsv_field(header_line_no, idx + 1, name)?.unwrap_or_default();
            if positions.insert(unescaped.clone(), idx).is_some() {
                return Err(tsv_error(
                    header_line_no,
                    idx + 1,
                    name,
                    "duplicate column in header",
                ));
            }
        }

        // Validate that every schema column is present.
        for (idx, col) in schema.columns.iter().enumerate() {
            if !positions.contains_key(&col.name) {
                return Err(tsv_error(
                    header_line_no,
                    idx + 1,
                    &col.name,
                    "column missing from TSV header",
                ));
            }
        }

        // Pre-allocate builders for each schema column.
        let mut builders: Vec<Box<dyn ArrayBuilder>> = Vec::with_capacity(schema.columns.len());
        for col in &schema.columns {
            let data_type = icefalldb_type_to_arrow(&col.r#type).ok_or_else(|| {
                IcefallDBError::SchemaMismatch {
                    column: col.name.clone(),
                    expected: "supported type".into(),
                    path: "tsv decoder".into(),
                }
            })?;
            builders.push(make_builder(&data_type));
        }

        for (zero_based, line) in lines {
            if line.is_empty() {
                continue;
            }
            let line_no = zero_based + 1;
            let fields = split_tsv_line(line.as_bytes())?;
            for (col_idx, col) in schema.columns.iter().enumerate() {
                let pos = positions[&col.name];
                let raw = fields
                    .get(pos)
                    .map(|c| String::from_utf8_lossy(c).to_string())
                    .unwrap_or_default();
                let unescaped = unescape_tsv_field(line_no, col_idx + 1, &raw)?;
                let data_type = icefalldb_type_to_arrow(&col.r#type).ok_or_else(|| {
                    IcefallDBError::SchemaMismatch {
                        column: col.name.clone(),
                        expected: "supported type".into(),
                        path: "tsv decoder".into(),
                    }
                })?;
                append_value(
                    builders[col_idx].as_mut(),
                    &data_type,
                    unescaped,
                    line_no,
                    col_idx + 1,
                    &raw,
                )?;
            }
        }

        let arrays: Vec<ArrayRef> = builders.into_iter().map(|mut b| b.finish()).collect();
        let batch = RecordBatch::try_new(arrow_schema, arrays)
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
        Ok(vec![batch])
    }
}

fn serialize_value(array: &ArrayRef, row: usize, data_type: &DataType) -> Result<Option<String>> {
    if array.is_null(row) {
        return Ok(None);
    }
    let value = match data_type {
        DataType::Int8 => array
            .as_primitive::<arrow::datatypes::Int8Type>()
            .value(row)
            .to_string(),
        DataType::Int16 => array
            .as_primitive::<arrow::datatypes::Int16Type>()
            .value(row)
            .to_string(),
        DataType::Int32 => array
            .as_primitive::<arrow::datatypes::Int32Type>()
            .value(row)
            .to_string(),
        DataType::Int64 => array
            .as_primitive::<arrow::datatypes::Int64Type>()
            .value(row)
            .to_string(),
        DataType::UInt8 => array
            .as_primitive::<arrow::datatypes::UInt8Type>()
            .value(row)
            .to_string(),
        DataType::UInt16 => array
            .as_primitive::<arrow::datatypes::UInt16Type>()
            .value(row)
            .to_string(),
        DataType::UInt32 => array
            .as_primitive::<arrow::datatypes::UInt32Type>()
            .value(row)
            .to_string(),
        DataType::UInt64 => array
            .as_primitive::<arrow::datatypes::UInt64Type>()
            .value(row)
            .to_string(),
        DataType::Float32 => array
            .as_primitive::<arrow::datatypes::Float32Type>()
            .value(row)
            .to_string(),
        DataType::Float64 => array
            .as_primitive::<arrow::datatypes::Float64Type>()
            .value(row)
            .to_string(),
        DataType::Boolean => array.as_boolean().value(row).to_string(),
        DataType::Utf8 => array.as_string::<i32>().value(row).to_string(),
        DataType::LargeUtf8 => array.as_string::<i64>().value(row).to_string(),
        DataType::Timestamp(unit, None) => format_timestamp(array, row, *unit)?,
        DataType::Binary => {
            return Ok(Some(serde_json::to_string(&base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                array.as_binary::<i32>().value(row),
            ))?));
        }
        DataType::LargeBinary => {
            return Ok(Some(serde_json::to_string(&base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                array.as_binary::<i64>().value(row),
            ))?));
        }
        DataType::FixedSizeBinary(_) => {
            return Ok(Some(serde_json::to_string(&base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                array.as_fixed_size_binary().value(row),
            ))?));
        }
        DataType::Decimal128(precision, scale) => {
            let v = array
                .as_primitive::<arrow::datatypes::Decimal128Type>()
                .value(row);
            return Ok(Some(serde_json::to_string(&format_decimal128(
                v, *precision, *scale,
            ))?));
        }
        DataType::Decimal256(precision, scale) => {
            let v = array
                .as_primitive::<arrow::datatypes::Decimal256Type>()
                .value(row);
            return Ok(Some(serde_json::to_string(&format_decimal256(
                v, *precision, *scale,
            ))?));
        }
        DataType::List(_) | DataType::LargeList(_) | DataType::Struct(_) | DataType::Map(_, _) => {
            let json = array_value_to_json(array, row, data_type)?;
            return Ok(Some(json.to_string()));
        }
        other => {
            return Err(IcefallDBError::TypeNotSupported(format!(
                "TSV serialization not supported for {}",
                other
            )))
        }
    };
    Ok(Some(escape_tsv_field(&value)))
}

fn format_timestamp(array: &ArrayRef, row: usize, unit: TimeUnit) -> Result<String> {
    use chrono::{DateTime, Utc};
    let micros: i64 = match unit {
        TimeUnit::Second => {
            let v = array
                .as_primitive::<arrow::datatypes::TimestampSecondType>()
                .value(row);
            v * 1_000_000
        }
        TimeUnit::Millisecond => {
            let v = array
                .as_primitive::<arrow::datatypes::TimestampMillisecondType>()
                .value(row);
            v * 1_000
        }
        TimeUnit::Microsecond => array
            .as_primitive::<arrow::datatypes::TimestampMicrosecondType>()
            .value(row),
        TimeUnit::Nanosecond => {
            let v = array
                .as_primitive::<arrow::datatypes::TimestampNanosecondType>()
                .value(row);
            v / 1_000
        }
    };
    let secs = micros / 1_000_000;
    let nanos = ((micros % 1_000_000) * 1_000) as u32;
    let dt = DateTime::from_timestamp(secs, nanos).ok_or_else(|| {
        IcefallDBError::TypeNotSupported(format!("timestamp out of range at row {}", row))
    })?;
    Ok(dt
        .with_timezone(&Utc)
        .to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true))
}

fn format_decimal128(value: i128, _precision: u8, scale: i8) -> String {
    let scale = scale as i32;
    let sign = if value < 0 { "-" } else { "" };
    let value = value.unsigned_abs().to_string();
    if scale <= 0 {
        let zeros = "0".repeat(scale.unsigned_abs() as usize);
        return format!("{}{}{}", sign, value, zeros);
    }
    let scale = scale as usize;
    let value = if value.len() <= scale {
        let pad = "0".repeat(scale - value.len() + 1);
        format!("{}{}", pad, value)
    } else {
        value
    };
    let (int_part, frac_part) = value.split_at(value.len() - scale);
    format!("{}{}.{}", sign, int_part, frac_part)
}

fn format_decimal256(value: arrow::datatypes::i256, _precision: u8, scale: i8) -> String {
    let scale = scale as i32;
    let sign = if value < arrow::datatypes::i256::from(0) {
        "-"
    } else {
        ""
    };
    let value = value.wrapping_abs().to_string();
    if scale <= 0 {
        let zeros = "0".repeat(scale.unsigned_abs() as usize);
        return format!("{}{}{}", sign, value, zeros);
    }
    let scale = scale as usize;
    let value = if value.len() <= scale {
        let pad = "0".repeat(scale - value.len() + 1);
        format!("{}{}", pad, value)
    } else {
        value
    };
    let (int_part, frac_part) = value.split_at(value.len() - scale);
    format!("{}{}.{}", sign, int_part, frac_part)
}

fn array_value_to_json(
    array: &ArrayRef,
    row: usize,
    data_type: &DataType,
) -> Result<serde_json::Value> {
    if array.is_null(row) {
        return Ok(serde_json::Value::Null);
    }
    Ok(match data_type {
        DataType::Int8 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Int8Type>()
                .value(row),
        ),
        DataType::Int16 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Int16Type>()
                .value(row),
        ),
        DataType::Int32 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Int32Type>()
                .value(row),
        ),
        DataType::Int64 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Int64Type>()
                .value(row),
        ),
        DataType::UInt8 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::UInt8Type>()
                .value(row),
        ),
        DataType::UInt16 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::UInt16Type>()
                .value(row),
        ),
        DataType::UInt32 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::UInt32Type>()
                .value(row),
        ),
        DataType::UInt64 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::UInt64Type>()
                .value(row),
        ),
        DataType::Float32 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Float32Type>()
                .value(row),
        ),
        DataType::Float64 => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Float64Type>()
                .value(row),
        ),
        DataType::Boolean => serde_json::Value::from(array.as_boolean().value(row)),
        DataType::Utf8 => serde_json::Value::from(array.as_string::<i32>().value(row)),
        DataType::LargeUtf8 => serde_json::Value::from(array.as_string::<i64>().value(row)),
        DataType::Binary => serde_json::Value::from(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            array.as_binary::<i32>().value(row),
        )),
        DataType::LargeBinary => serde_json::Value::from(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            array.as_binary::<i64>().value(row),
        )),
        DataType::FixedSizeBinary(_) => serde_json::Value::from(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            array.as_fixed_size_binary().value(row),
        )),
        DataType::Decimal128(_, _) => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Decimal128Type>()
                .value(row)
                .to_string(),
        ),
        DataType::Decimal256(_, _) => serde_json::Value::from(
            array
                .as_primitive::<arrow::datatypes::Decimal256Type>()
                .value(row)
                .to_string(),
        ),
        DataType::Timestamp(unit, None) => {
            serde_json::Value::from(format_timestamp(array, row, *unit)?)
        }
        DataType::List(field) | DataType::LargeList(field) => {
            let list_array = array.as_list::<i32>();
            let values = list_array.value(row);
            let mut arr = Vec::with_capacity(values.len());
            for i in 0..values.len() {
                arr.push(array_value_to_json(&values, i, field.data_type())?);
            }
            serde_json::Value::Array(arr)
        }
        DataType::Struct(fields) => {
            let struct_array = array.as_struct();
            let mut map = serde_json::Map::new();
            for (i, field) in fields.iter().enumerate() {
                let child = struct_array.column(i);
                let v = array_value_to_json(child, row, field.data_type())?;
                map.insert(field.name().clone(), v);
            }
            serde_json::Value::Object(map)
        }
        DataType::Map(map_field, _sorted) => {
            let map_array = array.as_map();
            let entries = map_array.value(row);
            let entry_fields = match map_field.data_type() {
                DataType::Struct(fields) => fields,
                _ => unreachable!("map field must be a struct"),
            };
            let key_field = &entry_fields[0];
            let value_field = &entry_fields[1];
            let key_array = entries.column(0);
            let value_array = entries.column(1);

            let string_keys = matches!(key_field.data_type(), DataType::Utf8 | DataType::LargeUtf8);
            if string_keys {
                let mut map = serde_json::Map::new();
                for i in 0..entries.len() {
                    let k = array_value_to_json(key_array, i, key_field.data_type())?;
                    let v = array_value_to_json(value_array, i, value_field.data_type())?;
                    if let serde_json::Value::String(s) = k {
                        map.insert(s, v);
                    } else {
                        unreachable!("non-string key for string-key map");
                    }
                }
                serde_json::Value::Object(map)
            } else {
                let mut arr = Vec::with_capacity(entries.len());
                for i in 0..entries.len() {
                    let mut obj = serde_json::Map::new();
                    obj.insert(
                        "key".to_string(),
                        array_value_to_json(key_array, i, key_field.data_type())?,
                    );
                    obj.insert(
                        "value".to_string(),
                        array_value_to_json(value_array, i, value_field.data_type())?,
                    );
                    arr.push(serde_json::Value::Object(obj));
                }
                serde_json::Value::Array(arr)
            }
        }
        other => {
            return Err(IcefallDBError::TypeNotSupported(format!(
                "JSON serialization not supported for {}",
                other
            )))
        }
    })
}

fn make_builder(data_type: &DataType) -> Box<dyn ArrayBuilder> {
    match data_type {
        DataType::Int8 => Box::new(Int8Builder::new()),
        DataType::Int16 => Box::new(Int16Builder::new()),
        DataType::Int32 => Box::new(Int32Builder::new()),
        DataType::Int64 => Box::new(Int64Builder::new()),
        DataType::UInt8 => Box::new(UInt8Builder::new()),
        DataType::UInt16 => Box::new(UInt16Builder::new()),
        DataType::UInt32 => Box::new(UInt32Builder::new()),
        DataType::UInt64 => Box::new(UInt64Builder::new()),
        DataType::Float32 => Box::new(Float32Builder::new()),
        DataType::Float64 => Box::new(Float64Builder::new()),
        DataType::Boolean => Box::new(BooleanBuilder::new()),
        DataType::Utf8 => Box::new(StringBuilder::new()),
        DataType::LargeUtf8 => Box::new(LargeStringBuilder::new()),
        DataType::Binary => Box::new(BinaryBuilder::new()),
        DataType::LargeBinary => Box::new(LargeBinaryBuilder::new()),
        DataType::FixedSizeBinary(size) => Box::new(FixedSizeBinaryBuilder::new(*size)),
        DataType::Timestamp(TimeUnit::Second, None) => {
            Box::new(TimestampSecondBuilder::new().with_data_type(data_type.clone()))
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            Box::new(TimestampMillisecondBuilder::new().with_data_type(data_type.clone()))
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            Box::new(TimestampMicrosecondBuilder::new().with_data_type(data_type.clone()))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            Box::new(TimestampNanosecondBuilder::new().with_data_type(data_type.clone()))
        }
        DataType::Decimal128(precision, scale) => Box::new(
            Decimal128Builder::new().with_data_type(DataType::Decimal128(*precision, *scale)),
        ),
        DataType::Decimal256(precision, scale) => Box::new(
            Decimal256Builder::new().with_data_type(DataType::Decimal256(*precision, *scale)),
        ),
        DataType::List(field) | DataType::LargeList(field) => {
            let inner = make_builder(field.data_type());
            Box::new(ListBuilder::new(inner).with_field(Arc::clone(field)))
        }
        DataType::Struct(fields) => {
            let field_builders: Vec<Box<dyn ArrayBuilder>> =
                fields.iter().map(|f| make_builder(f.data_type())).collect();
            Box::new(StructBuilder::new(fields.clone(), field_builders))
        }
        DataType::Map(map_field, _sorted) => {
            let entry_fields = match map_field.data_type() {
                DataType::Struct(fields) => fields,
                _ => unreachable!("map field must be a struct"),
            };
            let key_field = &entry_fields[0];
            let value_field = &entry_fields[1];
            let key_builder = make_builder(key_field.data_type());
            let value_builder = make_builder(value_field.data_type());
            Box::new(
                MapBuilder::new(None, key_builder, value_builder)
                    .with_keys_field(Arc::clone(key_field))
                    .with_values_field(Arc::clone(value_field)),
            )
        }
        other => panic!("unsupported builder for {}", other),
    }
}

fn append_null(builder: &mut dyn ArrayBuilder) {
    macro_rules! null_for {
        ($t:ty) => {
            if let Some(b) = builder.as_any_mut().downcast_mut::<$t>() {
                b.append_null();
                return;
            }
        };
    }
    null_for!(Int8Builder);
    null_for!(Int16Builder);
    null_for!(Int32Builder);
    null_for!(Int64Builder);
    null_for!(UInt8Builder);
    null_for!(UInt16Builder);
    null_for!(UInt32Builder);
    null_for!(UInt64Builder);
    null_for!(Float32Builder);
    null_for!(Float64Builder);
    null_for!(BooleanBuilder);
    null_for!(StringBuilder);
    null_for!(LargeStringBuilder);
    null_for!(BinaryBuilder);
    null_for!(LargeBinaryBuilder);
    null_for!(FixedSizeBinaryBuilder);
    null_for!(TimestampSecondBuilder);
    null_for!(TimestampMillisecondBuilder);
    null_for!(TimestampMicrosecondBuilder);
    null_for!(TimestampNanosecondBuilder);
    null_for!(Decimal128Builder);
    null_for!(Decimal256Builder);
    null_for!(ListBuilder<Box<dyn ArrayBuilder>>);
    null_for!(StructBuilder);
    if let Some(b) = builder
        .as_any_mut()
        .downcast_mut::<MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>>()
    {
        b.append(false).unwrap();
        return;
    }
    panic!("append_null not implemented for builder");
}

fn append_value(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
    value: Option<String>,
    line: usize,
    column: usize,
    raw: &str,
) -> Result<()> {
    if value.is_none() {
        append_null(builder);
        return Ok(());
    }
    let text = value.unwrap();
    match data_type {
        DataType::Int8 => {
            let v = text.parse::<i8>().map_err(|e| {
                tsv_error(line, column, raw, format!("invalid int8 '{}': {}", text, e))
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<Int8Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Int16 => {
            let v = text.parse::<i16>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid int16 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<Int16Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Int32 => {
            let v = text.parse::<i32>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid int32 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<Int32Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Int64 => {
            let v = text.parse::<i64>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid int64 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<Int64Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt8 => {
            let v = text.parse::<u8>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid uint8 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<UInt8Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt16 => {
            let v = text.parse::<u16>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid uint16 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<UInt16Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt32 => {
            let v = text.parse::<u32>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid uint32 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<UInt32Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt64 => {
            let v = text.parse::<u64>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid uint64 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<UInt64Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Float32 => {
            let v = text.parse::<f32>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid float32 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<Float32Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Float64 => {
            let v = text.parse::<f64>().map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid float64 '{}': {}", text, e),
                )
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<Float64Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Boolean => {
            let v = parse_bool(&text).map_err(|e| {
                tsv_error(line, column, raw, format!("invalid bool '{}': {}", text, e))
            })?;
            builder
                .as_any_mut()
                .downcast_mut::<BooleanBuilder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Utf8 => {
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(&text);
        }
        DataType::LargeUtf8 => {
            builder
                .as_any_mut()
                .downcast_mut::<LargeStringBuilder>()
                .unwrap()
                .append_value(&text);
        }
        DataType::Timestamp(unit, None) => {
            let micros = parse_timestamp(&text, *unit).map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid timestamp '{}': {}", text, e),
                )
            })?;
            match unit {
                TimeUnit::Second => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampSecondBuilder>()
                    .unwrap()
                    .append_value(micros),
                TimeUnit::Millisecond => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampMillisecondBuilder>()
                    .unwrap()
                    .append_value(micros),
                TimeUnit::Microsecond => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampMicrosecondBuilder>()
                    .unwrap()
                    .append_value(micros),
                TimeUnit::Nanosecond => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampNanosecondBuilder>()
                    .unwrap()
                    .append_value(micros),
            };
        }
        DataType::Binary
        | DataType::LargeBinary
        | DataType::FixedSizeBinary(_)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)
        | DataType::List(_)
        | DataType::LargeList(_)
        | DataType::Struct(_)
        | DataType::Map(_, _) => {
            let json: serde_json::Value = serde_json::from_str(&text).map_err(|e| {
                tsv_error(
                    line,
                    column,
                    raw,
                    format!("invalid JSON for {} '{}': {}", data_type, text, e),
                )
            })?;
            append_json_value(builder, data_type, json, line, column, raw)?;
        }
        other => {
            return Err(IcefallDBError::TypeNotSupported(format!(
                "TSV deserialization not supported for {}",
                other
            )))
        }
    }
    Ok(())
}

fn parse_bool(text: &str) -> std::result::Result<bool, String> {
    match text.to_lowercase().as_str() {
        "true" | "t" | "1" | "yes" | "y" => Ok(true),
        "false" | "f" | "0" | "no" | "n" => Ok(false),
        _ => Err(format!("expected true/false, got {}", text)),
    }
}

fn parse_timestamp(text: &str, unit: TimeUnit) -> std::result::Result<i64, String> {
    use chrono::{DateTime, NaiveDateTime};
    let dt = text
        .parse::<DateTime<chrono::Utc>>()
        .ok()
        .or_else(|| {
            NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f")
                .ok()
                .map(|naive| naive.and_utc())
        })
        .or_else(|| {
            NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S%.f")
                .ok()
                .map(|naive| naive.and_utc())
        })
        .ok_or_else(|| format!("expected RFC 3339 timestamp, got {}", text))?;
    let micros = dt.timestamp_micros();
    match unit {
        TimeUnit::Second => Ok(micros / 1_000_000),
        TimeUnit::Millisecond => Ok(micros / 1_000),
        TimeUnit::Microsecond => Ok(micros),
        TimeUnit::Nanosecond => Ok(micros * 1_000),
    }
}

fn append_json_value(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
    value: serde_json::Value,
    line: usize,
    column: usize,
    raw: &str,
) -> Result<()> {
    if value.is_null() {
        append_null(builder);
        return Ok(());
    }
    match data_type {
        DataType::Int8 => {
            let v = value
                .as_i64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected int8"))?
                as i8;
            builder
                .as_any_mut()
                .downcast_mut::<Int8Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Int16 => {
            let v = value
                .as_i64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected int16"))?
                as i16;
            builder
                .as_any_mut()
                .downcast_mut::<Int16Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Int32 => {
            let v = value
                .as_i64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected int32"))?
                as i32;
            builder
                .as_any_mut()
                .downcast_mut::<Int32Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Int64 => {
            let v = value
                .as_i64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected int64"))?;
            builder
                .as_any_mut()
                .downcast_mut::<Int64Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt8 => {
            let v = value
                .as_u64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected uint8"))?
                as u8;
            builder
                .as_any_mut()
                .downcast_mut::<UInt8Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt16 => {
            let v = value
                .as_u64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected uint16"))?
                as u16;
            builder
                .as_any_mut()
                .downcast_mut::<UInt16Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt32 => {
            let v = value
                .as_u64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected uint32"))?
                as u32;
            builder
                .as_any_mut()
                .downcast_mut::<UInt32Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::UInt64 => {
            let v = value
                .as_u64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected uint64"))?;
            builder
                .as_any_mut()
                .downcast_mut::<UInt64Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Float32 => {
            let v = value
                .as_f64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected float32"))?
                as f32;
            builder
                .as_any_mut()
                .downcast_mut::<Float32Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Float64 => {
            let v = value
                .as_f64()
                .ok_or_else(|| tsv_error(line, column, raw, "expected float64"))?;
            builder
                .as_any_mut()
                .downcast_mut::<Float64Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Boolean => {
            let v = value
                .as_bool()
                .ok_or_else(|| tsv_error(line, column, raw, "expected bool"))?;
            builder
                .as_any_mut()
                .downcast_mut::<BooleanBuilder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Utf8 => {
            let v = value
                .as_str()
                .ok_or_else(|| tsv_error(line, column, raw, "expected string"))?;
            builder
                .as_any_mut()
                .downcast_mut::<StringBuilder>()
                .unwrap()
                .append_value(v);
        }
        DataType::LargeUtf8 => {
            let v = value
                .as_str()
                .ok_or_else(|| tsv_error(line, column, raw, "expected string"))?;
            builder
                .as_any_mut()
                .downcast_mut::<LargeStringBuilder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Timestamp(unit, None) => {
            let s = value
                .as_str()
                .ok_or_else(|| tsv_error(line, column, raw, "expected timestamp string"))?;
            let micros = parse_timestamp(s, *unit)
                .map_err(|e| tsv_error(line, column, raw, format!("invalid timestamp: {}", e)))?;
            match unit {
                TimeUnit::Second => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampSecondBuilder>()
                    .unwrap()
                    .append_value(micros),
                TimeUnit::Millisecond => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampMillisecondBuilder>()
                    .unwrap()
                    .append_value(micros),
                TimeUnit::Microsecond => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampMicrosecondBuilder>()
                    .unwrap()
                    .append_value(micros),
                TimeUnit::Nanosecond => builder
                    .as_any_mut()
                    .downcast_mut::<TimestampNanosecondBuilder>()
                    .unwrap()
                    .append_value(micros),
            };
        }
        DataType::Binary | DataType::LargeBinary | DataType::FixedSizeBinary(_) => {
            let s = value
                .as_str()
                .ok_or_else(|| tsv_error(line, column, raw, "expected base64 string"))?;
            let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, s)
                .map_err(|e| tsv_error(line, column, raw, format!("invalid base64: {}", e)))?;
            match data_type {
                DataType::Binary => builder
                    .as_any_mut()
                    .downcast_mut::<BinaryBuilder>()
                    .unwrap()
                    .append_value(&bytes),
                DataType::LargeBinary => builder
                    .as_any_mut()
                    .downcast_mut::<LargeBinaryBuilder>()
                    .unwrap()
                    .append_value(&bytes),
                DataType::FixedSizeBinary(size) => {
                    if bytes.len() != *size as usize {
                        return Err(tsv_error(
                            line,
                            column,
                            raw,
                            format!(
                                "fixed_size_binary expected {} bytes, got {}",
                                size,
                                bytes.len()
                            ),
                        ));
                    }
                    let _ = builder
                        .as_any_mut()
                        .downcast_mut::<FixedSizeBinaryBuilder>()
                        .unwrap()
                        .append_value(&bytes);
                }
                _ => unreachable!(),
            }
        }
        DataType::Decimal128(precision, scale) => {
            let v = parse_decimal_json(&value, *precision, *scale)
                .ok_or_else(|| tsv_error(line, column, raw, "expected decimal"))?;
            builder
                .as_any_mut()
                .downcast_mut::<Decimal128Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::Decimal256(precision, scale) => {
            let v = parse_decimal256_json(&value, *precision, *scale)
                .ok_or_else(|| tsv_error(line, column, raw, "expected decimal256"))?;
            builder
                .as_any_mut()
                .downcast_mut::<Decimal256Builder>()
                .unwrap()
                .append_value(v);
        }
        DataType::List(field) | DataType::LargeList(field) => {
            let arr = value
                .as_array()
                .ok_or_else(|| tsv_error(line, column, raw, "expected list array"))?;
            let list_builder = builder
                .as_any_mut()
                .downcast_mut::<ListBuilder<Box<dyn ArrayBuilder>>>()
                .unwrap();
            for item in arr {
                append_json_value(
                    list_builder.values(),
                    field.data_type(),
                    item.clone(),
                    line,
                    column,
                    raw,
                )?;
            }
            list_builder.append(true);
        }
        DataType::Struct(fields) => {
            let obj = value
                .as_object()
                .ok_or_else(|| tsv_error(line, column, raw, "expected struct object"))?;
            let struct_builder = builder
                .as_any_mut()
                .downcast_mut::<StructBuilder>()
                .unwrap();
            for (i, field) in fields.iter().enumerate() {
                let field_value = obj
                    .get(field.name())
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                let field_builder = &mut struct_builder.field_builders_mut()[i];
                append_json_value(
                    field_builder.as_mut(),
                    field.data_type(),
                    field_value,
                    line,
                    column,
                    raw,
                )?;
            }
            struct_builder.append(true);
        }
        DataType::Map(map_field, _sorted) => {
            let obj = value
                .as_object()
                .ok_or_else(|| tsv_error(line, column, raw, "expected map object"))?;
            let map_builder = builder
                .as_any_mut()
                .downcast_mut::<MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>>()
                .unwrap();
            let entry_fields = match map_field.data_type() {
                DataType::Struct(fields) => fields,
                _ => unreachable!("map field must be a struct"),
            };
            let key_type = entry_fields[0].data_type();
            let value_type = entry_fields[1].data_type();
            for (k, v) in obj {
                append_json_value(
                    map_builder.keys(),
                    key_type,
                    serde_json::Value::String(k.clone()),
                    line,
                    column,
                    raw,
                )?;
                append_json_value(
                    map_builder.values(),
                    value_type,
                    v.clone(),
                    line,
                    column,
                    raw,
                )?;
            }
            map_builder.append(true).unwrap();
        }
        other => {
            return Err(IcefallDBError::TypeNotSupported(format!(
                "JSON deserialization not supported for {}",
                other
            )))
        }
    }
    Ok(())
}

fn parse_decimal_json(value: &serde_json::Value, _precision: u8, scale: i8) -> Option<i128> {
    match value {
        serde_json::Value::Number(n) => n.as_f64().map(|f| (f * 10f64.powi(scale as i32)) as i128),
        serde_json::Value::String(s) => parse_decimal_str(s, scale),
        _ => None,
    }
}

fn parse_decimal256_json(
    value: &serde_json::Value,
    _precision: u8,
    scale: i8,
) -> Option<arrow::datatypes::i256> {
    parse_decimal_json(value, 0, scale).map(arrow::datatypes::i256::from_i128)
}

fn parse_decimal_str(s: &str, scale: i8) -> Option<i128> {
    let scale = scale.max(0) as u32;
    let negative = s.starts_with('-');
    let s = s.trim_start_matches('-');
    let (int_part, frac_part) = if let Some(pos) = s.find('.') {
        (&s[..pos], &s[pos + 1..])
    } else {
        (s, "")
    };
    let frac_part = format!("{:0<width$}", frac_part, width = scale as usize);
    let frac_part = &frac_part[..(scale as usize).min(frac_part.len())];
    let combined = format!("{}{}", int_part, frac_part);
    let unscaled: i128 = combined.parse().ok()?;
    Some(if negative { -unscaled } else { unscaled })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int64Array;
    use arrow::datatypes::Schema as ArrowSchema;

    #[test]
    fn test_split_tsv_line() {
        let line = b"a\\tb\tc\td";
        let cells = split_tsv_line(line).unwrap();
        assert_eq!(cells.len(), 3);
        assert_eq!(cells[0], b"a\\tb");
        assert_eq!(cells[1], b"c");
        assert_eq!(cells[2], b"d");
    }

    #[test]
    fn test_tsv_roundtrip() {
        let schema = Schema {
            schema_id: 1,
            columns: vec![
                crate::metadata::Column::new("id", "int64", false),
                crate::metadata::Column::new("name", "utf8", true),
            ],
            partition_by: None,
            sort: None,
            agg_group_keys: None,
            row_group_target_rows: 1000,
            row_group_target_bytes: 1024 * 1024,
            dropped_columns: vec![],
            max_field_id: 0,
        };
        let arrow_schema = ArrowSchema::new(vec![
            arrow::datatypes::Field::new("id", DataType::Int64, false),
            arrow::datatypes::Field::new("name", DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(arrow_schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2])),
                Arc::new(arrow::array::StringArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        let tsv = TsvEncoder::encode(&batch);
        let decoded = TsvDecoder::decode(&tsv, &schema).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].num_rows(), 2);
    }
}
