//! Lossless conversion between JSON sidecar statistics and Arrow `ScalarValue`.
//!
//! The public direction (`json_to_scalar_value`) is used by the statistics builder
//! to turn per-row-group min/max values into DataFusion `ScalarValue`s.

use arrow::datatypes::{DataType, TimeUnit};
use base64::engine::{general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use datafusion::common::ScalarValue;
use serde_json::Value;

/// Convert a JSON value from a sidecar statistic into an Arrow `ScalarValue`.
///
/// Returns `None` when the type is unsupported or the value cannot be
/// reconstructed exactly (e.g. out-of-range integer, malformed base64). Callers
/// treat `None` as an absent statistic.
pub fn json_to_scalar_value(
    value: &serde_json::Value,
    data_type: &DataType,
) -> Option<ScalarValue> {
    match data_type {
        DataType::Int8 => to_i64(value)
            .filter(|&v| v >= i8::MIN as i64 && v <= i8::MAX as i64)
            .map(|v| ScalarValue::Int8(Some(v as i8))),
        DataType::Int16 => to_i64(value)
            .filter(|&v| v >= i16::MIN as i64 && v <= i16::MAX as i64)
            .map(|v| ScalarValue::Int16(Some(v as i16))),
        DataType::Int32 => to_i64(value)
            .filter(|&v| v >= i32::MIN as i64 && v <= i32::MAX as i64)
            .map(|v| ScalarValue::Int32(Some(v as i32))),
        DataType::Int64 => to_i64(value).map(|v| ScalarValue::Int64(Some(v))),

        DataType::UInt8 => to_u64(value)
            .filter(|&v| v <= u8::MAX as u64)
            .map(|v| ScalarValue::UInt8(Some(v as u8))),
        DataType::UInt16 => to_u64(value)
            .filter(|&v| v <= u16::MAX as u64)
            .map(|v| ScalarValue::UInt16(Some(v as u16))),
        DataType::UInt32 => to_u64(value)
            .filter(|&v| v <= u32::MAX as u64)
            .map(|v| ScalarValue::UInt32(Some(v as u32))),
        DataType::UInt64 => to_u64(value).map(|v| ScalarValue::UInt64(Some(v))),

        DataType::Float32 => parse_f32(value).map(|v| ScalarValue::Float32(Some(v))),
        DataType::Float64 => parse_f64(value).map(|v| ScalarValue::Float64(Some(v))),

        DataType::Decimal128(precision, scale) => parse_decimal(value, *precision, *scale)
            .map(|v| ScalarValue::Decimal128(Some(v), *precision, *scale)),

        DataType::Timestamp(TimeUnit::Second, tz) => {
            to_i64(value).map(|v| ScalarValue::TimestampSecond(Some(v), tz.clone()))
        }
        DataType::Timestamp(TimeUnit::Millisecond, tz) => {
            to_i64(value).map(|v| ScalarValue::TimestampMillisecond(Some(v), tz.clone()))
        }
        DataType::Timestamp(TimeUnit::Microsecond, tz) => {
            to_i64(value).map(|v| ScalarValue::TimestampMicrosecond(Some(v), tz.clone()))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
            to_i64(value).map(|v| ScalarValue::TimestampNanosecond(Some(v), tz.clone()))
        }

        DataType::Date32 => to_i64(value)
            .filter(|&v| v >= i32::MIN as i64 && v <= i32::MAX as i64)
            .map(|v| ScalarValue::Date32(Some(v as i32))),
        DataType::Date64 => to_i64(value).map(|v| ScalarValue::Date64(Some(v))),

        DataType::Time32(TimeUnit::Second) => to_i64(value)
            .filter(|&v| v >= i32::MIN as i64 && v <= i32::MAX as i64)
            .map(|v| ScalarValue::Time32Second(Some(v as i32))),
        DataType::Time32(TimeUnit::Millisecond) => to_i64(value)
            .filter(|&v| v >= i32::MIN as i64 && v <= i32::MAX as i64)
            .map(|v| ScalarValue::Time32Millisecond(Some(v as i32))),
        DataType::Time64(TimeUnit::Microsecond) => {
            to_i64(value).map(|v| ScalarValue::Time64Microsecond(Some(v)))
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            to_i64(value).map(|v| ScalarValue::Time64Nanosecond(Some(v)))
        }

        DataType::Utf8 => value
            .as_str()
            .map(|s| ScalarValue::Utf8(Some(s.to_string()))),
        DataType::LargeUtf8 => value
            .as_str()
            .map(|s| ScalarValue::LargeUtf8(Some(s.to_string()))),

        DataType::Binary => parse_binary(value).map(|b| ScalarValue::Binary(Some(b))),
        DataType::LargeBinary => parse_binary(value).map(|b| ScalarValue::LargeBinary(Some(b))),
        DataType::FixedSizeBinary(size) => parse_binary(value)
            .filter(|b| b.len() == *size as usize)
            .map(|b| ScalarValue::FixedSizeBinary(*size, Some(b))),

        DataType::Boolean => value.as_bool().map(|b| ScalarValue::Boolean(Some(b))),

        _ => None,
    }
}

fn to_i64(value: &Value) -> Option<i64> {
    match value {
        Value::Number(n) => n.as_i64().or_else(|| {
            // A unsigned value that fits in i64 (e.g. u64 <= i64::MAX).
            n.as_u64()
                .filter(|&v| v <= i64::MAX as u64)
                .map(|v| v as i64)
        }),
        _ => None,
    }
}

fn to_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64().or_else(|| {
            // A non-negative signed value.
            n.as_i64().filter(|&v| v >= 0).map(|v| v as u64)
        }),
        _ => None,
    }
}

fn parse_f32(value: &Value) -> Option<f32> {
    match value {
        Value::Number(n) => n.as_f64().map(|v| v as f32),
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f32::NAN),
            "Infinity" => Some(f32::INFINITY),
            "-Infinity" => Some(f32::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

fn parse_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => match s.as_str() {
            "NaN" => Some(f64::NAN),
            "Infinity" => Some(f64::INFINITY),
            "-Infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        },
        _ => None,
    }
}

/// Parse a JSON value as an unscaled i128 decimal with the given scale.
fn parse_decimal(value: &Value, _precision: u8, scale: i8) -> Option<i128> {
    match value {
        // A JSON number is interpreted as the already-unscaled integer value.
        Value::Number(n) => n
            .as_i64()
            .map(|v| v as i128)
            .or_else(|| n.as_u64().map(|v| v as i128)),
        Value::String(s) => parse_decimal_str(s, scale),
        _ => None,
    }
}

fn parse_decimal_str(s: &str, scale: i8) -> Option<i128> {
    let s = s.trim();
    let (negative, digits) = s
        .strip_prefix('-')
        .map(|rest| (true, rest))
        .unwrap_or((false, s));

    let scale = scale.max(0) as usize;

    // If the string contains a decimal point, interpret it as a human-readable
    // decimal and scale it. Otherwise the string is already the unscaled value.
    let unscaled_str = if let Some((int_part, frac_part)) = digits.split_once('.') {
        let frac_part = frac_part.trim_end_matches('0');
        if frac_part.len() > scale {
            return None;
        }
        let mut out = int_part.to_string();
        out.push_str(frac_part);
        let missing = scale - frac_part.len();
        out.push_str(&"0".repeat(missing));
        out
    } else {
        digits.to_string()
    };

    let mut value = unscaled_str.parse::<i128>().ok()?;
    if negative {
        value = value.checked_neg()?;
    }
    Some(value)
}

fn parse_binary(value: &Value) -> Option<Vec<u8>> {
    value.as_str().and_then(|s| BASE64_STANDARD.decode(s).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int64_value() {
        let value = Value::from(i64::MIN);
        let scalar = json_to_scalar_value(&value, &DataType::Int64).unwrap();
        assert_eq!(scalar, ScalarValue::Int64(Some(i64::MIN)));
    }

    #[test]
    fn test_uint64_above_2_53() {
        let v = (1u64 << 60) + 123;
        let value = Value::from(v);
        let scalar = json_to_scalar_value(&value, &DataType::UInt64).unwrap();
        assert_eq!(scalar, ScalarValue::UInt64(Some(v)));
    }

    #[test]
    fn test_float_special_values() {
        for (json, expected) in [
            (Value::String("NaN".into()), f64::NAN),
            (Value::String("Infinity".into()), f64::INFINITY),
            (Value::String("-Infinity".into()), f64::NEG_INFINITY),
        ] {
            let scalar = json_to_scalar_value(&json, &DataType::Float64).unwrap();
            let ScalarValue::Float64(Some(actual)) = scalar else {
                panic!("expected Float64 scalar");
            };
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, expected);
            }
        }

        let neg_zero = serde_json::Number::from_f64(-0.0).unwrap();
        let scalar =
            json_to_scalar_value(&Value::Number(neg_zero.clone()), &DataType::Float64).unwrap();
        let ScalarValue::Float64(Some(actual)) = scalar else {
            panic!("expected Float64 scalar");
        };
        assert_eq!(actual, -0.0);
        assert!(actual.is_sign_negative());
    }

    #[test]
    fn test_decimal_values() {
        let scalar = ScalarValue::Decimal128(Some(12345), 10, 2);
        let back =
            json_to_scalar_value(&Value::String("12345".into()), &DataType::Decimal128(10, 2))
                .unwrap();
        assert_eq!(back, scalar);

        // String form including the decimal point.
        let back2 = json_to_scalar_value(
            &Value::String("123.45".into()),
            &DataType::Decimal128(10, 2),
        )
        .unwrap();
        assert_eq!(back2, scalar);
    }

    #[test]
    fn test_timestamp_units() {
        let ts_us = Value::from(1_700_000_000_000_000i64);
        let scalar =
            json_to_scalar_value(&ts_us, &DataType::Timestamp(TimeUnit::Microsecond, None))
                .unwrap();
        assert_eq!(
            scalar,
            ScalarValue::TimestampMicrosecond(Some(1_700_000_000_000_000), None)
        );

        let ts_ns = Value::from(1_700_000_000_000_000_123i64);
        let scalar =
            json_to_scalar_value(&ts_ns, &DataType::Timestamp(TimeUnit::Nanosecond, None)).unwrap();
        assert_eq!(
            scalar,
            ScalarValue::TimestampNanosecond(Some(1_700_000_000_000_000_123), None)
        );
    }

    #[test]
    fn test_binary_value() {
        let bytes = vec![0u8, 1, 2, 255, 254];
        let encoded = BASE64_STANDARD.encode(&bytes);
        let scalar =
            json_to_scalar_value(&Value::String(encoded.clone()), &DataType::Binary).unwrap();
        assert_eq!(scalar, ScalarValue::Binary(Some(bytes.clone())));
    }

    #[test]
    fn test_utf8_value() {
        let json = Value::String("hello \n world".into());
        let scalar = json_to_scalar_value(&json, &DataType::Utf8).unwrap();
        assert_eq!(scalar, ScalarValue::Utf8(Some("hello \n world".into())));
    }

    #[test]
    fn test_fixed_size_binary_length_mismatch() {
        let bytes = vec![0u8, 1, 2];
        let encoded = BASE64_STANDARD.encode(&bytes);
        assert!(
            json_to_scalar_value(&Value::String(encoded), &DataType::FixedSizeBinary(4)).is_none()
        );
    }

    #[test]
    fn test_out_of_range_returns_none() {
        assert!(
            json_to_scalar_value(&Value::from(i32::MAX as i64 + 1), &DataType::Int32).is_none()
        );
        assert!(json_to_scalar_value(&Value::from(-1i64), &DataType::UInt32).is_none());
        assert!(json_to_scalar_value(&Value::from(256i64), &DataType::UInt8).is_none());
    }
}
