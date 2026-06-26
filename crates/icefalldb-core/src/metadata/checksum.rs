use serde_json::Value;
use sha2::{Digest, Sha256};

/// Computes a SHA-256 checksum of a JSON value.
///
/// The value is canonicalized before hashing so that the checksum is stable
/// regardless of object key ordering or insignificant whitespace. The returned
/// string has the form `sha256:<hex>`.
pub fn checksum_json(value: &Value) -> String {
    let canonical = canonicalize(value);
    let json = serde_json::to_string(&canonical).expect("canonical JSON value serializes to JSON");
    checksum_bytes(json.as_bytes())
}

/// Computes a SHA-256 checksum of raw bytes.
///
/// The returned string has the form `sha256:<hex>`.
pub fn checksum_bytes(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("sha256:{}", hex::encode(hasher.finalize()))
}

/// Returns a canonical JSON form of `value`.
///
/// The canonical form:
/// - Sorts object keys lexicographically.
/// - Preserves array order.
/// - Leaves scalars unchanged.
///
/// The returned value can be serialized with `serde_json::to_string` to produce
/// unambiguous, valid JSON suitable for checksumming.
fn canonicalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut items: Vec<_> = map.iter().collect();
            items.sort_by(|a, b| a.0.cmp(b.0));
            let sorted: serde_json::Map<String, Value> = items
                .into_iter()
                .map(|(k, v)| (k.clone(), canonicalize(v)))
                .collect();
            Value::Object(sorted)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(canonicalize).collect()),
        other => other.clone(),
    }
}
