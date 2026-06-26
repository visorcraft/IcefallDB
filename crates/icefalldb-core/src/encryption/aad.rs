//! AAD prefix derivation.
//!
//! The AAD prefix binds an encrypted Parquet file to a specific table and
//! schema id, preventing file-swap attacks: two files encrypted with the same
//! key but for different tables cannot be substituted for each other because
//! the GCM authentication tag will fail to verify.
//!
//! The prefix is stored in the file (via `with_aad_prefix_storage(true)`) so
//! readers do not need to recompute it — they only need it for *verification*.

use crate::encryption::keys::KeyIdentifier;

/// Derive a deterministic AAD prefix for a table.
///
/// The prefix encodes the table name and schema id so that:
/// - Swapping files between tables fails authentication.
/// - Bumping the schema id invalidates old files (forcing the reader to
///   re-acknowledge the new schema).
///
/// Format: `icefalldb:v1:<table>:<schema_id>` (ASCII, no trailing newline).
/// This is *not* secret — it appears in plaintext in the Parquet file footer.
pub fn table_aad_prefix(table: &str, schema_id: u64) -> Vec<u8> {
    format!("icefalldb:v1:{table}:{schema_id}").into_bytes()
}

/// Derive an AAD prefix from a table name and a `KeyIdentifier`.
///
/// Used when the schema id is not known at prefix-derivation time (e.g. the
/// reader has only the key identifier and the file path). The `KeyIdentifier`
/// is expected to encode the schema id in these deployments.
pub fn table_aad_prefix_from_id(table: &str, kid: &KeyIdentifier) -> Vec<u8> {
    format!("icefalldb:v1:{table}:{}", kid.0).into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aad_prefix_is_deterministic() {
        let a = table_aad_prefix("events", 3);
        let b = table_aad_prefix("events", 3);
        assert_eq!(a, b);
        assert_eq!(a, b"icefalldb:v1:events:3".to_vec());
    }

    #[test]
    fn aad_prefix_differs_per_table() {
        assert_ne!(table_aad_prefix("events", 1), table_aad_prefix("users", 1));
    }

    #[test]
    fn aad_prefix_differs_per_schema_id() {
        assert_ne!(table_aad_prefix("events", 1), table_aad_prefix("events", 2));
    }

    #[test]
    fn aad_prefix_from_id_matches_when_id_encodes_schema() {
        let kid = KeyIdentifier::new("3");
        let from_id = table_aad_prefix_from_id("events", &kid);
        let direct = table_aad_prefix("events", 3);
        assert_eq!(from_id, direct);
    }
}
