use crate::metadata::checksum::checksum_json;
use crate::{IcefallDBError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Reference to a row-index file (base snapshot + incremental deltas).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RowIndexRef {
    /// Path to the base row-index snapshot file, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    /// Ordered list of delta files applied on top of `base`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deltas: Vec<String>,
}

/// Reference to a secondary-index file (base snapshot + incremental deltas).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct IndexRef {
    /// Path to the base index snapshot file, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base: Option<String>,
    /// Ordered list of delta files applied on top of `base`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deltas: Vec<String>,
}

/// Reference to a row group data file and its companion metadata file.
///
/// Both `data` and `meta` are relative paths within the table directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct RowGroupEntry {
    /// Relative path to the row group data file (e.g. Parquet).
    pub data: String,
    /// Relative path to the row group metadata file.
    pub meta: String,
    /// Stable fragment identifier for this row group (monotonically increasing).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub fragment_id: u64,
    /// Relative path to the deletion vector file for this row group, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletes: Option<String>,
    /// Number of logically deleted rows in this row group.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub deleted_count: u64,
    /// Relative path to the additive-aggregate sidecar file for this row group,
    /// if one has been computed.  Absent for legacy fragments or
    /// on write paths that do not yet produce `.agg` files; the metadata-aggregate rule falls
    /// back to a full scan for any fragment that lacks this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agg: Option<String>,
}

/// A manifest lists the row groups that make up a table snapshot.
///
/// Manifests are stored as JSON files under `_manifests/` within the table
/// directory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct Manifest {
    /// Format version of the manifest file.
    pub format_version: u64,
    /// Sequence number of this manifest within the table history.
    pub sequence: u64,
    /// Id of the schema that applies to the row groups in this manifest.
    pub schema_id: u64,
    /// Row groups contained in this manifest.
    pub row_groups: Vec<RowGroupEntry>,
    /// Optional denormalized row counts, parallel to `row_groups`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_counts: Option<Vec<usize>>,
    /// Optional per-row-group partition values, keyed by row group data filename.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub partition_values: Option<HashMap<String, HashMap<String, Value>>>,
    /// Next row ID to assign when appending new rows.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub next_row_id: u64,
    /// Next fragment ID to assign when creating a new row group.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub next_fragment_id: u64,
    /// Generation reference for the `_rowindex` address map, if one exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rowindex_generation: Option<RowIndexRef>,
    /// Generation references for named secondary indexes.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub index_generations: HashMap<String, IndexRef>,
    /// Relative path to the snapshot checkpoint file for this manifest, if one
    /// was emitted.  Checkpoints are written atomically inside the commit and
    /// protected by `manifest_referenced_files`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<String>,
    /// Checksum of the immediately-preceding manifest (`<seq-1>.json`'s
    /// `checksum`), forming a hash chain. `None` for the genesis manifest and for
    /// manifests written before this field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_hash: Option<String>,
    /// RFC3339 UTC commit timestamp. `None` for pre-change manifests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub committed_at: Option<String>,
    /// Checksum of the manifest contents. Callers must clear this field before
    /// computing a self-checksum.
    pub checksum: String,
}

/// Returns `true` when the value is zero, used by `skip_serializing_if` to omit
/// default u64 fields from legacy manifests so their checksums stay stable.
#[inline]
fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

impl Manifest {
    /// Returns the relative path for a manifest with the given sequence number
    /// within the table directory.
    pub fn filename(seq: u64) -> String {
        format!("_manifests/{:09}.json", seq)
    }

    /// Computes a stable checksum of this manifest with the `checksum` field
    /// cleared.
    ///
    /// The returned checksum has the form `sha256:<hex>`.
    pub fn compute_checksum(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.checksum = String::new();
        let value = serde_json::to_value(&copy).map_err(IcefallDBError::Serialization)?;
        Ok(checksum_json(&value))
    }

    /// Verifies that the stored checksum matches the recomputed checksum.
    pub fn verify_checksum(&self) -> Result<bool> {
        Ok(self.checksum == Self::compute_checksum(self)?)
    }

    /// Set the chain link + commit time, then (re)compute the self-checksum so it
    /// covers both new fields. Call instead of assigning `checksum` directly.
    pub fn finalize(
        &mut self,
        parent_checksum: Option<String>,
        committed_at: String,
    ) -> Result<()> {
        self.parent_hash = parent_checksum;
        self.committed_at = Some(committed_at);
        self.checksum = self.compute_checksum()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrips_and_loads_legacy() {
        // legacy JSON with none of the new fields must deserialize with defaults
        let legacy = r#"{"format_version":1,"sequence":1,"schema_id":1,"row_groups":[{"data":"rg_a.parquet","meta":"rg_a.meta"}],"checksum":""}"#;
        let m: Manifest = serde_json::from_str(legacy).unwrap();
        assert_eq!(m.next_row_id, 0);
        assert_eq!(m.next_fragment_id, 0);
        assert!(m.rowindex_generation.is_none());
        assert!(m.index_generations.is_empty());
        assert_eq!(m.row_groups[0].fragment_id, 0);
        assert_eq!(m.row_groups[0].deleted_count, 0);
        assert!(m.row_groups[0].agg.is_none());
        assert!(m.checkpoint.is_none());
        // new fields round-trip
        let mut m2 = m.clone();
        m2.next_row_id = 42;
        m2.row_groups[0].fragment_id = 7;
        m2.row_groups[0].deletes = Some("_deletions/rg_a__v2.del".into());
        let s = serde_json::to_string(&m2).unwrap();
        assert_eq!(serde_json::from_str::<Manifest>(&s).unwrap(), m2);
    }

    /// The `agg` field must be absent from serialized JSON when `None`, so that
    /// legacy manifests keep a stable checksum.
    #[test]
    fn absent_agg_field_does_not_change_manifest_checksum() {
        // Build a manifest with no agg fields.
        let m_no_agg = Manifest {
            format_version: 1,
            sequence: 1,
            schema_id: 1,
            row_groups: vec![RowGroupEntry {
                data: "rg_a.parquet".into(),
                meta: "rg_a.meta".into(),
                fragment_id: 3,
                agg: None,
                ..Default::default()
            }],
            checksum: String::new(),
            ..Default::default()
        };
        let json_no_agg = serde_json::to_string(&m_no_agg).unwrap();
        // "agg" key must not appear in the JSON when the field is None.
        assert!(
            !json_no_agg.contains("\"agg\""),
            "agg key must be absent when None; got: {json_no_agg}"
        );

        // A manifest with an agg field set must differ.
        let mut m_with_agg = m_no_agg.clone();
        m_with_agg.row_groups[0].agg = Some("rg_a.agg".into());
        let json_with_agg = serde_json::to_string(&m_with_agg).unwrap();
        assert!(json_with_agg.contains("\"agg\""));
        // Their checksums must differ.
        assert_ne!(
            m_no_agg.compute_checksum().unwrap(),
            m_with_agg.compute_checksum().unwrap(),
            "manifest checksum must change when agg field is added"
        );
    }

    #[test]
    fn test_finalize_sets_chain_and_checksum() {
        let mut m1 = Manifest {
            format_version: 1,
            sequence: 1,
            ..Default::default()
        };
        m1.finalize(None, "2026-06-25T00:00:00+00:00".into())
            .unwrap();
        assert!(m1.parent_hash.is_none());
        assert_eq!(
            m1.committed_at.as_deref(),
            Some("2026-06-25T00:00:00+00:00")
        );
        assert!(m1.verify_checksum().unwrap());

        let mut m2 = Manifest {
            format_version: 1,
            sequence: 2,
            ..Default::default()
        };
        m2.finalize(
            Some(m1.checksum.clone()),
            "2026-06-25T00:00:01+00:00".into(),
        )
        .unwrap();
        assert_eq!(m2.parent_hash.as_deref(), Some(m1.checksum.as_str()));
        assert!(m2.verify_checksum().unwrap());
        // The chain link is the parent's self-checksum:
        assert_eq!(m2.parent_hash.as_ref().unwrap(), &m1.checksum);
    }

    #[test]
    fn test_legacy_manifest_without_new_fields_still_verifies() {
        // A manifest serialized without the new fields (defaults None) round-trips
        // and its checksum does not depend on the new fields being present.
        let mut m = Manifest {
            format_version: 1,
            sequence: 1,
            ..Default::default()
        };
        m.checksum = m.compute_checksum().unwrap();
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("parent_hash")); // skipped when None
        assert!(!json.contains("committed_at"));
        let back: Manifest = serde_json::from_str(&json).unwrap();
        assert!(back.verify_checksum().unwrap());
    }

    /// The `checkpoint` field must be absent from serialized JSON when `None`,
    /// so that legacy manifests keep a stable checksum.
    #[test]
    fn absent_checkpoint_field_does_not_change_manifest_checksum() {
        let m_no_cp = Manifest {
            format_version: 1,
            sequence: 1,
            schema_id: 1,
            row_groups: vec![RowGroupEntry {
                data: "rg_a.parquet".into(),
                meta: "rg_a.meta".into(),
                fragment_id: 3,
                ..Default::default()
            }],
            checkpoint: None,
            checksum: String::new(),
            ..Default::default()
        };
        let json_no_cp = serde_json::to_string(&m_no_cp).unwrap();
        assert!(
            !json_no_cp.contains("\"checkpoint\""),
            "checkpoint key must be absent when None; got: {json_no_cp}"
        );

        let mut m_with_cp = m_no_cp.clone();
        m_with_cp.checkpoint = Some("_checkpoints/000000001.json".into());
        let json_with_cp = serde_json::to_string(&m_with_cp).unwrap();
        assert!(json_with_cp.contains("\"checkpoint\""));
        assert_ne!(
            m_no_cp.compute_checksum().unwrap(),
            m_with_cp.compute_checksum().unwrap(),
            "manifest checksum must change when checkpoint field is added"
        );
    }
}
