use crate::metadata::checksum::checksum_json;
use crate::metadata::{ColumnChunkOffset, ColumnStats};
use crate::rowid::RowIdSegment;
use crate::{IcefallDBError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[inline]
fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

#[inline]
fn is_zero_usize(v: &usize) -> bool {
    *v == 0
}

/// Denormalized summary of a single fragment, sufficient to build a scan plan
/// without re-reading the per-fragment `.meta` sidecar.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FragmentSummary {
    /// Row-group identifier (no extension). In the Parquet-dedupe fast path this
    /// equals the `.meta` filename stem, which may differ from the data file stem.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub row_group: String,
    /// Relative path to the fragment data file.
    pub data: String,
    /// Relative path to the fragment metadata sidecar.
    pub meta: String,
    /// Relative path to the additive aggregate sidecar, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agg: Option<String>,
    /// Relative path to the deletion-vector file, if present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deletes: Option<String>,
    /// Stable fragment identifier.
    pub fragment_id: u64,
    /// Number of live rows in the fragment.
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub rows: usize,
    /// Number of logically deleted rows.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub deleted_count: u64,
    /// Per-column statistics, keyed by column name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub columns: HashMap<String, ColumnStats>,
    /// Optional per-column byte offsets into the Parquet data file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub column_offsets: Option<HashMap<String, ColumnChunkOffset>>,
    /// Stable row-ID segments describing the rows stored in this fragment.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub row_ids: Vec<RowIdSegment>,
    /// Optional sort order for the fragment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sort: Option<Vec<String>>,
    /// SHA-256 checksum of the Parquet data file (matches `RowGroupMeta.checksum`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub checksum: String,
    /// SHA-256 checksum of the metadata fields (matches `RowGroupMeta.meta_checksum`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub meta_checksum: String,
}

/// Denormalized snapshot checkpoint: everything the scan planner needs from the
/// per-fragment sidecars, collapsed into a single immutable file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SnapshotCheckpoint {
    /// Manifest sequence this checkpoint describes.
    pub sequence: u64,
    /// Schema id that applies to the fragments in this checkpoint.
    pub schema_id: u64,
    /// Per-fragment summaries in manifest order.
    pub fragments: Vec<FragmentSummary>,
    /// SHA-256 checksum of the checkpoint contents. Callers must clear this
    /// field before computing a self-checksum.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub checksum: String,
}

impl SnapshotCheckpoint {
    /// Relative path for the checkpoint file with the given sequence number.
    pub fn filename(seq: u64) -> String {
        format!("_checkpoints/{:09}.json", seq)
    }

    /// Computes a stable checksum of this checkpoint with the `checksum` field
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

    /// Relative path for the derived zero-copy archive sibling of the JSON
    /// checkpoint at `seq`.
    pub fn archive_filename(seq: u64) -> String {
        format!("_checkpoints/{:09}.rkyv", seq)
    }

    /// Serialize this checkpoint as a zero-copy `rkyv` archive (a derived,
    /// rebuildable cache written alongside the canonical JSON). On open the
    /// archive is read + validated and the checkpoint reconstructed WITHOUT a
    /// `serde_json` structural parse — the O(fragments) win on high-fragment
    /// tables. Column-stat min/max (arbitrary JSON) are stored as their JSON
    /// strings (round-trip byte-identical) since `rkyv` cannot archive
    /// `serde_json::Value`.
    pub fn to_archive_bytes(&self) -> Vec<u8> {
        let arch = ArchCheckpoint {
            sequence: self.sequence,
            schema_id: self.schema_id,
            fragments: self.fragments.iter().map(frag_to_arch).collect(),
            checksum: self.checksum.clone(),
        };
        rkyv::to_bytes::<rkyv::rancor::Error>(&arch)
            .map(|b| b.to_vec())
            .unwrap_or_default()
    }

    /// Reconstruct a checkpoint from [`Self::to_archive_bytes`]. Returns `None`
    /// for absent/garbage/incompatible bytes so the caller falls back to JSON.
    ///
    /// Uses `rkyv::access` (validate + read the archived structure in place) and
    /// builds the checkpoint directly — no intermediate owned deserialize — so
    /// the only structural work is the one allocation pass the caller needs
    /// anyway, with no `serde_json` tokenizing of paths/ids/row-ids.
    pub fn from_archive_bytes(bytes: &[u8]) -> Option<Self> {
        let arch = rkyv::access::<ArchivedArchCheckpoint, rkyv::rancor::Error>(bytes).ok()?;
        Some(Self {
            sequence: arch.sequence.to_native(),
            schema_id: arch.schema_id.to_native(),
            fragments: arch.fragments.iter().map(arch_frag_to_summary).collect(),
            checksum: arch.checksum.as_str().to_string(),
        })
    }
}

// ── rkyv mirror types ──────────────────────────────────────────────
// `rkyv` cannot derive on `serde_json::Value` or `HashMap`, so this parallel
// representation stores stat min/max as JSON strings and maps as sorted vecs.

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ArchColStats {
    min: Option<String>,
    max: Option<String>,
    nulls: u64,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ArchOffset {
    offset: u64,
    length: u64,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
enum ArchSeg {
    Range { start: u64, count: u64 },
    Sorted { ids: Vec<u64> },
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ArchFragment {
    row_group: String,
    data: String,
    meta: String,
    agg: Option<String>,
    deletes: Option<String>,
    fragment_id: u64,
    rows: u64,
    deleted_count: u64,
    columns: Vec<(String, ArchColStats)>,
    column_offsets: Option<Vec<(String, ArchOffset)>>,
    row_ids: Vec<ArchSeg>,
    sort: Option<Vec<String>>,
    checksum: String,
    meta_checksum: String,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
struct ArchCheckpoint {
    sequence: u64,
    schema_id: u64,
    fragments: Vec<ArchFragment>,
    checksum: String,
}

fn frag_to_arch(f: &FragmentSummary) -> ArchFragment {
    let mut columns: Vec<(String, ArchColStats)> = f
        .columns
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                ArchColStats {
                    min: v.min.as_ref().map(|x| x.to_string()),
                    max: v.max.as_ref().map(|x| x.to_string()),
                    nulls: v.nulls as u64,
                },
            )
        })
        .collect();
    columns.sort_by(|a, b| a.0.cmp(&b.0));
    let column_offsets = f.column_offsets.as_ref().map(|m| {
        let mut v: Vec<(String, ArchOffset)> = m
            .iter()
            .map(|(k, o)| {
                (
                    k.clone(),
                    ArchOffset {
                        offset: o.offset,
                        length: o.length,
                    },
                )
            })
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    });
    ArchFragment {
        row_group: f.row_group.clone(),
        data: f.data.clone(),
        meta: f.meta.clone(),
        agg: f.agg.clone(),
        deletes: f.deletes.clone(),
        fragment_id: f.fragment_id,
        rows: f.rows as u64,
        deleted_count: f.deleted_count,
        columns,
        column_offsets,
        row_ids: f
            .row_ids
            .iter()
            .map(|s| match s {
                RowIdSegment::Range { start, count } => ArchSeg::Range {
                    start: *start,
                    count: *count,
                },
                RowIdSegment::Sorted { ids } => ArchSeg::Sorted { ids: ids.clone() },
            })
            .collect(),
        sort: f.sort.clone(),
        checksum: f.checksum.clone(),
        meta_checksum: f.meta_checksum.clone(),
    }
}

fn arch_frag_to_summary(a: &ArchivedArchFragment) -> FragmentSummary {
    let columns: HashMap<String, ColumnStats> = a
        .columns
        .iter()
        .map(|entry| {
            (
                entry.0.as_str().to_string(),
                ColumnStats {
                    min: entry
                        .1
                        .min
                        .as_ref()
                        .and_then(|s| serde_json::from_str(s.as_str()).ok()),
                    max: entry
                        .1
                        .max
                        .as_ref()
                        .and_then(|s| serde_json::from_str(s.as_str()).ok()),
                    nulls: entry.1.nulls.to_native() as usize,
                },
            )
        })
        .collect();
    let column_offsets = a.column_offsets.as_ref().map(|v| {
        v.iter()
            .map(|entry| {
                (
                    entry.0.as_str().to_string(),
                    ColumnChunkOffset {
                        offset: entry.1.offset.to_native(),
                        length: entry.1.length.to_native(),
                    },
                )
            })
            .collect::<HashMap<_, _>>()
    });
    FragmentSummary {
        row_group: a.row_group.as_str().to_string(),
        data: a.data.as_str().to_string(),
        meta: a.meta.as_str().to_string(),
        agg: a.agg.as_ref().map(|s| s.as_str().to_string()),
        deletes: a.deletes.as_ref().map(|s| s.as_str().to_string()),
        fragment_id: a.fragment_id.to_native(),
        rows: a.rows.to_native() as usize,
        deleted_count: a.deleted_count.to_native(),
        columns,
        column_offsets,
        row_ids: a
            .row_ids
            .iter()
            .map(|s| match s {
                ArchivedArchSeg::Range { start, count } => RowIdSegment::Range {
                    start: start.to_native(),
                    count: count.to_native(),
                },
                ArchivedArchSeg::Sorted { ids } => RowIdSegment::Sorted {
                    ids: ids.iter().map(|x| x.to_native()).collect(),
                },
            })
            .collect(),
        sort: a
            .sort
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str().to_string()).collect()),
        checksum: a.checksum.as_str().to_string(),
        meta_checksum: a.meta_checksum.as_str().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_round_trips() {
        let mut cp = SnapshotCheckpoint {
            sequence: 7,
            schema_id: 1,
            fragments: vec![],
            checksum: String::new(),
        };
        cp.checksum = cp.compute_checksum().unwrap();
        assert!(cp.verify_checksum().unwrap());
        let bytes = serde_json::to_vec(&cp).unwrap();
        let back: SnapshotCheckpoint = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(cp, back);
        assert_eq!(
            SnapshotCheckpoint::filename(7),
            "_checkpoints/000000007.json"
        );
    }

    fn sample_checkpoint() -> SnapshotCheckpoint {
        let mut columns = HashMap::new();
        columns.insert(
            "id".to_string(),
            ColumnStats {
                min: Some(serde_json::json!(0)),
                max: Some(serde_json::json!(999)),
                nulls: 0,
            },
        );
        columns.insert(
            "category".to_string(),
            ColumnStats {
                min: Some(serde_json::json!("cat_0")),
                max: Some(serde_json::json!("cat_9")),
                nulls: 3,
            },
        );
        let mut offsets = HashMap::new();
        offsets.insert(
            "id".to_string(),
            ColumnChunkOffset {
                offset: 4,
                length: 8000,
            },
        );
        let mut cp = SnapshotCheckpoint {
            sequence: 42,
            schema_id: 1,
            fragments: vec![
                FragmentSummary {
                    row_group: "rg_abc".into(),
                    data: "rg_abc.parquet".into(),
                    meta: "rg_abc.meta".into(),
                    agg: Some("rg_abc.agg".into()),
                    deletes: None,
                    fragment_id: 0,
                    rows: 1000,
                    deleted_count: 0,
                    columns,
                    column_offsets: Some(offsets),
                    row_ids: vec![RowIdSegment::Range {
                        start: 0,
                        count: 1000,
                    }],
                    sort: Some(vec!["id".into()]),
                    checksum: "sha256:aaa".into(),
                    meta_checksum: "sha256:bbb".into(),
                },
                FragmentSummary {
                    row_group: String::new(),
                    data: "rg_def.parquet".into(),
                    meta: "rg_def.meta".into(),
                    agg: None,
                    deletes: Some("_deletions/rg_def.del".into()),
                    fragment_id: 1,
                    rows: 5,
                    deleted_count: 2,
                    columns: HashMap::new(),
                    column_offsets: None,
                    row_ids: vec![RowIdSegment::Sorted {
                        ids: vec![1000, 1002, 1005],
                    }],
                    sort: None,
                    checksum: String::new(),
                    meta_checksum: String::new(),
                },
            ],
            checksum: String::new(),
        };
        cp.checksum = cp.compute_checksum().unwrap();
        cp
    }

    /// The rkyv archive round-trips byte-equal to the original
    /// checkpoint (including JSON stat min/max and row-id segments), and garbage
    /// bytes fall back cleanly to `None`.
    #[test]
    fn rkyv_archive_round_trips_byte_equal() {
        let cp = sample_checkpoint();
        let bytes = cp.to_archive_bytes();
        assert!(!bytes.is_empty());
        let back =
            SnapshotCheckpoint::from_archive_bytes(&bytes).expect("valid archive must reconstruct");
        assert_eq!(
            cp, back,
            "archive must round-trip byte-equal to the JSON path"
        );
        assert!(
            back.verify_checksum().unwrap(),
            "reconstructed checksum holds"
        );

        // Garbage / truncated bytes fall back to None.
        assert!(SnapshotCheckpoint::from_archive_bytes(b"").is_none());
        assert!(SnapshotCheckpoint::from_archive_bytes(b"not an rkyv archive").is_none());
        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() / 2);
        assert!(SnapshotCheckpoint::from_archive_bytes(&truncated).is_none());
    }
}
