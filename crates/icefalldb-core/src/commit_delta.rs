use crate::metadata::{Manifest, RowGroupEntry};
use crate::reader::PlannedRowGroup;

/// Classification of a table mutation. Used by the query provider to decide
/// which cached state must be invalidated or recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitKind {
    /// No manifest sequence advanced (empty commit or no-op fallback).
    Noop,
    /// New rows appended to the table.
    Append,
    /// Table snapshot replaced with buffered rows (view refresh / overwrite).
    Replace,
    /// Rows marked deleted via deletion vectors.
    Delete,
    /// Rows updated in place via tombstone + patch fragment.
    Update,
    /// Atomic matched-update + unmatched-insert.
    Merge,
    /// Offline compaction / optimize rewrite.
    Compact,
}

/// Per-fragment change for fragments whose deletion vector changed without a
/// full rewrite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentDelta {
    pub fragment_id: u64,
    pub old_deletes: Option<String>,
    pub new_deletes: Option<String>,
    pub old_deleted_count: u64,
    pub new_deleted_count: u64,
}

/// Description of what changed during a writer commit.
///
/// The delta carries the previous and new manifests so that consumers can
/// derive exactly which fragments were added, removed, or modified without
/// re-reading unchanged sidecars. For a no-op commit `previous_sequence` equals
/// `new_sequence`.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitDelta {
    pub previous_sequence: u64,
    pub new_sequence: u64,
    pub kind: CommitKind,
    pub previous_manifest: Manifest,
    pub new_manifest: Manifest,
    /// Fully populated [`PlannedRowGroup`] values for every fragment added by
    /// this commit. Carrying them lets the query provider refresh its cached
    /// scan plan without re-reading `.meta`/manifest sidecars.
    pub added_row_groups: Vec<PlannedRowGroup>,
}

impl CommitDelta {
    /// Build a delta from two consecutive manifests.
    pub fn new(previous_manifest: &Manifest, new_manifest: &Manifest, kind: CommitKind) -> Self {
        Self {
            previous_sequence: previous_manifest.sequence,
            new_sequence: new_manifest.sequence,
            kind,
            previous_manifest: previous_manifest.clone(),
            new_manifest: new_manifest.clone(),
            added_row_groups: Vec::new(),
        }
    }

    /// Attach the row groups added by this commit.
    pub fn with_added_row_groups(mut self, added: Vec<PlannedRowGroup>) -> Self {
        self.added_row_groups = added;
        self
    }

    /// Returns true when the commit did not advance the manifest sequence.
    pub fn is_noop(&self) -> bool {
        self.previous_sequence == self.new_sequence
    }

    /// Returns true when the schema id changed between snapshots.
    pub fn schema_changed(&self) -> bool {
        self.previous_manifest.schema_id != self.new_manifest.schema_id
    }

    /// Row groups present in the new manifest but not the previous one.
    pub fn added_row_groups(&self) -> Vec<&RowGroupEntry> {
        let prev_ids: std::collections::HashSet<u64> = self
            .previous_manifest
            .row_groups
            .iter()
            .map(|rg| rg.fragment_id)
            .collect();
        self.new_manifest
            .row_groups
            .iter()
            .filter(|rg| !prev_ids.contains(&rg.fragment_id))
            .collect()
    }

    /// Fragment IDs present in the previous manifest but not the new one.
    pub fn removed_fragment_ids(&self) -> Vec<u64> {
        let new_ids: std::collections::HashSet<u64> = self
            .new_manifest
            .row_groups
            .iter()
            .map(|rg| rg.fragment_id)
            .collect();
        self.previous_manifest
            .row_groups
            .iter()
            .filter(|rg| !new_ids.contains(&rg.fragment_id))
            .map(|rg| rg.fragment_id)
            .collect()
    }

    /// Fragments whose deletion vector changed while their data/meta files stayed
    /// the same.
    pub fn updated_fragments(&self) -> Vec<FragmentDelta> {
        let prev_by_id: std::collections::HashMap<u64, &RowGroupEntry> = self
            .previous_manifest
            .row_groups
            .iter()
            .map(|rg| (rg.fragment_id, rg))
            .collect();
        let mut deltas = Vec::new();
        for new in &self.new_manifest.row_groups {
            if let Some(old) = prev_by_id.get(&new.fragment_id) {
                if old.data == new.data
                    && old.meta == new.meta
                    && (old.deletes != new.deletes || old.deleted_count != new.deleted_count)
                {
                    deltas.push(FragmentDelta {
                        fragment_id: new.fragment_id,
                        old_deletes: old.deletes.clone(),
                        new_deletes: new.deletes.clone(),
                        old_deleted_count: old.deleted_count,
                        new_deleted_count: new.deleted_count,
                    });
                }
            }
        }
        deltas
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{Manifest, RowGroupEntry};

    fn manifest_with_fragments(seq: u64, fragment_ids: &[u64]) -> Manifest {
        Manifest {
            format_version: 1,
            sequence: seq,
            schema_id: 1,
            row_groups: fragment_ids
                .iter()
                .map(|&id| RowGroupEntry {
                    data: format!("rg_{}.parquet", id),
                    meta: format!("rg_{}.meta", id),
                    fragment_id: id,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn added_row_groups_returns_only_new_fragments() {
        let prev = manifest_with_fragments(1, &[1, 2]);
        let new = manifest_with_fragments(2, &[1, 2, 3]);
        let delta = CommitDelta::new(&prev, &new, CommitKind::Append);
        let added: Vec<u64> = delta
            .added_row_groups()
            .iter()
            .map(|rg| rg.fragment_id)
            .collect();
        assert_eq!(added, vec![3]);
    }

    #[test]
    fn removed_fragment_ids_returns_dropped_fragments() {
        let prev = manifest_with_fragments(1, &[1, 2, 3]);
        let new = manifest_with_fragments(2, &[1, 3]);
        let delta = CommitDelta::new(&prev, &new, CommitKind::Compact);
        assert_eq!(delta.removed_fragment_ids(), vec![2]);
    }

    #[test]
    fn updated_fragments_detects_deletion_vector_changes() {
        let mut prev = manifest_with_fragments(1, &[1]);
        prev.row_groups[0].deletes = Some("_deletions/rg_1__v1.del".into());
        prev.row_groups[0].deleted_count = 1;

        let mut new = manifest_with_fragments(2, &[1]);
        new.row_groups[0].deletes = Some("_deletions/rg_1__v2.del".into());
        new.row_groups[0].deleted_count = 2;

        let delta = CommitDelta::new(&prev, &new, CommitKind::Delete);
        let deltas = delta.updated_fragments();
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].fragment_id, 1);
        assert_eq!(deltas[0].old_deleted_count, 1);
        assert_eq!(deltas[0].new_deleted_count, 2);
    }

    #[test]
    fn noop_delta_has_equal_sequences() {
        let prev = manifest_with_fragments(1, &[1]);
        let delta = CommitDelta::new(&prev, &prev, CommitKind::Append);
        assert!(delta.is_noop());
    }
}
