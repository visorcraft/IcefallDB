//! Per-table deferred-commit log for mutations (the "mutation WAL").
//!
//! A normal DELETE commit materializes a full new `Manifest` (O(fragments) of
//! JSON plus a SHA-256) and runs the multi-`fsync` atomic pointer-swap ceremony,
//! roughly seven `fsync`s to flip a handful of deletion-vector bits. The
//! mutation WAL lets a DELETE commit durably with the deletion-vector write plus
//! one appended log record, deferring the manifest materialization and pointer
//! swap to a periodic *checkpoint*.
//!
//! Durability model (opt-in; default commit path is unchanged):
//! - The `_manifest.json` pointer marks the last **checkpointed** sequence.
//! - `{table}/_wal/mutations.log` holds compact [`MutationRecord`]s with
//!   `sequence > pointer`. Each record inlines the bytes of the small artifacts
//!   it produced (deletion vectors, index deltas) as [`StagedArtifact`]s and is
//!   appended with a single `fsync` — those artifact files are written *without*
//!   their own `fsync`, so a DELETE costs ~1 `fsync`. A crash that loses an
//!   un-`fsync`ed file is healed by [`materialize`] on the next open.
//! - The **live** manifest = checkpoint manifest + [`apply`] of every record in
//!   sequence order ([`live_manifest`]). Every table-open path replays the log,
//!   so a fresh reader and crash recovery both see the deferred mutations.
//! - A crash mid-append leaves a torn trailing line; [`read_records`] stops at
//!   the first unparseable / checksum-failing record (that mutation never
//!   returned success), so recovery is exactly the durable prefix.
//! - Checkpoint folds the log into a real manifest via the normal swap path,
//!   then [`clear`]s the log. The referenced deletion-vector files are now held
//!   by the manifest, so GC keeps them.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use chrono::Utc;

use crate::metadata::checksum::checksum_bytes;
use crate::metadata::manifest::{IndexRef, RowGroupEntry, RowIndexRef};
use crate::metadata::Manifest;
use crate::storage::Storage;
use crate::Result;

/// Per-fragment deletion-vector update produced by a deferred DELETE.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FragmentDelete {
    pub fragment_id: u64,
    /// Relative path (under the table root) to the new `.del` file.
    pub deletes: String,
    pub deleted_count: u64,
}

/// A small artifact (deletion vector or secondary-index delta) whose bytes are
/// inlined in the WAL record so the record's single `fsync` makes them durable.
///
/// In WAL mode the file is written to `path` **without** its own `fsync`; the
/// durable copy is here. [`materialize`] rewrites the file from these bytes if a
/// crash lost the un-`fsync`ed copy, so the read path can always open it by path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedArtifact {
    /// Relative path under the table root.
    pub path: String,
    /// Hex-encoded file contents.
    pub hex: String,
}

/// One deferred mutation. Compact: it carries only the manifest diff, not a
/// full manifest, so appending it is O(fragments touched), not O(fragments).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationRecord {
    /// Manifest sequence this record produces (one past the prior state).
    pub sequence: u64,
    /// Sequence this record was built on top of (for sanity checks).
    pub base_sequence: u64,
    /// Per-fragment deletion-vector replacements.
    pub fragment_deletes: Vec<FragmentDelete>,
    /// New row-group entries appended by this mutation (UPDATE patch fragments,
    /// MERGE inserts). Empty for a plain DELETE.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fragment_adds: Vec<RowGroupEntry>,
    /// The full new secondary-index generation map (small; authoritative for
    /// the produced manifest). Empty when the table has no indexes.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub index_generations: HashMap<String, IndexRef>,
    /// New `_rowindex` generation, if this mutation appended a delta (UPDATE /
    /// MERGE relocate rows into a patch fragment). `None` for DELETE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rowindex_generation: Option<RowIndexRef>,
    /// New `next_fragment_id` high-water mark when this mutation added fragments.
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub next_fragment_id: u64,
    /// New `next_row_id` high-water mark (unchanged by DELETE, advanced by a
    /// MERGE insert).
    #[serde(default, skip_serializing_if = "is_zero_u64")]
    pub next_row_id: u64,
    /// Small artifacts (deletion vectors, index deltas) whose bytes are inlined
    /// so this record's single `fsync` makes the whole commit durable — their
    /// on-disk files are written without their own `fsync`. Empty for records
    /// written before the inlining change (back-compat: those files were
    /// `fsync`ed individually).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub staged_artifacts: Vec<StagedArtifact>,
    /// `sha256:<hex>` over this record with `checksum` blanked.
    pub checksum: String,
}

#[inline]
fn is_zero_u64(v: &u64) -> bool {
    *v == 0
}

/// Relative path (under the table root) of the mutation log.
pub fn wal_path(table: &str) -> String {
    format!("{table}/_wal/mutations.log")
}

impl MutationRecord {
    /// Compute the checksum over this record with the `checksum` field blanked,
    /// so it is stable and independent of the stored checksum.
    pub fn compute_checksum(&self) -> String {
        let mut bare = self.clone();
        bare.checksum = String::new();
        let json = serde_json::to_vec(&bare).expect("MutationRecord serializes");
        checksum_bytes(&json)
    }

    /// True when the stored checksum matches a freshly computed one.
    pub fn verify(&self) -> bool {
        self.checksum == self.compute_checksum()
    }

    /// Stamp the checksum field from the record's current contents.
    pub fn sealed(mut self) -> Self {
        self.checksum = self.compute_checksum();
        self
    }
}

/// Append `record` to the table's mutation log and `fsync` it.
pub async fn append(storage: &dyn Storage, table: &str, record: &MutationRecord) -> Result<()> {
    let path = wal_path(table);
    let mut line = serde_json::to_vec(record)?;
    line.push(b'\n');
    storage.append(&path, &line).await?;
    storage.sync_data(&path).await?;
    Ok(())
}

/// Read the durable prefix of the mutation log, sorted by sequence.
///
/// Stops at the first record that fails to parse or fails its checksum — a torn
/// trailing line from a crash mid-append. Returns an empty vec when no log
/// exists (the common, non-WAL case).
pub async fn read_records(storage: &dyn Storage, table: &str) -> Result<Vec<MutationRecord>> {
    let path = wal_path(table);
    let data = match storage.read(&path).await {
        Ok(d) => d,
        Err(e) if is_not_found(&e) => return Ok(Vec::new()),
        Err(e) => return Err(e),
    };
    let mut records = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }
        match serde_json::from_slice::<MutationRecord>(line) {
            Ok(rec) if rec.verify() => records.push(rec),
            // Torn / corrupt record: the durable prefix ends here. We stop
            // rather than skip — nothing after a gap can be trusted.
            _ => break,
        }
    }
    records.sort_by_key(|r| r.sequence);
    Ok(records)
}

/// Apply one record to `base`, returning the produced manifest with a fresh
/// checksum so it passes `verify_checksum`.
pub fn apply(base: &Manifest, record: &MutationRecord) -> Result<Manifest> {
    let mut m = base.clone();
    m.sequence = record.sequence;
    for fd in &record.fragment_deletes {
        if let Some(entry) = m
            .row_groups
            .iter_mut()
            .find(|e| e.fragment_id == fd.fragment_id)
        {
            entry.deletes = Some(fd.deletes.clone());
            entry.deleted_count = fd.deleted_count;
        }
    }
    // New fragments (UPDATE patch / MERGE insert) are appended.
    m.row_groups.extend(record.fragment_adds.iter().cloned());
    // The record always carries the authoritative new index-generation map.
    m.index_generations = record.index_generations.clone();
    if record.rowindex_generation.is_some() {
        m.rowindex_generation = record.rowindex_generation.clone();
    }
    if record.next_fragment_id != 0 {
        m.next_fragment_id = record.next_fragment_id;
    }
    if record.next_row_id != 0 {
        m.next_row_id = record.next_row_id;
    }
    // row_counts is denormalized/optional; drop any stale parallel array so it is
    // not mismatched against the new row_groups length.
    m.row_counts = None;
    // Drop the inherited snapshot-checkpoint sidecar: it describes the prior
    // snapshot's fragments (stale deleted_counts / missing the patch fragment).
    // A deferred mutation emits no new sidecar, so force readers onto the
    // per-fragment `.meta` path, which is always current.
    m.checkpoint = None;
    m.checksum = m.compute_checksum()?;
    Ok(m)
}

/// The live manifest = `base` (the checkpoint) folded with every log record
/// whose `sequence > base.sequence`, in order. Returns `base` unchanged when
/// the log is empty.
///
/// NOTE: the returned manifest's hash-chain fields (`parent_hash`,
/// `committed_at`) are NOT authoritative — this is an in-memory fold of pending
/// WAL records, not a committed `_manifests/<seq>.json`. Only the committed
/// on-disk manifests carry the canonical chain; `doctor`/`check` verify those.
/// Do not read `parent_hash`/`committed_at` off a `live_manifest` result.
pub async fn live_manifest(storage: &dyn Storage, table: &str, base: Manifest) -> Result<Manifest> {
    let records = read_records(storage, table).await?;
    let mut m = base;
    for rec in &records {
        if rec.sequence > m.sequence {
            // Reconstruct any inlined artifact files a crash left un-`fsync`ed,
            // so the read path can open them by path. Idempotent (same bytes,
            // immutable paths); a no-op in the common, no-crash case.
            materialize(storage, table, rec).await?;
            m = apply(&m, rec)?;
        }
    }
    Ok(m)
}

/// Write any of `record`'s inlined [`StagedArtifact`]s whose on-disk file is
/// missing (lost to a crash before its single covering `fsync`). The bytes are
/// authoritative; the file is a reconstructible cache for the read path.
pub async fn materialize(
    storage: &dyn Storage,
    table: &str,
    record: &MutationRecord,
) -> Result<()> {
    for art in &record.staged_artifacts {
        let abs = format!("{table}/{}", art.path);
        if !storage.exists(&abs).await? {
            let bytes = hex::decode(&art.hex).map_err(|e| {
                crate::IcefallDBError::Other(Box::new(std::io::Error::other(format!(
                    "mutation_wal: bad artifact hex for {}: {e}",
                    art.path
                ))))
            })?;
            storage.write(&abs, &bytes).await?;
        }
    }
    Ok(())
}

/// Fold the WAL into a fresh manifest and advance the `_manifest.json` pointer,
/// then clear the log. **The caller must already hold the table write lock.**
/// Returns `false` (no-op) when the log is empty.
///
/// After this returns, the deletion-vector files the WAL referenced are held by
/// the new manifest, so GC and compaction can run against the pointer normally.
pub async fn checkpoint_locked(storage: &dyn Storage, table: &str) -> Result<bool> {
    let records = read_records(storage, table).await?;
    if records.is_empty() {
        return Ok(false);
    }
    let base = load_pointer_manifest(storage, table).await?;
    // All records already folded into the pointer (e.g. by an interleaved
    // non-WAL commit that built on the replayed live state)? Just drop the log;
    // do not rewrite a manifest at an already-published sequence.
    if records.iter().all(|r| r.sequence <= base.sequence) {
        clear(storage, table).await?;
        return Ok(true);
    }
    let mut live = base;
    for rec in &records {
        if rec.sequence > live.sequence {
            // The new manifest will reference these inlined artifacts, so make
            // them durable now: reconstruct any missing file, then `fsync` it.
            // (At commit they were written without their own `fsync`.)
            materialize(storage, table, rec).await?;
            for art in &rec.staged_artifacts {
                storage.sync_data(&format!("{table}/{}", art.path)).await?;
            }
            live = apply(&live, rec)?;
        }
    }
    // Wire the chain: link this folded manifest to the highest on-disk manifest
    // below its sequence. A WAL fold of N>1 records publishes a single manifest
    // at the highest folded sequence, so the immediate predecessor
    // `<seq-1>.json` is usually absent (those intermediate sequences were never
    // written) — `parent_manifest_checksum` then scans `_manifests/` for the
    // highest surviving predecessor (the base), keeping the chain intact. `None`
    // (a true anchor) results only at genesis or when every predecessor is gone.
    let parent = crate::metadata::parent_manifest_checksum(storage, table, live.sequence).await?;
    live.finalize(parent, Utc::now().to_rfc3339())?;

    let manifest_path = format!("{table}/{}", Manifest::filename(live.sequence));
    storage
        .write(&manifest_path, &serde_json::to_vec(&live)?)
        .await?;
    storage.sync_data(&manifest_path).await?;
    write_pointer(storage, table, live.sequence).await?;
    clear(storage, table).await?;
    Ok(true)
}

/// Acquire the table write lock, then [`checkpoint_locked`]. No-op when the log
/// is empty. Read-only callers that need durability-sensitive consistency (GC,
/// compaction) call this so they never see a manifest pointer that lags behind
/// committed WAL records.
pub async fn checkpoint_if_pending(
    storage: &dyn Storage,
    table: &str,
    timeout: std::time::Duration,
) -> Result<bool> {
    let lock_path = format!("{table}/_write.lock");
    let _lock = storage.lock_exclusive(&lock_path, timeout).await?;
    checkpoint_locked(storage, table).await
}

/// Read the manifest the `_manifest.json` pointer references.
async fn load_pointer_manifest(storage: &dyn Storage, table: &str) -> Result<Manifest> {
    let pointer_path = format!("{table}/_manifest.json");
    let data = storage.read(&pointer_path).await?;
    let pointer: serde_json::Value = serde_json::from_slice(&data)?;
    let seq = pointer
        .get("latest")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| {
            crate::IcefallDBError::InvalidManifestPointer("missing or invalid 'latest'".into())
        })?;
    let manifest_path = format!("{table}/{}", Manifest::filename(seq));
    let manifest: Manifest = serde_json::from_slice(&storage.read(&manifest_path).await?)?;
    Ok(manifest)
}

/// Atomically point `_manifest.json` at `seq`.
async fn write_pointer(storage: &dyn Storage, table: &str, seq: u64) -> Result<()> {
    let pointer_path = format!("{table}/_manifest.json");
    let tmp_path = format!("{pointer_path}.tmp");
    let pointer = serde_json::json!({ "latest": seq });
    storage
        .write(&tmp_path, serde_json::to_vec(&pointer)?.as_slice())
        .await?;
    storage.sync(&tmp_path).await?;
    storage.rename(&tmp_path, &pointer_path).await?;
    storage.sync(&format!("{table}/")).await?;
    Ok(())
}

/// Delete the mutation log after a checkpoint has folded it into a manifest.
pub async fn clear(storage: &dyn Storage, table: &str) -> Result<()> {
    let path = wal_path(table);
    match storage.delete(&path).await {
        Ok(()) => Ok(()),
        Err(e) if is_not_found(&e) => Ok(()),
        Err(e) => Err(e),
    }
}

fn is_not_found(e: &crate::IcefallDBError) -> bool {
    matches!(e, crate::IcefallDBError::NotFound(_))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::manifest::RowGroupEntry;

    fn record(seq: u64, frag: u64, del: &str, count: u64) -> MutationRecord {
        MutationRecord {
            sequence: seq,
            base_sequence: seq - 1,
            fragment_deletes: vec![FragmentDelete {
                fragment_id: frag,
                deletes: del.to_string(),
                deleted_count: count,
            }],
            fragment_adds: vec![],
            index_generations: HashMap::new(),
            rowindex_generation: None,
            next_fragment_id: 0,
            next_row_id: 0,
            staged_artifacts: vec![],
            checksum: String::new(),
        }
        .sealed()
    }

    fn base_manifest() -> Manifest {
        let mut m = Manifest {
            format_version: 1,
            sequence: 5,
            schema_id: 1,
            row_groups: vec![
                RowGroupEntry {
                    data: "rg_a.parquet".into(),
                    meta: "rg_a.meta".into(),
                    fragment_id: 0,
                    deletes: None,
                    deleted_count: 0,
                    ..Default::default()
                },
                RowGroupEntry {
                    data: "rg_b.parquet".into(),
                    meta: "rg_b.meta".into(),
                    fragment_id: 1,
                    deletes: None,
                    deleted_count: 0,
                    ..Default::default()
                },
            ],
            row_counts: None,
            partition_values: None,
            next_row_id: 100,
            next_fragment_id: 2,
            rowindex_generation: None,
            index_generations: HashMap::new(),
            checkpoint: None,
            parent_hash: None,
            committed_at: None,
            checksum: String::new(),
        };
        m.checksum = m.compute_checksum().unwrap();
        m
    }

    #[test]
    fn checksum_round_trips_and_detects_tampering() {
        let rec = record(6, 0, "_deletions/rg_0__v1.del", 3);
        assert!(rec.verify());
        let mut tampered = rec.clone();
        tampered.fragment_deletes[0].deleted_count = 999;
        assert!(!tampered.verify(), "mutated record must fail verification");
    }

    #[test]
    fn apply_sets_deletes_on_the_named_fragment_only() {
        let base = base_manifest();
        let m = apply(&base, &record(6, 1, "_deletions/rg_1__v1.del", 4)).unwrap();
        assert_eq!(m.sequence, 6);
        // fragment 0 untouched
        assert_eq!(m.row_groups[0].deletes, None);
        assert_eq!(m.row_groups[0].deleted_count, 0);
        // fragment 1 updated
        assert_eq!(
            m.row_groups[1].deletes.as_deref(),
            Some("_deletions/rg_1__v1.del")
        );
        assert_eq!(m.row_groups[1].deleted_count, 4);
        // checksum is valid for the produced manifest
        assert!(m.verify_checksum().unwrap());
    }

    #[tokio::test]
    async fn append_then_read_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        append(&storage, "t", &record(6, 0, "_deletions/rg_0__v1.del", 1))
            .await
            .unwrap();
        append(&storage, "t", &record(7, 1, "_deletions/rg_1__v1.del", 2))
            .await
            .unwrap();
        let recs = read_records(&storage, "t").await.unwrap();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].sequence, 6);
        assert_eq!(recs[1].sequence, 7);
    }

    #[tokio::test]
    async fn read_records_empty_when_no_log() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        assert!(read_records(&storage, "t").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn read_records_stops_at_torn_trailing_line() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        append(&storage, "t", &record(6, 0, "_deletions/rg_0__v1.del", 1))
            .await
            .unwrap();
        // A corrupt (newline-terminated) line, then a syntactically-valid record
        // after it. We must stop at the corruption and NOT trust the record that
        // follows a gap — even though it would parse on its own.
        storage
            .append(&wal_path("t"), b"{\"sequence\":7,\"fragm\n")
            .await
            .unwrap();
        append(&storage, "t", &record(8, 1, "_deletions/rg_1__v1.del", 2))
            .await
            .unwrap();
        let recs = read_records(&storage, "t").await.unwrap();
        assert_eq!(
            recs.len(),
            1,
            "must stop at the gap, ignoring records after it"
        );
        assert_eq!(recs[0].sequence, 6);
    }

    #[tokio::test]
    async fn live_manifest_folds_records_in_order() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        // base seq 5; two deferred deletes at 6 (frag 0) and 7 (frag 1).
        append(&storage, "t", &record(6, 0, "_deletions/rg_0__v1.del", 1))
            .await
            .unwrap();
        append(&storage, "t", &record(7, 1, "_deletions/rg_1__v1.del", 2))
            .await
            .unwrap();
        let live = live_manifest(&storage, "t", base_manifest()).await.unwrap();
        assert_eq!(live.sequence, 7);
        assert_eq!(live.row_groups[0].deleted_count, 1);
        assert_eq!(live.row_groups[1].deleted_count, 2);
        assert!(live.verify_checksum().unwrap());
    }

    async fn write_pointer_manifest(storage: &dyn Storage, table: &str, m: &Manifest) {
        let path = format!("{table}/{}", Manifest::filename(m.sequence));
        storage
            .write(&path, &serde_json::to_vec(m).unwrap())
            .await
            .unwrap();
        let ptr = format!("{table}/_manifest.json");
        let body = serde_json::to_vec(&serde_json::json!({ "latest": m.sequence })).unwrap();
        storage.write(&ptr, &body).await.unwrap();
    }

    async fn read_pointer_seq(storage: &dyn Storage, table: &str) -> u64 {
        let data = storage
            .read(&format!("{table}/_manifest.json"))
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&data).unwrap();
        v.get("latest").unwrap().as_u64().unwrap()
    }

    #[tokio::test]
    async fn checkpoint_folds_wal_into_pointer_and_clears() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        write_pointer_manifest(&storage, "t", &base_manifest()).await; // pointer at seq 5
        append(&storage, "t", &record(6, 0, "_deletions/rg_0__v1.del", 1))
            .await
            .unwrap();
        append(&storage, "t", &record(7, 1, "_deletions/rg_1__v1.del", 2))
            .await
            .unwrap();

        let did = checkpoint_if_pending(&storage, "t", std::time::Duration::from_secs(5))
            .await
            .unwrap();
        assert!(did, "checkpoint with pending records returns true");

        // Pointer advanced to the live sequence; WAL cleared.
        assert_eq!(read_pointer_seq(&storage, "t").await, 7);
        assert!(read_records(&storage, "t").await.unwrap().is_empty());

        // The checkpointed manifest is a valid, normal manifest with the deletes.
        let m: Manifest = serde_json::from_slice(
            &storage
                .read(&format!("t/{}", Manifest::filename(7)))
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(m.verify_checksum().unwrap());
        assert_eq!(m.row_groups[0].deleted_count, 1);
        assert_eq!(m.row_groups[1].deleted_count, 2);

        // A second checkpoint with no pending records is a no-op.
        assert!(
            !checkpoint_if_pending(&storage, "t", std::time::Duration::from_secs(5))
                .await
                .unwrap()
        );
    }

    #[tokio::test]
    async fn materialize_reconstructs_a_missing_inlined_artifact() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        let mut rec = record(6, 0, "_deletions/rg_0__v1.del", 1);
        rec.staged_artifacts = vec![StagedArtifact {
            path: "_deletions/rg_0__v1.del".into(),
            hex: hex::encode([1u8, 2, 3, 4]),
        }];
        rec = rec.sealed();
        // File absent (a crash lost the un-fsynced copy) -> reconstructed.
        materialize(&storage, "t", &rec).await.unwrap();
        let got = storage.read("t/_deletions/rg_0__v1.del").await.unwrap();
        assert_eq!(got, vec![1u8, 2, 3, 4]);
        // Idempotent: a present file is not overwritten.
        storage
            .write("t/_deletions/rg_0__v1.del", &[9u8, 9])
            .await
            .unwrap();
        materialize(&storage, "t", &rec).await.unwrap();
        assert_eq!(
            storage.read("t/_deletions/rg_0__v1.del").await.unwrap(),
            vec![9u8, 9]
        );
    }

    #[tokio::test]
    async fn clear_removes_the_log() {
        let tmp = tempfile::tempdir().unwrap();
        let storage = crate::storage::local::LocalStorage::new(tmp.path()).unwrap();
        append(&storage, "t", &record(6, 0, "_deletions/rg_0__v1.del", 1))
            .await
            .unwrap();
        clear(&storage, "t").await.unwrap();
        assert!(read_records(&storage, "t").await.unwrap().is_empty());
        // Idempotent.
        clear(&storage, "t").await.unwrap();
    }
}
