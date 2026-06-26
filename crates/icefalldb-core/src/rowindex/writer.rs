//! Encoder and decoder for the `_rowindex` on-disk format.
//!
//! # File layout
//!
//! ```text
//! [header  32 bytes]
//! [segments  N × 24 bytes]
//! [block_index  B × 12 bytes]
//! [trailer  4 bytes: crc32 over everything that precedes]
//! ```
//!
//! ## Header (32 bytes, all fields little-endian)
//!
//! | offset | size | field          |
//! |--------|------|----------------|
//! |  0     |  4   | magic (`MRIX`) |
//! |  4     |  2   | format_version |
//! |  6     |  2   | _reserved      |
//! |  8     |  8   | segment_count  |
//! | 16     |  8   | file_length    |
//! | 24     |  8   | block_index_offset (byte offset from file start) |
//!
//! ## Segment (24 bytes, all fields little-endian)
//!
//! Sorted ascending by `start_row_id`.
//!
//! | offset | size | field        |
//! |--------|------|--------------|
//! |  0     |  8   | start_row_id |
//! |  8     |  8   | fragment_id  |
//! | 16     |  4   | start_offset |
//! | 20     |  4   | len          |
//!
//! ## Block index entry (12 bytes, all fields little-endian)
//!
//! Every `BLOCK_STRIDE`-th segment produces one entry so the remote reader can
//! binary-search to the right byte range without fetching the whole file.
//!
//! | offset | size | field                       |
//! |--------|------|-----------------------------|
//! |  0     |  8   | start_row_id of that segment |
//! |  8     |  4   | byte offset of that segment  |
//!
//! ## Trailer
//!
//! 4-byte CRC32 (IEEE polynomial 0xEDB88320, little-endian) computed over every
//! byte of the file that precedes it (header + segments + block index).

use crate::deletion::DeletionVector;
use crate::error::{IcefallDBError, Result};
use crate::metadata::manifest::RowIndexRef;
use crate::metadata::{Manifest, RowGroupMeta};
use crate::rowid::segment_ids;
use crate::rowindex::AddrSegment;
use crate::storage::Storage;

// Small helper so call sites stay terse.
fn fmt_err(msg: impl Into<String>) -> crate::error::IcefallDBError {
    crate::error::IcefallDBError::Other(msg.into().into())
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// ASCII `MRIX` — IcefallDB Row IndeX.
const MAGIC: [u8; 4] = *b"MRIX";
const FORMAT_VERSION: u16 = 1;
pub const HEADER_LEN: usize = 32;
pub const SEGMENT_LEN: usize = 24;
const BLOCK_ENTRY_LEN: usize = 12;
const TRAILER_LEN: usize = 4;
/// One block-index entry per this many segments (sparse index for remote seeks).
pub const BLOCK_STRIDE: usize = 1024;

// ---------------------------------------------------------------------------
// CRC32 (IEEE 802.3 / Ethernet polynomial 0xEDB88320 — reflected form)
// ---------------------------------------------------------------------------

fn make_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for i in 0u32..256 {
        let mut c = i;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        table[i as usize] = c;
    }
    table
}

fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(make_crc_table);
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = (crc >> 8) ^ table[idx];
    }
    crc ^ 0xFFFF_FFFF
}

// ---------------------------------------------------------------------------
// encode_idx
// ---------------------------------------------------------------------------

/// Encode a slice of [`AddrSegment`]s into the `_rowindex` binary format.
///
/// Segments are sorted by `start_row_id` before encoding (stable sort, so
/// equal `start_row_id` values retain their input order).
pub fn encode_idx(segs: &[AddrSegment]) -> Vec<u8> {
    // Sort a local copy so the caller's slice is unaffected.
    let mut sorted: Vec<AddrSegment> = segs.to_vec();
    sorted.sort_by_key(|s| s.start_row_id);

    let seg_count = sorted.len();
    let block_count = if seg_count == 0 {
        0
    } else {
        (seg_count - 1) / BLOCK_STRIDE + 1
    };
    let block_index_offset = HEADER_LEN + seg_count * SEGMENT_LEN;
    let file_len_no_trailer = block_index_offset + block_count * BLOCK_ENTRY_LEN;
    let total_len = file_len_no_trailer + TRAILER_LEN;

    let mut buf: Vec<u8> = Vec::with_capacity(total_len);

    // --- header ---
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
    buf.extend_from_slice(&(seg_count as u64).to_le_bytes());
    buf.extend_from_slice(&(total_len as u64).to_le_bytes());
    buf.extend_from_slice(&(block_index_offset as u64).to_le_bytes());
    debug_assert_eq!(buf.len(), HEADER_LEN);

    // --- segments ---
    for seg in &sorted {
        buf.extend_from_slice(&seg.start_row_id.to_le_bytes());
        buf.extend_from_slice(&seg.fragment_id.to_le_bytes());
        buf.extend_from_slice(&seg.start_offset.to_le_bytes());
        buf.extend_from_slice(&seg.len.to_le_bytes());
    }

    // --- block index ---
    for (i, seg) in sorted.iter().enumerate() {
        if i % BLOCK_STRIDE == 0 {
            let seg_byte_offset = HEADER_LEN + i * SEGMENT_LEN;
            buf.extend_from_slice(&seg.start_row_id.to_le_bytes());
            buf.extend_from_slice(&(seg_byte_offset as u32).to_le_bytes());
        }
    }
    debug_assert_eq!(buf.len(), file_len_no_trailer);

    // --- trailer: CRC32 over everything so far ---
    let checksum = crc32(&buf);
    buf.extend_from_slice(&checksum.to_le_bytes());
    debug_assert_eq!(buf.len(), total_len);

    buf
}

// ---------------------------------------------------------------------------
// DecodedIdx + decode_idx
// ---------------------------------------------------------------------------

/// The result of a successful [`decode_idx`] call.
pub struct DecodedIdx {
    segments: Vec<AddrSegment>,
}

impl DecodedIdx {
    /// The decoded, sorted segments.
    pub fn segments(&self) -> &[AddrSegment] {
        &self.segments
    }
}

/// Decode a buffer previously produced by [`encode_idx`].
///
/// Returns an error if:
/// - The buffer is too short to contain a valid header.
/// - The magic bytes do not match `MRIX`.
/// - The CRC32 trailer does not match the computed checksum.
/// - Field values are internally inconsistent (e.g. reported length mismatch).
pub fn decode_idx(buf: &[u8]) -> Result<DecodedIdx> {
    if buf.len() < HEADER_LEN + TRAILER_LEN {
        return Err(fmt_err("rowindex: buffer too short for header+trailer"));
    }

    // --- verify magic ---
    if buf[..4] != MAGIC {
        return Err(fmt_err("rowindex: bad magic bytes"));
    }

    // --- verify CRC32 ---
    let body = &buf[..buf.len() - TRAILER_LEN];
    let stored_crc = u32::from_le_bytes(
        buf[buf.len() - TRAILER_LEN..]
            .try_into()
            .map_err(|_| fmt_err("rowindex: internal error reading CRC trailer slice"))?,
    );
    let computed_crc = crc32(body);
    if stored_crc != computed_crc {
        return Err(fmt_err(
            "rowindex: CRC32 mismatch — file is corrupt or truncated",
        ));
    }

    // --- parse header fields ---
    let _version = u16::from_le_bytes(
        buf[4..6]
            .try_into()
            .map_err(|_| fmt_err("rowindex: internal error reading version field"))?,
    );
    // reserved: buf[6..8]
    let seg_count = u64::from_le_bytes(
        buf[8..16]
            .try_into()
            .map_err(|_| fmt_err("rowindex: internal error reading segment_count field"))?,
    ) as usize;
    let file_len = u64::from_le_bytes(
        buf[16..24]
            .try_into()
            .map_err(|_| fmt_err("rowindex: internal error reading file_length field"))?,
    ) as usize;
    let block_index_offset = u64::from_le_bytes(
        buf[24..32]
            .try_into()
            .map_err(|_| fmt_err("rowindex: internal error reading block_index_offset field"))?,
    ) as usize;

    if buf.len() != file_len {
        return Err(fmt_err(format!(
            "rowindex: buffer length {} != recorded file_length {}",
            buf.len(),
            file_len,
        )));
    }

    let expected_seg_end = HEADER_LEN + seg_count * SEGMENT_LEN;
    if block_index_offset != expected_seg_end {
        return Err(fmt_err(
            "rowindex: block_index_offset is inconsistent with segment count",
        ));
    }

    if buf.len() < expected_seg_end {
        return Err(fmt_err("rowindex: buffer truncated within segments region"));
    }

    // --- decode segments ---
    let mut segments = Vec::with_capacity(seg_count);
    for i in 0..seg_count {
        let off = HEADER_LEN + i * SEGMENT_LEN;
        let start_row_id = u64::from_le_bytes(buf[off..off + 8].try_into().map_err(|_| {
            fmt_err(format!(
                "rowindex: internal error reading start_row_id at segment {i}"
            ))
        })?);
        let fragment_id = u64::from_le_bytes(buf[off + 8..off + 16].try_into().map_err(|_| {
            fmt_err(format!(
                "rowindex: internal error reading fragment_id at segment {i}"
            ))
        })?);
        let start_offset =
            u32::from_le_bytes(buf[off + 16..off + 20].try_into().map_err(|_| {
                fmt_err(format!(
                    "rowindex: internal error reading start_offset at segment {i}"
                ))
            })?);
        let len = u32::from_le_bytes(buf[off + 20..off + 24].try_into().map_err(|_| {
            fmt_err(format!(
                "rowindex: internal error reading len at segment {i}"
            ))
        })?);
        segments.push(AddrSegment {
            start_row_id,
            fragment_id,
            start_offset,
            len,
        });
    }

    Ok(DecodedIdx { segments })
}

// ---------------------------------------------------------------------------
// derive_base
// ---------------------------------------------------------------------------

/// Derive the `_rowindex` base from a manifest, considering only live rows.
///
/// For each fragment in the manifest this function:
/// 1. Loads the fragment's [`RowGroupMeta`] (for `row_ids`).
/// 2. If the fragment has a deletion vector, loads it too.
/// 3. Walks the fragment's row-id sequence in physical order (offset 0, 1, …),
///    skipping any offset marked as deleted in the deletion vector.
/// 4. Collects `(row_id, fragment_id, physical_offset)` for every live row.
///
/// After walking all fragments the live records are sorted by `row_id` and
/// coalesced into [`AddrSegment`]s.  A contiguous run extends while the
/// `row_id` increments by 1, the `fragment_id` is the same, and the physical
/// `offset` also increments by 1.
///
/// Returns an error if any `row_id` appears more than once in the live set
/// (that would indicate data corruption — e.g. a row tombstoned in one
/// fragment but also live in another without the tombstone being applied).
pub async fn derive_base(
    storage: &dyn Storage,
    table: &str,
    manifest: &Manifest,
) -> Result<Vec<AddrSegment>> {
    // Collect (row_id, fragment_id, physical_offset) for every live row.
    let mut live: Vec<(u64, u64, u32)> = Vec::new();

    for entry in &manifest.row_groups {
        // Load the fragment's metadata (row_ids live here).
        let meta_path = format!("{}/{}", table, entry.meta);
        let meta_bytes = storage.read(&meta_path).await?;
        let meta: RowGroupMeta =
            serde_json::from_slice(&meta_bytes).map_err(IcefallDBError::Serialization)?;

        // Load the deletion vector if one exists.
        let dv: Option<DeletionVector> = if let Some(del_path) = &entry.deletes {
            let del_full = format!("{}/{}", table, del_path);
            let del_bytes = storage.read(&del_full).await?;
            let dv = DeletionVector::deserialize(&del_bytes).map_err(IcefallDBError::Io)?;
            Some(dv)
        } else {
            None
        };

        // Walk row-ids in physical order; the offset counter spans all segments.
        let mut offset: u32 = 0;
        for seg in &meta.row_ids {
            for row_id in segment_ids(seg) {
                if dv.as_ref().is_some_and(|d| d.contains(offset)) {
                    // Dead row — skip it.
                } else {
                    live.push((row_id, entry.fragment_id, offset));
                }
                offset += 1;
            }
        }
    }

    // Sort by row_id.
    live.sort_unstable_by_key(|&(row_id, _, _)| row_id);

    // Assert exactly one live address per row_id (duplicates = corruption).
    for window in live.windows(2) {
        let (id_a, _, _) = window[0];
        let (id_b, _, _) = window[1];
        if id_a == id_b {
            return Err(IcefallDBError::Other(
                format!(
                    "derive_base: row_id {} appears more than once in the live set \
                     — data corruption or missing tombstone",
                    id_a
                )
                .into(),
            ));
        }
    }

    // Coalesce consecutive live records into AddrSegments.
    let mut segments: Vec<AddrSegment> = Vec::new();
    for (row_id, fragment_id, offset) in live {
        if let Some(last) = segments.last_mut() {
            // Extend the current run if row_id, fragment, and offset are all consecutive.
            let expected_row_id = last.start_row_id + u64::from(last.len);
            let expected_offset = last.start_offset + last.len;
            if last.fragment_id == fragment_id
                && row_id == expected_row_id
                && offset == expected_offset
            {
                last.len += 1;
                continue;
            }
        }
        segments.push(AddrSegment {
            start_row_id: row_id,
            fragment_id,
            start_offset: offset,
            len: 1,
        });
    }

    Ok(segments)
}

// ---------------------------------------------------------------------------
// rebuild
// ---------------------------------------------------------------------------

/// Regenerate the `_rowindex` base from the manifest's current live-row set.
///
/// This function:
/// 1. Calls [`derive_base`] to compute the fresh live-row address segments.
/// 2. Encodes them with [`encode_idx`].
/// 3. Writes the bytes to `_rowindex/base__v<seq>.idx` (where `<seq>` is
///    `manifest.sequence`, zero-padded to 9 digits, matching the convention
///    used by [`Manifest::filename`]).
/// 4. Returns a [`RowIndexRef`] with `base` set to that path and `deltas`
///    empty — callers should persist this into the next manifest.
pub async fn rebuild(
    storage: &dyn Storage,
    table: &str,
    manifest: &Manifest,
) -> Result<RowIndexRef> {
    let segments = derive_base(storage, table, manifest).await?;
    let bytes = encode_idx(&segments);
    // The padding width (9 digits) matches Manifest::filename: format!("_manifests/{:09}.json", seq).
    let rel_path = format!("_rowindex/base__v{:09}.idx", manifest.sequence);
    let storage_path = format!("{table}/{rel_path}");
    storage.write(&storage_path, &bytes).await?;
    Ok(RowIndexRef {
        base: Some(rel_path),
        deltas: vec![],
    })
}

// ---------------------------------------------------------------------------
// verify
// ---------------------------------------------------------------------------

/// Check that the on-disk `_rowindex` generation recorded in `manifest`
/// matches a fresh [`derive_base`] from the same manifest.
///
/// ## `None` generation
///
/// If `manifest.rowindex_generation` is `None`, the function checks that the
/// fresh derive is also empty (no live rows produce no segments).  If the
/// derive *would* produce segments for a table that has no index yet, that is
/// a consistency error and `Err` is returned.  An empty derive with no
/// generation is silently `Ok` — the table simply has no row-id index yet.
///
/// ## Error conditions
///
/// Returns `Err` with a descriptive message if:
/// - Any `row_id` present in the fresh derive is absent from the on-disk map.
/// - Any `row_id` present in the fresh derive resolves to a *different*
///   `(fragment_id, offset)` in the on-disk map.
/// - The on-disk map claims `None` for a row that the fresh derive maps.
pub async fn verify(storage: &dyn Storage, table: &str, manifest: &Manifest) -> Result<()> {
    use crate::rowindex::reader::AddressMap;

    // Derive the fresh ground-truth from the manifest.
    let fresh = derive_base(storage, table, manifest).await?;

    // Handle the None-generation case.
    let gen = match &manifest.rowindex_generation {
        Some(g) => g,
        None => {
            // No index on disk — acceptable only when the fresh derive is also empty.
            if fresh.is_empty() {
                return Ok(());
            }
            return Err(IcefallDBError::Other(
                "verify: manifest has no rowindex_generation but the table has live rows; \
                 run rebuild first"
                    .into(),
            ));
        }
    };

    // Load the on-disk generation.
    let am = AddressMap::open(storage, table, gen).await?;

    // Expand every AddrSegment from the fresh derive and compare.
    for seg in &fresh {
        for i in 0..seg.len {
            let row_id = seg.start_row_id + u64::from(i);
            let expected = (seg.fragment_id, seg.start_offset + i);
            match am.lookup(row_id) {
                Some(got) if got == expected => {} // ok
                Some(got) => {
                    return Err(IcefallDBError::Other(
                        format!(
                            "verify: row_id {row_id} mismatch — \
                             on-disk map says (fragment={}, offset={}) \
                             but fresh derive says (fragment={}, offset={})",
                            got.0, got.1, expected.0, expected.1
                        )
                        .into(),
                    ));
                }
                None => {
                    return Err(IcefallDBError::Other(
                        format!(
                            "verify: row_id {row_id} is present in the fresh derive \
                             (fragment={}, offset={}) but the on-disk map cannot resolve it",
                            expected.0, expected.1
                        )
                        .into(),
                    ));
                }
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deletion::DeletionVector;
    use crate::metadata::{Manifest, RowGroupEntry, RowGroupMeta};
    use crate::rowid::RowIdSegment;
    use crate::rowindex::{lookup_in_base, AddrSegment};
    use crate::storage::memory::MemoryStorage;
    use std::sync::Arc;

    #[test]
    fn idx_file_roundtrips_with_checksum() {
        let segs = vec![
            AddrSegment {
                start_row_id: 0,
                fragment_id: 1,
                start_offset: 0,
                len: 100,
            },
            AddrSegment {
                start_row_id: 100,
                fragment_id: 2,
                start_offset: 0,
                len: 50,
            },
        ];
        let buf = encode_idx(&segs);
        let decoded = decode_idx(&buf).unwrap(); // verifies magic+crc
        assert_eq!(decoded.segments(), segs.as_slice());
        let mut bad = buf.clone();
        *bad.last_mut().unwrap() ^= 0xFF;
        assert!(decode_idx(&bad).is_err()); // crc mismatch detected
    }

    #[test]
    fn encode_sorts_by_start_row_id() {
        let segs = vec![
            AddrSegment {
                start_row_id: 200,
                fragment_id: 3,
                start_offset: 0,
                len: 10,
            },
            AddrSegment {
                start_row_id: 0,
                fragment_id: 1,
                start_offset: 0,
                len: 100,
            },
            AddrSegment {
                start_row_id: 100,
                fragment_id: 2,
                start_offset: 0,
                len: 50,
            },
        ];
        let buf = encode_idx(&segs);
        let decoded = decode_idx(&buf).unwrap();
        let ids: Vec<u64> = decoded.segments().iter().map(|s| s.start_row_id).collect();
        assert_eq!(ids, vec![0, 100, 200]);
    }

    #[test]
    fn empty_index_roundtrips() {
        let buf = encode_idx(&[]);
        let decoded = decode_idx(&buf).unwrap();
        assert!(decoded.segments().is_empty());
    }

    #[test]
    fn bad_magic_is_rejected() {
        let segs = vec![AddrSegment {
            start_row_id: 0,
            fragment_id: 1,
            start_offset: 0,
            len: 10,
        }];
        let mut buf = encode_idx(&segs);
        buf[0] = b'X'; // corrupt magic
        assert!(decode_idx(&buf).is_err());
    }

    #[test]
    fn block_index_present_for_large_input() {
        // 1025 segments → 2 block index entries
        let segs: Vec<AddrSegment> = (0u64..1025)
            .map(|i| AddrSegment {
                start_row_id: i * 10,
                fragment_id: i,
                start_offset: 0,
                len: 10,
            })
            .collect();
        let buf = encode_idx(&segs);
        let decoded = decode_idx(&buf).unwrap();
        assert_eq!(decoded.segments().len(), 1025);
    }

    #[test]
    fn crc32_known_value() {
        // CRC32("123456789") == 0xCBF43926 per IEEE standard
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    // -----------------------------------------------------------------------
    // derive_base: updated row resolves to patch fragment, not tombstoned original
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn derive_base_uses_live_rows_only() {
        // fragment A: ids 0,1,2,3 at offsets 0,1,2,3; offset 1 is deleted
        //             (row_id 1 was updated → row 1 in A is tombstoned)
        // fragment B (patch): id 1 at offset 0
        //
        // derived base must map:
        //   1 → (frag_b_id, 0)   — only the live copy in B
        //   0 → (frag_a_id, 0)
        //   2 → (frag_a_id, 2)

        let storage = Arc::new(MemoryStorage::new());
        let frag_a_id: u64 = 10;
        let frag_b_id: u64 = 20;

        // --- Write fragment A's meta ---
        let meta_a = RowGroupMeta {
            row_group: "rg_a".to_string(),
            schema_id: 1,
            rows: 4,
            columns: Default::default(),
            column_offsets: None,
            sort: None,
            row_ids: vec![RowIdSegment::Range { start: 0, count: 4 }],
            checksum: String::new(),
            meta_checksum: String::new(),
        };
        let meta_a_bytes = serde_json::to_vec(&meta_a).unwrap();
        storage.write("t/rg_a.meta", &meta_a_bytes).await.unwrap();

        // --- Write fragment A's deletion vector (offset 1 deleted) ---
        let mut dv_a = DeletionVector::default();
        dv_a.union_offsets([1u32]);
        let del_a_bytes = dv_a.serialize();
        storage
            .write("t/_deletions/rg_a.del", &del_a_bytes)
            .await
            .unwrap();

        // --- Write fragment B's meta ---
        let meta_b = RowGroupMeta {
            row_group: "rg_b".to_string(),
            schema_id: 1,
            rows: 1,
            columns: Default::default(),
            column_offsets: None,
            sort: None,
            row_ids: vec![RowIdSegment::Sorted { ids: vec![1] }],
            checksum: String::new(),
            meta_checksum: String::new(),
        };
        let meta_b_bytes = serde_json::to_vec(&meta_b).unwrap();
        storage.write("t/rg_b.meta", &meta_b_bytes).await.unwrap();

        // --- Build a minimal manifest ---
        let manifest = Manifest {
            format_version: 1,
            sequence: 1,
            schema_id: 1,
            row_groups: vec![
                RowGroupEntry {
                    data: "rg_a.parquet".to_string(),
                    meta: "rg_a.meta".to_string(),
                    fragment_id: frag_a_id,
                    deletes: Some("_deletions/rg_a.del".to_string()),
                    deleted_count: 1,
                    agg: None,
                },
                RowGroupEntry {
                    data: "rg_b.parquet".to_string(),
                    meta: "rg_b.meta".to_string(),
                    fragment_id: frag_b_id,
                    deletes: None,
                    deleted_count: 0,
                    agg: None,
                },
            ],
            checksum: String::new(),
            ..Default::default()
        };

        let base = derive_base(storage.as_ref(), "t", &manifest).await.unwrap();

        // row_id 1 must resolve to fragment B (patch), not A (tombstoned)
        assert_eq!(lookup_in_base(&base, 1), Some((frag_b_id, 0)));
        // row_id 0 and 2 resolve to fragment A
        assert_eq!(lookup_in_base(&base, 0), Some((frag_a_id, 0)));
        assert_eq!(lookup_in_base(&base, 2), Some((frag_a_id, 2)));
    }

    // -----------------------------------------------------------------------
    // Helper: build a storage + manifest with two fragments (A has a deletion)
    // -----------------------------------------------------------------------

    async fn make_two_fragment_storage() -> (Arc<MemoryStorage>, Manifest) {
        let storage = Arc::new(MemoryStorage::new());
        let frag_a_id: u64 = 10;
        let frag_b_id: u64 = 20;

        // Fragment A: row_ids 0,1,2,3 — offset 1 deleted (tombstone for update).
        let meta_a = RowGroupMeta {
            row_group: "rg_a".to_string(),
            schema_id: 1,
            rows: 4,
            columns: Default::default(),
            column_offsets: None,
            sort: None,
            row_ids: vec![RowIdSegment::Range { start: 0, count: 4 }],
            checksum: String::new(),
            meta_checksum: String::new(),
        };
        storage
            .write("t/rg_a.meta", &serde_json::to_vec(&meta_a).unwrap())
            .await
            .unwrap();

        let mut dv_a = DeletionVector::default();
        dv_a.union_offsets([1u32]);
        storage
            .write("t/_deletions/rg_a.del", &dv_a.serialize())
            .await
            .unwrap();

        // Fragment B (patch): row_id 1 at offset 0.
        let meta_b = RowGroupMeta {
            row_group: "rg_b".to_string(),
            schema_id: 1,
            rows: 1,
            columns: Default::default(),
            column_offsets: None,
            sort: None,
            row_ids: vec![RowIdSegment::Sorted { ids: vec![1] }],
            checksum: String::new(),
            meta_checksum: String::new(),
        };
        storage
            .write("t/rg_b.meta", &serde_json::to_vec(&meta_b).unwrap())
            .await
            .unwrap();

        let manifest = Manifest {
            format_version: 1,
            sequence: 7,
            schema_id: 1,
            row_groups: vec![
                RowGroupEntry {
                    data: "rg_a.parquet".to_string(),
                    meta: "rg_a.meta".to_string(),
                    fragment_id: frag_a_id,
                    deletes: Some("_deletions/rg_a.del".to_string()),
                    deleted_count: 1,
                    agg: None,
                },
                RowGroupEntry {
                    data: "rg_b.parquet".to_string(),
                    meta: "rg_b.meta".to_string(),
                    fragment_id: frag_b_id,
                    deletes: None,
                    deleted_count: 0,
                    agg: None,
                },
            ],
            checksum: String::new(),
            ..Default::default()
        };

        (storage, manifest)
    }

    // -----------------------------------------------------------------------
    // rebuild + verify — happy path
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn verify_passes_for_rebuilt_index() {
        let (storage, mut manifest) = make_two_fragment_storage().await;

        // Rebuild writes _rowindex/base__v000000007.idx and returns a RowIndexRef.
        let gen = rebuild(storage.as_ref(), "t", &manifest).await.unwrap();

        // The base path must encode the sequence number with 9-digit padding.
        assert_eq!(gen.base.as_deref(), Some("_rowindex/base__v000000007.idx"));
        assert!(gen.deltas.is_empty());

        // Record the generation in the manifest and verify.
        manifest.rowindex_generation = Some(gen);
        verify(storage.as_ref(), "t", &manifest).await.unwrap();
    }

    // -----------------------------------------------------------------------
    // rebuild + verify — corrupt delta is detected
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn verify_fails_on_corrupt_generation() {
        let (storage, mut manifest) = make_two_fragment_storage().await;

        // Build a valid base.
        let mut gen = rebuild(storage.as_ref(), "t", &manifest).await.unwrap();

        // Craft a delta that remaps row_id 0 to a WRONG (fragment_id, offset).
        // The fresh derive says row_id 0 → (fragment 10, offset 0).
        // We'll write a delta claiming row_id 0 → (fragment 99, offset 42).
        let bad_seg = AddrSegment {
            start_row_id: 0,
            fragment_id: 99,
            start_offset: 42,
            len: 1,
        };
        let bad_bytes = encode_idx(&[bad_seg]);
        storage
            .write("t/_rowindex/bad_delta.idx", &bad_bytes)
            .await
            .unwrap();
        gen.deltas.push("_rowindex/bad_delta.idx".into());

        // Store the corrupt generation in the manifest.
        manifest.rowindex_generation = Some(gen);

        // verify must catch the mismatch and return Err.
        let result = verify(storage.as_ref(), "t", &manifest).await;
        assert!(
            result.is_err(),
            "verify should have returned Err for a corrupt delta but returned Ok"
        );
    }
}
