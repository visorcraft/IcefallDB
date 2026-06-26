//! Reader for the `_rowindex` on-disk format.
//!
//! # Two reader back-ends
//!
//! - [`MmapBase`]: local files — `mmap` the `.idx` file and binary-search
//!   without loading the whole segment array into RAM.  This is the fast path
//!   for writers running on the same host as the data.
//!
//! - [`RangedBase`]: remote / object-store files — reads the full file via the
//!   [`Storage`] trait and decodes it with [`decode_idx`].  The block index
//!   embedded in the file format would allow true ranged reads (one small HTTP
//!   range to hit the block-index entry, one more to fetch the candidate
//!   segment window), but that optimisation is left for after the POC.
//!
//! Both expose the same `lookup(row_id) -> Option<(fragment_id, offset)>`
//! interface and delegate the containment check to [`lookup_in_base`], which is
//! the pure, unit-tested core.

use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::Result;
use crate::metadata::manifest::RowIndexRef;
use crate::rowindex::writer::{HEADER_LEN, SEGMENT_LEN};
use crate::rowindex::{decode_idx, AddrSegment};
use crate::storage::Storage;

// ---------------------------------------------------------------------------
// Byte-layout constants (imported from writer.rs to avoid divergence)
// ---------------------------------------------------------------------------

/// `segment_count` lives at header offset 8, 8 bytes.
const HDR_OFF_SEG_COUNT: usize = 8;

// Intra-segment field offsets (relative to the start of each segment).
const SEG_OFF_START_ROW_ID: usize = 0; // u64 (8 bytes)
const SEG_OFF_FRAGMENT_ID: usize = 8; // u64 (8 bytes)
const SEG_OFF_START_OFFSET: usize = 16; // u32 (4 bytes)
                                        // SEG_OFF_LEN = 20 (u32, 4 bytes) — only needed at the candidate segment.
const SEG_OFF_LEN: usize = 20;

// ---------------------------------------------------------------------------
// Pure core: lookup_in_base
// ---------------------------------------------------------------------------

/// Find which (fragment, offset) a `row_id` belongs to, given a slice of
/// sorted [`AddrSegment`]s.
///
/// Uses a predecessor search (`partition_point`) followed by a **mandatory
/// containment check** (`start_row_id <= row_id < start_row_id + len`).
/// Rows that fall in gaps between segments, or past the last segment, return
/// `None`.
///
/// On a hit returns `(fragment_id, start_offset + (row_id - start_row_id) as u32)`.
pub fn lookup_in_base(segments: &[AddrSegment], row_id: u64) -> Option<(u64, u32)> {
    // Find the last segment whose start_row_id <= row_id.
    let idx = segments.partition_point(|s| s.start_row_id <= row_id);
    if idx == 0 {
        return None; // row_id is before the very first segment
    }
    let seg = &segments[idx - 1];
    // Containment: row_id must be strictly inside [start_row_id, start_row_id + len)
    if row_id < seg.start_row_id.saturating_add(u64::from(seg.len)) {
        let offset = seg.start_offset + (row_id - seg.start_row_id) as u32;
        Some((seg.fragment_id, offset))
    } else {
        None // row_id is past the end of the predecessor segment (gap or past end)
    }
}

// ---------------------------------------------------------------------------
// MmapBase — local file, mmap'd, binary search without decoding full array
// ---------------------------------------------------------------------------

/// Local-file reader that `mmap`s the `.idx` file and binary-searches the
/// segment region without loading the whole array into heap memory.
///
/// Opening is cheap (one `mmap` syscall); each `lookup` does O(log n) byte
/// reads within the mmap, and decodes only the final candidate segment.
pub struct MmapBase {
    mmap: Mmap,
    seg_count: usize,
}

impl MmapBase {
    /// Open and memory-map a `_rowindex` file at `path`.
    ///
    /// Does **not** verify the CRC32 trailer — that is done by the writer at
    /// write-time (or explicitly via [`decode_idx`] if a full integrity check
    /// is needed).
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(crate::error::IcefallDBError::Io)?;
        // SAFETY: The standard `memmap2::Mmap::map` requirement is that the
        // underlying file must not be concurrently truncated.  `.idx` files are
        // written once and never mutated in place, so this is safe.
        let mmap = unsafe { Mmap::map(&file) }.map_err(crate::error::IcefallDBError::Io)?;

        if mmap.len() < HEADER_LEN {
            return Err(crate::error::IcefallDBError::Other(
                "rowindex: file too short for header".into(),
            ));
        }

        // Read segment_count from the header (offset 8, 8 bytes LE).
        let seg_count = u64::from_le_bytes(
            mmap[HDR_OFF_SEG_COUNT..HDR_OFF_SEG_COUNT + 8]
                .try_into()
                .map_err(|_| {
                    crate::error::IcefallDBError::Other(
                        "rowindex: internal error reading segment_count".into(),
                    )
                })?,
        ) as usize;

        Ok(Self { mmap, seg_count })
    }

    /// Return the `start_row_id` of segment `i` by reading directly from the
    /// mmap buffer — no heap allocation, no full segment decode.
    #[inline]
    fn seg_start_row_id(&self, i: usize) -> u64 {
        let off = HEADER_LEN + i * SEGMENT_LEN + SEG_OFF_START_ROW_ID;
        u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap())
    }

    /// Decode the full segment at index `i` from the mmap buffer.
    #[inline]
    fn decode_seg(&self, i: usize) -> AddrSegment {
        let base = HEADER_LEN + i * SEGMENT_LEN;
        let start_row_id = u64::from_le_bytes(
            self.mmap[base + SEG_OFF_START_ROW_ID..base + SEG_OFF_START_ROW_ID + 8]
                .try_into()
                .unwrap(),
        );
        let fragment_id = u64::from_le_bytes(
            self.mmap[base + SEG_OFF_FRAGMENT_ID..base + SEG_OFF_FRAGMENT_ID + 8]
                .try_into()
                .unwrap(),
        );
        let start_offset = u32::from_le_bytes(
            self.mmap[base + SEG_OFF_START_OFFSET..base + SEG_OFF_START_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        let len = u32::from_le_bytes(
            self.mmap[base + SEG_OFF_LEN..base + SEG_OFF_LEN + 4]
                .try_into()
                .unwrap(),
        );
        AddrSegment {
            start_row_id,
            fragment_id,
            start_offset,
            len,
        }
    }

    /// Look up `row_id` and return `(fragment_id, offset)` if it is contained
    /// by a segment.
    pub fn lookup(&self, row_id: u64) -> Option<(u64, u32)> {
        // Binary predecessor search: find the last index whose start_row_id <= row_id.
        // partition_point is a slice method, so we do the search manually over [0, seg_count).
        if self.seg_count == 0 {
            return None;
        }
        let mut lo = 0usize;
        let mut hi = self.seg_count; // exclusive
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if self.seg_start_row_id(mid) <= row_id {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // lo is now the largest index with start_row_id <= row_id, unless even index 0 fails.
        if self.seg_start_row_id(lo) > row_id {
            return None;
        }
        let seg = self.decode_seg(lo);
        // Reuse the pure containment logic.
        lookup_in_base(std::slice::from_ref(&seg), row_id)
    }
}

// ---------------------------------------------------------------------------
// RangedBase — remote / Storage-backed reader
// ---------------------------------------------------------------------------

/// Remote-file reader that fetches the full `.idx` file via the [`Storage`]
/// trait and decodes it with [`decode_idx`].
///
/// The block index embedded in the file format enables future optimisation to
/// two ranged reads (block-index fetch + candidate-segment fetch), avoiding a
/// full file download for large indexes.  That optimisation is left as a later
/// improvement; for the POC a full read is acceptable because `.idx` files are
/// compact.
pub struct RangedBase {
    storage: Arc<dyn Storage>,
    path: String,
}

impl RangedBase {
    /// Create a reader backed by `storage`, reading from `path`.
    pub fn new(storage: Arc<dyn Storage>, path: impl Into<String>) -> Self {
        Self {
            storage,
            path: path.into(),
        }
    }

    /// Look up `row_id`.  This is `async` because it may perform I/O.
    pub async fn lookup(&self, row_id: u64) -> Result<Option<(u64, u32)>> {
        let buf = self.storage.read(&self.path).await?;
        let decoded = decode_idx(&buf)?;
        Ok(lookup_in_base(decoded.segments(), row_id))
    }
}

// ---------------------------------------------------------------------------
// AddressMap — LSM read path: base + ordered deltas
// ---------------------------------------------------------------------------

/// Combined address map consisting of an optional base index plus an ordered
/// sequence of per-commit delta indexes.
///
/// On lookup, deltas are checked NEWEST-FIRST (i.e. in reverse order of how
/// they are stored in [`RowIndexRef::deltas`], which is oldest-first).  The
/// first delta that contains the row ID wins; if no delta matches, the base is
/// consulted.
///
/// Both the base and each delta are stored as decoded `Vec<AddrSegment>` to
/// avoid lifetime and `Send` complications from holding an `Mmap` across `await`
/// points.  The mmap-zero-copy base path can be wired in a follow-up once the
/// POC stabilises.
pub struct AddressMap {
    base: Vec<AddrSegment>,
    deltas: Vec<Vec<AddrSegment>>,
}

impl AddressMap {
    /// Construct an `AddressMap` directly from decoded segment slices.
    ///
    /// `base_segments` is the base index (may be empty for a table with no base
    /// yet).  `deltas` is an ordered slice where the *last* element is the
    /// newest delta (i.e. oldest-first, the same ordering used in
    /// [`RowIndexRef::deltas`]).
    pub fn from_parts(base_segments: Vec<AddrSegment>, deltas: Vec<Vec<AddrSegment>>) -> Self {
        Self {
            base: base_segments,
            deltas,
        }
    }

    /// Open the address map described by `gen` from `storage`, rooted under
    /// the table directory `table`.
    ///
    /// File paths stored in `gen` are relative to the table directory; this
    /// function builds the storage-relative path as `"{table}/{rel_path}"`.
    ///
    /// An empty `gen` (no base, no deltas) is valid and produces an
    /// `AddressMap` that returns `None` for all lookups.
    pub async fn open(storage: &dyn Storage, table: &str, gen: &RowIndexRef) -> Result<Self> {
        // Decode the base, if present.
        let base = if let Some(ref rel) = gen.base {
            let path = format!("{table}/{rel}");
            let buf = storage.read(&path).await?;
            let decoded = decode_idx(&buf)?;
            decoded.segments().to_vec()
        } else {
            Vec::new()
        };

        // Decode each delta eagerly (deltas are small).
        let mut deltas = Vec::with_capacity(gen.deltas.len());
        for rel in &gen.deltas {
            let path = format!("{table}/{rel}");
            let buf = storage.read(&path).await?;
            let decoded = decode_idx(&buf)?;
            deltas.push(decoded.segments().to_vec());
        }

        Ok(Self { base, deltas })
    }

    /// Look up `row_id`, checking deltas NEWEST-FIRST, then the base.
    ///
    /// Returns `Some((fragment_id, offset))` on a hit, or `None` if the row ID
    /// is not present in any layer.
    pub fn lookup(&self, row_id: u64) -> Option<(u64, u32)> {
        // Iterate deltas in reverse (newest first).
        for delta in self.deltas.iter().rev() {
            if let Some(result) = lookup_in_base(delta, row_id) {
                return Some(result);
            }
        }
        // Fall through to the base.
        lookup_in_base(&self.base, row_id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rowindex::{encode_idx, AddrSegment};

    // -----------------------------------------------------------------------
    // Pure core: containment check
    // -----------------------------------------------------------------------

    #[test]
    fn lookup_requires_containment() {
        let segs = vec![
            AddrSegment {
                start_row_id: 0,
                fragment_id: 1,
                start_offset: 0,
                len: 1,
            }, // id 0 only
            AddrSegment {
                start_row_id: 2,
                fragment_id: 1,
                start_offset: 1,
                len: 1,
            }, // id 2 only
        ];
        assert_eq!(lookup_in_base(&segs, 0), Some((1, 0)));
        assert_eq!(lookup_in_base(&segs, 2), Some((1, 1)));
        assert_eq!(lookup_in_base(&segs, 1), None); // hole — MUST NOT map to offset 1
        assert_eq!(lookup_in_base(&segs, 9), None); // past end
    }

    #[test]
    fn lookup_empty_slice() {
        assert_eq!(lookup_in_base(&[], 0), None);
        assert_eq!(lookup_in_base(&[], 100), None);
    }

    #[test]
    fn lookup_multi_row_segment() {
        let segs = vec![AddrSegment {
            start_row_id: 10,
            fragment_id: 7,
            start_offset: 50,
            len: 5,
        }];
        // Rows 10..=14 are inside; 15 is past end; 9 is before start.
        assert_eq!(lookup_in_base(&segs, 9), None);
        assert_eq!(lookup_in_base(&segs, 10), Some((7, 50)));
        assert_eq!(lookup_in_base(&segs, 12), Some((7, 52)));
        assert_eq!(lookup_in_base(&segs, 14), Some((7, 54)));
        assert_eq!(lookup_in_base(&segs, 15), None);
    }

    // -----------------------------------------------------------------------
    // On-disk paths: MmapBase and RangedBase agree with lookup_in_base
    // -----------------------------------------------------------------------

    fn make_test_segs() -> Vec<AddrSegment> {
        vec![
            AddrSegment {
                start_row_id: 0,
                fragment_id: 1,
                start_offset: 0,
                len: 3,
            },
            AddrSegment {
                start_row_id: 5,
                fragment_id: 2,
                start_offset: 10,
                len: 2,
            },
            AddrSegment {
                start_row_id: 10,
                fragment_id: 3,
                start_offset: 0,
                len: 5,
            },
        ]
    }

    #[test]
    fn mmap_base_matches_pure_lookup() {
        let segs = make_test_segs();
        let encoded = encode_idx(&segs);

        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), &encoded).unwrap();

        let reader = MmapBase::open(tmp.path()).expect("MmapBase::open should succeed");

        let probe_ids: &[u64] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 100];
        for &row_id in probe_ids {
            let expected = lookup_in_base(&segs, row_id);
            let got = reader.lookup(row_id);
            assert_eq!(
                got, expected,
                "MmapBase::lookup({row_id}) mismatch: got {got:?}, expected {expected:?}"
            );
        }
    }

    #[tokio::test]
    async fn ranged_base_matches_pure_lookup() {
        use crate::storage::memory::MemoryStorage;

        let segs = make_test_segs();
        let encoded = encode_idx(&segs);

        let store = Arc::new(MemoryStorage::new());
        store.write("test.idx", &encoded).await.unwrap();

        let reader = RangedBase::new(store, "test.idx");

        let probe_ids: &[u64] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 100];
        for &row_id in probe_ids {
            let expected = lookup_in_base(&segs, row_id);
            let got = reader.lookup(row_id).await.unwrap();
            assert_eq!(
                got, expected,
                "RangedBase::lookup({row_id}) mismatch: got {got:?}, expected {expected:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // AddressMap: LSM read path tests
    // -----------------------------------------------------------------------

    /// Acceptance criterion from the brief: newest delta overrides the base for
    /// a row that has been relocated.
    #[test]
    fn delta_overrides_base() {
        let base = vec![AddrSegment {
            start_row_id: 5,
            fragment_id: 1,
            start_offset: 5,
            len: 1,
        }];
        let delta = vec![AddrSegment {
            start_row_id: 5,
            fragment_id: 9,
            start_offset: 0,
            len: 1,
        }]; // relocated
        let am = AddressMap::from_parts(base, vec![delta]);
        assert_eq!(am.lookup(5), Some((9, 0)));
    }

    /// A row relocated twice must resolve to its LATEST address (newest delta wins).
    #[test]
    fn newest_delta_wins_when_relocated_twice() {
        let base = vec![AddrSegment {
            start_row_id: 3,
            fragment_id: 1,
            start_offset: 3,
            len: 1,
        }];
        // delta[0] = oldest relocation: row 3 -> fragment 2
        let delta_old = vec![AddrSegment {
            start_row_id: 3,
            fragment_id: 2,
            start_offset: 0,
            len: 1,
        }];
        // delta[1] = newest relocation: row 3 -> fragment 7
        let delta_new = vec![AddrSegment {
            start_row_id: 3,
            fragment_id: 7,
            start_offset: 99,
            len: 1,
        }];
        // deltas stored oldest-first; lookup must reverse iterate
        let am = AddressMap::from_parts(base, vec![delta_old, delta_new]);
        assert_eq!(am.lookup(3), Some((7, 99)));
    }

    /// An empty AddressMap (no base, no deltas) must return None for any lookup.
    #[test]
    fn empty_address_map_returns_none() {
        let am = AddressMap::from_parts(vec![], vec![]);
        assert_eq!(am.lookup(0), None);
        assert_eq!(am.lookup(u64::MAX), None);
    }

    /// Rows not in any delta must still resolve to the base.
    #[test]
    fn base_fallthrough_for_unrelocated_rows() {
        let base = vec![AddrSegment {
            start_row_id: 0,
            fragment_id: 1,
            start_offset: 0,
            len: 10,
        }];
        // Delta only relocates row 5; other rows should still resolve via the base.
        let delta = vec![AddrSegment {
            start_row_id: 5,
            fragment_id: 9,
            start_offset: 0,
            len: 1,
        }];
        let am = AddressMap::from_parts(base, vec![delta]);
        // Relocated row: delta wins.
        assert_eq!(am.lookup(5), Some((9, 0)));
        // Non-relocated rows: base wins.
        assert_eq!(am.lookup(0), Some((1, 0)));
        assert_eq!(am.lookup(3), Some((1, 3)));
        // Row past the base: None.
        assert_eq!(am.lookup(10), None);
    }

    /// End-to-end test: encode base + delta to MemoryStorage, open via
    /// `AddressMap::open`, and confirm newest-first override.
    #[tokio::test]
    async fn open_from_storage_delta_overrides_base() {
        use crate::metadata::manifest::RowIndexRef;
        use crate::storage::memory::MemoryStorage;

        let base_segs = vec![
            AddrSegment {
                start_row_id: 5,
                fragment_id: 1,
                start_offset: 5,
                len: 1,
            },
            AddrSegment {
                start_row_id: 10,
                fragment_id: 1,
                start_offset: 10,
                len: 5,
            },
        ];
        let delta_segs = vec![
            AddrSegment {
                start_row_id: 5,
                fragment_id: 9,
                start_offset: 0,
                len: 1,
            }, // relocate row 5
        ];

        let store = Arc::new(MemoryStorage::new());
        store
            .write("tbl/base.idx", &encode_idx(&base_segs))
            .await
            .unwrap();
        store
            .write("tbl/delta0.idx", &encode_idx(&delta_segs))
            .await
            .unwrap();

        let gen = RowIndexRef {
            base: Some("base.idx".into()),
            deltas: vec!["delta0.idx".into()],
        };

        let am = AddressMap::open(store.as_ref(), "tbl", &gen).await.unwrap();

        // Delta overrides base for row 5.
        assert_eq!(am.lookup(5), Some((9, 0)));
        // Row 10 falls through to base.
        assert_eq!(am.lookup(10), Some((1, 10)));
        // Row 12 is in the base range [10, 15).
        assert_eq!(am.lookup(12), Some((1, 12)));
        // Row not in any layer.
        assert_eq!(am.lookup(99), None);
    }

    /// End-to-end test: a row relocated twice via two delta files resolves to
    /// the latest delta when opened from storage.
    #[tokio::test]
    async fn open_from_storage_newest_delta_wins_twice_relocated() {
        use crate::metadata::manifest::RowIndexRef;
        use crate::storage::memory::MemoryStorage;

        let base_segs = vec![AddrSegment {
            start_row_id: 7,
            fragment_id: 1,
            start_offset: 7,
            len: 1,
        }];
        // Oldest delta: row 7 -> fragment 2
        let delta0 = vec![AddrSegment {
            start_row_id: 7,
            fragment_id: 2,
            start_offset: 0,
            len: 1,
        }];
        // Newest delta: row 7 -> fragment 5
        let delta1 = vec![AddrSegment {
            start_row_id: 7,
            fragment_id: 5,
            start_offset: 42,
            len: 1,
        }];

        let store = Arc::new(MemoryStorage::new());
        store
            .write("tbl/base.idx", &encode_idx(&base_segs))
            .await
            .unwrap();
        store
            .write("tbl/d0.idx", &encode_idx(&delta0))
            .await
            .unwrap();
        store
            .write("tbl/d1.idx", &encode_idx(&delta1))
            .await
            .unwrap();

        let gen = RowIndexRef {
            base: Some("base.idx".into()),
            // deltas stored oldest-first
            deltas: vec!["d0.idx".into(), "d1.idx".into()],
        };

        let am = AddressMap::open(store.as_ref(), "tbl", &gen).await.unwrap();
        // Must resolve to the newest delta (fragment 5, offset 42).
        assert_eq!(am.lookup(7), Some((5, 42)));
    }
}
