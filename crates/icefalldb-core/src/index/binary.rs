//! Compact, mmap-friendly binary layout for a [`BTreeIndex`]'s postings.
//!
//! This is a **derived, optional cache** written alongside the canonical JSON
//! index (the JSON stays the source of truth and stays auditable). The binary
//! form exists so that opening a large indexed table does not parse the whole
//! `BTreeMap` into memory: the reader binary-searches a fixed-size key directory
//! and decodes only the matched key's postings (the mmap reader maps the bytes;
//! this module defines the format and a slice-backed reader).
//!
//! Layout (all integers little-endian, so the format is endianness-portable):
//!
//! ```text
//! header (32 bytes):
//!   [0..8]   magic = b"BLZIDX01"
//!   [8..16]  snapshot_sequence : u64
//!   [16..24] key_count         : u64
//!   [24..32] def_len           : u64  length of the definition JSON blob
//! definition  : `IndexDefinition` as JSON (so an mmap'd index is self-describing)
//! (pad to 8)
//! directory (key_count records, 32 bytes each, sorted by key):
//!   key_off    : u64  absolute byte offset of the key bytes
//!   key_len    : u64  length of the key in bytes
//!   post_off   : u64  absolute byte offset of the postings (u64 LE each)
//!   post_count : u64  number of u64 row-ids
//! key blob    : concatenated UTF-8 key bytes (BTreeMap order)
//! postings    : row-ids, 8 bytes each (u64 LE), grouped per key
//! ```
//!
//! Keys are written in `BTreeMap` iteration order, which is the same ordering
//! `lookup` binary-searches with — `String`/`str` ordering equals UTF-8 byte
//! order. Any inconsistency (bad magic, truncation, out-of-range offset) makes
//! [`BinaryIndexRef::parse`] return `None` so the caller falls back to JSON.

use super::{BTreeIndex, IndexDefinition};

const MAGIC: &[u8; 8] = b"BLZIDX01";
const HEADER_LEN: usize = 32;
const DIR_ENTRY_LEN: usize = 32;

fn align8(n: usize) -> usize {
    (n + 7) & !7
}

fn u64_at(bytes: &[u8], off: usize) -> Option<u64> {
    let end = off.checked_add(8)?;
    let slice = bytes.get(off..end)?;
    Some(u64::from_le_bytes(slice.try_into().ok()?))
}

/// Serialize an index's postings into the binary format described above.
pub fn serialize(index: &BTreeIndex) -> Vec<u8> {
    let key_count = index.entries.len();
    // The definition is embedded as JSON so an mmap'd index is self-describing
    // (the reader needs the column/unique without parsing the postings or the
    // catalog). serde_json on a single small struct never fails in practice.
    let def_json = serde_json::to_vec(&index.definition).unwrap_or_default();

    // Compute region offsets. Postings are 8-byte aligned from the file start
    // (the mmap reader maps from a page-aligned base, so postings land 8-aligned and
    // a future zero-copy `&[u64]` view is possible).
    let dir_off = align8(HEADER_LEN + def_json.len());
    let key_blob_off = dir_off + key_count * DIR_ENTRY_LEN;
    let key_blob_len: usize = index.entries.keys().map(|k| k.len()).sum();
    let postings_off = align8(key_blob_off + key_blob_len);
    let total_postings: usize = index.entries.values().map(|v| v.len()).sum();
    let total_len = postings_off + total_postings * 8;

    let mut buf = vec![0u8; total_len];
    buf[0..8].copy_from_slice(MAGIC);
    buf[8..16].copy_from_slice(&index.snapshot_sequence.to_le_bytes());
    buf[16..24].copy_from_slice(&(key_count as u64).to_le_bytes());
    buf[24..32].copy_from_slice(&(def_json.len() as u64).to_le_bytes());
    buf[HEADER_LEN..HEADER_LEN + def_json.len()].copy_from_slice(&def_json);

    let mut key_cursor = key_blob_off;
    let mut post_cursor = postings_off;
    for (i, (key, ids)) in index.entries.iter().enumerate() {
        let dir = dir_off + i * DIR_ENTRY_LEN;
        buf[dir..dir + 8].copy_from_slice(&(key_cursor as u64).to_le_bytes());
        buf[dir + 8..dir + 16].copy_from_slice(&(key.len() as u64).to_le_bytes());
        buf[dir + 16..dir + 24].copy_from_slice(&(post_cursor as u64).to_le_bytes());
        buf[dir + 24..dir + 32].copy_from_slice(&(ids.len() as u64).to_le_bytes());

        buf[key_cursor..key_cursor + key.len()].copy_from_slice(key.as_bytes());
        key_cursor += key.len();

        for id in ids {
            buf[post_cursor..post_cursor + 8].copy_from_slice(&id.to_le_bytes());
            post_cursor += 8;
        }
    }

    buf
}

/// A read-only view over binary-index bytes (a slice today, an mmap in 1.3).
pub struct BinaryIndexRef<'a> {
    bytes: &'a [u8],
    key_count: usize,
    snapshot_sequence: u64,
    dir_off: usize,
    definition: IndexDefinition,
}

impl<'a> BinaryIndexRef<'a> {
    /// Validate and wrap `bytes`. Returns `None` for any malformed input
    /// (bad magic, truncated header/directory, undecodable definition) so the
    /// caller falls back to JSON.
    pub fn parse(bytes: &'a [u8]) -> Option<Self> {
        if bytes.len() < HEADER_LEN || &bytes[0..8] != MAGIC {
            return None;
        }
        let snapshot_sequence = u64_at(bytes, 8)?;
        let key_count = u64_at(bytes, 16)? as usize;
        let def_len = u64_at(bytes, 24)? as usize;
        let def_end = HEADER_LEN.checked_add(def_len)?;
        let def_bytes = bytes.get(HEADER_LEN..def_end)?;
        let definition: IndexDefinition = serde_json::from_slice(def_bytes).ok()?;
        let dir_off = align8(def_end);
        // The directory must fit within the buffer.
        let dir_end = dir_off.checked_add(key_count.checked_mul(DIR_ENTRY_LEN)?)?;
        if bytes.len() < dir_end {
            return None;
        }
        Some(Self {
            bytes,
            key_count,
            snapshot_sequence,
            dir_off,
            definition,
        })
    }

    /// The snapshot sequence this binary index was built for.
    pub fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    /// The index definition (column, unique flag, name) embedded in the file.
    pub fn definition(&self) -> &IndexDefinition {
        &self.definition
    }

    /// Number of distinct keys.
    pub fn key_count(&self) -> usize {
        self.key_count
    }

    /// `(key, postings)` for the `i`-th directory entry, or `None` if any
    /// recorded offset/length is out of range (garbage → caller falls back).
    fn entry(&self, i: usize) -> Option<(&'a str, Vec<u64>)> {
        let dir = self.dir_off + i * DIR_ENTRY_LEN;
        let key_off = u64_at(self.bytes, dir)? as usize;
        let key_len = u64_at(self.bytes, dir + 8)? as usize;
        let post_off = u64_at(self.bytes, dir + 16)? as usize;
        let post_count = u64_at(self.bytes, dir + 24)? as usize;

        let key_bytes = self.bytes.get(key_off..key_off.checked_add(key_len)?)?;
        let key = std::str::from_utf8(key_bytes).ok()?;

        let post_bytes_end = post_off.checked_add(post_count.checked_mul(8)?)?;
        let post_bytes = self.bytes.get(post_off..post_bytes_end)?;
        let ids = post_bytes
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        Some((key, ids))
    }

    /// Just the key at directory index `i` (for binary search), bounds-checked.
    fn key_at(&self, i: usize) -> Option<&'a str> {
        let dir = self.dir_off + i * DIR_ENTRY_LEN;
        let key_off = u64_at(self.bytes, dir)? as usize;
        let key_len = u64_at(self.bytes, dir + 8)? as usize;
        let key_bytes = self.bytes.get(key_off..key_off.checked_add(key_len)?)?;
        std::str::from_utf8(key_bytes).ok()
    }

    /// Row IDs recorded for `value`; empty if the key is absent (matching
    /// [`BTreeIndex::lookup`]). Binary search over the sorted key directory —
    /// only the matched key's postings are decoded.
    pub fn lookup_checked(&self, value: &str) -> Option<Vec<u64>> {
        let mut lo = 0usize;
        let mut hi = self.key_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.key_at(mid) {
                Some(k) => match value.cmp(k) {
                    std::cmp::Ordering::Equal => {
                        return self.entry(mid).map(|(_, ids)| ids);
                    }
                    std::cmp::Ordering::Less => hi = mid,
                    std::cmp::Ordering::Greater => lo = mid + 1,
                },
                None => return None,
            }
        }
        Some(Vec::new())
    }

    /// Row IDs recorded for `value`; empty if the key is absent or if the
    /// derived cache is malformed. Production callers that can fall back to the
    /// canonical JSON/scan path should use [`Self::lookup_checked`] instead.
    pub fn lookup(&self, value: &str) -> Vec<u64> {
        self.lookup_checked(value).unwrap_or_default()
    }
}

/// An mmap-backed binary index: owns the file mapping and answers `lookup` by
/// binary-searching the mapped directory, decoding only the matched key's
/// postings. Opening it does **not** read or parse the whole index — the OS
/// pages in only the directory pages and the looked-up key's postings.
pub struct MmapBinaryIndex {
    mmap: memmap2::Mmap,
    definition: IndexDefinition,
    snapshot_sequence: u64,
}

impl MmapBinaryIndex {
    /// The embedded index definition (column, unique flag, name).
    pub fn definition(&self) -> &IndexDefinition {
        &self.definition
    }

    /// The snapshot sequence this binary index was built for.
    pub fn snapshot_sequence(&self) -> u64 {
        self.snapshot_sequence
    }

    /// Row ids recorded for `value` (empty if absent), by binary search over
    /// the mapped directory.
    pub fn lookup(&self, value: &str) -> Vec<u64> {
        self.lookup_checked(value).unwrap_or_default()
    }

    /// Row ids recorded for `value`, or `None` when the mapped derived cache is
    /// malformed and the caller should fall back to the canonical path.
    pub fn lookup_checked(&self, value: &str) -> Option<Vec<u64>> {
        BinaryIndexRef::parse(&self.mmap)?.lookup_checked(value)
    }
}

/// Try to mmap the derived binary index that sits next to the JSON base at
/// `base_rel` (a table-relative `…/base__v<seq>.json` path; the `.idx` sibling
/// is mapped). Returns `None` — so the caller falls back to the JSON base — when
/// the file is absent or fails validation.
pub fn open_mmap_binary_index(
    local_root: &std::path::Path,
    table: &str,
    base_rel: &str,
) -> Option<MmapBinaryIndex> {
    let stem = base_rel.strip_suffix(".json")?;
    let path = local_root.join(table).join(format!("{stem}.idx"));
    let file = std::fs::File::open(&path).ok()?;
    // SAFETY: a versioned `.idx` is uniquely named per generation and never
    // rewritten; the legacy `_indexes/<name>.idx` is replaced via tmp+rename, so
    // an update swaps the directory entry rather than mutating this inode in
    // place. Either way the bytes backing an opened mapping are never modified
    // underneath us. (Only reached for local storage, where this holds.)
    let mmap = unsafe { memmap2::Mmap::map(&file).ok()? };
    let (definition, snapshot_sequence) = {
        let r = BinaryIndexRef::parse(&mmap)?;
        (r.definition().clone(), r.snapshot_sequence())
    };
    Some(MmapBinaryIndex {
        mmap,
        definition,
        snapshot_sequence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{BTreeIndex, IndexDefinition};
    use std::collections::BTreeMap;

    fn sample_index() -> BTreeIndex {
        let mut entries: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        // Multi-posting key, single-posting keys, and lexicographically tricky
        // neighbours so binary search ordering is exercised.
        entries.insert("alpha".into(), vec![1, 7, 42, 1000]);
        entries.insert("beta".into(), vec![2]);
        entries.insert("gamma".into(), vec![3, 9]);
        entries.insert("10".into(), vec![11]);
        entries.insert("2".into(), vec![22, 23]);
        BTreeIndex {
            definition: IndexDefinition {
                name: "idx".into(),
                table: "t".into(),
                column: "c".into(),
                unique: false,
            },
            snapshot_sequence: 17,
            entries,
        }
    }

    #[test]
    fn binary_lookup_byte_equal_to_json_backed_index() {
        let json = sample_index();
        let bytes = serialize(&json);
        let bin = BinaryIndexRef::parse(&bytes).expect("valid binary index must parse");

        assert_eq!(bin.snapshot_sequence(), json.snapshot_sequence);
        assert_eq!(bin.key_count(), json.entries.len());
        assert_eq!(bin.definition().column, json.definition.column);
        assert_eq!(bin.definition().name, json.definition.name);
        assert_eq!(bin.definition().unique, json.definition.unique);

        // Every present key resolves byte-equal to the JSON index.
        for (key, ids) in &json.entries {
            assert_eq!(&bin.lookup(key), ids, "postings for key {key:?} must match");
        }

        // Missing keys (before, between, after the key range) resolve to empty,
        // matching BTreeIndex::lookup.
        for missing in ["", "0", "aaa", "alpz", "zzz"] {
            assert_eq!(bin.lookup(missing), json.lookup(missing).to_vec());
            assert!(bin.lookup(missing).is_empty());
        }
    }

    #[test]
    fn empty_index_round_trips() {
        let json = BTreeIndex {
            definition: IndexDefinition {
                name: "idx".into(),
                table: "t".into(),
                column: "c".into(),
                unique: false,
            },
            snapshot_sequence: 0,
            entries: BTreeMap::new(),
        };
        let bytes = serialize(&json);
        let bin = BinaryIndexRef::parse(&bytes).expect("empty binary index must parse");
        assert_eq!(bin.key_count(), 0);
        assert!(bin.lookup("anything").is_empty());
    }

    #[test]
    fn garbage_bytes_do_not_parse() {
        assert!(BinaryIndexRef::parse(b"").is_none());
        assert!(BinaryIndexRef::parse(b"not-a-magic-header-padding").is_none());
        // Valid magic but a key_count that overflows the buffer.
        let mut bad = vec![0u8; HEADER_LEN];
        bad[0..8].copy_from_slice(MAGIC);
        bad[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(BinaryIndexRef::parse(&bad).is_none());
    }
}
