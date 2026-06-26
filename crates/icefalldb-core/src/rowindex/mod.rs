pub mod reader;
pub mod writer;

pub use reader::{lookup_in_base, AddressMap, MmapBase, RangedBase};
pub use writer::{decode_idx, derive_base, encode_idx, rebuild, verify, DecodedIdx};

/// A single entry in the `_rowindex` file.
///
/// Maps a contiguous range of stable row IDs `[start_row_id, start_row_id + len)`
/// to the physical location `(fragment_id, start_offset)` within that fragment's
/// Parquet file. Fixed width: 24 bytes, all fields little-endian.
///
/// Layout (little-endian):
/// ```text
/// offset  size  field
///   0       8   start_row_id  (u64)
///   8       8   fragment_id   (u64)
///  16       4   start_offset  (u32)
///  20       4   len           (u32)
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AddrSegment {
    /// First stable row ID in the range covered by this entry.
    pub start_row_id: u64,
    /// Fragment (row-group) that physically holds these rows.
    pub fragment_id: u64,
    /// Byte offset of the first row within the fragment's Parquet file.
    pub start_offset: u32,
    /// Number of rows covered by this entry.
    pub len: u32,
}

impl AddrSegment {
    pub const ENCODED_LEN: usize = 24;
}
