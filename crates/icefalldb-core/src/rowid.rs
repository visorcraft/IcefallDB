use serde::{Deserialize, Serialize};

/// A compact encoding of stable row identifiers assigned to a fragment.
///
/// `Range` covers the common case where all rows in a fragment receive a
/// contiguous block of IDs.  `Sorted` handles the rare case where a fragment
/// contains rows whose IDs are not contiguous (e.g. after a merge that
/// combines non-adjacent ranges).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum RowIdSegment {
    Range { start: u64, count: u64 },
    Sorted { ids: Vec<u64> },
}

/// Allocate `count` contiguous row IDs starting at `*next`, bump `*next`
/// past the allocated range, and return the resulting [`RowIdSegment::Range`].
pub fn allocate_range(next: &mut u64, count: u64) -> RowIdSegment {
    let start = *next;
    *next += count;
    RowIdSegment::Range { start, count }
}

/// Iterate over every row ID described by `seg` in ascending order.
pub fn segment_ids(seg: &RowIdSegment) -> impl Iterator<Item = u64> + '_ {
    match seg {
        RowIdSegment::Range { start, count } => SegmentIter::Range(*start..*start + *count),
        RowIdSegment::Sorted { ids } => SegmentIter::Sorted(ids.iter().copied()),
    }
}

// Private iterator helper that unifies the two cases without boxing.
enum SegmentIter<'a> {
    Range(std::ops::Range<u64>),
    Sorted(std::iter::Copied<std::slice::Iter<'a, u64>>),
}

impl Iterator for SegmentIter<'_> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        match self {
            SegmentIter::Range(r) => r.next(),
            SegmentIter::Sorted(i) => i.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_fragment_is_one_contiguous_range() {
        let mut next = 100u64;
        let seg = allocate_range(&mut next, 3);
        assert_eq!(
            seg,
            RowIdSegment::Range {
                start: 100,
                count: 3
            }
        );
        assert_eq!(next, 103);
        assert_eq!(segment_ids(&seg).collect::<Vec<_>>(), vec![100, 101, 102]);
    }
}
