//! Range coalescing helper for column-aware Parquet reads.

use std::ops::Range;

/// Sort `ranges` by start and merge any two where `next.start <= prev.end + window`.
///
/// The merged ranges are returned; the input vector is drained and left empty.
pub fn coalesce_ranges(ranges: &mut Vec<Range<u64>>, window: u64) -> Vec<Range<u64>> {
    ranges.sort_by_key(|r| r.start);
    let mut out: Vec<Range<u64>> = Vec::new();
    for next in ranges.drain(..) {
        if let Some(prev) = out.last_mut() {
            if next.start <= prev.end.saturating_add(window) {
                prev.end = prev.end.max(next.end);
                continue;
            }
        }
        out.push(next);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disjoint_ranges_stay_separate() {
        let mut ranges = vec![0..10, 20..30, 40..50];
        let coalesced = coalesce_ranges(&mut ranges, 5);
        assert_eq!(coalesced, vec![0..10, 20..30, 40..50]);
        assert!(ranges.is_empty());
    }

    #[test]
    fn test_overlapping_ranges_merge() {
        let mut ranges = vec![0..10, 5..15, 20..30];
        let coalesced = coalesce_ranges(&mut ranges, 0);
        assert_eq!(coalesced, vec![0..15, 20..30]);
    }

    #[test]
    fn test_window_merges_nearby_ranges() {
        let mut ranges = vec![0..10, 15..25, 30..40];
        let coalesced = coalesce_ranges(&mut ranges, 5);
        assert_eq!(coalesced, vec![0..40]);
    }

    #[test]
    fn test_unsorted_input() {
        let mut ranges = vec![50..60, 10..20, 15..25, 0..5];
        let coalesced = coalesce_ranges(&mut ranges, 0);
        assert_eq!(coalesced, vec![0..5, 10..25, 50..60]);
    }

    #[test]
    fn test_empty_input() {
        let mut ranges: Vec<Range<u64>> = vec![];
        let coalesced = coalesce_ranges(&mut ranges, 10);
        assert!(coalesced.is_empty());
    }
}
