use roaring::RoaringBitmap;

/// A deletion vector tracks which physical row offsets within a fragment have
/// been logically deleted. It is persisted as a roaring bitmap in a
/// `_deletions/rg_<id>__v<seq>.del` file.
#[derive(Debug, Default, Clone)]
pub struct DeletionVector(RoaringBitmap);

impl DeletionVector {
    /// Mark all given row offsets as deleted. Re-inserting an already-deleted
    /// offset is a no-op (idempotent).
    pub fn union_offsets(&mut self, offsets: impl IntoIterator<Item = u32>) {
        for off in offsets {
            self.0.insert(off);
        }
    }

    /// Number of distinct deleted offsets.
    pub fn cardinality(&self) -> u64 {
        self.0.len()
    }

    /// Returns `true` if `off` has been marked as deleted.
    pub fn contains(&self, off: u32) -> bool {
        self.0.contains(off)
    }

    /// Iterate over all deleted row offsets in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter()
    }

    /// Serialize to bytes suitable for writing to a `.del` file.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        self.0
            .serialize_into(&mut buf)
            .expect("serialization into Vec<u8> is infallible");
        buf
    }

    /// Deserialize from bytes previously produced by [`DeletionVector::serialize`].
    pub fn deserialize(bytes: &[u8]) -> Result<Self, std::io::Error> {
        let bitmap = RoaringBitmap::deserialize_from(bytes)?;
        Ok(Self(bitmap))
    }
}

#[cfg(test)]
mod tests {
    use super::DeletionVector;

    #[test]
    fn iter_round_trips_with_contains() {
        let mut dv = DeletionVector::default();
        dv.union_offsets([5u32, 10, 100, 999]);
        let collected: Vec<u32> = dv.iter().collect();
        assert_eq!(collected, vec![5, 10, 100, 999]);
        for off in &collected {
            assert!(dv.contains(*off));
        }
        assert!(!dv.contains(6));
    }

    #[test]
    fn deletion_union_is_idempotent() {
        let mut dv = DeletionVector::default();
        dv.union_offsets([3, 17, 902]);
        dv.union_offsets([17, 902]); // overlap
        assert_eq!(dv.cardinality(), 3);
        assert!(dv.contains(3) && !dv.contains(4));
        let bytes = dv.serialize();
        assert_eq!(
            DeletionVector::deserialize(&bytes).unwrap().cardinality(),
            3
        );
    }
}
