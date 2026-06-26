//! Step 1: establish the locate bar.
//!
//! Times the CURRENT secondary-index locate path used for monotonic-but-non-affine
//! integer keys — the mmap binary index `lookup_checked` (string binary search) —
//! against a hypothetical probed near-linear learned index (slope/intercept +
//! max-error bound + bounded local binary search) and the JSON B-tree, on the
//! same key set. This quantifies whether a probed learned index has headroom over
//! the binary-index fallback.
//!
//! Run: `cargo run --release -p icefalldb-core --example bench_locate [n]`

use std::collections::BTreeMap;
use std::time::Instant;

use icefalldb_core::index::binary::{serialize, BinaryIndexRef};
use icefalldb_core::{BTreeIndex, IndexDefinition};

/// Deterministic PRNG (SplitMix64-ish) so the bench is reproducible without a dep.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1_000_000);

    // Monotonic-NON-affine integer keys: cumulative random gaps in 1..=9 (so the
    // exact-affine learned model would NOT fit; the binary index is the path).
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut keys: Vec<i64> = Vec::with_capacity(n);
    let mut k: i64 = 0;
    for _ in 0..n {
        k += 1 + (next(&mut rng) % 9) as i64;
        keys.push(k);
    }

    // Canonical JSON B-tree: string key -> [row_id]. row_id = position.
    let mut entries: BTreeMap<String, Vec<u64>> = BTreeMap::new();
    for (i, &key) in keys.iter().enumerate() {
        entries.insert(key.to_string(), vec![i as u64]);
    }
    let def = IndexDefinition {
        name: "k_idx".into(),
        table: "t".into(),
        column: "k".into(),
        unique: true,
    };
    let btree = BTreeIndex {
        definition: def,
        snapshot_sequence: 1,
        entries,
    };

    // Derived binary index (the production fallback path for non-affine keys).
    let bytes = serialize(&btree);
    let bin = BinaryIndexRef::parse(&bytes).expect("binary parse");

    // Probed near-linear learned index over the sorted keys.
    let key_min = keys[0] as f64;
    let key_max = keys[n - 1] as f64;
    let slope = (n as f64 - 1.0) / (key_max - key_min);
    let predict = |key: i64| -> usize {
        (((key as f64 - key_min) * slope).round()).clamp(0.0, (n - 1) as f64) as usize
    };
    let mut max_err: usize = 0;
    for (i, &key) in keys.iter().enumerate() {
        max_err = max_err.max(predict(key).abs_diff(i));
    }
    let learned_locate = |q: &str| -> Option<u64> {
        let key: i64 = q.parse().ok()?;
        let p = predict(key);
        let lo = p.saturating_sub(max_err);
        let hi = (p + max_err + 1).min(n);
        keys[lo..hi]
            .binary_search(&key)
            .ok()
            .map(|idx| (lo + idx) as u64)
    };

    // Query workload: 100k random existing keys, as strings (the SQL locate path).
    let q_count = 100_000usize;
    let mut qrng = 0xdead_beef_cafe_babeu64;
    let queries: Vec<String> = (0..q_count)
        .map(|_| keys[(next(&mut qrng) as usize) % n].to_string())
        .collect();

    let mut sink = 0u64;

    let t = Instant::now();
    for q in &queries {
        if let Some(ids) = bin.lookup_checked(q) {
            sink ^= ids.first().copied().unwrap_or(0);
        }
    }
    let bin_ns = t.elapsed().as_nanos() as f64 / q_count as f64;

    let t = Instant::now();
    for q in &queries {
        if let Some(r) = learned_locate(q) {
            sink ^= r;
        }
    }
    let learned_ns = t.elapsed().as_nanos() as f64 / q_count as f64;

    let t = Instant::now();
    for q in &queries {
        if let Some(ids) = btree.lookup(q).first() {
            sink ^= *ids;
        }
    }
    let json_ns = t.elapsed().as_nanos() as f64 / q_count as f64;

    println!(
        "n={n}  max_err={max_err}  (key range {}..{})",
        keys[0],
        keys[n - 1]
    );
    println!("binary-index   locate: {bin_ns:8.1} ns/op  (string binary search + Vec alloc)");
    println!("probed-learned locate: {learned_ns:8.1} ns/op  (predict + bounded i64 search)");
    println!("json-btree     locate: {json_ns:8.1} ns/op  (BTreeMap::get)");
    let pt_us = 50_000.0; // measured point-lookup p95 ~50 ms (mutations/open_metrics.json)
    println!(
        "locate as a fraction of a ~50ms point-lookup: binary {:.4}%, learned {:.4}%",
        bin_ns / 1000.0 / pt_us * 100.0,
        learned_ns / 1000.0 / pt_us * 100.0
    );
    std::hint::black_box(sink);
}
