//! Tests for IcefallDB physical optimizer rule feature flags.

use datafusion::execution::config::SessionConfig;
use icefalldb_query::session::IcefallDBConfig;

#[test]
fn test_disabled_rule_flags_are_preserved() {
    let cfg = SessionConfig::new()
        .with_option_extension(IcefallDBConfig::default())
        .set_str("icefalldb.sorted_group_by", "false")
        .set_str("icefalldb.lookup_join", "false");

    let extracted = cfg
        .options()
        .extensions
        .get::<IcefallDBConfig>()
        .expect("IcefallDBConfig extension should be present");

    assert!(
        !extracted.sorted_group_by,
        "sorted_group_by should be disabled"
    );
    assert!(!extracted.lookup_join, "lookup_join should be disabled");
    assert!(extracted.metadata_aggregate);
    assert!(extracted.tiny_build_join);
    assert!(extracted.dynamic_filter_pushdown);
    assert_eq!(extracted.tiny_build_join_threshold, 4096);
}
