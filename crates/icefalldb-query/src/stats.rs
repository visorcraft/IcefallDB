//! Build DataFusion `Statistics` from a IcefallDB `ScanPlan`.
//!
//! Row counts are exact because they come from the manifest / sidecar metadata.
//! Column null counts are summed across row groups. Min/max values are exact
//! only when every row group publishes comparable sidecar statistics and the
//! JSON values can be reconstructed as `ScalarValue`s.

use arrow::datatypes::Schema as ArrowSchema;
use datafusion::common::stats::Precision;
use datafusion::common::ScalarValue;
use datafusion::physical_plan::{ColumnStatistics, Statistics};
use icefalldb_core::ScanPlan;

use crate::scalar_codec::json_to_scalar_value;

/// Build DataFusion `Statistics` from a IcefallDB `ScanPlan`.
///
/// `arrow_schema` supplies the column order and data types used to decode
/// sidecar min/max values. Statistics for columns not present in the schema are
/// ignored.
pub fn scan_plan_statistics(scan_plan: &ScanPlan, arrow_schema: &ArrowSchema) -> Statistics {
    // Count only live (non-deleted) rows.  `rg.meta.rows` is the physical row
    // count in the Parquet file; `rg.deleted_count` is the number of rows that
    // have been logically deleted via a deletion vector.  DataFusion's
    // `AggregateStatistics` physical optimizer rule folds COUNT(*) to
    // `num_rows` when it is `Precision::Exact`, so this value MUST exclude
    // deleted rows to avoid stale counts after a DELETE commit.
    let num_rows: usize = scan_plan
        .row_groups
        .iter()
        .map(|rg| rg.meta.rows.saturating_sub(rg.deleted_count as usize))
        .sum();

    let column_statistics: Vec<ColumnStatistics> = arrow_schema
        .fields()
        .iter()
        .map(|field| {
            let name = field.name();
            let data_type = field.data_type();
            column_stats(scan_plan, name, data_type)
        })
        .collect();

    Statistics {
        num_rows: Precision::Exact(num_rows),
        total_byte_size: Precision::Absent,
        column_statistics,
    }
}

fn column_stats(
    scan_plan: &ScanPlan,
    name: &str,
    data_type: &arrow::datatypes::DataType,
) -> ColumnStatistics {
    let mut null_count: usize = 0;
    let mut min_value: Option<ScalarValue> = None;
    let mut max_value: Option<ScalarValue> = None;
    let mut stats_complete = true;

    for rg in &scan_plan.row_groups {
        let Some(col_stats) = rg.meta.columns.get(name) else {
            // Column missing from this row group's sidecar; treat null count as
            // zero but mark min/max absent because we cannot prove bounds.
            stats_complete = false;
            continue;
        };

        null_count += col_stats.nulls;

        if stats_complete {
            match (&col_stats.min, &col_stats.max) {
                (Some(min_json), Some(max_json)) => {
                    let Some(rg_min) = json_to_scalar_value(min_json, data_type) else {
                        stats_complete = false;
                        continue;
                    };
                    let Some(rg_max) = json_to_scalar_value(max_json, data_type) else {
                        stats_complete = false;
                        continue;
                    };

                    min_value = Some(match min_value {
                        Some(current) => pick_min(&current, &rg_min),
                        None => rg_min,
                    });
                    max_value = Some(match max_value {
                        Some(current) => pick_max(&current, &rg_max),
                        None => rg_max,
                    });
                }
                _ => {
                    stats_complete = false;
                }
            }
        }
    }

    let min_value = if stats_complete {
        min_value.map(Precision::Exact).unwrap_or(Precision::Absent)
    } else {
        Precision::Absent
    };
    let max_value = if stats_complete {
        max_value.map(Precision::Exact).unwrap_or(Precision::Absent)
    } else {
        Precision::Absent
    };

    ColumnStatistics {
        null_count: Precision::Exact(null_count),
        max_value,
        min_value,
        distinct_count: Precision::Absent,
        sum_value: Precision::Absent,
        byte_size: Precision::Absent,
    }
}

fn pick_min(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    match a.partial_cmp(b) {
        Some(std::cmp::Ordering::Greater) => b.clone(),
        _ => a.clone(),
    }
}

fn pick_max(a: &ScalarValue, b: &ScalarValue) -> ScalarValue {
    match a.partial_cmp(b) {
        Some(std::cmp::Ordering::Less) => b.clone(),
        _ => a.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};
    use icefalldb_core::metadata::{ColumnStats, RowGroupMeta};
    use icefalldb_core::PlannedRowGroup;
    use serde_json::Value;

    fn make_scan_plan(row_groups: Vec<(usize, Vec<(String, ColumnStats)>)>) -> ScanPlan {
        let mut planned = Vec::new();
        for (rows, columns) in row_groups {
            let mut meta = RowGroupMeta {
                rows,
                columns: columns.into_iter().collect(),
                ..Default::default()
            };
            // The checksums are not used by statistics building.
            meta.checksum.clear();
            meta.meta_checksum.clear();
            planned.push(PlannedRowGroup {
                meta,
                ..Default::default()
            });
        }
        ScanPlan {
            table: "test".into(),
            schema: icefalldb_core::metadata::Schema {
                schema_id: 1,
                columns: vec![],
                partition_by: None,
                sort: None,
                agg_group_keys: None,
                row_group_target_rows: 1000,
                row_group_target_bytes: 1024,
                dropped_columns: vec![],
                max_field_id: 0,
            },
            row_groups: planned,
        }
    }

    #[test]
    fn test_exact_row_count_and_nulls() {
        let scan = make_scan_plan(vec![
            (
                100,
                vec![(
                    "id".into(),
                    ColumnStats {
                        min: Some(Value::from(1)),
                        max: Some(Value::from(10)),
                        nulls: 5,
                    },
                )],
            ),
            (
                200,
                vec![(
                    "id".into(),
                    ColumnStats {
                        min: Some(Value::from(11)),
                        max: Some(Value::from(20)),
                        nulls: 3,
                    },
                )],
            ),
        ]);
        let arrow = ArrowSchema::new(vec![Field::new("id", DataType::Int64, true)]);
        let stats = scan_plan_statistics(&scan, &arrow);

        assert_eq!(stats.num_rows, Precision::Exact(300));
        let col = stats.column_statistics[0].clone();
        assert_eq!(col.null_count, Precision::Exact(8));
        assert_eq!(col.min_value, Precision::Exact(ScalarValue::Int64(Some(1))));
        assert_eq!(
            col.max_value,
            Precision::Exact(ScalarValue::Int64(Some(20)))
        );
    }

    #[test]
    fn test_missing_min_max_becomes_absent() {
        let scan = make_scan_plan(vec![(
            10,
            vec![(
                "id".into(),
                ColumnStats {
                    min: None,
                    max: None,
                    nulls: 0,
                },
            )],
        )]);
        let arrow = ArrowSchema::new(vec![Field::new("id", DataType::Int64, true)]);
        let stats = scan_plan_statistics(&scan, &arrow);
        let col = stats.column_statistics[0].clone();
        assert_eq!(col.null_count, Precision::Exact(0));
        assert_eq!(col.min_value, Precision::Absent);
        assert_eq!(col.max_value, Precision::Absent);
    }
}
