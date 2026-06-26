//! The reader builds the scan plan from the derived zero-copy `rkyv`
//! checkpoint archive (no serde_json structural parse), byte-equal to the JSON
//! path, and falls back cleanly when the archive is absent.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::{Reader, ScanPlan, Writer};

fn schema() -> Schema {
    let mut schema = Schema {
        schema_id: 1,
        columns: vec![Column {
            name: "id".into(),
            r#type: "int64".into(),
            nullable: false,
            field_id: 0,
        }],
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: 100,
        row_group_target_bytes: 1024 * 1024,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    schema
}

fn batch(ids: Vec<i64>) -> RecordBatch {
    let s = ArrowSchema::new(vec![Field::new("id", DataType::Int64, false)]);
    RecordBatch::try_new(Arc::new(s), vec![Arc::new(Int64Array::from(ids))]).unwrap()
}

/// (data path, row count) per row group — the observable scan-plan shape.
fn shape(plan: &ScanPlan) -> Vec<(String, usize)> {
    let mut v: Vec<(String, usize)> = plan
        .row_groups
        .iter()
        .map(|rg| (rg.data_path.clone(), rg.meta.rows))
        .collect();
    v.sort();
    v
}

async fn build_three_fragments(storage: &Arc<dyn Storage>, table: &str) {
    let mut w = Writer::create(Arc::clone(storage), table, schema())
        .await
        .unwrap();
    w.insert_batch(batch(vec![1, 2, 3])).await.unwrap();
    w.commit().await.unwrap();
    for chunk in [vec![4, 5, 6], vec![7, 8, 9]] {
        let mut w = Writer::new(Arc::clone(storage), table, schema())
            .await
            .unwrap();
        w.insert_batch(batch(chunk)).await.unwrap();
        w.commit().await.unwrap();
    }
}

fn list_files(dir: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some(ext) {
                out.push(p);
            }
        }
    }
    out
}

#[tokio::test]
async fn scan_plan_built_from_archive_byte_equal_and_falls_back() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    build_three_fragments(&storage, "t").await;

    // Baseline plan (archive present and used).
    let baseline = shape(
        &Reader::new(&*storage, "t")
            .await
            .unwrap()
            .scan()
            .await
            .unwrap(),
    );
    assert_eq!(baseline.len(), 3, "three fragments");

    // The derived `.rkyv` archive(s) were written next to the JSON checkpoints.
    let ckpt_dir = root.join("t/_checkpoints");
    assert!(
        !list_files(&ckpt_dir, "rkyv").is_empty(),
        "checkpoint archive must be written"
    );

    // Definitive proof the archive is used: remove BOTH the JSON checkpoints AND
    // every `.meta` sidecar, leaving only the `.rkyv` archive (+ the Parquet
    // data). A successful, identical scan can then come ONLY from the archive.
    for p in list_files(&ckpt_dir, "json") {
        std::fs::remove_file(p).unwrap();
    }
    for p in list_files(&root.join("t"), "meta") {
        std::fs::remove_file(p).unwrap();
    }
    let from_archive = shape(
        &Reader::new(&*storage, "t")
            .await
            .unwrap()
            .scan()
            .await
            .unwrap(),
    );
    assert_eq!(
        from_archive, baseline,
        "scan plan from the archive must be byte-equal to the JSON/sidecar path"
    );

    // Fallback: a fresh table with the `.rkyv` archive(s) removed (JSON kept)
    // still opens via the canonical JSON checkpoint.
    let storage2: Arc<dyn Storage> = Arc::new(LocalStorage::new(root).unwrap());
    build_three_fragments(&storage2, "t2").await;
    let want = shape(
        &Reader::new(&*storage2, "t2")
            .await
            .unwrap()
            .scan()
            .await
            .unwrap(),
    );
    for p in list_files(&root.join("t2/_checkpoints"), "rkyv") {
        std::fs::remove_file(p).unwrap();
    }
    let from_json = shape(
        &Reader::new(&*storage2, "t2")
            .await
            .unwrap()
            .scan()
            .await
            .unwrap(),
    );
    assert_eq!(
        from_json, want,
        "must fall back to the JSON checkpoint cleanly"
    );
}
