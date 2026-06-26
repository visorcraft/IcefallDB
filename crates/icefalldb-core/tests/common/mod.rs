use arrow::array::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use std::path::{Path, PathBuf};

pub fn write_snappy_parquet(path: &Path, batch: &RecordBatch) {
    let props = parquet::file::properties::WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(path).unwrap(),
        batch.schema().clone(),
        Some(props),
    )
    .unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

pub fn find_parquet_path(entries: &[String]) -> Option<PathBuf> {
    entries
        .iter()
        .find(|e| {
            let name = std::path::Path::new(e)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("");
            name.ends_with(".parquet")
        })
        .map(PathBuf::from)
}
