use anyhow::Context;
use arrow::array::RecordBatch;
use arrow::ipc::reader::{FileReader, StreamReader};
use bytes::Bytes;
use clap::{Parser, Subcommand, ValueEnum};
use fs2::FileExt;
use futures::stream::StreamExt;
use icefalldb_core::catalog::Catalog;
use icefalldb_core::check::{CheckResult, Severity};
use icefalldb_core::metadata::{Column, Schema};
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::writer::validate_table;
#[cfg(feature = "encryption")]
use icefalldb_core::WriterOptionsFull;
use icefalldb_core::{
    build_btree_index, list_snapshots, Checker, Compactor, DatabaseCatalog, Doctor,
    GarbageCollector, IcefallDBError, IndexDefinition, InsertParquetOutcome, Reader, TsvDecoder,
    TsvEncoder, Writer,
};
#[cfg(feature = "encryption")]
use icefalldb_query::icefalldb_encrypted_session;
use icefalldb_query::result_cache::{is_cacheable_select, EvictPolicy, ResultCache};
use icefalldb_query::{
    execute_sql, icefalldb_session, IcefallDBTableProvider, ProviderConfig, QueryError,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Duration;

mod client;
#[cfg(feature = "encryption")]
mod encryption;

#[derive(Debug, Parser)]
#[command(name = "icefalldb")]
#[command(about = "IcefallDB command-line tool")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Create a new IcefallDB table.
    Create {
        db: PathBuf,
        table: String,
        /// Path to a JSON schema file.
        #[arg(long)]
        schema: Option<PathBuf>,
    },
    /// Create a new IcefallDB table through the central database catalog.
    CreateTable {
        db: PathBuf,
        table: String,
        /// Path to a JSON schema file.
        #[arg(long)]
        schema: Option<PathBuf>,
    },
    /// Drop a IcefallDB table from the central database catalog.
    DropTable { db: PathBuf, table: String },
    /// Insert records from a file into a IcefallDB table.
    Insert {
        db: PathBuf,
        table: String,
        file: PathBuf,
    },
    /// Import a TSV file into a IcefallDB table, inferring the schema if needed.
    Import {
        db: PathBuf,
        table: String,
        file: PathBuf,
        /// Encrypt the whole table (Parquet Modular Encryption, footer key).
        #[arg(long)]
        encrypt: bool,
        /// Encrypt only this column with its own key (repeatable). Without
        /// `--encrypt`, the other columns stay plaintext.
        #[arg(long = "encrypt-column", value_name = "COLUMN")]
        encrypt_column: Vec<String>,
        /// Also encrypt the Parquet footer (default: footer left plaintext so
        /// page-index reads stay fast).
        #[arg(long)]
        encrypt_footer: bool,
        /// JSON key file mapping key ids to hex keys. Default: read keys from
        /// `ICEFALLDB_KEY_*` environment variables.
        #[arg(long, value_name = "PATH")]
        key_file: Option<PathBuf>,
    },
    /// Export a IcefallDB table to a TSV file.
    Export {
        db: PathBuf,
        table: String,
        file: PathBuf,
    },
    /// Validate a IcefallDB table.
    Check {
        db: PathBuf,
        table: String,
        /// JSON key file for reading an encrypted table. Default: read keys
        /// from `ICEFALLDB_KEY_*` environment variables.
        #[arg(long, value_name = "PATH")]
        key_file: Option<PathBuf>,
    },
    /// Diagnose and optionally repair a IcefallDB table.
    Doctor {
        db: PathBuf,
        table: String,
        /// Actually perform repairs instead of only reporting issues.
        #[arg(long)]
        repair: bool,
    },
    /// Compact a IcefallDB table offline.
    Compact { db: PathBuf, table: String },
    /// Run garbage collection on a IcefallDB table.
    Gc {
        db: PathBuf,
        table: String,
        /// Number of recent snapshots to retain.
        #[arg(long, default_value = "3")]
        retain_snapshots: usize,
    },
    /// Compact a IcefallDB table to ZSTD-1, then garbage-collect old snapshots/files.
    Optimize {
        /// Path to the IcefallDB database directory.
        #[arg(value_name = "DB")]
        db: PathBuf,
        /// Table to optimize.
        #[arg(value_name = "TABLE")]
        table: String,
        /// Number of recent snapshots to retain after optimization.
        #[arg(long, default_value = "1", value_name = "N")]
        retain_snapshots: usize,
        /// Column to sort output row groups by. May be repeated for composite keys.
        #[arg(long = "sort", value_name = "KEY")]
        sort_keys: Vec<String>,
    },
    /// Create an index on a table column.
    CreateIndex {
        /// Path to the IcefallDB database directory.
        #[arg(value_name = "DB")]
        db: PathBuf,
        /// Table containing the column to index.
        #[arg(value_name = "TABLE")]
        table: String,
        /// Column to index.
        #[arg(value_name = "COLUMN")]
        column: String,
        /// Optional index name; defaults to `{table}_{column}_idx`.
        #[arg(long, value_name = "NAME")]
        name: Option<String>,
        /// Index type (e.g., btree).
        #[arg(long, default_value = "btree", value_name = "TYPE")]
        index_type: String,
        /// Mark the index as unique (required for MERGE key indexes).
        #[arg(long)]
        unique: bool,
    },
    /// Create a materialized view definition.
    CreateView {
        db: PathBuf,
        view: String,
        query_file: PathBuf,
    },
    /// Refresh a materialized view.
    RefreshView { db: PathBuf, view: String },
    /// Export a IcefallDB table snapshot to Iceberg metadata.
    IcebergExport {
        db: PathBuf,
        table: String,
        output: PathBuf,
        /// Snapshot sequence to export. Defaults to the latest snapshot.
        #[arg(long)]
        snapshot: Option<u64>,
    },
    /// List the retained snapshot history for a IcefallDB table.
    Snapshots { db: PathBuf, table: String },
    /// Execute a SQL query against one or more IcefallDB tables using the DataFusion engine.
    Query {
        /// Path to a IcefallDB table directory, or a database directory when --table is used.
        path: PathBuf,
        /// SQL query to execute.
        sql: String,
        /// Additional tables to register. The table in `path` is always registered.
        #[arg(short = 't', long = "table")]
        extra_tables: Vec<String>,
        /// Output format.
        #[arg(long, value_enum, default_value = "json")]
        format: QueryOutputFormat,
        /// Result cache budget in MiB (0 disables). Eligible SELECT results are
        /// cached under <db>/_query_cache and shared with the Python adapter.
        #[arg(long, default_value = "1024", value_name = "N")]
        result_cache_mb: u64,
        /// OPTIONAL: route this statement to a running daemon at this URL
        /// (e.g. http://127.0.0.1:8080) so it pays table-open + engine-startup
        /// once. Absent ⇒ the standalone in-process path (unchanged).
        #[arg(long)]
        server: Option<String>,
        /// Read the table AS OF snapshot sequence N instead of the latest.
        /// Useful for time-travel queries. Returns an error if the snapshot
        /// manifest for N is absent.
        #[arg(long)]
        snapshot: Option<u64>,
        /// JSON key file for reading an encrypted table. Default: read keys
        /// from `ICEFALLDB_KEY_*` environment variables.
        #[arg(long, value_name = "PATH")]
        key_file: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum QueryOutputFormat {
    Json,
    Csv,
}

/// Writer-only commands run on a **current-thread** tokio runtime: they use only
/// `icefalldb-core`'s `Writer` (async file I/O), never the DataFusion query session
/// or its multi-thread scan pool, so spawning that pool is pure startup waste.
/// Query and rewrite commands keep the multi-thread runtime for parallel scans.
fn build_runtime(cmd: &Commands) -> std::io::Result<tokio::runtime::Runtime> {
    let writer_only = matches!(
        cmd,
        Commands::Create { .. }
            | Commands::CreateTable { .. }
            | Commands::DropTable { .. }
            | Commands::Insert { .. }
            | Commands::Import { .. }
            | Commands::CreateIndex { .. }
            | Commands::CreateView { .. }
    );
    // A/B benchmark override: force the multi-thread pool even for writer-only
    // commands, to measure the lean runtime's startup saving.
    if writer_only && std::env::var_os("ICEFALLDB_FORCE_MULTI_THREAD").is_none() {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
    }
}

fn main() -> ExitCode {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let kind = e.kind();
            if kind == clap::error::ErrorKind::DisplayHelp
                || kind == clap::error::ErrorKind::DisplayVersion
            {
                e.print().expect("failed to print CLI message");
                return ExitCode::SUCCESS;
            }
            e.print().expect("failed to print CLI error");
            return ExitCode::from(2);
        }
    };

    let runtime = match build_runtime(&cli.command) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("icefalldb: failed to start runtime: {e}");
            return ExitCode::from(2);
        }
    };

    runtime.block_on(async move {
    // Optional instrumentation: when ICEFALLDB_REPORT_FSYNCS / _SESSIONS are set,
    // print the durability barriers and query sessions this invocation built
    // (used by python/benchmarks/perf/*). A writer-only command must report
    // sessions=0 — proof the lean write path never builds the query stack.
    let report_fsyncs = std::env::var_os("ICEFALLDB_REPORT_FSYNCS").is_some();
    let report_sessions = std::env::var_os("ICEFALLDB_REPORT_SESSIONS").is_some();
    let fsyncs_before = icefalldb_core::storage::local::global_fsync_count();

    let exit_code = match cli.command {
        Commands::Create { db, table, schema } => {
            match run_create(&db, &table, schema.as_deref()).await {
                Ok(()) => ExitCode::from(0),
                Err(e) => {
                    eprintln!("icefalldb create failed: {:#}", e);
                    ExitCode::from(1)
                }
            }
        }
        Commands::CreateTable { db, table, schema } => {
            match run_create_table(&db, &table, schema.as_deref()).await {
                Ok(()) => ExitCode::from(0),
                Err(e) => {
                    eprintln!("icefalldb create-table failed: {:#}", e);
                    ExitCode::from(1)
                }
            }
        }
        Commands::DropTable { db, table } => match run_drop_table(&db, &table).await {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb drop-table failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Insert { db, table, file } => match run_insert(&db, &table, &file).await {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb insert failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Import {
            db,
            table,
            file,
            encrypt,
            encrypt_column,
            encrypt_footer,
            key_file,
        } => {
            let enc = ImportEncryption {
                whole_table: encrypt,
                columns: encrypt_column,
                encrypt_footer,
                key_file,
            };
            match run_import(&db, &table, &file, enc).await {
                Ok(rows) => {
                    println!("imported {} rows into {}", rows, table);
                    ExitCode::from(0)
                }
                Err(e) => {
                    eprintln!("icefalldb import failed: {:#}", e);
                    ExitCode::from(1)
                }
            }
        }
        Commands::Export { db, table, file } => match run_export(&db, &table, &file).await {
            Ok(rows) => {
                println!("exported {} rows from {}", rows, table);
                ExitCode::from(0)
            }
            Err(e) => {
                eprintln!("icefalldb export failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Check { db, table, key_file } => match run_check(&db, &table, key_file.as_deref()).await {
            Ok(result) => {
                print_issues(&result);
                if result.passed {
                    println!("check passed");
                    ExitCode::from(0)
                } else {
                    println!("check failed");
                    ExitCode::from(1)
                }
            }
            Err(e) => {
                eprintln!("icefalldb check failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Compact { db, table } => match run_compact(&db, &table).await {
            Ok(result) => {
                println!(
                    "compacted {}: {} input row groups -> {} output row groups, {} rows -> {} rows",
                    table,
                    result.input_row_groups,
                    result.output_row_groups,
                    result.input_rows,
                    result.output_rows
                );
                ExitCode::from(0)
            }
            Err(e) => {
                eprintln!("icefalldb compact failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Gc {
            db,
            table,
            retain_snapshots,
        } => match run_gc(&db, &table, retain_snapshots).await {
            Ok(result) => {
                println!(
                    "gc completed: deleted {} files, retained snapshots {:?}",
                    result.deleted.len(),
                    result.retained_snapshots
                );
                ExitCode::from(0)
            }
            Err(e) => {
                eprintln!("icefalldb gc failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Optimize {
            db,
            table,
            retain_snapshots,
            sort_keys,
        } => match run_optimize(&db, &table, retain_snapshots, sort_keys).await {
            Ok(result) => {
                println!(
                    "optimized {}: {} input row groups -> {} output row groups, {} rows -> {} rows, deleted {} files, retained snapshots {:?}",
                    table,
                    result.compaction.input_row_groups,
                    result.compaction.output_row_groups,
                    result.compaction.input_rows,
                    result.compaction.output_rows,
                    result.gc.deleted.len(),
                    result.gc.retained_snapshots
                );
                ExitCode::from(0)
            }
            Err(e) => {
                eprintln!("icefalldb optimize failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::CreateIndex {
            db,
            table,
            column,
            name,
            index_type,
            unique,
        } => match run_create_index(&db, &table, &column, name.as_deref(), &index_type, unique)
            .await
        {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb create-index failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::CreateView {
            db,
            view,
            query_file,
        } => match run_create_view(&db, &view, &query_file).await {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb create-view failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::RefreshView { db, view } => match run_refresh_view(&db, &view).await {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb refresh-view failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::IcebergExport {
            db,
            table,
            output,
            snapshot,
        } => match run_iceberg_export(&db, &table, &output, snapshot).await {
            Ok(metadata_path) => {
                println!("{}", metadata_path.display());
                ExitCode::from(0)
            }
            Err(e) => {
                eprintln!("icefalldb iceberg-export failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Snapshots { db, table } => match run_snapshots(&db, &table).await {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb snapshots failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Query {
            path,
            sql,
            extra_tables,
            format,
            result_cache_mb,
            server,
            snapshot,
            key_file,
        } => match run_query(&path, &sql, &extra_tables, format, result_cache_mb, server.as_deref(), snapshot, key_file.as_deref()).await {
            Ok(()) => ExitCode::from(0),
            Err(e) => {
                eprintln!("icefalldb query failed: {:#}", e);
                ExitCode::from(1)
            }
        },
        Commands::Doctor { db, table, repair } => {
            if repair {
                match run_doctor_repair(&db, &table).await {
                    Ok(result) => {
                        for action in &result.actions {
                            println!("{} {}: {}", action.kind, action.path, action.detail);
                        }
                        if !result.healthy {
                            println!("unrepairable issues remain");
                            ExitCode::from(1)
                        } else if result.repaired {
                            println!("repaired");
                            ExitCode::from(0)
                        } else {
                            println!("no repairs needed");
                            ExitCode::from(0)
                        }
                    }
                    Err(e) => {
                        eprintln!("icefalldb doctor failed: {:#}", e);
                        ExitCode::from(1)
                    }
                }
            } else {
                match run_doctor_diagnose(&db, &table).await {
                    Ok(result) => {
                        for issue in &result.issues {
                            println!("{} {}: {}", issue.kind, issue.path, issue.detail);
                        }
                        if result.healthy {
                            println!("table is healthy");
                        } else {
                            println!("issues found");
                        }
                        ExitCode::from(0)
                    }
                    Err(e) => {
                        eprintln!("icefalldb doctor failed: {:#}", e);
                        ExitCode::from(1)
                    }
                }
            }
        }
    };

    if report_fsyncs {
        let delta = icefalldb_core::storage::local::global_fsync_count() - fsyncs_before;
        eprintln!("fsyncs={delta}");
    }
    if report_sessions {
        eprintln!("sessions={}", icefalldb_query::session_build_count());
    }
    exit_code
    })
}

async fn run_create(db: &Path, table: &str, schema_path: Option<&Path>) -> anyhow::Result<()> {
    validate_table(table)?;

    let schema = if let Some(path) = schema_path {
        let data = std::fs::read(path)
            .with_context(|| format!("reading schema file {}", path.display()))?;
        serde_json::from_slice(&data)
            .with_context(|| format!("parsing schema file {}", path.display()))?
    } else {
        Schema {
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
            row_group_target_rows: 1_000_000,
            row_group_target_bytes: 134_217_728,
            max_field_id: 0,
            dropped_columns: vec![],
        }
    };

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    // Delegate schema validation, field-ID assignment, and atomic table
    // initialization to the shared core helper. It acquires the writer lock,
    // re-checks `_schema.json` under the lock, and writes the schema files and
    // empty manifest pointer atomically.
    Writer::create(Arc::clone(&storage), table, schema)
        .await
        .with_context(|| format!("creating table '{}'", table))?;

    // Register the newly created table in the central catalog so the daemon
    // can serve it.  `register_existing_table` is idempotent: if the entry
    // already exists (e.g. an interrupted prior run wrote it) it is a no-op.
    let catalog = DatabaseCatalog::new(Arc::clone(&storage));
    let guard = catalog
        .acquire_lock(std::time::Duration::from_secs(30))
        .await
        .with_context(|| "acquiring catalog lock for registration")?;
    catalog
        .register_existing_table(&guard, table)
        .await
        .with_context(|| format!("registering table '{}' in catalog", table))?;

    println!("created table {}", table);
    Ok(())
}

async fn run_create_table(
    db: &Path,
    table: &str,
    schema_path: Option<&Path>,
) -> anyhow::Result<()> {
    let schema = if let Some(path) = schema_path {
        let data = std::fs::read(path)
            .with_context(|| format!("reading schema file {}", path.display()))?;
        serde_json::from_slice(&data)
            .with_context(|| format!("parsing schema file {}", path.display()))?
    } else {
        Schema {
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
            row_group_target_rows: 1_000_000,
            row_group_target_bytes: 134_217_728,
            max_field_id: 0,
            dropped_columns: vec![],
        }
    };

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let catalog = DatabaseCatalog::new(storage);
    let guard = catalog
        .acquire_lock(std::time::Duration::from_secs(30))
        .await
        .with_context(|| "acquiring catalog lock")?;
    catalog
        .create_table(&guard, table, &schema)
        .await
        .with_context(|| format!("creating table '{}'", table))?;

    println!("created table {}", table);
    Ok(())
}

async fn run_drop_table(db: &Path, table: &str) -> anyhow::Result<()> {
    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let catalog = DatabaseCatalog::new(storage);
    let guard = catalog
        .acquire_lock(std::time::Duration::from_secs(30))
        .await
        .with_context(|| "acquiring catalog lock")?;
    catalog
        .drop_table(&guard, table)
        .await
        .with_context(|| format!("dropping table '{}'", table))?;

    println!("dropped table {}", table);
    Ok(())
}

async fn run_insert(db: &Path, table: &str, file: &Path) -> anyhow::Result<()> {
    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let schema = load_table_schema(storage.as_ref(), table)
        .await
        .with_context(|| format!("loading schema for table {}", table))?;

    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    if ext == "parquet" {
        let mut writer = icefalldb_core::Writer::new(Arc::clone(&storage), table, schema.clone())
            .await
            .with_context(|| format!("opening writer for table {}", table))?;
        match writer
            .insert_parquet(file.to_str().unwrap_or(""))
            .await
            .with_context(|| format!("inserting parquet into table {}", table))?
        {
            InsertParquetOutcome::FastPath { rows } => {
                println!("inserted {} rows into {}", rows, table);
                return Ok(());
            }
            InsertParquetOutcome::Incompatible => {
                // Fall through to the decode/re-encode path.
            }
        }
    }

    let batches = read_batches(file)
        .await
        .with_context(|| format!("reading input file {}", file.display()))?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    let mut writer = icefalldb_core::Writer::new(Arc::clone(&storage), table, schema)
        .await
        .with_context(|| format!("opening writer for table {}", table))?;
    for batch in batches {
        writer
            .insert_batch(batch)
            .await
            .with_context(|| format!("inserting into table {}", table))?;
    }
    writer
        .commit()
        .await
        .with_context(|| format!("committing inserts to table {}", table))?;

    println!("inserted {} rows into {}", total_rows, table);
    Ok(())
}

/// Encryption options for `import`, carried independently of the optional
/// `encryption` feature. Conversion to a real key set happens under
/// `#[cfg(feature = "encryption")]` in [`run_import`]; with the feature off,
/// `encrypt_footer`/`key_file` are unused, hence the conditional `allow`.
#[cfg_attr(not(feature = "encryption"), allow(dead_code))]
struct ImportEncryption {
    whole_table: bool,
    columns: Vec<String>,
    encrypt_footer: bool,
    key_file: Option<PathBuf>,
}

impl ImportEncryption {
    fn enabled(&self) -> bool {
        self.whole_table || !self.columns.is_empty()
    }
}

async fn run_import(
    db: &Path,
    table: &str,
    file: &Path,
    enc: ImportEncryption,
) -> anyhow::Result<usize> {
    validate_table(table)?;

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let data = tokio::fs::read(file)
        .await
        .with_context(|| format!("reading TSV file {}", file.display()))?;

    let schema = if storage.exists(&format!("{}/_schema.json", table)).await? {
        load_table_schema(storage.as_ref(), table)
            .await
            .with_context(|| format!("loading schema for table {}", table))?
    } else {
        infer_schema_from_tsv(&data)
            .with_context(|| format!("inferring schema from {}", file.display()))?
    };

    let batches =
        TsvDecoder::decode(&data, &schema).map_err(|e| anyhow::anyhow!("decoding TSV: {}", e))?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    let mut writer = if enc.enabled() {
        #[cfg(feature = "encryption")]
        {
            // `--encrypt` (whole table) and `--encrypt-column` (specific columns)
            // are mutually exclusive: combining them is ambiguous about which
            // columns end up encrypted. Require exactly one mode.
            if enc.whole_table && !enc.columns.is_empty() {
                anyhow::bail!(
                    "use --encrypt to encrypt the whole table OR --encrypt-column \
                     for specific columns, not both"
                );
            }
            // A mistyped --encrypt-column would otherwise silently leave the
            // intended column in plaintext. Reject names that are not columns.
            let col_names: std::collections::HashSet<&str> =
                schema.columns.iter().map(|c| c.name.as_str()).collect();
            for c in &enc.columns {
                if !col_names.contains(c.as_str()) {
                    anyhow::bail!(
                        "--encrypt-column '{}' is not a column of table '{}'",
                        c,
                        table
                    );
                }
            }
            let spec = encryption::EncryptSpec {
                whole_table: enc.whole_table,
                columns: enc.columns.clone(),
                encrypt_footer: enc.encrypt_footer,
                key_file: enc.key_file.clone(),
            };
            let cfg = encryption::write_config(&spec, table, schema.schema_id)
                .await
                .with_context(|| format!("building encryption config for '{}'", table))?;
            Writer::new_with_full(
                Arc::clone(&storage),
                table,
                schema,
                WriterOptionsFull::new().with_encryption(cfg),
            )
            .await
            .with_context(|| format!("opening encrypted writer for table {}", table))?
        }
        #[cfg(not(feature = "encryption"))]
        {
            let _ = &enc;
            anyhow::bail!(
                "this `icefalldb` build was compiled without the `encryption` feature; \
                 rebuild with default features to create encrypted tables"
            );
        }
    } else {
        Writer::new(Arc::clone(&storage), table, schema)
            .await
            .with_context(|| format!("opening writer for table {}", table))?
    };

    for batch in batches {
        writer
            .insert_batch(batch)
            .await
            .with_context(|| format!("inserting into table {}", table))?;
    }
    writer
        .commit()
        .await
        .with_context(|| format!("committing import to table {}", table))?;

    Ok(total_rows)
}

async fn run_export(db: &Path, table: &str, file: &Path) -> anyhow::Result<usize> {
    validate_table(table)?;

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let schema = load_table_schema(storage.as_ref(), table)
        .await
        .with_context(|| format!("loading schema for table {}", table))?;
    let arrow_schema = schema.arrow_schema().ok_or_else(|| {
        anyhow::anyhow!(
            "schema for table '{}' contains unsupported Arrow types",
            table
        )
    })?;

    let mut batches: Vec<RecordBatch> = Vec::new();
    let manifest_pointer_path = format!("{}/_manifest.json", table);
    let latest = if storage.exists(&manifest_pointer_path).await? {
        let pointer_data = storage.read(&manifest_pointer_path).await?;
        let pointer: serde_json::Value = serde_json::from_slice(&pointer_data)
            .with_context(|| format!("parsing manifest pointer for {}", table))?;
        pointer["latest"].as_u64().unwrap_or(0)
    } else {
        0
    };

    if latest > 0 {
        let reader = Reader::new(storage.as_ref(), table)
            .await
            .with_context(|| format!("opening reader for table {}", table))?;
        let plan = reader.scan().await?;
        for rg in &plan.row_groups {
            let mut stream = reader.read_row_group(rg).await?;
            while let Some(batch) = stream.next().await {
                batches.push(batch?);
            }
        }
    }

    let batch = if batches.is_empty() {
        RecordBatch::new_empty(Arc::new(arrow_schema))
    } else {
        arrow::compute::concat_batches(&Arc::new(arrow_schema), &batches)
            .context("concatenating export batches")?
    };
    let total_rows = batch.num_rows();

    let tsv = TsvEncoder::encode(&batch);
    tokio::fs::write(file, tsv)
        .await
        .with_context(|| format!("writing TSV file {}", file.display()))?;

    Ok(total_rows)
}

fn infer_schema_from_tsv(data: &[u8]) -> anyhow::Result<Schema> {
    let mut lines = Vec::new();
    for line in data.split(|&b| b == b'\n') {
        let line = if line.ends_with(b"\r") {
            &line[..line.len() - 1]
        } else {
            line
        };
        if !line.is_empty() {
            lines.push(line);
        }
    }
    if lines.is_empty() {
        anyhow::bail!("TSV file is empty");
    }

    let header = icefalldb_core::split_tsv_line(lines[0])?;
    let mut column_types: Vec<Option<&str>> = vec![None; header.len()];

    for line in lines.iter().skip(1).take(100) {
        let cells = icefalldb_core::split_tsv_line(line)?;
        for (i, cell) in cells.iter().enumerate().take(header.len()) {
            let trimmed = String::from_utf8_lossy(cell).trim().to_string();
            if trimmed.is_empty() {
                continue;
            }
            let inferred = infer_value_type(&trimmed);
            column_types[i] = Some(promote_type(column_types[i], inferred));
        }
    }

    let columns: Vec<Column> = header
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            let name = String::from_utf8_lossy(name).to_string();
            let type_str = column_types[i].unwrap_or("utf8");
            Column::new(name, type_str, true)
        })
        .collect();

    let mut schema = Schema {
        schema_id: 1,
        columns,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: icefalldb_core::DEFAULT_ROW_GROUP_TARGET_ROWS,
        row_group_target_bytes: icefalldb_core::DEFAULT_ROW_GROUP_TARGET_BYTES,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    Ok(schema)
}

fn infer_value_type(value: &str) -> &'static str {
    let value = value.trim();
    if value == "true" || value == "false" {
        return "bool";
    }
    if value.parse::<i64>().is_ok() {
        return "int64";
    }
    if value.parse::<f64>().is_ok() {
        return "float64";
    }
    if chrono::DateTime::parse_from_rfc3339(value).is_ok()
        || chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f").is_ok()
        || chrono::NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.f").is_ok()
    {
        return "timestamp[us]";
    }
    "utf8"
}

fn promote_type(current: Option<&'static str>, inferred: &'static str) -> &'static str {
    // Precedence: bool < int64 < float64 < timestamp[us] < utf8.
    let score = |t: &str| match t {
        "bool" => 0,
        "int64" => 1,
        "float64" => 2,
        "timestamp[us]" => 3,
        _ => 4,
    };
    match current {
        None => inferred,
        Some(t) => {
            if score(inferred) > score(t) {
                inferred
            } else {
                t
            }
        }
    }
}

async fn run_create_view(db: &Path, view: &str, query_file: &Path) -> anyhow::Result<()> {
    validate_table(view)?;

    let query = tokio::fs::read_to_string(query_file)
        .await
        .with_context(|| format!("reading query file {}", query_file.display()))?;
    validate_view_sql(&query).with_context(|| "validating view query")?;
    let normalized = extract_view_query(&query)?;

    let views_dir = db.join("views");
    tokio::fs::create_dir_all(&views_dir)
        .await
        .with_context(|| format!("creating views directory {}", views_dir.display()))?;

    let view_file = views_dir.join(format!("{}.sql", view));
    tokio::fs::write(&view_file, normalized)
        .await
        .with_context(|| format!("writing view definition {}", view_file.display()))?;

    println!("created view {}", view);
    Ok(())
}

async fn run_refresh_view(db: &Path, view: &str) -> anyhow::Result<()> {
    validate_table(view)?;

    let views_dir = db.join("views");
    let query_file = views_dir.join(format!("{}.sql", view));
    if !query_file.is_file() {
        anyhow::bail!("view '{}' does not exist at {}", view, query_file.display());
    }
    let query = tokio::fs::read_to_string(&query_file)
        .await
        .with_context(|| format!("reading view definition {}", query_file.display()))?;
    if query.trim().is_empty() {
        anyhow::bail!("view definition is empty");
    }

    // Ensure DuckDB CLI is available.
    if which::which("duckdb").is_err() {
        anyhow::bail!(
            "duckdb CLI not found in PATH; install DuckDB to use refresh-view \
             (https://duckdb.org/docs/installation)"
        );
    }

    let db_abs = std::path::absolute(db)
        .with_context(|| format!("resolving database directory {}", db.display()))?;
    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    // Build a preamble that registers every IcefallDB table (and the view's own
    // derived table, if it exists) as a DuckDB temp view. The file list is read
    // from each table's latest manifest so refresh sees a consistent snapshot
    // and never reads unreferenced Parquet files directly.
    let view_table = format!("views/{}", view);
    let view_table_dir = views_dir.join(view);

    // Acquire the writer lock on the view table for the entire refresh. The
    // lock is held from before any DuckDB work until after the new manifest
    // pointer is durable. The Writer is told not to re-acquire the lock, so
    // the CLI's file handle keeps the lock alive until it drops at the end of
    // this function.
    tokio::fs::create_dir_all(&view_table_dir)
        .await
        .with_context(|| format!("creating view table directory {}", view_table_dir.display()))?;
    let lock_path = view_table_dir.join("_write.lock");
    let _lock_guard = tokio::task::spawn_blocking({
        let lock_path = lock_path.clone();
        move || -> anyhow::Result<FlockGuard> {
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&lock_path)
                .with_context(|| format!("opening view lock file {}", lock_path.display()))?;
            file.lock_exclusive().with_context(|| {
                format!(
                    "acquiring exclusive writer lock for view table {}",
                    lock_path.display()
                )
            })?;
            Ok(FlockGuard(file))
        }
    })
    .await
    .map_err(|e| anyhow::anyhow!("lock acquisition task panicked: {}", e))??;

    let preamble = build_refresh_preamble(db, storage.as_ref(), &view_table)
        .await
        .with_context(|| "building refresh preamble")?;

    let query_text = extract_view_query(&query)?;

    // Open or create the derived view table while already holding the writer
    // lock. If the derived table does not exist yet, infer its schema from a
    // LIMIT 0 preflight. Tell the writer not to re-acquire the lock, because
    // the CLI's FlockGuard already holds it for the whole refresh.
    let writer_options = icefalldb_core::WriterOptions {
        lock_timeout: std::time::Duration::from_secs(30),
        assume_lock_held: true,
    };
    let mut writer = if view_table_dir.join("_schema.json").is_file() {
        let schema = load_table_schema(storage.as_ref(), &view_table)
            .await
            .with_context(|| format!("loading schema for view table {}", view_table))?;
        icefalldb_core::Writer::new_with_options(
            Arc::clone(&storage),
            &view_table,
            schema,
            writer_options,
        )
        .await
        .with_context(|| format!("opening writer for view table {}", view_table))?
    } else {
        let inferred = infer_schema_from_query(&preamble, &query_text)
            .await
            .with_context(|| "inferring schema from refresh query")?;
        icefalldb_core::Writer::create_with_options(
            Arc::clone(&storage),
            &view_table,
            inferred,
            writer_options,
        )
        .await
        .with_context(|| format!("creating view table {}", view_table))?
    };

    let tmp_parquet = tempfile::NamedTempFile::with_suffix_in(".parquet", &db_abs)
        .with_context(|| "creating temporary parquet file")?;
    let tmp_path = tmp_parquet.path().to_path_buf();
    let tmp_path_str = tmp_path
        .to_string_lossy()
        .replace('\\', "/")
        .replace('\'', "''");
    let sql = format!(
        "{} COPY ({}) TO '{}' (FORMAT PARQUET);",
        preamble, query_text, tmp_path_str
    );
    run_duckdb_command(&sql)?;

    let batches = read_parquet_batches(&tmp_path)
        .await
        .with_context(|| "reading refresh output parquet")?;
    for batch in batches {
        writer
            .insert_batch(batch)
            .await
            .with_context(|| format!("inserting refresh output into view table {}", view_table))?;
    }
    writer.replace().await.with_context(|| {
        format!(
            "committing refresh replacement for view table {}",
            view_table
        )
    })?;

    // Explicitly close the temp file so the deletion below succeeds on Windows.
    drop(tmp_parquet);
    let _ = tokio::fs::remove_file(&tmp_path).await;

    println!("refreshed view {}", view);
    Ok(())
}

/// Advisory lock guard that releases an exclusive `flock()` on drop.
///
/// The CLI uses this directly (instead of [`Storage::lock_exclusive`]) so it
/// can hold the view-table writer lock across the entire refresh, including
/// DuckDB execution and `Writer::replace`, without releasing between internal
/// writer calls.
struct FlockGuard(std::fs::File);

impl Drop for FlockGuard {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.0);
    }
}

/// Infer a IcefallDB schema from a DuckDB `DESCRIBE` of the view query.
///
/// This avoids writing a temporary Parquet file, which is important because
/// `COPY ... LIMIT 0` can produce an empty/invalid Parquet file.
async fn infer_schema_from_query(preamble: &str, query: &str) -> anyhow::Result<Schema> {
    let describe_sql = format!("{} DESCRIBE ({});", preamble, query);
    let output = std::process::Command::new("duckdb")
        .arg("-csv")
        .arg("-c")
        .arg(&describe_sql)
        .output()
        .with_context(|| "running duckdb DESCRIBE")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("duckdb DESCRIBE failed: {}", stderr);
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty DESCRIBE output"))?;
    let column_name_idx = header
        .split(',')
        .position(|h| h.eq_ignore_ascii_case("column_name"))
        .ok_or_else(|| anyhow::anyhow!("missing column_name in DESCRIBE output"))?;
    let column_type_idx = header
        .split(',')
        .position(|h| h.eq_ignore_ascii_case("column_type"))
        .ok_or_else(|| anyhow::anyhow!("missing column_type in DESCRIBE output"))?;

    let mut columns = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        if fields.len() <= column_name_idx.max(column_type_idx) {
            anyhow::bail!("malformed DESCRIBE output line: {}", line);
        }
        let name = fields[column_name_idx].trim().to_string();
        let duckdb_type = fields[column_type_idx].trim();
        columns.push(Column {
            name,
            r#type: duckdb_type_to_icefalldb(duckdb_type)?,
            nullable: true,
            field_id: 0,
        });
    }

    let mut schema = Schema {
        schema_id: 1,
        columns,
        partition_by: None,
        sort: None,
        agg_group_keys: None,
        row_group_target_rows: icefalldb_core::DEFAULT_ROW_GROUP_TARGET_ROWS,
        row_group_target_bytes: icefalldb_core::DEFAULT_ROW_GROUP_TARGET_BYTES,
        dropped_columns: vec![],
        max_field_id: 0,
    };
    schema.assign_field_ids(None);
    Ok(schema)
}

/// Parse a simple CSV line. Does not handle commas inside quoted fields, which
/// is fine for the identifiers and type names returned by `DESCRIBE`.
fn parse_csv_line(line: &str) -> Vec<&str> {
    line.split(',').collect()
}

/// Map a DuckDB type name to a IcefallDB type string.
fn duckdb_type_to_icefalldb(duckdb_type: &str) -> anyhow::Result<String> {
    let normalized = duckdb_type.trim().to_uppercase();
    if normalized == "TIMESTAMP WITH TIME ZONE" {
        anyhow::bail!("IcefallDB does not support time-zoned timestamps");
    }
    let base = normalized.split_whitespace().next().unwrap_or(&normalized);
    let icefalldb = match base {
        "TINYINT" => "int8",
        "SMALLINT" => "int16",
        "INTEGER" => "int32",
        "BIGINT" => "int64",
        "UTINYINT" => "uint8",
        "USMALLINT" => "uint16",
        "UINTEGER" => "uint32",
        "UBIGINT" => "uint64",
        "FLOAT" => "float32",
        "DOUBLE" => "float64",
        "VARCHAR" => "utf8",
        "BOOLEAN" => "bool",
        "TIMESTAMP" => "timestamp[us]",
        other => anyhow::bail!("unsupported DuckDB type for IcefallDB schema: {}", other),
    };
    Ok(icefalldb.to_string())
}

async fn load_table_schema(storage: &dyn Storage, table: &str) -> anyhow::Result<Schema> {
    // Prefer the catalog when a valid manifest pointer exists.
    if let Ok(catalog) = Catalog::load(storage, table).await {
        if let Some(schema) = catalog.latest_schema() {
            return Ok(schema.clone());
        }
    }

    let schema_pointer_path = format!("{}/_schema.json", table);
    let schema_pointer_data = storage.read(&schema_pointer_path).await.with_context(|| {
        format!(
            "reading schema pointer {} (table may not exist)",
            schema_pointer_path
        )
    })?;
    let schema_pointer: serde_json::Value = serde_json::from_slice(&schema_pointer_data)
        .with_context(|| format!("parsing schema pointer {}", schema_pointer_path))?;
    let schema_id = schema_pointer
        .get("latest")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("missing or invalid 'latest' in {}", schema_pointer_path))?;

    let schema_path = format!("{}/{}", table, Schema::filename(schema_id));
    let schema_data = storage
        .read(&schema_path)
        .await
        .with_context(|| format!("reading schema file {}", schema_path))?;
    let schema = serde_json::from_slice(&schema_data)
        .with_context(|| format!("parsing schema file {}", schema_path))?;
    Ok(schema)
}

async fn read_batches(path: &Path) -> anyhow::Result<Vec<RecordBatch>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "arrow" => read_arrow_batches(path).await,
        "parquet" => read_parquet_batches(path).await,
        other => Err(anyhow::anyhow!(
            "unsupported input format '{}'; expected .arrow or .parquet",
            other
        )),
    }
}

async fn read_arrow_batches(path: &Path) -> anyhow::Result<Vec<RecordBatch>> {
    let data = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading arrow file {}", path.display()))?;

    // Try IPC file format first.
    let cursor = std::io::Cursor::new(data.clone());
    if let Ok(reader) = FileReader::try_new(cursor, None) {
        return reader
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("reading arrow IPC file");
    }

    // Fall back to IPC stream format.
    let cursor = std::io::Cursor::new(data);
    let reader = StreamReader::try_new(cursor, None)
        .context("arrow file is neither a valid IPC file nor stream")?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("reading arrow IPC stream")
}

async fn read_parquet_batches(path: &Path) -> anyhow::Result<Vec<RecordBatch>> {
    let data = tokio::fs::read(path)
        .await
        .with_context(|| format!("reading parquet file {}", path.display()))?;
    let bytes = Bytes::from(data);
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .with_context(|| format!("opening parquet file {}", path.display()))?;
    let reader = builder
        .build()
        .context("building parquet record batch reader")?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("reading parquet batches")
}

async fn run_compact(
    db: &Path,
    table: &str,
) -> icefalldb_core::Result<icefalldb_core::CompactionResult> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)?;
    Compactor::new(&storage, table).compact().await
}

async fn run_gc(
    db: &Path,
    table: &str,
    retain_snapshots: usize,
) -> icefalldb_core::Result<icefalldb_core::GcResult> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)?;
    GarbageCollector::new(&storage, table, retain_snapshots)
        .run()
        .await
}

/// Combined result returned by [`run_optimize`].
struct OptimizeResult {
    compaction: icefalldb_core::CompactionResult,
    gc: icefalldb_core::GcResult,
}

async fn run_optimize(
    db: &Path,
    table: &str,
    retain_snapshots: usize,
    sort_keys: Vec<String>,
) -> icefalldb_core::Result<OptimizeResult> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)?;
    let compaction = Compactor::with_options(
        &storage,
        table,
        icefalldb_core::CompactionOptions {
            force: true,
            sort_keys,
            ..icefalldb_core::CompactionOptions::default()
        },
    )
    .compact()
    .await?;
    let gc = GarbageCollector::new(&storage, table, retain_snapshots)
        .run()
        .await?;
    Ok(OptimizeResult { compaction, gc })
}

async fn run_create_index(
    db: &Path,
    table: &str,
    column: &str,
    name: Option<&str>,
    index_type: &str,
    unique: bool,
) -> anyhow::Result<()> {
    validate_table(table)?;

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let catalog = DatabaseCatalog::new(storage.clone());
    let lock = catalog
        .acquire_lock(Duration::from_secs(30))
        .await
        .with_context(|| "acquiring catalog lock")?;

    // Validate that the column exists in the table's latest schema.
    let schema = load_table_schema(storage.as_ref(), table)
        .await
        .with_context(|| format!("loading schema for table {}", table))?;
    if !schema.columns.iter().any(|c| c.name == column) {
        anyhow::bail!("column '{}' not found in table '{}'", column, table);
    }

    let default_name = format!("{}_{}_idx", table, column);
    let name = name.unwrap_or(&default_name);

    // Build the index in memory first so a build failure leaves the catalog
    // unchanged. For empty tables there is no manifest, so nothing to build.
    let manifest = Catalog::load(storage.as_ref(), table)
        .await
        .ok()
        .and_then(|c| c.latest_manifest().cloned());
    let index = if let Some(manifest) = manifest {
        let definition = IndexDefinition {
            name: name.to_string(),
            table: table.to_string(),
            column: column.to_string(),
            unique,
        };
        Some(
            build_btree_index(storage.as_ref(), &definition, &manifest)
                .await
                .with_context(|| format!("building index '{}' for table '{}'", name, table))?,
        )
    } else {
        None
    };

    catalog
        .create_index_definition_with_options(&lock, name, table, column, index_type, unique)
        .await
        .map_err(|e| match e {
            icefalldb_core::IcefallDBError::TableAlreadyExists(_) => {
                icefalldb_core::IcefallDBError::Other(
                    format!("index '{}' already exists", name).into(),
                )
            }
            other => other,
        })
        .with_context(|| format!("creating index '{}'", name))?;

    if let Some(index) = index {
        index
            .save(storage.as_ref())
            .await
            .with_context(|| format!("saving index '{}'", name))?;
    }

    println!("created index {} on {}.{}", name, table, column);
    Ok(())
}

async fn run_check(
    db: &Path,
    table: &str,
    key_file: Option<&Path>,
) -> icefalldb_core::Result<CheckResult> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)?;
    let options = {
        #[cfg(feature = "encryption")]
        {
            let provider = encryption::provider_from(key_file);
            icefalldb_core::check::CheckOptions::new().with_key_provider(provider)
        }
        #[cfg(not(feature = "encryption"))]
        {
            let _ = key_file;
            icefalldb_core::check::CheckOptions::new()
        }
    };
    let checker = Checker::new_with_options(&storage, table, options);
    checker.check().await
}

async fn run_snapshots(db: &Path, table: &str) -> anyhow::Result<()> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)
        .with_context(|| format!("opening database directory {}", db.display()))?;
    let snaps = list_snapshots(&storage, table)
        .await
        .with_context(|| format!("listing snapshots for table '{}'", table))?;
    println!(
        "{:>10}  {:<25}  {:>12}  {:>9}  parent",
        "sequence", "committed_at", "rows", "fragments"
    );
    if snaps.is_empty() {
        println!("(no snapshots)");
        return Ok(());
    }
    let mut any_wal_folded = false;
    for s in snaps {
        let hash = s
            .parent_hash
            .as_deref()
            .map(|h| &h[..h.len().min(16)])
            .unwrap_or("-");
        let wal_note = if s.wal_folded {
            any_wal_folded = true;
            " *"
        } else {
            ""
        };
        println!(
            "{:>10}  {:<25}  {:>12}  {:>9}  {}{}",
            s.sequence,
            s.committed_at.unwrap_or_default(),
            s.rows,
            s.fragments,
            hash,
            wal_note,
        );
    }
    if any_wal_folded {
        println!("* rows include WAL-folded live-row adjustments (DELETE-only exact; UPDATE/MERGE may undercount until checkpoint)");
    }
    Ok(())
}

async fn run_doctor_diagnose(
    db: &Path,
    table: &str,
) -> icefalldb_core::Result<icefalldb_core::DiagnosisResult> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)?;
    let doctor = Doctor::new(&storage, table);
    doctor.diagnose().await
}

async fn run_doctor_repair(
    db: &Path,
    table: &str,
) -> icefalldb_core::Result<icefalldb_core::RepairResult> {
    validate_table(table)?;
    let storage = LocalStorage::new(db)?;
    let doctor = Doctor::new(&storage, table);
    doctor.repair().await
}

async fn run_iceberg_export(
    db: &Path,
    table: &str,
    output: &Path,
    snapshot: Option<u64>,
) -> anyhow::Result<PathBuf> {
    validate_table(table)?;

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(db)
            .with_context(|| format!("opening database directory {}", db.display()))?,
    );

    let db_abs = std::path::absolute(db)
        .with_context(|| format!("resolving database directory {}", db.display()))?;
    let table_root = db_abs.join(table);
    let table_root_uri = format!("file://{}", table_root.to_string_lossy().replace('\\', "/"));

    icefalldb_core::iceberg::export_table(
        storage.as_ref(),
        table,
        output,
        snapshot,
        &table_root_uri,
    )
    .await
    .with_context(|| format!("exporting table '{}' to Iceberg", table))
}

fn provider_config_from_env_or_default() -> anyhow::Result<ProviderConfig> {
    let mut config = ProviderConfig::default();
    if let Ok(v) = std::env::var("ICEFALLDB_QUERY_BATCH_SIZE") {
        config.batch_size = v.parse().context("invalid ICEFALLDB_QUERY_BATCH_SIZE")?;
    }
    if let Ok(v) = std::env::var("ICEFALLDB_QUERY_TARGET_PARTITIONS") {
        config.target_partitions = v
            .parse()
            .context("invalid ICEFALLDB_QUERY_TARGET_PARTITIONS")?;
    }
    Ok(config)
}

#[allow(clippy::too_many_arguments)]
async fn run_query(
    path: &Path,
    sql: &str,
    extra_tables: &[String],
    format: QueryOutputFormat,
    result_cache_mb: u64,
    server: Option<&str>,
    snapshot: Option<u64>,
    key_file: Option<&Path>,
) -> anyhow::Result<()> {
    // Guard: --snapshot and --server are mutually exclusive.  Time-travel reads
    // require the standalone engine; the daemon always returns latest data and
    // has no API to request a historical snapshot.
    if snapshot.is_some() && server.is_some() {
        anyhow::bail!(
            "--snapshot and --server cannot be used together; \
             time-travel reads require the standalone engine"
        );
    }
    // The HTTP server has no encryption key provider and its own plaintext
    // result cache, so it cannot serve encrypted tables. A `--key-file` is only
    // meaningful for the standalone encrypted-read path.
    if key_file.is_some() && server.is_some() {
        anyhow::bail!(
            "--key-file and --server cannot be used together; \
             the server does not decrypt encrypted tables"
        );
    }
    // The server cannot serve encrypted tables. Reject up front when the local
    // path resolves to an encrypted table (a table directory, or any `--table`
    // under a database directory, carrying an `_encryption.json` marker).
    if server.is_some() {
        let encrypted = path.join("_encryption.json").is_file()
            || extra_tables
                .iter()
                .any(|t| path.join(t).join("_encryption.json").is_file());
        if encrypted {
            anyhow::bail!(
                "cannot query an encrypted table via --server; the server does not \
                 decrypt encrypted tables. Query it locally instead."
            );
        }
    }

    // OPTIONAL daemon path: route to a running server (which pays open once). The
    // default (no --server) standalone path below is unchanged.
    if let Some(url) = server {
        let daemon = client::DaemonClient::new(url)?;
        if client::is_mutation(sql) {
            let affected = daemon.mutate(sql)?;
            println!("{affected}");
        } else {
            println!("{}", daemon.query(sql)?);
        }
        return Ok(());
    }

    let path = std::path::absolute(path)
        .with_context(|| format!("resolving query path {}", path.display()))?;

    // Determine whether `path` points at a table directory or a database directory.
    let (db_path, tables): (PathBuf, Vec<String>) = if path.join("_manifest.json").is_file() {
        let db_path = path
            .parent()
            .with_context(|| format!("table directory {} has no parent", path.display()))?
            .to_path_buf();
        let table_name = path
            .file_name()
            .with_context(|| format!("table directory {} has no name", path.display()))?
            .to_string_lossy()
            .to_string();
        let mut tables = vec![table_name];
        tables.extend(extra_tables.iter().cloned());
        (db_path, tables)
    } else {
        if extra_tables.is_empty() {
            anyhow::bail!(
                "{} is not a table directory and no --table flags were provided",
                path.display()
            );
        }
        (path, extra_tables.to_vec())
    };

    let storage: Arc<dyn Storage> = Arc::new(
        LocalStorage::new(&db_path)
            .with_context(|| format!("opening database directory {}", db_path.display()))?,
    );

    // Encrypted-table read path: when any registered table carries an
    // `_encryption.json` marker, route to the encrypted reader (which builds an
    // encryption-aware session and decrypts via the key provider).
    #[cfg(feature = "encryption")]
    {
        let mut any_encrypted = false;
        for table in &tables {
            if encryption::read_marker(&storage, table).await?.is_some() {
                any_encrypted = true;
                break;
            }
        }
        if any_encrypted {
            return run_query_encrypted(&storage, &tables, sql, format, snapshot, key_file).await;
        }
    }
    #[cfg(not(feature = "encryption"))]
    {
        let _ = key_file;
        for table in &tables {
            if storage
                .exists(&format!("{}/_encryption.json", table))
                .await?
            {
                anyhow::bail!(
                    "table '{}' is encrypted but this `icefalldb` build was compiled \
                     without the `encryption` feature",
                    table
                );
            }
        }
    }

    let provider_config = provider_config_from_env_or_default()?;
    let ctx = icefalldb_session(
        std::cmp::max(1, provider_config.target_partitions),
        provider_config.batch_size,
    );

    // Collect snapshot sequences while registering providers so we can build
    // the result-cache key without re-reading manifests.
    let mut table_names: Vec<String> = Vec::with_capacity(tables.len());
    let mut snapshots: Vec<u64> = Vec::with_capacity(tables.len());
    for table in &tables {
        validate_table(table)?;
        let provider = if let Some(seq) = snapshot {
            // Time-travel read: build a provider pinned to the historical snapshot.
            // Map SnapshotNotFound to a clean, user-facing error message.
            IcefallDBTableProvider::new_at_snapshot(
                Arc::clone(&storage),
                table,
                provider_config,
                seq,
            )
            .await
            .map_err(|e| match e {
                QueryError::Core(IcefallDBError::SnapshotNotFound(n)) => {
                    anyhow::anyhow!("snapshot {} not found for table '{}'", n, table)
                }
                other => anyhow::anyhow!("{}", other)
                    .context(format!("loading table '{}' at snapshot {}", table, seq)),
            })?
        } else {
            IcefallDBTableProvider::new(Arc::clone(&storage), table, provider_config)
                .await
                .with_context(|| format!("loading table '{}'", table))?
        };
        table_names.push(table.clone());
        snapshots.push(provider.pinned_sequence());
        ctx.register_table(table, std::sync::Arc::new(provider))
            .map_err(|e| anyhow::anyhow!("registering table '{}': {}", table, e))?;
    }

    // Route mutation statements (DELETE, UPDATE, MERGE) through execute_sql so
    // that changes are committed to storage, the provider is refreshed, and the
    // affected-row count is reported.  Everything else (SELECT, DDL, multi-table
    // read queries) falls through to ctx.sql.
    //
    // For DELETE and UPDATE the target table is unambiguous (single-table DML),
    // so we require tables.len() == 1.  For MERGE the source is usually an
    // inline subquery — only the target table needs to be registered, so we
    // also require tables.len() == 1 and pass that table as the table_root.
    // execute_sql re-parses the target from the SQL AST and uses table_root
    // only to open the Writer.
    //
    // Mutations are never routed when `--snapshot` is set: a time-travel read
    // is inherently read-only, and mutating via a historical scan plan would be
    // incorrect.
    let sql_upper = sql.trim_start().to_uppercase();

    // Guard: mutations are not allowed on a past snapshot.  Without this check
    // the mutation branches below are simply skipped (snapshot.is_none() is
    // false), and the statement falls through to ctx.sql() which produces a
    // confusing DataFusion error instead of a clear user message.
    if snapshot.is_some()
        && (sql_upper.starts_with("DELETE")
            || sql_upper.starts_with("UPDATE")
            || sql_upper.starts_with("MERGE"))
    {
        anyhow::bail!(
            "--snapshot is read-only; DELETE/UPDATE/MERGE are not allowed on a past snapshot"
        );
    }

    if snapshot.is_none() && sql_upper.starts_with("DELETE") && tables.len() == 1 {
        let affected = execute_sql(&ctx, Arc::clone(&storage), &tables[0], sql)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        eprintln!("{affected} row(s) deleted");
        return Ok(());
    }
    if snapshot.is_none() && sql_upper.starts_with("UPDATE") && tables.len() == 1 {
        let affected = execute_sql(&ctx, Arc::clone(&storage), &tables[0], sql)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        eprintln!("{affected} row(s) updated");
        return Ok(());
    }
    if snapshot.is_none() && sql_upper.starts_with("MERGE") && tables.len() == 1 {
        let affected = execute_sql(&ctx, Arc::clone(&storage), &tables[0], sql)
            .await
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        eprintln!("{affected} row(s) affected");
        return Ok(());
    }

    // Result cache: <db_path>/_query_cache, shared with the Python adapter.
    // The cache dir is always rooted at the database directory so a CLI result
    // is readable by any adapter opened against the same db and vice-versa.
    let cache_max_bytes = result_cache_mb.saturating_mul(1024 * 1024);
    let cache = ResultCache::new(
        db_path.join("_query_cache"),
        cache_max_bytes,
        EvictPolicy::Lru,
    )
    .map_err(|e| anyhow::anyhow!("result cache: {e}"))?;

    // Cache-hit fast path: return stored result without running DataFusion.
    if cache.enabled() && is_cacheable_select(sql, &table_names) {
        if let Some(batches) = cache
            .get(sql, &table_names, &snapshots)
            .map_err(|e| anyhow::anyhow!("result cache get: {e}"))?
        {
            return match format {
                QueryOutputFormat::Json => write_json(&batches),
                QueryOutputFormat::Csv => write_csv(&batches),
            };
        }
    }

    let df = ctx.sql(sql).await.map_err(|e| anyhow::anyhow!("{}", e))?;
    let batches = df.collect().await.map_err(|e| anyhow::anyhow!("{}", e))?;

    // Cache the result for future queries with the same SQL + snapshots.
    if cache.enabled() && is_cacheable_select(sql, &table_names) {
        cache
            .put(sql, &table_names, &snapshots, &batches)
            .map_err(|e| anyhow::anyhow!("result cache put: {e}"))?;
    }

    match format {
        QueryOutputFormat::Json => write_json(&batches),
        QueryOutputFormat::Csv => write_csv(&batches),
    }
}

/// Read path for encrypted tables: build an encryption-aware session, register
/// each table (decrypting via the key provider resolved from a `--key-file` or
/// `ICEFALLDB_KEY_*` env vars), and run a read-only query. Encrypted results
/// bypass the on-disk result cache, which stores plaintext Arrow IPC and would
/// otherwise defeat at-rest encryption.
#[cfg(feature = "encryption")]
async fn run_query_encrypted(
    storage: &Arc<dyn Storage>,
    tables: &[String],
    sql: &str,
    format: QueryOutputFormat,
    snapshot: Option<u64>,
    key_file: Option<&Path>,
) -> anyhow::Result<()> {
    if snapshot.is_some() {
        anyhow::bail!("time-travel reads (--snapshot) on encrypted tables are not yet supported");
    }
    let sql_upper = sql.trim_start().to_uppercase();
    if sql_upper.starts_with("DELETE")
        || sql_upper.starts_with("UPDATE")
        || sql_upper.starts_with("MERGE")
    {
        anyhow::bail!(
            "mutations on encrypted tables are not yet supported via the CLI; \
             encrypted tables are currently read-only from `icefalldb query`"
        );
    }

    let provider_config = provider_config_from_env_or_default()?;
    let key_provider = encryption::provider_from(key_file);
    let ctx = icefalldb_encrypted_session(
        std::cmp::max(1, provider_config.target_partitions),
        provider_config.batch_size,
        Arc::clone(&key_provider),
    );

    for table in tables {
        validate_table(table)?;
        if let Some(marker) = encryption::read_marker(storage, table).await? {
            let provider = encryption::open_encrypted_provider(
                storage,
                table,
                provider_config,
                Arc::clone(&key_provider),
                &marker,
            )
            .await?;
            ctx.register_table(table, Arc::new(provider))
                .map_err(|e| anyhow::anyhow!("registering encrypted table '{}': {}", table, e))?;
        } else {
            let provider = IcefallDBTableProvider::new(Arc::clone(storage), table, provider_config)
                .await
                .with_context(|| format!("loading table '{}'", table))?;
            ctx.register_table(table, Arc::new(provider))
                .map_err(|e| anyhow::anyhow!("registering table '{}': {}", table, e))?;
        }
    }

    let df = ctx.sql(sql).await.map_err(|e| anyhow::anyhow!("{}", e))?;
    let batches = df.collect().await.map_err(|e| anyhow::anyhow!("{}", e))?;
    match format {
        QueryOutputFormat::Json => write_json(&batches),
        QueryOutputFormat::Csv => write_csv(&batches),
    }
}

fn write_json(batches: &[RecordBatch]) -> anyhow::Result<()> {
    let stdout = std::io::stdout();
    // Keep null keys in the JSON output so consumers see a consistent set of
    // columns even when every selected value is NULL.
    let mut writer = arrow::json::WriterBuilder::new()
        .with_explicit_nulls(true)
        .build::<_, arrow::json::writer::JsonArray>(stdout.lock());
    for batch in batches {
        writer
            .write(batch)
            .context("serializing query results to JSON")?;
    }
    writer.finish().context("finishing JSON output")?;
    println!();
    Ok(())
}

fn write_csv(batches: &[RecordBatch]) -> anyhow::Result<()> {
    let stdout = std::io::stdout();
    let mut writer = arrow::csv::WriterBuilder::new()
        .with_header(true)
        .build(stdout.lock());
    for batch in batches {
        writer
            .write(batch)
            .context("serializing query results to CSV")?;
    }
    Ok(())
}

fn print_issues(result: &CheckResult) {
    for issue in &result.issues {
        let severity_label = match issue.severity {
            Severity::Error => "ERROR",
            Severity::Warning => "WARNING",
            Severity::Info => "INFO",
        };
        println!("{} {}: {}", severity_label, issue.code, issue.message);
    }
}

/// Run a single DuckDB CLI command and surface any error.
///
/// DuckDB may print errors to stderr and still exit with status 0 when the
/// command string contains multiple statements or certain fatal errors, so
/// stderr is also checked for the word "Error".
fn run_duckdb_command(sql: &str) -> anyhow::Result<()> {
    let output = std::process::Command::new("duckdb")
        .arg("-c")
        .arg(sql)
        .output()
        .with_context(|| "running duckdb CLI")?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() || stderr.contains("Error") {
        anyhow::bail!("duckdb query failed: {}", stderr);
    }
    Ok(())
}

/// Build the DuckDB preamble that registers every IcefallDB table as a temp view.
///
/// The view's own derived table is also registered if it already exists, so
/// self-referential view definitions can see the previous snapshot. Empty
/// source tables are registered as zero-row relations with the correct schema.
async fn build_refresh_preamble(
    db: &Path,
    storage: &dyn Storage,
    view_table: &str,
) -> anyhow::Result<String> {
    let mut preamble = String::new();

    for entry in std::fs::read_dir(db)
        .with_context(|| format!("reading database directory {}", db.display()))?
    {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let table_dir = entry.path();
        if !table_dir.join("_manifest.json").is_file() || !table_dir.join("_schema.json").is_file()
        {
            continue;
        }
        let table_name = table_dir.file_name().unwrap().to_string_lossy();
        if table_name == "views" {
            continue;
        }
        preamble.push_str(&register_table_as_temp_view(storage, &table_name, &table_dir).await?);
    }

    // Register the view's own derived table if it already exists.
    let view_dir = db.join(view_table);
    if view_dir.join("_manifest.json").is_file() && view_dir.join("_schema.json").is_file() {
        preamble.push_str(&register_table_as_temp_view(storage, view_table, &view_dir).await?);
    }

    Ok(preamble)
}

/// Register a single IcefallDB table as a DuckDB temp view.
async fn register_table_as_temp_view(
    storage: &dyn Storage,
    table_name: &str,
    table_dir: &Path,
) -> anyhow::Result<String> {
    let catalog = Catalog::load(storage, table_name)
        .await
        .with_context(|| format!("loading catalog for table {}", table_name))?;
    let latest = catalog.latest_manifest();

    let has_data = latest.map(|m| !m.row_groups.is_empty()).unwrap_or(false);
    if !has_data {
        // Empty table: register a zero-row relation with the current schema.
        let schema = load_table_schema(storage, table_name)
            .await
            .with_context(|| format!("loading schema for empty table {}", table_name))?;
        let columns: Vec<String> = schema
            .columns
            .iter()
            .map(|col| {
                let duckdb_type = icefalldb_type_to_duckdb(&col.r#type)?;
                Ok(format!(
                    "CAST(NULL AS {}) AS \"{}\"",
                    duckdb_type,
                    col.name.replace('"', "\"\"")
                ))
            })
            .collect::<anyhow::Result<_>>()?;
        let select_list = columns.join(", ");
        return Ok(format!(
            "CREATE TEMP VIEW IF NOT EXISTS \"{}\" AS SELECT {} WHERE FALSE;\n",
            table_name.replace('"', "\"\""),
            select_list
        ));
    }

    let mut files = Vec::new();
    for rg in &latest.unwrap().row_groups {
        let data_path = table_dir.join(&rg.data);
        let resolved = std::path::absolute(&data_path)
            .with_context(|| format!("resolving row group path {}", data_path.display()))?;
        files.push(format!(
            "'{}'",
            resolved
                .to_string_lossy()
                .replace('\\', "/")
                .replace('\'', "''")
        ));
    }
    let files_list = files.join(", ");
    Ok(format!(
        "CREATE TEMP VIEW IF NOT EXISTS \"{}\" AS SELECT * FROM read_parquet([{}]);\n",
        table_name.replace('"', "\"\""),
        files_list
    ))
}

/// Map a IcefallDB type string to a DuckDB type name.
fn icefalldb_type_to_duckdb(type_str: &str) -> anyhow::Result<&'static str> {
    match type_str {
        "int8" => Ok("TINYINT"),
        "int16" => Ok("SMALLINT"),
        "int32" => Ok("INTEGER"),
        "int64" => Ok("BIGINT"),
        "uint8" => Ok("UTINYINT"),
        "uint16" => Ok("USMALLINT"),
        "uint32" => Ok("UINTEGER"),
        "uint64" => Ok("UBIGINT"),
        "float32" => Ok("FLOAT"),
        "float64" => Ok("DOUBLE"),
        "utf8" | "string" => Ok("VARCHAR"),
        "large_utf8" => Ok("VARCHAR"),
        "bool" => Ok("BOOLEAN"),
        "timestamp[us]" | "timestamp" => Ok("TIMESTAMP"),
        other => anyhow::bail!("unsupported IcefallDB type for DuckDB: {}", other),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SqlTokenKind {
    Word,
    Semicolon,
    Comment,
    String,
    Other,
}

/// A minimal SQL tokenizer that recognizes single-line (`--`) and block (`/* */`)
/// comments, single/double-quoted strings (with doubled-quote escapes), and
/// semicolons. It is used to validate view definitions without being confused
/// by semicolons inside string literals.
fn tokenize_sql(sql: &str) -> Vec<(SqlTokenKind, usize, usize)> {
    let mut tokens = Vec::new();
    let bytes = sql.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '-' && i + 1 < bytes.len() && bytes[i + 1] == b'-' {
            let start = i;
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            tokens.push((SqlTokenKind::Comment, start, i));
            continue;
        }
        if c == '/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            if i + 1 < bytes.len() {
                i += 2;
            }
            tokens.push((SqlTokenKind::Comment, start, i));
            continue;
        }
        if c == '\'' || c == '"' {
            let quote = c;
            let start = i;
            i += 1;
            while i < bytes.len() {
                if bytes[i] == quote as u8 {
                    if i + 1 < bytes.len() && bytes[i + 1] == quote as u8 {
                        i += 2;
                        continue;
                    }
                    i += 1;
                    break;
                }
                i += 1;
            }
            tokens.push((SqlTokenKind::String, start, i));
            continue;
        }
        if c == ';' {
            tokens.push((SqlTokenKind::Semicolon, i, i + 1));
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        while i < bytes.len() {
            let ch = bytes[i] as char;
            if ch.is_whitespace() || ch == ';' {
                break;
            }
            i += 1;
        }
        let kind = if sql[start..i]
            .chars()
            .next()
            .map(|ch| ch.is_alphabetic() || ch == '_')
            .unwrap_or(false)
        {
            SqlTokenKind::Word
        } else {
            SqlTokenKind::Other
        };
        tokens.push((kind, start, i));
    }
    tokens
}

/// Split SQL into top-level statement ranges, ignoring semicolons inside
/// strings and comments. Empty trailing statements are omitted.
fn split_sql_statements(sql: &str) -> Vec<(usize, usize)> {
    let tokens = tokenize_sql(sql);
    let mut statements = Vec::new();
    let mut stmt_start: Option<usize> = None;
    for (kind, start, _end) in tokens {
        if kind == SqlTokenKind::Semicolon {
            if let Some(s) = stmt_start.take() {
                statements.push((s, start));
            }
        } else if stmt_start.is_none() && kind != SqlTokenKind::Comment {
            stmt_start = Some(start);
        }
    }
    if let Some(s) = stmt_start {
        statements.push((s, sql.len()));
    }
    statements
}

/// Validate that `sql` is a single SELECT or WITH statement.
fn validate_view_sql(sql: &str) -> anyhow::Result<()> {
    extract_view_query(sql).map(|_| ())
}

/// Extract the single top-level SELECT/WITH statement from a view query,
/// normalizing away a trailing semicolon.
fn extract_view_query(sql: &str) -> anyhow::Result<String> {
    let ranges = split_sql_statements(sql);
    let statements: Vec<&str> = ranges
        .iter()
        .map(|(s, e)| sql[*s..*e].trim())
        .filter(|t| !t.is_empty())
        .collect();
    if statements.is_empty() {
        anyhow::bail!("query is empty");
    }
    if statements.len() > 1 {
        anyhow::bail!("query must contain a single SELECT statement");
    }
    let first = statements[0];
    let first_word = tokenize_sql(first)
        .into_iter()
        .find(|(kind, _, _)| *kind == SqlTokenKind::Word)
        .map(|(_, start, end)| first[start..end].to_lowercase())
        .unwrap_or_default();
    if first_word != "select" && first_word != "with" {
        anyhow::bail!("query must be a single SELECT statement");
    }
    Ok(first.to_string())
}
