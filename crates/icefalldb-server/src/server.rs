use crate::error::ServerError;
use crate::sql_insert::parse_insert_values;
use crate::transaction::TransactionManager;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use datafusion::execution::context::SessionContext;
use datafusion::sql::parser::DFParser;
use icefalldb_core::catalog::Catalog;
use icefalldb_core::storage::local::LocalStorage;
use icefalldb_core::storage::Storage;
use icefalldb_core::wal::Wal;
use icefalldb_core::{build_btree_index, DatabaseCatalog, IndexDefinition};
use icefalldb_query::result_cache::{is_cacheable_select, EvictPolicy, ResultCache};
use icefalldb_query::{icefalldb_session, IcefallDBTableProvider, ProviderConfig};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlparser::ast::Statement as SqlStatement;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// HTTP SQL server for IcefallDB.
///
/// The server registers tables from the central catalog (with legacy directory
/// fallback) at startup, including empty tables. The initial MVP materializes
/// full query results in memory before serializing to JSON, which limits
/// result-set size to available memory. DDL mutations (CREATE TABLE / DROP
/// TABLE) require a server restart to be visible in the registered DataFusion
/// catalog.
#[derive(Clone)]
pub struct Server {
    _db_path: PathBuf,
    ctx: Arc<SessionContext>,
    tx_manager: Arc<TransactionManager>,
    storage: Arc<dyn Storage>,
    /// Serializes daemon-side commit+refresh sequences (`/mutate`, `/tx/commit`).
    /// The on-disk writer is already single-writer via `flock`, but the locate /
    /// pre-image read happens before that lock and `apply_committed_delta`
    /// requires the provider's pinned sequence to equal the delta's
    /// `previous_sequence` — so two concurrent mutations could read a stale
    /// snapshot (lost update) or fail the apply. Holding this across the whole
    /// locate→commit→refresh makes each mutation see the prior one's snapshot.
    mutate_lock: Arc<tokio::sync::Mutex<()>>,
    /// On-disk result cache for eligible SELECTs. Stored as Arrow IPC under
    /// `<db>/_query_cache`. Each cache key encodes the SQL text and the
    /// `pinned_sequence` of every registered table, so snapshot advances
    /// naturally produce key misses without an explicit eviction step. `/mutate`
    /// calls `clear()` to drop all entries after committing a mutation.
    result_cache: ResultCache,
    /// Registered table names (set at startup; DDL requires a restart).
    /// Kept here so the `/sql` handler can snapshot their sequences without
    /// iterating the DataFusion catalog.
    table_names: Vec<String>,
}

impl Server {
    /// Start the server with the default result-cache budget (1 GiB).
    pub async fn new(db_path: &Path) -> Result<Self, ServerError> {
        Self::new_with_cache_mb(db_path, 1024).await
    }

    /// Start the server with an explicit result-cache budget in MiB (`0` disables).
    pub async fn new_with_cache_mb(
        db_path: &Path,
        result_cache_mb: u64,
    ) -> Result<Self, ServerError> {
        let storage: Arc<dyn Storage> = Arc::new(LocalStorage::new(db_path)?);
        let ctx = icefalldb_session(num_cpus::get(), 8192);
        let config = ProviderConfig::default();

        // Discover tables using the central catalog or legacy directory scan.
        let tables = list_tables(Arc::clone(&storage)).await?;
        let mut registered = Vec::with_capacity(tables.len());
        for table in &tables {
            // The server has no encryption key provider and keeps a plaintext
            // result cache, so it cannot serve encrypted tables. Skip them: a
            // query that references one fails cleanly ("table not found")
            // instead of leaking decrypted rows through the cache.
            if storage.exists(&format!("{table}/_encryption.json")).await? {
                eprintln!(
                    "icefalldb-server: skipping encrypted table '{table}' \
                     (encrypted tables are not served over HTTP)"
                );
                continue;
            }
            let provider = IcefallDBTableProvider::new(Arc::clone(&storage), table, config)
                .await
                .map_err(|e| ServerError::Internal(format!("table {table}: {e}")))?;
            ctx.register_table(table, Arc::new(provider))
                .map_err(|e| ServerError::Internal(format!("register {table}: {e}")))?;
            registered.push(table.clone());
        }
        let tables = registered;

        let wal = Arc::new(Wal::open(Arc::clone(&storage)).await?);
        icefalldb_core::recovery::apply_committed_transactions(Arc::clone(&storage)).await?;
        let tx_manager = Arc::new(TransactionManager::new(
            Arc::clone(&wal),
            Arc::clone(&storage),
        ));

        let cache_dir = db_path.join("_query_cache");
        let result_cache = ResultCache::new(
            &cache_dir,
            result_cache_mb.saturating_mul(1024 * 1024),
            EvictPolicy::Lru,
        )
        .map_err(|e| ServerError::Internal(format!("result cache: {e}")))?;

        Ok(Self {
            _db_path: db_path.to_path_buf(),
            ctx: Arc::new(ctx),
            tx_manager,
            storage,
            mutate_lock: Arc::new(tokio::sync::Mutex::new(())),
            result_cache,
            table_names: tables,
        })
    }

    /// The registered DataFusion session (providers are refreshed incrementally
    /// by the mutation handlers).
    pub(crate) fn ctx(&self) -> &SessionContext {
        &self.ctx
    }

    /// The storage backend backing the registered tables.
    pub(crate) fn storage(&self) -> Arc<dyn Storage> {
        Arc::clone(&self.storage)
    }

    /// The lock serializing daemon-side commit+refresh sequences.
    pub(crate) fn mutate_lock(&self) -> Arc<tokio::sync::Mutex<()>> {
        Arc::clone(&self.mutate_lock)
    }

    /// The on-disk result cache (shared across all clones via the same dir).
    pub(crate) fn result_cache(&self) -> &ResultCache {
        &self.result_cache
    }

    pub async fn serve(self, addr: &str) -> Result<(), ServerError> {
        let app = Router::new()
            .route("/sql", post(sql_handler))
            .route("/mutate", post(crate::mutate_handler::mutate_handler))
            .route("/index", post(create_index_handler))
            .route("/tables", get(list_tables_handler))
            .route("/tx/begin", post(begin_handler))
            .route("/tx/sql", post(tx_sql_handler))
            .route("/tx/commit", post(commit_handler))
            .route("/tx/rollback", post(rollback_handler))
            .with_state(self);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;
        Ok(())
    }
}

impl Server {
    /// Start the server on a random free port and return the base URL.
    pub async fn start_for_test(
        self,
    ) -> Result<(String, tokio::task::JoinHandle<()>), ServerError> {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let app = Router::new()
            .route("/sql", post(sql_handler))
            .route("/mutate", post(crate::mutate_handler::mutate_handler))
            .route("/index", post(create_index_handler))
            .route("/tables", get(list_tables_handler))
            .route("/tx/begin", post(begin_handler))
            .route("/tx/sql", post(tx_sql_handler))
            .route("/tx/commit", post(commit_handler))
            .route("/tx/rollback", post(rollback_handler))
            .with_state(self);
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        Ok((format!("http://{}", addr), handle))
    }
}

async fn list_tables(storage: Arc<dyn Storage>) -> Result<Vec<String>, ServerError> {
    let catalog = DatabaseCatalog::new(Arc::clone(&storage));
    let data = catalog.load().await?;
    // The catalog is authoritative once it exists: a registered table set (even
    // after the last table is dropped, leaving it empty) must NOT be overridden
    // by a directory scan, or a dropped table whose data files linger on disk
    // (drop is reversible) would be resurrected.
    if storage.exists("_catalog.json").await.unwrap_or(false) {
        let mut names: Vec<String> = data.tables.keys().cloned().collect();
        names.sort();
        return Ok(names);
    }
    // Directory-scan fallback for catalog-less databases (tables made with
    // `create`, never registered centrally).
    let mut tables = Vec::new();
    for name in storage.list("").await? {
        // A table is a directory containing `_manifest.json`. Stray files at the
        // db root (`_catalog.json`, `_catalog.lock`, a leftover `.tsv`) make the
        // `<name>/_manifest.json` probe fail with "Not a directory"; treat any
        // probe error as "not a table" rather than failing startup.
        if storage
            .exists(&format!("{}/_manifest.json", name))
            .await
            .unwrap_or(false)
        {
            tables.push(name);
        }
    }
    tables.sort();
    Ok(tables)
}

#[derive(Deserialize)]
struct SqlRequest {
    sql: String,
    /// Optional historical snapshot sequence. When present the query is executed
    /// against that snapshot rather than the latest registered state. The result
    /// is NOT cached (the per-request session is ephemeral and the snapshot key
    /// is already embedded in the session itself).
    #[serde(default)]
    snapshot: Option<u64>,
}

#[derive(Serialize)]
struct SqlResponse {
    data: Vec<Value>,
}

async fn sql_handler(
    State(server): State<Server>,
    Json(req): Json<SqlRequest>,
) -> Result<Json<SqlResponse>, ServerError> {
    // When a specific historical snapshot is requested: build a fresh
    // per-request SessionContext pinned to that sequence. This does NOT mutate
    // the shared registered session and the result is NOT cached (the session is
    // ephemeral; skipping the cache is the simplest correct option — the
    // snapshot sequence is part of the provider state, so there is no collision
    // risk, but caching is unnecessary overhead for infrequent time-travel reads).
    let batches = if let Some(snap) = req.snapshot {
        let fresh_ctx = icefalldb_session(num_cpus::get(), 8192);
        let config = ProviderConfig::default();
        for table in &server.table_names {
            let provider = IcefallDBTableProvider::new_at_snapshot(
                Arc::clone(&server.storage),
                table.as_str(),
                config,
                snap,
            )
            .await
            .map_err(ServerError::from)?;
            fresh_ctx
                .register_table(table, Arc::new(provider))
                .map_err(|e| ServerError::Internal(format!("register {table}: {e}")))?;
        }
        let df = fresh_ctx.sql(&req.sql).await?;
        df.collect().await?
    } else {
        let ctx = Arc::clone(&server.ctx);
        let table_names = &server.table_names;
        let cache = &server.result_cache;

        // Collect the current snapshot sequence for each registered table. The
        // sequences are included in the cache key so an advancing snapshot
        // automatically produces a key miss without requiring explicit
        // invalidation (except after a mutation, where we call `clear()`).
        let snapshots = collect_snapshots(&ctx, table_names).await;

        // For eligible SELECTs: attempt a cache hit before running the query.
        if cache.enabled() && is_cacheable_select(&req.sql, table_names) {
            if let Ok(Some(cached)) = cache.get(&req.sql, table_names, &snapshots) {
                cached
            } else {
                let df = ctx.sql(&req.sql).await?;
                let result = df.collect().await?;
                // Best-effort write; ignore errors so the non-cache path is unchanged.
                let _ = cache.put(&req.sql, table_names, &snapshots, &result);
                result
            }
        } else {
            let df = ctx.sql(&req.sql).await?;
            df.collect().await?
        }
    };

    let mut buf = Vec::new();
    let mut writer = arrow::json::WriterBuilder::new()
        .with_explicit_nulls(true)
        .build::<_, arrow::json::writer::JsonArray>(&mut buf);
    for batch in &batches {
        writer
            .write(batch)
            .map_err(|e| ServerError::Internal(format!("json encoding: {e}")))?;
    }
    writer
        .finish()
        .map_err(|e| ServerError::Internal(format!("json encoding: {e}")))?;
    let rows: Vec<Value> = serde_json::from_slice(&buf)
        .map_err(|e| ServerError::Internal(format!("json parsing: {e}")))?;
    Ok(Json(SqlResponse { data: rows }))
}

/// Gather the current `pinned_sequence` for every registered IcefallDB table.
/// Returns 0 for any table whose provider cannot be downcast (e.g. DDL-only).
async fn collect_snapshots(ctx: &SessionContext, table_names: &[String]) -> Vec<u64> {
    let mut snapshots = Vec::with_capacity(table_names.len());
    for name in table_names {
        let seq = if let Ok(provider) = ctx.table_provider(name).await {
            (provider.as_ref() as &dyn std::any::Any)
                .downcast_ref::<IcefallDBTableProvider>()
                .map(|p| p.pinned_sequence())
                .unwrap_or(0)
        } else {
            0
        };
        snapshots.push(seq);
    }
    snapshots
}

#[derive(Serialize)]
struct TablesResponse {
    tables: Vec<String>,
}

async fn create_index_handler(
    State(server): State<Server>,
    Json(req): Json<SqlRequest>,
) -> Result<Json<Value>, ServerError> {
    let (index_name, table, column) = parse_create_index(&req.sql).ok_or_else(|| {
        ServerError::BadRequest("expected CREATE INDEX name ON table (column)".into())
    })?;

    let value = tokio::task::spawn_blocking(move || {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            let catalog = DatabaseCatalog::new(Arc::clone(&server.storage));
            let guard = catalog
                .acquire_lock(Duration::from_secs(30))
                .await
                .map_err(ServerError::from)?;
            catalog
                .create_index_definition(&guard, &index_name, &table, &column, "btree")
                .await?;

            let cat = Catalog::load(server.storage.as_ref(), &table).await?;
            let manifest = cat.latest_manifest().ok_or_else(|| {
                ServerError::NotFound(format!("table {} has no committed manifest", table))
            })?;
            let definition = IndexDefinition {
                name: index_name.clone(),
                table: table.clone(),
                column: column.clone(),
                unique: false,
            };
            let index = build_btree_index(server.storage.as_ref(), &definition, manifest).await?;
            index.save(server.storage.as_ref()).await?;

            Ok::<_, ServerError>(json!({"status": "created", "index": index_name}))
        })
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(value))
}

async fn list_tables_handler(
    State(server): State<Server>,
) -> Result<Json<TablesResponse>, ServerError> {
    let tables = list_tables(Arc::clone(&server.storage)).await?;
    Ok(Json(TablesResponse { tables }))
}

#[derive(Serialize)]
struct BeginResponse {
    tx_id: String,
}

async fn begin_handler(State(server): State<Server>) -> Result<Json<BeginResponse>, ServerError> {
    let tx_id = server.tx_manager.begin().await?;
    Ok(Json(BeginResponse { tx_id }))
}

#[derive(Deserialize)]
struct TxSqlRequest {
    tx_id: String,
    sql: String,
}

async fn tx_sql_handler(
    State(server): State<Server>,
    Json(req): Json<TxSqlRequest>,
) -> Result<Json<Value>, ServerError> {
    let (table, batch) = parse_insert_values(&*server.storage, &req.sql).await?;
    server
        .tx_manager
        .add_insert(&req.tx_id, &table, batch)
        .await?;
    Ok(Json(json!({"status": "ok"})))
}

#[derive(Deserialize)]
struct TxIdRequest {
    tx_id: String,
}

async fn commit_handler(
    State(server): State<Server>,
    Json(req): Json<TxIdRequest>,
) -> Result<Json<Value>, ServerError> {
    // Serialize with /mutate: both commit + refresh the shared providers.
    let _guard = server.mutate_lock().lock_owned().await;
    let deltas = server.tx_manager.commit(&req.tx_id).await?;

    // Incrementally refresh the registered providers so subsequent queries see
    // the committed snapshot without re-reading manifests or sidecars.
    for (table, delta) in &deltas {
        if let Ok(provider) = server.ctx.table_provider(table).await {
            if let Some(provider) =
                (provider.as_ref() as &dyn std::any::Any).downcast_ref::<IcefallDBTableProvider>()
            {
                provider.apply_committed_delta(delta).await?;
            }
        }
    }

    Ok(Json(json!({"status": "committed"})))
}

async fn rollback_handler(
    State(server): State<Server>,
    Json(req): Json<TxIdRequest>,
) -> Result<Json<Value>, ServerError> {
    server.tx_manager.rollback(&req.tx_id).await?;
    Ok(Json(json!({"status": "rolled back"})))
}

fn parse_create_index(sql: &str) -> Option<(String, String, String)> {
    let statements = DFParser::parse_sql(sql).ok()?;
    let statement = statements.into_iter().next()?;
    let datafusion::sql::parser::Statement::Statement(stmt) = statement else {
        return None;
    };
    let SqlStatement::CreateIndex(create_index) = *stmt else {
        return None;
    };

    let index_name = create_index.name.as_ref()?.to_string();
    let table = create_index.table_name.to_string();
    let column = create_index.columns.first()?.column.expr.to_string();
    Some((index_name, table, column))
}
