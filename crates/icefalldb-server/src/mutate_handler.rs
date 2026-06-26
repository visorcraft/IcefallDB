//! `POST /mutate` — run a single `DELETE`/`UPDATE`/`MERGE` through the daemon's
//! own writer and incrementally refresh the registered provider.
//!
//! [`execute_sql`] both commits the mutation and applies the resulting
//! `CommitDelta` to the provider registered in the server's `SessionContext`
//! (via `apply_committed_delta` — no full reload), so a long-lived daemon pays
//! table open + engine startup **once** across many mutations. The daemon is an
//! optional performance opt-in; the stateless CLI/in-process paths are unchanged.

use crate::error::ServerError;
use crate::server::Server;
use axum::extract::State;
use axum::Json;
use icefalldb_query::{execute_sql, mutation_target_table};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub(crate) struct MutateRequest {
    pub sql: String,
}

#[derive(Serialize)]
pub(crate) struct MutateResponse {
    pub affected: u64,
}

pub(crate) async fn mutate_handler(
    State(server): State<Server>,
    Json(req): Json<MutateRequest>,
) -> Result<Json<MutateResponse>, ServerError> {
    let table = mutation_target_table(&req.sql).ok_or_else(|| {
        ServerError::BadRequest("expected a single-table DELETE/UPDATE/MERGE".into())
    })?;
    // Serialize the locate→commit→refresh against concurrent mutations so a
    // request cannot read a pre-image from a snapshot another commit is about to
    // supersede (lost update / stale-provider apply failure).
    let _guard = server.mutate_lock().lock_owned().await;
    let affected = execute_sql(server.ctx(), server.storage(), &table, &req.sql).await?;
    // Invalidate the result cache so follow-up SELECTs see the new snapshot.
    // Best-effort: a cache clear failure must not fail the mutation response.
    let _ = server.result_cache().clear();
    Ok(Json(MutateResponse { affected }))
}
