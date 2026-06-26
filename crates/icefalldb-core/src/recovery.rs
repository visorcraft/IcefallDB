use crate::catalog::Catalog;
use crate::storage::Storage;
use crate::wal::{LogEntry, LogEntryBody, WalReader};
use crate::{is_not_found, IcefallDBError, Result, Writer};
use arrow::ipc::reader::StreamReader;
use std::collections::BTreeMap;
use std::io::Cursor;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct RecoveryState {
    pub committed: Vec<String>,
    pub aborted: Vec<String>,
    pub undecided: Vec<String>,
}

pub async fn recover(storage: &dyn Storage) -> Result<RecoveryState> {
    let reader = WalReader::new(storage);
    let entries = reader.entries().await?;
    let mut by_tx: BTreeMap<String, Vec<&LogEntry>> = BTreeMap::new();
    for entry in &entries {
        by_tx
            .entry(entry.tx_id().to_string())
            .or_default()
            .push(entry);
    }

    let mut state = RecoveryState::default();
    for (tx_id, entries) in by_tx {
        let mut decided = false;
        for entry in &entries {
            match &entry.body {
                LogEntryBody::Commit { .. } => {
                    state.committed.push(tx_id.clone());
                    decided = true;
                    break;
                }
                LogEntryBody::Rollback { .. } => {
                    state.aborted.push(tx_id.clone());
                    decided = true;
                    break;
                }
                _ => {}
            }
        }
        if !decided {
            state.undecided.push(tx_id.clone());
        }
    }
    Ok(state)
}

/// Re-apply committed transactions whose effects are not yet visible in the
/// current manifests. This is idempotent because row groups are content-addressed.
pub async fn apply_committed_transactions(storage: Arc<dyn Storage>) -> Result<()> {
    let reader = WalReader::new(&*storage);
    let entries = match reader.entries().await {
        Ok(entries) => entries,
        Err(e) if is_not_found(&e) => return Ok(()),
        Err(e) => return Err(e),
    };
    let state = recover(&*storage).await?;

    let mut inserts_by_tx_table: BTreeMap<
        String,
        BTreeMap<String, Vec<arrow::record_batch::RecordBatch>>,
    > = BTreeMap::new();

    for tx_id in &state.committed {
        inserts_by_tx_table.insert(tx_id.clone(), BTreeMap::new());
    }

    for entry in &entries {
        if let LogEntryBody::Insert {
            tx_id,
            table,
            rows_ipc,
        } = &entry.body
        {
            if !state.committed.contains(tx_id) {
                continue;
            }
            let cursor = Cursor::new(rows_ipc);
            let stream_reader = StreamReader::try_new(cursor, None)
                .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
            let mut batches = Vec::new();
            for batch in stream_reader {
                batches.push(batch.map_err(|e| IcefallDBError::Other(Box::new(e)))?);
            }
            let tx_map = inserts_by_tx_table.get_mut(tx_id).unwrap();
            tx_map.entry(table.clone()).or_default().extend(batches);
        }
    }

    for (_tx_id, tables) in inserts_by_tx_table {
        for (table, batches) in tables {
            let catalog = Catalog::load(&*storage, &table).await?;
            let schema = catalog
                .latest_schema()
                .ok_or_else(|| IcefallDBError::SchemaNotFound {
                    path: format!("{}/_schema.json", table),
                })?;
            let mut writer = Writer::new(Arc::clone(&storage), &table, schema.clone()).await?;
            for batch in &batches {
                writer.insert_batch(batch.clone()).await?;
            }
            writer.commit().await?;
        }
    }

    Ok(())
}
