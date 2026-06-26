use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use icefalldb_core::storage::{LockGuard, Storage};
use icefalldb_core::wal::{LogEntryBody, Wal};
use icefalldb_core::{IcefallDBError, Result, Writer, WriterOptions};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use uuid::Uuid;

pub struct Transaction {
    pub tx_id: String,
    /// Buffered inserts per table.
    pub inserts: BTreeMap<String, Vec<RecordBatch>>,
}

impl Transaction {
    pub fn new(tx_id: String) -> Self {
        Self {
            tx_id,
            inserts: BTreeMap::new(),
        }
    }

    pub fn add_batch(&mut self, table: &str, batch: RecordBatch) {
        self.inserts
            .entry(table.to_string())
            .or_default()
            .push(batch);
    }
}

pub struct TransactionManager {
    wal: Arc<Wal>,
    storage: Arc<dyn Storage>,
    /// Active transactions.
    transactions: Mutex<BTreeMap<String, Transaction>>,
}

impl TransactionManager {
    pub fn new(wal: Arc<Wal>, storage: Arc<dyn Storage>) -> Self {
        Self {
            wal,
            storage,
            transactions: Mutex::new(BTreeMap::new()),
        }
    }

    pub async fn begin(&self) -> Result<String> {
        let tx_id = Uuid::new_v4().to_string();
        self.wal
            .append(LogEntryBody::Begin {
                tx_id: tx_id.clone(),
            })
            .await?;
        self.transactions
            .lock()
            .await
            .insert(tx_id.clone(), Transaction::new(tx_id.clone()));
        Ok(tx_id)
    }

    pub async fn add_insert(&self, tx_id: &str, table: &str, batch: RecordBatch) -> Result<()> {
        // Serialize the batch to Arrow IPC for the WAL before buffering it.
        let rows_ipc = record_batch_to_ipc(&batch)?;
        self.wal
            .append(LogEntryBody::Insert {
                tx_id: tx_id.to_string(),
                table: table.to_string(),
                rows_ipc,
            })
            .await?;

        let mut txs = self.transactions.lock().await;
        let tx = txs
            .get_mut(tx_id)
            .ok_or_else(|| IcefallDBError::Other("transaction not found".into()))?;
        tx.add_batch(table, batch);
        Ok(())
    }

    pub async fn commit(&self, tx_id: &str) -> Result<Vec<(String, icefalldb_core::CommitDelta)>> {
        // Remove the transaction from active set before doing I/O.
        let tx = {
            let mut txs = self.transactions.lock().await;
            txs.remove(tx_id)
                .ok_or_else(|| IcefallDBError::Other("transaction not found".into()))?
        };

        // Acquire table writer locks in deterministic order.
        let tables: Vec<String> = tx.inserts.keys().cloned().collect();
        let mut guards: Vec<Box<dyn LockGuard>> = Vec::new();
        for table in &tables {
            let guard = self
                .storage
                .lock_exclusive(&format!("{}/_write.lock", table), Duration::from_secs(30))
                .await?;
            guards.push(guard);
        }

        // Write all row groups for all tables. Because we already hold the
        // writer locks, tell the Writer not to acquire them again.
        let writer_options = WriterOptions {
            lock_timeout: Duration::from_secs(30),
            assume_lock_held: true,
        };
        let mut deltas = Vec::new();
        for (table, batches) in &tx.inserts {
            let catalog = icefalldb_core::catalog::Catalog::load(&*self.storage, table).await?;
            let schema = catalog
                .latest_schema()
                .ok_or_else(|| IcefallDBError::SchemaNotFound {
                    path: format!("{}/_schema.json", table),
                })?;
            let mut writer = Writer::new_with_options(
                Arc::clone(&self.storage),
                table,
                schema.clone(),
                writer_options,
            )
            .await?;
            for batch in batches {
                writer.insert_batch(batch.clone()).await?;
            }
            let delta = writer.commit().await?;
            deltas.push((table.clone(), delta));
        }

        // After all data and manifests are durable, record the commit in the WAL.
        self.wal
            .append(LogEntryBody::Commit {
                tx_id: tx_id.to_string(),
            })
            .await?;

        // Locks are released when guards drop here.
        Ok(deltas)
    }

    pub async fn rollback(&self, tx_id: &str) -> Result<()> {
        {
            let mut txs = self.transactions.lock().await;
            if txs.remove(tx_id).is_none() {
                return Err(IcefallDBError::Other("transaction not found".into()));
            }
        }
        self.wal
            .append(LogEntryBody::Rollback {
                tx_id: tx_id.to_string(),
            })
            .await?;
        Ok(())
    }
}

fn record_batch_to_ipc(batch: &RecordBatch) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut buf, batch.schema().as_ref())
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
        writer
            .write(batch)
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
        writer
            .finish()
            .map_err(|e| IcefallDBError::Other(Box::new(e)))?;
    }
    Ok(buf)
}
