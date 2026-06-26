use crate::storage::Storage;
use crate::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum LogEntryBody {
    Begin {
        tx_id: String,
    },
    Insert {
        tx_id: String,
        table: String,
        /// Arrow IPC bytes for the rows being inserted.
        rows_ipc: Vec<u8>,
    },
    Commit {
        tx_id: String,
    },
    Rollback {
        tx_id: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub lsn: u64,
    #[serde(flatten)]
    pub body: LogEntryBody,
}

impl LogEntry {
    pub fn tx_id(&self) -> &str {
        match &self.body {
            LogEntryBody::Begin { tx_id, .. }
            | LogEntryBody::Insert { tx_id, .. }
            | LogEntryBody::Commit { tx_id, .. }
            | LogEntryBody::Rollback { tx_id, .. } => tx_id,
        }
    }
}

const WAL_DIR: &str = "_wal";
const SEGMENT_SIZE: u64 = 16 * 1024 * 1024;

struct WalState {
    next_lsn: u64,
}

pub struct Wal {
    storage: Arc<dyn Storage>,
    state: Mutex<WalState>,
}

impl Wal {
    pub async fn open(storage: Arc<dyn Storage>) -> Result<Self> {
        // Determine the next LSN from existing segments.
        let mut next_lsn = 0u64;
        let reader = WalReader::new(&*storage);
        if let Ok(entries) = reader.entries().await {
            for entry in entries {
                if entry.lsn >= next_lsn {
                    next_lsn = entry.lsn + 1;
                }
            }
        }
        Ok(Self {
            storage,
            state: Mutex::new(WalState { next_lsn }),
        })
    }

    /// Append a body to the WAL and return the assigned log entry.
    pub async fn append(&self, body: LogEntryBody) -> Result<LogEntry> {
        let mut state = self.state.lock().await;
        let lsn = state.next_lsn;
        state.next_lsn += 1;

        let segment = segment_for_lsn(lsn);
        let path = format!("{}/{:020}.log", WAL_DIR, segment);
        let entry = LogEntry { lsn, body };

        let mut line = serde_json::to_vec(&entry)?;
        line.push(b'\n');
        self.storage.append(&path, &line).await?;
        self.storage.sync_data(&path).await?;
        Ok(entry)
    }

    pub async fn next_lsn(&self) -> u64 {
        self.state.lock().await.next_lsn
    }
}

fn segment_for_lsn(lsn: u64) -> u64 {
    lsn / SEGMENT_SIZE
}

pub struct WalReader<'a> {
    storage: &'a dyn Storage,
}

impl<'a> WalReader<'a> {
    pub fn new(storage: &'a dyn Storage) -> Self {
        Self { storage }
    }

    pub async fn entries(&self) -> Result<Vec<LogEntry>> {
        let mut entries = Vec::new();
        let mut segment_names: Vec<String> = self
            .storage
            .list(WAL_DIR)
            .await?
            .into_iter()
            .filter(|n| n.ends_with(".log"))
            .collect();
        segment_names.sort();

        for name in segment_names {
            let data = self.storage.read(&name).await?;
            for line in data.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                let entry: LogEntry = serde_json::from_slice(line)?;
                entries.push(entry);
            }
        }
        entries.sort_by_key(|e| e.lsn);
        Ok(entries)
    }
}
