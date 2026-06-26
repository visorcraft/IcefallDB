//! Optional thin HTTP/1.1 client for the `--server` daemon path.
//!
//! Hand-rolled over `std::net::TcpStream` so the CLI gains no HTTP-client
//! dependency. It is used only when `--server <url>` is passed; the default
//! (no `--server`) standalone path never touches this module.

use std::io::{Read, Write};
use std::net::TcpStream;

use anyhow::{anyhow, bail, Context};

/// A minimal client for a running `icefalldb-server` daemon.
pub struct DaemonClient {
    host_port: String,
}

impl DaemonClient {
    /// Parse `http://host:port` into a connectable target.
    pub fn new(url: &str) -> anyhow::Result<Self> {
        let host_port = url
            .trim_end_matches('/')
            .strip_prefix("http://")
            .ok_or_else(|| anyhow!("--server URL must start with http:// (got {url})"))?
            .to_string();
        Ok(Self { host_port })
    }

    /// POST a JSON `body` to `path` and return the response body. Errors on a
    /// connection failure (daemon down) or a non-2xx status.
    fn post(&self, path: &str, body: &str) -> anyhow::Result<String> {
        let mut stream = TcpStream::connect(&self.host_port)
            .with_context(|| format!("connecting to daemon at {}", self.host_port))?;
        let req = format!(
            "POST {path} HTTP/1.1\r\nHost: {hp}\r\nContent-Type: application/json\r\n\
             Content-Length: {len}\r\nConnection: close\r\n\r\n{body}",
            hp = self.host_port,
            len = body.len(),
        );
        stream.write_all(req.as_bytes())?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw)?;
        let text = String::from_utf8_lossy(&raw);
        let (head, resp_body) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| anyhow!("malformed HTTP response from daemon"))?;
        let status_line = head.lines().next().unwrap_or("");
        let ok = status_line
            .split_whitespace()
            .nth(1)
            .is_some_and(|code| code.starts_with('2'));
        if !ok {
            bail!("daemon returned {status_line}: {resp_body}");
        }
        Ok(resp_body.to_string())
    }

    /// Route a SELECT to `/sql`; return the `data` rows as JSON.
    pub fn query(&self, sql: &str) -> anyhow::Result<String> {
        let resp = self.post("/sql", &serde_json::json!({ "sql": sql }).to_string())?;
        let v: serde_json::Value = serde_json::from_str(&resp)?;
        Ok(serde_json::to_string(v.get("data").unwrap_or(&v))?)
    }

    /// Route a DELETE/UPDATE/MERGE to `/mutate`; return the affected-row count.
    pub fn mutate(&self, sql: &str) -> anyhow::Result<u64> {
        let resp = self.post("/mutate", &serde_json::json!({ "sql": sql }).to_string())?;
        let v: serde_json::Value = serde_json::from_str(&resp)?;
        Ok(v.get("affected").and_then(|a| a.as_u64()).unwrap_or(0))
    }
}

/// True for SQL that mutates (routed to `/mutate` rather than `/sql`).
pub fn is_mutation(sql: &str) -> bool {
    let head = sql.trim_start().to_ascii_uppercase();
    head.starts_with("UPDATE") || head.starts_with("DELETE") || head.starts_with("MERGE")
}
