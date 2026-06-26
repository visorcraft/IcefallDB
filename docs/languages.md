# Using IcefallDB from other languages

IcefallDB is written in Rust and ships a Python package, but you are not limited
to those two. There are three ways to reach your data from any language:

1. The **Python package** - see [Using IcefallDB from Python](python.md).
2. The **HTTP server** - run a small server and talk to it over the web, from
   any language that can make an HTTP request.
3. **Reading the files directly** - because tables are plain Parquet, almost any
   data tool can open them.

## From Rust

IcefallDB's core *is* a Rust library, so a Rust program can embed it directly
rather than going through a separate process. The two crates are
`icefalldb-core` (the storage, writer, and reader) and `icefalldb-query` (the SQL
engine). The command-line tool and the server are both thin wrappers over these,
so their source is the best worked example of the API. (The crates are not
published to crates.io yet, so depend on them by path or Git for now.)

## Over HTTP, from any language

The HTTP server opens a database once and answers queries over a simple JSON API.
This is the way to use IcefallDB from JavaScript, Go, Ruby, Java, or anything
else.

### Start the server

Build it (once) and run it, pointing at your database folder:

```bash
cargo build --release -p icefalldb-server
./target/release/icefalldb-server --db /tmp/mydb
```

It prints `icefalldb-server listening on http://127.0.0.1:8080`. Options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--db <path>` | (required) | The database folder to serve. |
| `--port <n>` | `8080` | Port to listen on. |
| `--host <addr>` | `127.0.0.1` | Address to bind. Use `0.0.0.0` to accept connections from other machines. |
| `--result-cache-mb <n>` | `1024` | Result-cache size in MiB; `0` disables it. |

### The endpoints

| Method & path | Body you send | What you get back |
| --- | --- | --- |
| `POST /sql` | `{"sql": "SELECT ...", "snapshot": N}` (`snapshot` optional) | `{"data": [ {row}, {row}, ... ]}` - one object per row |
| `POST /mutate` | `{"sql": "DELETE/UPDATE/MERGE ..."}` | `{"affected": N}` - number of rows changed |
| `GET /tables` | (nothing) | `{"tables": ["orders", ...]}` |

Errors come back as `{"error": "message"}` with an HTTP status of `400`
(bad request), `404` (e.g. snapshot not found), or `500` (server error).

There is also a small set of transaction endpoints (`/tx/begin`, `/tx/sql`,
`/tx/commit`, `/tx/rollback`) for inserting many rows across tables in one atomic
batch. Use them when you need several inserts to commit together.

### Talk to it with curl

```bash
# Query
curl -s http://127.0.0.1:8080/sql \
  -H 'Content-Type: application/json' \
  -d '{"sql": "SELECT category, SUM(amount) AS revenue FROM orders GROUP BY category"}'
# -> {"data":[{"category":"books","revenue":24.0}, ...]}

# Query an old version
curl -s http://127.0.0.1:8080/sql \
  -H 'Content-Type: application/json' \
  -d '{"sql": "SELECT COUNT(*) FROM orders", "snapshot": 1}'

# Change rows
curl -s http://127.0.0.1:8080/mutate \
  -H 'Content-Type: application/json' \
  -d '{"sql": "DELETE FROM orders WHERE status = '\''cancelled'\''"}'
# -> {"affected":1}
```

### From JavaScript

```javascript
const res = await fetch("http://127.0.0.1:8080/sql", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ sql: "SELECT * FROM orders WHERE amount > 10" }),
});
const { data } = await res.json();
console.log(data);   // array of row objects
```

### From Python (without the package)

```python
import requests

r = requests.post("http://127.0.0.1:8080/sql",
                  json={"sql": "SELECT * FROM orders"})
print(r.json()["data"])
```

### Security note

The server has **no built-in authentication or HTTPS**. On a trusted local
machine that is fine. To expose it to a network, put it behind a reverse proxy
(such as nginx or Caddy) that adds TLS and access control, and bind the server
itself to `127.0.0.1`.

## Reading the files directly

Because each table is just Parquet files in a folder, any tool that reads
Parquet can read your data with no IcefallDB code at all -
[pandas](https://pandas.pydata.org/), [Polars](https://pola.rs/), DuckDB, Apache
Spark, and many others:

```python
import pandas as pd
df = pd.read_parquet("/tmp/mydb/orders")   # reads the Parquet files in the folder
```

```sql
-- DuckDB, reading the files straight off disk
SELECT * FROM read_parquet('/tmp/mydb/orders/*.parquet');
```

This is great for quick exploration and for plugging IcefallDB tables into
existing pipelines. **One caveat:** reading the raw files bypasses IcefallDB's
own bookkeeping, so it does *not* hide rows you have deleted but not yet
compacted, and it cannot read [encrypted](encryption.md) tables. If a table has
recent edits, either query it through IcefallDB (which applies them correctly) or
run `icefalldb optimize` first to fold the edits into the files.
