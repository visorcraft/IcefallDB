# Using IcefallDB from Python

The `icefalldb` Python package lets you read and change tables from your own
programs, and hand results straight to [pandas](https://pandas.pydata.org/) or
[Arrow](https://arrow.apache.org/). If you have not installed it, see
[Installation](installation.md).

## Connecting

```python
import icefalldb

con = icefalldb.attach("/tmp/mydb")
```

`attach` opens a database folder and returns a connection you run queries on. If
you point it at a single table's folder instead, it opens just that table.

`attach` takes several options. The ones you are most likely to use:

| Option | Default | What it does |
| --- | --- | --- |
| `engine` | `"icefalldb"` | Which engine runs your queries: `"icefalldb"` (smart router, recommended), `"duckdb"` (reads only), or `"datafusion"` (native engine). See [the engines](querying.md#the-engines). |
| `snapshot` | `None` | Open the table read-only as it was at snapshot number `N` (time travel). |
| `tables` | `None` (all) | A list of specific table names to open, instead of every table in the folder. |
| `result_cache_mb` | `1024` | Size of the on-disk result cache in MiB. `0` disables it. |
| `verify_data_checksums` | `True` | Check each data file's checksum when opening. Set `False` to open faster on data you trust. |

```python
con = icefalldb.attach("/tmp/mydb", engine="duckdb")          # read-only, fast
con = icefalldb.attach("/tmp/mydb", snapshot=3)               # the table at version 3
con = icefalldb.attach("/tmp/mydb", tables=["orders"])        # just one table
```

## Running queries and getting results

Call `.sql(...)` with a `SELECT` statement. Then turn the result into whatever
shape you want:

```python
result = con.sql("SELECT category, SUM(amount) AS revenue FROM orders GROUP BY category")

result.fetchall()   # list of tuples:  [('books', 24.0), ('games', 39.5), ...]
result.df()         # a pandas DataFrame
result.arrow()      # an Apache Arrow table (zero-copy, no conversion cost)
```

`.fetchall()`, `.df()`, and `.arrow()` work the same way no matter which engine
ran the query, so you can switch engines without changing how you read results.

## Changing rows

To delete, update, or merge rows, you have two equivalent options:

```python
# Option 1: run it like any other statement
con.sql("DELETE FROM orders WHERE status = 'cancelled'")

# Option 2: use .mutate(), which returns how many rows changed
n = con.mutate("UPDATE orders SET amount = amount * 1.1 WHERE category = 'books'")
print(f"updated {n} rows")
```

Use `.mutate()` when you want the count of affected rows. With the default
`"icefalldb"` engine, changes are sent to the native engine automatically and
your next `SELECT` sees them.

> Changing rows and time-travel reads from Python need the **native extension**
> from [Installation](installation.md). Plain `SELECT`s work without it.

## Time travel

List a table's snapshots, then open the one you want:

```python
import icefalldb

for snap in icefalldb.snapshots("/tmp/mydb", "orders"):
    print(snap["sequence"], snap["committed_at"], snap["rows"])

old = icefalldb.attach("/tmp/mydb", snapshot=1)
print(old.sql("SELECT COUNT(*) FROM orders").fetchall())
# old.mutate(...) would raise an error - historical reads are read-only
```

## Reading a whole table fast, without SQL

If you just want a table's data as an Arrow table (and do not need SQL),
`read_arrow_table` skips the query engine and reads the Parquet files directly.
It is the fastest way to pull a full table into Python:

```python
import icefalldb

table = icefalldb.read_arrow_table("/tmp/mydb", table="orders", columns=["order_id", "amount"])
df = table.to_pandas()
```

`columns=` is optional; omit it to read every column. (Because this reads the raw
files, it does not apply pending deletes the way a SQL query does - use it on
tables you have not edited, or run `optimize` first.)

## A complete example

```python
import icefalldb

con = icefalldb.attach("/tmp/mydb")

# Read
top = con.sql(
    "SELECT category, SUM(amount) AS revenue "
    "FROM orders WHERE status = 'paid' GROUP BY category ORDER BY revenue DESC"
).df()
print(top)

# Write
removed = con.mutate("DELETE FROM orders WHERE amount < 1.00")
print(f"removed {removed} tiny orders")

# The cache makes the identical read above effectively free the second time.
```

## When things go wrong

- `IcefallDBError` - the database or table path is wrong, or a file failed its
  integrity check.
- `ValueError` - an invalid option, such as `engine="duckdb"` together with
  `snapshot=` (DuckDB cannot time-travel).
- Trying to change rows on a snapshot-pinned (historical) connection raises an
  error, because the past is read-only.

## See also

- The concepts behind all of this: [Querying your data](querying.md).
- Calling IcefallDB from other languages: [other languages](languages.md).
