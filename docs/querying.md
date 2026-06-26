# Querying your data

This page explains how asking questions of an IcefallDB table actually works:
the query language, the engines that run your queries, how editing rows behaves,
reading old versions, and why the same query is almost free the second time. It
applies whether you use the command line, Python, or the HTTP server.

## Queries are SQL

IcefallDB speaks **SQL**, the same query language used by virtually every
database. If you have written SQL before, it works as you expect:

```sql
SELECT category, SUM(amount) AS revenue
FROM orders
WHERE status = 'paid'
GROUP BY category
ORDER BY revenue DESC
```

`SELECT` reads data. `WHERE` filters rows. `GROUP BY` collapses rows into groups
and `SUM`, `COUNT`, `AVG`, `MIN`, `MAX` summarise them (these summaries are called
**aggregates**). Joins across tables work too.

## The engines

A query has to be *run* by something. IcefallDB can use two engines over the
exact same files, and a smart router that picks between them. You choose with the
`engine` setting in Python (the command-line `query` always uses the native
engine).

- **`icefalldb` (the router, and the default in Python).** Recommended. For each
  statement it picks the faster path automatically: plain `SELECT`s go to DuckDB,
  which is very fast at scanning columns, while edits and certain whole-table
  summaries go to the native engine. You do not have to think about it.
- **`duckdb`.** Uses [DuckDB](https://duckdb.org/) only. Great for read-only
  analysis on tables you are not editing. It cannot make edits, cannot read old
  snapshots, and cannot read encrypted tables.
- **`datafusion` (the native engine).** IcefallDB's own engine, built on
  [Apache DataFusion](https://datafusion.apache.org/). It is the one that knows
  about pending edits, the aggregate cache, time travel, and encryption. The
  command-line tool uses this engine.

The good news: with the default router you usually do not need to choose. The
options exist for when you want to force one path.

## Changing rows: mutations

Editing data is called a **mutation**. IcefallDB supports three:

- **`DELETE`** removes rows that match a condition.
- **`UPDATE`** changes column values in rows that match a condition.
- **`MERGE`** combines insert-and-update in one statement (often called "upsert"):
  for each incoming row, update it if the key already exists, otherwise insert it.
  `MERGE` requires a **unique index** on the key column first (see
  [`create-index`](cli.md)).

```sql
DELETE FROM orders WHERE status = 'cancelled';
UPDATE orders SET amount = amount * 1.1 WHERE category = 'books';
```

Two things are worth knowing about how this behaves:

1. **Your Parquet files are never rewritten in place.** A `DELETE` does not go
   back and edit the data file. IcefallDB records which rows are gone in a small
   side file and skips them when reading. Edits are therefore cheap and your
   original files stay intact. Later, `optimize` folds the changes in and tidies
   up.
2. **Every mutation is atomic.** It either fully happens or does not happen at
   all, even if the machine loses power mid-write. You never end up with a
   half-applied change.

Each mutation targets one table at a time.

## Reading old versions: snapshots and time travel

IcefallDB keeps older versions of a table as numbered, frozen **snapshots**, so
you can look at the table as it was at an earlier point. Snapshots are retained
until you remove them with `gc`/`optimize`. (Edits are committed quickly and
folded into new numbered snapshots as the table is maintained, so the snapshot
list grows over time rather than on every single edit.)

List them:

```bash
icefalldb snapshots /tmp/mydb orders
```

Read one by adding its number:

```bash
icefalldb query /tmp/mydb/orders "SELECT COUNT(*) FROM orders" --snapshot 3
```

In Python, pass `snapshot=` when you connect (see [Python](python.md)). Reading an
old snapshot is **read-only** - it shows you the past without changing the
present. This is useful for audits ("what did this table say last Tuesday?"),
debugging, and reproducible reports.

## Why the second run is almost instant

IcefallDB is built for the kind of analytical queries you run again and again.
Two things make repeats fast:

- **The result cache.** When a `SELECT` finishes, IcefallDB saves its result on
  disk (under `_query_cache` inside the database). Run the same query again and
  it hands back the saved result, skipping the work entirely. The moment you
  change any row in a table the query touches, its cached results stop being used
  automatically - so you never see stale answers. You can change the cache size
  or turn it off (`--result-cache-mb` on the CLI, `result_cache_mb=` in Python).
- **Aggregates from metadata.** For whole-table summaries like `COUNT(*)`,
  `MIN`, `MAX`, `SUM`, and `AVG`, IcefallDB keeps small running totals next to
  each chunk of data. It can answer these by adding up the totals instead of
  scanning the rows, so they come back in well under a millisecond - and the
  answers are exact, not estimates.

The one query the cache cannot speed up is one that returns a huge number of
rows: there, the time is spent handing you the result, not computing it.

## Next steps

- Run queries from the terminal: [The command-line tool](cli.md).
- Run them from a program: [Python](python.md) or
  [other languages](languages.md).
