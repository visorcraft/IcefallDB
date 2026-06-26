# The command-line tool

`icefalldb` is the program you run in a terminal to work with tables. This page
lists every command. If you have not built it yet, see
[Installation](installation.md).

## How the arguments are shaped

Almost every command starts with two things:

- a **database** path - the folder that holds your tables (e.g. `/tmp/mydb`)
- a **table** name - a table inside that database (e.g. `orders`)

So most commands read as `icefalldb <command> <database> <table> ...`.

The one exception is `query`, which takes a path to a **single table's folder**
(e.g. `/tmp/mydb/orders`) instead of the database plus a table name. This is
explained under [Querying](#querying) below.

---

## Creating tables and loading data

### `import` - load a TSV, creating the table

```bash
icefalldb import <database> <table> <file.tsv>
```

Loads a tab-separated file. The table is created automatically and its column
types are inferred from the data. The quickest way to get data in.

```bash
icefalldb import /tmp/mydb orders orders.tsv
```

### `insert` - add Parquet or Arrow data to an existing table

```bash
icefalldb insert <database> <table> <file.parquet | file.arrow>
```

Appends rows from a Parquet or Arrow file. The table must already exist and its
columns must match the file. If the source is Parquet with a matching layout,
IcefallDB copies it without re-encoding (fast).

### `create` - make an empty table

```bash
icefalldb create <database> <table> [--schema <schema.json>]
```

Creates an empty table. With `--schema`, you supply a JSON file describing the
columns; without it, you get a single `id` column to start from. Use this when
you want to define the columns yourself before loading data with `insert`.

### `create-table` / `drop-table` - manage tables in the catalog

```bash
icefalldb create-table <database> <table> [--schema <schema.json>]
icefalldb drop-table   <database> <table>
```

Like `create`, but also registers (or removes) the table in the database's
central catalog, which the HTTP server reads at startup.

### `export` - write a table out to TSV

```bash
icefalldb export <database> <table> <file.tsv>
```

The reverse of `import`: dumps a table to a tab-separated file.

---

## Querying

### `query` - run SQL

```bash
icefalldb query <table-folder> "<SQL>" [options]
```

Runs a SQL statement. The first argument is the path to the table's folder. The
SQL goes in quotes.

```bash
icefalldb query /tmp/mydb/orders "SELECT * FROM orders WHERE amount > 10"
```

**To query more than one table** (for example, to join them), point the first
argument at the **database** folder and name the extra tables with `-t`:

```bash
icefalldb query /tmp/mydb "SELECT * FROM orders o JOIN customers c ON o.cust_id = c.id" \
  -t orders -t customers
```

Options:

| Option | What it does |
| --- | --- |
| `--format json` \| `csv` | Output format for results. Default is `json`. |
| `--snapshot <N>` | Read the table as it was at snapshot number `N` (time travel). Read-only. |
| `--server <URL>` | Send the query to a running server instead of opening the files directly (see [languages](languages.md)). Cannot be combined with `--snapshot`. |
| `--result-cache-mb <N>` | Size limit for the on-disk result cache, in MiB. `0` turns it off. Default `1024`. |

The same `query` command also runs the data-changing statements below.

### `snapshots` - list a table's history

```bash
icefalldb snapshots <database> <table>
```

Shows every retained version of the table: its sequence number, commit time, row
count, and more. Use the sequence numbers with `query --snapshot`.

---

## Changing data

`DELETE`, `UPDATE`, and `MERGE` are written as SQL and run through `query`,
against a single table:

```bash
icefalldb query /tmp/mydb/orders "DELETE FROM orders WHERE status = 'cancelled'"
icefalldb query /tmp/mydb/orders "UPDATE orders SET amount = amount * 1.1 WHERE category = 'books'"
```

`MERGE` (insert-or-update, sometimes called "upsert") needs a **unique index**
on the key column first - see `create-index` below.

### `create-index` - speed up lookups and enable MERGE

```bash
icefalldb create-index <database> <table> <column> [--unique] [--name <name>] [--index-type btree]
```

Builds an index on a column. An index makes "find the row where this column
equals X" fast. Add `--unique` if each value appears at most once; a unique index
is required to use that column as a `MERGE` key.

```bash
icefalldb create-index /tmp/mydb orders order_id --unique
```

---

## Maintenance

Over time, edits leave a table with extra files and old versions. These commands
tidy up. They are optional - the table works without them - but they keep things
fast and small.

### `optimize` - the all-in-one cleanup

```bash
icefalldb optimize <database> <table> [--retain-snapshots <N>] [--sort <column>]
```

Rewrites the table compactly (good compression), folds in pending edits, and
removes old versions. `--retain-snapshots N` keeps the `N` most recent versions
(default `1`). `--sort column` physically orders the rows by a column, which can
make range queries faster; repeat `--sort` for multiple columns.

```bash
icefalldb optimize /tmp/mydb orders --retain-snapshots 2 --sort order_id
```

### `compact` and `gc` - the individual pieces

```bash
icefalldb compact <database> <table>                          # merge small files together
icefalldb gc      <database> <table> [--retain-snapshots <N>] # delete old, unreferenced versions
```

`optimize` does both of these together; use them separately if you want finer
control.

### `check` and `doctor` - verify and repair

```bash
icefalldb check  <database> <table>            # report integrity problems
icefalldb doctor <database> <table> [--repair] # diagnose, and fix with --repair
```

`check` reports any problems it finds and leaves the table untouched. `doctor`
diagnoses, and with `--repair` actually fixes what it safely can.

---

## Other commands

### `create-view` / `refresh-view` - saved queries

```bash
icefalldb create-view  <database> <view> <query.sql>   # define a view from a SELECT in a file
icefalldb refresh-view <database> <view>               # (re)compute it
```

A **view** is a saved query that you can treat like a table. `create-view` stores
the definition; `refresh-view` runs it and saves the result. (`refresh-view`
needs the standalone DuckDB program on your `PATH`.)

### `iceberg-export` - hand off to Apache Iceberg

```bash
icefalldb iceberg-export <database> <table> <output-folder> [--snapshot <N>]
```

Writes Iceberg metadata for the table so other tools in the
[Apache Iceberg](https://iceberg.apache.org/) ecosystem can read it. This is a
one-way export; IcefallDB's own files remain the source of truth.

---

To understand what these commands are doing underneath - snapshots, mutations,
the engines, the cache - read [Querying your data](querying.md).
