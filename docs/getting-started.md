# Getting started

This page takes you from nothing to a working table you can query, in a few
minutes. We will use the command-line tool the whole way through. By the end you
will have created a table, loaded data into it, queried it, changed some rows,
and looked at the files on disk.

If the `icefalldb` command is not yet installed, do the short build in
[Installation](installation.md) first (it is two commands), then come back here.

## Step 1: make a little data file

IcefallDB can load data from a **TSV** file - a plain text file where each row is
a line and the columns are separated by tabs. The first line holds the column
names.

Create a file called `orders.tsv`. The spaces between the values below must be
single **tab** characters, not spaces:

```text
order_id	category	amount	status
1	books	9.99	paid
2	games	39.50	paid
3	books	14.00	cancelled
4	music	5.25	paid
```

You do not have to describe the column types in advance. IcefallDB looks at the
data and figures them out for you (here: `order_id` is a whole number, `amount`
is a decimal, the rest are text).

## Step 2: load it into a table

A **database** in IcefallDB is just a folder that holds one or more tables. Pick
any path; IcefallDB creates the folder if it does not exist.

```bash
icefalldb import /tmp/mydb orders orders.tsv
```

That command means: *into the database at `/tmp/mydb`, create a table named
`orders`, and fill it from `orders.tsv`.* The `import` command creates the table
and infers its columns in one step.

## Step 3: look at what was created

This is the part that makes IcefallDB different from most databases - there is
nothing hidden. Have a look:

```bash
ls /tmp/mydb/orders
```

Among the files you will see are a Parquet file holding your four rows, a
`_schema.json` describing the columns, and a `_manifest.json` (the table's
table-of-contents), alongside a handful of small bookkeeping files. They are all
ordinary files - you can copy the folder, check it into Git, or inspect it with
other tools.

## Step 4: query it

Queries are written in **SQL**, the standard language for asking questions of
tabular data. Use the `query` command. The first argument points at the table
folder, and the second is the SQL, in quotes:

```bash
icefalldb query /tmp/mydb/orders "SELECT * FROM orders"
```

(A `SELECT *` also shows two helper columns, `_rowid` and `_rowaddr`, that
IcefallDB uses internally to keep track of rows across edits. You can ignore
them, or just name the columns you want instead of `*`.)

Try a real question - total revenue per category, counting only paid orders:

```bash
icefalldb query /tmp/mydb/orders \
  "SELECT category, SUM(amount) AS revenue, COUNT(*) AS n
   FROM orders
   WHERE status = 'paid'
   GROUP BY category"
```

Results come back as JSON by default. If you would rather have a spreadsheet-style
table, add `--format csv`.

## Step 5: change some rows

Most plain-file data sets are read-only - to fix one wrong row you would rewrite
the whole file. IcefallDB lets you change rows in place with the same `query`
command. Editing rows is called a **mutation**.

```bash
# Remove the cancelled order
icefalldb query /tmp/mydb/orders "DELETE FROM orders WHERE status = 'cancelled'"

# Give every books order a 10% bump
icefalldb query /tmp/mydb/orders \
  "UPDATE orders SET amount = amount * 1.1 WHERE category = 'books'"
```

Run the `SELECT * FROM orders` query again and you will see the changes. Behind
the scenes IcefallDB did **not** rewrite your Parquet file; it recorded the
changes alongside it and applies them when you read. The old version still
exists - which leads to the next step.

## Step 6: see the table's history

IcefallDB keeps older versions of a table as numbered **snapshots**, so you can
look at the table as it was in the past. List the ones it currently has:

```bash
icefalldb snapshots /tmp/mydb orders
```

Each line shows a sequence number, when it was committed, and how many rows that
version held. Read the table as it looked at a given number by adding
`--snapshot N`. Snapshot 1 is the table right after you imported it, before the
delete and update:

```bash
icefalldb query /tmp/mydb/orders "SELECT COUNT(*) FROM orders" --snapshot 1
```

That returns the original count, while a plain query (without `--snapshot`) shows
the current, edited table. Reading an old snapshot is **time travel**: looking at
the past without un-doing your later changes, and always read-only.

(IcefallDB commits edits quickly and folds them into new numbered snapshots as a
table is maintained, so the list grows over time rather than on every single
edit.)

## Step 7: do the same from Python (optional)

If you installed the Python package, the same table is one import away:

```python
import icefalldb

con = icefalldb.attach("/tmp/mydb")          # open the database
rows = con.sql("SELECT category, SUM(amount) FROM orders GROUP BY category").fetchall()
print(rows)

con.mutate("DELETE FROM orders WHERE amount < 1.00")   # change rows from Python too
```

See [Using IcefallDB from Python](python.md) for the full story.

## Where to go next

- You created and queried a table. To understand *how* queries and edits behave
  (and why the second time you run a query it is almost instant), read
  **[Querying your data](querying.md)**.
- For every command and option, see **[The command-line tool](cli.md)**.
- To call IcefallDB from a program, see **[Python](python.md)** or
  **[other languages](languages.md)**.
