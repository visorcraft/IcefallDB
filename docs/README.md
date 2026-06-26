# IcefallDB documentation

Welcome. This is the user guide for **IcefallDB** - a system for storing and
querying tables of data that live on disk as ordinary files you can open and
read yourself.

If you have never used IcefallDB before, read the pages in this order. Each one
builds on the last.

## Start here

1. **[Getting started](getting-started.md)** - install it, create your first
   table, put some data in, and run a query. About ten minutes, start to finish.
2. **[Installation](installation.md)** - the full setup details: the
   command-line tool, the Python package, and the optional speed-up extension.

## Everyday use

3. **[Querying your data](querying.md)** - how queries actually work: writing
   `SELECT`s, changing rows with `DELETE`/`UPDATE`/`MERGE`, reading old versions
   of a table, and why repeated queries come back almost instantly.
4. **[The command-line tool](cli.md)** - a reference for every `icefalldb`
   command: creating tables, loading data, querying, and maintenance.
5. **[Using IcefallDB from Python](python.md)** - the Python package in depth:
   connecting, running queries, getting results into pandas or Arrow, and making
   changes.

## Going further

6. **[Using IcefallDB from other languages](languages.md)** - Python, Rust, and
   the HTTP server that lets *any* language (JavaScript, Go, anything that can
   make a web request) talk to IcefallDB. Plus reading the raw files directly.
7. **[Encryption](encryption.md)** - locking individual tables (or even
   individual columns) so the data on disk is unreadable without a key.

## What is IcefallDB, in one paragraph?

Each IcefallDB table is just a folder. Inside that folder are
[Parquet](https://parquet.apache.org/) files (a compact, column-oriented file
format that analytics tools everywhere already understand) holding the data, and
small plain-text JSON files describing it. There is no hidden binary database
file - you can look at a table with everyday tools like `ls`, `cat`, and `git`.
On top of those files, IcefallDB runs a real SQL engine, so you get the
convenience of a database (queries, row edits, history) without giving up the
openness of plain files.

> **A note on names:** the project, the command-line program, and the Python
> package are all called **`icefalldb`** (lower-case when you type it). When you
> see `IcefallDB` with capitals in this guide, that is just the product name in
> prose.
