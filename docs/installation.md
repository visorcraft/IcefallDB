# Installation

IcefallDB is built from source. There are two pieces, and you only need the ones
you plan to use:

- The **command-line tool** (`icefalldb`) - a single program you run in a
  terminal. This is all you need to create tables, load data, and run queries.
- The **Python package** (`icefalldb`) - for using IcefallDB inside Python
  programs.

There is no installer or package-manager download yet (it is not on crates.io or
PyPI). You clone the repository and build it. That sounds heavier than it is -
it is a couple of commands.

## Prerequisites

| You want to use... | You need |
| --- | --- |
| The command-line tool | [Rust](https://www.rust-lang.org/tools/install) 1.80 or newer |
| The Python package | Python 3.9 or newer (plus Rust, if you build the optional extension) |

Rust ships with `cargo`, the build tool used below. On Linux/macOS the official
one-line installer at the link above is the usual way to get it.

## Part 1: the command-line tool

```bash
git clone https://github.com/visorcraft/IcefallDB.git
cd IcefallDB
cargo build --release -p icefalldb-cli
```

The first build downloads and compiles dependencies, so it takes a few minutes.
When it finishes, the program is at `target/release/icefalldb`. Add that folder
to your `PATH` so you can type `icefalldb` from anywhere:

```bash
export PATH="$PWD/target/release:$PATH"
```

(That line lasts for the current terminal session. To make it permanent, add it
to your shell's startup file, e.g. `~/.bashrc` or `~/.config/fish/config.fish`.)

Check it worked:

```bash
icefalldb --help
```

You should see the list of commands. (There is no `--version` flag; `--help` is
the quick smoke test.)

### Optional: the HTTP server

If you want to query IcefallDB over the network from other languages, also build
the server (covered in [Using IcefallDB from other languages](languages.md)):

```bash
cargo build --release -p icefalldb-server
```

## Part 2: the Python package

First make a **virtual environment** - a private, self-contained Python setup so
IcefallDB's dependencies do not mix with the rest of your system:

```bash
cd IcefallDB
python3 -m venv python/.venv
source python/.venv/bin/activate
```

Then install the package. This pulls in [DuckDB](https://duckdb.org/), the
engine IcefallDB uses to run fast read queries from Python:

```bash
pip install -e python/
```

For working with results as Arrow tables or pandas DataFrames, also install the
extras:

```bash
pip install -e "python/[dev]"
```

Check it worked:

```bash
python3 -c "import icefalldb; print('ok')"
```

### Optional but recommended: the native extension

The steps above are enough to **read** tables from Python. To **change** rows
(`DELETE`/`UPDATE`/`MERGE`) and to do **time-travel** reads from Python, build
the native extension. It compiles IcefallDB's Rust engine into something Python
can call directly:

```bash
pip install maturin
VIRTUAL_ENV=$(pwd)/python/.venv python/.venv/bin/maturin develop \
  -m crates/icefalldb-query-py/Cargo.toml
```

Check it is available:

```bash
python3 -c "import icefalldb_query_py; print('native engine ready')"
```

If that import fails, reading still works; only the write and time-travel
features need it.

## A note on DuckDB

The Python package uses DuckDB for its fast read path, and `pip` installs it for
you automatically - you do not need to install DuckDB separately. (One command,
`refresh-view`, also looks for the standalone DuckDB command-line program on your
`PATH`, but you only need that if you use materialized views.)

## You are set up

Jump to [Getting started](getting-started.md) to create your first table, or to
[The command-line tool](cli.md) for the full command reference.
