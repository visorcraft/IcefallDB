# Contributing to IcefallDB

Thanks for helping improve IcefallDB. This document covers how to propose a
change, what a good pull request looks like, and the checks your change has to
pass.

If anything here is unclear or out of date, open an issue or a pull request.

## Code of conduct

Be kind, be specific, assume good faith. Disagree about the technical details,
not the person. Reviews stay focused on the change.

## How to propose a change

IcefallDB uses a standard **fork, branch, pull request** workflow on GitHub.

1. **Fork** [`visorcraft/IcefallDB`](https://github.com/visorcraft/IcefallDB) to
   your account.
2. **Clone** your fork and add the upstream remote:

   ```sh
   git clone git@github.com:<you>/IcefallDB.git
   cd IcefallDB
   git remote add upstream https://github.com/visorcraft/IcefallDB.git
   ```

3. **Branch** from `master` with a short, descriptive, kebab-case name, for
   example `fix-merge-unique-check` or `docs/python-examples`:

   ```sh
   git fetch upstream
   git switch -c my-change upstream/master
   ```

4. **Make focused commits** - one logical change per commit. Run the checks
   below before pushing.
5. **Open a pull request** against `master`. In the description, cover:
   - **What.** A one-paragraph summary of the change.
   - **Why.** Bug fix, new feature, or docs? Link the issue if there is one.
   - **How to test.** The exact commands a reviewer should run.
   - **Risk.** What might break, and what you did not test.

## Before you push: the checks

Your change must pass the same gates that CI runs. From the repository root:

```sh
# Rust
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

If you touched the Python adapter (virtual environment at `python/.venv`):

```sh
python/.venv/bin/ruff check python
python/.venv/bin/ruff format --check python
python/.venv/bin/python -m pytest python/tests -q
```

If your Rust change affects the Python extension, rebuild it first so the Python
tests exercise your code:

```sh
VIRTUAL_ENV=$(pwd)/python/.venv python/.venv/bin/maturin develop \
  -m crates/icefalldb-query-py/Cargo.toml
```

If a check fails, fix the cause rather than working around it. Do not silence a
clippy lint with `#[allow(...)]` unless you explain why in the pull request.

## What we look for in review

- The change does one thing and does it well.
- New behavior ships with tests, placed next to the code (a unit test in `src/`,
  or a file under the crate's `tests/`).
- `icefalldb-core` stays the source of truth for storage and metadata; all
  filesystem access goes through its async `Storage` trait.
- The on-disk format invariants are respected. The canonical JSON and Parquet
  files are authoritative; derived `.idx` / `.model` / `.agg` / checkpoint files
  are optional accelerators and must never become the only copy of the truth.
- The change keeps tables inspectable: plain Parquet plus plain JSON, no opaque
  binary state.

The conventions and format invariants are documented in
[`AGENTS.md`](AGENTS.md); skim it before a larger change. End-user docs live in
[`docs/`](docs/).

## Licensing of contributions

IcefallDB is dual-licensed under [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE). By submitting a contribution, you agree that it may
be distributed under both licenses, with no additional terms.
