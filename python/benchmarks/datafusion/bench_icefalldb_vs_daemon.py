#!/usr/bin/env python3
"""IcefallDB-vs-daemon query benchmark: the two real deployment shapes.

Measures the 9-query matrix two ways, using the `icefalldb` engine for everything
(no manual per-query engine selection) so the router does its own routing:

  * icefalldb (in-process)  -- `icefalldb.attach(engine="icefalldb")`: DuckDB-routed scans
    + native metadata/mutations + the persistent result cache (cache-through).
  * daemon              -- a long-lived `icefalldb-server` hit over HTTP `/sql`.
    The server runs the NATIVE DataFusion engine + result cache (it does NOT do
    icefalldb's DuckDB scan-routing today), so its cold scans are slower than the
    in-process icefalldb path; warm repeats are cache hits plus HTTP/JSON overhead.

For each query we report `cold` (first call on an empty cache) and `warm`
(median of `--warm-iters` repeats, i.e. result-cache hits).

NOTE on the daemon: `icefalldb-server` only serves tables registered in the
central catalog (`_catalog.json`). A database built with `icefalldb create`
(rather than `create-table`) has an empty catalog `tables` map, so the server
registers no tables and the daemon section is SKIPPED with a warning. Register
the tables via the catalog (or use a `create-table`-built db) to benchmark the
daemon. See the "daemon limitations" notes in perf/RESULTS.md.

Usage (dataset built by generate_events.py --scale 1m):
    python3 bench_icefalldb_vs_daemon.py [--warm-iters N] [--port P] [--skip-daemon]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DB_PATH = REPO / "target" / "tmp" / "datafusion_bench_db"
SERVER_BIN = REPO / "target" / "release" / "icefalldb-server"

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(HERE))
import icefalldb  # noqa: E402
from run_icefalldb_query_bench import QUERIES  # noqa: E402  (name, sql, tables)


def _discover_tables(db: Path) -> list[str]:
    return sorted(p.name for p in db.iterdir() if (p / "_manifest.json").is_file())


def _clear_cache(db: Path) -> None:
    shutil.rmtree(db / "_query_cache", ignore_errors=True)


def _timed(fn: Callable[[], Any]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def bench_icefalldb(db: Path, warm_iters: int) -> dict[str, tuple[float, float]]:
    """In-process icefalldb engine: cold (first call) + warm (median of repeats)."""
    _clear_cache(db)
    con = icefalldb.attach(
        db, engine="icefalldb", tables=_discover_tables(db), verify_data_checksums=False
    )
    out: dict[str, tuple[float, float]] = {}
    for name, sql, _tables in QUERIES:
        cold = _timed(lambda s=sql: con.sql(s).fetchall())
        warm = median(
            [_timed(lambda s=sql: con.sql(s).fetchall()) for _ in range(warm_iters)]
        )
        out[name] = (cold, warm)
    return out


def bench_daemon(
    db: Path, port: int, warm_iters: int
) -> dict[str, tuple[float, float]] | None:
    """Long-lived icefalldb-server over HTTP /sql: cold + warm. None if unusable."""
    if not SERVER_BIN.exists():
        print(f"daemon skipped: server binary not found at {SERVER_BIN}")
        print("  build it: cargo build --release -p icefalldb-server")
        return None
    _clear_cache(db)
    url = f"http://127.0.0.1:{port}/sql"
    proc = subprocess.Popen(
        [
            str(SERVER_BIN),
            "--db",
            str(db),
            "--port",
            str(port),
            "--result-cache-mb",
            "1024",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def post(sql: str) -> None:
        data = json.dumps({"sql": sql}).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp.read()

    try:
        # Wait for readiness; require that the server actually registered tables.
        tables_url = f"http://127.0.0.1:{port}/tables"
        registered: list[str] = []
        for _ in range(150):
            try:
                with urllib.request.urlopen(tables_url, timeout=5) as resp:
                    registered = json.loads(resp.read()).get("tables", [])
                break
            except Exception:
                time.sleep(0.1)
        if not registered:
            print(
                "daemon skipped: server registered NO tables (the db's "
                "_catalog.json has an empty `tables` map — built with `icefalldb "
                "create`, not catalog-registered). See the daemon-limitation "
                "note in this script's docstring."
            )
            return None
        out: dict[str, tuple[float, float]] = {}
        for name, sql, _tables in QUERIES:
            cold = _timed(lambda s=sql: post(s))
            warm = median([_timed(lambda s=sql: post(s)) for _ in range(warm_iters)])
            out[name] = (cold, warm)
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _fmt(x: float) -> str:
    return f"{x:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-iters", type=int, default=5)
    parser.add_argument("--port", type=int, default=8799)
    parser.add_argument("--skip-daemon", action="store_true")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    if not args.db.is_dir():
        print(f"ERROR: dataset not found at {args.db}", file=sys.stderr)
        print("Generate it: python3 generate_events.py --scale 1m", file=sys.stderr)
        return 1

    print(f"DB: {args.db}  warm-iters: {args.warm_iters}")
    print("Running in-process icefalldb ...", flush=True)
    icefalldb = bench_icefalldb(args.db, args.warm_iters)
    daemon: dict[str, tuple[float, float]] | None = None
    if not args.skip_daemon:
        print("Running daemon (icefalldb-server /sql) ...", flush=True)
        daemon = bench_daemon(args.db, args.port, args.warm_iters)

    print("\n| Query | icefalldb cold | icefalldb warm | daemon cold | daemon warm |")
    print("|---|---:|---:|---:|---:|")
    for name, _sql, _tables in QUERIES:
        bc, bw = icefalldb[name]
        if daemon is not None:
            dc, dw = daemon[name]
            dcell, wcell = _fmt(dc), _fmt(dw)
        else:
            dcell = wcell = "-"
        print(f"| {name} | {_fmt(bc)} | {_fmt(bw)} | {dcell} | {wcell} |")
    print(
        "\nms; cold = first call on empty cache, warm = median repeat (cache hit). "
        "icefalldb = in-process engine='icefalldb'; daemon = icefalldb-server /sql (native engine)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
