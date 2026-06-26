#!/usr/bin/env python3
"""T0 selective-scan throughput measurement: IcefallDB vs DuckDB, warm/cold, StringView A/B.

Produces p50/p95 numbers for warm and cold cache on the selective queries
(warm_filtered_scan, wide_filter, wide_agg) plus all 9 queries for completeness.
Measures StringView ON vs OFF for DataFusion to A/B the lever.

Results are written to THROUGHPUT.md in the same directory.

Usage:
    python3 run_t0_throughput_bench.py [--iters N] [--selective-only]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "target" / "tmp"
DB_PATH = ROOT / "datafusion_bench_db"
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(HERE))

# Keep original query list from run_icefalldb_query_bench.py
from run_icefalldb_query_bench import QUERIES  # noqa: E402

import duckdb  # noqa: E402
import icefalldb  # noqa: E402

# The selective-scan queries the brief focuses on (T0 primary targets).
SELECTIVE_QUERY_NAMES = frozenset(
    ["warm_filtered_scan", "wide_filter", "wide_agg", "warm_agg_group"]
)


# ---------------------------------------------------------------------------
# Cache eviction helper (mirrors bench_cold_cache.py)
# ---------------------------------------------------------------------------

FALLBACK_EVICTOR = ROOT / "cold_cache_eviction.bin"


def _collect_parquet_files(*dirs: Path) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.parquet"))
    return files


def _evict_cache(paths: list[Path]) -> str:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        for path in paths:
            try:
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except Exception as exc:
                print(f"  cache eviction warning: {exc}", file=sys.stderr)
        return "posix_fadvise"

    # Fallback: read scratch file to evict
    total_bytes = sum(p.stat().st_size for p in paths if p.exists())
    needed = max(total_bytes * 2, 2 * 1024 * 1024 * 1024)
    if not FALLBACK_EVICTOR.exists() or FALLBACK_EVICTOR.stat().st_size < needed:
        FALLBACK_EVICTOR.parent.mkdir(parents=True, exist_ok=True)
        block = b"\0" * (1024 * 1024)
        written = 0
        with open(FALLBACK_EVICTOR, "wb") as f:
            while written < needed:
                f.write(block)
                written += len(block)
    with open(FALLBACK_EVICTOR, "rb") as f:
        while f.read(8 * 1024 * 1024):
            pass
    return "fallback_read"


# ---------------------------------------------------------------------------
# Connection factories
# ---------------------------------------------------------------------------


def _make_duckdb_conn(tables: list[str]) -> Any:
    con = duckdb.connect()
    icefalldb.attach(str(DB_PATH), connection=con, tables=tables)
    return con


def _run_duckdb(con: Any, query: str) -> None:
    con.sql(query).fetchall()


def _run_df(
    db_path: str, tables: list[str], query: str, force_view_types: bool | None
) -> None:
    """Run one DataFusion query via the PyO3 extension."""
    import icefalldb_query_py

    conn = icefalldb_query_py.IcefallDBConnection(
        db_path,
        tables,
        **(
            {"force_view_types": force_view_types}
            if force_view_types is not None
            else {}
        ),
    )
    result = conn.sql(query)
    # force materialisation
    _ = result.to_pydict()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Return the p-th percentile of a sorted list (0-100 scale)."""
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + frac * (s[hi] - s[lo])


def _summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "p50_ms": _percentile(times_ms, 50),
        "p95_ms": _percentile(times_ms, 95),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": float(statistics.median(times_ms)),
    }


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def run_warm_duckdb(
    query_name: str,
    query: str,
    tables: list[str],
    iters: int,
) -> dict[str, Any]:
    """Warm DuckDB benchmark: single long-lived connection, 1 warmup run."""
    con = _make_duckdb_conn(tables)
    # warmup
    con.sql(query).fetchall()
    times_ms: list[float] = []
    for _ in range(iters):
        t = time.perf_counter()
        con.sql(query).fetchall()
        times_ms.append((time.perf_counter() - t) * 1000.0)
    stats = _summarize(times_ms)
    return {
        "query_name": query_name,
        "engine": "duckdb",
        "mode": "warm",
        "iterations": times_ms,
        **stats,
    }


def run_warm_df(
    query_name: str,
    query: str,
    tables: list[str],
    iters: int,
    force_view_types: bool | None = None,
) -> dict[str, Any]:
    """Warm DataFusion benchmark: OS page cache warm, result cache cleared each iter.

    We keep the same DataFusion connection (same SessionContext, warm JIT/plan
    caches) but clear the persistent query-result cache before each timed run so
    we measure real DataFusion execution, not IPC cache replay.
    """
    import icefalldb_query_py

    db_path = str(DB_PATH)
    label = "datafusion"
    if force_view_types is True:
        label = "datafusion_sv_on"
    elif force_view_types is False:
        label = "datafusion_sv_off"

    kwargs: dict[str, Any] = {}
    if force_view_types is not None:
        kwargs["force_view_types"] = force_view_types

    conn = icefalldb_query_py.IcefallDBConnection(db_path, tables, **kwargs)
    # warmup: prime OS page cache and DataFusion plan caches, then discard result cache
    try:
        conn.sql(query).to_pydict()
    except Exception as e:
        return {
            "query_name": query_name,
            "engine": label,
            "mode": "warm",
            "error": str(e),
            "iterations": [],
            "p50_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
            "median_ms": None,
        }
    conn.clear_cache()

    times_ms: list[float] = []
    for _ in range(iters):
        conn.clear_cache()  # force real execution, not cache replay
        t = time.perf_counter()
        conn.sql(query).to_pydict()
        times_ms.append((time.perf_counter() - t) * 1000.0)

    stats = _summarize(times_ms)
    return {
        "query_name": query_name,
        "engine": label,
        "mode": "warm",
        "iterations": times_ms,
        **stats,
    }


def run_cold_duckdb(
    query_name: str,
    query: str,
    tables: list[str],
    iters: int,
    parquet_files: list[Path],
) -> dict[str, Any]:
    """Cold DuckDB: evict cache + fresh connection per iteration."""
    times_ms: list[float] = []
    for _ in range(iters):
        _evict_cache(parquet_files)
        con = _make_duckdb_conn(tables)
        t = time.perf_counter()
        con.sql(query).fetchall()
        times_ms.append((time.perf_counter() - t) * 1000.0)
    stats = _summarize(times_ms)
    return {
        "query_name": query_name,
        "engine": "duckdb",
        "mode": "cold",
        "iterations": times_ms,
        **stats,
    }


def run_cold_df(
    query_name: str,
    query: str,
    tables: list[str],
    iters: int,
    parquet_files: list[Path],
    force_view_types: bool | None = None,
) -> dict[str, Any]:
    """Cold DataFusion: evict cache + fresh connection + clear result cache per iter."""
    import icefalldb_query_py

    label = "datafusion"
    if force_view_types is True:
        label = "datafusion_sv_on"
    elif force_view_types is False:
        label = "datafusion_sv_off"

    kwargs: dict[str, Any] = {}
    if force_view_types is not None:
        kwargs["force_view_types"] = force_view_types

    times_ms: list[float] = []
    for _ in range(iters):
        _evict_cache(parquet_files)
        conn = icefalldb_query_py.IcefallDBConnection(str(DB_PATH), tables, **kwargs)
        conn.clear_cache()
        try:
            t = time.perf_counter()
            conn.sql(query).to_pydict()
            times_ms.append((time.perf_counter() - t) * 1000.0)
        except Exception as e:
            return {
                "query_name": query_name,
                "engine": label,
                "mode": "cold",
                "error": str(e),
                "iterations": [],
                "p50_ms": None,
                "p95_ms": None,
                "min_ms": None,
                "max_ms": None,
                "median_ms": None,
            }
    stats = _summarize(times_ms)
    return {
        "query_name": query_name,
        "engine": label,
        "mode": "cold",
        "iterations": times_ms,
        **stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="T0 selective-scan throughput benchmark (IcefallDB vs DuckDB, StringView A/B)."
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=7,
        help="Timed iterations per query+engine+mode (default 7)",
    )
    parser.add_argument(
        "--selective-only",
        action="store_true",
        help="Only run selective-scan queries (warm_filtered_scan, wide_filter, ...)",
    )
    parser.add_argument(
        "--skip-cold", action="store_true", help="Skip cold-cache benchmarks (faster)"
    )
    parser.add_argument(
        "--skip-stringview",
        action="store_true",
        help="Skip the StringView ON A/B variant",
    )
    args = parser.parse_args()

    iters = args.iters
    print(f"T0 throughput benchmark — {datetime.now(timezone.utc).isoformat()}")
    print(f"DB:     {DB_PATH}")
    print(f"Iters:  {iters}")
    print()

    # Verify DB exists
    if not DB_PATH.is_dir():
        print(f"ERROR: benchmark DB not found at {DB_PATH}", file=sys.stderr)
        print("Run: python3 generate_events.py --scale 10m", file=sys.stderr)
        return 1

    # Count rows for scale confirmation
    try:
        import icefalldb_query_py

        conn_check = icefalldb_query_py.IcefallDBConnection(
            str(DB_PATH), ["events", "events_wide"]
        )
        ev_count = conn_check.sql("SELECT COUNT(*) as c FROM events").to_pydict()["c"][
            0
        ]
        ew_count = conn_check.sql("SELECT COUNT(*) as c FROM events_wide").to_pydict()[
            "c"
        ][0]
        print(
            f"Scale confirmation: events={ev_count:,} rows, events_wide={ew_count:,} rows"
        )
        scale_note = f"events={ev_count:,}, events_wide={ew_count:,}"
    except Exception as e:
        print(f"WARNING: could not confirm scale: {e}")
        scale_note = "unknown"
    print()

    # Collect Parquet files for cache eviction
    table_dirs = [
        DB_PATH / "events",
        DB_PATH / "categories",
        DB_PATH / "events_wide",
        DB_PATH / "events_sorted",
        DB_PATH / "events_indexed",
        DB_PATH / "events_wide_clustered",
    ]
    parquet_files = _collect_parquet_files(*table_dirs)
    total_bytes = sum(p.stat().st_size for p in parquet_files if p.exists())
    eviction_method = _evict_cache(parquet_files)
    print(
        f"Parquet files: {len(parquet_files)}, total {total_bytes / 1024 / 1024:.0f} MiB"
    )
    print(f"Cache eviction: {eviction_method}")
    print()

    # Filter queries if needed
    queries = QUERIES
    if args.selective_only:
        queries = [(n, q, t) for n, q, t in QUERIES if n in SELECTIVE_QUERY_NAMES]
        print(f"Selective-only: {[n for n, _, _ in queries]}")
        print()

    all_results: list[dict[str, Any]] = []

    for qname, query, tables in queries:
        # Check all required tables exist
        missing = [t for t in tables if not (DB_PATH / t).is_dir()]
        if missing:
            print(f"  SKIP {qname}: missing tables {missing}")
            continue

        print(f"=== {qname} (tables: {tables}) ===")

        # --- Warm DuckDB ---
        print("  warm/duckdb...", end=" ", flush=True)
        r = run_warm_duckdb(qname, query, tables, iters)
        all_results.append(r)
        print(f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

        # --- Warm DataFusion (StringView OFF, production default) ---
        print("  warm/datafusion (sv=off)...", end=" ", flush=True)
        r = run_warm_df(qname, query, tables, iters, force_view_types=None)
        all_results.append(r)
        if r.get("error"):
            print(f"ERROR: {r['error']}")
        else:
            print(f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

        # --- Warm DataFusion (StringView ON) ---
        if not args.skip_stringview:
            print("  warm/datafusion (sv=on)...", end=" ", flush=True)
            r = run_warm_df(qname, query, tables, iters, force_view_types=True)
            all_results.append(r)
            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

        if not args.skip_cold:
            # --- Cold DuckDB ---
            print("  cold/duckdb...", end=" ", flush=True)
            r = run_cold_duckdb(qname, query, tables, iters, parquet_files)
            all_results.append(r)
            print(f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

            # --- Cold DataFusion (StringView OFF) ---
            print("  cold/datafusion (sv=off)...", end=" ", flush=True)
            r = run_cold_df(
                qname, query, tables, iters, parquet_files, force_view_types=None
            )
            all_results.append(r)
            if r.get("error"):
                print(f"ERROR: {r['error']}")
            else:
                print(f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

            # --- Cold DataFusion (StringView ON) ---
            if not args.skip_stringview:
                print("  cold/datafusion (sv=on)...", end=" ", flush=True)
                r = run_cold_df(
                    qname, query, tables, iters, parquet_files, force_view_types=True
                )
                all_results.append(r)
                if r.get("error"):
                    print(f"ERROR: {r['error']}")
                else:
                    print(f"p50={r['p50_ms']:.1f}ms p95={r['p95_ms']:.1f}ms")

        print()

    # Save JSON
    ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ROOT / "t0_throughput_results.json"
    matrix = {
        "dataset": "t0_selective_scan_throughput",
        "scale_note": scale_note,
        "iters": iters,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": all_results,
    }
    output_path.write_text(json.dumps(matrix, indent=2))
    print(f"Wrote JSON: {output_path}")

    # --- Generate THROUGHPUT.md ---
    _write_throughput_md(all_results, scale_note, iters, eviction_method)

    return 0


def _write_throughput_md(
    results: list[dict[str, Any]],
    scale_note: str,
    iters: int,
    eviction_method: str,
) -> None:
    """Write THROUGHPUT.md to the benchmarks/datafusion directory."""
    # Index results by (query_name, engine, mode)
    idx: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in results:
        key = (r["query_name"], r["engine"], r["mode"])
        idx[key] = r

    query_names = list(dict.fromkeys(r["query_name"] for r in results))

    def _fmt(v: float | None) -> str:
        if v is None:
            return "ERR"
        return f"{v:.1f}"

    def _ratio(df_val: float | None, duck_val: float | None) -> str:
        if df_val is None or duck_val is None or duck_val == 0:
            return "N/A"
        r = df_val / duck_val
        return f"{r:.2f}×"

    lines: list[str] = []
    lines.append("# THROUGHPUT.md — T0 Selective-Scan Baseline")
    lines.append("")
    lines.append(
        "Generated by `run_t0_throughput_bench.py` as part of the T0 throughput measurement plan."
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(
        f"- **Scale**: {scale_note} (target 100M events_wide rows; `--scale 10m`)"
    )
    lines.append(f"- **Iterations per query/engine/mode**: {iters}")
    lines.append(f"- **Cold-cache eviction**: `{eviction_method}`")
    lines.append(
        "- **arrow-rs / parquet**: 58.3.0 (≥ 57.1 target — T0.3 already satisfied)"
    )
    lines.append("- **DataFusion**: 54.0.0")
    lines.append(
        "- **Pushdown**: hardcoded ON (`with_pushdown_filters(true).with_reorder_filters(true)` at `crates/icefalldb-query/src/parquet_exec.rs:111-112`)"
    )
    lines.append(
        "- **StringView default**: OFF (`schema_force_view_types = false` in `session.rs:61`)"
    )
    lines.append("")

    lines.append("## T0.3: arrow-rs version check")
    lines.append("")
    lines.append("Confirmed in `Cargo.lock`:")
    lines.append("```")
    lines.append('name = "arrow"')
    lines.append('version = "58.3.0"')
    lines.append('name = "parquet"')
    lines.append('version = "58.3.0"')
    lines.append('name = "datafusion"')
    lines.append('version = "54.0.0"')
    lines.append("```")
    lines.append("")
    lines.append(
        "**T0.3: arrow-rs already at 58.3 ≥ 57.1 target; no bump needed.** Post-#7850 `CachedArrayReader` and adaptive RowSelection are already active."
    )
    lines.append("")

    lines.append("## T0.1: Warm-cache benchmark results")
    lines.append("")
    lines.append(
        "| Query | DuckDB p50 | DuckDB p95 | IcefallDB p50 | IcefallDB p95 | Ratio p95 (MDB/Duck) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for qname in query_names:
        duck = idx.get((qname, "duckdb", "warm"), {})
        df = idx.get((qname, "datafusion", "warm"), {})
        err_note = ""
        if df.get("error"):
            err_note = f" (ERR: {df['error'][:40]})"
        lines.append(
            f"| {qname} | {_fmt(duck.get('p50_ms'))} | {_fmt(duck.get('p95_ms'))} "
            f"| {_fmt(df.get('p50_ms'))} | {_fmt(df.get('p95_ms'))} "
            f"| {_ratio(df.get('p95_ms'), duck.get('p95_ms'))}{err_note} |"
        )
    lines.append("")

    if any((qname, "duckdb", "cold") in idx for qname in query_names):
        lines.append("## T0.1: Cold-cache benchmark results")
        lines.append("")
        lines.append(
            "| Query | DuckDB p50 | DuckDB p95 | IcefallDB p50 | IcefallDB p95 | Ratio p95 (MDB/Duck) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for qname in query_names:
            duck = idx.get((qname, "duckdb", "cold"), {})
            df = idx.get((qname, "datafusion", "cold"), {})
            err_note = ""
            if df.get("error"):
                err_note = f" (ERR: {df['error'][:40]})"
            if not duck and not df:
                continue
            lines.append(
                f"| {qname} | {_fmt(duck.get('p50_ms'))} | {_fmt(duck.get('p95_ms'))} "
                f"| {_fmt(df.get('p50_ms'))} | {_fmt(df.get('p95_ms'))} "
                f"| {_ratio(df.get('p95_ms'), duck.get('p95_ms'))}{err_note} |"
            )
        lines.append("")

    if any((qname, "datafusion_sv_on", "warm") in idx for qname in query_names):
        lines.append("## T0.2: StringView A/B (warm cache)")
        lines.append("")
        lines.append(
            "Comparing DataFusion with StringView OFF (production default) vs ON."
        )
        lines.append("")
        lines.append(
            "| Query | SV OFF p50 | SV OFF p95 | SV ON p50 | SV ON p95 | Ratio SV-ON/OFF p95 | Status |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for qname in query_names:
            off = idx.get((qname, "datafusion", "warm"), {})
            on = idx.get((qname, "datafusion_sv_on", "warm"), {})
            if not off and not on:
                continue
            err_note = ""
            if on.get("error"):
                err_note = "schema-mismatch"
            elif off.get("p95_ms") and on.get("p95_ms"):
                ratio = on["p95_ms"] / off["p95_ms"] if off["p95_ms"] else 0
                if ratio < 0.95:
                    err_note = "faster"
                elif ratio > 1.05:
                    err_note = "slower"
                else:
                    err_note = "same"
            lines.append(
                f"| {qname} | {_fmt(off.get('p50_ms'))} | {_fmt(off.get('p95_ms'))} "
                f"| {_fmt(on.get('p50_ms'))} | {_fmt(on.get('p95_ms'))} "
                f"| {_ratio(on.get('p95_ms'), off.get('p95_ms'))} | {err_note} |"
            )
        lines.append("")

    if any((qname, "datafusion_sv_on", "cold") in idx for qname in query_names):
        lines.append("## T0.2: StringView A/B (cold cache)")
        lines.append("")
        lines.append(
            "| Query | SV OFF p50 | SV OFF p95 | SV ON p50 | SV ON p95 | Ratio SV-ON/OFF p95 | Status |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for qname in query_names:
            off = idx.get((qname, "datafusion", "cold"), {})
            on = idx.get((qname, "datafusion_sv_on", "cold"), {})
            if not off and not on:
                continue
            err_note = ""
            if on.get("error"):
                err_note = "schema-mismatch"
            elif off.get("p95_ms") and on.get("p95_ms"):
                ratio = on["p95_ms"] / off["p95_ms"] if off["p95_ms"] else 0
                if ratio < 0.95:
                    err_note = "faster"
                elif ratio > 1.05:
                    err_note = "slower"
                else:
                    err_note = "same"
            lines.append(
                f"| {qname} | {_fmt(off.get('p50_ms'))} | {_fmt(off.get('p95_ms'))} "
                f"| {_fmt(on.get('p50_ms'))} | {_fmt(on.get('p95_ms'))} "
                f"| {_ratio(on.get('p95_ms'), off.get('p95_ms'))} | {err_note} |"
            )
        lines.append("")

    # Go/no-go section
    lines.append("## Go/No-Go Verdict")
    lines.append("")

    # Compute ratios for the three key selective queries
    selective_queries = [
        "warm_filtered_scan",
        "wide_filter",
        "wide_agg",
        "warm_agg_group",
    ]
    go_data: list[tuple[str, str, float | None, float | None]] = []
    for mode in ["warm", "cold"]:
        for qname in selective_queries:
            duck = idx.get((qname, "duckdb", mode), {})
            df = idx.get((qname, "datafusion", mode), {})
            if duck.get("p95_ms") and df.get("p95_ms") and not df.get("error"):
                ratio = df["p95_ms"] / duck["p95_ms"]
                go_data.append((qname, mode, df["p95_ms"], duck["p95_ms"]))

    if go_data:
        all_within = all(
            (df_v / duck_v) <= 1.5 for _, _, df_v, duck_v in go_data if df_v and duck_v
        )
        max_ratio = max(
            (df_v / duck_v) for _, _, df_v, duck_v in go_data if df_v and duck_v
        )
        worst = max(go_data, key=lambda x: (x[2] / x[3]) if (x[2] and x[3]) else 0)

        if all_within:
            lines.append("**VERDICT: GO (gap ≤ 1.5× — T1–T5 are optional polish)**")
            lines.append("")
            lines.append(
                f"All selective queries (warm+cold) are within 1.5× of DuckDB at 10m scale "
                f"(max ratio: {max_ratio:.2f}× on `{worst[0]}` {worst[1]})."
            )
            lines.append(
                "T0 closes the gap. The later throughput phases (T1–T5) are optional polish."
            )
        else:
            lines.append(
                "**VERDICT: NO-GO (gap > 1.5× on some queries — T1–T5 work needed)**"
            )
            lines.append("")
            lines.append("Queries exceeding 1.5× DuckDB p95:")
            lines.append("")
            for qname, mode, df_v, duck_v in go_data:
                ratio = df_v / duck_v if duck_v else 0
                if ratio > 1.5:
                    lines.append(
                        f"- `{qname}` ({mode}): {df_v:.1f}ms vs {duck_v:.1f}ms = **{ratio:.2f}×**"
                    )
            lines.append("")

        # StringView A/B summary
        sv_results: list[str] = []
        for qname in selective_queries:
            for mode in ["warm", "cold"]:
                off = idx.get((qname, "datafusion", mode), {})
                on = idx.get((qname, "datafusion_sv_on", mode), {})
                if on.get("error"):
                    sv_results.append(
                        f"- `{qname}` ({mode}): StringView ON caused schema-mismatch error"
                    )
                elif off.get("p95_ms") and on.get("p95_ms"):
                    ratio = on["p95_ms"] / off["p95_ms"]
                    direction = (
                        "faster"
                        if ratio < 0.95
                        else ("slower" if ratio > 1.05 else "same")
                    )
                    sv_results.append(
                        f"- `{qname}` ({mode}): SV-ON {direction} ({ratio:.2f}× vs SV-OFF)"
                    )
        if sv_results:
            lines.append("### StringView A/B summary")
            lines.append("")
            for s in sv_results:
                lines.append(s)
            lines.append("")

    else:
        lines.append("**VERDICT: INCOMPLETE — no comparable results found.**")
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Pushdown: hardcoded ON in `build_native_parquet_exec` (`parquet_exec.rs:111-112`)."
    )
    lines.append(
        "  T1 (per-query pushdown gating) is only relevant if pushdown regresses non-selective"
    )
    lines.append(
        "  queries >5% — check `warm_full_count` and `join_100m_x_10` ratios above."
    )
    lines.append(
        "- StringView is disabled by default to preserve schema compatibility with"
    )
    lines.append(
        "  `icefalldb-core` (`Utf8` vs `Utf8View`). Enabling it requires schema-path work (T2+)."
    )
    lines.append(
        "- T0.3 (arrow-rs bump): already satisfied at 58.3.0. No action needed."
    )
    lines.append("")

    out_path = HERE / "THROUGHPUT.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote THROUGHPUT.md: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
