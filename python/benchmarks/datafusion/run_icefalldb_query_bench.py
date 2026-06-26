#!/usr/bin/env python3
"""9-query matrix runner for IcefallDB's DataFusion query-engine benchmarks.

This runner compares the IcefallDB query engine against DuckDB (the oracle) on the
benchmark matrix defined in the DataFusion query-engine spec.

Usage:
    python3 run_icefalldb_query_bench.py --scale 1m --engine duckdb
    python3 run_icefalldb_query_bench.py --scale 10m --engine duckdb --iters 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "target" / "tmp"
DB_PATH = ROOT / "datafusion_bench_db"

sys.path.insert(0, str(REPO / "python"))

import icefalldb  # noqa: E402

# Each entry is (query_name, sql, tables).
QUERIES: list[tuple[str, str, list[str]]] = [
    (
        "warm_agg_group",
        """SELECT category, COUNT(*) AS event_count, AVG(value) AS avg_value
FROM events
WHERE ts >= TIMESTAMP '2024-03-01 00:00:00'
  AND ts <  TIMESTAMP '2024-04-01 00:00:00'
GROUP BY category
ORDER BY event_count DESC
LIMIT 10""",
        ["events"],
    ),
    (
        "warm_filtered_scan",
        """SELECT COUNT(*) AS cnt, SUM(value) AS total
FROM events
WHERE category = 'cat_0' AND value > 0.5""",
        ["events"],
    ),
    (
        "warm_full_count",
        """SELECT COUNT(*) AS cnt FROM events""",
        ["events"],
    ),
    (
        "join_100m_x_10",
        """SELECT c.category_name, COUNT(*) AS event_count
FROM events e
JOIN categories c ON e.category = c.category_name
GROUP BY c.category_name
ORDER BY event_count DESC""",
        ["events", "categories"],
    ),
    (
        "wide_agg",
        """SELECT SUM(int_a) AS s_a, AVG(float_a) AS avg_a,
       SUM(int_b) AS s_b, AVG(float_b) AS avg_b,
       AVG(float_c) AS avg_c, SUM(int_c) AS s_c
FROM events_wide
WHERE int_d > 100 AND float_d < 0.5 AND status = 'ok'""",
        ["events_wide"],
    ),
    (
        "wide_filter",
        """SELECT COUNT(*) AS cnt
FROM events_wide
WHERE int_a > 500000 AND float_b > 0.5
  AND int_c < 500000 AND status = 'ok'""",
        ["events_wide"],
    ),
    (
        "sorted_time_window",
        """SELECT category, COUNT(*) AS event_count, AVG(value) AS avg_value
FROM events_sorted
WHERE ts >= TIMESTAMP '2024-03-01 00:00:00'
  AND ts <  TIMESTAMP '2024-04-01 00:00:00'
GROUP BY category
ORDER BY event_count DESC
LIMIT 10""",
        ["events_sorted"],
    ),
    (
        "indexed_equality",
        """SELECT COUNT(*) AS cnt, SUM(value) AS total
FROM events_indexed
WHERE category = 'cat_0' AND value > 0.5""",
        ["events_indexed"],
    ),
    (
        "clustered_wide_filter",
        """SELECT COUNT(*) AS cnt
FROM events_wide_clustered
WHERE int_a > 500000 AND float_b > 0.5
  AND int_c < 500000 AND status = 'ok'""",
        ["events_wide_clustered"],
    ),
]


def _attach_engine(db_path: Path, tables: list[str], engine: str) -> Any:
    """Attach ``tables`` from ``db_path`` using the requested engine."""
    if engine == "duckdb":
        import duckdb

        con = duckdb.connect()
        icefalldb.attach(str(db_path), connection=con, tables=tables)
        return con

    if engine in ("datafusion", "icefalldb"):
        return icefalldb.attach(str(db_path), tables=tables, engine=engine)

    raise ValueError(f"unsupported engine: {engine!r}")


def _run_query(connection: Any, query: str) -> list[Any]:
    """Execute ``query`` on ``connection`` and return rows as a Python list."""
    sql_result = connection.sql(query)
    # DuckDB's .sql() returns a DuckDBPyRelation with .fetchall().
    if hasattr(sql_result, "fetchall"):
        return sql_result.fetchall()
    # Future DataFusion path may return an Arrow-backed relation.
    if hasattr(sql_result, "to_arrow_table"):
        table = sql_result.to_arrow_table()
        return [tuple(row.values()) for row in table.to_pylist()]
    raise TypeError(f"unsupported query result type: {type(sql_result)!r}")


def benchmark_query(
    db_path: Path,
    tables: list[str],
    query: str,
    engine: str,
    iters: int,
    warmup: bool = True,
    clear_cache_per_iter: bool = False,
) -> list[float]:
    """Run ``query`` ``iters`` times and return wall-clock timings in milliseconds.

    A single connection is used for the warmup and all timed iterations, matching
    the warm-cache methodology of the original Seafowl-vs-DuckDB benchmark.
    """
    connection = _attach_engine(db_path, tables, engine)
    if warmup:
        _run_query(connection, query)
        if clear_cache_per_iter and hasattr(connection, "clear_cache"):
            connection.clear_cache()

    times_ms: list[float] = []
    for _ in range(iters):
        if clear_cache_per_iter and hasattr(connection, "clear_cache"):
            connection.clear_cache()
        start = time.perf_counter()
        _run_query(connection, query)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def summarize(times_ms: list[float]) -> dict[str, float]:
    """Return min/max/median statistics for a list of timings."""
    return {
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": float(statistics.median(times_ms)),
    }


def print_markdown_table(results: list[dict[str, Any]]) -> None:
    """Print a Markdown table of the benchmark results."""
    print("| Query | Engine | Min (ms) | Median (ms) | Max (ms) |")
    print("|---|---|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['query_name']} | {r['engine']} | "
            f"{r['min_ms']:.3f} | {r['median_ms']:.3f} | {r['max_ms']:.3f} |"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the IcefallDB DataFusion 9-query benchmark matrix."
    )
    parser.add_argument(
        "--scale",
        choices=["1m", "10m"],
        default="1m",
        help="Dataset scale to benchmark",
    )
    parser.add_argument(
        "--engine",
        choices=["duckdb", "datafusion", "icefalldb"],
        default="duckdb",
        help="Query engine to benchmark",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of iterations per query (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for JSON result files",
    )
    parser.add_argument(
        "--clear-cache-per-iter",
        action="store_true",
        help="Clear the DataFusion query-result cache before each timed iteration",
    )
    args = parser.parse_args()

    scale_rows = {"1m": 1_000_000, "10m": 10_000_000}[args.scale]
    print(f"Dataset scale: {args.scale} ({scale_rows:,} event rows)")
    print(f"IcefallDB db:      {DB_PATH}")
    print(f"Engine:        {args.engine}")
    print(f"Iterations:    {args.iters}")
    print()

    results: list[dict[str, Any]] = []
    for qname, query, tables in QUERIES:
        print(f"=== Query: {qname} ===")
        print(query.replace("\n", " "))
        print()

        try:
            times_ms = benchmark_query(
                DB_PATH,
                tables,
                query,
                args.engine,
                args.iters,
                clear_cache_per_iter=args.clear_cache_per_iter,
            )
        except Exception as exc:  # e.g. a table the engine cannot attach
            print(f"  {args.engine}: SKIPPED ({type(exc).__name__}: {exc})")
            print()
            results.append(
                {
                    "query_name": qname,
                    "engine": args.engine,
                    "skipped": True,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        stats = summarize(times_ms)
        print(
            f"  {args.engine}: "
            f"min={stats['min_ms']:.3f}ms "
            f"median={stats['median_ms']:.3f}ms "
            f"max={stats['max_ms']:.3f}ms"
        )
        print()

        results.append(
            {
                "query_name": qname,
                "engine": args.engine,
                "iterations": times_ms,
                "median_ms": stats["median_ms"],
                "min_ms": stats["min_ms"],
                "max_ms": stats["max_ms"],
            }
        )

    matrix = {
        "dataset": "icefalldb_query_bench",
        "scale": args.scale,
        "iters": args.iters,
        "clear_cache_per_iter": args.clear_cache_per_iter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queries": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"icefalldb_query_bench_{args.scale}_{args.engine}.json"
    )
    output_path.write_text(json.dumps(matrix, indent=2))
    print(f"Wrote results to {output_path}")
    print()

    print("## Results")
    print_markdown_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
