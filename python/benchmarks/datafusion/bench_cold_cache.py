#!/usr/bin/env python3
"""Cold-cache variant of the IcefallDB DataFusion 9-query benchmark matrix.

Evicts Parquet files from the OS page cache with ``posix_fadvise`` before each
timed query, then reuses the query list and timing code from
``run_icefalldb_query_bench.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "target" / "tmp"
DB_PATH = ROOT / "datafusion_bench_db"
FALLBACK_EVICTOR = ROOT / "cold_cache_eviction.bin"

# Reuse the query matrix and the core timing helper from the warm runner.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_icefalldb_query_bench import (  # noqa: E402
    QUERIES,
    _attach_engine,
    _run_query,
    print_markdown_table,
    summarize,
)


def collect_parquet_files(*dirs: Path) -> list[Path]:
    """Collect every Parquet file under the given directories."""
    files: list[Path] = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.parquet"))
    return files


def _build_fallback_evictor(total_bytes: int) -> Callable[[], None]:
    """Return a function that reads a ~2x dataset scratch file to evict cache."""
    needed = max(total_bytes * 2, 2 * 1024 * 1024 * 1024)
    block = b"\0" * (1024 * 1024)

    if not FALLBACK_EVICTOR.exists() or FALLBACK_EVICTOR.stat().st_size < needed:
        FALLBACK_EVICTOR.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with open(FALLBACK_EVICTOR, "wb") as f:
            while written < needed:
                f.write(block)
                written += len(block)

    def evict() -> None:
        with open(FALLBACK_EVICTOR, "rb") as f:
            while f.read(8 * 1024 * 1024):
                pass

    return evict


def evict_cache(paths: list[Path], total_bytes: int) -> str:
    """Evict the given Parquet files from the OS page cache.

    Returns the method used (``posix_fadvise`` or ``fallback_read``).
    """
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        for path in paths:
            try:
                fd = os.open(str(path), os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
            except Exception as exc:  # pragma: no cover
                print(f"  cache eviction warning for {path}: {exc}", file=sys.stderr)
        return "posix_fadvise"

    evictor = _build_fallback_evictor(total_bytes)
    evictor()
    return "fallback_read"


def benchmark_query_cold(
    db_path: Path,
    tables: list[str],
    query: str,
    engine: str,
    iters: int,
    parquet_files: list[Path],
    total_bytes: int,
) -> list[float]:
    """Run ``query`` ``iters`` times with cache eviction before each iteration."""
    times_ms: list[float] = []
    for _ in range(iters):
        evict_cache(parquet_files, total_bytes)
        connection = _attach_engine(db_path, tables, engine)
        start = time.perf_counter()
        _run_query(connection, query)
        times_ms.append((time.perf_counter() - start) * 1000.0)
    return times_ms


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the IcefallDB DataFusion cold-cache benchmark matrix."
    )
    parser.add_argument(
        "--scale",
        choices=["1m", "10m"],
        default="1m",
        help="Dataset scale to benchmark",
    )
    parser.add_argument(
        "--engine",
        choices=["duckdb", "datafusion"],
        default="duckdb",
        help="Query engine to benchmark",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=5,
        help="Number of cold-cache iterations per query (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Directory for JSON result files",
    )
    args = parser.parse_args()

    table_dirs = [DB_PATH / "events", DB_PATH / "categories", DB_PATH / "events_wide"]
    parquet_files = collect_parquet_files(*table_dirs)
    total_bytes = sum(p.stat().st_size for p in parquet_files)
    eviction_method = evict_cache(parquet_files, total_bytes)

    print(f"Dataset scale:    {args.scale}")
    print(f"IcefallDB db:         {DB_PATH}")
    print(f"Engine:           {args.engine}")
    print(f"Iterations:       {args.iters}")
    print(f"Cache eviction:   {eviction_method}")
    print(f"Parquet files:    {len(parquet_files)}")
    print(f"Total size:       {total_bytes / 1024 / 1024:.2f} MiB")
    print()

    results: list[dict[str, Any]] = []
    for qname, query, tables in QUERIES:
        print(f"=== Query: {qname} ===")
        print(query.replace("\n", " "))
        print()

        try:
            times_ms = benchmark_query_cold(
                DB_PATH,
                tables,
                query,
                args.engine,
                args.iters,
                parquet_files,
                total_bytes,
            )
        except (ValueError, NotImplementedError) as exc:
            print(f"  skipped: {exc}")
            print()
            continue

        stats = summarize(times_ms)
        print(
            f"  {args.engine} (cold): "
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
        "dataset": "icefalldb_query_bench_cold",
        "scale": args.scale,
        "iters": args.iters,
        "clear_cache_per_iter": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queries": results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        args.output_dir / f"icefalldb_query_bench_cold_{args.scale}_{args.engine}.json"
    )
    output_path.write_text(json.dumps(matrix, indent=2))
    print(f"Wrote results to {output_path}")
    print()

    print("## Cold-cache results")
    print_markdown_table(results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
