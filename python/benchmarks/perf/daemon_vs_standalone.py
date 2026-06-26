#!/usr/bin/env python3
"""Single-record UPDATE latency via three execution modes, to show what the
optional daemon buys: it pays table-open + engine-startup **once**, bringing
one-shot latency toward the warm in-process numbers.

Modes (per-op median over N updates of distinct rows; fresh table clone per mode
so each starts from identical state):

  * **one-shot CLI** — `icefalldb query <t> 'UPDATE …'` per op: a fresh process that
    pays tokio + the DataFusion session build + table open + commit every time.
  * **daemon** — a single long-lived `icefalldb-server` (opens once); each op is the
    lean `icefalldb query … --server <url>` thin client (no DataFusion session) →
    HTTP → the daemon's commit + incremental refresh.
  * **in-process PyO3** — one `IcefallDBConnection` (opens once); each op is just
    `conn.mutate('UPDATE …')` (commit + refresh, no per-op startup/open).

The daemon is an optional opt-in; IcefallDB remains fully usable with no server.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from benchmarks.mutations.generate import make_table  # noqa: E402

DEFAULT_SCALES = [100, 1_000_000]
REPEATS = 9
TABLE = "bench_data"


def _bin(name: str) -> Path:
    for c in (REPO_ROOT / f"target/release/{name}", REPO_ROOT / f"target/debug/{name}"):
        if c.is_file() and os.access(str(c), os.X_OK):
            return c
    raise RuntimeError(f"{name} not found; build with: cargo build --release")


def _run_cli(args: list[str]) -> None:
    subprocess.run(
        [str(_bin("icefalldb")), *args], cwd=REPO_ROOT, check=True, capture_output=True
    )


def _schema(n: int) -> dict:
    return {
        "schema_id": 1,
        "columns": [
            {"name": "id", "type": "int64", "nullable": True, "field_id": 1},
            {"name": "value", "type": "int64", "nullable": True, "field_id": 2},
            {"name": "category", "type": "utf8", "nullable": True, "field_id": 3},
        ],
        "sort": ["id"],
        "row_group_target_rows": max(1, n),
        "row_group_target_bytes": 1 << 40,
        "dropped_columns": [],
        "max_field_id": 3,
    }


def _build_base(db: Path, n: int) -> None:
    if db.exists():
        shutil.rmtree(db)
    db.mkdir(parents=True)
    data = make_table(n)
    schema_path = db / "s.json"
    schema_path.write_text(json.dumps(_schema(n)))
    tsv = db / "d.tsv"
    ids = data.column("id").to_pylist()
    vals = data.column("value").to_pylist()
    cats = data.column("category").to_pylist()
    with tsv.open("w") as f:
        f.write("id\tvalue\tcategory\n")
        for i, v, c in zip(ids, vals, cats):
            f.write(f"{i}\t{v}\t{c}\n")
    try:
        # create-table registers the table in the central catalog so the daemon
        # discovers it via the catalog (not the directory fallback).
        _run_cli(["create-table", str(db), TABLE, "--schema", str(schema_path)])
        _run_cli(["import", str(db), TABLE, str(tsv)])
        _run_cli(["create-index", str(db), TABLE, "id", "--unique"])
    finally:
        schema_path.unlink(missing_ok=True)
        tsv.unlink(missing_ok=True)


def _clone(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _update_sql(target_id: int) -> str:
    return f'UPDATE "{TABLE}" SET value = 424242 WHERE id = {target_id}'


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_port(port: int, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("daemon did not bind in time")


def _measure_cli(db: Path, ids: list[int]) -> list[float]:
    out = []
    for tid in ids:
        t0 = time.perf_counter()
        _run_cli(["query", str(db / TABLE), _update_sql(tid)])
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _measure_daemon(db: Path, ids: list[int]) -> list[float]:
    port = _free_port()
    proc = subprocess.Popen(
        [str(_bin("icefalldb-server")), "--db", str(db), "--port", str(port)],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_port(port)
        url = f"http://127.0.0.1:{port}"
        out = []
        for tid in ids:
            t0 = time.perf_counter()
            _run_cli(["query", str(db / TABLE), _update_sql(tid), "--server", url])
            out.append((time.perf_counter() - t0) * 1000.0)
        return out
    finally:
        proc.terminate()
        proc.wait(timeout=10)


_PYO3_PROBE = """
import sys, time, json
sys.path.insert(0, sys.argv[1])
import icefalldb_query_py
db, ids = sys.argv[2], json.loads(sys.argv[3])
conn = icefalldb_query_py.IcefallDBConnection(db, ["bench_data"])
conn.table_count()  # finish opening before timing
out = []
for tid in ids:
    t0 = time.perf_counter()
    conn.mutate(f'UPDATE "bench_data" SET value = 424242 WHERE id = {tid}')
    out.append((time.perf_counter() - t0) * 1000.0)
print(json.dumps(out))
"""


def _measure_pyo3(db: Path, ids: list[int]) -> list[float]:
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            _PYO3_PROBE,
            str(REPO_ROOT / "python"),
            str(db),
            json.dumps(ids),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"pyo3 probe failed: {res.stderr or res.stdout}")
    return json.loads(res.stdout.strip())


def run(out_dir: Path, scales: list[int], repeats: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "_base"
    tmp = out_dir / "_tmp"
    results: dict[str, dict] = {}
    for n in scales:
        print(f"\n=== rows={n:,} ===")
        _build_base(base, n)
        ids = [min(n - 1, i * max(1, n // repeats)) for i in range(repeats)]
        modes = {}
        for name, fn in (
            ("one_shot_cli", _measure_cli),
            ("daemon", _measure_daemon),
            ("in_process_pyo3", _measure_pyo3),
        ):
            clone = tmp / name
            _clone(base, clone)
            samples = fn(clone, ids)
            modes[name] = {
                "median_ms": median(samples),
                "min_ms": min(samples),
                "max_ms": max(samples),
            }
            shutil.rmtree(clone, ignore_errors=True)
            print(f"  {name:16s} median={modes[name]['median_ms']:.1f}ms")
        results[str(n)] = modes
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return results


def _markdown(results: dict, scales: list[int]) -> str:
    lines = [
        "| rows | one-shot CLI (ms) | daemon (ms) | in-process PyO3 (ms) |",
        "|-----:|------------------:|------------:|---------------------:|",
    ]
    for n in scales:
        r = results[str(n)]
        lines.append(
            f"| {n:,} | {r['one_shot_cli']['median_ms']:.1f} | "
            f"{r['daemon']['median_ms']:.1f} | {r['in_process_pyo3']['median_ms']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scales",
        type=lambda s: [int(x) for x in s.split(",")],
        default=DEFAULT_SCALES,
    )
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "target/tmp/perf_daemon")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    results = run(args.out_dir, args.scales, args.repeats)
    json_path = args.json_out or (args.out_dir / "daemon_vs_standalone.json")
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote JSON: {json_path}\n")
    print(_markdown(results, args.scales))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
