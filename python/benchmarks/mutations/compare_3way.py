#!/usr/bin/env python3
"""Single-row INSERT / UPDATE cost: IcefallDB vs DuckDB-native vs Duck-on-Parquet.

Three representations of the same table (id unique key, value int, category str)
at 100 and 1,000,000 rows; each backend performs ONE single-row mutation and we
report the median wall-clock over N fresh-state runs.

Backends / methodology (kept honest — they differ by necessity):
  * IcefallDB        — UPDATE: in-process ``IcefallDBConnection.mutate()`` (open
                     excluded, warm); INSERT: ``icefalldb insert`` CLI e2e — the
                     binding has no in-process row-append, so this carries CLI
                     process startup (~4-10 ms). Both append/patch only the
                     changed row(s); the base fragment is never rewritten.
  * DuckDB-native  — in-process ``con.execute()`` against the ``.duckdb`` file
                     (open excluded for UPDATE). In-place row mutation.
  * Duck-on-Parquet— Parquet is immutable, so a single-row change = rewrite the
                     WHOLE file (read all rows -> apply -> write). This is the
                     cost IcefallDB exists to avoid; shown for contrast.

Each timed run starts from identical on-disk state (fresh clone / fresh output
file). Run from the repo root with the venv active and the release binary built:

    python python/benchmarks/mutations/compare_3way.py [--scales 100,1000000] [--runs 7]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from statistics import median

import duckdb
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))

from benchmarks.mutations.generate import (  # noqa: E402
    _icefalldb_cli,
    build_icefalldb_table,
    build_duckdb_table,
    build_parquet_copy,
    make_table,
)
from icefalldb.producer import write_icefalldb_ready_parquet  # noqa: E402

import icefalldb_query_py  # noqa: E402  (PyO3 binding, built via maturin)

TABLE = "bench_data"


def _t(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _build_base(base: Path, data: pa.Table) -> dict:
    """Build the three representations once; clone per run later."""
    db = base / "icefalldb_db"
    db.mkdir(parents=True)
    build_icefalldb_table(db, TABLE, data, n_fragments=1)
    subprocess.run(
        [str(_icefalldb_cli()), "create-index", "--unique", str(db), TABLE, "id"],
        check=True,
        capture_output=True,
    )
    duck = base / "bench.duckdb"
    build_duckdb_table(duck, data)
    con = duckdb.connect(str(duck))
    con.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS idx ON {TABLE} (id)")
    con.close()
    parquet = base / "bench.parquet"
    build_parquet_copy(parquet, data)
    return {"db": db, "duck": duck, "parquet": parquet}


def _clone_db(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def bench_update(base: dict, data: pa.Table, work: Path, runs: int) -> dict:
    n = len(data)
    kid = n // 2  # contiguous ids 0..n-1
    sql = f"UPDATE {TABLE} SET value = value + 1 WHERE id = {kid}"
    icefalldb, duck, ponp = [], [], []
    for i in range(runs):
        # IcefallDB: warm in-process mutate (open excluded), fresh clone.
        cdb = _clone_db(base["db"], work / f"u_b_{i}")
        conn = icefalldb_query_py.IcefallDBConnection(str(cdb), [TABLE])
        icefalldb.append(_t(lambda: conn.mutate(sql)))
        shutil.rmtree(cdb, ignore_errors=True)

        # DuckDB-native: fresh copy, time the in-place UPDATE.
        cduck = work / f"u_d_{i}.duckdb"
        shutil.copy(base["duck"], cduck)
        con = duckdb.connect(str(cduck))
        duck.append(_t(lambda: con.execute(sql)))
        con.close()
        cduck.unlink()

        # Duck-on-Parquet: rewrite the whole file with the row changed.
        out = work / f"u_p_{i}.parquet"
        src = str(base["parquet"])
        rewrite = (
            f"COPY (SELECT id, CASE WHEN id = {kid} THEN value + 1 ELSE value END "
            f"AS value, category FROM read_parquet('{src}')) "
            f"TO '{out}' (FORMAT PARQUET)"
        )
        c = duckdb.connect()
        ponp.append(_t(lambda: c.execute(rewrite)))
        c.close()
        out.unlink(missing_ok=True)
    return {
        "IcefallDB": median(icefalldb) * 1000,
        "DuckDB-native": median(duck) * 1000,
        "Duck-on-Parquet": median(ponp) * 1000,
    }


def bench_insert(base: dict, data: pa.Table, work: Path, runs: int) -> dict:
    n = len(data)
    # one new row appended past the contiguous id range
    row = pa.table(
        {
            "id": pa.array([n], pa.int64()),
            "value": pa.array([12345], pa.int64()),
            "category": pa.array(["cat_0"], pa.string()),
        }
    )
    one_parquet = work / "one_row.parquet"
    write_icefalldb_ready_parquet(one_parquet, row, row_group_size=1)
    cli = str(_icefalldb_cli())

    icefalldb, duck, ponp = [], [], []
    for i in range(runs):
        # IcefallDB: CLI e2e append of a 1-row parquet (new fragment).
        cdb = _clone_db(base["db"], work / f"i_b_{i}")
        icefalldb.append(
            _t(
                lambda: subprocess.run(
                    [cli, "insert", str(cdb), TABLE, str(one_parquet)],
                    check=True,
                    capture_output=True,
                )
            )
        )
        shutil.rmtree(cdb, ignore_errors=True)

        # DuckDB-native: fresh copy, time the in-place INSERT.
        cduck = work / f"i_d_{i}.duckdb"
        shutil.copy(base["duck"], cduck)
        con = duckdb.connect(str(cduck))
        duck.append(
            _t(lambda: con.execute(f"INSERT INTO {TABLE} VALUES ({n}, 12345, 'cat_0')"))
        )
        con.close()
        cduck.unlink()

        # Duck-on-Parquet: rewrite the whole file with one extra row.
        out = work / f"i_p_{i}.parquet"
        src = str(base["parquet"])
        rewrite = (
            f"COPY (SELECT * FROM read_parquet('{src}') UNION ALL "
            f"SELECT {n} AS id, 12345 AS value, 'cat_0' AS category) "
            f"TO '{out}' (FORMAT PARQUET)"
        )
        c = duckdb.connect()
        ponp.append(_t(lambda: c.execute(rewrite)))
        c.close()
        out.unlink(missing_ok=True)
    one_parquet.unlink(missing_ok=True)
    return {
        "IcefallDB": median(icefalldb) * 1000,
        "DuckDB-native": median(duck) * 1000,
        "Duck-on-Parquet": median(ponp) * 1000,
    }


BACKENDS = ["IcefallDB", "DuckDB-native", "Duck-on-Parquet"]


def _print_matrix(title: str, data: dict[int, dict], scales: list[int]) -> None:
    print(f"\n## {title} — single-row latency (median ms)\n")
    hdr = f"{'Backend':<18}" + "".join(f"{f'{n:,} rows':>18}" for n in scales)
    print(hdr)
    print("-" * len(hdr))
    for b in BACKENDS:
        line = f"{b:<18}" + "".join(f"{data[n][b]:>18,.2f}" for n in scales)
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="100,1000000")
    ap.add_argument("--runs", type=int, default=7)
    args = ap.parse_args()
    scales = [int(s) for s in args.scales.split(",")]

    upd: dict[int, dict] = {}
    ins: dict[int, dict] = {}
    for n in scales:
        print(f"Preparing {n:,} rows ...", flush=True)
        data = make_table(n)
        base_dir = Path(tempfile.mkdtemp(prefix=f"cmp3_{n}_base_"))
        work = Path(tempfile.mkdtemp(prefix=f"cmp3_{n}_work_"))
        try:
            base = _build_base(base_dir, data)
            print("  UPDATE ...", flush=True)
            upd[n] = bench_update(base, data, work, args.runs)
            print("  INSERT ...", flush=True)
            ins[n] = bench_insert(base, data, work, args.runs)
        finally:
            shutil.rmtree(base_dir, ignore_errors=True)
            shutil.rmtree(work, ignore_errors=True)

    _print_matrix("Point UPDATE (1 row by id)", upd, scales)
    _print_matrix("Single-row INSERT (append 1 row)", ins, scales)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
