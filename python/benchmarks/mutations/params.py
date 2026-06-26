"""Fixed benchmark parameters for the mutations benchmark suite.

All values are pinned per the mutations design spec so results are
reproducible and comparable across runs without fuzzy bars.
"""

# ---------------------------------------------------------------------------
# Full benchmark parameters
# ---------------------------------------------------------------------------

# Dataset sizes in rows.
SIZES: list[int] = [1_000_000, 10_000_000, 100_000_000]

# Fragment counts per dataset — ~10 (large fragments) and ~10 000 (small
# fragments) to exercise snapshot-open cost vs fragment count.
FRAGMENTS: list[int] = [10, 10_000]

# Mutation selectivity levels.  "point" means exactly 1 row; the floats are
# fractions of the table (0.1% / 1% / 10%).
SELECTIVITY: list[str | float] = ["point", 0.001, 0.01, 0.10]

# CDC/MERGE mix: (update_fraction, insert_fraction) over a 10%-of-table batch.
CDC_MIX: tuple[float, float] = (0.80, 0.20)

# Minimum timed iterations per workload cell (p50/p95 over ≥ 20 runs).
RUNS: int = 20

# ---------------------------------------------------------------------------
# SMOKE profile — fast local validation (≈ seconds, not minutes)
# ---------------------------------------------------------------------------

SMOKE_SIZE: int = 100_000
SMOKE_FRAGMENTS: list[int] = [4]
SMOKE_SELECTIVITY: list[str | float] = ["point", 0.01, 0.10]
SMOKE_RUNS: int = 3
