#!/usr/bin/env python3
"""Generate a full matrix comparison chart from the value benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from compare import BACKENDS, run_for_rows  # noqa: E402

SIZES = [10_000, 100_000, 1_000_000]
COLORS = {
    "IcefallDB": "#1f77b4",
    "Parquet": "#ff7f0e",
    "DuckDB-on-Parquet": "#2ca02c",
    "SQLite": "#d62728",
    "CSV": "#9467bd",
}


def main() -> int:
    data = {rows: run_for_rows(rows) for rows in SIZES}
    labels = [f"{n:,}" for n in SIZES]
    x = np.arange(len(SIZES))
    width = 0.15
    backend_names = [b.name for b in BACKENDS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("IcefallDB Value Comparison Matrix", fontsize=16, fontweight="bold")

    def plot_metric(
        ax,
        metric_key: str,
        ylabel: str,
        log_scale: bool = False,
        annotate_ms: bool = False,
    ) -> None:
        for i, backend in enumerate(backend_names):
            values = [data[n][backend][metric_key] for n in SIZES]
            offset = width * (i - len(backend_names) / 2 + 0.5)
            bars = ax.bar(
                x + offset, values, width, label=backend, color=COLORS[backend]
            )
            for bar in bars:
                height = bar.get_height()
                if annotate_ms:
                    label = f"{height * 1000:.1f}"
                elif height >= 100:
                    label = f"{height:,.0f}"
                else:
                    label = f"{height:.1f}"
                ax.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plot_metric(
        axes[0, 0],
        "write_rows_per_s",
        "Rows / second",
        log_scale=True,
    )
    axes[0, 0].set_title("Write Throughput")

    plot_metric(
        axes[0, 1],
        "read_all_s",
        "Seconds",
        log_scale=True,
        annotate_ms=True,
    )
    axes[0, 1].set_title("Read-all Latency")

    plot_metric(
        axes[1, 0],
        "aggregate_s",
        "Seconds",
        log_scale=True,
        annotate_ms=True,
    )
    axes[1, 0].set_title("Aggregate Latency (COUNT + AVG)")

    plot_metric(
        axes[1, 1],
        "size_per_row",
        "Bytes / row",
        log_scale=False,
    )
    axes[1, 1].set_title("Storage Size")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(backend_names),
        bbox_to_anchor=(0.5, 0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = REPO_ROOT / "python" / "benchmarks" / "comparison_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved chart to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
