"""
Benchmark: grite, paraminer, glcm, pglcm, sgrite — with and without RC pruning.

Usage:
    uv run python benchmarks/pruning_benchmark.py
    uv run python benchmarks/pruning_benchmark.py --output results/pruning.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from gpmf.benchmarking import run_benchmark

MIN_SUPPORTS = [0.5, 0.6, 0.7]
N_RUNS = 3

PRUNABLE = ["grite", "glcm", "pglcm", "sgrite"]
UNPRUNABLE = ["paraminer"]


def main(output: str | None = None) -> None:
    data = pd.read_csv(Path(__file__).parents[1] / "data" / "toy.txt", sep=" ")
    dataset_name = "toy"
    print(f"Dataset: {dataset_name}  shape={data.shape}")

    no_prune_kwargs = {algo: {"use_rc_pruning": False} for algo in PRUNABLE}
    prune_kwargs = {algo: {"use_rc_pruning": True} for algo in PRUNABLE}

    print("Running without pruning …")
    report_no = run_benchmark(
        algorithms=PRUNABLE + UNPRUNABLE,
        data=data,
        min_supports=MIN_SUPPORTS,
        n_runs=N_RUNS,
        dataset_name=dataset_name,
        algorithm_kwargs=no_prune_kwargs,
    )

    print("Running with RC pruning …")
    report_yes = run_benchmark(
        algorithms=PRUNABLE,
        data=data,
        min_supports=MIN_SUPPORTS,
        n_runs=N_RUNS,
        dataset_name=dataset_name,
        algorithm_kwargs=prune_kwargs,
    )

    df_no = report_no.to_dataframe()
    df_no["use_rc_pruning"] = False

    df_yes = report_yes.to_dataframe()
    df_yes["use_rc_pruning"] = True

    df = pd.concat([df_no, df_yes], ignore_index=True)

    summary = (
        df.groupby(["algorithm", "use_rc_pruning", "min_support"])[
            ["execution_time", "peak_memory_bytes", "final_memory_bytes", "n_patterns", "throughput"]
        ]
        .mean()
        .round(4)
    )

    print("\n" + "=" * 80)
    print(summary.to_string())
    print("=" * 80)

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nFull results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Path to save CSV results")
    args = parser.parse_args()
    main(output=args.output)
