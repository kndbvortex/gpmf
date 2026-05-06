from __future__ import annotations

from typing import Dict

import numpy as np

from ..core.result import MiningResult


def pattern_count(result: MiningResult) -> int:
    return len(result.patterns)


def execution_time(result: MiningResult) -> float:
    return result.execution_time or 0.0


def throughput(result: MiningResult) -> float:
    t = execution_time(result)
    if t <= 0:
        return float("inf")
    return len(result.patterns) / t


def support_stats(result: MiningResult) -> Dict[str, float]:
    supports = [p.support for p in result.patterns]
    if not supports:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "median": 0.0}
    arr = np.array(supports)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def length_stats(result: MiningResult) -> Dict[str, float]:
    lengths = [len(p) for p in result.patterns]
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0.0, "std": 0.0}
    arr = np.array(lengths)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def redundancy_ratio(result: MiningResult) -> float:
    patterns = result.patterns
    if len(patterns) < 2:
        return 0.0

    pattern_sets = [frozenset(p.to_string()) for p in patterns]
    redundant = 0
    n = len(pattern_sets)
    for i in range(n):
        for j in range(n):
            if i != j and pattern_sets[i].issubset(pattern_sets[j]):
                if patterns[i].support <= patterns[j].support:
                    redundant += 1
                    break
    return redundant / n


def extract_metrics(result: MiningResult) -> Dict[str, float]:
    ss = support_stats(result)
    ls = length_stats(result)
    return {
        "n_patterns": pattern_count(result),
        "execution_time": execution_time(result),
        "throughput": throughput(result),
        "support_min": ss["min"],
        "support_max": ss["max"],
        "support_mean": ss["mean"],
        "support_std": ss["std"],
        "length_min": ls["min"],
        "length_max": ls["max"],
        "length_mean": ls["mean"],
    }
