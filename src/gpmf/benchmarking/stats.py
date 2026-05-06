from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_ranks(
    results_matrix: np.ndarray,
    lower_better: bool = True,
) -> np.ndarray:
    n_estimators, n_datasets = results_matrix.shape
    ranks = np.zeros_like(results_matrix, dtype=float)
    for j in range(n_datasets):
        col = results_matrix[:, j]
        order = np.argsort(col) if lower_better else np.argsort(-col)
        for rank, idx in enumerate(order, start=1):
            ranks[idx, j] = rank
    return ranks


def check_friedman(ranks: np.ndarray) -> Tuple[float, float]:
    from scipy.stats import friedmanchisquare

    k, n = ranks.shape
    args = [ranks[i] for i in range(k)]
    stat, p_value = friedmanchisquare(*args)
    return float(stat), float(p_value)


def nemenyi_test(
    ordered_avg_ranks: np.ndarray,
    n_datasets: int,
    alpha: float = 0.05,
) -> np.ndarray:
    k = len(ordered_avg_ranks)
    q_alpha = {0.10: 1.960, 0.05: 2.241, 0.01: 2.576}
    q = q_alpha.get(alpha, 2.241)
    cd = q * np.sqrt(k * (k + 1) / (6 * n_datasets))
    diff_matrix = np.abs(ordered_avg_ranks[:, None] - ordered_avg_ranks[None, :])
    return diff_matrix < cd


def wilcoxon_test(
    results: np.ndarray,
    labels: List[str],
    lower_better: bool = True,
) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    n = len(labels)
    p_values = np.ones((n, n))
    stats = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            x, y = results[i], results[j]
            diff = x - y
            if np.all(diff == 0):
                continue
            stat, p = wilcoxon(x, y)
            stats[i, j] = stats[j, i] = stat
            p_values[i, j] = p_values[j, i] = p

    df_p = pd.DataFrame(p_values, index=labels, columns=labels)
    df_s = pd.DataFrame(stats, index=labels, columns=labels)
    return df_p, df_s


def average_ranks(
    results_matrix: np.ndarray,
    lower_better: bool = True,
) -> np.ndarray:
    ranks = compute_ranks(results_matrix, lower_better=lower_better)
    return ranks.mean(axis=1)
