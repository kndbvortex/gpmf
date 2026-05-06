from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..core.dataset import GradualDataset
from ..factory import AlgorithmRegistry


def split_data(
    data: pd.DataFrame,
    k: int,
    random_state: Optional[int] = None,
) -> List[pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(len(data))
    return [
        data.iloc[chunk].reset_index(drop=True)
        for chunk in np.array_split(indices, k)
    ]


def pattern_stability(
    algorithm_name: str,
    data: pd.DataFrame,
    k: int = 5,
    min_support: float = 0.5,
    algorithm_kwargs: Optional[Dict] = None,
) -> pd.DataFrame:
    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    subsets = split_data(data, k)
    subset_pattern_sets: List[set] = []

    for subset in subsets:
        try:
            algo_cls = AlgorithmRegistry.get(algorithm_name)
            algo = algo_cls(min_support=min_support, **algorithm_kwargs)
            patterns = algo.mine(subset)
            subset_pattern_sets.append({str(p.to_string()) for p in patterns})
        except Exception:
            subset_pattern_sets.append(set())

    all_patterns = set().union(*subset_pattern_sets)
    rows = []
    for pattern_str in sorted(all_patterns):
        presence = [pattern_str in s for s in subset_pattern_sets]
        rows.append({
            "pattern": pattern_str,
            **{f"subset_{i}": v for i, v in enumerate(presence)},
            "n_subsets": sum(presence),
            "frequency": round(sum(presence) / k, 4),
        })

    return pd.DataFrame(rows).sort_values("frequency", ascending=False).reset_index(drop=True)
