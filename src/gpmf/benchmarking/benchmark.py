from __future__ import annotations

import tracemalloc
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

from ..core.dataset import GradualDataset
from ..factory import AlgorithmRegistry
from .metrics import extract_metrics
from .results import BenchmarkEntry, BenchmarkReport


def run_benchmark(
    algorithms: Sequence[str],
    data: Union[pd.DataFrame, str, GradualDataset],
    min_supports: Optional[Sequence[float]] = None,
    n_runs: int = 1,
    dataset_name: str = "dataset",
    algorithm_kwargs: Optional[Dict[str, Dict]] = None,
) -> BenchmarkReport:
    if min_supports is None:
        min_supports = [0.5]
    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    entries: List[BenchmarkEntry] = []

    for algo_name in algorithms:
        kwargs = algorithm_kwargs.get(algo_name, {})
        for min_sup in min_supports:
            for run_id in range(n_runs):
                algo_cls = AlgorithmRegistry.get(algo_name)
                algo = algo_cls(min_support=min_sup, **kwargs)
                tracemalloc.start()
                result = algo.mine_and_get_result(data)
                final_mem, peak_mem = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                m = extract_metrics(result)
                entry = BenchmarkEntry(
                    algorithm=algo_name,
                    dataset_name=dataset_name,
                    min_support=min_sup,
                    run_id=run_id,
                    execution_time=m["execution_time"],
                    n_patterns=int(m["n_patterns"]),
                    support_min=m["support_min"],
                    support_max=m["support_max"],
                    support_mean=m["support_mean"],
                    support_std=m["support_std"],
                    length_min=int(m["length_min"]),
                    length_max=int(m["length_max"]),
                    length_mean=m["length_mean"],
                    throughput=m["throughput"],
                    peak_memory_bytes=peak_mem,
                    final_memory_bytes=final_mem,
                )
                entries.append(entry)

    return BenchmarkReport(entries)


def compare_algorithms(
    algorithms: Sequence[str],
    datasets: Dict[str, Union[pd.DataFrame, str, GradualDataset]],
    min_support: float = 0.5,
    n_runs: int = 1,
    algorithm_kwargs: Optional[Dict[str, Dict]] = None,
) -> BenchmarkReport:
    if algorithm_kwargs is None:
        algorithm_kwargs = {}

    entries: List[BenchmarkEntry] = []

    for dataset_name, data in datasets.items():
        report = run_benchmark(
            algorithms=algorithms,
            data=data,
            min_supports=[min_support],
            n_runs=n_runs,
            dataset_name=dataset_name,
            algorithm_kwargs=algorithm_kwargs,
        )
        entries.extend(report.entries)

    return BenchmarkReport(entries)
