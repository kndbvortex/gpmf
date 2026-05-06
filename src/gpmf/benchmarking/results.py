from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class BenchmarkEntry:
    algorithm: str
    dataset_name: str
    min_support: float
    run_id: int
    execution_time: float
    n_patterns: int
    support_min: float
    support_max: float
    support_mean: float
    support_std: float
    length_min: int
    length_max: int
    length_mean: float
    throughput: float
    peak_memory_bytes: int
    final_memory_bytes: int
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("extra")
        d.update(self.extra)
        return d


class BenchmarkReport:
    def __init__(self, entries: List[BenchmarkEntry], timestamp: Optional[datetime] = None):
        self.entries = entries
        self.timestamp = timestamp or datetime.now()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([e.to_dict() for e in self.entries])

    def summary(self) -> pd.DataFrame:
        df = self.to_dataframe()
        numeric_cols = [
            "execution_time", "n_patterns", "support_mean",
            "length_mean", "throughput", "peak_memory_bytes", "final_memory_bytes",
        ]
        available = [c for c in numeric_cols if c in df.columns]
        group_cols = ["algorithm", "dataset_name", "min_support"]
        return (
            df.groupby(group_cols)[available]
            .agg(["mean", "std"])
            .round(4)
        )

    def save_csv(self, filepath: str) -> None:
        self.to_dataframe().to_csv(filepath, index=False)

    def save_json(self, filepath: str) -> None:
        data = {
            "timestamp": self.timestamp.isoformat(),
            "entries": [e.to_dict() for e in self.entries],
        }
        Path(filepath).write_text(json.dumps(data, indent=2))

    @classmethod
    def load_csv(cls, filepath: str) -> "BenchmarkReport":
        df = pd.read_csv(filepath)
        fixed_cols = {
            "algorithm", "dataset_name", "min_support", "run_id",
            "execution_time", "n_patterns", "support_min", "support_max",
            "support_mean", "support_std", "length_min", "length_max",
            "length_mean", "throughput", "peak_memory_bytes", "final_memory_bytes",
        }
        entries = []
        for _, row in df.iterrows():
            base = {k: row[k] for k in fixed_cols if k in row}
            extra = {k: row[k] for k in row.index if k not in fixed_cols}
            entries.append(BenchmarkEntry(**base, extra=extra))
        return cls(entries)

    @classmethod
    def load_json(cls, filepath: str) -> "BenchmarkReport":
        data = json.loads(Path(filepath).read_text())
        timestamp = datetime.fromisoformat(data["timestamp"])
        fixed_cols = {
            "algorithm", "dataset_name", "min_support", "run_id",
            "execution_time", "n_patterns", "support_min", "support_max",
            "support_mean", "support_std", "length_min", "length_max",
            "length_mean", "throughput", "peak_memory_bytes", "final_memory_bytes",
        }
        entries = []
        for d in data["entries"]:
            base = {k: d[k] for k in fixed_cols if k in d}
            extra = {k: d[k] for k in d if k not in fixed_cols}
            entries.append(BenchmarkEntry(**base, extra=extra))
        return cls(entries, timestamp=timestamp)

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"BenchmarkReport(entries={len(self.entries)}, timestamp={self.timestamp.isoformat()})"
