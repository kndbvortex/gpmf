# Getting started

## Installation

```bash
pip install gradual-mining
```

Pre-built wheels are available for Linux, macOS and Windows (x86_64 and arm64).

## Basic usage

All algorithms share the same fit/mine interface:

```python
from gpmf.algorithms.graank import GRAANK
import pandas as pd

df = pd.read_csv("data.csv")

algo = GRAANK(min_support=0.5)
patterns = algo.fit(df).get_patterns()

for p in patterns:
    print(p.to_string(), ":", p.support)
```

You can also pass a CSV path directly:

```python
patterns = GRAANK(min_support=0.5).mine("data.csv")
```

## Choosing an algorithm

Two support families exist in GPMF. Pick based on which interpretation fits your task — see [Algorithms](algorithms/README.md) for the formal relationship.

| I want... | Use |
|-----------|-----|
| Pair-based correlations (standard baseline) | GRAANK |
| Longest ordered subsequence | GRITE or SGrite |
| Closed patterns (compact output) | GLCM, PGLCM, or ParaMiner |
| Large dataset, approximate result | ACO-GRAANK |
| Temporal lag discovery | TGrad |
| Seasonal / periodic patterns | MSGP |
| Contrast two populations | EmergingGP |

## Working with results

```python
result = algo.mine_and_get_result(df)

print(result.algorithm)        # 'GRAANK'
print(result.execution_time)   # seconds
print(result.n_patterns)       # number of patterns found

# Filter and sort
result.filter_by_support(0.7)
result.sort_by_support()

# Export
result.to_csv("patterns.csv")
result.to_json("patterns.json")
```

## Bundled datasets

```python
from gpmf.datasets import load_example, load_air_quality

df = load_example()        # 4-row toy dataset
df = load_air_quality()    # UCI Air Quality
```

## CLI

```bash
gradual-mine --list                               # list available algorithms
gradual-mine graank data.csv --min-support 0.5
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs 4
gradual-mine msgp data.csv --min-support 0.5 --cycle-size 12
```
