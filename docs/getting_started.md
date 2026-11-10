# Getting started

## Installation

```bash
pip install gradual-mining
```

Wheels are published for Linux, macOS and Windows (x86_64 and arm64).

## Basic usage

All algorithms are used the same way:

```python
import pandas as pd
from gpmf.algorithms.graank import GRAANK

df = pd.read_csv("data.csv")

algo = GRAANK(min_support=0.5)
patterns = algo.fit(df).get_patterns()

for p in patterns:
    print(p.to_string(), ":", p.support)
```

You can also give a CSV path directly:

```python
patterns = GRAANK(min_support=0.5).mine("data.csv")
```

GPMF works on the numeric columns. Date columns are recognised and left out of the mining (TGrad uses them to measure time lags). Numeric column labels are kept as attribute names, and your DataFrame is not modified.

## Which algorithm?

GPMF has two families of support, and their thresholds do not mean the same thing. The [algorithm overview](algorithms/README.md) explains how they relate.

| You want | Use |
|----------|-----|
| The standard baseline, support counted on pairs of rows | GRAANK |
| The longest chain of rows that follows the pattern | GRITE or SGrite |
| A compact output with closed patterns only | GLCM, PGLCM or ParaMiner |
| A fast approximate answer on a large dataset | ACO-GRAANK |
| Co-variations that appear with a time lag | TGrad |
| Co-variations that come back at every cycle | MSGP |
| What distinguishes one dataset from another | EmergingGP |

## Working with results

```python
result = algo.mine_and_get_result(df)

print(result.algorithm)        # 'GRAANK'
print(result.execution_time)   # seconds
print(result.n_patterns)       # number of patterns

result.filter_by_support(0.7)
result.sort_by_support()

result.to_csv("patterns.csv")
result.to_json("patterns.json")
```

## Bundled datasets

```python
from gpmf.datasets import load_example, load_air_quality

df = load_example()        # a toy dataset of four rows
df = load_air_quality()    # the UCI Air Quality dataset
```

## Command line

```bash
gradual-mine --list                                    # list the algorithms
gradual-mine graank data.csv --min-support 0.5
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs 4
gradual-mine sgrite data.csv --min-support 0.3 --use-rc-pruning
gradual-mine graank data.csv --min-support 0.5 --maximal
```

The command line covers the options shared by all algorithms. Algorithm-specific settings, such as the cycle size of MSGP, are set from Python.
