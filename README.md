# GPMF: Gradual Pattern Mining Framework

GPMF is a Python library for mining gradual patterns, rules such as "the more A increases, the more B decreases". It brings the main algorithms of the literature under one scikit-learn style interface, so you can switch from one to another by changing a single line.

## Installation

```bash
pip install gradual-mining
```

Wheels are published for Linux, macOS and Windows (x86_64 and arm64). ParaMiner comes with a Rust extension that is compiled into the wheel. If the extension cannot be loaded, GPMF falls back to the pure Python version on its own.

## Quick start

```python
from gpmf.algorithms.graank import GRAANK

model = GRAANK(min_support=0.5)
patterns = model.mine("data.csv")

for p in patterns:
    print(p.to_string(), ":", p.support)
```

Every algorithm has the same interface:

```python
model.fit(data)                # a CSV path, a pandas DataFrame or a GradualDataset
patterns = model.get_patterns()
result = model.get_result()    # patterns plus execution time and metadata
```

Date columns are detected and left out of the mining. Numeric column labels are kept as attribute names, and the DataFrame you pass in is never modified.

### ParaMiner with the Rust extension

```python
from gpmf.algorithms.closed.paraminer_algorithm import ParaMiner

model = ParaMiner(min_support=0.5, use_rust=True, num_threads=4)
patterns = model.mine("data.csv")
```

## Algorithms

| Algorithm | Key | Authors | Reference |
|-----------|-----|---------|-----------|
| **GRAANK** | `graank` | Laurent, Lesot & Rifqi (2009) | [paper](https://doi.org/10.1007/978-3-642-04957-6_33) |
| **GRITE** | `grite` | Di-Jorio, Laurent & Teisseire (2009) | [paper](https://doi.org/10.1007/978-3-642-03915-7_26) |
| **SGrite** | `sgrite` / `sgopt` / `sg1` / `sgb1` / `sgb2` | Tayou Djamegni, Tabueu Fotso & Kenmogne (2021) | [paper](https://doi.org/10.1016/j.mlwa.2021.100068) |
| **ParaMiner** | `paraminer` | Negrevergne, Termier, Rousset & Méhaut (2014) | [paper](https://doi.org/10.1007/s10618-013-0313-2) |
| **ACO-GRAANK** | `ant-graank` | Owuor, Runkler, Laurent, Orero & Menya (2021) | [paper](https://doi.org/10.1007/s13042-021-01390-w) |
| **TGrad** | `tgrad` | Owuor, Laurent & Orero (2019) | [paper](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883) |
| **MSGP** | `msgp` | Lonlac, Doniec, Lujak & Lecoeuche (2020) | [paper](https://doi.org/10.1007/978-3-030-59065-9_16) |
| **GLCM** | `glcm` | Do, Termier, Laurent et al. (2015) | [paper](https://doi.org/10.1007/s10115-014-0749-8) |
| **PGLCM** | `pglcm` | Do, Termier, Laurent et al. (2015) | [paper](https://doi.org/10.1007/s10115-014-0749-8) |

### Row-Column pruning

| Criterion | Works with | Authors |
|-----------|------------|---------|
| Row-Column pruning (`--use-rc-pruning`) | GRITE, SGrite (all variants), GLCM, PGLCM, ParaMiner (Python and Rust) | Kamga Nguifo, Lonlac, Fleury & Mephu Nguifo (2025), [paper](https://doi.org/10.1109/FUZZ62266.2025.11152072) |

The pruning removes, before the longest path is computed, the rows and columns of a precedence matrix that cannot belong to a chain long enough to reach the support threshold. The patterns and their supports stay the same; only the work shrinks.

## Command line

```bash
# list the algorithms
gradual-mine --list

# mine patterns
gradual-mine graank data.csv --min-support 0.5

# save the results
gradual-mine graank data.csv --min-support 0.5 --output results.json --csv results.csv

# run on several cores
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs -1

# Row-Column pruning (algorithms that do not support it refuse the option)
gradual-mine sgrite data.csv --min-support 0.3 --use-rc-pruning

# maximal patterns only (GRAANK)
gradual-mine graank data.csv --min-support 0.5 --maximal
```

Parameters that are specific to one algorithm, such as the cycle size of MSGP or the target column of TGrad, are set from Python.

## Development

```bash
git clone https://github.com/kndbvortex/gpmf
cd gpmf
uv sync --dev
uv run pytest
```

To build the Rust extension locally:

```bash
pip install maturin
maturin develop --release
```

## License

MIT
