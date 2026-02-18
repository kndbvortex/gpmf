# GPMF (Gradual Pattern Mining Framework)

A unified Python package for gradual pattern mining with multiple algorithms and a clean, SPMF-like interface.

## Features

- **Unified API** - Simple interface inspired by SPMF
- **Multiple algorithms** - GRAANK, GRITE, ParaMiner, ACO-GRAANK, TGrad, MSGP, GLCM, PGLCM


## Installation

```bash
uv sync
```

## Quick Start

```python
from gradual_mining.algorithms.closed.paraminer_algorithm import ParaMiner

miner = ParaMiner(min_support=0.5, num_threads=4)
miner.fit('data.csv')

patterns = miner.get_patterns()

for pattern in patterns:
    print(f"{pattern.to_string()} : {pattern.support}")
```

ParaMiner can also be used directly, with optional Rust acceleration and multi-threading:

```python
from gradual_mining.algorithms.closed.paraminer_algorithm import ParaMiner

miner = ParaMiner(min_support=0.5, use_rust=True, num_threads=4)
miner.fit('data.csv')
patterns = miner.get_patterns()
```

Or via its dedicated CLI:

```bash
python -m gradual_mining.algorithms.closed.paraminer data.csv 0.5
```


## Available Algorithms

| Name | CLI key | Authors | Paper                                                                                                                                                        |
|------|---------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **GRAANK** | `graank` | Laurent, Lesot & Rifqi (2010) | [GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets](https://www.lirmm.fr/~laurent/POLYTECH/IG4/DM/RessourcesProjet/LaurentLesotRifqi.pdf) |
| **GRITE** | `grite` | Di-Jorio, Laurent & Teisseire (2009) | [Mining Frequent Gradual Itemsets from Large Databases](https://hal.inrae.fr/hal-02592797v1)                                                                 |
| **SGrite** | `sgrite`, `sgopt`, `sg1`, `sgb1`, `sgb2` | Tayou Djamegni, Tabueu Fotso & Kenmogne (2021) | [A novel algorithm for extracting frequent gradual patterns](https://doi.org/10.1016/j.mlwa.2021.100068)                                                     |
| **ParaMiner** | `paraminer` | Négrevergne, Termier, Rousset & Méhaut (2014) | [Para Miner: a generic pattern mining algorithm for multi-core architectures](https://doi.org/10.1007/s10618-013-0313-2)                                     |
| **ACO-GRAANK** | `ant-graank`, `aco-graank` | Owuor, Laurent & Orero (2019) | [Mining Fuzzy-Temporal Gradual Patterns](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883)                                                                     |
| **TGrad** | `tgrad`, `t-graank`, `temporal` | Owuor, Laurent & Orero (2019) | [Mining Fuzzy-Temporal Gradual Patterns](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883)                                                                     |
| **MSGP** | `msgp`, `seasonal` | Lonlac, Doniec, Lujak & Lecoeuche (2020) | [Mining Frequent Seasonal Gradual Patterns](https://doi.org/10.1007/978-3-030-59065-9_16)                                                                    |
| **GLCM** | `glcm` | Do, Termier, Laurent, Negrevergne, Jeudy & Gacias (2015) | [PGLCM: Efficient Parallel Mining of Closed Frequent Gradual Itemsets](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01381085v1)                                      |
| **PGLCM** | `pglcm` | Do, Termier, Laurent, Negrevergne, Jeudy & Gacias (2015) | [PGLCM: Efficient Parallel Mining of Closed Frequent Gradual Itemsets](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01381085v1)                                      |

## Condition

The following works propose pruning criteria that can be integrated into the algorithms above to reduce the search space and execution time without altering the output.

| Criterion | Applies to | Authors | Paper |
|-----------|-----------|---------|-------|
| **Row–Column pruning** — keep transaction *t* iff `ML_t + MC_t ≥ minsup − 1` | GRITE, ParaMiner | Kamga Nguifo, Lonlac, Fleury & Mephu Nguifo (2025) | [Mining Faster, Not Harder: A New Criterion for Gradual Pattern Mining](https://doi.org/10.1109/FUZZ62266.2025.11152072) |

## Command-Line Interface

### Generic CLI (all algorithms)

```bash
# List available algorithms
gradual-mine --list

# Mine patterns
gradual-mine graank data.csv --min-support 0.5 --output results.json

# Parallel execution
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs -1

# Disable Row–Column pruning (Kamga Nguifo et al., 2025) for comparison
gradual-mine sgrite data.csv --min-support 0.5 --use-rc-pruning false
```

## Development

```bash
uv sync --dev
uv run pytest
uv run black gradual_mining/
uv run mypy gradual_mining/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
