# GPMF — Gradual Pattern Mining Framework

A Python library for gradual pattern mining with a unified scikit-learn-style API.

## Installation

```bash
pip install gradual-mining
```

Pre-built wheels are available for Linux, macOS and Windows (x86_64 and arm64). The ParaMiner algorithm includes an optional Rust extension for multi-core acceleration — it is compiled into the wheel automatically and falls back to a pure-Python implementation if unavailable.

## Quick start

```python
from gpmf.algorithms.graank import GRAANK

model = GRAANK(min_support=0.5)
patterns = model.mine("data.csv")

for p in patterns:
    print(p.to_string(), ":", p.support)
```

All algorithms share the same interface:

```python
model.fit(data)              # data can be a CSV path, DataFrame, or GradualDataset
patterns = model.get_patterns()
result   = model.get_result()  # includes execution time, metadata
```

### ParaMiner with Rust acceleration

```python
from gpmf.algorithms.closed.paraminer_algorithm import ParaMiner

model = ParaMiner(min_support=0.5, use_rust=True, num_threads=4)
patterns = model.mine("data.csv")
```

## Available algorithms

| Algorithm | Key | Authors | Reference |
|-----------|-----|---------|-----------|
| **GRAANK** | `graank` | Laurent, Lesot & Rifqi (2010) | [paper](https://www.lirmm.fr/~laurent/POLYTECH/IG4/DM/RessourcesProjet/LaurentLesotRifqi.pdf) |
| **GRITE** | `grite` | Di-Jorio, Laurent & Teisseire (2009) | [paper](https://hal.inrae.fr/hal-02592797v1) |
| **SGrite** | `sgrite` / `sgopt` / `sg1` / `sgb1` / `sgb2` | Tayou Djamegni, Tabueu Fotso & Kenmogne (2021) | [paper](https://doi.org/10.1016/j.mlwa.2021.100068) |
| **ParaMiner** | `paraminer` | Négrevergne, Termier, Rousset & Méhaut (2014) | [paper](https://doi.org/10.1007/s10618-013-0313-2) |
| **ACO-GRAANK** | `ant-graank` | Owuor, Laurent & Orero (2019) | [paper](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883) |
| **TGrad** | `tgrad` | Owuor, Laurent & Orero (2019) | [paper](https://doi.org/10.1109/FUZZ-IEEE.2019.8858883) |
| **MSGP** | `msgp` | Lonlac, Doniec, Lujak & Lecoeuche (2020) | [paper](https://doi.org/10.1007/978-3-030-59065-9_16) |
| **GLCM** | `glcm` | Do, Termier, Laurent et al. (2015) | [paper](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01381085v1) |
| **PGLCM** | `pglcm` | Do, Termier, Laurent et al. (2015) | [paper](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01381085v1) |

### Pruning criterion

| Criterion | Applies to | Authors |
|-----------|-----------|---------|
| Row–Column pruning (`--use-rc-pruning`) | GRITE, SGrite, GLCM | Kamga Nguifo, Lonlac, Fleury & Mephu Nguifo (2025) — [paper](https://doi.org/10.1109/FUZZ62266.2025.11152072) |

## CLI

```bash
# List available algorithms
gradual-mine --list

# Mine patterns
gradual-mine graank data.csv --min-support 0.5

# Save results
gradual-mine graank data.csv --min-support 0.5 --output results.json --csv results.csv

# Parallel execution
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs -1
```

## Development

```bash
git clone https://github.com/your-org/gpmf
cd gpmf
uv sync --dev
uv run pytest
```

To build with the Rust extension locally:

```bash
pip install maturin
maturin develop --release
```

## License

MIT
