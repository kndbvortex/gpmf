---
hide:
  - navigation
  - toc
---

# GPMF: Gradual Pattern Mining Framework

GPMF finds gradual patterns in numerical data, rules such as "the more A increases, the more B decreases". It gathers the main mining algorithms of the literature behind one scikit-learn style interface, so comparing them takes one line.

## Installation

```bash
pip install gradual-mining
```

## Quick start

```python
from gpmf.algorithms.graank import GRAANK

algo = GRAANK(min_support=0.5)
patterns = algo.mine("data.csv")

for p in patterns:
    print(p.to_string(), ":", p.support)
```

## Algorithms

| Algorithm | Key | Family | Support |
|-----------|-----|--------|---------|
| [GRAANK](algorithms/graank.md) | `graank` | Core | Pairs |
| [GRITE](algorithms/grite.md) | `grite` | Core | Chains |
| [SGrite](algorithms/sgrite.md) | `sgrite` | Core | Chains |
| [ParaMiner](algorithms/paraminer.md) | `paraminer` | Closed | Chains |
| [GLCM](algorithms/glcm.md) | `glcm` | Closed | Chains |
| [PGLCM](algorithms/pglcm.md) | `pglcm` | Closed | Chains |
| [ACO-GRAANK](algorithms/ant_graank.md) | `ant-graank` | Swarm | Pairs |
| [TGrad](algorithms/tgrad.md) | `tgrad` | Temporal | Pairs |
| [MSGP](algorithms/msgp.md) | `msgp` | Seasonal | Cycles |
| [EmergingGP](algorithms/emerging_gp.md) | none (Python only) | Contrast | Pairs |

[Get started](getting_started.md){ .md-button .md-button--primary }
[Algorithm overview](algorithms/README.md){ .md-button }
