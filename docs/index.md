---
hide:
  - navigation
  - toc
---

# GPMF — Gradual Pattern Mining Framework

A unified Python library for gradual pattern mining with a scikit-learn-style API.

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

## Available algorithms

| Algorithm | Key | Category | Support |
|-----------|-----|----------|---------|
| [GRAANK](algorithms/graank.md) | `graank` | Core | Pair-based |
| [GRITE](algorithms/grite.md) | `grite` | Core | Chain-based |
| [SGrite](algorithms/sgrite.md) | `sgrite` | Core | Chain-based |
| [ParaMiner](algorithms/paraminer.md) | `paraminer` | Closed | Chain-based |
| [GLCM](algorithms/glcm.md) | `glcm` | Closed | Chain-based |
| [PGLCM](algorithms/pglcm.md) | `pglcm` | Closed | Chain-based |
| [ACO-GRAANK](algorithms/ant_graank.md) | `ant-graank` | Swarm | Pair-based |
| [TGrad](algorithms/tgrad.md) | `tgrad` | Temporal | Pair-based |
| [MSGP](algorithms/msgp.md) | `msgp` | Seasonal | Periodic |
| [EmergingGP](algorithms/emerging_gp.md) | — | Contrast | Pair-based |

[Get started](getting_started.md){ .md-button .md-button--primary }
[Algorithm overview](algorithms/README.md){ .md-button }
