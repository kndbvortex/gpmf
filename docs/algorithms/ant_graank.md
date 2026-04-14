# ACO-GRAANK

Ant Colony Optimisation heuristic for gradual pattern mining. Trades completeness for scalability on large datasets where exhaustive enumeration is impractical.

## Overview

ACO-GRAANK adapts the ant colony optimisation metaheuristic to the gradual pattern mining task. A population of ants constructs candidate patterns by probabilistically selecting gradual items according to pheromone intensities and a heuristic based on single-item support. High-support patterns reinforce the pheromone trails of their constituent items, guiding future ants toward promising regions of the search space. Pheromone evaporation introduces forgetting, preventing premature convergence.

Support is evaluated using the same bitmap mechanism as GRAANK, so support values are directly comparable between the two algorithms. ACO-GRAANK is the appropriate choice when bitmap memory is not the bottleneck but exhaustive candidate enumeration is too slow.

## Parameters

**min_support** : float, default=0.5
    Minimum pair-based support threshold. Must satisfy 0 < min_support ≤ 1.

**max_iter** : int, default=10
    Number of ACO iterations. More iterations improve recall at the cost of runtime.

**evaporation_factor** : float, default=0.5
    Pheromone evaporation rate ρ. Must satisfy 0 < evaporation_factor < 1. Lower values favour exploitation of known good patterns; higher values favour exploration.

## Support definition

Pair-based, identical to GRAANK:

```
supp(P) = |{(i,j) : i<j, all items of P satisfied by (oi, oj)}| / C(n, 2)
```

## Examples

```python
>>> from gpmf.algorithms.swarm.ant_graank import AntGRAANK
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 2, 3, 4]})
>>> algo = AntGRAANK(min_support=0.5, max_iter=20, evaporation_factor=0.3)
>>> patterns = algo.fit(df).get_patterns()
```

Via CLI:

```bash
gradual-mine ant-graank data.csv --min-support 0.5
```

## Notes

ACO-GRAANK is **not complete**: it may miss frequent patterns, especially in high-dimensional spaces with many infrequent items. Increasing `max_iter` improves recall.

Support values produced by ACO-GRAANK are directly comparable to GRAANK output, since both use the pair-based measure.

## See Also

[GRAANK](graank.md)
    Exact exhaustive version with the same support definition.

## References

[1] Owuor, D. O., Runkler, T., Laurent, A., Orero, J. O., & Menya, E. O. (2021). Ant colony optimization for mining gradual patterns. *International Journal of Machine Learning and Cybernetics*, 12, 2989–3009. https://doi.org/10.1007/s13042-021-01390-w
