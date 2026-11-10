# ACO-GRAANK

Ant colony search for gradual patterns: faster than GRAANK on large problems, at the price of completeness.

## Overview

ACO-GRAANK replaces the exhaustive search of GRAANK with an ant colony. Each ant builds a candidate pattern by picking gradual items at random, with probabilities driven by pheromone levels. Patterns with a high support reinforce the pheromone of their items, which steers later ants towards promising regions. Evaporation slowly erases old trails so that the colony does not lock onto its first findings.

Support is computed with the same bitmaps as GRAANK, so the supports of the two algorithms can be compared directly. ACO-GRAANK is useful when the number of candidates makes the exhaustive search too slow.

## Parameters

**min_support** : float, default=0.5
    Minimum support on pairs, with 0 < min_support ≤ 1.

**max_iter** : int, default=10
    Number of iterations of the colony. More iterations find more patterns and take longer.

**evaporation_factor** : float, default=0.5
    Pheromone evaporation rate, strictly between 0 and 1. A low value keeps exploiting the trails already found; a high value explores more.

## Support

Support on pairs, as in GRAANK:

```
supp(P) = |{(i, j) : i < j, the pair (o_i, o_j) respects every item of P}| / C(n, 2)
```

## Example

```python
import pandas as pd
from gpmf.algorithms.swarm.ant_graank import AntGRAANK

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 2, 3, 4]})
patterns = AntGRAANK(min_support=0.5, max_iter=20, evaporation_factor=0.3).fit(df).get_patterns()
```

From the command line:

```bash
gradual-mine ant-graank data.csv --min-support 0.5
```

## Notes

ACO-GRAANK is not complete: it can miss frequent patterns, especially when there are many attributes and many infrequent items. Raising `max_iter` helps. Since the search is random, two runs can return different sets.

## See also

[GRAANK](graank.md): the exact search with the same support.

## Reference

[1] Owuor, D. O., Runkler, T., Laurent, A., Orero, J. O., & Menya, E. O. (2021). Ant colony optimization for mining gradual patterns. *International Journal of Machine Learning and Cybernetics*, 12, 2989-3009. https://doi.org/10.1007/s13042-021-01390-w
