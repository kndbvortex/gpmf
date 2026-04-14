# GRAANK

Apriori-like gradual pattern mining using bitmap concordance matrices and rank correlations.

## Overview

GRAANK (GRAdual rANKing) extracts frequent gradual patterns from numerical datasets by encoding attribute orderings as binary concordance matrices and combining them with bitwise AND operations. It generalises the Apriori principle to the gradual setting: a pattern is frequent if enough object pairs satisfy all its ordering constraints simultaneously.

GRAANK is the standard baseline for gradual pattern mining. Its pair-based support measure is shared by ACO-GRAANK, TGrad, and EmergingGP, making results directly comparable across these algorithms. For datasets where memory is not a constraint, GRAANK is the recommended starting point.

## Parameters

**min_support** : float, default=0.5
    Minimum fraction of valid object pairs. Must satisfy 0 < min_support ≤ 1.

**n_jobs** : int, default=1
    Reserved for future parallelisation. Currently unused.

## Support definition

Pair-based. For a dataset of n objects and a pattern P:

```
supp(P) = |{(i,j) : i<j, all items of P satisfied by (oi, oj)}| / C(n, 2)
```

A pair (oi, oj) satisfies item `Ak+` if `oi.Ak < oj.Ak`, and `Ak-` if `oi.Ak > oj.Ak`.

## Examples

```python
>>> from gpmf.algorithms.graank import GRAANK
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 2, 3, 4]})
>>> algo = GRAANK(min_support=0.5)
>>> patterns = algo.fit(df).get_patterns()
>>> for p in patterns:
...     print(p.to_string(), p.support)
['A+', 'C+'] 1.0
['A+', 'B-'] 1.0
['B-', 'C+'] 1.0
```

Via the unified miner:

```python
>>> from gpmf import GradualPatternMiner
>>> miner = GradualPatternMiner('graank', 'data.csv', min_support=0.5)
>>> patterns = miner.mine()
```

Via CLI:

```bash
gradual-mine graank data.csv --min-support 0.5
```

## Notes

GRAANK returns only **maximal** frequent gradual patterns. Sub-patterns are pruned as longer ones are discovered, following the Apriori anti-monotony property.

A pattern and its complement `{A+, B-}` / `{A-, B+}` represent identical object orderings. Only one canonical form appears in the output.

Memory scales as O(n²) per gradual item, which limits practical use to datasets of roughly 10 000 rows and 20 attributes. For larger datasets, consider GRITE or SGrite (chain-based, no n² memory footprint). For closed patterns, prefer GLCM or ParaMiner.

## See Also

[GRITE](grite.md)
    Chain-based alternative; same conceptual task, different support definition.

[ACO-GRAANK](ant_graank.md)
    Heuristic extension of GRAANK for large datasets.

[EmergingGP](emerging_gp.md)
    Uses GRAANK internally to contrast two datasets.

## References

[1] Laurent, A., Lesot, M.-J., & Rifqi, M. (2009). GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. *Flexible Query Answering Systems*, LNCS 5822, 382–393. https://doi.org/10.1007/978-3-642-04957-6_33
