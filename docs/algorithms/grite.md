# GRITE

Precedence-graph approach to gradual pattern mining that measures support via the longest ordered subsequence.

## Overview

GRITE (GRadual ITEms) represents attribute orderings as directed acyclic graphs and computes pattern support as the length of the longest path through each graph. Unlike GRAANK, which counts all valid pairs, GRITE asks for the longest chain of objects that can be totally ordered according to the pattern. This chain-based view scales better to dense, nearly-sorted data and avoids enumerating all O(n²) pairs.

The support definition in GRITE is fundamentally different from GRAANK's. The two measures cannot be compared at the same threshold value. Use GRITE when the chain interpretation is appropriate for your task, or when pair-based algorithms run out of memory.

## Parameters

**min_support** : float, default=0.5
    Minimum chain length relative to n−1. Must satisfy 0 < min_support ≤ 1. A value of 1.0 requires all n objects to be totally ordered by the pattern.

## Support definition

Chain-based. For a dataset of n objects, build a DAG where i→j exists if all items of P are satisfied by (oi, oj). Then:

```
supp(P) = length_of_longest_path(DAG_P) / (n − 1)
```

This is not comparable to pair-based support. Asymptotically, `supp_pair(P) ≥ supp_chain(P)²`, but no exact equivalence exists in general.

## Examples

```python
>>> from gpmf.algorithms.grite import GRITE
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
>>> algo = GRITE(min_support=0.5)
>>> patterns = algo.fit(df).get_patterns()
>>> for p in patterns:
...     print(p.to_string(), p.support)
```

Via CLI:

```bash
gradual-mine grite data.csv --min-support 0.5
```

## Notes

The `path` attribute of each returned `GradualPattern` stores the row indices forming the longest chain.

GRITE may produce fewer or different patterns than GRAANK at the same numerical threshold, because the two support measures rank patterns differently. For an improved implementation of the same chain-based measure, prefer SGrite.

## See Also

[SGrite](sgrite.md)
    Faster drop-in replacement for GRITE using a single DAG sweep.

[GRAANK](graank.md)
    Pair-based alternative with a different support definition.

[GLCM](glcm.md)
    Closed-pattern miner using the same chain-based support.

## References

[1] Di-Jorio, L., Laurent, A., & Teisseire, M. (2009). Mining Frequent Gradual Itemsets from Large Databases. *Advances in Intelligent Data Analysis VIII*, IDA 2009, LNCS 5772. https://doi.org/10.1007/978-3-642-03915-7_15
