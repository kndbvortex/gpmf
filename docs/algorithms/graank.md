# GRAANK

Level-wise mining of gradual patterns with bitmap concordance matrices.

## Overview

GRAANK (GRAdual rANKing) turns each gradual item into a binary matrix that says, for every pair of rows, whether the pair goes in the right direction. Combining items is then a bitwise AND of their matrices, and the support of a pattern is the share of pairs that survive. The search follows the Apriori principle: a pattern can only be frequent if its sub-patterns are.

GRAANK is the usual baseline. ACO-GRAANK, TGrad and EmergingGP count support the same way, so their results can be compared with it directly. If your data fit in memory, it is a good place to start.

## Parameters

**min_support** : float, default=0.5
    Minimum share of pairs of rows that respect the pattern, with 0 < min_support ≤ 1.

**maximal** : bool, default=False
    Keep only maximal patterns. A pattern is dropped when it, or its complement, is a strict subset of another pattern of the output.

**n_jobs** : int, default=1
    Accepted for a uniform interface. GRAANK itself runs on one core.

## Support

Support on pairs. For n rows and a pattern P:

```
supp(P) = |{(i, j) : i < j, the pair (o_i, o_j) respects every item of P}| / C(n, 2)
```

A pair (o_i, o_j) respects `Ak+` when `o_i.Ak < o_j.Ak`, and `Ak-` when `o_i.Ak > o_j.Ak`.

## Example

```python
import pandas as pd
from gpmf.algorithms.graank import GRAANK

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 2, 3, 4]})

for p in GRAANK(min_support=0.5, maximal=True).fit(df).get_patterns():
    print(p.to_string(), p.support)
# two patterns with support 1.0: {0+, 1-, 2+} and its complement {0-, 1+, 2-}
```

The order of the items inside a pattern can change from one run to the next; the pattern is the same set.

Without `maximal=True`, the output also contains the pairs `{0+, 1-}`, `{0+, 2+}` and so on, because GRAANK keeps a pattern when it was found in one orientation and its extension in the other.

Through the unified miner:

```python
from gpmf import GradualPatternMiner

patterns = GradualPatternMiner('graank', 'data.csv', min_support=0.5).mine()
```

From the command line:

```bash
gradual-mine graank data.csv --min-support 0.5
gradual-mine graank data.csv --min-support 0.5 --maximal
```

## Notes

By default GRAANK behaves as published: a sub-pattern is removed once a longer pattern is found, but a pattern can stay when only its complement is contained in a longer one. Use `maximal=True` to get the maximal set, which is what so4gp returns.

Each gradual item needs an n × n matrix, so memory grows with the square of the number of rows. In practice this limits GRAANK to roughly 10 000 rows and 20 attributes. For larger data, GRITE and SGrite count support on chains and avoid this cost. For closed patterns, use GLCM or ParaMiner.

## See also

[GRITE](grite.md): the same task with support counted on chains.

[ACO-GRAANK](ant_graank.md): a faster, approximate search with the same support.

[EmergingGP](emerging_gp.md): uses GRAANK to compare two datasets.

## Reference

[1] Laurent, A., Lesot, M.-J., & Rifqi, M. (2009). GRAANK: Exploiting Rank Correlations for Extracting Gradual Itemsets. *Flexible Query Answering Systems (FQAS 2009)*, LNCS 5822, 382-393. https://doi.org/10.1007/978-3-642-04957-6_33
