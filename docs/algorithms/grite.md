# GRITE

Gradual pattern mining with precedence graphs, where support is the longest chain of rows.

## Overview

GRITE (GRadual ITEmset) represents each pattern as a precedence graph between rows and measures support as the longest path in that graph. Where GRAANK counts every pair of rows that goes in the right direction, GRITE asks how many rows can be lined up in one order that respects the pattern.

The two supports are different measures, so their thresholds cannot be compared (see [the overview](README.md)). GRITE is the natural choice when the chain reading fits your question, or when the pair matrices of GRAANK do not fit in memory.

## Parameters

**min_support** : float, default=0.5
    Minimum length of the longest chain, as a share of the number of rows, with 0 < min_support ≤ 1. A value of 1.0 asks for every row to be ordered by the pattern.

**use_rc_pruning** : bool, default=False
    Apply Row-Column pruning before computing the longest path. The patterns and supports do not change; the graphs get smaller.

**n_jobs** : int, default=1
    Accepted for a uniform interface. GRITE runs on one core.

## Support

Support on chains. Build the graph where i → j when the pair (o_i, o_j) respects every item of P (with a non-strict order: ties are allowed). Then:

```
supp(P) = number of rows on the longest path of the graph / n
```

## Example

```python
import pandas as pd
from gpmf.algorithms.grite import GRITE

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 3, 2, 4]})

for p in GRITE(min_support=0.5).fit(df).get_patterns():
    print(p.to_string(), p.support)
# ['0+', '1-'] 1.0
# ['0-', '1+'] 1.0
# ['0+', '2+'] 0.75
# and the other frequent patterns, in both orientations
```

From the command line:

```bash
gradual-mine grite data.csv --min-support 0.5
gradual-mine grite data.csv --min-support 0.5 --use-rc-pruning
```

## Notes

Each returned pattern has a `path` attribute with the row indices of its longest chain.

At the same numerical threshold, GRITE and GRAANK can return different patterns, because the two supports rank patterns differently. SGrite computes the same kind of support faster, with a strict order instead of a non-strict one.

## See also

[SGrite](sgrite.md): a faster algorithm for chain support.

[GRAANK](graank.md): the same task with support counted on pairs.

[GLCM](glcm.md): closed patterns with the same support and the same tie convention.

## References

[1] Di-Jorio, L., Laurent, A., & Teisseire, M. (2009). Mining Frequent Gradual Itemsets from Large Databases. *Advances in Intelligent Data Analysis VIII (IDA 2009)*, LNCS 5772, 297-308. https://doi.org/10.1007/978-3-642-03915-7_26

[2] Kamga Nguifo, D., Lonlac, J., Fleury, A., & Mephu Nguifo, E. (2025). Mining Faster, Not Harder: A New Criterion for Gradual Pattern Mining. *IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2025)*, 1-6. https://doi.org/10.1109/FUZZ62266.2025.11152072
