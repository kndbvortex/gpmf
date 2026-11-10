# GLCM

Closed frequent gradual patterns on one core, adapted from the LCM algorithm.

## Overview

GLCM (Gradual LCM) brings the Linear-time Closed itemset Miner to gradual patterns. It walks the pattern space depth-first and applies a closure operator at each node, so it outputs closed patterns directly instead of filtering them afterwards. Each closed pattern is visited once, which makes the running time linear in the size of the output.

GLCM is the single-core closed miner of GPMF. PGLCM runs the same search on several cores.

## Parameters

**min_support** : float, default=0.5
    Minimum chain support, with 0 < min_support ≤ 1.

**max_pattern_size** : int, default=0
    Largest number of items in a pattern. 0 means no limit. Useful on data with many correlated attributes.

**use_rc_pruning** : bool, default=False
    Apply Row-Column pruning before computing the longest chain. The result does not change.

## Support

Support on chains with a non-strict order, as in GRITE:

```
supp(P) = number of rows on the longest chain / n
```

## Example

```python
import pandas as pd
from gpmf.algorithms.closed.glcm import GLCM

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 3, 2, 4]})

for p in GLCM(min_support=0.5).fit(df).get_patterns():
    print(p.to_string(), p.support)
# ['0+', '1-'] 1.0
# ['0+', '1-', '2+'] 0.75
# ['0+', '1-', '2-'] 0.5

# with a size limit and Row-Column pruning
patterns = GLCM(min_support=0.5, max_pattern_size=3, use_rc_pruning=True).fit(df).get_patterns()
```

From the command line:

```bash
gradual-mine glcm data.csv --min-support 0.5
```

## Notes

GLCM and PGLCM return the same patterns. PGLCM is the better choice when several cores are free and the dataset has enough attributes to split the work.

When Row-Column pruning leaves fewer rows than the minimum chain length, GLCM skips the support computation of that candidate.

## See also

[PGLCM](pglcm.md): the same search on several cores.

[ParaMiner](paraminer.md): closed patterns with a Rust implementation.

[GRITE](grite.md): all frequent patterns with the same support.

## References

[1] Do, T. D. T., Termier, A., Laurent, A., Negrevergne, B., Omidvar-Tehrani, B., & Amer-Yahia, S. (2015). PGLCM: Efficient Parallel Mining of Closed Frequent Gradual Itemsets. *Knowledge and Information Systems*, 43(3), 497-527. https://doi.org/10.1007/s10115-014-0749-8

[2] Kamga Nguifo, D., Lonlac, J., Fleury, A., & Mephu Nguifo, E. (2025). Mining Faster, Not Harder: A New Criterion for Gradual Pattern Mining. *IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2025)*, 1-6. https://doi.org/10.1109/FUZZ62266.2025.11152072
