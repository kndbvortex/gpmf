# GLCM

Single-threaded closed frequent gradual pattern miner adapted from the LCM algorithm. Mining time is linear in the number of closed patterns.

## Overview

GLCM (Gradual LCM) adapts the Linear-time Closed itemset Miner to the gradual setting. It performs a depth-first traversal of the pattern space and applies a closure operator at each node to identify closed patterns directly, without post-processing. The key complexity guarantee is that each closed pattern is visited exactly once, making mining time linear in the output size.

GLCM is the fastest single-core closed gradual pattern miner in GPMF. For multi-core execution, use PGLCM, which parallelises GLCM's independent subtrees.

## Parameters

**min_support** : float, default=0.5
    Minimum chain-based support. Must satisfy 0 < min_support ≤ 1.

**max_pattern_size** : int, default=0
    Maximum pattern length. Set to 0 for no limit. Useful to limit output on datasets with many correlated attributes.

**use_rc_pruning** : bool, default=False
    Enable Row-Column pruning before longest-path computation. Reduces the size of concordance matrices on datasets with many tied values.

## Support definition

Chain-based:

```
supp(P) = length_of_longest_chain(DAG_P) / n
```

Note: GLCM divides by n, whereas GRITE and SGrite divide by n−1. Support values differ slightly at the same threshold.

## Examples

```python
>>> from gpmf.algorithms.closed.glcm import GLCM
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 3, 2, 4]})
>>> algo = GLCM(min_support=0.5)
>>> patterns = algo.fit(df).get_patterns()
```

With pattern size limit and RC pruning:

```python
>>> algo = GLCM(min_support=0.5, max_pattern_size=3, use_rc_pruning=True)
>>> patterns = algo.fit(df).get_patterns()
```

Via CLI:

```bash
gradual-mine glcm data.csv --min-support 0.5
```

## Notes

GLCM and PGLCM produce identical results. PGLCM is preferable when more than one CPU core is available.

RC pruning reduces concordance matrix size before longest-path computation and is compatible with any support threshold.

## See Also

[PGLCM](pglcm.md)
    Multi-core parallelisation of GLCM producing identical results.

[ParaMiner](paraminer.md)
    Alternative closed miner with Rust acceleration.

[SGrite](sgrite.md)
    Non-closed miner using the same chain-based support.

## References

[1] Do, T. D. T., Termier, A., Laurent, A., Negrevergne, B., Omidvar-Tehrani, B., & Amer-Yahia, S. (2015). PGLCM: Efficient Parallel Mining of Closed Frequent Gradual Itemsets. *Knowledge and Information Systems*, 43(3), 497–527. https://hal-lirmm.ccsd.cnrs.fr/lirmm-01381085v1

[2] Kamga Nguifo, E., Lonlac, J., Fleury, G., & Mephu Nguifo, E. (2025). Row-Column Pruning for Gradual Pattern Mining. *IEEE FUZZ 2025*.
    https://doi.org/10.1109/FUZZ62266.2025.11152072
