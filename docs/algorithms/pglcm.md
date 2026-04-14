# PGLCM

Multi-core parallelisation of GLCM. Each starting attribute's subtree is independent, allowing threads to work without synchronisation.

## Overview

PGLCM (Parallel GLCM) dispatches the top-level search subtrees of GLCM across multiple worker threads using `ThreadPoolExecutor`. Because the subtrees rooted at different starting attributes share no mutable state, no locking is needed. The result is a near-linear speedup proportional to the number of independent subtrees, which equals the number of attributes.

PGLCM produces exactly the same patterns as GLCM. It is a pure performance optimisation with no change to the algorithm or the output.

## Parameters

**min_support** : float, default=0.5
    Minimum chain-based support. Must satisfy 0 < min_support ≤ 1.

**n_jobs** : int, default=1
    Number of worker threads. Pass `-1` to use all available cores.

**max_pattern_size** : int, default=0
    Maximum pattern length. Set to 0 for no limit.

**use_rc_pruning** : bool, default=False
    Enable Row-Column pruning. Passed through to each GLCM subtree.

## Support definition

Identical to GLCM:

```
supp(P) = length_of_longest_chain(DAG_P) / n
```

## Examples

```python
>>> from gpmf.algorithms.closed.pglcm import PGLCM
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 3, 2, 4]})
>>> algo = PGLCM(min_support=0.5, n_jobs=-1)
>>> patterns = algo.fit(df).get_patterns()
```

Via CLI:

```bash
gradual-mine pglcm data.csv --min-support 0.5 --n-jobs -1
```

## Notes

Parallelism efficiency is bounded by the number of attributes: a dataset with five attributes has at most five independent subtrees, so `n_jobs > 5` provides no additional benefit.

On small datasets or datasets with few attributes, GLCM may be faster due to thread spawning overhead.

## See Also

[GLCM](glcm.md)
    Single-threaded version producing identical results.

[ParaMiner](paraminer.md)
    Alternative multi-core closed miner with a Rust extension.

## References

[1] Do, T. D. T., Termier, A., Laurent, A., Negrevergne, B., Omidvar-Tehrani, B., & Amer-Yahia, S. (2015). PGLCM: Efficient Parallel Mining of Closed Frequent Gradual Itemsets. *Knowledge and Information Systems*, 43(3), 497–527. https://hal-lirmm.ccsd.cnrs.fr/lirmm-01381085v1
