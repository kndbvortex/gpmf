# SGrite

Improved GRITE using a single bottom-up DAG sweep with Lemma 1 symmetry pruning.

## Overview

SGrite (Simplified GRITE) replaces GRITE's multiple-pass graph processing with a single bottom-up sweep, and exploits a symmetry lemma to halve the search space. It offers four variants that trade DFS against BFS traversal and two support algorithms against each other. In all configurations, SGrite is strictly faster than GRITE on the same data while producing identical or more compact results.

`sgb1` is the recommended default. It combines DFS traversal with Algorithm 3 (which applies Lemma 1 to skip complement patterns) and delivers the best overall performance in the original evaluation.

## Parameters

**min_support** : float, default=0.5
    Minimum chain-based support. Must satisfy 0 < min_support ≤ 1.

**variant** : str, default='sgb1'
    Algorithm variant. One of `'sgb1'`, `'sgb2'`, `'sgopt'`, `'sg1'`.

**use_rc_pruning** : bool, default=False
    Enable Row-Column pruning before longest-path computation. Reduces DAG size on datasets with many tied values. Compatible with all variants.

### Variants

| Key       | Traversal | Support algorithm                  |
| --------- | --------- | ---------------------------------- |
| `sgb1`  | DFS       | Algorithm 3 (Lemma 1), recommended |
| `sgb2`  | BFS       | Algorithm 3 (Lemma 1)              |
| `sgopt` | DFS       | Algorithm 2                        |
| `sg1`   | BFS       | Algorithm 2                        |

Algorithm 3 uses Lemma 1 to avoid processing a pattern when its complement has already been seen, halving the effective search space. Algorithm 2 explores the full space without this shortcut.

## Support definition

Identical to GRITE:

```
supp(P) = length_of_longest_path(DAG_P) / (n − 1)
```

## Examples

```python
>>> from gpmf.algorithms.sgrite import SGrite
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
>>> algo = SGrite(min_support=0.5)
>>> patterns = algo.fit(df).get_patterns()
```

With explicit variant and RC pruning:

```python
>>> algo = SGrite(min_support=0.5, variant='sgopt', use_rc_pruning=True)
>>> patterns = algo.fit(df).get_patterns()
```

Via CLI:

```bash
gradual-mine sgrite data.csv --min-support 0.5
gradual-mine sgb1   data.csv --min-support 0.5
gradual-mine sgopt  data.csv --min-support 0.5 --use-rc-pruning
```

## Notes

SGrite and GRITE share the same support definition. Results are directly comparable between the two algorithms.

`sgb1` and `sgb2` may output fewer patterns than GRITE when Lemma 1 eliminates complement patterns. Both forms of a complementary pair carry equivalent information.

RC pruning (`use_rc_pruning=True`) is independent of the chosen variant and provides additional speedups on data with frequent tied attribute values.

## See Also

[GRITE](grite.md)
    Original algorithm that SGrite improves upon.

[GLCM](glcm.md)
    Closed-pattern miner using the same chain-based support.

## References

[1] Tayou Djamegni, C., Tabueu Fotso Laurent, C., & Kenmogne Edith, B. (2021). A novel algorithm for extracting frequent gradual patterns. *Machine Learning with Applications*, 6, 100068. https://doi.org/10.1016/j.mlwa.2021.100068

[2] Kamga Nguifo, E., Lonlac, J., Fleury, G., & Mephu Nguifo, E. (2025). Row-Column Pruning for Gradual Pattern Mining. *IEEE FUZZ 2025*.
    https://doi.org/10.1109/FUZZ62266.2025.11152072
