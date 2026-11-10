# SGrite

A faster GRITE: one bottom-up sweep of the graph, and a symmetry lemma that halves the search.

## Overview

SGrite (Simplified GRITE) replaces the several passes of GRITE over the precedence graph with a single bottom-up sweep. It also uses a symmetry lemma: a pattern and its complement have the same support, so only one of the two needs to be processed. The four variants combine a depth-first or breadth-first traversal with one of two support algorithms.

`sgb1` is the default and the variant we recommend. It explores depth-first and applies the lemma, and it was the fastest in the original evaluation.

## Parameters

**min_support** : float, default=0.5
    Minimum chain support, with 0 < min_support ≤ 1.

**variant** : str, default='sgb1'
    One of `'sgb1'`, `'sgb2'`, `'sgopt'`, `'sg1'`.

**use_rc_pruning** : bool, default=False
    Apply Row-Column pruning before computing the longest path. It works with every variant and leaves the result unchanged.

### Variants

| Key | Traversal | Support algorithm |
| --- | --------- | ----------------- |
| `sgb1` | depth-first | Algorithm 3 (with the symmetry lemma), recommended |
| `sgb2` | breadth-first | Algorithm 3 (with the symmetry lemma) |
| `sgopt` | depth-first | Algorithm 2 |
| `sg1` | breadth-first | Algorithm 2 |

Algorithm 3 skips a pattern when its complement has already been processed. Algorithm 2 explores both.

## Support

The same longest-chain support as GRITE, divided by the number of rows, but with a strict order: two rows with the same value do not follow each other.

```
supp(P) = number of rows on the longest path of the graph / n
```

## Example

```python
import pandas as pd
from gpmf.algorithms.sgrite import SGrite

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
patterns = SGrite(min_support=0.5).fit(df).get_patterns()

# another variant, with Row-Column pruning
patterns = SGrite(min_support=0.5, variant='sgopt', use_rc_pruning=True).fit(df).get_patterns()
```

From the command line:

```bash
gradual-mine sgrite data.csv --min-support 0.5
gradual-mine sgb1 data.csv --min-support 0.5
gradual-mine sgopt data.csv --min-support 0.5 --use-rc-pruning
```

## Notes

On data without ties, SGrite and GRITE give the same supports. With ties they can differ, since GRITE uses a non-strict order and SGrite a strict one.

`sgb1` and `sgb2` can return fewer patterns than GRITE because they keep one pattern of each complementary pair. Both forms carry the same information.

Row-Column pruning helps most on large datasets where many rows cannot be part of a long enough chain.

## See also

[GRITE](grite.md): the algorithm SGrite improves on.

[GLCM](glcm.md): closed patterns with chain support.

## References

[1] Tayou Djamegni, C., Tabueu Fotso, L. C., & Kenmogne, E. B. (2021). A novel algorithm for extracting frequent gradual patterns. *Machine Learning with Applications*, 5, 100068. https://doi.org/10.1016/j.mlwa.2021.100068

[2] Kamga Nguifo, D., Lonlac, J., Fleury, A., & Mephu Nguifo, E. (2025). Mining Faster, Not Harder: A New Criterion for Gradual Pattern Mining. *IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2025)*, 1-6. https://doi.org/10.1109/FUZZ62266.2025.11152072
