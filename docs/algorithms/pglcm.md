# PGLCM

GLCM on several cores. The subtrees of the search are independent, so each worker mines its own.

## Overview

PGLCM (Parallel GLCM) hands the top-level subtrees of the GLCM search to a pool of worker processes. Each subtree starts from a different attribute and shares nothing that can change, so the workers need no locking. GPMF uses processes rather than threads because the search is pure Python, and threads would be serialised by the interpreter lock. Each worker receives the encoded matrices once, when it starts.

PGLCM returns exactly the same patterns as GLCM. Only the running time changes.

## Parameters

**min_support** : float, default=0.5
    Minimum chain support, with 0 < min_support ≤ 1.

**n_jobs** : int, default=1
    Number of worker processes. `-1` uses every available core.

**max_pattern_size** : int, default=0
    Largest number of items in a pattern. 0 means no limit.

**use_rc_pruning** : bool, default=False
    Apply Row-Column pruning in every subtree.

## Support

The same as GLCM:

```
supp(P) = number of rows on the longest chain / n
```

## Example

```python
import pandas as pd
from gpmf.algorithms.closed.pglcm import PGLCM

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [1, 3, 2, 4]})

if __name__ == "__main__":
    patterns = PGLCM(min_support=0.5, n_jobs=-1).fit(df).get_patterns()
```

Keep the call under `if __name__ == "__main__":` in scripts. Worker processes re-import the main module on Windows and macOS.

From the command line:

```bash
gradual-mine pglcm data.csv --min-support 0.5 --n-jobs -1
```

## Notes

The useful number of workers is bounded by the number of attributes: with five attributes there are at most five subtrees, so more than five workers bring nothing.

On small datasets, or with few attributes, starting the processes can cost more than it saves, and GLCM is faster.

## See also

[GLCM](glcm.md): the single-core version, with the same output.

[ParaMiner](paraminer.md): another multi-core closed miner, written in Rust.

## Reference

[1] Do, T. D. T., Termier, A., Laurent, A., Negrevergne, B., Omidvar-Tehrani, B., & Amer-Yahia, S. (2015). PGLCM: Efficient Parallel Mining of Closed Frequent Gradual Itemsets. *Knowledge and Information Systems*, 43(3), 497-527. https://doi.org/10.1007/s10115-014-0749-8
