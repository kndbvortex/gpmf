# ParaMiner

A generic parallel miner of closed frequent patterns, here applied to gradual patterns.

## Overview

ParaMiner explores the pattern space depth-first and jumps from one closed pattern to the next with a closure operator. At every node it shrinks the dataset it works on (EL-reduction), so deeper nodes cost less. Independent subtrees are spread across cores without any locking.

For gradual patterns, ParaMiner is sound and complete: it returns exactly the closed frequent patterns. GPMF ships a Rust version, compiled into the wheel, and a pure Python version used when the extension is missing. In the Rust version, the transaction sets are stored as bitsets and the candidate tests of a node run in parallel. Both versions support Row-Column pruning.

ParaMiner counts support on chains, like GRITE and SGrite.

## Parameters

**min_support** : float, default=0.5
    Minimum chain support, with 0 < min_support ≤ 1.

**use_rust** : bool, default=True
    Use the Rust extension when it is available, and fall back to Python otherwise.

**num_threads** : int or None, default=None
    Number of threads of the Rust version. `None` lets the extension choose.

**use_rc_pruning** : bool, default=False
    Apply Row-Column pruning when computing supports. The result does not change.

**verbose** : bool, default=False
    Print progress information.

## Support

Support on chains, with a strict order where ties are broken by the row index, as in the original ParaMiner implementation:

```
supp(P) = max{ |T*| : T* is a subset of T totally ordered by P }
```

ParaMiner reports this support as a number of rows. Divide it by the number of rows to compare it with the other chain miners.

## Example

```python
import pandas as pd
from gpmf.algorithms.closed.paraminer_algorithm import ParaMiner

df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [2, 3, 4, 5]})

for p in ParaMiner(min_support=0.5, num_threads=4).fit(df).get_patterns():
    print(p.to_string(), p.support)
# ['0+', '1-', '2+'] 4
# ['1+', '2-'] 4
```

To force the Python version, pass `use_rust=False`. To check that the extension is installed:

```python
from gpmf.algorithms.closed.paraminer import RUST_AVAILABLE
print(RUST_AVAILABLE)
```

From the command line:

```bash
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs 4
gradual-mine paraminer data.csv --min-support 0.5 --use-rc-pruning
```

## Notes

A pattern is closed when no larger pattern has the same support. The closed patterns are a compact and lossless summary of all frequent patterns.

The extension is built with maturin for the `abi3-py39` target, so one wheel works for every Python version from 3.9.

Because ties are broken by the row index, reordering the rows of a dataset with ties can change the result. On data without ties this does not happen.

## See also

[GLCM](glcm.md): closed patterns on a single core.

[PGLCM](pglcm.md): closed patterns on several cores, in pure Python.

[SGrite](sgrite.md): all frequent patterns with chain support.

## References

[1] Negrevergne, B., Termier, A., Rousset, M.-C., & Méhaut, J.-F. (2014). ParaMiner: A Generic Pattern Mining Algorithm for Multi-Core Architectures. *Data Mining and Knowledge Discovery*, 28(3), 593-633. https://doi.org/10.1007/s10618-013-0313-2

[2] Kamga Nguifo, D., Lonlac, J., Fleury, A., & Mephu Nguifo, E. (2025). Mining Faster, Not Harder: A New Criterion for Gradual Pattern Mining. *IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2025)*, 1-6. https://doi.org/10.1109/FUZZ62266.2025.11152072
