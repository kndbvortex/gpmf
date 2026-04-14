# ParaMiner

Generic parallel algorithm for mining closed frequent gradual patterns using a closure operator and EL-reduction for dataset compression.

## Overview

ParaMiner is a depth-first closed pattern miner built on the theory of strongly accessible set systems. Its core contribution is a novel dataset reduction technique (EL-reduction) that shrinks the working dataset at each node of the search tree, combined with a parallel execution strategy that distributes independent subtrees across CPU cores without synchronisation.

Applied to gradual patterns, ParaMiner provides formal soundness and completeness guarantees: it returns exactly the set of closed frequent gradual patterns and nothing else. Mining time is linear in the number of closed patterns. An optional Rust extension compiled via PyO3 provides approximately a 10× speedup over the pure Python implementation on large datasets.

ParaMiner uses the same chain-based (gradual) support as GRITE and SGrite. It is the most efficient closed gradual pattern miner in GPMF.

## Parameters

**min_support** : float, default=0.5
    Minimum chain-based support. Must satisfy 0 < min_support ≤ 1.

**use_rust** : bool, default=True
    Use the Rust extension if available. Falls back to pure Python automatically if the extension is not installed.

**num_threads** : int, default=1
    Number of parallel threads. Effective only when `use_rust=True`.

## Support definition

Chain-based (gradual support):

```
supp(P) = max{ |T*| : T* ⊆ T, T* totally ordered by P } / |T|
```

## Examples

```python
>>> from gpmf.algorithms.closed.paraminer_algorithm import ParaMiner
>>> import pandas as pd
>>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1], 'C': [2, 3, 4, 5]})
>>> algo = ParaMiner(min_support=0.5, use_rust=False)
>>> patterns = algo.fit(df).get_patterns()
```

With Rust acceleration:

```python
>>> algo = ParaMiner(min_support=0.5, use_rust=True, num_threads=4)
>>> patterns = algo.fit(df).get_patterns()
```

Check Rust availability:

```python
>>> from gpmf.algorithms.closed.paraminer import RUST_AVAILABLE
>>> print(RUST_AVAILABLE)
True
```

Via CLI:

```bash
gradual-mine paraminer data.csv --min-support 0.5 --n-jobs 4
```

## Notes

**Closed patterns** are patterns where no strict super-pattern has the same support. They form a compact lossless representation of the full frequent pattern set.

The Rust extension is compiled at build time via Maturin with `abi3-py39` for broad wheel compatibility. If it is absent, the Python fallback is used transparently.

Thread efficiency depends on the number of independent subtrees, which is bounded by the number of attributes.

## See Also

[GLCM](glcm.md)
    Faster than ParaMiner on single-core for medium-sized datasets.

[PGLCM](pglcm.md)
    Parallel GLCM; an alternative multi-core closed pattern miner.

[SGrite](sgrite.md)
    Non-closed miner using the same chain-based support.

## References

[1] Negrevergne, B., Termier, A., Rousset, M.-C., & Mehaut, J.-F. (2014). ParaMiner: A Generic Pattern Mining Algorithm for Multi-Core Architectures. *Data Mining and Knowledge Discovery*, 28(3), 593–633. https://doi.org/10.1007/s10618-013-0313-2
