# TGrad

Temporal extension of GRAANK that discovers gradual patterns with a fuzzy time lag between co-varying attributes.

## Overview

TGrad (Temporal Gradual patterns) addresses datasets where co-variations between attributes are not simultaneous but delayed. It transforms the dataset by pairing each observation with the observation recorded `s` time steps later, then runs GRAANK on each transformed version. Patterns that survive across multiple time steps are assigned a fuzzy time lag and a representativeness score measuring their stability over time.

A typical output pattern reads: "the more exercise, the less stress approximately one month later."

TGrad requires at least one datetime column in the dataset to derive time differences. If no datetime column is present, no temporal transformation is applied.

## Parameters

**min_support** : float, default=0.5
    Minimum pair-based support within each transformed dataset. Must satisfy 0 < min_support ≤ 1.

**target_col** : int, default=0
    Index of the datetime column in the dataset. Used to compute time differences between rows.

**min_rep** : float, default=0.5
    Minimum representativeness. A pattern must hold at this fraction of evaluated time steps to be included in the output.

**n_jobs** : int, default=1
    Number of parallel workers for evaluating time steps.

## Support definition

Pair-based, evaluated on temporally shifted pairs:

```
supp(P, step) = pair-based support of P on the dataset shifted by `step` time units
```

## Examples

```python
>>> from gpmf.algorithms.temporal.tgrad import TGrad
>>> import pandas as pd
>>> df = pd.DataFrame({
...     'date': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03',
...                             '2021-01-04', '2021-01-05']),
...     'A': [1, 2, 3, 4, 5],
...     'B': [5, 4, 3, 2, 1],
... })
>>> algo = TGrad(min_support=0.5, target_col=0, min_rep=0.5)
>>> patterns = algo.fit(df).get_patterns()
>>> for p in patterns:
...     print(p)
```

Via CLI:

```bash
gradual-mine tgrad data.csv --min-support 0.5 --target-col 0
```

## Output

Returns `TemporalGradualPattern` objects extending `GradualPattern` with a list of `TimeLag` instances. Each `TimeLag` carries a `timestamp`, a `support`, and a direction `sign` (`+`, `-`, or `~`).

## Notes

`min_rep` controls pattern stability: a pattern that holds only at one specific lag receives low representativeness and is filtered out if `min_rep` is above the corresponding threshold.

`n_jobs > 1` parallelises evaluation across time steps and scales linearly with the number of distinct lags.

## See Also

[GRAANK](graank.md)
    Non-temporal baseline that TGrad extends.

[MSGP](msgp.md)
    Alternative for datasets with a known periodic structure.

## References

[1] Owuor, D., Laurent, A., & Orero, J. (2019). Mining Fuzzy-Temporal Gradual Patterns. *IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2019)*. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883
