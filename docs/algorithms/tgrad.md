# TGrad

Gradual patterns with a time lag: "the more A, the less B, about one month later".

## Overview

TGrad (Temporal Gradual patterns, also called T-GRAANK) looks for co-variations that are delayed rather than simultaneous. For a time step s, it pairs each value of the target attribute with the values of the other attributes s rows later, and runs GRAANK on this shifted table. It repeats this for every step. A pattern found at a step gets a fuzzy time lag, computed from the time differences between the paired rows.

A typical result reads: "the more exercise, the less stress, about one month later".

TGrad needs a time column to measure the lags. GPMF detects it on its own, and it can hold numbers or date strings such as `2021-01-31`. Without a time column, no lag is computed.

## Parameters

**min_support** : float, default=0.5
    Minimum support on pairs in each shifted table, with 0 < min_support ≤ 1.

**target_col** : int, default=0
    Index of the target attribute, counting every column of the data, the time column included. The target keeps its own rows and the other attributes are shifted. Every pattern contains the target.

**min_rep** : float, default=0.5
    Minimum representativeness, which bounds how many steps are tried: a step s is used while the shifted table still holds at least this share of the rows.

**n_jobs** : int, default=1
    Number of worker processes used to evaluate the steps.

## Support

Support on pairs, computed on the table shifted by s steps:

```
supp(P, s) = support on pairs of P in the table where the attributes other than the target are shifted by s rows
```

## Example

```python
import pandas as pd
from gpmf.algorithms.temporal.tgrad import TGrad

df = pd.DataFrame({
    'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06'],
    'A': [1, 2, 3, 4, 5, 6],
    'B': [6, 5, 4, 3, 2, 1],
    'C': [1, 3, 2, 5, 4, 6],
})

for p in TGrad(min_support=0.5, target_col=1, min_rep=0.5).fit(df).get_patterns():
    print(p.to_string(), p.support, [lag.to_string() for lag in p.time_lags])
# ['2-', '1+'] 1.0 ['~86400.0']
# and the other patterns, one line per pattern and time step
```

Here column 1 (`A`) is the target, and every pattern contains `1+`. The lag `~86400.0` is one day, in seconds. The order of the items inside a pattern can change from one run to the next.

## Output

TGrad returns `TemporalGradualPattern` objects. They extend `GradualPattern` with a list of `TimeLag` objects, each with a `timestamp` (in the unit of the time column, seconds for dates), a `support` and a `sign` (`+`, `-` or `~` for "about").

A pattern is reported relative to the target and once per time step where it is frequent, so the same items can appear with several lags.

## Notes

The command line does not expose `target_col`: use TGrad from Python.

Setting `n_jobs` above 1 evaluates the steps in parallel. It pays off when there are many steps to try.

## See also

[GRAANK](graank.md): the algorithm TGrad runs on each shifted table.

[MSGP](msgp.md): for data with a known cycle, such as days or years.

## Reference

[1] Owuor, D., Laurent, A., & Orero, J. (2019). Mining Fuzzy-Temporal Gradual Patterns. *IEEE International Conference on Fuzzy Systems (FUZZ-IEEE 2019)*, 1-6. https://doi.org/10.1109/FUZZ-IEEE.2019.8858883
