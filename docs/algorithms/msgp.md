# MSGP

Mining Seasonal Gradual Patterns. Discovers gradual patterns that recur periodically across seasons or cycles.

## Overview

MSGP combines gradual pattern extraction with periodic pattern mining. It splits the dataset into cycles of fixed size, computes which gradual patterns hold within each cycle, and then uses SPMF's MPFPS-BFS algorithm to identify patterns that recur with bounded periodicity across cycles. The result is a set of gradual patterns annotated with their seasonal regularity.

A typical use case is monthly sensor data: MSGP can discover that "when temperature increases, humidity decreases" holds every summer with high regularity, but not in winter.

MSGP is the only algorithm in GPMF that requires an external Java dependency (SPMF).

## Parameters

**min_support** : float, default=0.5
    Minimum within-cycle support threshold.

**cycle_size** : int, default=8
    Number of rows per cycle. Should divide the dataset length evenly; incomplete trailing cycles are discarded.

**min_ra** : float or None, default=None
    Minimum ratio of cycles in which the pattern must appear. `None` computes a default from the dataset size and `cycle_size`.

**max_periodicity** : int or None, default=None
    Maximum gap (in cycles) between two consecutive occurrences. `None` defaults to the total number of cycles.

**max_std** : float or None, default=None
    Maximum standard deviation of the inter-occurrence gap. `None` defaults to the dataset size.

## Support definition

Periodic support: a pattern is included if it appears in at least `min_ra` of the observed cycles with inter-occurrence gap ≤ `max_periodicity` and gap standard deviation ≤ `max_std`.

## Examples

```python
>>> from gpmf.algorithms.seasonal.msgp import MSGP
>>> import pandas as pd
>>> df = pd.read_csv('monthly_measurements.csv')   # 36 rows, 12-month cycle
>>> algo = MSGP(min_support=0.5, cycle_size=12)
>>> patterns = algo.fit(df).get_patterns()
>>> for p in patterns:
...     print(p.to_string(), p.support)
```

Via CLI:

```bash
gradual-mine msgp data.csv --min-support 0.5 --cycle-size 12
```

## Notes

MSGP requires SPMF to be installed and its JAR path configured:

```python
>>> from gpmf.config import config
>>> config.spmf_path = '/path/to/spmf.jar'
```

Download SPMF from the [SPMF website](https://www.philippe-fournier-viger.com/spmf/).

## See Also

[TGrad](tgrad.md)
    Alternative for datasets where the temporal lag is unknown and must be discovered.

## References

[1] Lonlac, J., Doniec, A., Lujak, M., & Lecoeuche, S. (2020). Mining Frequent Seasonal Gradual Patterns. *Advances in Intelligent Data Analysis XIX*, DaWaK 2020, LNCS 12393, 197–207. https://doi.org/10.1007/978-3-030-59065-9_16
