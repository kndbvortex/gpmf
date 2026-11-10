# MSGP

Seasonal gradual patterns: co-variations that come back at every cycle, such as each day or each year.

## Overview

MSGP (Mining Seasonal Gradual Patterns) cuts the series into consecutive cycles of fixed length. Each cycle becomes one sequence: its k-th element holds the gradual items that hold between observations k and k + 1 of the cycle. A periodic pattern miner from SPMF (MPFPS-BFS) then looks for itemsets that are periodic inside many of these sequences. The result is a set of gradual patterns that recur at the same moments of the cycle.

A typical use is hourly sensor data with a daily cycle: MSGP can find that "temperature rises while humidity falls" holds every morning in most days.

MSGP is the only algorithm of GPMF that needs Java, through SPMF.

## Parameters

**min_support** : float, default=0.5
    Seasonal support: the share of cycles in which the pattern must be periodic. A value above 1 is read as a number of cycles.

**cycle_size** : int, default=8
    Number of observations in one cycle. An incomplete last cycle is dropped, and the transition between two cycles is not used.

**min_ra** : float or None, default=None
    Share of the transitions of a cycle on which the pattern must hold for it to count as periodic in that cycle. `None` means 0.5.

**max_periodicity** : int or None, default=None
    Largest gap allowed between two occurrences inside a cycle. `None` means the cycle size.

**max_std** : int or None, default=None
    Largest standard deviation allowed for these gaps. `None` means the cycle size.

**spmf_path** : str or None, default=None
    Path to `spmf.jar`. When it is not given, GPMF uses the configuration or the `SPMF_PATH` environment variable.

## Support

A pattern is periodic in a cycle when it occurs on at least `min_ra` × (cycle_size - 1) transitions of that cycle, with gaps of at most `max_periodicity` and a gap standard deviation of at most `max_std`. Its support is the share of cycles in which it is periodic, and it is kept when that share reaches `min_support`.

In MSGP, "rise together" and "fall together" are different patterns, because the moment of the cycle matters.

## Example

```python
import pandas as pd
from gpmf.algorithms.seasonal.msgp import MSGP

df = pd.read_csv('hourly_measurements.csv')   # one row per hour
algo = MSGP(min_support=0.5, cycle_size=24, spmf_path='/path/to/spmf.jar')

for p in algo.fit(df).get_patterns():
    print(p.to_string(), p.support)
```

## Installing SPMF

MSGP calls SPMF through the `spmf` Python package and needs Java:

```bash
pip install spmf
```

Download `spmf.jar` from the [SPMF website](https://www.philippe-fournier-viger.com/spmf/), then give its path in one of three ways:

```python
algo = MSGP(min_support=0.5, cycle_size=24, spmf_path='/path/to/spmf.jar')

# or once for the session
from gpmf.config import config
config.spmf_path = '/path/to/spmf.jar'
```

or set the `SPMF_PATH` environment variable. Recent SPMF releases need a recent Java runtime; check the SPMF website for the version.

The command line does not expose `cycle_size`: use MSGP from Python.

## See also

[TGrad](tgrad.md): when the delay between attributes is unknown and has to be found.

## Reference

[1] Lonlac, J., Doniec, A., Lujak, M., & Lecoeuche, S. (2020). Mining Frequent Seasonal Gradual Patterns. *Big Data Analytics and Knowledge Discovery (DaWaK 2020)*, LNCS 12393, 197-207. https://doi.org/10.1007/978-3-030-59065-9_16
