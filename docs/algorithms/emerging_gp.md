# EmergingGP

Gradual patterns that are frequent in one dataset and not in another.

## Overview

EmergingGP implements the emerging gradual patterns of Laurent, Lesot and Rifqi (2015). Given a reference dataset D1 and a target dataset D2, it returns the gradual patterns that hold in D2 but not in D1. They describe what is specific to D2 in the way its attributes vary together.

The method works on borders. It mines each dataset separately and keeps the patterns of D2 that are not covered by a pattern of D1. Because support is anti-monotone, a pattern covered by a frequent pattern of D1 is itself frequent in D1, so this difference never needs the full list of frequent patterns.

EmergingGP takes two datasets, so unlike the other algorithms it does not derive from `BaseAlgorithm` and has no command line key.

## Parameters

**min_support_d1** : float, default=0.5
    Threshold in the reference dataset D1. Patterns that reach it in D1 count as present there and are excluded.

**min_support_d2** : float, default=0.5
    Threshold in the target dataset D2. Only patterns that reach it in D2 are candidates.

**base_algo** : str, default='graank'
    Algorithm used to mine each dataset, given by its key in `AlgorithmRegistry`. Its support applies to both thresholds.

## Support

It comes from `base_algo`, so with the default GRAANK it is the support on pairs. A pattern P emerges from D1 to D2 when:

```
supp_D2(P) >= min_support_d2   (frequent in the target)
supp_D1(P) <  min_support_d1   (not frequent in the reference)
```

## Example

```python
import pandas as pd
from gpmf.algorithms.emerging_gp import EmergingGP

d1 = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})   # A and B move in opposite directions
d2 = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 3, 4]})   # A and B move together

egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
for p in egp.fit(d1, d2).get_patterns():
    print(p.to_string(), p.support)
# ['0+', '1+'] 1.0
# ['0-', '1-'] 1.0
```

The two lines are the same pattern read in both directions. To see what is specific to D1, swap the arguments:

```python
patterns_in_d1 = egp.fit(d2, d1).get_patterns()
```

## Notes

The output contains the patterns of D2 returned by the base algorithm that no pattern of D1 covers. With GRAANK these are mostly maximal patterns, and their sub-patterns that also emerge are not listed one by one.

Both datasets must have the same columns in the same order.

Lowering `min_support_d1` makes it easier for a pattern to count as present in D1, so fewer patterns emerge. Raising `min_support_d2` also gives fewer patterns.

## See also

[GRAANK](graank.md): the default base algorithm.

## Reference

[1] Laurent, A., Lesot, M.-J., & Rifqi, M. (2015). Mining Emerging Gradual Patterns. *Conference of the International Fuzzy Systems Association and the European Society for Fuzzy Logic and Technology (IFSA-EUSFLAT 2015)*. https://doi.org/10.2991/ifsa-eusflat-15.2015.234
