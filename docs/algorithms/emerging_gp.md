# EmergingGP

Mines gradual patterns that are frequent in a target dataset but absent from a reference dataset.

## Overview

EmergingGP implements the emerging gradual pattern framework of Laurent, Lesot, and Rifqi (2015). Given a reference dataset D1 and a target dataset D2, it identifies gradual patterns that co-vary significantly in D2 but not in D1. Such patterns characterise what is structurally specific to D2 compared to D1 in terms of attribute co-variations.

The algorithm is border-based: it mines the maximal frequent patterns of each dataset separately and then computes their set difference. The anti-monotone property of gradual pattern support guarantees that the set of frequent patterns is convex and can be compactly represented by its maximal elements, making the set difference tractable without enumerating all frequent patterns explicitly.

Unlike all other GPMF algorithms, EmergingGP takes two datasets and does not inherit from `BaseAlgorithm`.

## Parameters

**min_support_d1** : float, default=0.5
    Support threshold for the reference dataset D1. Patterns with support ≥ min_support_d1 in D1 are considered present there and excluded from the output.

**min_support_d2** : float, default=0.5
    Support threshold for the target dataset D2. Only patterns with support ≥ min_support_d2 in D2 are candidates.

**base_algo** : str, default='graank'
    Name of the base algorithm used to mine each dataset. Any key accepted by `AlgorithmRegistry` is valid. The support family of the chosen algorithm applies to both thresholds.

## Support definition

Determined by `base_algo`. With the default `'graank'`, pair-based support is used. A pattern P emerges from D1 to D2 if:

```
supp_D2(P) >= min_support_d2   (frequent in target)
supp_D1(P) <  min_support_d1   (absent from reference)
```

## Examples

```python
>>> from gpmf.algorithms.emerging_gp import EmergingGP
>>> import pandas as pd
>>> d1 = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})   # anti-correlated
>>> d2 = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 3, 4]})   # positively correlated
>>> egp = EmergingGP(min_support_d1=0.5, min_support_d2=0.5)
>>> patterns = egp.fit(d1, d2).get_patterns()
>>> for p in patterns:
...     print(p.to_string(), p.support)
['A+', 'B+'] 1.0
```

Find what is specific to D1 by swapping the arguments:

```python
>>> patterns_in_d1 = egp.fit(d2, d1).get_patterns()
```

## Notes

The output contains **maximal** emerging patterns: patterns from D2's border that are not subsumed by D1's border. Sub-patterns of these maximals that are also emerging are not enumerated explicitly.

The two datasets must share the same column names and number of attributes.

Lowering `min_support_d1` makes it harder to qualify as "present" in D1, increasing the number of emerging patterns. Raising `min_support_d2` has the opposite effect.

## See Also

[GRAANK](graank.md)
    Default base algorithm used internally by EmergingGP.

## References

[1] Laurent, A., Lesot, M.-J., & Rifqi, M. (2015). Mining Emerging Gradual Patterns. *IFSA-EUSFLAT 2015*, Gijon, Spain.
    https://doi.org/10.2991/ifsa-eusflat-15.2015.234
    HAL: https://hal.science/hal-01160222v1
