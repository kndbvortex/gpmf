# Algorithms

GPMF provides implementations of 10 gradual pattern mining algorithms spanning five categories. Each algorithm page documents its parameters, support definition, usage examples, and original reference.

## Algorithm index

| Algorithm                 | Key                                                | Category |
| ------------------------- | -------------------------------------------------- | -------- |
| [GRAANK](graank.md)          | `graank`                                         | Core     |
| [GRITE](grite.md)            | `grite`                                          | Core     |
| [SGrite](sgrite.md)          | `sgrite`, `sgopt`, `sg1`, `sgb1`, `sgb2` | Core     |
| [ParaMiner](paraminer.md)    | `paraminer`                                      | Closed   |
| [GLCM](glcm.md)              | `glcm`                                           | Closed   |
| [PGLCM](pglcm.md)            | `pglcm`                                          | Closed   |
| [ACO-GRAANK](ant_graank.md)  | `ant-graank`, `aco-graank`                     | Swarm    |
| [TGrad](tgrad.md)            | `tgrad`                                          | Temporal |
| [MSGP](msgp.md)              | `msgp`                                           | Seasonal |
| [EmergingGP](emerging_gp.md) |                                                    | Contrast |

## Support definitions

Two incompatible support families coexist in the gradual pattern literature. The exact relationship between them was established by Lonlac et al. [1].

**Pair-based support** (Kendall's τ) used by GRAANK, ACO-GRAANK, TGrad, EmergingGP

```
supp_τ(P) = |{(i,j) : i<j, all items of P satisfied by (oi, oj)}| / C(n, 2)
```

**Chain-based support** (gradual support) used by GRITE, SGrite, ParaMiner, GLCM, PGLCM

```
supp_G(P) = max{ |T*| : T* ⊆ T, T* totally ordered by P } / |T|
```

### Formal relationship

Let λc be a chain-based support threshold (absolute count) and λ'c a pair-based support threshold (absolute count). Lonlac et al. [1] prove the following:

**Theorem.** If λ'c ≤ λc × (λc − 1) / 2, then every pattern frequent under chain-based support at threshold λc is also frequent under pair-based support at threshold λ'c.

In normalised form, this means that any pattern found by GRITE/SGrite/ParaMiner at chain support s is also found by GRAANK at pair support s', provided:

```
s' ≤ s² × n(n−1) / (n−1)²  ≈  s²   (for large n)
```

The reverse inclusion does not hold in general: GRAANK at the corresponding threshold typically finds strictly more patterns than GRITE. Experiments in [1] confirm this: fixing thresholds according to the theorem, GRAANK always extracts more patterns than GRITE.

**Consequence for users:** do not compare threshold values across the two families. Do not interpret a support of 0.5 from GRAANK as equivalent to 0.5 from GRITE.

## References

[1] Lonlac, J., Tchide, B., Bomgni, A., Doniec, A., & Mephu Nguifo, E. (2024). Revisiting Frequent (Closed) Gradual Itemsets Mining. *IEEE ICTAI 2024*, 913–920. https://doi.org/10.1109/ICTAI62512.2024.00132
