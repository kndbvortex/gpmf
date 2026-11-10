# Algorithms

GPMF implements ten gradual pattern mining algorithms in five families. Each page gives the parameters, the support the algorithm uses, an example and the original reference.

## Index

| Algorithm | Key | Family |
| --------- | --- | ------ |
| [GRAANK](graank.md) | `graank` | Core |
| [GRITE](grite.md) | `grite` | Core |
| [SGrite](sgrite.md) | `sgrite`, `sgopt`, `sg1`, `sgb1`, `sgb2` | Core |
| [ParaMiner](paraminer.md) | `paraminer` | Closed |
| [GLCM](glcm.md) | `glcm` | Closed |
| [PGLCM](pglcm.md) | `pglcm` | Closed |
| [ACO-GRAANK](ant_graank.md) | `ant-graank`, `aco-graank` | Swarm |
| [TGrad](tgrad.md) | `tgrad` | Temporal |
| [MSGP](msgp.md) | `msgp` | Seasonal |
| [EmergingGP](emerging_gp.md) | none (Python only) | Contrast |

## Reading a pattern

A pattern is a list of gradual items. The item `'0+'` reads "column 0 increases" and `'2-'` reads "column 2 decreases". Columns are numbered in the order of your data, so you can get the names back with:

```python
names = list(df.columns)
readable = [names[int(item[:-1])] + item[-1] for item in pattern.to_string()]
```

A pattern and its complement, for instance `{0+, 1-}` and `{0-, 1+}`, describe the same ordering of the rows read in opposite directions.

## Two kinds of support

The literature uses two support measures that do not mean the same thing. Their exact relation was shown by Lonlac et al. [1].

**Support on pairs** (a Kendall's tau style count), used by GRAANK, ACO-GRAANK, TGrad and EmergingGP. It is the share of pairs of rows that respect every item of the pattern:

```
supp_pairs(P) = |{(i, j) : i < j, the pair (o_i, o_j) respects every item of P}| / C(n, 2)
```

**Support on chains** (gradual support), used by GRITE, SGrite, ParaMiner, GLCM and PGLCM. It is the length of the longest sequence of rows that the pattern orders completely, divided by the number of rows:

```
supp_chain(P) = max{ |T*| : T* is a subset of T totally ordered by P } / |T|
```

### How the two relate

Let λc be a chain threshold and λ'c a pair threshold, both as absolute counts. Lonlac et al. [1] prove that if λ'c ≤ λc × (λc - 1) / 2, every pattern frequent for chains at λc is also frequent for pairs at λ'c. With relative thresholds, a pattern found by GRITE, SGrite or ParaMiner at chain support s is found by GRAANK at pair support s' as long as s' ≤ s² × n(n - 1) / (n - 1)², which is about s² when n is large.

The reverse does not hold. At thresholds matched this way, GRAANK usually returns more patterns than GRITE, and the experiments of [1] confirm it.

In practice, never compare thresholds across the two families: a support of 0.5 in GRAANK and a support of 0.5 in GRITE are different requirements.

### Ties

Chain miners also differ in how they treat two rows with the same value on an attribute:

- GRITE, GLCM and PGLCM use a non-strict order (≥ and ≤), as in Di-Jorio et al. (2009);
- SGrite and its variants use a strict order, as in Tayou Djamegni et al. (2021);
- ParaMiner uses a strict order and breaks ties with the row index, as the original ParaMiner code does. Its result can therefore depend on the order of the rows.

On data without ties the three conventions agree. On data with many ties, compare chain miners that share a convention.

## References

[1] Lonlac, J., Tchide, B. F., Bomgni, A., Doniec, A., & Mephu Nguifo, E. (2024). Revisiting Frequent (Closed) Gradual Itemsets Mining. *IEEE ICTAI 2024*, 913-920. https://doi.org/10.1109/ICTAI62512.2024.00132
