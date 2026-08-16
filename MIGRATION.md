# Migrating from MoMPy 1.0 to 1.1

**Short version:** `MomentProblem`, `BlockMomentProblem`, `StateMomentProblem`
and `NPAProblem` are now one class. `to_cvxpy` and `to_cvxpy_block` are now
one function. `MoMPy.MoM` / `MoMPy.BloM` are untouched — if you only ever used
those, there is nothing to do.

## 1. One class instead of four

1.0 grew a class per combination of two independent flags — does the trace
close cyclically? are words identified with their reversal? — plus a separate
block-matrix variant on top. That is really one build procedure with two
booleans and a size, so 1.1 collapses all four into one:

| 1.0 | 1.1 |
|---|---|
| `MomentProblem(m, a)` | `MomentProblem(m, a, dim=1)` |
| `StateMomentProblem(m, a)` / `NPAProblem(m, a)` | `MomentProblem(m, a, dim=1, cyclicity=False)` |
| `BlockMomentProblem(m, a)` | `MomentProblem(m, a, dim=1, cyclicity=False, hermitian=False)` |

`tracial` is renamed `cyclicity` (same meaning: identify a word with its
cyclic rotations). `hermitian` is unchanged. **`dim` is new and mandatory** —
declare the block side-length up front, even when it is `1`. Declaring it
here, once, is what lets `to_cvxpy` build the right shape automatically
instead of needing a separate block-aware function (next section).

`MomentMatrix` gains `.dim` and `.cyclicity` attributes so a built matrix
still tells you what it was built with; `BlockMomentMatrix` is gone, since a
single `MomentMatrix` now covers both cases.

## 2. One `to_cvxpy` instead of two

`to_cvxpy_block(matrix, dim)` and `CvxpyBlockModel` are gone.
`to_cvxpy(matrix)` now reads `matrix.dim` (set from the `MomentProblem` that
built it) and produces `dim x dim` CVXPY blocks automatically when
`dim > 1`, plain scalars when `dim == 1` — the same `CvxpyModel` indexes to
whichever shape applies:

```python
# 1.0
bm = BlockMomentProblem(monomials, algebra).build()
model = to_cvxpy_block(bm, dim=2)
model[[R[0], M[1][0]]]              # 2x2 CVXPY expression

# 1.1
mm = MomentProblem(monomials, algebra, dim=2, cyclicity=False).build()
model = mm.to_cvxpy()               # dim=2 comes from mm.dim -- or to_cvxpy(mm)
model[[R[0], M[1][0]]]              # 2x2 CVXPY expression, same as before
```

`to_cvxpy(matrix, dim=..., complex=...)` still accepts overrides for the odd
case where you want a different block size or real-only blocks for one call;
omit them and both follow what `MomentProblem` declared (`complex` defaults
to `dim > 1`).

The PSD constraint `to_cvxpy` builds is now always the symmetrised
`G + G.H >> 0`, in every case (scalar or block, `hermitian` True or False).
This is not a behavioural change: for `dim == 1` it is exactly the old
`G >> 0` whenever `hermitian=True` was used to build the matrix (the
classifier already makes `G` exactly symmetric there, so `G + G.H` is just
`2G`, and scaling a PSD constraint by a positive constant does not change its
feasible set or the optimal value of anything defined over the same
variables), and exactly the old `G + G.T >> 0` whenever `hermitian=False`
was used and the variables are real (`.H` and `.T` coincide). For blocks it
is the same convexification `to_cvxpy_block` already used.

## 3. Nothing else changes

Zero-pinning, `normalise_identity`, `.apply()`, and every `MomentMatrix`
method (`.index_of`, `.normalisation_constraints`, `.to_legacy`, ...) behave
exactly as in 1.0 for `dim=1`. `MoMPy.MoM` and `MoMPy.BloM` build the unified
`MomentProblem` internally now, but their own function signatures —
`MomentMatrix(...)`, `BlockMatrix(...)`, `fmap`, `normalisation_contraints`,
etc. — have not changed at all.

---

# Migrating from MoMPy 0.x to 1.0

**Short version:** your existing scripts still run. Two bug fixes change the
numbers they produce, both making the relaxation tighter and more correct. If
you have published results, re-run them rather than assuming they carry over.

---

## 1. Nothing to change immediately

The functional interface is preserved:

```python
from MoMPy.MoM import *

[G, map_table, S, list_of_eq_indices, Mexp] = MomentMatrix(
    S_1, S_2, S_high, rank_1_projectors, orthogonal_projectors, commuting_pairs)

G_var_vec[fmap(map_table, [R[x], M[y][b]])]
```

`map_table` subclasses `list`, so `map_table[-1][-1]`, `term[0]`, iteration and
slicing all behave exactly as before — but lookups now go through a hash index
instead of scanning every word in every class.

`MoMPy.BloM` is likewise unchanged in shape: `BlockMatrix`,
`block_normalisation_contraints`, `check_if_id_BloM`.

---

## 2. Behavioural changes you must know about

### 2.1 `G[r][c]` now always equals `fmap(map_table, Mexp[r][c])`

**This is the important one.**

0.x built the matrix by computing the lower triangle and then copying it across
the diagonal:

```python
Moment_Matrix[i][j] = fmap(map_table, Mexp[i][j])
Moment_Matrix[j][i] = Moment_Matrix[i][j]
```

That forces the matrix to be symmetric. But the equivalence classes never
recorded that a word is equivalent to its reversal, so `fmap` and the matrix
could disagree. On a three-operator test case **12 of 81 entries disagreed**:

```
G[1][7] = 6  but  fmap(map_table, [1, 2, 3]) = 10
```

If you constrained a variable via `fmap`, you were sometimes constraining a
different variable from the one sitting at that position in `G`.

The symmetrisation is mathematically fine — it amounts to working with
`Re Tr(w)`, and `Re Tr(w) = Re Tr(w†)`. The fix is to put the reversal into the
equivalence relation instead of patching the matrix afterwards. 1.0 does that,
so the two always agree, and there is a regression test asserting it for every
entry.

**Effect on your results:** slightly fewer variables, and any constraint you
wrote through `fmap` now genuinely lands on the entry you meant.

### 2.2 Equivalence classes are now fully closed

0.x guarded its closure loop with

```python
diff = len(commuting_pairs)   # zero when no commuting pairs were declared
while diff > 0:
    ...
```

so with no commuting pairs the loop never ran and multi-step reductions were
only found when some other monomial happened to bridge them. The merging step
also mutated `id_elements` while iterating over it, silently skipping entries.

1.0 computes the true closure. Measured on the repository's own examples:

| Scenario | 0.x variables | 1.0 variables |
|---|---|---|
| NPA CHSH level 1 | 34 | 34 |
| NPA CHSH level 1+AB | 130 | 98 |
| PAM dimension, 3rd order | 237 | 178 |
| PAM dimension, nX=4 | 407 | 298 |

Every difference is a *merge*: 1.0's partition is strictly coarser, never finer.
Fewer variables means a tighter relaxation, so bounds can come out **lower** than
before. They were valid upper bounds before, just looser than intended.

### 2.3 `normalisation_contraints` returns more constraints

0.x hard-coded `dd = 0` and `yy = 0`, so it only inspected each class's first
stored word and only matched the first POVM outcome. Whether a constraint was
generated depended on which word the search happened to store first.

1.0 scans every word of every class and deduplicates. You get a complete,
non-redundant set. The return shape is unchanged, so the loop in the old README
still works.

### 2.4 `check_if_id_BloM` no longer raises

Its body referenced a `commuting_pairs` name that was never a parameter, so
every call raised `NameError`. It is reimplemented as a table lookup returning
`[found, is_zero, index]`. The trailing arguments are accepted and ignored.

### 2.5 `Commute` handles repeated labels

0.x removed the element *by value* (`v_copy.remove(store)`), which picked the
wrong occurrence in words containing repeated labels. Now it swaps by index.

---

## 3. If you need the old numbers

Pin the previous release:

```bash
pip install "MoMPy<2"
```

There is no flag to restore the old behaviour, because the old behaviour was
order-dependent — it is not a well-defined mode to reproduce.

---

## 4. Porting to the new API

The new API is optional. It mainly removes bookkeeping.

### Before

```python
w_R, w_M, S_1, cc = [], [], [], 1
for x in range(nX):
    S_1 += [cc]; w_R += [cc]; cc += 1
for b in range(nB):
    w_M += [[]]
    for y in range(nY):
        S_1 += [cc]; w_M[b] += [cc]; cc += 1

rank_1_projectors  = [w_R[x] for x in range(nX)]
rank_1_projectors += [w_M[b][y] for y in range(nY) for b in range(nB)]
orthogonal_projectors = [[w_M[b][y] for b in range(nB)] for y in range(nY)]
commuting_pairs = [[[w_R[x] for x in range(nX)], [w_R[x] for x in range(nX)]]]

[G, map_table, S, eq, Mexp] = MomentMatrix(
    S_1, [], S_high, rank_1_projectors, orthogonal_projectors, commuting_pairs)
```

### After

```python
from MoMPy import OperatorSet, MomentProblem

ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)
M = ops.add_povm_family(nY, nB)        # note: M[y][b], transposed vs w_M[b][y]
ops.declare_commuting(R, R)

mm = MomentProblem(monomials, ops.algebra()).build()
```

### Translation table

| 0.x | 1.0 |
|---|---|
| `fmap(map_table, w)` | `mm.index_of(w)` — raises on a miss |
| `fmap(...) == 'ERROR: ...'` | `mm.get(w) is None` |
| `map_table[-1][-1]` | `mm.zero_index` |
| `fmap(map_table, [0])` | `mm.identity_index` |
| `np.unique(G)` | `mm.variable_indices` |
| `Mexp[r][c]` | `mm.word_at(r, c)` |
| `S` | `mm.monomials` |
| `normalisation_contraints(M[y], identities)` | `mm.normalisation_constraints(M[y])` |
| `normalisation_contraints_2compatibility(B, M, ids)` | `mm.marginal_constraints(B, M)` |
| `check_if_id(...)` | `mm.get(w)` |
| manual `cp.bmat` loop | `mm.to_cvxpy()` |

`mm.to_legacy()` returns the old five-tuple if you want to mix styles.

### Building the SDP

Before:

```python
G_var_vec = {}
for element in list_of_eq_indices:
    if element == map_table[-1][-1]:
        G_var_vec[element] = 0.0
    else:
        G_var_vec[element] = cp.Variable()

lis = []
for r in range(len(G)):
    lis += [[]]
    for c in range(len(G)):
        lis[r] += [G_var_vec[G[r][c]]]
MomMat = cp.bmat(lis)
```

After:

```python
model = mm.to_cvxpy()
ct = list(model.constraints)      # includes G >> 0 and the zero pinning
```

`cp.bmat` creates one CVXPY object per matrix entry and becomes the bottleneck
well before the solver does. `to_cvxpy` allocates a single vector variable and
gathers it through the index matrix, so the whole matrix is one atom.

### One trap worth flagging

`to_cvxpy` does **not** constrain `Tr(1) == 1` by default. In a tracial
relaxation `Tr(1)` is the Hilbert-space dimension. Pinning it to 1 forces
dimension 1 and quietly turns, for example, state discrimination into random
guessing. Add it yourself only when you mean the state-vector NPA convention:

```python
ct.append(model.identity == 1)
```

---

## 5. Summary of fixes

| # | Bug | Effect |
|---|---|---|
| 1 | Matrix symmetrised but classes were not | `G[r][c]` disagreed with `fmap` on 12/81 entries |
| 2 | `check_if_id_BloM` used an undefined name | `NameError` on every call |
| 3 | `id_elements.remove()` during iteration | Merges silently skipped |
| 4 | Closure loop skipped when no commuting pairs | Up to 27% too many variables; looser bound |
| 5 | `normalisation_contraints` fixed `dd=0, yy=0` | Constraints silently missing |
| 6 | `Commute` removed by value | Wrong swap on repeated labels |
