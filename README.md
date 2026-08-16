# MoMPy

**Moment matrices for SDP hierarchy relaxations.**

MoMPy builds the moment matrix of a semidefinite relaxation and works out, for
you, which of its entries are forced to be equal or zero by the algebraic
properties of your operators — rank-1 projectors, orthogonal measurements,
commutation. You describe the operators; MoMPy hands back a matrix of SDP
variable indices ready to drop into CVXPY.

```python
from MoMPy import OperatorSet, MomentProblem

ops = OperatorSet()
R = ops.add_family(3, idempotent=True)     # three pure states
M = ops.add_povm_family(2, 2)              # M[y][b]: two binary measurements
ops.declare_commuting(R, R)                # the states commute with each other

monomials  = list(R) + [m for row in M for m in row]
monomials += [[R[x], M[y][b]] for x in range(3) for y in range(2) for b in range(2)]

mm = MomentProblem(monomials, ops.algebra(), dim=1).build()
print(mm.summary())
```

```
MomentMatrix: 20 x 20 (19 monomials + identity)
  block size (dim)   : 1
  SDP variables      : 64
  compression        : 400 entries -> 64 variables (6.2x)
  zero entries       : 64
  distinct words seen: 885
  build time         : 0.012 s
```


> ### `cyclicity`: tracial or state moments? Read this before your first build.
>
> `MomentProblem(..., cyclicity=True)` (the default) uses **tracial** moments
> `Tr(u v†)`, which are cyclic. `cyclicity=False` uses **state** moments
> `<psi|u v†|psi>`, which are not.
>
> Cyclicity is valid when your figure of merit really is a trace with the state
> *inside* the algebra — prepare-and-measure scenarios, `Tr(rho_x M_b)`. It is
> **not** valid for Bell/NPA problems. Imposing it there over-constrains the
> program: CHSH at level 1+AB returns 2.0000 instead of Tsirelson's 2.8284, so
> it is not an upper bound on the quantum value at all. At level 1 both agree,
> which makes the error easy to miss.
>
> | Problem | Use |
> |---|---|
> | Bell, NPA, device-independent | `cyclicity=False` |
> | Prepare-and-measure, dimension witnesses | `cyclicity=True` (the default) |
> | Unsure | `cyclicity=False` (fewer relations, so never invalid) |

---

## Contents

- [Installation](#installation)
- [What problem this solves](#what-problem-this-solves)
- [Tutorial: a prepare-and-measure scenario](#tutorial-a-prepare-and-measure-scenario)
- [Building the SDP](#building-the-sdp)
- [Block moment matrices](#block-moment-matrices)
- [API reference](#api-reference)
- [Performance](#performance)
- [Upgrading from 0.x](#upgrading-from-1x)

---

## Installation

```bash
pip install MoMPy            # core, needs only numpy
pip install MoMPy[cvxpy]     # plus the CVXPY helpers
```

From a checkout:

```bash
pip install -e ".[dev]"
pytest
```

---

## What problem this solves

Take a prepare-and-measure scenario. Alice encodes a message `x` in a quantum
state `R[x]` and sends it to Bob, who measures with `M[y][b]` and observes `b`.
The observable statistics are `p(b|x,y) = Tr(R[x] @ M[y][b])`, and you want to
maximise some linear functional of them over *all* states and measurements.

That optimisation is not an SDP. The standard relaxation makes it one: list
monomials in your operators, `L = {1, R[x], M[y][b], R[x] R[x'], R[x] M[y][b], ...}`,
and form the matrix `G[u,v] = Tr(u v†)` over `u, v ∈ L`. `G` is positive
semidefinite by construction and your objective lives inside it, so maximising
over PSD `G` gives an upper bound.

The tedious part is that many entries of `G` are secretly the same variable.
If `R[x]` is a pure state then `Tr(R[x])` and `Tr(R[x] R[x])` are equal. If
`M[y][b]` is a projective measurement then `Tr(R[x] M[y][0] M[y][1])` is
identically zero. Miss these identifications and your relaxation is looser than
it should be; get them wrong and it is not a valid bound at all.

MoMPy finds them. You declare the properties, it computes the equivalence
classes and returns the matrix.

**Applicable to** any optimisation expressible as an SDP relaxation over traces
of operator monomials: NPA / device-independent bounds, prepare-and-measure
scenarios, dimension witnesses, randomness certification, joint measurability.

---

## Tutorial: a prepare-and-measure scenario

### 1. Allocate operators

Operators are integer labels. `OperatorSet` allocates them and remembers their
properties, so you never keep a counter by hand. **Label `0` is reserved for
the identity** and is added to the matrix automatically.

```python
from MoMPy import OperatorSet

nX, nY, nB = 3, 2, 2

ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)     # R[x],   pure states
M = ops.add_povm_family(nY, nB)             # M[y][b], projective measurements
```

`add_povm_family` registers each measurement's outcomes as an orthogonal set
*and* as projectors, which is the usual projective assumption. Override with
`add_povm(n, idempotent=False, orthogonal=False)` if you need something else.

### 2. Declare the relations

Three kinds of relation are supported:

| Relation | Meaning | How to declare |
|---|---|---|
| **Idempotent** | `P @ P == P` | `add_family(..., idempotent=True)` or `ops.declare_idempotent([...])` |
| **Orthogonal** | `P_i @ P_j == 0` for `i != j` | `add_povm(...)` or `ops.declare_orthogonal([...])` |
| **Commuting** | `a @ b == b @ a` | `ops.declare_commuting(A, B)` |

```python
ops.declare_commuting(R, R)                 # every R[x] commutes with every R[x']
```

`declare_commuting(A, B)` means *every* label in `A` commutes with *every* label
in `B`. Pass the same list twice for "all of these commute with each other".

### 3. Choose your monomials

The hierarchy level is just which monomials you include. Longer words give a
tighter bound and a bigger matrix.

```python
monomials  = list(R)                                      # first order
monomials += [m for row in M for m in row]
monomials += [[R[x], M[y][b]]                             # second order
              for x in range(nX) for y in range(nY) for b in range(nB)]
monomials += [[R[x], R[xx], R[xxx]]                       # some third order
              for x in range(nX) for xx in range(nX) for xxx in range(nX)]
```

A monomial is a bare label or a list of labels read left to right as a product.
For the standard "all words up to length k" there is a shortcut:

```python
from MoMPy import generate_monomials
monomials = generate_monomials(list(R) + flat_M, level=2)
```

### 4. Build

```python
from MoMPy import MomentProblem

mm = MomentProblem(monomials, ops.algebra(), dim=1).build(progress=True)
```

`dim` is the one parameter with no default: it is the side length of the
block that will back each entry once you reach `to_cvxpy` (see
[Block moment matrices](#block-moment-matrices) below). `dim=1` is the
ordinary scalar moment matrix used throughout this tutorial section.

`mm.matrix` is an integer NumPy array: `mm.matrix[r, c]` is the index of the SDP
variable at that position. Equal indices mean the same variable.

Look up the variable for any monomial:

```python
mm.index_of([R[0], M[1][0]])     # the variable holding Tr(R0 M10)
mm.identity_index                # the variable holding Tr(1)
mm.zero_index                    # the class of monomials forced to zero
mm.equivalents([R[0]])           # every monomial equal to Tr(R0)
```

---

## Building the SDP

### With the CVXPY helper

```python
model = mm.to_cvxpy()
ct = list(model.constraints)          # G >> 0, and zeros pinned to zero
```

Index the model by monomial or by variable index:

```python
model[[R[0], M[1][0]]]     # scalar expression for Tr(R0 M10)
model.identity             # Tr(1)
```

> **`Tr(1)` is the dimension, not 1.** In a tracial relaxation the identity
> variable equals the Hilbert-space dimension. MoMPy deliberately does *not*
> constrain it. Add `ct.append(model.identity == 1)` only if you are using the
> state-vector NPA convention where moments are `<psi| w |psi>`.

### Normalisation constraints

`sum_b M[y][b] == 1` is a linear relation between variables, so it must be added
to the program. MoMPy finds every place it applies:

```python
for y in range(nY):
    ct += model.apply(mm.normalisation_constraints(M[y]))
```

For joint measurability, where a parent POVM marginalises onto a single
operator:

```python
ct += model.apply(mm.marginal_constraints(joint=B_labels, marginal=M[0][0]))
```

### Problem-specific constraints

```python
ct += [model[[R[x]]] == 1.0 for x in range(nX)]              # states are normalised
ct += [model[[R[x], R[xx]]] >= d for x in range(nX) for xx in range(nX)]
```

### Solve

```python
import cvxpy as cp

W = sum(model[[R[x], M[0][x]]] for x in range(nX))
problem = cp.Problem(cp.Maximize(W), ct)
problem.solve(solver=cp.SCS)
print(problem.value)
```

Any SDP solver works — SCS and Clarabel ship with CVXPY; MOSEK is free with an
academic licence.

### Without CVXPY

Nothing ties you to CVXPY. Allocate one variable per index and read the matrix:

```python
variables = {i: make_variable() for i in mm.variable_indices}
variables[mm.zero_index] = 0.0
G = [[variables[mm.matrix[r, c]] for c in range(mm.n)] for r in range(mm.n)]
```

Constraint objects expose plain integers via `.lhs` and `.rhs`, so
`mm.normalisation_constraints(...)` is usable with any modelling layer.

---

## Block moment matrices

Set `dim=d` for `d > 1` and every entry of the matrix becomes a `d x d` block
instead of a scalar — for relaxations whose "moments" are themselves
operators on a `d`-dimensional Hilbert space, rather than numbers. Cyclicity
practically never holds for these: `u v` and `v u` are genuinely different
blocks, so `cyclicity=False` is the right choice for essentially every block
hierarchy, and `hermitian=False` too whenever a block and its adjoint are
meant to be different blocks (the general case).

```python
bm = MomentProblem(monomials, ops.algebra(), dim=d, cyclicity=False, hermitian=False).build()
model = bm.to_cvxpy()            # dim x dim CVXPY blocks, read straight off bm.dim
model[[R[0], M[1][0]]]           # a dim x dim expression, not a scalar
```

`to_cvxpy` is the same function used for scalar matrices above — it reads
`matrix.dim` and builds scalars or blocks accordingly, so there is nothing
extra to call or import for the block case. Everything else (constraints,
`.apply()`, `.normalisation_constraints()`, indexing by monomial or by
variable index) works exactly as in the scalar walkthrough above.

---

## API reference

### Describing a problem

| Object | Purpose |
|---|---|
| `OperatorSet` | Allocates labels, records properties, emits an `Algebra` |
| `Algebra(idempotents, orthogonal_sets, commuting_pairs)` | The relations, if you prefer to build them by hand |
| `generate_monomials(letters, level)` | All words up to a given length |
| `MomentProblem(monomials, algebra, *, dim, cyclicity=True, hermitian=True, dedupe=True)` | One class for every relaxation: scalar or block, tracial or state |
| `MomentProblem.from_levels(letters, level, extra=..., dim=...)` | Shortcut constructor |

One class covers what used to be four: `MomentProblem(m, a, dim=1)` is the
tracial relaxation `Tr(u v†)`; add `cyclicity=False` for state moments
`<psi|u v†|psi>` — **use this for NPA/Bell** — and `dim=d>1` for a block
hierarchy whose entries are `d x d` operators (see
[Block moment matrices](#block-moment-matrices)).

### `MomentProblem.build(progress=False)` → `MomentMatrix`

| Attribute | Meaning |
|---|---|
| `.matrix` | `(n, n)` integer array of variable indices |
| `.n`, `.shape` | Matrix size |
| `.monomials` | Generating monomials, excluding the identity |
| `.word_at(r, c)` | Explicit operator word behind an entry |
| `.words` | Full nested list of words (built lazily) |
| `.map_table` | `MapTable`: monomial → index |
| `.variable_indices`, `.n_variables` | The distinct variables present |
| `.zero_index`, `.identity_index` | Reserved classes |
| `.has_zeros` | Whether orthogonality forced anything to zero |
| `.stats` | Build diagnostics |
| `.index_of(w)`, `.get(w, default)` | Lookup; `index_of` raises `UnknownMonomial` |
| `.equivalents(w)` | All monomials sharing `w`'s variable |
| `.summary()` | Human-readable report |
| `.normalisation_constraints(povm)` | `sum(povm) == 1` constraints |
| `.marginal_constraints(joint, marginal)` | `sum(joint) == marginal` constraints |
| `.to_cvxpy(dim=None, psd=True, complex=None, normalise_identity=False)` | CVXPY model, scalar or block per `.dim` |
| `.to_legacy()` | The 0.x five-tuple |
| `.dim`, `.cyclicity`, `.hermitian` | The three flags the matrix was built with |

### Options

- **`dim`** — side length of the block each SDP variable becomes in
  `to_cvxpy`. No default: declare it explicitly, even as `dim=1` for an
  ordinary scalar matrix.
- **`cyclicity`** — identify each word with its cyclic rotations, i.e. treat
  an entry as a trace `Tr(u v)` rather than an operator product `u v`. Default
  `True`. See the callout above — `False` is what NPA/Bell problems need.
- **`hermitian`** — identify each word with its reversal. For Hermitian
  operators this says the moment matrix is real symmetric, i.e. the variables
  are `Re Tr(w)`. Default `True`. Set `False` to build a complex Hermitian
  SDP, or for a block hierarchy where a block and its adjoint should be
  independent.
- **`dedupe`** — drop repeated monomials, which only add linearly dependent
  rows and columns. Default `True`.

---

## Performance

Version 2 replaces the per-monomial linear scans with canonical tuple words, a
breadth-first closure that memoises every word it has already seen, and a
union-find over classes. Each distinct word is expanded exactly once for the
whole build, and monomial lookup is a dict probe rather than a scan over every
word in every class.

Measured on the scenarios in `examples/`:

| Scenario | Matrix | 0.x | 1.x | Speedup |
|---|---|---|---|---|
| NPA CHSH level 1 | 9×9 | 0.01 s | 0.007 s | ~1× |
| NPA CHSH level 1+AB | 25×25 | 0.02 s | 0.012 s | 2× |
| PAM dimension, 3rd order | 84×84 | 41.7 s | 0.041 s | **1027×** |
| PAM dimension, 2nd+3rd order | 105×105 | 52.4 s | 0.057 s | **919×** |
| PAM dimension, nX=4 | 137×137 | 528 s | 0.178 s | **2960×** |

Scaling is now roughly linear in the number of matrix entries:

| Scenario | Matrix | Entries | Variables | Time |
|---|---|---|---|---|
| NPA 3 settings, 3 outcomes, 1+AB | 100×100 | 10 000 | 1 370 | 0.47 s |
| NPA 5 settings, 3 outcomes, 1+AB | 256×256 | 65 536 | 11 237 | 3.7 s |
| PAM 6 states, order 3 | 287×287 | 82 369 | 381 | 0.86 s |
| PAM 8 states, order 3 | 639×639 | 408 321 | 1 670 | 5.8 s |

---

## Correctness

The equivalence classes are checked against a deliberately naive brute-force
closure oracle over 720 randomised scenarios, covering tracial and block modes
with and without reversal symmetry. The induced partitions match exactly.

On top of that, `tests/test_physics.py` solves real SDPs (CHSH → 2√2, a fully
commutative algebra → the local bound 2, state discrimination → 1) and plugs
explicit matrices in for the operator labels to confirm numerically that every
monomial sharing a variable really does have the same trace and that the zero
class really vanishes.

```bash
pytest                     # everything
pytest tests/test_api.py   # fast unit tests only
```

---

## Upgrading from 0.x

**Your existing scripts keep working.** `from MoMPy.MoM import *` still gives
you `MomentMatrix`, `fmap`, `normalisation_contraints` and friends, returning
the same five outputs.

Two fixes do change the numbers you get, both in the direction of a tighter and
more correct relaxation. See [`MIGRATION.md`](MIGRATION.md) for the details and
for how to port to the new API.

---

## Citing and contact

Author: Carles Roch i Carceller — <chalswater@gmail.com>
Repository: <https://github.com/chalswater/MoMPy> · MIT licence.
