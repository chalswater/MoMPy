# Changelog

## 3.0.0

Collapses `MomentProblem`, `BlockMomentProblem`, `StateMomentProblem` and
`NPAProblem` into a single `MomentProblem`, and `to_cvxpy` / `to_cvxpy_block`
into a single `to_cvxpy`. See [`MIGRATION.md`](MIGRATION.md).

### Changed

- `MomentProblem(monomials, algebra, *, dim, cyclicity=True, hermitian=True, dedupe=True)`
  replaces the four-class hierarchy. `tracial` is renamed `cyclicity`. `dim`
  is new and required: the block side-length backing each entry once the
  problem reaches `to_cvxpy`.
- `MomentMatrix` gains `.dim` and `.cyclicity`; `BlockMomentMatrix` is gone —
  a single `MomentMatrix` now covers both scalar and block matrices.
- `to_cvxpy(matrix, *, dim=None, psd=True, complex=None, normalise_identity=False, name="g")`
  replaces `to_cvxpy` + `to_cvxpy_block`. Reads `matrix.dim` unless
  overridden; builds scalars for `dim == 1`, `dim x dim` CVXPY blocks
  otherwise. The PSD constraint is always the symmetrised `G + G.H >> 0`,
  which is exactly `G >> 0` in every case the old `hermitian=True` branch
  used it (the two differ only by an immaterial positive factor of 2) and
  exactly `G + G.T >> 0` in every real-valued case the old `hermitian=False`
  branch used it.
- `CvxpyModel` replaces `CvxpyModel` + `CvxpyBlockModel`; indexing returns a
  scalar or a `dim x dim` expression depending on `matrix.dim`.
- `MoMPy.MoM` and `MoMPy.BloM` build the unified `MomentProblem` internally;
  their own function signatures are unchanged.

### Removed

- `BlockMomentProblem`, `StateMomentProblem`, `NPAProblem`,
  `BlockMomentMatrix`, `to_cvxpy_block`, `CvxpyBlockModel`.

## 2.0.0

Rewrite of the equivalence engine and restructure into a proper package.
Existing scripts using `MoMPy.MoM` / `MoMPy.BloM` continue to work; see
[`MIGRATION.md`](MIGRATION.md) for the two fixes that change results.

### Fixed

- **Matrix and lookup table disagreed.** The moment matrix was symmetrised
  after construction while the equivalence classes were not, so `G[r][c]` and
  `fmap(map_table, Mexp[r][c])` could return different variables (12 of 81
  entries on a small test case). Word reversal is now part of the equivalence
  relation, so the two always agree.
- **`check_if_id_BloM` raised `NameError` on every call** — its body referenced
  a `commuting_pairs` name that was never a parameter.
- **Equivalence classes were not fully closed.** The closure loop was skipped
  entirely when no commuting pairs were declared, and the merging step mutated
  the class list while iterating over it. Up to 27% more variables than
  necessary, giving a looser relaxation.
- **`normalisation_contraints` missed constraints.** It inspected only each
  class's first stored word and only the first POVM outcome.
- **`Commute` mishandled repeated labels**, removing by value rather than by
  index.
- `to_cvxpy` does not constrain `Tr(1) == 1`: in a tracial relaxation that is
  the Hilbert-space dimension, and pinning it to 1 silently forces dimension 1.

### Performance

Canonical tuple words, memoised breadth-first closure and union-find replace
the per-monomial linear scans. Each distinct word is expanded once per build;
monomial lookup is a dict probe rather than a scan over every word in every
class.

| Scenario | 1.x | 2.0 | Speedup |
|---|---|---|---|
| PAM dimension, 3rd order (84x84) | 41.7 s | 0.041 s | 1027x |
| PAM dimension, 2nd+3rd (105x105) | 52.4 s | 0.057 s | 919x |
| PAM dimension, nX=4 (137x137) | 528 s | 0.178 s | 2960x |

A 639x639 matrix (408k entries) now builds in 5.8 s.

### Added

- `Algebra`, `OperatorSet`, `MomentProblem`, `BlockMomentProblem`,
  `MomentMatrix`, `BlockMomentMatrix`, `MapTable`, `LinearConstraint`.
- `generate_monomials`, `MomentProblem.from_levels`.
- `mm.to_cvxpy()` building the matrix as a single CVXPY atom instead of one
  object per entry.
- `mm.summary()`, `mm.stats`, `mm.equivalents()`, `mm.word_at()`.
- `marginal_constraints` for joint measurability, replacing
  `normalisation_contraints_2compatibility`.
- Type hints and `py.typed`; `pyproject.toml` packaging.
- Test suite: 720 randomised differential tests against a brute-force oracle,
  SDP tests against known optima, and numerical realisability tests using
  explicit operator matrices.

### Changed

- `map_table` is now a `MapTable`, a `list` subclass with O(1) lookup. Indexing
  by integer or slice behaves as before; indexing by a monomial returns its
  variable index.
- Class members are sorted shortest-first, so `map_table[i][0][0]` is the
  simplest representative rather than an arbitrary one.
- Explicit words (`Mexp`) are built lazily; use `mm.word_at(r, c)` for one-off
  access.
