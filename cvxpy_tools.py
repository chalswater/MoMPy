"""Optional CVXPY glue.

Turning a moment matrix into a CVXPY expression the obvious way -- one
variable per class, assembled with ``cp.bmat`` -- costs a Python-level object
per matrix entry and becomes the bottleneck well before the solver does.

:func:`to_cvxpy` instead allocates a single flat CVXPY variable and gathers it
into place with one constant indexing operation, so the whole moment matrix
is one CVXPY atom regardless of size. The same function handles both scalar
moment matrices (``matrix.dim == 1``, one number per class) and block moment
matrices (``matrix.dim > 1``, one ``dim x dim`` operator per class -- see
:class:`~MoMPy.problem.MomentProblem`), reading which one applies from
``matrix.dim`` so there is nothing to choose between at the call site.

CVXPY is not a hard dependency; import this module only if you have it.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .algebra import IDENTITY_LABEL

__all__ = ["CvxpyModel", "to_cvxpy"]


def _require_cvxpy():
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CVXPY is needed for MoMPy.cvxpy_tools; install it with "
            "`pip install cvxpy` or `pip install MoMPy[cvxpy]`"
        ) from exc
    return cp


class CvxpyModel:
    """A moment matrix expressed as CVXPY objects.

    Covers both the scalar case (``dim == 1``: one number per class) and the
    block case (``dim > 1``: one ``dim x dim`` expression per class). Which
    one applies is fixed at construction time from the ``matrix`` that built
    it, and determines what indexing returns.

    Attributes
    ----------
    G : cvxpy.Expression
        The moment matrix, already respecting every operator equivalence:
        shape ``(n, n)`` when ``dim == 1``, ``(n*dim, n*dim)`` otherwise.
    vector : cvxpy.Variable
        The underlying flat CVXPY variable everything else is built from.
        For ``dim == 1`` this is one entry per class; for ``dim > 1`` each
        class occupies a contiguous run of ``dim*dim`` entries, reshaped to a
        ``dim x dim`` block wherever it is indexed.
    constraints : list
        Structural constraints: the (symmetrised) PSD requirement on ``G``,
        and zeros pinned to the zero matrix.

    Indexing accepts either a variable index or a monomial, and returns a
    scalar or a ``dim x dim`` expression to match ``matrix.dim``::

        model[[R[0], M[1][0]]]     # by monomial
        model[7]                   # by variable index
    """

    __slots__ = ("G", "vector", "constraints", "_matrix", "_position", "_dim")

    def __init__(self, G, vector, constraints, matrix, position, dim) -> None:
        self.G = G
        self.vector = vector
        self.constraints = constraints
        self._matrix = matrix
        self._position = position
        self._dim = dim

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            pos = self._position[int(key)]
        else:
            pos = self._position[self._matrix.index_of(key)]
        d = self._dim
        if d == 1:
            return self.vector[pos]
        cp = _require_cvxpy()
        return cp.reshape(
            self.vector[pos * d * d:(pos + 1) * d * d], (d, d), order="C"
        )

    def variable(self, index: int):
        """The expression (scalar, or ``dim x dim``) for SDP variable ``index``."""
        return self[int(index)]

    @property
    def identity(self):
        """The expression for the identity class: ``Tr(1)`` or ``eye(dim)``."""
        return self[(IDENTITY_LABEL,)]

    def as_dict(self) -> dict:
        """``{variable index: expression}``, for code expecting a mapping."""
        return {idx: self[idx] for idx in self._position}

    def apply(self, constraints) -> list:
        """Turn :class:`~MoMPy.constraints.LinearConstraint` objects into CVXPY ones.

        Works unchanged for both cases: a ``LinearConstraint`` only ever
        combines its operands with ``+`` and ``==``, and both are just as
        meaningful between ``dim x dim`` expressions as between scalars.
        """
        variables = self.as_dict()
        return [ct.apply(variables) for ct in constraints]


def to_cvxpy(matrix, *, dim: int | None = None, psd: bool = True,
             complex: bool | None = None, normalise_identity: bool = False,
             name: str = "g"):
    """Build a :class:`CvxpyModel` from a moment matrix.

    One function for both scalar and block moment matrices: it reads
    ``matrix.dim`` -- declared once, on the
    :class:`~MoMPy.problem.MomentProblem` that built ``matrix`` -- and builds
    plain scalars for ``dim == 1`` or ``dim x dim`` CVXPY blocks otherwise.
    Either way, the whole moment matrix is a single CVXPY atom: one flat
    variable, gathered into the ``(n, n)`` (or ``(n*dim, n*dim)``) matrix with
    one constant indexing operation, never one Python object per entry.

    Parameters
    ----------
    matrix:
        A built :class:`~MoMPy.matrix.MomentMatrix`.
    dim:
        Override the block size instead of using ``matrix.dim``. Rarely
        needed -- it exists for solving the same built matrix at a different
        block size without rebuilding it -- since ordinarily the size you
        want is exactly the one declared when you built ``matrix``.
    psd:
        Include a PSD constraint on ``model.G`` in ``model.constraints``:
        always the symmetrised ``G + G.H >> 0``. This is exactly equivalent
        to a bare ``G >> 0`` whenever ``G`` is already Hermitian -- in
        particular whenever ``matrix`` was built with ``hermitian=True`` and
        ``dim == 1``, where the classifier already makes ``G`` exactly
        symmetric, so ``G + G.H`` is just ``2G`` and the two constraints have
        the same feasible set -- and otherwise it is the same convexification
        a ``hermitian=False`` build has always used, generalised from numbers
        to ``dim x dim`` blocks: valid whether or not the mirror position
        happens to hold the transpose class, and exact whenever ``G`` happens
        to be Hermitian anyway.
    complex:
        Allocate a complex CVXPY variable. Defaults to ``dim > 1`` (a block
        ``u v^\\dagger`` need not be real even when ``u`` and ``v`` are real
        operators, so blocks default to complex) and to ``False`` for
        ``dim == 1`` (the ordinary real-valued scalar relaxation). Override
        either way explicitly if you know better -- e.g. ``complex=False``
        for a cheaper real-only block relaxation, or ``complex=True`` for a
        complex Hermitian *scalar* SDP.
    normalise_identity:
        Include ``Tr(1) == 1`` (``dim == 1``) or ``block(1) == eye(dim)``
        (``dim > 1``). **Off by default and deliberately so.** In a tracial
        formulation ``Tr(1)`` is the Hilbert-space dimension, which is
        usually either free or bounded by the problem; pinning it is only
        correct for the state-vector NPA convention, or, in the block case,
        when nothing else already pins row/column 0 to the bare identity
        (e.g. when it additionally carries a branch weight ``q``, where
        ``block(1) == q * eye(dim)`` is the constraint you actually want,
        added by hand instead).
    name:
        Name passed to the underlying CVXPY variable.
    """
    cp = _require_cvxpy()
    dim = int(matrix.dim if dim is None else dim)
    if complex is None:
        complex = dim > 1

    indices = matrix.variable_indices
    position = {int(v): i for i, v in enumerate(indices)}
    n_vars = len(indices)
    n = matrix.n

    constraints = []
    zero_index = matrix.zero_index

    if dim == 1:
        vector = cp.Variable(n_vars, complex=complex, name=name)

        # Gather: map every matrix entry to its slot in the vector, then
        # reshape -- the whole matrix stays a single CVXPY atom.
        flat = np.searchsorted(indices, matrix.matrix.ravel())
        G = cp.reshape(vector[flat], (n, n), order="C")

        if zero_index in position:
            constraints.append(vector[position[zero_index]] == 0)
        if normalise_identity:
            identity_index = matrix.get((IDENTITY_LABEL,))
            if identity_index is not None and identity_index in position:
                constraints.append(vector[position[identity_index]] == 1)
    else:
        vector = cp.Variable(n_vars * dim * dim, complex=complex, name=name)

        # Every entry of the big matrix is one dim x dim class, placed at a
        # fixed (row-block, col-block) position -- a 0/1 selection,
        # independent of the variable values, so it is precomputed once as a
        # constant sparse matrix and applied with a single matmul instead of
        # a per-entry cp.bmat (one Python object per matrix position, exactly
        # what the scalar gather-reshape above also avoids). A literal
        # generalisation of that gather-reshape trick -- stacking every
        # class into one (n_vars, dim, dim) variable and indexing with a 4D
        # reshape/transpose -- is also correct, but lands on CVXPY's much
        # slower SCIPY canonicalisation backend as soon as any expression has
        # more than two dimensions; flattening every block into one vector
        # and applying a single constant sparse selector keeps every
        # expression here at ndim <= 2.
        classes = np.searchsorted(indices, matrix.matrix)  # (n, n) position-in-`indices`
        r_idx, c_idx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        i_idx, j_idx = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")

        out_idx = ((r_idx[:, :, None, None] * dim + i_idx[None, None, :, :]) * (n * dim)
                   + (c_idx[:, :, None, None] * dim + j_idx[None, None, :, :]))
        in_idx = (classes[:, :, None, None] * (dim * dim)
                  + i_idx[None, None, :, :] * dim + j_idx[None, None, :, :])

        selector = sp.csr_matrix(
            (np.ones(out_idx.size), (out_idx.ravel(), in_idx.ravel())),
            shape=((n * dim) ** 2, n_vars * dim * dim),
        )

        G = cp.reshape(selector @ vector, (n * dim, n * dim), order="C")

        if zero_index in position:
            pos = position[zero_index]
            constraints.append(
                vector[pos * dim * dim:(pos + 1) * dim * dim] == np.zeros(dim * dim)
            )
        if normalise_identity:
            identity_index = matrix.get((IDENTITY_LABEL,))
            if identity_index is not None and identity_index in position:
                pos = position[identity_index]
                constraints.append(
                    vector[pos * dim * dim:(pos + 1) * dim * dim] == np.eye(dim).ravel()
                )

    if psd:
        constraints.append(G + G.H >> 0)

    return CvxpyModel(G, vector, constraints, matrix, position, dim)
