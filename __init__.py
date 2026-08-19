"""MoMPy -- moment matrices for SDP hierarchy relaxations.

Quick start
-----------
Describe your operators, declare their algebraic properties, declare the
block size, and build::

    from MoMPy import OperatorSet, MomentProblem

    ops = OperatorSet()
    R = ops.add_family(3, idempotent=True)          # three pure states
    M = ops.add_povm_family(2, 2)                   # M[y][b], projective
    ops.declare_commuting(R, R)                     # the states commute

    monomials  = list(R) + [m for row in M for m in row]
    monomials += [[R[x], M[y][b]] for x in range(3)
                  for y in range(2) for b in range(2)]

    mm = MomentProblem(monomials, ops.algebra(), dim=1).build()

    print(mm.summary())
    mm.index_of([R[0], M[1][0]])        # the SDP variable for Tr(R0 M10)

``dim`` is the side length of the block behind each entry: ``dim=1`` above is
the ordinary scalar moment matrix; set ``dim=d`` for a block hierarchy whose
moments are ``d x d`` operators; ``to_cvxpy`` builds ``d x d`` blocks
automatically either way. ``cyclicity`` and ``hermitian`` are the other two
flags -- see :class:`~MoMPy.problem.MomentProblem` for when to set each.

Then hand it to a solver::

    model = mm.to_cvxpy()
    constraints = model.constraints + model.apply(
        mm.normalisation_constraints(M[0])
    )

The pre-2.0 functional interface still works unchanged::

    from MoMPy.MoM import MomentMatrix, fmap, normalisation_contraints

See ``MIGRATION.md`` for what changed and why.
"""

from __future__ import annotations

from .algebra import IDENTITY_LABEL, Algebra, Word, as_word, as_words
from .constraints import (
    LinearConstraint,
    marginal_constraints,
    normalisation_constraints,
)
from .equivalence import Classifier, Rewriter, UnionFind
from .matrix import (
    FMAP_ERROR,
    MapTable,
    MomentMatrix,
    UnknownMonomial,
)
from .monomials import OperatorSet, generate_monomials
from .problem import MomentProblem

__version__ = "1.1.0"

__all__ = [
    "__version__",
    # describing a problem
    "Algebra",
    "OperatorSet",
    "generate_monomials",
    "MomentProblem",
    # results
    "MomentMatrix",
    "MapTable",
    "UnknownMonomial",
    # constraints
    "LinearConstraint",
    "normalisation_constraints",
    "marginal_constraints",
    # words and internals
    "Word",
    "as_word",
    "as_words",
    "IDENTITY_LABEL",
    "Rewriter",
    "Classifier",
    "UnionFind",
    "FMAP_ERROR",
]


def __getattr__(name):
    # Lazily expose the CVXPY helpers so that importing MoMPy never requires
    # CVXPY to be installed.
    if name in ("to_cvxpy", "CvxpyModel"):
        from . import cvxpy_tools

        return getattr(cvxpy_tools, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
