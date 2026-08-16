"""Linear constraints that the moment matrix cannot encode by itself.

Positivity and the operator relations are baked into the matrix by
:mod:`MoMPy.problem`.  Normalisation is not: ``sum_b M_b == 1`` is a linear
relation *between* SDP variables, so it has to be added to the program.

:func:`normalisation_constraints` finds every place a POVM appears inside a
known monomial and emits the corresponding relation.  Unlike the original
implementation it scans every monomial in every class -- not just the class
representative -- and deduplicates the result, so it produces a complete set of
constraints without flooding the solver with copies of the same one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .algebra import IDENTITY_LABEL, Word

__all__ = [
    "LinearConstraint",
    "normalisation_constraints",
    "marginal_constraints",
]


class LinearConstraint:
    """``sum(variables at lhs) == variable at rhs``.

    Attributes
    ----------
    lhs : tuple[int, ...]
        Variable indices to be summed.
    rhs : int
        Variable index the sum must equal.
    words : tuple
        The monomials the constraint came from, for debugging.
    """

    __slots__ = ("lhs", "rhs", "words")

    def __init__(self, lhs: Sequence[int], rhs: int, words=()) -> None:
        self.lhs = tuple(int(x) for x in lhs)
        self.rhs = int(rhs)
        self.words = tuple(words)

    @property
    def key(self):
        """Canonical identity of the constraint, used for deduplication."""
        return (tuple(sorted(self.lhs)), self.rhs)

    def is_trivial(self) -> bool:
        """True when the constraint says nothing (a single term equal to itself)."""
        return len(self.lhs) == 1 and self.lhs[0] == self.rhs

    def apply(self, variables):
        """Return ``sum(variables[i] for i in lhs) == variables[rhs]``.

        Works with any mapping whose values support ``+`` and ``==``, which
        includes a CVXPY variable dict.
        """
        total = variables[self.lhs[0]]
        for i in self.lhs[1:]:
            total = total + variables[i]
        return total == variables[self.rhs]

    def __iter__(self):
        return iter((self.lhs, self.rhs))

    def __eq__(self, other):
        if not isinstance(other, LinearConstraint):
            return NotImplemented
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"LinearConstraint(sum{list(self.lhs)} == {self.rhs})"


def _substitution_sites(words: Iterable[Word], targets: frozenset):
    """Yield ``(prefix, suffix)`` for every position holding a target label.

    A ``(prefix, suffix)`` pair is a monomial with one slot punched out.  Two
    different words that punch out to the same pair generate the same
    constraint, so yielding the hole rather than the word is what makes
    deduplication cheap.
    """
    seen: set = set()
    for word in words:
        for i, label in enumerate(word):
            if label in targets:
                hole = (word[:i], word[i + 1:])
                if hole not in seen:
                    seen.add(hole)
                    yield hole


def _resolve(matrix, word: Word):
    """Variable index for ``word``, or None if it is not in the hierarchy."""
    if not word:
        word = (IDENTITY_LABEL,)
    return matrix.map_table.get(word)


def normalisation_constraints(matrix, povm: Sequence, *, dedupe: bool = True):
    """Constraints implied by ``sum(povm) == identity``.

    For every known monomial containing a POVM element at some position, the
    sum over outcomes at that position equals the monomial with the position
    deleted.

    Parameters
    ----------
    matrix:
        A built :class:`~MoMPy.matrix.MomentMatrix`.
    povm:
        The labels of one measurement's outcomes, summing to the identity.
    dedupe:
        Drop repeated and trivial constraints.

    Returns
    -------
    list[LinearConstraint]

    Examples
    --------
    >>> cts = mm.normalisation_constraints(M[0])        # doctest: +SKIP
    >>> problem_constraints = [c.apply(variables) for c in cts]  # doctest: +SKIP
    """
    labels = [int(x) for x in povm]
    if len(labels) < 2:
        raise ValueError("a POVM needs at least two outcomes to normalise")
    targets = frozenset(labels)

    out: list = []
    emitted: set = set()
    for prefix, suffix in _substitution_sites(matrix.map_table.words(), targets):
        lhs = []
        ok = True
        for label in labels:
            idx = _resolve(matrix, prefix + (label,) + suffix)
            if idx is None:
                ok = False
                break
            lhs.append(idx)
        if not ok:
            continue
        rhs = _resolve(matrix, prefix + suffix)
        if rhs is None:
            continue
        ct = LinearConstraint(lhs, rhs, words=(prefix, suffix))
        if dedupe:
            if ct.is_trivial() or ct.key in emitted:
                continue
            emitted.add(ct.key)
        out.append(ct)
    return out


def marginal_constraints(matrix, joint: Sequence, marginal, *, dedupe: bool = True):
    """Constraints implied by ``sum(joint) == marginal``.

    The joint-measurability workhorse: ``joint`` is the list of labels of a
    parent POVM's outcomes that marginalise onto the single operator
    ``marginal``.

    This is the corrected successor to the original
    ``normalisation_contraints_2compatibility``.
    """
    labels = [int(x) for x in joint]
    if len(labels) < 2:
        raise ValueError("need at least two joint outcomes to marginalise")
    marginal = int(marginal)
    targets = frozenset(labels)

    out: list = []
    emitted: set = set()
    for prefix, suffix in _substitution_sites(matrix.map_table.words(), targets):
        lhs = []
        ok = True
        for label in labels:
            idx = _resolve(matrix, prefix + (label,) + suffix)
            if idx is None:
                ok = False
                break
            lhs.append(idx)
        if not ok:
            continue
        rhs = _resolve(matrix, prefix + (marginal,) + suffix)
        if rhs is None:
            continue
        ct = LinearConstraint(lhs, rhs, words=(prefix, suffix))
        if dedupe:
            if ct.is_trivial() or ct.key in emitted:
                continue
            emitted.add(ct.key)
        out.append(ct)
    return out
