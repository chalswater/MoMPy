"""Helpers for building the monomial list that generates a hierarchy.

The original workflow was to allocate integer labels by hand with a running
counter.  :class:`OperatorSet` does that bookkeeping for you while still
handing back plain integers, so everything stays compatible with code that
manipulates labels directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import product

from .algebra import Algebra

__all__ = ["generate_monomials", "OperatorSet"]


def generate_monomials(letters: Sequence[int], level: int = 1,
                       *, include_identity: bool = False) -> list:
    """All words over ``letters`` of length 1..``level``.

    Parameters
    ----------
    letters:
        The single-operator labels.
    level:
        Maximum word length.  ``level=1`` gives NPA level 1, ``level=2`` gives
        level 2, and so on.
    include_identity:
        Prepend the identity word ``(0,)``.  Off by default because the moment
        matrix adds the identity row and column itself.

    Examples
    --------
    >>> generate_monomials([1, 2], 2)
    [(1,), (2,), (1, 1), (1, 2), (2, 1), (2, 2)]
    """
    if level < 1:
        raise ValueError("level must be at least 1")
    letters = [int(x) for x in letters]
    out: list = [(0,)] if include_identity else []
    for length in range(1, level + 1):
        out.extend(product(letters, repeat=length))
    return out


class OperatorSet:
    """Allocates operator labels and records their algebraic properties.

    Wraps the "keep a counter and increment it" pattern from the original
    examples, and accumulates the rank-1 / orthogonality / commutation
    declarations so an :class:`~MoMPy.algebra.Algebra` can be produced in one
    call.

    Examples
    --------
    >>> ops = OperatorSet()
    >>> R = ops.add_family(3, idempotent=True)            # 3 pure states
    >>> M = ops.add_povm_family(2, 2, idempotent=True)    # M[y][b]
    >>> ops.declare_commuting(R, R)
    >>> alg = ops.algebra()
    >>> sorted(alg.idempotents)
    [1, 2, 3, 4, 5, 6, 7]
    """

    def __init__(self, start: int = 1) -> None:
        if start <= 0:
            raise ValueError("labels must be positive; 0 is reserved for the identity")
        self._next = int(start)
        self.labels: list = []
        self._idempotents: list = []
        self._orthogonal: list = []
        self._commuting: list = []

    # -- allocation --------------------------------------------------------

    def add(self, *, idempotent: bool = False) -> int:
        """Allocate and return a single new operator label."""
        label = self._next
        self._next += 1
        self.labels.append(label)
        if idempotent:
            self._idempotents.append(label)
        return label

    def add_family(self, count: int, *, idempotent: bool = False) -> list:
        """Allocate ``count`` independent labels, e.g. a set of states."""
        return [self.add(idempotent=idempotent) for _ in range(count)]

    def add_povm(self, n_outcomes: int, *, idempotent: bool = True,
                 orthogonal: bool = True) -> list:
        """Allocate one measurement's worth of labels.

        The outcomes are registered as an orthogonal set (and by default as
        projectors), which is the usual projective-measurement assumption.
        """
        labels = [self.add(idempotent=idempotent) for _ in range(n_outcomes)]
        if orthogonal and n_outcomes > 1:
            self._orthogonal.append(list(labels))
        return labels

    def add_povm_family(self, n_settings: int, n_outcomes: int, **kwargs) -> list:
        """Allocate ``n_settings`` measurements; returns ``M[setting][outcome]``."""
        return [self.add_povm(n_outcomes, **kwargs) for _ in range(n_settings)]
    
    
    def add_tensor(self, *shape: int, idempotent: bool = False):
        """Allocate an arbitrary-dimensional array of operator labels."""
        if not shape:
            raise ValueError("at least one dimension must be specified")
        if any(n <= 0 for n in shape):
            raise ValueError("all dimensions must be positive")
    
        def build(dim):
            if dim == len(shape):
                return self.add(idempotent=idempotent)
            return [build(dim + 1) for _ in range(shape[dim])]
    
        return build(0)
    

    # -- declarations ------------------------------------------------------

    def declare_idempotent(self, labels: Iterable[int]) -> None:
        self._idempotents.extend(int(x) for x in labels)

    def declare_orthogonal(self, labels: Iterable[int]) -> None:
        self._orthogonal.append([int(x) for x in labels])

    def declare_commuting(self, a: Iterable[int], b: Iterable[int]) -> None:
        """Declare that every label in ``a`` commutes with every label in ``b``."""
        self._commuting.append(([int(x) for x in a], [int(x) for x in b]))

    # -- output ------------------------------------------------------------

    def algebra(self) -> Algebra:
        """Bundle the declarations into an :class:`~MoMPy.algebra.Algebra`."""
        return Algebra(
            idempotents=self._idempotents,
            orthogonal_sets=self._orthogonal,
            commuting_pairs=self._commuting,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self):
        return iter(self.labels)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"OperatorSet({len(self.labels)} operators, next label {self._next})"
