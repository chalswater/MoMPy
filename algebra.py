"""Operator algebra: the relations that words of operators are quotiented by.

A *word* is a tuple of integer operator labels, read left to right as an
operator product.  The label ``0`` is reserved for the identity.

An :class:`Algebra` records the structural relations obeyed by the operators:

``idempotents``
    Labels ``P`` with ``P @ P == P`` (rank-1 projectors, or any projector).

``orthogonal_sets``
    Groups ``{P_0, ..., P_k}`` with ``P_i @ P_j == 0`` for ``i != j``.
    Membership of an orthogonal set does *not* by itself imply idempotency;
    list the labels in ``idempotents`` too if they are projectors.

``commuting_pairs``
    Pairs ``(A, B)`` of label collections such that every ``a in A`` commutes
    with every ``b in B``.  Pass ``(A, A)`` to say that all of ``A`` commutes
    with itself.

The class is immutable once constructed and exposes O(1) lookup tables that the
rewriting engine uses in its inner loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

__all__ = ["Algebra", "Word", "as_word", "as_words", "IDENTITY_LABEL"]

#: A word is a tuple of integer operator labels.
Word = tuple

#: Reserved label denoting the identity operator.
IDENTITY_LABEL = 0


def as_word(obj) -> Word:
    """Coerce ``obj`` to a word (tuple of ints).

    Accepts a bare label (``3`` -> ``(3,)``) or any iterable of labels.
    """
    if isinstance(obj, int):
        return (obj,)
    return tuple(obj)


def as_words(objs: Iterable) -> list:
    """Coerce an iterable of monomials to a list of words."""
    return [as_word(o) for o in objs]


class Algebra:
    """Structural relations obeyed by a set of operators.

    Parameters
    ----------
    idempotents:
        Labels satisfying ``P @ P == P``.
    orthogonal_sets:
        Iterable of label collections that are mutually orthogonal.
    commuting_pairs:
        Iterable of ``(A, B)`` label-collection pairs; every element of ``A``
        commutes with every element of ``B``.

    Examples
    --------
    >>> alg = Algebra(idempotents=[1, 2, 3],
    ...               orthogonal_sets=[[2, 3]],
    ...               commuting_pairs=[([1], [1])])
    >>> alg.are_orthogonal(2, 3)
    True
    >>> alg.commute(1, 2)
    False
    """

    __slots__ = (
        "idempotents",
        "orthogonal_sets",
        "commuting_pairs",
        "_ortho",
        "_commutes",
        "_trivial",
    )

    def __init__(
        self,
        idempotents: Iterable[int] = (),
        orthogonal_sets: Iterable[Iterable[int]] = (),
        commuting_pairs: Iterable[Sequence[Iterable[int]]] = (),
    ) -> None:
        self.idempotents = frozenset(int(x) for x in idempotents)
        self.orthogonal_sets = tuple(
            frozenset(int(x) for x in group) for group in orthogonal_sets
        )
        self.commuting_pairs = tuple(
            (tuple(int(x) for x in a), tuple(int(x) for x in b))
            for a, b in commuting_pairs
        )

        # label -> set of labels it is orthogonal to (O(1) adjacency test)
        ortho: dict[int, set[int]] = {}
        for group in self.orthogonal_sets:
            for a in group:
                bucket = ortho.setdefault(a, set())
                bucket.update(group)
                bucket.discard(a)
        self._ortho: Mapping[int, frozenset[int]] = {
            k: frozenset(v) for k, v in ortho.items() if v
        }

        # label -> set of labels it commutes with (O(1) adjacency test)
        commutes: dict[int, set[int]] = {}
        for a_list, b_list in self.commuting_pairs:
            for a in a_list:
                bucket = commutes.setdefault(a, set())
                bucket.update(b_list)
            for b in b_list:
                bucket = commutes.setdefault(b, set())
                bucket.update(a_list)
        self._commutes: Mapping[int, frozenset[int]] = {
            k: frozenset(v) for k, v in commutes.items() if v
        }

        self._trivial = not (self.idempotents or self._ortho or self._commutes)

    # -- introspection -----------------------------------------------------

    @property
    def is_trivial(self) -> bool:
        """True when no relation at all was declared (free algebra)."""
        return self._trivial

    def is_idempotent(self, label: int) -> bool:
        return label in self.idempotents

    def are_orthogonal(self, a: int, b: int) -> bool:
        """True when ``a @ b == 0``, i.e. distinct members of one orthogonal set."""
        if a == b:
            return False
        partners = self._ortho.get(a)
        return partners is not None and b in partners

    def commute(self, a: int, b: int) -> bool:
        """True when ``a @ b == b @ a`` was declared."""
        if a == b:
            return True
        partners = self._commutes.get(a)
        return partners is not None and b in partners

    # -- internal fast tables ---------------------------------------------

    @property
    def ortho_table(self) -> Mapping[int, frozenset[int]]:
        return self._ortho

    @property
    def commute_table(self) -> Mapping[int, frozenset[int]]:
        return self._commutes

    # -- convenience -------------------------------------------------------

    def with_(self, **changes) -> Algebra:
        """Return a copy with some fields replaced."""
        return Algebra(
            idempotents=changes.get("idempotents", self.idempotents),
            orthogonal_sets=changes.get("orthogonal_sets", self.orthogonal_sets),
            commuting_pairs=changes.get("commuting_pairs", self.commuting_pairs),
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Algebra(idempotents={sorted(self.idempotents)}, "
            f"orthogonal_sets={[sorted(g) for g in self.orthogonal_sets]}, "
            f"commuting_pairs={[(list(a), list(b)) for a, b in self.commuting_pairs]})"
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Algebra):
            return NotImplemented
        return (
            self.idempotents == other.idempotents
            and set(self.orthogonal_sets) == set(other.orthogonal_sets)
            and self._commutes == other._commutes
        )

    def __hash__(self) -> int:
        return hash((self.idempotents, self.orthogonal_sets))
