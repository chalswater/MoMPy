"""Backwards-compatible functional API for tracial moment matrices.

Existing scripts that do::

    from MoMPy.MoM import *
    [G, map_table, S, list_of_eq_indices, Mexp] = MomentMatrix(
        S_1, S_2, S_high, rank_1, orthogonal_projectors, commuting_pairs)

keep working.  Everything here is a thin wrapper over the current engine, so
you get the speed-ups without touching your code.

Two behavioural fixes are worth knowing about, because they change results:

* ``G[r][c]`` and ``fmap(map_table, Mexp[r][c])`` now always agree.  Previously
  the matrix was symmetrised after the fact while the equivalence classes were
  not, so the variable you looked up was sometimes not the variable sitting in
  the matrix.
* Equivalence classes are now closed under all the declared relations, not just
  the ones the old control flow happened to reach.  Classes therefore merge
  more often, which means fewer SDP variables and a tighter relaxation.

Both changes make the relaxation *more* correct.  If you need to reproduce old
numbers exactly, pin the previous release.
"""

from __future__ import annotations

from .algebra import Algebra, as_word
from .matrix import FMAP_ERROR
from .problem import MomentProblem

__all__ = [
    "MomentMatrix",
    "fmap",
    "normalisation_contraints",
    "normalisation_contraints_2compatibility",
    "check_if_id",
    "Permute",
    "Commute",
    "Commute_new",
    "reverse_list",
]


# ---------------------------------------------------------------------------
# Small utilities kept for source compatibility
# ---------------------------------------------------------------------------


def Permute(v):
    """Cyclic permutation: move the last entry to the front."""
    return [v[-1]] + list(v[:-1])


def reverse_list(lista):
    """Reverse a list."""
    return list(reversed(lista))


def Commute_new(vec, i, j):
    """Return a copy of ``vec`` with positions ``i`` and ``j`` interchanged."""
    n = len(vec)
    if i < 0 or j < 0 or i >= n or j >= n:
        raise IndexError("Index out of range.")
    out = list(vec)
    out[i], out[j] = out[j], out[i]
    return out


def Commute(v, index):
    """Swap ``v[index]`` with the entry after it, wrapping cyclically.

    Reimplemented directly; the original removed the element *by value*, which
    silently misbehaved when a word contained repeated labels.
    """
    n = len(v)
    if n == 0:
        return []
    index %= n
    out = list(v)
    if index == n - 1:
        return [out[-1]] + out[:-1]
    out[index], out[index + 1] = out[index + 1], out[index]
    return out


def fmap(table, i):
    """Map a monomial to its SDP variable index.

    Returns the historical error string rather than raising, so the guard
    pattern in the old examples still works::

        if fmap(map_table, w) == 'ERROR: The value does not appear in the mapping rule':
            ...

    With a :class:`~MoMPy.matrix.MapTable` this is a single dict lookup.  A
    plain list of ``[members, index]`` rows still works via a linear scan.
    """
    lookup = getattr(table, "_lookup", None)
    if lookup is not None:
        return lookup.get(as_word(i), FMAP_ERROR)

    # Fallback for hand-built tables.
    key = list(i) if not isinstance(i, list) else i
    for members, index in table:
        if key in members:
            return index
    return FMAP_ERROR


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def MomentMatrix(
    S_1,
    S_2,
    higher_order_elements,
    rank_1_projectors,
    orthogonal_projectors,
    commuting_pairs,
    progress: bool = False,
):
    """Generate a tracial SDP moment matrix.

    Parameters
    ----------
    S_1:
        First-order operator labels (the identity is added automatically).
    S_2:
        Labels whose ordered pairs form the second-order monomials.  Pass
        ``[]`` to skip.
    higher_order_elements:
        Explicit monomials of any length, each a list of labels.
    rank_1_projectors:
        Labels obeying ``P @ P == P``.
    orthogonal_projectors:
        Lists of mutually orthogonal labels.
    commuting_pairs:
        Pairs ``[A, B]`` of label lists; every element of ``A`` commutes with
        every element of ``B``.
    progress:
        Print a progress line while building.

    Returns
    -------
    (Moment_Matrix, map_table, S, list_of_eq_indices, Mexp)
        Same five outputs as before.  ``map_table`` is now a
        :class:`~MoMPy.matrix.MapTable`, which still behaves as the old list of
        ``[members, index]`` rows but supports O(1) lookup.
    """
    monomials = [[v] for v in S_1]
    monomials += [[h, k] for k in S_2 for h in S_2]
    monomials += [list(w) for w in higher_order_elements]

    algebra = Algebra(
        idempotents=rank_1_projectors,
        orthogonal_sets=orthogonal_projectors,
        commuting_pairs=commuting_pairs,
    )

    problem = MomentProblem(
        monomials, algebra, dim=1, cyclicity=True, hermitian=True, dedupe=False
    )
    result = problem.build(progress=progress)
    return result.to_legacy()


# ---------------------------------------------------------------------------
# Constraint generators
# ---------------------------------------------------------------------------


def _sites(list_identities, targets):
    """Yield ``(word, position)`` for each occurrence of a target label.

    ``list_identities`` is the old ``[term[0] for term in map_table]`` shape: a
    list of equivalence groups.  Every word of every group is inspected, not
    just the group representative -- that omission was why the old function
    missed constraints whenever the representative happened not to contain the
    operator being normalised.
    """
    seen = set()
    for group in list_identities:
        if not group:
            continue
        words = group if isinstance(group[0], (list, tuple)) else [group]
        for word in words:
            for pos, label in enumerate(word):
                if label in targets:
                    key = (tuple(word[:pos]), tuple(word[pos + 1:]))
                    if key not in seen:
                        seen.add(key)
                        yield list(word), pos


def normalisation_contraints(element, list_identities_in):
    """Lists of monomials whose sum equals the final monomial in each list.

    ``element`` is a POVM: the labels of one measurement's outcomes, which sum
    to the identity.  Each returned entry is
    ``[w_with_M0, w_with_M1, ..., w_without]`` of length ``len(element) + 1``.

    Note
    ----
    This now scans every monomial of every equivalence class and deduplicates,
    so it returns a complete, non-redundant set.  The old version looked only
    at ``term[0]`` and matched only ``element[0]``.
    """
    element = list(element)
    targets = frozenset(element)
    output = []
    for word, pos in _sites(list_identities_in, targets):
        block = []
        for label in element:
            replaced = list(word)
            replaced[pos] = label
            block.append(replaced)
        without = word[:pos] + word[pos + 1:]
        if not without:
            without = [0]
        block.append(without)
        output.append(block)
    return output


def normalisation_contraints_2compatibility(Belement, Melement, list_identities_in):
    """Marginalisation constraints for joint measurability.

    ``Belement`` are the outcomes of a joint POVM whose marginal is the single
    operator ``Melement``.  Each returned entry is
    ``[w_with_B0, ..., w_with_Bk, w_with_M]``.
    """
    Belement = list(Belement)
    targets = frozenset(Belement)
    output = []
    for word, pos in _sites(list_identities_in, targets):
        block = []
        for label in Belement:
            replaced = list(word)
            replaced[pos] = label
            block.append(replaced)
        marginal = list(word)
        marginal[pos] = Melement
        block.append(marginal)
        output.append(block)
    return output


def check_if_id(element, map_table, rank_1_projectors, commuting_elements,
                orthogonal_projectors):
    """Check whether ``element`` coincides with a monomial already in the table.

    Returns ``[found, is_zero, index]``.

    Kept for compatibility; :meth:`MoMPy.MapTable.get` is the direct
    replacement and is a single dict lookup.
    """
    index = fmap(map_table, element)
    if index == FMAP_ERROR:
        return [False, False, None]
    zero_index = getattr(map_table, "_zero_index", None)
    if zero_index is None:
        zero_index = map_table[-1][1]
    return [True, index == zero_index, index]
