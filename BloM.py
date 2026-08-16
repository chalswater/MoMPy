"""Backwards-compatible functional API for block moment matrices.

The block hierarchy drops cyclicity of the trace: entries are operators rather
than numbers, so ``u v`` and ``v u`` are different and the first and last
letters of a word are not adjacent.

Existing scripts continue to work::

    from MoMPy.BloM import *
    [G, map_table, S, list_of_eq_indices, Mexp] = BlockMatrix(
        S_1, S_2, S_high, rank_1, orthogonal_projectors, commuting_pairs)

Fixes relative to the original module:

* ``check_if_id_BloM`` referenced an undefined name and raised ``NameError``
  on every call.  It is reimplemented here as a table lookup.
* Equivalence classes are now properly closed under commutation and
  idempotency instead of depending on the order words happened to be visited.
"""

from __future__ import annotations

from .algebra import Algebra
from .matrix import FMAP_ERROR
from .MoM import (
    Commute,
    Commute_new,
    Permute,
    fmap,
    normalisation_contraints_2compatibility,
    reverse_list,
)
from .MoM import _sites as _sites
from .problem import MomentProblem

__all__ = [
    "BlockMatrix",
    "fmap",
    "block_normalisation_contraints",
    "normalisation_contraints_2compatibility",
    "check_if_id_BloM",
    "Permute",
    "Commute",
    "Commute_new",
    "reverse_list",
]


def BlockMatrix(
    S_1,
    S_2,
    higher_order_elements,
    rank_1_projectors,
    orthogonal_projectors,
    commuting_pairs,
    progress: bool = False,
):
    """Generate a block moment matrix (no trace, so no cyclicity).

    Same arguments and same five return values as
    :func:`MoMPy.MoM.MomentMatrix`; see that docstring for details.
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
        monomials, algebra, dim=1, cyclicity=False, hermitian=False, dedupe=False
    )
    result = problem.build(progress=progress)
    return result.to_legacy()


def block_normalisation_contraints(element, list_identities_in):
    """Lists of monomials whose sum equals the final monomial in each list.

    Block-matrix counterpart of :func:`MoMPy.MoM.normalisation_contraints`;
    the implementation is shared, since normalisation does not care whether the
    entries are traces or operators.
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


def check_if_id_BloM(element, map_table, rank_1_projectors=None,
                     commuting_elements=None, orthogonal_projectors=None):
    """Check whether ``element`` coincides with a monomial already in the table.

    Returns ``[found, is_zero, index]``.  The original raised ``NameError``
    because its body referred to a ``commuting_pairs`` variable that was never
    a parameter; the trailing arguments are accepted and ignored so that old
    call sites keep working.
    """
    index = fmap(map_table, element)
    if index == FMAP_ERROR:
        return [False, False, None]
    zero_index = getattr(map_table, "_zero_index", None)
    if zero_index is None:
        zero_index = map_table[-1][1]
    return [True, index == zero_index, index]
