"""Result objects: the map table and the moment matrix itself."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .algebra import IDENTITY_LABEL, Word, as_word

__all__ = [
    "MapTable",
    "MomentMatrix",
    "UnknownMonomial",
    "FMAP_ERROR",
]

#: The sentinel the historical ``fmap`` returned on a miss.  Kept verbatim
#: because existing user scripts compare against this exact string.
FMAP_ERROR = "ERROR: The value does not appear in the mapping rule"


class UnknownMonomial(KeyError):
    """Raised when a monomial does not appear anywhere in the hierarchy."""


class MapTable(list):
    """Maps monomials to SDP variable indices.

    Subclasses :class:`list` so that it is still a sequence of
    ``[members, index]`` rows -- exactly the shape the original package
    returned, which keeps old scripts working.  The difference is the private
    hash index, which turns lookup from a linear scan over every word in every
    class into a single dict probe.

    Examples
    --------
    >>> table[[1, 2]]          # doctest: +SKIP
    7
    >>> table.index_of([1, 2]) # doctest: +SKIP
    7
    >>> table[-1][1]           # doctest: +SKIP
    23
    """

    __slots__ = ("_lookup", "_zero_index")

    def __init__(self, rows: Sequence, lookup: dict, zero_index: int) -> None:
        super().__init__(rows)
        self._lookup = lookup
        self._zero_index = int(zero_index)

    # -- lookup ------------------------------------------------------------

    def index_of(self, monomial) -> int:
        """Return the SDP variable index of ``monomial``.

        Raises :class:`UnknownMonomial` if it is not in the hierarchy.
        """
        try:
            return self._lookup[as_word(monomial)]
        except KeyError:
            raise UnknownMonomial(
                f"monomial {list(as_word(monomial))!r} does not appear in the "
                f"hierarchy; add it to the monomial list or check the labels"
            ) from None

    def get(self, monomial, default=None):
        """Like :meth:`index_of` but returns ``default`` instead of raising."""
        return self._lookup.get(as_word(monomial), default)

    def __contains__(self, item) -> bool:
        try:
            key = as_word(item)
        except TypeError:
            return False
        return key in self._lookup

    def __getitem__(self, key):
        # Integers and slices keep list semantics (``table[-1][1]`` etc.);
        # anything word-like is treated as a monomial lookup.
        if isinstance(key, (int, np.integer, slice)):
            return list.__getitem__(self, key)
        return self.index_of(key)

    def __call__(self, monomial) -> int:
        return self.index_of(monomial)

    # -- introspection -----------------------------------------------------

    @property
    def zero_index(self) -> int:
        """Index of the class of monomials that are identically zero."""
        return self._zero_index

    @property
    def n_variables(self) -> int:
        return len(self)

    def words(self) -> Iterable[Word]:
        return self._lookup.keys()

    def members(self, index: int) -> list:
        """All monomials sharing the SDP variable ``index``."""
        return list.__getitem__(self, index)[0]


class MomentMatrix:
    """A built moment matrix and everything needed to use it in an SDP.

    Attributes
    ----------
    matrix : numpy.ndarray
        Integer array of shape ``(n, n)``.  Entry ``[r, c]`` is the SDP
        variable index for ``Tr(u_r u_c†)`` (or, when ``dim > 1``, for the
        class whose ``dim x dim`` block sits at that position).
    monomials : list[Word]
        The generating monomials, *excluding* the identity that occupies row
        and column 0.
    words : list[list[Word]]
        ``words[r][c]`` is the explicit operator word behind ``matrix[r, c]``.
        Built lazily; prefer :meth:`word_at` for one-off access.
    map_table : MapTable
        Monomial -> variable index.
    cyclicity : bool
        Whether the build identified words with their cyclic rotations.  See
        :class:`~MoMPy.problem.MomentProblem`.
    hermitian : bool
        Whether the build identified words with their reversal.
    dim : int
        Side length of the block :func:`~MoMPy.cvxpy_tools.to_cvxpy` gives
        each entry.  ``1`` means a plain scalar moment matrix.
    """

    __slots__ = (
        "matrix",
        "monomials",
        "map_table",
        "algebra",
        "cyclicity",
        "hermitian",
        "dim",
        "_stats",
        "_words_cache",
    )

    def __init__(self, matrix, monomials, map_table, algebra, cyclicity: bool,
                 hermitian: bool, dim: int, stats: dict | None = None) -> None:
        self.matrix = matrix
        self.monomials = monomials
        self.map_table = map_table
        self.algebra = algebra
        self.cyclicity = cyclicity
        self.hermitian = hermitian
        self.dim = dim
        self._stats = stats or {}
        self._words_cache = None

    # -- explicit words ----------------------------------------------------

    def word_at(self, r: int, c: int) -> Word:
        """The operator word behind ``matrix[r, c]``, computed on demand.

        Row/column 0 hold the identity, so ``word_at(r, c)`` is
        ``monomials[r-1] + reversed(monomials[c-1])``.
        """
        mons = self.monomials
        n = len(mons) + 1
        if not (0 <= r < n and 0 <= c < n):
            raise IndexError(f"index ({r}, {c}) out of range for {n}x{n} matrix")
        if r == 0 and c == 0:
            return (IDENTITY_LABEL,)
        if r == 0:
            return mons[c - 1][::-1]
        if c == 0:
            return mons[r - 1]
        return mons[r - 1] + mons[c - 1][::-1]

    @property
    def words(self) -> list:
        """Full ``n x n`` nested list of explicit words.

        Materialised lazily and cached: for a large hierarchy this is the
        single biggest object in the build, and most callers never need it.
        """
        if self._words_cache is None:
            n = self.n
            self._words_cache = [
                [self.word_at(r, c) for c in range(n)] for r in range(n)
            ]
        return self._words_cache

    # -- basic properties --------------------------------------------------

    def __len__(self) -> int:
        return self.matrix.shape[0]

    @property
    def n(self) -> int:
        """Side length of the matrix (number of monomials plus the identity)."""
        return int(self.matrix.shape[0])

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def zero_index(self) -> int:
        return self.map_table.zero_index

    @property
    def variable_indices(self) -> np.ndarray:
        """Sorted array of the variable indices that actually occur."""
        return np.unique(self.matrix)

    @property
    def n_variables(self) -> int:
        """Number of distinct SDP variables in the matrix."""
        return int(self.variable_indices.size)

    @property
    def has_zeros(self) -> bool:
        return bool((self.matrix == self.zero_index).any())

    @property
    def stats(self) -> dict:
        """Build diagnostics (timings, words expanded, ...)."""
        return dict(self._stats)

    # -- lookup ------------------------------------------------------------

    def index_of(self, monomial) -> int:
        """SDP variable index of ``monomial``.  Raises :class:`UnknownMonomial`."""
        return self.map_table.index_of(monomial)

    def get(self, monomial, default=None):
        return self.map_table.get(monomial, default)

    def __getitem__(self, key):
        """``mm[[1, 2]]`` -> variable index; ``mm[0, 3]`` -> matrix entry."""
        if isinstance(key, tuple) and len(key) == 2 and all(
            isinstance(k, (int, np.integer)) for k in key
        ):
            return int(self.matrix[key])
        return self.index_of(key)

    def __contains__(self, monomial) -> bool:
        return monomial in self.map_table

    @property
    def identity_index(self) -> int:
        """Variable index of ``Tr(1)``, normally constrained to 1."""
        return self.map_table.index_of((IDENTITY_LABEL,))

    def equivalents(self, monomial) -> list:
        """Every monomial known to be equal to ``monomial``."""
        return self.map_table.members(self.index_of(monomial))

    # -- constraints -------------------------------------------------------

    def normalisation_constraints(self, povm, *, dedupe: bool = True):
        """Constraints from ``sum(povm) == identity``.  See :mod:`MoMPy.constraints`."""
        from .constraints import normalisation_constraints

        return normalisation_constraints(self, povm, dedupe=dedupe)

    def marginal_constraints(self, joint, marginal, *, dedupe: bool = True):
        """Constraints from ``sum(joint) == marginal``.  See :mod:`MoMPy.constraints`."""
        from .constraints import marginal_constraints

        return marginal_constraints(self, joint, marginal, dedupe=dedupe)

    # -- SDP interfaces ----------------------------------------------------

    def to_cvxpy(self, **kwargs):
        """Build a CVXPY model for this matrix.  See :mod:`MoMPy.cvxpy_tools`."""
        from .cvxpy_tools import to_cvxpy

        return to_cvxpy(self, **kwargs)

    # -- interop -----------------------------------------------------------

    def to_legacy(self):
        """Return the 5-tuple the pre-2.0 functional API returned."""
        explicit = [[list(w) for w in row] for row in self.words]
        monomials = [list(w) for w in self.monomials]
        return (
            self.matrix,
            self.map_table,
            monomials,
            self.variable_indices,
            explicit,
        )

    def summary(self) -> str:
        """A short human-readable report on the build."""
        n = self.n
        entries = n * n
        dim_line = (
            f"  block size (dim)   : {self.dim}"
            if self.dim == 1
            else f"  block size (dim)   : {self.dim} ({self.dim}x{self.dim} per entry)"
        )
        lines = [
            f"{type(self).__name__}: {n} x {n} "
            f"({len(self.monomials)} monomials + identity)",
            dim_line,
            f"  SDP variables      : {self.n_variables}",
            f"  compression        : {entries} entries -> {self.n_variables} "
            f"variables ({entries / max(self.n_variables, 1):.1f}x)",
            f"  zero entries       : {int((self.matrix == self.zero_index).sum())}",
            f"  distinct words seen: {self._stats.get('distinct_words', '?')}",
        ]
        if "build_seconds" in self._stats:
            lines.append(f"  build time         : {self._stats['build_seconds']:.3f} s")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<{type(self).__name__} {self.n}x{self.n}, "
            f"{self.n_variables} variables, dim={self.dim}>"
        )
