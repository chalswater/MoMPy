"""Problem definitions: describe a hierarchy, then build its moment matrix."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Sequence

import numpy as np

from .algebra import IDENTITY_LABEL, Algebra, Word, as_words
from .equivalence import ZERO_CLASS, Classifier, Rewriter
from .matrix import MapTable, MomentMatrix

__all__ = ["MomentProblem"]


def _sort_key(word: Word):
    """Order words shortest-first, then lexicographically.

    Used to pick each class's representative, so that ``map_table[i][0][0]`` is
    the simplest member rather than whichever word the search happened to hit
    first.
    """
    return (len(word), word)


class _ProgressPrinter:
    """Minimal carriage-return progress line; no dependency on tqdm."""

    __slots__ = ("_label", "_total", "_last", "_stream", "_enabled")

    def __init__(self, label: str, total: int, enabled: bool, stream=None) -> None:
        self._label = label
        self._total = max(int(total), 1)
        self._last = -1
        self._stream = stream if stream is not None else sys.stderr
        self._enabled = bool(enabled)

    def update(self, done: int) -> None:
        if not self._enabled:
            return
        pct = int(done * 100 / self._total)
        if pct == self._last:
            return
        self._last = pct
        self._stream.write(f"\r{self._label}: {pct:3d}%")
        self._stream.flush()

    def close(self, message: str = "") -> None:
        if not self._enabled:
            return
        self._stream.write(f"\r{self._label}: done. {message}\n")
        self._stream.flush()


class MomentProblem:
    """A moment-matrix relaxation: describe it, then call :meth:`build`.

    One class covers every combination that used to need a different name in
    MoMPy 2.0. Two independent booleans control what gets identified during
    the build, and one mandatory size controls what each entry becomes once
    the matrix reaches :func:`~MoMPy.cvxpy_tools.to_cvxpy`.

    Parameters
    ----------
    monomials:
        The generating monomials, each a label or a sequence of labels.  Do
        *not* include the identity: it is added automatically as row/column 0.
    algebra:
        The operator relations.  An empty :class:`~MoMPy.algebra.Algebra` means
        no relations are imposed.
    dim:
        Side length of the block that will back each entry in CVXPY: declare
        it here, once, and :func:`~MoMPy.cvxpy_tools.to_cvxpy` picks it up
        automatically -- there is no separate "block" function to call later.
        ``dim=1`` is the ordinary scalar moment matrix, where each entry is a
        single number; ``dim=d>1`` makes every entry a ``d x d`` block
        instead, for relaxations whose "moments" are themselves operators on
        a Hilbert space of dimension ``d`` (the block-matrix hierarchy of
        arXiv:2603.19388). There is no default: declare the dimension your
        relaxation actually needs before building, even if that is ``1``.
    cyclicity:
        Identify a word with its cyclic rotations -- i.e. treat each entry as
        a *trace*, ``Tr(u v) == Tr(v u)``, rather than as an *operator
        product* ``u v``. Set this True only when the quantity you are
        relaxing really is a trace with the state folded into the algebra --
        typically a prepare-and-measure scenario, where ``Tr(rho_x M_b)`` is
        a genuine trace of an operator product. Leave it False for moments
        taken in a fixed external state, ``Gamma[u,v] = <psi| u v^dagger
        |psi>`` -- the usual NPA / Bell-nonlocality setting -- and for any
        block hierarchy (``dim > 1``): none of these satisfy
        ``Tr(uv) == Tr(vu)``, and imposing cyclicity anyway silently
        identifies moments that are genuinely different, which can push a
        relaxation's optimum *below* the true value, i.e. it stops being a
        valid relaxation at all. Concretely, for CHSH with monomials
        ``{A, B, AB}``:

        ==============  ==========
        ``cyclicity``   CHSH bound
        ==============  ==========
        ``True``          2.000000
        ``False``         2.828427
        ==============  ==========

        The ``cyclicity=True`` answer is below Tsirelson's bound, so it is not
        an upper bound on the quantum value at all. When unsure, leave it
        False: it imposes fewer relations, so it is never the invalid choice.
    hermitian:
        Identify a word with its reversal.  For Hermitian generators this is
        the statement that the moment matrix is real symmetric, i.e. that the
        variables are ``Re Tr(w)`` rather than ``Tr(w)``.  Leave it True
        unless you specifically want a complex Hermitian SDP.
    dedupe:
        Drop repeated monomials.  Duplicates only add linearly-dependent rows
        and columns, so this is on by default.

    Examples
    --------
    >>> from MoMPy import Algebra, MomentProblem
    >>> alg = Algebra(idempotents=[1, 2], orthogonal_sets=[[1, 2]])
    >>> mm = MomentProblem([1, 2, [1, 2]], alg, dim=1).build()
    >>> mm.index_of([1, 2]) == mm.zero_index
    True

    Migration from 2.0
    -------------------
    2.0's four classes are all this one class now, plus the new mandatory
    ``dim`` (use ``dim=1`` to reproduce every 2.0 scalar result exactly, and
    note ``tracial`` is renamed ``cyclicity``):

    - ``MomentProblem(m, a)`` -> ``MomentProblem(m, a, dim=1)``
    - ``StateMomentProblem(m, a)`` / ``NPAProblem(m, a)`` ->
      ``MomentProblem(m, a, dim=1, cyclicity=False)``
    - ``BlockMomentProblem(m, a)`` ->
      ``MomentProblem(m, a, dim=1, cyclicity=False, hermitian=False)``

    ``BlockMomentMatrix`` is likewise gone: a single :class:`MomentMatrix`
    now carries ``.dim`` and covers both cases, and
    :func:`~MoMPy.cvxpy_tools.to_cvxpy` replaces both ``to_cvxpy`` and
    ``to_cvxpy_block``.
    """

    __slots__ = ("monomials", "algebra", "cyclicity", "hermitian", "dim", "_dedupe")

    def __init__(
        self,
        monomials: Iterable,
        algebra: Algebra | None = None,
        *,
        dim: int,
        cyclicity: bool = True,
        hermitian: bool = True,
        dedupe: bool = True,
    ) -> None:
        words = as_words(monomials)
        if dedupe:
            seen: set = set()
            unique = []
            for w in words:
                if w not in seen:
                    seen.add(w)
                    unique.append(w)
            words = unique
        self.monomials: list = words
        self.algebra = algebra if algebra is not None else Algebra()
        self.cyclicity = bool(cyclicity)
        self.hermitian = bool(hermitian)
        dim = int(dim)
        if dim < 1:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        self.dim = dim
        self._dedupe = dedupe

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_levels(
        cls,
        letters: Sequence[int],
        level: int = 1,
        *,
        extra: Iterable | None = None,
        algebra: Algebra | None = None,
        **kwargs,
    ) -> MomentProblem:
        """Build from all words in ``letters`` up to length ``level``.

        ``extra`` appends further monomials of any length (the usual way to
        add a partial next level, e.g. NPA "1 + AB"). ``dim`` (and any other
        keyword :class:`MomentProblem` takes) passes straight through, so it
        still has to be supplied here -- e.g.
        ``MomentProblem.from_levels(letters, 2, dim=1, cyclicity=False)``.
        """
        from .monomials import generate_monomials

        mons = generate_monomials(letters, level)
        if extra:
            mons.extend(as_words(extra))
        return cls(mons, algebra, **kwargs)

    @property
    def n(self) -> int:
        """Side length of the moment matrix that will be built."""
        return len(self.monomials) + 1

    # -- the build ---------------------------------------------------------

    def build(self, *, progress: bool = False, progress_stream=None):
        """Construct the moment matrix.

        Parameters
        ----------
        progress:
            Print a progress line to stderr while classifying.
        """
        start = time.perf_counter()

        mons = self.monomials
        n = len(mons) + 1
        rev = [w[::-1] for w in mons]
        identity = (IDENTITY_LABEL,)

        rewriter = Rewriter(
            self.algebra, tracial=self.cyclicity, hermitian=self.hermitian
        )
        classifier = Classifier(rewriter)
        classify = classifier.classify

        raw = np.zeros((n, n), dtype=np.int64)

        # When words are identified with their reversals, entry (c, r) is the
        # reversal of entry (r, c), so the upper triangle determines the whole
        # matrix and we halve the classification work.
        symmetric = self.hermitian
        bar = _ProgressPrinter(
            "Building moment matrix", n, progress, progress_stream
        )

        # A local word -> root cache short-circuits the (very common) case of
        # the same word appearing at many positions.
        cache: dict = {}

        for r in range(n):
            row = raw[r]
            first = r if symmetric else 0
            left = identity if r == 0 else mons[r - 1]
            for c in range(first, n):
                if r == 0 and c == 0:
                    word = identity
                elif r == 0:
                    word = rev[c - 1]
                elif c == 0:
                    word = left
                else:
                    word = left + rev[c - 1]
                root = cache.get(word)
                if root is None:
                    root = classify(word)
                    cache[word] = root
                row[c] = root
            bar.update(r + 1)

        # Roots recorded during the sweep can be stale after later merges.
        resolve = classifier.resolve
        stale = np.unique(raw)
        remap = {int(s): resolve(int(s)) for s in stale}
        if any(k != v for k, v in remap.items()):
            raw = np.vectorize(remap.__getitem__, otypes=[np.int64])(raw)

        if symmetric:
            iu = np.triu_indices(n, 1)
            raw[(iu[1], iu[0])] = raw[iu]

        matrix, map_table = self._finalise(classifier, raw)

        elapsed = time.perf_counter() - start
        stats = {
            "build_seconds": elapsed,
            "distinct_words": len(classifier),
            "words_expanded": classifier.words_expanded,
            "n_classes": len(map_table),
        }
        bar.close(f"{n}x{n}, {len(np.unique(matrix))} variables, {elapsed:.2f}s")

        return MomentMatrix(
            matrix=matrix,
            monomials=mons,
            map_table=map_table,
            algebra=self.algebra,
            cyclicity=self.cyclicity,
            hermitian=self.hermitian,
            dim=self.dim,
            stats=stats,
        )

    # -- internals ---------------------------------------------------------

    def _finalise(self, classifier: Classifier, raw: np.ndarray):
        """Renumber disjoint-set roots into contiguous SDP variable indices.

        Indices are assigned in row-major order of first appearance, so the
        identity is 0 and the numbering is reproducible.  The class of zeros is
        always last, which is the convention ``map_table[-1][1]`` relies on.
        """
        zero_root = classifier.resolve(ZERO_CLASS)

        flat = raw.ravel()
        uniq, first_seen = np.unique(flat, return_index=True)
        order = [int(x) for x in uniq[np.argsort(first_seen)]]

        # Zeros go last, and are always present even if the class is empty so
        # that map_table[-1] is reliably the zero row.
        order = [r for r in order if r != zero_root]
        order.append(zero_root)

        public = {root: i for i, root in enumerate(order)}
        zero_index = public[zero_root]

        classes = classifier.classes()

        rows = []
        lookup: dict = {}
        for index, root in enumerate(order):
            members = classes.get(root, [])
            members.sort(key=_sort_key)
            rows.append([[list(w) for w in members], index])
            for w in members:
                lookup[w] = index

        map_table = MapTable(rows, lookup, zero_index)

        # Vectorised renumbering of the matrix.
        keys = np.array(sorted(public), dtype=np.int64)
        vals = np.array([public[int(k)] for k in keys], dtype=np.int64)
        matrix = vals[np.searchsorted(keys, raw)]

        return matrix, map_table

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}({len(self.monomials)} monomials, "
            f"cyclicity={self.cyclicity}, hermitian={self.hermitian}, "
            f"dim={self.dim})"
        )
