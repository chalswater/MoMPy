"""Word rewriting and equivalence-class discovery.

This module is the performance-critical core of MoMPy.  It answers one
question fast: *given a word of operators, which SDP variable does it belong
to?*

Two words are identified when one can be transformed into the other by a
sequence of the moves in :class:`Rewriter`:

============  ========================================================
move          condition
============  ========================================================
rotate        tracial mode only; ``Tr(u v) == Tr(v u)``
reverse       hermitian mode only; ``Re Tr(w) == Re Tr(w†)``
swap          adjacent labels declared commuting
collapse      adjacent equal labels that are idempotent (``P P -> P``)
zero          adjacent distinct labels from one orthogonal set
============  ========================================================

Algorithm
---------
:class:`Classifier` performs a breadth-first exploration of the moves from
each query word, then unions the result into a disjoint-set forest.  Two
properties make this near-linear in the total number of distinct words rather
than quadratic in the number of matrix entries:

1. **Memoisation.** Every word ever reached is recorded in a dict.  When a
   search meets an already-classified word it stops expanding there and simply
   unions the two classes.  This is sound because a word only enters the dict
   after its own neighbourhood has been fully explored, so its class is
   already closed under the moves.

2. **Union-find.** Classes are merged in near-constant amortised time, so a
   late discovery that two long-separate classes coincide costs nothing.  The
   *zero* class is a sticky root: unioning anything with it makes the whole
   merged class zero, which is how zeros propagate backwards through
   previously-classified words.

The result is that each distinct word is expanded exactly once over the whole
build, no matter how many matrix entries reference it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .algebra import Algebra, Word

__all__ = ["Rewriter", "Classifier", "UnionFind", "ZERO_CLASS"]

#: Reserved disjoint-set id for the class of words that are identically zero.
ZERO_CLASS = 0


class UnionFind:
    """Disjoint-set forest with path halving and union by rank.

    ``ZERO_CLASS`` is a sticky root: any union involving it keeps it as the
    representative, so "this class is zero" is never lost by a later merge.
    """

    __slots__ = ("_parent", "_rank")

    def __init__(self) -> None:
        self._parent: list[int] = [ZERO_CLASS]
        self._rank: list[int] = [0]

    def new(self) -> int:
        """Allocate a fresh singleton class and return its id."""
        idx = len(self._parent)
        self._parent.append(idx)
        self._rank.append(0)
        return idx

    def find(self, x: int) -> int:
        parent = self._parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        """Merge the classes of ``a`` and ``b``; return the surviving root."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        # The zero class always wins, whatever the ranks say.
        if ra == ZERO_CLASS:
            self._parent[rb] = ra
            return ra
        if rb == ZERO_CLASS:
            self._parent[ra] = rb
            return rb
        rank = self._rank
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        return ra

    def __len__(self) -> int:
        return len(self._parent)


class Rewriter:
    """Generates the one-step moves available on a word.

    Parameters
    ----------
    algebra:
        The operator relations.
    tracial:
        If True, words stand for traces, so cyclic rotations are equivalent
        and the last/first letters count as adjacent.  Set False for block
        moment matrices, whose entries are operators rather than numbers.
    hermitian:
        If True, a word is identified with its reversal.  For a tracial matrix
        of Hermitian operators this is exactly the statement that the moment
        matrix is real symmetric, i.e. that we work with ``Re Tr(w)``.
    """

    __slots__ = ("algebra", "tracial", "hermitian", "_idem", "_ortho", "_comm")

    def __init__(self, algebra: Algebra, *, tracial: bool = True,
                 hermitian: bool = True) -> None:
        self.algebra = algebra
        self.tracial = bool(tracial)
        self.hermitian = bool(hermitian)
        # Bind lookup tables to slots: attribute access in the inner loop is
        # measurably hotter than anything else in the build.
        self._idem = algebra.idempotents
        self._ortho = algebra.ortho_table
        self._comm = algebra.commute_table

    # -- predicates --------------------------------------------------------

    def is_zero(self, w: Word) -> bool:
        """True when two adjacent letters are distinct members of one orthogonal set."""
        n = len(w)
        if n < 2:
            return False
        ortho = self._ortho
        if not ortho:
            return False
        for i in range(n - 1):
            a = w[i]
            b = w[i + 1]
            if a != b:
                partners = ortho.get(a)
                if partners is not None and b in partners:
                    return True
        if self.tracial and n > 2:
            a = w[-1]
            b = w[0]
            if a != b:
                partners = ortho.get(a)
                if partners is not None and b in partners:
                    return True
        return False

    # -- moves -------------------------------------------------------------

    def neighbours(self, w: Word) -> Iterator[Word]:
        """Yield every word reachable from ``w`` by a single move.

        Wrap-around adjacency in tracial mode is *not* enumerated directly:
        one rotation exposes the wrapped pair as an ordinary adjacent pair, and
        the breadth-first search closes over rotations anyway.  Emitting a
        single rotation instead of all ``n`` of them keeps the branching factor
        low without changing the reachable set.
        """
        n = len(w)
        if n < 2:
            return

        if self.tracial:
            yield w[1:] + w[:1]
        if self.hermitian:
            yield w[::-1]

        idem = self._idem
        comm = self._comm
        for i in range(n - 1):
            a = w[i]
            b = w[i + 1]
            if a == b:
                if a in idem:
                    yield w[:i] + w[i + 1:]          # P P -> P
            else:
                partners = comm.get(a)
                if partners is not None and b in partners:
                    yield w[:i] + (b, a) + w[i + 2:]  # a b -> b a

    def reduce(self, w: Word) -> Word:
        """Apply idempotent collapses until none remain.

        A cheap pre-normalisation.  It is confluent on its own but *not* when
        combined with commutation, so it is a shortcut rather than a canonical
        form -- :class:`Classifier` still explores the full class.
        """
        idem = self._idem
        if not idem:
            return w
        changed = True
        while changed and len(w) > 1:
            changed = False
            for i in range(len(w) - 1):
                if w[i] == w[i + 1] and w[i] in idem:
                    w = w[:i] + w[i + 1:]
                    changed = True
                    break
        return w


class Classifier:
    """Assigns words to equivalence classes, memoising as it goes.

    The public entry point is :meth:`classify`.  Class ids returned are
    disjoint-set roots and may change as later merges happen, so always call
    :meth:`resolve` (or use :meth:`finalise`) before comparing them.
    """

    __slots__ = ("rewriter", "_class_of", "_members", "_uf", "_expansions")

    def __init__(self, rewriter: Rewriter) -> None:
        self.rewriter = rewriter
        self._class_of: dict[Word, int] = {}
        self._members: dict[int, list[Word]] = {ZERO_CLASS: []}
        self._uf = UnionFind()
        self._expansions = 0

    # -- queries -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._class_of)

    @property
    def words_expanded(self) -> int:
        """How many distinct words were expanded (a proxy for real work done)."""
        return self._expansions

    def resolve(self, class_id: int) -> int:
        """Map a possibly-stale class id to its current representative."""
        return self._uf.find(class_id)

    def known(self, word: Word) -> bool:
        return word in self._class_of

    def lookup(self, word: Word):
        """Return the current class of a already-seen word, else None."""
        cid = self._class_of.get(word)
        return None if cid is None else self._uf.find(cid)

    # -- the main routine --------------------------------------------------

    def classify(self, word: Word) -> int:
        """Return the equivalence class of ``word``, exploring it if new."""
        class_of = self._class_of
        uf = self._uf

        cid = class_of.get(word)
        if cid is not None:
            return uf.find(cid)

        rewriter = self.rewriter
        is_zero_fn = rewriter.is_zero
        neighbours = rewriter.neighbours

        fresh: list[Word] = []          # words not seen before this search
        stack: list[Word] = [word]
        seen: set[Word] = {word}
        touched: list[int] = []         # pre-existing classes met on the way
        zero = False

        while stack:
            current = stack.pop()
            fresh.append(current)
            if is_zero_fn(current):
                zero = True
            for nxt in neighbours(current):
                if nxt in seen:
                    continue
                seen.add(nxt)
                existing = class_of.get(nxt)
                if existing is None:
                    stack.append(nxt)
                else:
                    # Already classified: its own class is closed under the
                    # moves, so stop here and merge instead of re-walking it.
                    touched.append(existing)

        self._expansions += len(fresh)

        root = uf.new()
        members = self._members
        members[root] = fresh
        for w in fresh:
            class_of[w] = root

        for other in touched:
            root = self._merge(root, other)
        if zero:
            root = self._merge(root, ZERO_CLASS)
        return root

    def _merge(self, a: int, b: int) -> int:
        uf = self._uf
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            return ra
        root = uf.union(ra, rb)
        loser = rb if root == ra else ra
        members = self._members
        moved = members.pop(loser, None)
        if moved:
            members[root].extend(moved)
        return root

    def mark_zero(self, word: Word) -> int:
        """Force ``word`` (and its whole class) into the zero class."""
        cid = self.classify(word)
        return self._merge(cid, ZERO_CLASS)

    # -- results -----------------------------------------------------------

    def classes(self) -> dict:
        """Return ``{root: [words...]}`` for the current, fully-merged state."""
        uf = self._uf
        out: dict[int, list[Word]] = {}
        for root, words in self._members.items():
            true_root = uf.find(root)
            if true_root == root:
                out[root] = words
            else:  # pragma: no cover - _merge keeps this from happening
                out.setdefault(true_root, []).extend(words)
        return out

    def zero_words(self) -> list:
        return self._members.get(self._uf.find(ZERO_CLASS), [])

    def all_words(self) -> Iterable[Word]:
        return self._class_of.keys()
