"""How does the new engine scale as the hierarchy grows?"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/home/claude/work/pkg")

from MoMPy import MomentProblem, OperatorSet


def npa(n_parties_settings, n_outcomes, level_extra=True):
    """NPA-style: two parties, ``n`` settings each, level 1 (+AB)."""
    ops = OperatorSet()
    A = ops.add_povm_family(n_parties_settings, n_outcomes)
    B = ops.add_povm_family(n_parties_settings, n_outcomes)
    fa = [x for row in A for x in row]
    fb = [x for row in B for x in row]
    ops.declare_commuting(fa, fb)
    mons = fa + fb
    if level_extra:
        mons += [[a, b] for a in fa for b in fb]
    return mons, ops.algebra()


def pam(n_states, n_settings, n_outcomes, order=3):
    """Prepare-and-measure with commuting pure states, third-order words."""
    ops = OperatorSet()
    R = ops.add_family(n_states, idempotent=True)
    M = ops.add_povm_family(n_settings, n_outcomes)
    ops.declare_commuting(R, R)
    fm = [x for row in M for x in row]
    mons = list(R) + fm
    mons += [[r, m] for r in R for m in fm]
    mons += [[r, s] for r in R for s in R]
    if order >= 3:
        mons += [[r, s, t] for r in R for s in R for t in R]
    return mons, ops.algebra()


CASES = [
    ("NPA 2 settings 2 out, 1+AB", *npa(2, 2)),
    ("NPA 3 settings 2 out, 1+AB", *npa(3, 2)),
    ("NPA 4 settings 2 out, 1+AB", *npa(4, 2)),
    ("NPA 3 settings 3 out, 1+AB", *npa(3, 3)),
    ("NPA 4 settings 3 out, 1+AB", *npa(4, 3)),
    ("NPA 5 settings 3 out, 1+AB", *npa(5, 3)),
    ("PAM 3 states, order 3",      *pam(3, 2, 2)),
    ("PAM 4 states, order 3",      *pam(4, 2, 2)),
    ("PAM 5 states, order 3",      *pam(5, 2, 2)),
    ("PAM 6 states, order 3",      *pam(6, 2, 2)),
    ("PAM 8 states, order 3",      *pam(8, 3, 2)),
]

print(f"{'scenario':<30} {'matrix':>9} {'entries':>10} {'vars':>7} "
      f"{'words':>9} {'seconds':>9} {'us/entry':>9}")
print("-" * 92)
for label, mons, alg in CASES:
    t0 = time.perf_counter()
    mm = MomentProblem(mons, alg).build()
    dt = time.perf_counter() - t0
    entries = mm.n * mm.n
    print(f"{label:<30} {mm.n:>4}x{mm.n:<4} {entries:>10} {mm.n_variables:>7} "
          f"{mm.stats['distinct_words']:>9} {dt:>8.3f}s "
          f"{dt / entries * 1e6:>8.2f}")
