#!/usr/bin/env python3
"""NPA hierarchy: Tsirelson's bound for CHSH.

Two parties, two binary measurements each. The NPA level-1 relaxation of the
CHSH value converges to 2*sqrt(2) ~ 2.8284, which is the exact quantum maximum.

Run:  python npa_chsh.py
"""

from __future__ import annotations

import math
import time

import cvxpy as cp

from MoMPy import MomentProblem, OperatorSet

nA = nB = 2          # outcomes
nX = nY = 2          # settings
LEVEL_1_PLUS_AB = True    # products AB; safe here because moments are state moments


def build():
    ops = OperatorSet()
    A = ops.add_povm_family(nX, nA)     # A[x][a]
    B = ops.add_povm_family(nY, nB)     # B[y][b]

    flat_a = [m for row in A for m in row]
    flat_b = [m for row in B for m in row]

    # Alice's and Bob's operators act on different subsystems, so their
    # operators commute with each other. Alice's own operators do NOT commute
    # with each other across different measurement settings x (that is
    # exactly the assumption whose absence lets CHSH exceed the classical
    # bound), so only the cross-party commutation is declared.
    ops.declare_commuting(flat_a, flat_b)

    monomials = flat_a + flat_b
    if LEVEL_1_PLUS_AB:
        monomials += [[a, b] for a in flat_a for b in flat_b]

    start = time.perf_counter()
    # State moments (cyclicity=False): the moments are <psi| u v^dagger |psi>
    # in a fixed but unspecified state, not a trace, so cyclic rotations of a
    # word are not equivalent -- this is the setting NPA / Bell problems need.
    mm = MomentProblem(monomials, ops.algebra(), dim=1, cyclicity=False).build()
    print(f"moment matrix built in {time.perf_counter() - start:.3f} s")
    print(mm.summary())
    return mm, A, B


def solve(mm, A, B):
    model = mm.to_cvxpy()
    ct = list(model.constraints)

    # NPA convention: the moments are <psi| w |psi>, so the identity is 1.
    ct.append(model.identity == 1)

    for x in range(nX):
        ct.append(sum(model[[A[x][a]]] for a in range(nA)) == model.identity)
    for y in range(nY):
        ct.append(sum(model[[B[y][b]]] for b in range(nB)) == model.identity)

    def correlator(x, y):
        return sum(
            (-1) ** (a + b) * model[[A[x][a], B[y][b]]]
            for a in range(nA)
            for b in range(nB)
        )

    chsh = (correlator(0, 0) + correlator(1, 0)
            + correlator(0, 1) - correlator(1, 1))

    problem = cp.Problem(cp.Maximize(chsh), ct)
    problem.solve(solver='MOSEK')
    return problem.value


if __name__ == "__main__":
    mm, A, B = build()
    value = solve(mm, A, B)
    print(f"\nCHSH bound      : {value:.6f}")
    print(f"Tsirelson 2*sqrt(2): {2 * math.sqrt(2):.6f}")
    print(f"Local bound        : 2.000000")
