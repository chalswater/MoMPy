#!/usr/bin/env python3
"""Prepare-and-measure: minimum-error state discrimination.

Alice prepares one of nX pure states and sends it to Bob, who tries to identify
which. With the Hilbert-space dimension unconstrained the states can be made
mutually orthogonal, so the bound is 1. Constraining the pairwise overlaps
Tr(R[x] R[x']) >= c makes the problem non-trivial and the bound drops.

This example also shows the third-order monomials that made the original
implementation slow: with nX = 3 they add 27 words of length three.

Run:  python pam_state_discrimination.py
"""

from __future__ import annotations

import time

import cvxpy as cp

from MoMPy import MomentProblem, OperatorSet

nX = 4          # state preparations, one per message
nB = 4          # measurement outcomes, one guess per message
THIRD_ORDER = True

def build():
    ops = OperatorSet()
    R = ops.add_family(nX, idempotent=True)     # pure states
    M = ops.add_povm(nB)                        # one projective measurement

    # Pure states R[x] = |psi_x><psi_x| all commute in the relaxation.
    ops.declare_commuting(R, R)

    monomials = list(R) + list(M)
    monomials += [[r, m] for r in R for m in M]
    monomials += [[r, s] for r in R for s in R]
    if THIRD_ORDER:
        monomials += [[r, s, t] for r in R for s in R for t in R]

    start = time.perf_counter()
    mm = MomentProblem(monomials, ops.algebra(), dim=1).build(progress=True)
    print(f"moment matrix built in {time.perf_counter() - start:.3f} s")
    print(mm.summary())
    return mm, R, M


def solve(mm, R, M, min_overlap=None):
    model = mm.to_cvxpy()
    ct = list(model.constraints)

    # Tr(1) is the dimension here, so it is left free.
    # States are normalised.
    ct += [model[[R[x]]] == 1.0 for x in range(nX)]

    # sum_b M_b == identity, propagated through every monomial.
    ct += model.apply(mm.normalisation_constraints(M))

    if min_overlap is not None:
        ct += [
            model[[R[x], R[xx]]] >= min_overlap
            for x in range(nX)
            for xx in range(nX)
            if x != xx
        ]

    success = sum(model[[R[x], M[x]]] for x in range(nX)) / nX
    problem = cp.Problem(cp.Maximize(success), ct)
    problem.solve(solver='MOSEK')
    return problem.value


if __name__ == "__main__":
    mm, R, M = build()

    print(f"\n{'min overlap':>12}  {'success probability':>20}")
    print("-" * 36)
    print(f"{'unbounded':>12}  {solve(mm, R, M):>20.6f}")
    for c in (0.1, 0.3, 0.5, 0.7):
        print(f"{c:>12.2f}  {solve(mm, R, M, min_overlap=c):>20.6f}")
