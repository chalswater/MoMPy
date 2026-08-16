#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:53:34 2026

@author: carles
"""

import cvxpy as cp
from MoMPy import MomentProblem, OperatorSet

nX = 3
ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)   # pure states
M = ops.add_povm(nX)                      # one projective measurement
# ops.declare_commuting(R, R) # for classical implementation (comment for quantum bound)

monomials  = list(R) + list(M)
monomials += [[r, m] for r in R for m in M]
monomials += [[r, s] for r in R for s in R]
monomials += [[r, s, t] for r in R for s in R for t in R]   # third order

mm = MomentProblem(monomials, ops.algebra(), dim=1).build()

def bound(min_overlap):
    model = mm.to_cvxpy()
    ct = list(model.constraints)
    ct += [model[[R[x]]] == 1.0 for x in range(nX)]     # Tr(rho) = 1
    ct += model.apply(mm.normalisation_constraints(M))  # sum_b M_b = 1
    if min_overlap is not None:
        ct += [model[[R[x], R[xx]]] >= min_overlap
               for x in range(nX) for xx in range(nX) if x != xx]
    P = sum(model[[R[x], M[x]]] for x in range(nX)) / nX
    cp.Problem(cp.Maximize(P), ct).solve(solver=cp.SCS)
    return P.value
