#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:57:18 2026

@author: carles
"""

import cvxpy as cp
from MoMPy import MomentProblem, OperatorSet

nX, nK = 3 , 1
ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)   # preparations
M = ops.add_povm(nX)                      # guessing measurement
P = ops.add_family(nK,idempotent=True)    # the heralding projector
ops.declare_commuting(R,R)

monomials  = list(R) + list(M) + list(P)
monomials += [[r, m] for r in R for m in M]
monomials += [[r, p] for r in R for p in P]
monomials += [[r, s] for r in R for s in R]

mm = MomentProblem(monomials, ops.algebra()).build()

def bound(omega):
    model = mm.to_cvxpy()
    ct = list(model.constraints)
    ct += [model[[R[x]]] == 1.0 for x in range(nX)]
    ct += [model[[P[k]]] == 1.0 for k in range(nK)]
    ct += [model[[R[x], P[k]]] == 1.0 - omega for x in range(nX) for k in range(nK)]
    ct += model.apply(mm.normalisation_constraints(M))
    P_succ = sum(model[[R[x], M[x]]] for x in range(nX)) / nX
    cp.Problem(cp.Maximize(P_succ), ct).solve(solver=cp.MOSEK)
    return P_succ.value
