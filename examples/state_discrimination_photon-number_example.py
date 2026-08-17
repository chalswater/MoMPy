import cvxpy as cp
from MoMPy import MomentProblem, OperatorSet
 
nX = 3
nK = 1
ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)   # preparations
M = ops.add_povm(nX)                      # guessing measurement
P = ops.add_family(nK, idempotent=True)   # the photon-number projector
ops.declare_commuting(R,R) # for classical bound (comment for quantum)
 
monomials  = list(R) + list(M) + list(P)
monomials += [[r, m] for r in R for m in M]
monomials += [[r, p] for r in R for p in P]
monomials += [[r, s] for r in R for s in R]
 
mm = MomentProblem(monomials, ops.algebra(), dim=1).build()
 
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
