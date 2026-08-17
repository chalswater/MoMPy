import cvxpy as cp
from MoMPy import MomentProblem, OperatorSet
 
nX, nY, nB = 3, 2, 2
ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)     # preparations
M = ops.add_povm_family(nY, nB)             # M[y][b], two dichotomic measurements
ops.declare_commuting(R, R)
 
monomials  = list(R) + [m for row in M for m in row]
monomials += [[r, m] for r in R for row in M for m in row]
monomials += [[r, s] for r in R for s in R]
monomials += [[r, s, t] for r in R for s in R for t in R]
 
mm = MomentProblem(monomials, ops.algebra(), dim=1).build()
 
def W_max(dim):
    model = mm.to_cvxpy()
    ct = list(model.constraints)
    ct.append(model.identity == dim)                    # pin Tr(1) = D
    ct += [model[[R[x]]] == 1.0 for x in range(nX)]
    for y in range(nY):
        ct += model.apply(mm.normalisation_constraints(M[y]))
 
    def E(x, y):
        return sum((-1)**b * model[[R[x], M[y][b]]] for b in range(nB))
 
    W = E(0,0) + E(0,1) + E(1,0) - E(1,1) - E(2,0)
    cp.Problem(cp.Maximize(W), ct).solve(solver=cp.MOSEK)
    return W.value
