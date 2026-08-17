import cvxpy as cp
import numpy as np
from MoMPy import OperatorSet, MomentProblem
 
nX, nY, nB, d = 3, 2, 2, 2
 
ops = OperatorSet()
R = ops.add_family(nX, idempotent=True)      # three known preparations
M = ops.add_povm_family(nY, nB)              # M[y][b]: two known qubit bases
algebra = ops.algebra()
 
monomials  = list(R) + [m for row in M for m in row]
monomials += [[r, m] for r in R for row in M for m in row]
monomials += [[r, s] for r in R for s in R]
 
mm = MomentProblem(monomials, algebra, dim=1).build()                      # scalar, tracial
bm = MomentProblem(monomials, algebra, dim=d,
                    cyclicity=False, hermitian=False).build()              # block-valued, same relations


def critical_visibility(rho, meas, xstar, ystar):
    """Largest v for which Eve can guess Bob's outcome at (xstar, ystar) for
    certain, given the fixed noisy ensemble {rho[x]} and known measurement."""
    branches_B = [bm.to_cvxpy() for _ in range(nB)]  # one branch per guess l; dim=d from bm
    branches_G = [mm.to_cvxpy() for _ in range(nB)]
 
    v = cp.Variable(nonneg=True)
    q = cp.Variable(nB, nonneg=True)                       # Pr(Eve's branch = l)
 
    branches = branches_B + branches_G
    ct = [c for br in branches for c in br.constraints]
    ct += [v <= 1, cp.sum(q) == 1]
 
    for l in range(nB):                                    # tie block traces to scalars
        for r in range(mm.n):
            for c in range(mm.n):
                ct.append(cp.trace(branches_B[l][bm.matrix[r, c]]) == branches_G[l][mm.matrix[r, c]])
 
    for l in range(nB):
        ct.append(branches_B[l][bm.identity_index] == q[l] * np.eye(d))
        for x in range(nX):
            ct.append(cp.real(cp.trace(branches_B[l][bm.index_of([R[x]])])) == q[l])
 
    for x in range(nX):                                    # branches average to the noisy ensemble
        ct.append(sum(branches_B[l][bm.index_of([R[x]])] for l in range(nB))
                  == v * rho[x] + (1 - v) * np.eye(d) / d)
 
    for x in range(nX):                                    # ... and reproduce every observed statistic
        for y in range(nY):
            for b in range(nB):
                observed = cp.real(sum(cp.trace(branches_B[l][bm.index_of([R[x], M[y][b]])])
                                        for l in range(nB)))
                target = cp.real(cp.trace((v * rho[x] + (1 - v) * np.eye(d) / d) @ meas[y][b]))
                ct.append(observed == target)
 
    p_guess = sum(cp.real(cp.trace(branches_B[l][bm.index_of([R[xstar], M[ystar][l]])]))
                  for l in range(nB))
    ct.append(p_guess == 1)                                # Eve succeeds with certainty
 
    cp.Problem(cp.Maximize(v), ct).solve(solver=cp.SCS)
    return v.value
