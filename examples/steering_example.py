import cvxpy as cp
import numpy as np
from itertools import product
from MoMPy import OperatorSet, MomentProblem
 
sx = np.array([[0, 1], [1, 0]], complex)
sy = np.array([[0, -1j], [1j, 0]], complex)
sz = np.array([[1, 0], [0, -1]], complex)
paulis, d = [sx, sy, sz], 2
LAM = list(product([0, 1], repeat=3))       # lambda = (a_x, a_y, a_z)
 
def steering_bound(jointly_measurable):
    ops = OperatorSet()
    if jointly_measurable:
        # no sharpness assumed: the only requirement is Eq. (17)
        A = ops.add_povm_family(3, 2, idempotent=False, orthogonal=False)
        G = ops.add_povm(8)                  # global POVM, projective by Naimark
    else:
        A = ops.add_povm_family(3, 2)        # sharp: extremal for this criterion
        G = []
    monomials = [m for row in A for m in row] + list(G)
 
    bm = MomentProblem(monomials, ops.algebra(), dim=d,
                       cyclicity=False, hermitian=False).build()   # Theta = Tr_A
 
    model = bm.to_cvxpy()
    ct = list(model.constraints)
    rho_B = model[bm.identity_index]                    # Theta(1) = rho_B
    ct.append(cp.real(cp.trace(rho_B)) == 1)            # Tr rho_B = 1
    for k in range(3):                                  # sum_a sigma_{a|k} = rho_B
        ct.append(model[[A[k][0]]] + model[[A[k][1]]] == rho_B)
 
    if jointly_measurable:
        ct += model.apply(bm.normalisation_constraints(list(G)))
        for k in range(3):                              # Eq. (17), via marginals
            for a in (0, 1):
                joint = [G[i] for i, lam in enumerate(LAM) if lam[k] == a]
                ct += model.apply(bm.marginal_constraints(joint, A[k][a]))
 
    Q = sum(cp.real(cp.trace(paulis[k] @ (model[[A[k][0]]] - model[[A[k][1]]])))
            for k in range(3))
    cp.Problem(cp.Maximize(Q), ct).solve(solver=cp.SCS, eps=1e-9)
    return Q.value
