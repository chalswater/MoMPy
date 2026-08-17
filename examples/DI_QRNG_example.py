import cvxpy as cp
from MoMPy import OperatorSet, MomentProblem
 
ops = OperatorSet()
A = ops.add_povm_family(2, 2)          # A[x][a]
B = ops.add_povm_family(2, 2)          # B[y][b]
flat_a = [m for row in A for m in row]
flat_b = [m for row in B for m in row]
ops.declare_commuting(flat_a, flat_b)
 
monomials  = flat_a + flat_b
monomials += [[a, b] for a in flat_a for b in flat_b]
mm = MomentProblem(monomials, ops.algebra(), dim=1, cyclicity=False).build()
 
def guessing_probability(S_obs):
    branches = [mm.to_cvxpy() for _ in range(2)]         # one branch per guess l in {0, 1}
    ct = []
    for br in branches:
        ct += br.constraints
    ct.append(sum(br.identity for br in branches) == 1.0)
    for x in range(2):
        for br in branches:
            ct += br.apply(mm.normalisation_constraints([A[x][0], A[x][1]]))
    for y in range(2):
        for br in branches:
            ct += br.apply(mm.normalisation_constraints([B[y][0], B[y][1]]))
 
    def corr(x, y):
        return sum((-1)**(a + b) * sum(br[[A[x][a], B[y][b]]] for br in branches)
                   for a in range(2) for b in range(2))
 
    S = corr(0, 0) + corr(1, 0) + corr(0, 1) - corr(1, 1)
    ct.append(S == S_obs)                                # the observed CHSH value
 
    p_guess = sum(branches[l][[A[0][l]]] for l in range(2))   # Eve guesses l, scores if a = l
    cp.Problem(cp.Maximize(p_guess), ct).solve(solver=cp.MOSEK)
    return p_guess.value
