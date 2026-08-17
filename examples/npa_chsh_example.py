import cvxpy as cp
from MoMPy import OperatorSet, MomentProblem
 
ops = OperatorSet()
A = ops.add_povm_family(2, 2)          # A[x][a]
B = ops.add_povm_family(2, 2)          # B[y][b]
flat_a = [m for row in A for m in row]
flat_b = [m for row in B for m in row]
ops.declare_commuting(flat_a, flat_b)  # distinct Hilbert space
# ops.declare_commuting(flat_a, flat_a)  # uncomment for local bound
 
monomials  = flat_a + flat_b
monomials += [[a, b] for a in flat_a for b in flat_b]   # level 1 + AB
 
mm = MomentProblem(monomials, ops.algebra(), dim=1, cyclicity=False).build()
 
model = mm.to_cvxpy()
ct = list(model.constraints)
ct.append(model.identity == 1)                          # <psi|1|psi> = 1
for x in range(2):
    ct.append(sum(model[[A[x][a]]] for a in range(2)) == model.identity)
for y in range(2):
    ct.append(sum(model[[B[y][b]]] for b in range(2)) == model.identity)
 
def corr(x, y):
    return sum((-1)**(a + b) * model[[A[x][a], B[y][b]]]
               for a in range(2) for b in range(2))
 
S = corr(0, 0) + corr(1, 0) + corr(0, 1) - corr(1, 1)
cp.Problem(cp.Maximize(S), ct).solve(solver=cp.MOSEK)
print(S.value)          # 2.828427  =  2 * sqrt(2)
