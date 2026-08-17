import cvxpy as cp
from MoMPy import OperatorSet, MomentProblem
 
ops = OperatorSet()
A = ops.add_povm_family(2, 2)          # A[x][a]
B = ops.add_povm_family(2, 2)          # B[y][b]
C = ops.add_povm_family(2, 2)          # C[z][c]
flat_a = [m for row in A for m in row]
flat_b = [m for row in B for m in row]
flat_c = [m for row in C for m in row]
ops.declare_commuting(flat_a, flat_b)  # distinct parties
ops.declare_commuting(flat_a, flat_c)
ops.declare_commuting(flat_b, flat_c)

# ops.declare_commuting(flat_a, flat_a) # classical Alice
# ops.declare_commuting(flat_b, flat_b) # classical Bob
 
monomials  = flat_a + flat_b + flat_c
monomials += [[a, b] for a in flat_a for b in flat_b]   # level 1 + AB + AC + BC
monomials += [[a, c] for a in flat_a for c in flat_c]
monomials += [[b, c] for b in flat_b for c in flat_c]
monomials += [[a, b, c] for a in flat_a for b in flat_b for c in flat_c]  # + ABC
 
mm = MomentProblem(monomials, ops.algebra(), dim=1, cyclicity=False).build()
 
model = mm.to_cvxpy()
ct = list(model.constraints)
ct.append(model.identity == 1)
for x in range(2):
    ct.append(sum(model[[A[x][a]]] for a in range(2)) == model.identity)
for y in range(2):
    ct.append(sum(model[[B[y][b]]] for b in range(2)) == model.identity)
for z in range(2):
    ct.append(sum(model[[C[z][c]]] for c in range(2)) == model.identity)
 
def corr3(x, y, z):
    return sum((-1)**(a + b + c) * model[[A[x][a], B[y][b], C[z][c]]]
               for a in range(2) for b in range(2) for c in range(2))
 
M = corr3(0,0,1) + corr3(0,1,0) + corr3(1,0,0) - corr3(1,1,1)
cp.Problem(cp.Maximize(M), ct).solve(solver=cp.MOSEK)
print(M.value)          # 4.000000
