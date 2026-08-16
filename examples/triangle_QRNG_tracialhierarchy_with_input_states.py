#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:35:27 2025

@author: carles
"""

import numpy as np
import cvxpy as cp
from cvxpy import *
import time
import chaospy # Needed for Gauss-Radau quadrature weigths and nodes
import scipy.special as sps # used to compute error function
# to integrate
from scipy.integrate import quad
import scipy.integrate as integrate
from scipy.integrate import dblquad

# Moment matrix generators
from MoMPy.MoM import *

import json

import itertools

import warnings
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Functions                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def embed_two_qubit_operator(M, q1, q2, nqubits=6):
    """
    Embed a 4x4 operator M acting on qubits (q1,q2)
    into an nqubits-qubit Hilbert space.

    Qubit numbering starts from 0.
    """
    dim = 2**nqubits

    # reshape operator into tensor with indices
    # (out1,out2,in1,in2)
    Mt = M.reshape(2,2,2,2)

    O = np.zeros((dim, dim), dtype=complex)

    for i in range(dim):
        in_bits = [(i >> (nqubits-1-k)) & 1 for k in range(nqubits)]

        for j in range(dim):
            out_bits = [(j >> (nqubits-1-k)) & 1 for k in range(nqubits)]

            # spectator qubits must be unchanged
            ok = True
            for k in range(nqubits):
                if k not in (q1,q2):
                    if in_bits[k] != out_bits[k]:
                        ok = False
                        break
            if not ok:
                continue

            O[j,i] = Mt[
                out_bits[q1],
                out_bits[q2],
                in_bits[q1],
                in_bits[q2]
            ]

    return O

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def embed_A(M):
    # Alice acts on qubits (A_AB,A_CA) = (0,5)
    return embed_two_qubit_operator(M, 0, 5)

def embed_B(M):
    # Bob acts on qubits (B_AB,B_BC) = (1,2)
    return embed_two_qubit_operator(M, 1, 2)

def embed_C(M):
    # Charlie acts on qubits (C_BC,C_CA) = (3,4)
    return embed_two_qubit_operator(M, 3, 4)

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def Pabc_obs_generator(u):

    phi = np.array([1,0,0,1], dtype=complex) / np.sqrt(2)
    Psi = np.kron(np.kron(phi, phi), phi)
    rho = np.outer(Psi, Psi.conj())
    
    v00 = np.array([1.0,0.0,0.0,0.0])
    v01 = np.array([0.0,1.0,0.0,0.0])
    v10 = np.array([0.0,0.0,1.0,0.0])
    v11 = np.array([0.0,0.0,0.0,1.0])
        
    Xi0 = u * v00 + np.sqrt(1.0-u**2) * v11
    Xi1 = np.sqrt(1.0-u**2) * v00 - u * v11
    
    M = {}
    M[0] = np.outer(v01,v01)
    M[1] = np.outer(v10,v10)
    M[2] = np.outer(Xi0,Xi0)
    M[3] = np.outer(Xi1,Xi1)
    
    P = np.zeros((4,4,4))
    
    for a in range(4):
        OA = embed_A(M[a])
    
        for b in range(4):
            OB = embed_B(M[b])
    
            for c in range(4):
                OC = embed_C(M[c])
    
                O = OA @ OB @ OC
                P[a,b,c] = np.real(np.trace(rho @ O))

    return P

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def deterministic_strategies(B, nX, nY):
    """
    Generate all deterministic strategies D(b|x,y).

    Returns
    -------
    strategies : list of np.ndarray
        Each element has shape (B, nX, nY)
    """
    strategies = []

    # Each strategy is a function f : (x,y) -> b
    # Represent it as a tuple of length nX*nY with values in {0,...,B-1}
    for outputs in itertools.product(range(B), repeat=nX * nY):
        D = np.zeros((B, nX, nY))

        idx = 0
        for x in range(nX):
            for y in range(nY):
                b = outputs[idx]
                D[b, x, y] = 1.0
                idx += 1

        strategies.append(D)

    return strategies

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def deltaF(x,xx):

    """ Delta function """
    
    if x == xx:
        return 1.0
    else:
        return 0.0
   
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def triangle_QRNG(nA,nB,nC,nZ,nK,omega,Pabc_obs):
    
    [w_R,w_A,w_B,w_C,w_P] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els
    
    nL = nA  
    
    #----------------------------------------------------#
    #                  CREATE VARIABLES                  #
    #----------------------------------------------------#

    G_var_vec = {}
    for l in range(nL):
        G_var_vec[l] = {}
        for element in list_of_eq_indices:
            if element == map_table[-1][-1]:
                G_var_vec[l][element] = 0.0 # Zeros form orthogonal projectors
            else:
                G_var_vec[l][element] = cp.Variable()

    q = cp.Variable(nL,nonneg=True)

    #--------------------------------------------------#
    #                  BUILD MATRICES                  #
    #--------------------------------------------------#
    
    G = {}
    for l in range(nL): 
        lis = []
        for r in range(len(G_new)):
            lis += [[]]
            for c in range(len(G_new)):
                lis[r] += [G_var_vec[l][G_new[r][c]]]
        G[l] = cp.bmat(lis)

    #------------------------------------------------------#
    #                  CREATE CONSTRAINTS                  #
    #------------------------------------------------------#
    
    ct = []
    
    ct += [ sum([ q[l] for l in range(nL) ]) == 1.0 ]
   
    ct += [ G[l] >> 0.0 for l in range(nL) ]
    
    # ct += [ G_var_vec[fmap(map_table,[0])] == 2.0 ]
    
    # ----------------------------------------------------------------------------------------

    # Rank-1 projectors
    ct += [ G_var_vec[l][fmap(map_table,[w_R[z]])] == q[l] for z in range(nZ) for l in range(nL)]
    ct += [ G_var_vec[l][fmap(map_table,[w_P[k]])] == q[l] for k in range(nK) for l in range(nL)]
    
    ct += [ sum([ G_var_vec[l][fmap(map_table,[w_R[z],w_P[k]])] for l in range(nL) ]) >= 1.0 - omega[k] for z in range(nZ) for k in range(nK) ]
        
    # Normalisation
    ct += [ sum([ G_var_vec[l][fmap(map_table,[w_A[a]])] for a in range(nA) ]) == G_var_vec[l][fmap(map_table,[0])] for l in range(nL) ]
    ct += [ sum([ G_var_vec[l][fmap(map_table,[w_B[b]])] for b in range(nB) ]) == G_var_vec[l][fmap(map_table,[0])] for l in range(nL) ]
    ct += [ sum([ G_var_vec[l][fmap(map_table,[w_C[c]])] for c in range(nC) ]) == G_var_vec[l][fmap(map_table,[0])] for l in range(nL) ]
    
    # Probabilities
    pabc = {}
    for a in range(nA):
        pabc[a] = {}
        for b in range(nB):
            pabc[a][b] = {}
            for c in range(nC):
                pabc[a][b][c] = sum([ G_var_vec[l][fmap(map_table,[w_R[z],w_A[a],w_B[b],w_C[c]])] for l in range(nL) ])
    
    ct += [ pabc[a][b][c] == Pabc_obs[a][b][c] for a in range(nA) for b in range(nB) for c in range(nC) ]
    
    # Check LHV models with deterministic assignements
    # DA = deterministic_strategies(nA, nX, nZ)
    # DB = deterministic_strategies(nB, nY, nZ)
        
    # nL = len(DA)
    # q = cp.Variable((nL,nL),nonneg=True)
    # ct += [ sum([ q[la][lb] for la in range(nL) for lb in range(nL) ]) == 1.0 ]
    # ct += [ pabxyz[a][b][x][y][z] == sum([ q[la][lb] * DA[la][a][x][z] * DB[lb][b][y][z] for la in range(nL) for lb in range(nL) ]) for a in range(nA) for b in range(nB) for x in range(nX) for y in range(nY) for z in range(nZ) ]
    
    zstar = 0
    pg = sum([ G_var_vec[a][fmap(map_table,[w_R[zstar],w_A[a]])] for a in range(nA) ])
    
    # pg = sum([ (G_var_vec[l0 * nA**0 + l1 * nA**1][fmap(map_table,[w_R[zstar],w_A[l0][0]])] + \
    #             G_var_vec[l0 * nA**0 + l1 * nA**1][fmap(map_table,[w_R[zstar],w_A[l1][1]])] ) / (2) for l0 in range(nB) for l1 in range(nB) ])
        
    # pg = sum([ (G_var_vec[l0 * nA**0 + l1 * nA**1][fmap(map_table,[w_R[0],w_A[l0][xstar]])] + \
    #             G_var_vec[l0 * nA**0 + l1 * nA**1][fmap(map_table,[w_R[1],w_A[l1][xstar]])] ) / (2) for l0 in range(nB) for l1 in range(nB) ])
        
    # pg = sum([ (G_var_vec[l00 * nA**0 + l01 * nA**1 + l10 * nA**2 + l11 * nA**3][fmap(map_table,[w_R[0],w_A[l00][0]])] + \
    #             G_var_vec[l00 * nA**0 + l01 * nA**1 + l10 * nA**2 + l11 * nA**3][fmap(map_table,[w_R[1],w_A[l10][0]])] + \
    #             G_var_vec[l00 * nA**0 + l01 * nA**1 + l10 * nA**2 + l11 * nA**3][fmap(map_table,[w_R[0],w_A[l01][1]])] + \
    #             G_var_vec[l00 * nA**0 + l01 * nA**1 + l10 * nA**2 + l11 * nA**3][fmap(map_table,[w_R[1],w_A[l11][1]])] ) / (4) for l00 in range(nB) for l01 in range(nB) for l10 in range(nB) for l11 in range(nB) ])
    
    # ct += [ pg == 1.0 ]
    
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#

    obj = cp.Maximize(pg)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=True, mosek_params=mosek_params)

    except SolverError:
        something = 10

    return pg.value

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        MAIN CODE                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

nA = 4
nB = 4
nC = 4
nL = 4
nZ = 1 # number of state preparations
nK = 1 # number of restricted components

#---------------------------------------------------------------------#
#                        Collect all monomials                        #
#---------------------------------------------------------------------#

# Track operators in the tracial matrix
w_R = [] # Prepared quantum state
w_A = [] # Measurement in Alice
w_B = [] # Measurement in Bob
w_C = [] # Measurement in Charlie
w_P = [] # Projector onto the restricted components

S_1 = [] # List of first order elements
cc = 1

for z in range(nZ):
    S_1 += [cc]
    w_R += [cc]
    cc += 1

for a in range(nA): 
    S_1 += [cc]
    w_A += [cc]
    cc += 1

for b in range(nB): 
    S_1 += [cc]
    w_B += [cc]
    cc += 1
        
for c in range(nC): 
    S_1 += [cc]
    w_C += [cc]
    cc += 1
    
for k in range(nK):
    S_1 += [cc]
    w_P += [cc]
    cc += 1

# Additional higher order elements
S_high = [] # Uncomment if we only allow up to some 2nd order elements in the hierarchy  

# Second order elements
some_second = False
if some_second == True:

    S_high += [[w_R[z],w_R[zz]]   for z in range(nZ) for zz in range(nZ) ]
    S_high += [[w_R[z],w_A[a]] for a in range(nA) for z in range(nZ) ]
    S_high += [[w_R[z],w_B[b]] for b in range(nB) for z in range(nZ) ]

some_third = True
if some_third == True:
    
    S_high += [[w_R[z],w_R[zz],w_R[zzz]] for z in range(nZ) for zz in range(nZ) for zzz in range(nZ) ]
    # S_high += [[w_P[k],w_B[b][y],w_R[z]] for k in range(nK) for b in range(nB) for y in range(nY) for z in range(nZ) ]
    S_high += [[w_A[a],w_B[b],w_R[z]] for a in range(nA) for b in range(nB) for z in range(nZ) ]
    S_high += [[w_A[a],w_B[b],w_C[c]] for a in range(nA) for b in range(nB) for c in range(nC) ]

# Set the operational rules within the SDP relaxation
rank_1_projectors = []
rank_1_projectors += [ w_R[z] for z in range(nZ) ]
rank_1_projectors += [ w_A[a] for a in range(nA) ]
rank_1_projectors += [ w_B[b] for b in range(nB) ] 
rank_1_projectors += [ w_C[c] for c in range(nC) ]
rank_1_projectors += [ w_P[k] for k in range(nK) ]

orthogonal_projectors = []
orthogonal_projectors += [ [ w_A[a] for a in range(nA) ] ]
orthogonal_projectors += [ [ w_B[b] for b in range(nB) ] ]
orthogonal_projectors += [ [ w_C[c] for c in range(nC) ] ]
orthogonal_projectors += [ [ w_P[k] for k in range(nK) ] ]

commuting_pairs = [] # commuting elements (wxcept with elements in "list_states"
commuting_pairs += [ [ [w_A[a] for a in range(nA)] , [w_B[b] for b in range(nB)] ] ]
commuting_pairs += [ [ [w_A[a] for a in range(nA)] , [w_C[c] for c in range(nC)] ] ]
commuting_pairs += [ [ [w_B[b] for b in range(nB)] , [w_C[c] for c in range(nC)] ] ]

# commuting_pairs += [ [ [w_A[a][x] for x in range(nX) for a in range(nA)] , [w_A[a][x] for x in range(nX) for a in range(nA)] ] ]
# commuting_pairs += [ [ [ w_R[z] for z in range(nZ) ] , [ w_R[z] for z in range(nZ) ] ] ]
# commuting_pairs += [ [ [ w_R[z] for z in range(nZ) ] , [w_A[a][x] for x in range(nX) for a in range(nA)] ] ]
# commuting_pairs += [ [ [ w_R[z] for z in range(nZ) ] , [w_B[b][y] for y in range(nY) for b in range(nB)] ] ]

print('Rank-1 projectors',rank_1_projectors)
print('Orthogonal projectors',orthogonal_projectors)
print('commuting elements',commuting_pairs)

# Collect rules and generate SDP relaxation matrix
start = time.process_time()
# [G_new,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S_1,S_1,S_high,rank_1_projectors,orthogonal_projectors,commuting_pairs)
end = time.process_time()

print('Gamma matrix generated in',end-start,'s')
print('Matrix size:',np.shape(G_new))

monomials = [w_R,w_A,w_B,w_C,w_P]
gamma_matrix_els = [G_new,map_table,S,list_of_eq_indices,Mexp]

# ---------------------------------------------------------------------------------
# Begin code ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------

N = 10
vec = np.linspace(0.0,0.5,N)
out_vec = [[],[]]

for i in range(1):
    
    # alpha = np.sqrt(0.0)
    # r = vec[i]

    # vacs = create_displaced_vacua(2, [alpha,alpha], [0.0,0.0])
    # state = qgt.tensor_product(vacs)    
    # state.two_mode_squeezing(r, modes=[0, 1])
    
    # density_state = qgt.density_matrix_number_basis(state, n_cutoff=nK)
    # density_state = density_state.reshape(nK**2,nK**2)    

    # # omega = {k: 1.0 - density_state[k][k][k][k] for k in range(nK)}
    # omega = {k: 1.0 - density_state[k][k] for k in range(nK)}
    
    u = 0.8
    Pabc_obs = Pabc_obs_generator(u)
    
    omega = {k: vec[i] for k in range(nK)}
    # omega = {k: 0.15 for k in range(nK)}
    
    # Wobs = 4*np.sqrt(2)*np.sqrt(omega[0]*(1-omega[0])) 
    
    out = triangle_QRNG(nA,nB,nC,nZ,nK,omega,Pabc_obs)
    
    out_vec[0] += [vec[i]]
    out_vec[1] += [out]
    
    # print(4*np.sqrt(omega[0]*(1-omega[0])) )
    print(omega,out)
    
    
