#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:28:14 2026

@author: carles
"""

import numpy as np
import cvxpy as cp
from cvxpy import *
import time

# Import MoMPy
from MoMPy.MoM import *

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Functions                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def NPA_MoMPy(nA,nB,nX,nY):
    
    [w_A,w_B] = monomials
    [G_new,map_table,S,list_of_eq_indices,Mexp] = gamma_matrix_els
        
    #----------------------------------------------------#
    #                  CREATE VARIABLES                  #
    #----------------------------------------------------#

    G_var_vec = {}
    for element in list_of_eq_indices:
        if element == map_table[-1][-1]:
            G_var_vec[element] = 0.0 # Zeros form orthogonal projectors
        else:
            G_var_vec[element] = cp.Variable()

    #--------------------------------------------------#
    #                  BUILD MATRICES                  #
    #--------------------------------------------------#
    
    G = {}
    lis = []
    for r in range(len(G_new)):
        lis += [[]]
        for c in range(len(G_new)):
            lis[r] += [G_var_vec[G_new[r][c]]]
    G = cp.bmat(lis)

    #------------------------------------------------------#
    #                  CREATE CONSTRAINTS                  #
    #------------------------------------------------------#
    
    ct = []
   
    ct += [ G >> 0.0 ]
    ct += [ G == G.H ]
    
    # ----------------------------------------------------------------------------------------

    ct += [ G_var_vec[fmap(map_table,[0])] == 1 ]

    # Normalisation
    ct += [ sum([ G_var_vec[fmap(map_table,[w_A[a][x]])] for a in range(nA) ]) == G_var_vec[fmap(map_table,[0])] for x in range(nX) ]
    ct += [ sum([ G_var_vec[fmap(map_table,[w_B[b][y]])] for b in range(nB) ]) == G_var_vec[fmap(map_table,[0])] for y in range(nY) ]
    
    # Probabilities
    pabxy = {}
    for a in range(nA):
        pabxy[a] = {}
        for b in range(nB):
            pabxy[a][b] = {}
            for x in range(nX):
                pabxy[a][b][x] = {}
                for y in range(nY):
                    pabxy[a][b][x][y] = G_var_vec[fmap(map_table,[w_A[a][x],w_B[b][y]])]
    
    # Correlators
    ABxy = {}
    for x in range(nX):
        ABxy[x] = {}
        for y in range(nY):
            ABxy[x][y] = sum([ (-1)**(a+b)*pabxy[a][b][x][y] for a in range(nA) for b in range(nB) ])

    # CHSH (objective function)
    CHSH = ABxy[0][0] + ABxy[1][0] + ABxy[0][1] - ABxy[1][1]
    
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#

    obj = cp.Maximize(CHSH)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10

    return CHSH.value

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        MAIN CODE                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

nA = 2 # Alice's measurement outcomes
nB = 2 # Bob's measurement outcomes
nX = 2 # Alice's measurement settings
nY = 2 # Bob's measurement settings

#---------------------------------------------------------------------#
#                        Collect all monomials                        #
#---------------------------------------------------------------------#

# Track operators in the tracial matrix
w_A = [] # Measurement in Alice
w_B = [] # Measurement in Bob

S_1 = [] # List of first order elements
cc = 1

for a in range(nA): 
    w_A += [[]]
    for x in range(nX):
        S_1 += [cc]
        w_A[a] += [cc]
        cc += 1

for b in range(nB): 
    w_B += [[]]
    for y in range(nY):
        S_1 += [cc]
        w_B[b] += [cc]
        cc += 1
        
# Additional higher order elements
S_high = [] # Uncomment if we only allow up to some 2nd order elements in the hierarchy  

# Second order elements
some_second = False
if some_second == True:

    S_high += [[w_A[a][x],w_B[b][y]] for a in range(nA) for x in range(nX) for b in range(nB) for y in range(nY) ]

# Set the operational rules within the SDP relaxation
rank_1_projectors = []
rank_1_projectors += [ w_A[a][x] for x in range(nX) for a in range(nA)]
rank_1_projectors += [ w_B[b][y] for y in range(nY) for b in range(nB)]

orthogonal_projectors = []
orthogonal_projectors += [ [ w_A[a][x] for a in range(nA) ] for x in range(nX) ]
orthogonal_projectors += [ [ w_B[b][y] for b in range(nB) ] for y in range(nY) ]

commuting_pairs = [] # commuting pairs of elements

print('Rank-1 projectors',rank_1_projectors)
print('Orthogonal projectors',orthogonal_projectors)
print('commuting elements',commuting_pairs)

# Collect rules and generate SDP relaxation matrix
start = time.process_time()
[G_new,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S_1,[],S_high,rank_1_projectors,orthogonal_projectors,commuting_pairs)
end = time.process_time()

print('Gamma matrix generated in',end-start,'s')
print('Matrix size:',np.shape(G_new))

monomials = [w_A,w_B]
gamma_matrix_els = [G_new,map_table,S,list_of_eq_indices,Mexp]

# ---------------------------------------------------------------------------------
# Begin code ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------
    
out = NPA_MoMPy(nA,nB,nX,nY)
    
print(out)
    
    
