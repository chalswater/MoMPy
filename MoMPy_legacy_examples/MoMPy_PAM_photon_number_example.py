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

import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        Functions                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

def PAM_photon_number_MoMPy(nA,nB,nX,nY,nK,omega):
    
    ''' PAM with dimension restrictions '''
    
    [w_R,w_M,w_P] = monomials
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

    # Normalisation of quantum states
    ct += [ G_var_vec[fmap(map_table,[w_R[x]])] == 1.0 for x in range(nX) ]
    ct += [ G_var_vec[fmap(map_table,[w_P[k]])] == 1.0 for k in range(nK) ]
    
    ct += [ G_var_vec[fmap(map_table,[w_R[x],w_P[k]])] == 1.0 - omega[k][x] for k in range(nK) for x in range(nX) ]

    # Normalisation
    ct += [ sum([ G_var_vec[fmap(map_table,[w_M[b][y]])] for b in range(nB) ]) == G_var_vec[fmap(map_table,[0])] for y in range(nY) ]
    
    # Probabilities
    pbxy = {}
    for b in range(nB):
        pbxy[b] = {}
        for x in range(nX):
            pbxy[b][x] = {}
            for y in range(nY):
                pbxy[b][x][y] = G_var_vec[fmap(map_table,[w_R[x],w_M[b][y]])]

    # Quantum state discrimiantion success probability
    W = sum([ pbxy[x][x][0]/nX for x in range(nX) ])
    
    #----------------------------------------------------------------#
    #                  RUN THE SDP and WRITE OUTPUT                  #
    #----------------------------------------------------------------#

    obj = cp.Maximize(W)
    prob = cp.Problem(obj,ct)

    output = []

    try:
        mosek_params = {
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
            }
        prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

    except SolverError:
        something = 10

    return W.value

#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#                                        MAIN CODE                                         #
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#

nB = 2 # Bob's measurement outcomes
nX = 2 # Quantum state preparations
nY = 1 # Bob's measurement settings
nK = 1 # Number of restricted photon number components 

#---------------------------------------------------------------------#
#                        Collect all monomials                        #
#---------------------------------------------------------------------#

# Track operators in the tracial matrix
w_R = [] # Measurement in Alice
w_M = [] # Measurement in Bob
w_P = [] # Projectors onto photon-number components

S_1 = [] # List of first order elements
cc = 1

for x in range(nX):
    S_1 += [cc]
    w_R += [cc]
    cc += 1

for b in range(nB): 
    w_M += [[]]
    for y in range(nY):
        S_1 += [cc]
        w_M[b] += [cc]
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

    S_high += [[w_R[x],w_M[b][y]] for x in range(nX) for b in range(nB) for y in range(nY) ]
    S_high += [[w_R[x],w_R[xx]] for x in range(nX) for xx in range(nX) ]
    S_high += [[w_R[x],w_P[k]] for x in range(nX) for k in range(nK) ]

# Second order elements
some_third = False
if some_third == True:

    S_high += [[w_R[x],w_R[xx],w_R[xxx]] for x in range(nX) for xx in range(nX) for xxx in range(nX) ]

# Set the operational rules within the SDP relaxation
rank_1_projectors = []
rank_1_projectors += [ w_R[x] for x in range(nX) ]
rank_1_projectors += [ w_M[b][y] for y in range(nY) for b in range(nB)]
rank_1_projectors += [ w_P[k] for k in range(nK) ]

orthogonal_projectors = []
orthogonal_projectors += [ [ w_M[b][y] for b in range(nB) ] for y in range(nY) ]

commuting_pairs = [] # commuting pairs of elements
# commuting_pairs += [ [ [ w_R[x] for x in range(nX) ] , [ w_R[x] for x in range(nX) ] ] ]

print('Rank-1 projectors',rank_1_projectors)
print('Orthogonal projectors',orthogonal_projectors)
print('commuting elements',commuting_pairs)

# Collect rules and generate SDP relaxation matrix
start = time.process_time()
[G_new,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S_1,S_1,S_high,rank_1_projectors,orthogonal_projectors,commuting_pairs)
end = time.process_time()

print('Gamma matrix generated in',end-start,'s')
print('Matrix size:',np.shape(G_new))

monomials = [w_R,w_M,w_P]
gamma_matrix_els = [G_new,map_table,S,list_of_eq_indices,Mexp]

# ---------------------------------------------------------------------------------
# Begin code ----------------------------------------------------------------------
# ---------------------------------------------------------------------------------

N = 20
vec = np.linspace(0.0,1.0-1.0/nX,N)
out_vec = [[],[]]

for i in range(N):   
    
    omega = {}
    for k in range(nK):
        omega[k] = {}
        for x in range(nX):
            omega[k][x] = vec[i]

    out = PAM_photon_number_MoMPy(nA,nB,nX,nY,nK,omega)
        
    print(out)
    
    out_vec[0] += [ vec[i] ]
    out_vec[1] += [ out ]
        
plt.plot(out_vec[0],out_vec[1])
    
