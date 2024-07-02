# MoMPy: Moment matrix generation and managing package for SDP hierarchies.

This is a package to generate moment matrices for SDP hierarchies. The package contains the following relevant functions:

 - MomentMatrix: generates the moment matrix with operational equivalences already taken into account (except normalisation).

 - normalisation_contraints: takes into account normalisation constraints. This is to be called inside the SDP and added as a constraint!

More information in the code files. The package is still in development phase.

How does it work?

It's simple. Suppose you have an optimisation problem involving traces of operators A_{a}, B_{y,b} and C_{k} as for example:

maximise Tr[A_{a} * B_{y,b}] 
s.t. Tr[C_{k} * B_{y,b}] >= c_{k,y,b}



1) Define a set of lists of scalar numbers. Each different scalar represents a different operator. to keep track of each operator easily, we suggest to use the following notations:

 - List of operators:
     
    A = [] # Operator A
    B = [] # Operator B
    C = [] # Operator C
    
 - List where we will store the operators:
    
    S = [] # List of first order elements
    
 - Store the operators:
 
    # A has indices A[a]
    cc = 1    
    for a in range(nA):
        S += [cc]
        R += [cc]
        cc += 1
    
    # B has indices B[y][b]
    for y in range(nY):
        B += [[]]
        for b in range(nB): 
            S += [cc]
            B[y] += [cc]
            cc += 1
            
    # C has indices [k]
    for k in range(nK):
        S += [cc]
        C += [cc]
        cc += 1

2) Declare operational relations. These consists in the following:

 - Operators are rank-1: rank_1_projectors
 - Operators are orthogonal for different specific indices: orthogonal_projectors
 - Operators commute with every other element: commuting_variables
 - Operators may not commute with some other operators (which we call states): list_states
 
 For example, suppose all operators are rank-1, 

    rank_1_projectors = []#w_R
    rank_1_projectors += [w_B[y][b] for y in range(nY) for b in range(nB)]
    rank_1_projectors += [w_P[k] for k in range(nK)]

 operators B are orthogonal for indices [b] for every [y], and same for P but for incides [k]

    
    orthogonal_projectors = []
    orthogonal_projectors += [ w_B[y] for y in range(nY)]
    orthogonal_projectors += [ w_P ] 

 and nothing else for now (for simplicity),

    list_states = [] 
    commuting_variables = [] 
    
    
3) If we include 1st order elements, we write S as the first entry of the function. 
If additionally we want to automatically include all 2nd order elements, we write S as the second entry as well. 
If we need additional specific elements of higher order elements, we can include them in the list S_high as for example,

    S_high = []
    for a in range(nA):
        for aa in range(nA):
            S_high += [[A[a],A[aa]]]
            
    for a in range(nA):
        for b in range(nB):
            for y in range(nY):
                S_high += [[A[a],B[y][b]]]
        
    for k in range(nK):
        for b in range(nB):
            for y in range(nY):
                S_high += [[C[k],B[y][b]]]
            
    for a in range(nA):
        for k in range(nK):
              S_high += [[C[k],A[a]]]

Here we included the specific seconnd order elements [[A,A],[A,B],[C,B],[C,A]], but we can include any other higher order elemetns if required.

4) Call MomentMatrix inbuilt function as follows:

[MoMatrix,map_table,S,list_of_eq_indices,Mexp] = MomentMatrix(S,S,S_high,rank_1_projectors,orthogonal_projectors,commuting_variables,list_states)
    
This function returns:

 - MoMatrix: matrix of scalar indices that represent different quantities within the moment matrix. To be used as indices to label SDP variables.
 - map_table: table to map from explicit operators to indices in MoMatrix. This shall be used with the inbuilt matrix: fmap(map_table,i) as
 
        fmap(map_table,[A[a],B[y][b]]) returns the index corresponding to the variable that represents Tr[A[a] * B[y][b]].
        
 - S: list of first order elements that we wrote as input
 - list_of_eq_indices: complete list of unique indices that appear in MoMatrix. These are ordered from lowest to highest.
 - Mexp: Moment matrix with explicit operators as we defined them in the beginning.
 
 
    
    
    
    
    
    
    
    
    
    
    
