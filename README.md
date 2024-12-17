# MoMPy: Moment matrix generation and managing package for SDP hierarchies.

## Introduction

This is a package available in Python to generate moment matrices for SDP hierarchies. The package is built to be intuitive and easy to use. It contains only two relevant functions:

 - **MomentMatrix:** generates the moment matrix with operational equivalences already taken into account (except normalisation).

 - **normalisation_contraints:** takes into account normalisation constraints. This is to be called inside the SDP and added as a constraint!

See the sections below to get more information on the functions and how to use the package.

This package is in constant development.

## Basiscs and applicable senarios for SDP relaxations

Consider a scenario of two parties. One party (Alice) encodes classical messages x={0,1,...,nX-1} in quantum states {R[x]}. These states are sent to a second party (Bob) who based on the value of a classical input y={0,1,...,nY-1} performs a measurement {M[y][b]} with outcome b={0,1,...,nB}. Alice and Bob can extract the observable correlations through the Born rule: p(b|x,y) = Tr( R[x] @ M[b][y] ). Now, Alice and Bob are given the task of obtaining the maximum of a linear function on the probabilities W = sum(c[b][x][y] * p(b|x,y) ) over all possible state preparations and measurements. To render this optimisation as a semidefinite program (SDP), a relaxation is required. That is, from the list of relevant operators O = {id, R[x], M[y][b]} sample monomials L = {id, R[x], M[y][b], R[x] @ R[xx], R[x] @ M[y][b], M[y][b] @ M[yy][bb], ... } up to a certain order. With those monomials, one then builds a matrix G = Tr(u @ v), for u and v being all monomials in L. Then, since by construction G is positive-semidefinite, and W will appear in the elements of G, one can optimise W given that G is positive semidefinite and get a good approximate solution to the problem.

The problem in building such SDP relaxations lies in building the moment matrix G and identify all elements that are equivalent by intrinsic properties of the operators. For instance, if R[x] are pure states, then the elements in G Tr(R[x]) and Tr(R[x]@R[x]) are equivalent. This package takes care of this burden for you. Specifically, you can specify properties of the relevant operators such as rank-1, orthogonality, commutativity, ... and the package give you the SDP moment matrix. 

> [!NOTE]
> The package is applicable to any optimisation problem that can be rendered as a SDP relaxation with full traces.

## Use the package

Here we detial the first steps towards th propper use of the package.

### Installation

To install teh package you only need to download and install it from PyPi using pip. Just write the following command in your terminal:

```
pip install MoMPy
```

Once the pacakge is installed, you are rready to use it.

### Identify the list of relevant operators and sample monomials

The first step to build the SDP hierarchy is to identify and list the relevant operators in your scenario. To illustrate how can you simply do that in python, let's take an example. Consdier the prepare-and-measure scenario from the beginning: Alice prepares quantum states R[x] and Bob performs measurements M[y][b]. These will be our relevant operators. 

> [!IMPORTANT]
> The identity also belongs to the list of relevant operators. However, we do not need to take care of since the package will automatically incorporate it.

To list them we do the following:

1. Define empty lists to store the operators
```
R = []
M = []
```
2. Create a list to store all sampled monomials from the list of operators
```
S = []
```
3. Define an ancilla variable that will count all elements we will go through
```
cc = 1
```
4. Sample all operators in the list S
```
for x in range(nX):
    S += [cc]
    R += [cc]
    cc += 1

for y in range(nY):
    M += [[]]
    for b in range(nB):
        S += [cc]
        M[y] += [cc]
        cc += 1
```
Up to now, we have stored in the list S all elements of first order in our hierarchy. Now we can include higher order monomials to the list. To do so, we will add them in a new list as follows:
```
S_high = []

for x in range(nX):
    for xx in range(nX):
        S_high += [[R[x],R[xx]]]

for y in range(nY):
    for b in range(nB):
        for x in range(nX):
            S_high += [[R[x],M[y][b]]]

for y in range(nY):
    for b in range(nB):
        for yy in range(nY):
            for bb in range(nB):
                S_high += [[M[y][b],M[yy][bb]]]
```
Now in S we have all elements up to first order and in S_high all elements in second order. We could add higher order elements if interested. For this case, we will stay in this level of the hierarchy.

### Write all operational properties of the operators

The next step is to identify and write down which properties the operators will have to obey when building the moment matrix. The package can incorporate teh following properties:

1. **Rank-1 projectors**: If an operator R is rank-1, that is if R@R = R.
```
rank_1 = []
```
2. **Orthogonal projetors**: If the operators are orthogonal projectors, that is if R[x]@R[xx] = R[x] if x == xx or = 0 if x != xx.
```
orthogonal_projectors = []
```
3. **Commuting operators**: If the operator R commutes with any other operator in the list.
```
commuting_operators = []
```
4. **Exceptions for cummutativity**: If the operators R listed to commute with any other operators do not commute with an operator in the list.
```
exceptions_comm = []
```
In our exxample, consider that R and M are rank-1, M are orthogonal projectors for each distinct input y, and R commute with everything. This can be described with:
```
rank_1 += [R[x] for x in range(nX)]
rank_1 += [M[y][b] for y in range(nY) for b in range(nB)]
orthogonal_projectors += [ M[y] for y in range(nY) ]
commuting_operators += [R[x] for x in range(nX)]
```

### Call the pacakge to create the SDP moment matrix

Now we have all ingredients to build the moment matrix for our SDP relaxation. To do so, we will import first the necessary tools to build moment matrices. These are found within the _MOM_ part of the package. Then, one calls the function _MomentMatrix_ as follows:
```
from MoMPy.MoM import *
[G,map_table,S_out,list_of_eq_indices,Gexp] = MomentMatrix(S,[],S_high,rank_1,orthogonal_projectors,commuting_operators,exceptions_comm)
```
The function returns a list of outputs. These are:
1. **G**: Moment matrix with indices. Each index represents an SDP variable.
2. **map_table**: Table used to map lists of operators to SDP variable indices. It takes into account all equivalence relations.
3. **S_out**: Complete list of all monomials in our scenario.
4. **list_of_eq_indices**: This that contains lists with all monomials that are equivalent given the properties we indicated.
5. **Gexp**: The moment matrix, but in each element one finds a list of monomials that are building each variable.

From this big list of outputs, we will mainly only use the two most important ones: the Moment Matrix **G** and the table **map_table** to access all elements in the Moment Matrix.











The main use of the MoMPy package is to build the matrices of moments and identify all equivalence relations in an autamized manner. 

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
 
 
    
    
    
    
    
    
    
    
    
    
    
