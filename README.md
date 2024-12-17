# MoMPy: Moment matrix generation and managing package for SDP hierarchies.

## Introduction

This is a package available in Python to generate moment matrices for SDP hierarchies. The package is built to be intuitive and easy to use. It contains only two relevant functions:

 - **MomentMatrix:** generates the moment matrix with operational equivalences already taken into account (except normalisation).

 - **normalisation_contraints:** takes into account normalisation constraints. This is to be called inside the SDP and added as a constraint!

See the sections below to get more information on the functions and how to use the package.

This package is in constant development.

## Basics and applicable senarios for SDP relaxations

Consider a scenario of two parties. One party (Alice) encodes classical messages x={0,1,...,nX-1} in quantum states {R[x]}. These states are sent to a second party (Bob) who based on the value of a classical input y={0,1,...,nY-1} performs a measurement {M[y][b]} with outcome b={0,1,...,nB}. Alice and Bob can extract the observable correlations through the Born rule: p(b|x,y) = Tr( R[x] @ M[b][y] ). Now, Alice and Bob are given the task of obtaining the maximum of a linear function on the probabilities W = sum(c[b][x][y] * p(b|x,y) ) over all possible state preparations and measurements. To render this optimisation as a semidefinite program (SDP), a relaxation is required. That is, from the list of relevant operators O = {id, R[x], M[y][b]} sample monomials L = {id, R[x], M[y][b], R[x] @ R[xx], R[x] @ M[y][b], M[y][b] @ M[yy][bb], ... } up to a certain order. With those monomials, one then builds a matrix G = Tr(u @ v), for u and v being all monomials in L. Then, since by construction G is positive-semidefinite, and W will appear in the elements of G, one can optimise W given that G is positive semidefinite and get a good approximate solution to the problem.

The problem in building such SDP relaxations lies in building the moment matrix G and identify all elements that are equivalent by intrinsic properties of the operators. For instance, if R[x] are pure states, then the elements in G Tr(R[x]) and Tr(R[x]@R[x]) are equivalent. This package takes care of this burden for you. Specifically, you can specify properties of the relevant operators such as rank-1, orthogonality, commutativity, ... and the package give you the SDP moment matrix. 

> [!NOTE]
> The package is applicable to any optimisation problem that can be rendered as a SDP relaxation with full traces.

## Use of the package: build your first moment matrix!

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

## Use the moment matrix to build SDP hierarchies

Now all complicatd numerical work is done! Here we detail how to properly use the moment matrix to build the SDP relaxation. We will take the example above to illustrate the steps.

> [!NOTE]
> To build a semidefinite program, we use the package CVXPY which can be freely downloaded from using pip.

### Step 1: Define and organize your SDP variables

First things first: we need to define the SDP variables. In our relaxation, these are essentially the elements in the moment matrix. However, with our tool, we do not need to define the moment matrix as a variable itself, but only the list of non-identical elements in the matrix G. This list is essentially the variable _list_of_eq_indices_ extracted from the _MomentMatrix_ function. Therefore, we define a vector of variables, call it _G_var_vec_, as follows
```
G_var_vec = {}
for element in list_of_eq_indices:
    if element == map_table[-1][-1]:
        G_var_vec[element] = 0.0
    else:
        G_var_vec[element] = cp.Variable()
```
What we did here is define a dictionary _G_var_vec_ in which we store the SDP variables. The last element from the _list_of_eq_indices_ (i.e. the element that is equal to _map_table[-1][-1]_) contains the variables that are zero because of the orthogonality properties we specifried above. 

With this vector we can construct the moment matrix _MomMat_ as follows
```
lis = []
for r in range(len(G)):
    lis += [[]]
    for c in range(len(G)):
        lis[r] += [[ G[r][c] ]]
MomMat = cp.bmat(lis)
```

Now _MomMat_ is the matrix containing all the moments in our SDP relaxation, and all equivalence relations considering intrinsic properties of the operators are already accounted for.

> [!NOTE]
> To access each element one has to use the _map_table_ outputted from the Moment Matrix function, with the _fmap_ function. For example, the variable corresponding to Tr(R[x] @ M[y][b]) is accessed through
> ```
> G_var_vec[fmap(map_table,[R[x],M[y][b]])]
> ```
> And the identity element Tr(id) is recovered from choosing the element **[0]** as follows
> ```
> G_var_vec[fmap(map_table,[0])]
> ```


### Step 2: Specify all non-trivial constraints

Now that we properly defined all SDP vairables, we need to add the constraints that we are missing. These can be for example normalisation constraints, certain values of traces or bounded elements which are problem-dependent. 

First, define a list where we will store all constraints:
```
ct = []
```

#### Normalisation

Let us start adding normalisation constraints. We can do it two ways: by hand or using a tool provided by the package.

1. **Normalisation by hand**: Assume that the operator M[y][b] when summed over all "b" one recovers the identity. This implies that the following elements need to be normalised
```
ct += [ sum([ G_var_vec[fmap(map_table,[M[y][b]])] for b in range(nB) ]) == G_var_vec[fmap(map_table,[0])] for y in range(nY) ]
ct += [ sum([ G_var_vec[fmap(map_table,[R[x],M[y][b]])] for b in range(nB) ]) == G_var_vec[fmap(map_table,[R[x]])] for y in range(nY) for x in range(nX) ]
```
and so on including all elements that intervene in the hierarchy.

2. **Normalisation using the package**: The package offers a function that can take care of the normalisation of one operator that appears in the hierarchy. This is done by using the function _normalisation_contraints_. To sue this tool, we suggest to use this method
```
for y in range(nY):
    map_table_copy = map_table[:]
    
    identities = [ term[0] for term in map_table_copy]
    norm_cts = normalisation_contraints(M[y],identities)
    
    for gg in range(len(norm_cts)):
        the_elements = [fmap(map_table,norm_cts[gg][jj]) for jj in range(nB+1) ]
        an_element_is_not_in_the_list = False
        for hhh in range(len(the_elements)):
            if the_elements[hhh] == 'ERROR: The value does not appear in the mapping rule':
                an_element_is_not_in_the_list = True
        if an_element_is_not_in_the_list == False:
            ct += [ sum([ G_var_vec[fmap(map_table,norm_cts[gg][jj])] for jj in range(nB) ]) == G_var_vec[fmap(map_table,norm_cts[gg][nB])]  ]
```
It may not be the most intuitive solution, but it is effective, and takes care of the normalisation of M[y][b] that affects all elements in the hierarchy.

#### Exact values or bounds on certain elements

Other constraints might involve idntifyig the exact value of some elements in the matrix, or at least some bounds. For example, in our example, the trace of quantum states we know must return the identity. Therefore,
```
ct += [ G_var_vec[fmap(map_table,[R[x]])] == 1.0 for x in range(nX)]
```

Additionally, assume that we want to bound the inner product of all state preparations to be at least equal to _d_ (to be specificed). This can be implemented with
```
ct += [ G_var_vec[fmap(map_table,[R[x],R[xx]])] >= d for x in range(nX) for xx in range(nX) ]
```

### Step 3: Define object function and run SDP

Now we only need to define the object function. In our example, we can take the simple state discrimination success probability and only a single input _y_ in Bob's device. That is,
```
W = sum([ G_var_vec[fmap(map_table,[R[x],M[0][x]])] for x in range(nX) ])
```

We aim to maximise the success probability _W_. We can do this using CVXPY as follows
```
obj = cp.Maximize(W)
prob = cp.Problem(obj,ct)

try:
    mosek_params = {
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-1
        }
    prob.solve(solver='MOSEK',verbose=False, mosek_params=mosek_params)

except SolverError:
    something = 10
```

Here we are using the solver _MOSEK_ which can be freely used with an academic license. The solution of the SDP can be accessed through _W.value_.


    
    
    
    
    
    
    
    
    
