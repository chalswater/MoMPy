#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:41:00 2024

@author: carles
"""

import numpy as np

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Permute(v):
    """ Cyclical permutation """
    return [v[-1]] + v[:-1]

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def Commute(v,index):
    """ Commute v[index] in v """
    store = v[index]

    v_copy = [ii for ii in v]
    v_copy.remove(store)
    
    out = []
    entered = False
    for jj in range(len(v)):
        if jj == index + 1:
            out += [store]
            entered = True
        else:
            if entered == True:
                out += [v_copy[jj-1]]
            else:
                if jj < len(v_copy):
                    out += [v_copy[jj]]

    if len(out) < len(v):
        out = [store] + out
                
    return out

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def fmap(table,i):
    ''' A funciton that maps a value on table[item][0] --> table[item][1] in the same row'''
    found = False
    for item in range(len(table)):
        if i in table[item][0]: # For non-commuting variables, i is a vector!
            found = True
            item_saved = item

    if found == True:
        return table[item_saved][1]
    else:
        return 'ERROR: The value does not appear in the mapping rule'
    
#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def reverse_list(lista):
    return[k for k in reversed(lista)]

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def check_if_id(element,map_table,rank_1_projectors,commuting_elements,orthogonal_projectors):
    
    """ 
        Checks if an element that may be not in the matrix is identical to 
        other elements that are in there (according to the rules)
    """

    id_elements = [map_table[i][0] for i in range(len(map_table))]
    group_of_zeros = id_elements[-1]
    
    equivalences_vec = [] # vector of equivalent elements

    missed_zero = False
    index_id = None
    found_one = False

    # Identical elements by cyclic permutation in traces
    new_vec = element[:]
    for ii in range(len(element)+1):
        if new_vec not in equivalences_vec: # if it is not in the list already, add it
            equivalences_vec += [new_vec]    
        if new_vec in group_of_zeros:    
            missed_zero = True
        new_vec = Permute(new_vec) # Permute members (monomials) of the element
        
    # Identical elements by being rank-1 projectors (B^2 = B)
    if len(element) >= 2:
        for ii in range(len(element)): 
            if element[ii] in rank_1_projectors:
                if element[ii] == element[np.mod(ii+1,len(element))]:
                    new_vec = [element[jj] for jj in range(len(element)) if jj != ii]
                    if new_vec not in equivalences_vec:  #if it is not in the list already, add it
                        equivalences_vec += [new_vec]    
                    if new_vec in group_of_zeros:    
                        missed_zero = True
        
    # Identical elements by commuting rules
    # When you commute an element, you completely chenge the whole set of elements, and so one need to re-check ciclicity and rank-1
    new_vec = element[:]
    new_lists = [] # Each time we commute an element we generate a new element which we will store here
    new_lists += [new_vec]
    diff = 10
    already_counted = []
    while diff > 0: # While we keep adding elements in "new_lists" we will do the following
        len_ini = len(new_lists)
        for term in new_lists:
            if term not in already_counted:
                already_counted += [term]
                new_vec = term[:]
                for hh in range(len(term)):
                    if term[hh] in commuting_elements: #if the element is in the list of commuting elements
                        for ii in range(len(new_vec)+1):
                            new_vec = Commute(new_vec,np.mod(hh+ii,len(new_vec))) # permute members (monomials) of the element
                            if new_vec not in new_lists:
                                new_lists += [new_vec]
                            if new_vec not in equivalences_vec:
                                equivalences_vec += [new_vec]
                            if new_vec in group_of_zeros:    
                                missed_zero = True    
                
                new_vec = term[:]
                for ii in range(len(new_vec)+1): # Include also the cyclic permutations of the new elements
                    if new_vec not in equivalences_vec: # if it is not in the list already, add it
                        equivalences_vec += [new_vec]   
                    #if new_vec not in new_lists:
                    #    new_lists += [new_vec]
                    if new_vec in group_of_zeros:    
                        missed_zero = True
                    new_vec = Permute(new_vec) # Permute members (monomials) of the element
                
                new_vec = term[:]
                if len(new_vec) >= 2:
                    for ii in range(len(new_vec)): # Also include all rank-1 element reductions that can happen
                        if new_vec[ii] in rank_1_projectors:
                            if new_vec[ii] == new_vec[np.mod(ii+1,len(new_vec))]:
                                new_vec_2 = [new_vec[jj] for jj in range(len(new_vec)) if jj != ii]
                                if new_vec_2 not in equivalences_vec:  #if it is not in the list already, add it
                                    equivalences_vec += [new_vec_2]    
                                #if new_vec not in new_lists:
                                #    new_lists += [new_vec]
                                if new_vec_2 in group_of_zeros:    
                                    missed_zero = True
                                    
                # Check if the new element is zero or not (and we did not count it initially)
                new_vec = term[:]
                if len(new_vec) >= 2:
                    for ii in range(len(new_vec)):
                        for kk in range(len(orthogonal_projectors)):
                            if new_vec[ii] in orthogonal_projectors[kk] and new_vec[np.mod(ii+1,len(new_vec))] in orthogonal_projectors[kk]:
                                    if new_vec[ii] != new_vec[np.mod(ii+1,len(new_vec))]:
                                        new_vec_2 = [new_vec[jj] for jj in range(len(new_vec))]
                                        if new_vec_2 not in group_of_zeros:  #if it is not in the list already, add it
                                            group_of_zeros += [new_vec_2]
                                            missed_zero = True

        len_fin = len(new_lists)
        diff = len_fin - len_ini
        
    
    #print(equivalences_vec)
        
    if missed_zero == False: 
        # Check if any of the elements in the equivalences is already accounted for
        
        saved_id = []
        for el in equivalences_vec:
            if found_one == False:
                for ids in id_elements:
                    if found_one == False:
                        if el in ids:
                            found_one = True
                            index_id = id_elements.index(ids)
                
    return [found_one,missed_zero,index_id]

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def normalisation_contraints(element,list_identities_in):
    
    """ 
        Def: 
            List of elements which summed are equal to the last element in each list 
            
        Arguments:
            
            element: list of elements that require normalisation
            
            list_identities_in: lost of equivalences
    
    """
    
    # element: list of elements that require normalisation
    # list_identities_in: list of exsisting equivalences
    
    output_ct = []
    already_counted = []
    
    list_identities = list_identities_in[:]

    for term in list_identities:
        dd = 0
        for pp in range(len(term[dd])):
            yy = 0
            string = []
            if term[dd][pp] == element[yy]:
                new_input = term[dd][:]
                copy_without_element = new_input[:]
                del copy_without_element[pp]
                if len(copy_without_element) < 1:
                    copy_without_element += [0]
                for hh in range(len(element)):
                    new_input[pp] = element[hh]
                    copy = new_input[:]
                    string += [copy]
                string += [copy_without_element]

            if len(string) > 1:
                output_ct += [string]

    """
    
    Output:
        
        list of indices which added are equal to the last one
        
    """

    return output_ct

#------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------

def MomentMatrix(S_1,S_2,higher_order_elements,rank_1_projectors,orthogonal_projectors,commuting_elements,states):
    
    """ 
        Generator of SDP hierarchy Moment matrix
        
        S_1:
            list of first order elements (zero order not included)
            
        S_2:
            list of elements to generate second order terms (zero order not included)
            
        higher_order_elements:
            list of specific elements of higher orders
            
        rank_1_projectors:
            list of rank-1 projectors
            
        orthogonal_projectors:
            lists of orthogonal projectors
            
        commuting_elements:
            list of elements that commute with everything EXCEPT with the states
            
        states:
            delcare what are your states that don't commute with anything (only used for commutation constraints)        
    """    
    
    S = []
    for v in S_1:
        S += [[v]] # Add first order monomials

    for k in S_2:
        for h in S_2:
            S += [[h,k]] # Compute second order monomials
            
    S += higher_order_elements # Add additional higher order elements

    Mexp = [] # Here we store the matrix for all explicit elements
    complete_list_explicit = [] # Here we store all explicit elements in a list
    for r in range(len(S)+1):
        Mexp += [[]]
        for c in range(len(S)+1):
            if r == 0 and c > 0:
                Mexp[r] += [reverse_list(S[c-1])]
            elif c == 0 and r > 0:
                Mexp[r] += [S[r-1]]
            elif c == 0 and r == 0:
                Mexp[r] += [[0]]
            else:
                Mexp[r] += [S[r-1]+reverse_list(S[c-1])]
            if r >= c:
                complete_list_explicit += [ Mexp[r][c] ]
            
    
    # Detect the elemenets that are equal to zero by orthogonality rules
    group_of_zeros = [] # Group of zeros 
    for r in range(len(S)+1):
        for c in range(len(S)+1):
                
            # Orthogonal projectors
            if len(Mexp[r][c]) >= 2:
                for ii in range(len(Mexp[r][c])):
                    for kk in range(len(orthogonal_projectors)):
                        if Mexp[r][c][ii] in orthogonal_projectors[kk] and Mexp[r][c][np.mod(ii+1,len(Mexp[r][c]))] in orthogonal_projectors[kk]:
                                if Mexp[r][c][ii] != Mexp[r][c][np.mod(ii+1,len(Mexp[r][c]))]:
                                    new_vec = [Mexp[r][c][jj] for jj in range(len(Mexp[r][c]))]
                                    if new_vec not in group_of_zeros:  #if it is not in the list already, add it
                                        group_of_zeros += [new_vec]
                                        if Mexp[r][c] in complete_list_explicit:
                                            complete_list_explicit.remove(Mexp[r][c]) # Remove element that was already taken into account

    # Apply symmetry rules 
    id_elements = [] # list containing groups of identical elements
    ss = 0
    for element in complete_list_explicit:     
        
        print(f'\r Building Moment Matrix: {np.round(ss/(len(complete_list_explicit)),2)}%'+'\r',end=' ')
        equivalences_vec = [] # vector of equivalent elements
    
        missed_zero = False
    
        # Identical elements by cyclic permutation in traces
        new_vec = element[:]
        for ii in range(len(element)+1):
            if new_vec not in equivalences_vec: # if it is not in the list already, add it
                equivalences_vec += [new_vec]    
            if new_vec in group_of_zeros:    
                missed_zero = True
            new_vec = Permute(new_vec) # Permute members (monomials) of the element
            
        # Identical elements by being rank-1 projectors (B^2 = B)
        if len(element) >= 2:
            for ii in range(len(element)): 
                if element[ii] in rank_1_projectors:
                    if element[ii] == element[np.mod(ii+1,len(element))]:
                        new_vec = [element[jj] for jj in range(len(element)) if jj != ii]
                        if new_vec not in equivalences_vec:  #if it is not in the list already, add it
                            equivalences_vec += [new_vec]    
                        if new_vec in group_of_zeros:    
                            missed_zero = True
            
        # Identical elements by commuting rules
        # When you commute an element, you completely change the whole set of elements, and so one needs to re-check ciclicity and rank-1
        new_vec = element[:]
        new_lists = equivalences_vec
        diff = len(commuting_elements) # if there are no commuting elements, we don't need to do this re-checking
        already_counted = []
        while diff > 0: # While we keep adding elements in "new_lists" we will do the following
            len_ini = len(new_lists)
            for term in new_lists:
                if term not in already_counted:
                    already_counted += [term]
                    new_vec = term[:]
                    for hh in range(len(term)):
                        if term[hh] in commuting_elements: #if the element is in the list of commuting elements
                            for ii in range(len(new_vec)+1):
                                new_vec = Commute(new_vec,np.mod(hh+ii,len(new_vec))) # permute members (monomials) of the element
                                if new_vec not in new_lists:
                                    new_lists += [new_vec]
                                if new_vec not in equivalences_vec:
                                    equivalences_vec += [new_vec]
                                if new_vec in group_of_zeros:    
                                    missed_zero = True    
                    
                    new_vec = term[:]
                    for ii in range(len(new_vec)+1): # Include also the cyclic permutations of the new elements
                        if new_vec not in equivalences_vec: # if it is not in the list already, add it
                            equivalences_vec += [new_vec]   
                        if new_vec in group_of_zeros:    
                            missed_zero = True
                        new_vec = Permute(new_vec) # Permute members (monomials) of the element
                    
                    new_vec = term[:]
                    if len(new_vec) >= 2:
                        for ii in range(len(new_vec)): # Also include all rank-1 element reductions that can happen
                            if new_vec[ii] in rank_1_projectors:
                                if new_vec[ii] == new_vec[np.mod(ii+1,len(new_vec))]:
                                    new_vec_2 = [new_vec[jj] for jj in range(len(new_vec)) if jj != ii]
                                    if new_vec_2 not in equivalences_vec:  #if it is not in the list already, add it
                                        equivalences_vec += [new_vec_2]    
                                    if new_vec_2 in group_of_zeros:    
                                        missed_zero = True
                                        
                    # Check if the new element is zero or not (and we did not count it initially)
                    new_vec = term[:]
                    if len(new_vec) >= 2:
                        for ii in range(len(new_vec)):
                            for kk in range(len(orthogonal_projectors)):
                                if new_vec[ii] in orthogonal_projectors[kk] and new_vec[np.mod(ii+1,len(new_vec))] in orthogonal_projectors[kk]:
                                        if new_vec[ii] != new_vec[np.mod(ii+1,len(new_vec))]:
                                            new_vec_2 = [new_vec[jj] for jj in range(len(new_vec))]
                                            if new_vec_2 not in group_of_zeros:  #if it is not in the list already, add it
                                                group_of_zeros += [new_vec_2]
                                                missed_zero = True

            len_fin = len(new_lists)
            diff = len_fin - len_ini
        
        if missed_zero == True: # if we detected that an element is a zero, then all is zero
            for el in equivalences_vec:
                if el not in group_of_zeros:
                    group_of_zeros += [el]
        else:
            # Check if any of the elements in the equivalences is already accounted for
            found_one = False
            saved_id = []
            for el in equivalences_vec:
                for ids in id_elements:
                    if el in ids:
                        found_one = True
                        saved_id += [ids]
                        id_elements.remove(ids)
    
            # If so, merge all elements in one without repetitions: 
            if found_one == True:
                for tt in range(len(saved_id)):
                    for iss in saved_id[tt]:
                        if iss not in equivalences_vec:
                            equivalences_vec += [iss]
        
            # Add the equivalences in a bigger list of identities
            id_elements += [equivalences_vec]
        ss += 1

    # Add the zeros
    id_elements += [group_of_zeros] 
        
    Moment_Matrix = np.zeros((len(Mexp),len(Mexp)),dtype=int) # This is the output Gama matrix with numbers in each element
    map_table = [] # Table to map explicit elements to the number group of identities
    already_counted = []
    already_counted_2 = []

    ll = 0
    for l in range(int(len(id_elements))):
        if id_elements[l] not in already_counted_2:
            map_table += [[id_elements[l],ll]]
            already_counted_2 += [id_elements[l]]
            already_counted += id_elements[l]   
            ll += 1
            
    for j in range(len(Mexp)):
        for i in range(j,len(Mexp)):
            Moment_Matrix[i][j] = fmap(map_table,Mexp[i][j])
            Moment_Matrix[j][i] = Moment_Matrix[i][j]
    
    list_of_eq_indices = np.unique(Moment_Matrix) # Unique elements in M

    """
        Outputs:
            
        Moment_Matrix: 
            Moment Matrix with indices of equivalent elements
            
        map_table:
            table used to map explicit elements to their equivalence index in Moment_Matrix
            the last element in map_table are all the zeros emerging from orthogonality
            
        S:
            complete list of all elements
            
        list_of_eq_indices:
            list of all equivalence indices in Moment_Matrix
            
        Mexp:
            Moment matrix with all explicit elements
        
    """
    
    return Moment_Matrix,map_table,S,list_of_eq_indices,Mexp


