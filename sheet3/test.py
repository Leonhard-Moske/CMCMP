# %%
import numpy as np
import itertools as it
import sympy

def genBasis(L, s): #L number of spins, s total spin
    posSpins = np.linspace(-s, s, int((2*s+1)), dtype = float)#generate List of all possible z projections
    #print("possible Spins ", posSpins)
    tmp = it.product(posSpins, repeat = L) # generate object that itterates over all combinations
    return np.asarray([i for i in tmp]) # assemble the list

# %%
def Splus(oldcoef, state, site, s): # site counted from 0 , passed with copy
                        # return new coefficient and new state 
    if (state[site] >= s): # if state is s the operator will return 0
        return 0, np.zeros(L)
    coef = np.sqrt(s*(s+1) - state[site] * (state[site] + 1)) #calculate new coef
    state[site] = state[site] + 1 # update spin state
    return coef*oldcoef, state 

def Sminus(oldcoef, state, site, s): # analog to Splus
    if (state[site] <= -s ):
        return 0, np.zeros(L)
    coef = np.sqrt(s*(s+1) - state[site] * (state[site] - 1))
    state[site] = state[site] - 1
    return coef*oldcoef, state 

def Sz(coef, state, site): 
    return state[site]*coef, state

def Hamiltonian(J, statei, L, statej, s):
    H = 0
    for i in range(L-1): #apply the operators to every site until L-1
        coef1, state1 = Splus(*Sminus(1, np.copy(statei), i+1, s), i, s)
        coef2, state2 = Sminus(*Splus(1, np.copy(statei), i+1, s), i, s)
        coef3, state3 = Sz(*Sz(1, np.copy(statei), i+1), i)
        if np.array_equal(state1, statej): #compare if (<i|H)|j> is 0 for every term
            H += coef1*J*0.5
        if np.array_equal(state2, statej):
            H += coef2*J*0.5
        if np.array_equal(state3, statej):
            H += coef3 * J
    return H

def calcMatrix(Basis, J, L, s):# calculate the basis by calculating the hamiltonian with all combination from the basis
    Matrix = np.empty((len(Basis),len(Basis)))
    for i, j in it.product(range(len(Basis)), repeat=2):
        Matrix[i,j] = Hamiltonian(J, Basis[i], L, Basis[j], s)
    return Matrix


# %%
def totalZ(state, L):
    totalZ = 0
    for i in range(L):
        totalZ += Sz(1, state, i)[0]
    return totalZ

def totalZmatrix(Basis, L):# calculate the basis by calculating the hamiltonian with all combination from the basis
    Matrix = np.zeros((len(Basis),len(Basis))) # since we know that the basisstates are eigenstates of totalZ we only have to compute the diagonal 
    for i in range(len(Basis)):
        Matrix[i,i] = totalZ(Basis[i], L)
    return Matrix

def splitBasisTotZ(Basis, L):
    splitBasis= []
    sortedBasis = sorted(Basis, key = lambda state : (totalZ(state, L)))
    tmpBasis = []
    oldz = totalZ(sortedBasis[0],L)
    for state in sortedBasis:
        z = totalZ(state, L)
        if z != oldz:
            splitBasis.append(tmpBasis)
            tmpBasis = [state]
            oldz = z
        else:
            tmpBasis.append(state)
            oldz = z
    splitBasis.append(tmpBasis)  
    return splitBasis

def inversionOperator(state):
    return np.flip(state)

def inversionMatrix(Basis): 
    Matrix = np.empty((len(Basis), len(Basis)))
    for i in range(len(Basis)):
        for k in range(len(Basis)):
            if (np.array_equal(Basis[i], inversionOperator(np.copy(Basis[k])))):
                Matrix[i,k] = 1
            else:
                Matrix[i,k] = 0
    return Matrix

def genFamily(state, operator, L): # generate the family of a state with an operator
    """generate the family of a state with an operator"""
    family = np.asarray([state])
    for r in range(L):
        family = np.append(family, [operator(family[-1])], axis=0)
    return np.unique(family,axis=0)

def getRepresentativ(state, operator, L): # the representativ is the first entry in the sortet family
    tmp = genFamily(state,operator, L)
    for i in range(L):
       tmp = tmp[tmp[:,i].argsort()]
    return tmp[0]

def computeNormalization(state, k, operator): #k in multiple of pi
    #check if k*familysize is 2*z where z is whole number
    fn = len(genFamily(state, operator, L))
    if((fn*k)%2 == 0): # check if state is normalizable
        return 1/np.sqrt(fn) 
    else:
        return 0

def allRepresentatives(basis, operator, L):
    return np.unique([getRepresentativ(state, operator, L) for state in basis], axis = 0)

def calcMatrixInversionsymmetrie(basis, J, L, s, operator ,k):
    representatives = allRepresentatives(basis, operator, L)
    normalizable = np.empty((0,L)) # get number of states
    for rep in representatives:
        if(computeNormalization(rep, k, operator) != 0):
            normalizable = np.append(normalizable, [rep], axis = 0)
    Matrix = np.empty((len(normalizable),len(normalizable)))
    for i, j in it.product(range(len(normalizable)), repeat=2):
        Matrix[i,j] = 2 * ( Hamiltonian(J, normalizable[i], L, normalizable[j], s) /
                        (computeNormalization(normalizable[i], k, operator) *computeNormalization(normalizable[j], k, operator)))
        Matrix[i,j] += 2 * k*(Hamiltonian(J, normalizable[i], L, operator(normalizable[j]), s)  / 
                        (computeNormalization(normalizable[i], k, operator) *computeNormalization(normalizable[j], k, operator)))

    return Matrix


# %%
import sympy

L = 4
J = 1
s = 0.5

testindex = 3

print(genBasis(L, s)[testindex])

print(genFamily(genBasis(L,s)[testindex], inversionOperator, L))

print(getRepresentativ(genBasis(L,s)[testindex], inversionOperator, L))

print(computeNormalization(genBasis(L,s)[testindex], 0, inversionOperator))
print(1/np.sqrt(2))
# normalization is 1 if f is 1
# normalization is 1/sqrt2 if f is 2
# so my states are never duplicates

counter = 0
repr0 = np.empty((0,L))

for state in genBasis(L, s):
    #if(computeNormalization(state, k,inversionOperator) != 0):
        # print(repr)
        #print(state, getRepresentativ(state, inversionOperator, L))
        #print(genFamily(state, inversionOperator, L))
    #print(state, getRepresentativ(state, inversionOperator, L))
    # if(np.any(np.all(getRepresentativ(state,inversionOperator,L) != repr0, axis=0))):
    #     counter += 1
    repr0 = np.append(repr0, [getRepresentativ(state, inversionOperator,L)], axis = 0)
    #     #print(counter, state, getRepresentativ(state, inversionOperator, L))
    

print(np.unique(repr0,axis = 0))


print(allRepresentatives(genBasis(L,s), inversionOperator, L))

# %%
repr1 = np.empty((0,L))


for rep in np.unique(repr0,axis = 0):
    for k in [0,1]:
        if(computeNormalization(rep, k,inversionOperator) != 0):
            print("counts", rep, k)
            repr1 = np.append(repr1, [getRepresentativ(rep, inversionOperator,L)], axis = 0)
        else:
            print("norm 0",rep)



# %%
print(calcMatrixInversionsymmetrie(genBasis(L,s), J, L, s, inversionOperator, 1))

# %%
import sympy



iMatrix = inversionMatrix(genBasis(L, s))
hMatrix = calcMatrix(genBasis(L,s),J,L,s)


#sympy.Matrix(np.matmul(iMatrix, hMatrix) - np.matmul(hMatrix, iMatrix))

# %%
# print(splitBasisInversion(genBasis(L,s), L)[0])

# ham = calcMatrix(splitBasisInversion(genBasis(L,s), L)[0], J, L, s)
# sympy.Matrix(ham)



# %%
from timeit import default_timer as timer
import matplotlib.pyplot as plt
J = 1
s = 0.5

n = 2

measureTime = []
times = []
Length = []

for L in range(1,n+1):
    start = timer()
    Basis = splitBasisInversion(genBasis(L,s), L)# sort the basis by the total spin
    allEvalues = []
    for stotal in Basis:
        matrix = calcMatrix(stotal, J, L, s)
        #print(matrix)
        Evalues , Evectors = np.linalg.eigh(matrix) # calculate the eigenvalues
        allEvalues = np.append(allEvalues , Evalues)
    allEvalues = allEvalues.flatten()
    #print(allEvalues)
    if len(allEvalues) >= 4: # print the lowest 4 Eigenvalues
        plt.scatter([1/L]*4, np.sort(allEvalues)[:4], color = "blue")
    else:
        plt.scatter([1/L]*len(allEvalues), np.sort(allEvalues), color = "blue")
    times.append(timer() - start)
    Length.append(L)

measureTime.append([Length, times])
times = []
Length = []

plt.title(f"lowest 4 Eigenvalues of a {s}-Spin system in the Heisenberg model with rearanged basis")
plt.xlabel(r"$\frac{1}{L}$", fontsize = 15)
plt.ylabel("Eigenvalue", fontsize = 15)
plt.grid()
plt.show()

s = 1
    
for L in range(1,n+1):
    start = timer()
    #print(genBasis(L,s), J, L, s)
    Basis = splitBasisInversion(genBasis(L,s), L)# sort the basis by the total spin
    #print(Basis)
    allEvalues = []
    #print(allEvalues)
    for stotal in Basis:
        matrix = calcMatrix(stotal, J, L, s)
        Evalues , Evectors = np.linalg.eigh(matrix) # calculate the eigenvalues
        #print(L, Evalues, allEvalues)
        allEvalues = np.append(allEvalues , Evalues)
    allEvalues = allEvalues.flatten()
    if len(allEvalues) >= 4: # print the lowest 4 Eigenvalues
        plt.scatter([1/L]*4, np.sort(allEvalues)[:4], color = "blue")
    else:
        plt.scatter([1/L]*len(allEvalues), np.sort(allEvalues), color = "blue")
    times.append(timer() - start)
    Length.append(L)
    
measureTime.append([Length, times])
times = []
Length = []


plt.title(f"lowest 4 Eigenvalues of a {s}-Spin system in the Heisenberg model with rearanged basis")
plt.xlabel(r"$\frac{1}{L}$", fontsize = 15)
plt.ylabel("Eigenvalue", fontsize = 15)
plt.grid()
plt.show()


