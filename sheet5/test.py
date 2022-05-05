# %% [markdown]
# # Sheet5 Leonhard Moske
# 
# ## Exercise 13

# %%
import numpy as np
import itertools as it
import sympy

def genBasis(L, s): #L number of spins, s total spin
    posSpins = np.linspace(-s, s, int((2*s+1)), dtype = float)#generate List of all possible z projections
    #print("possible Spins ", posSpins)
    tmp = it.product(posSpins, repeat = L) # generate object that itterates over all combinations
    return np.asarray([i for i in tmp]) # assemble the list

def Splus(oldcoef, state, site): # site counted from 0 , passed with copy
                        # return new coefficient and new state 
    if (state[site] == 0.5): #s): # if state is s the operator will return 0
        return 0, np.zeros(L)
    # coef = 1  #s = 1/2 #coef = np.sqrt(s*(s+1) - state[site] * (state[site] + 1)) #calculate new coef
    state[site] = 0.5 #state[site] + 1 # update spin state
    return oldcoef, state 

def Sminus(oldcoef, state, site): # analog to Splus
    if (state[site] == -0.5): # -s ):
        return 0, np.zeros(L)
    # coef = 1 #coef = np.sqrt(s*(s+1) - state[site] * (state[site] - 1))
    state[site] = -0.5 # state[site] - 1
    return oldcoef, state #coef*oldcoef 

def Sz(coef, state, site): 
    return state[site]*coef, state

def Hamiltonian(J, statei, L, statej, s):
    H = 0
    # if (np.array_equal(statei, statej)):
    #     for i in range(L):
    #         H += statei[i]*statei[(i+1)%L] * J
            #H += Sz(*Sz(1, np.copy(statei), (i+1)%L), i)[0] * J
    for i in range(L): #apply the operators to every site until L-1
        coef1, state1 = Splus(*Sminus(1, np.copy(statei),(i+1)%L), i)
        coef2, state2 = Sminus(*Splus(1, np.copy(statei), (i+1)%L), i)
        #coef3, state3 = Sz(*Sz(1, np.copy(statei), i+1), i)
        if np.array_equal(state1, statej): #compare if (<i|H)|j> is 0 for every term
            H += coef1*J*0.5
        if np.array_equal(state2, statej):
            H += coef2*J*0.5
        #if np.array_equal(state3, statej):
        #    H += coef3 * J
    return H

def HamiltonianDiag(J, statei, L):
    H = 0
    for i in range(L):
        H += statei[i]*statei[(i+1)%L] * J
    return H


def calcMatrix(Basis, J, L, s):# calculate the basis by calculating the hamiltonian with all combination from the basis
    Matrix = np.empty((len(Basis),len(Basis)))
    for i in range(len(Basis)):
        Matrix[i,i] = HamiltonianDiag(J, Basis[i], L)
    for i, j in it.permutations(range(len(Basis)), repeat=2):
        Matrix[i,j] = Hamiltonian(J, Basis[i], L, Basis[j], s)
    return Matrix 


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

# %%
import numpy as np

def csr(matrix): #matrix has to be rows in first dimension
    elements , coloumns, pointer = [] , [], [0]
    for row in matrix:
        elements = np.append(elements, row[np.nonzero(row)])
        coloumns = np.append(coloumns, np.nonzero(row))
        pointer = np.append(pointer,np.count_nonzero(row) + pointer[-1])
    return elements, [int(c) for c in coloumns], pointer

def matrixVectorproduct(Matrix,vector):
    e, c, p  = csr(Matrix)
    returnVec = []
    for i in range(len(vector)):
        returnVec.append(np.sum(np.multiply(np.take(vector,c[p[i]:p[i+1]]),e[p[i]:p[i+1]])))
    return returnVec

matrix = np.asanyarray([[0,0,1,0],[3,2,0,0],[0,0,4,0],[0,0,0,0]])
print(csr(matrix))
vec = np.asarray([1,2,3,4])

print(matrixVectorproduct(matrix,vec))

L = 9

ham = calcMatrix(genBasis(L,0.5),1,L,0.5)
print(csr(ham))



# %% [markdown]
# ## Exercise 14

# %%
# %%
from scipy.sparse import csr_matrix 

def lancoz(A, v_un, m): #A is scipy sparse matrix
    n = len(v_un)
    alpha = np.empty(m)
    beta = np.empty(m)
    v = np.empty((m+1,n))
    v0 = v_un / np.linalg.norm(v_un)
    v[0] = v0
    beta[0] = 0
    w = A.dot(v[0])
    alpha[0] = np.dot(w,v[0])
    w = np.add(w, -1*alpha[0]*v[0])
    beta[1] = np.linalg.norm(w)
    v[1] = w/beta[1]
    for j in range(1,m-1):
        # print(v)
        w = np.add(A.dot(v[j]), (-1)*beta[j]*v[j-1])
        alpha[j] =  np.dot(w,v[j])
        w = np.add(w, -1*alpha[j]*v[j]) 
        beta[j+1] = np.linalg.norm(w)
        v[j+1] = w/beta[j+1]
    return v, alpha, beta[1:]

L = 8
m = 8

mat = csr_matrix(calcMatrix(genBasis(L,0.5),1,L,0.5))
H = calcMatrix(genBasis(L,0.5),1,L,0.5)

v0 = np.random.rand(2**L)
v , alpha, beta = lancoz(mat,v0,m)

T = np.zeros((m,m))

for i in range(m):
    T[i,i] = alpha[i]
for i in range(m-1):
    T[i,i+1] = beta[i]
    T[i+1,i] = beta[i]

eigenT = np.linalg.eigvals(T)
eigenH = np.linalg.eigvals(H)

print(np.unique(eigenH), eigenT)


