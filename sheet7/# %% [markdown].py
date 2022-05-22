# %% [markdown]
# # Sheet7 Leonhard Moske
# 

# %%
import numpy as np
import itertools as it
import sympy

def pullGaussWfFactors(mean, std, N):
    return np.random.normal(mean, std, N)

def genBasis(L, s): #L number of spins, s total spin
    posSpins = np.linspace(-s, s, int((2*s+1)), dtype = float)#generate List of all possible z projections
    #print("possible Spins ", posSpins)
    tmp = it.product(posSpins, repeat = L) # generate object that itterates over all combinations
    return np.asarray([i for i in tmp]) # assemble the list


def Splus(oldcoef, state, site, s): # site counted from 0 as tupel , passed with copy
                        # return new coefficient and new state 
    if (state[site] >= s): # if state is s the operator will return 0
        return 0, np.zeros(state.shape)
    coef = np.sqrt(s*(s+1) - state[site] * (state[site] + 1)) #calculate new coef
    state[site] = state[site] + 1 # update spin state
    return coef*oldcoef, state 

def Sminus(oldcoef, state, site, s): # analog to Splus
    if (state[site] <= -s ):
        return 0, np.zeros(state.shape)
    coef = np.sqrt(s*(s+1) - state[site] * (state[site] - 1))
    state[site] = state[site] - 1
    return coef*oldcoef, state 

def Sz(coef, state, site): 
    return state[site]*coef, state

def Hamiltonian(statei , statej, s, J, Lx, Ly):
    H = 0
    for i in range(Lx*Ly): #apply the operators to every site in x direction
        # print("index,",i)
        if i%Lx == 0:
            # print("x Bounds",i, i+Lx-1)
            coef1, state1 = Splus(*Sminus(1, np.copy(statei), i+Lx-1, s), i, s)
            coef2, state2 = Sminus(*Splus(1, np.copy(statei), i+Lx-1, s), i, s)
            coef3, state3 = Sz(*Sz(1, np.copy(statei), i+Lx-1), i)
            if np.array_equal(state1, statej): #compare if (<i|H)|j> is 0 for every term
                H += coef1*J*0.5
            if np.array_equal(state2, statej):
                H += coef2*J*0.5
            if np.array_equal(state3, statej):
                H += coef3 * J
        else:
            # print("x normal", i, i-1)
            coef1, state1 = Splus(*Sminus(1, np.copy(statei), i-1, s), i, s)
            coef2, state2 = Sminus(*Splus(1, np.copy(statei), i-1, s), i, s)
            coef3, state3 = Sz(*Sz(1, np.copy(statei), i-1), i)
            if np.array_equal(state1, statej): #compare if (<i|H)|j> is 0 for every term
                H += coef1*J*0.5
            if np.array_equal(state2, statej):
                H += coef2*J*0.5
            if np.array_equal(state3, statej):
                H += coef3 * J
        # apply in y direction
        coef1, state1 = Splus(*Sminus(1, np.copy(statei), i-Lx, s), i, s)
        coef2, state2 = Sminus(*Splus(1, np.copy(statei), i-Lx, s), i, s)
        coef3, state3 = Sz(*Sz(1, np.copy(statei), i-Lx), i)
        if np.array_equal(state1, statej): #compare if (<i|H)|j> is 0 for every term
            H += coef1*J*0.5
        if np.array_equal(state2, statej):
            H += coef2*J*0.5
        if np.array_equal(state3, statej):
            H += coef3 * J
    return H

def calcMatrix(Basis, J, Lx, Ly, s):# calculate the basis by calculating the hamiltonian with all combination from the basis
    Matrix = np.empty((len(Basis),len(Basis)))
    for i, j in it.product(range(len(Basis)), repeat=2):
        # print(i,j)
        Matrix[i,j] = Hamiltonian(Basis[i], Basis[j], s,J, Lx, Ly)
    return Matrix

def totalZ(state, Lx,Ly):
    totalZ = 0
    for i in range(Lx*Ly):
        totalZ += Sz(1, state, i)[0]
    return totalZ

def splitBasisTotZ(Basis, Lx,Ly):
    splitBasis= []
    sortedBasis = sorted(Basis, key = lambda state : (totalZ(state, Lx,Ly)))
    tmpBasis = []
    oldz = totalZ(sortedBasis[0],Lx,Ly)
    for state in sortedBasis:
        z = totalZ(state, Lx,Ly)
        if z != oldz:
            splitBasis.append(tmpBasis)
            tmpBasis = [state]
            oldz = z
        else:
            tmpBasis.append(state)
            oldz = z
    splitBasis.append(tmpBasis)  
    return splitBasis

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
        w = np.add(A.dot(v[j]), (-1)*beta[j]*v[j-1])
        alpha[j] =  np.dot(w,v[j])
        w = np.add(w, -1*alpha[j]*v[j]) 
        beta[j+1] = np.linalg.norm(w)
        v[j+1] = w/beta[j+1]
    return v, alpha, beta[1:]

def wrapLancoz(csr_matrix, L, m, v0):
    
    # v0 = np.random.rand(2**L)
    v, alpha, beta = lancoz(csr_matrix, v0, m)

    T = np.zeros((m,m))

    for i in range(m):
        T[i,i] = alpha[i]
    for i in range(m-1):
        T[i,i+1] = beta[i]
        T[i+1,i] = beta[i]

    return T , v

from scipy.linalg import eig, inv


def diagonalizeT(T, beta):
    eVa, eVe = eig(T)

    D = np.zeros(np.shape(T))

    np.fill_diagonal(D, np.exp(np.multiply(eVa.real, -beta/2)))

    return D, eVe



def realtimeEvolution(ham, beta, Lx,Ly,m,v0):
    T , vTrans = wrapLancoz(csr_matrix(ham),Lx*Ly,m, v0)
    print(T)
    vTrans = np.delete(vTrans, m, 0)

    D, eVe = diagonalizeT(T, beta)


    expBetaH = np.linalg.multi_dot ([np.transpose(vTrans),eVe,D,inv(eVe)])#([v,inv(eVe),D,eVe])
    return np.matmul(expBetaH, [1] + [0]*(m-1))

def norm(phi):
    return np.vdot(phi,phi)

# %%
Lx = 2
Ly = 3
s = 0.5
J = 1

m = 200

beta = 0.05
Nbeta = 2# 300
betas = np.arange(2*beta,beta*(Nbeta + 2),beta)


ham = calcMatrix(genBasis(Lx*Ly,s),J,Lx,Ly,s)
ham2 = np.matmul(ham,ham)

Nsamples = 3

Es = []
E2s = []
cVs = []

for k in range(Nsamples):

    phi = pullGaussWfFactors(0,1/(2**(Lx*Ly)),2**(Lx*Ly))


    phiPrime = realtimeEvolution(ham, beta, Lx,Ly, m, phi)


    E = []
    E2 = []
    cV = []
    cV.append(betas[0]**2*(np.vdot(phiPrime, ham2.dot(phiPrime)) - np.vdot(phiPrime, ham.dot(phiPrime))**2))
    for i in range(1, Nbeta):
        print(k,i)
        phiPrime = realtimeEvolution(ham, beta, Lx,Ly, m, phiPrime)


    #print(np.vdot(phiPrime, ham2.dot(phiPrime)))

    # print(beta**2*(np.vdot(phiPrime, ham2.dot(phiPrime)) - np.vdot(phiPrime, ham.dot(phiPrime))**2))
        E.append(np.vdot(phiPrime, ham.dot(phiPrime))/norm(phiPrime))
        E2.append(np.vdot(phiPrime, ham2.dot(phiPrime))/norm(phiPrime))
        cV.append((betas[i]**2)*(np.vdot(phiPrime, ham2.dot(phiPrime))/norm(phiPrime) - (np.vdot(phiPrime, ham.dot(phiPrime))/norm(phiPrime))**2))

        phiPrime = phiPrime / norm(phiPrime)

    Es.append(E)
    E2s.append(E2)
    cVs.append(cV)


# %%
import matplotlib.pyplot as plt
plt.plot(np.mean(Es, axis = 0))
plt.show()
plt.plot(np.mean(E2s, axis = 0))
plt.show()
plt.plot(1/betas,np.mean(cVs, axis = 0))
plt.show()

# %% [markdown]
# The weird behavoir of the heat capacity comes from statistical divergins in the calculation. It appears in some of the times

# %%
# spin
m = 100
splitBasis = splitBasisTotZ(genBasis(Lx*Ly,s),Lx,Ly)

for basis in splitBasis:
    ham = calcMatrix(basis,J,Lx,Ly,s)
    print(ham)

    ham2 = np.matmul(ham,ham)    
    phi = pullGaussWfFactors(0,1/(np.shape(ham)[0]),np.shape(ham)[0])
    print(phi)

    phiPrime = realtimeEvolution(ham, beta, Lx,Ly, np.shape(ham)[0], phi)


