# %%
import numpy as np
import scipy
import matplotlib.pyplot as plt
import itertools


hopping=1


def make_basis(L):
    basis = []
    for s in itertools.product((0,1), repeat=L):
        basis.append(s)
    return basis

# %%
def density(state, site):
    return state[site]

def cdag(state,site):
    if state[site]==1:
        return None
    else:
        retstate = list(state)
        retstate[site]=1
        retstate=tuple(retstate)
        nfermions = sum(state[:site])
        
        if nfermions%2==0:
            return (retstate, 1)
        else:
            return (retstate, -1)
        
def c(state,site):
    if state[site]==0:
        return None
    else:
        retstate = list(state)
        retstate[site]=0
        retstate=tuple(retstate)
        nfermions = sum(state[:site])
        
        if nfermions%2==0:
            return (retstate, 1)
        else:
            return (retstate, -1)
            

def createH(L):
    basis = make_basis(L)
    dim=len(basis)
    H=np.zeros((dim,dim))
    
    for row,s in enumerate(basis):
        diagmatel = 0.0
        for i in range(L):
            diagmatel+= density(s,i)*density(s,(i+1)%L)
        H[row,row]=diagmatel
        
        for i in range(L):
            ret = c(s,(i+1)%L)
            if ret!=None:
                s2,sign = ret
                
                ret2 = cdag(s2,i)
                if ret2!=None:
                    s3,sign2=ret2
                    
                    col=basis.index(s3)
                    H[row,col]=sign*sign2*hopping
                    H[col,row]=sign*sign2*hopping
    return H
        
        

# %%
def translate(state):
    state2 = tuple(list(state[1:])+[state[0]])
    parity = sum(state[1:])
    if state[0]==1 and parity%2!=0:
        return state2, -1
    else:
        return state2, 1

# %%
L = 4

testindex = 9

print(make_basis(L)[testindex])
print(translate(make_basis(L)[testindex])[0])
print(translate(make_basis(L)[testindex])[1])

# %%
import itertools as it

def translateOp(statei, statej, params = None):
    if np.all(np.equal(statei,translate(statej)[0])):
        return translate(statej)[1]
    else:
        return 0

def genMatrix(operator, basis):
    Matrix = np.empty((len(basis), len(basis)))
    for i, j in it.product(range(len(basis)), repeat=2):
        Matrix[i,j] = operator(basis[i],basis[j])
    return Matrix

# %%
print(genMatrix(translateOp,make_basis(L)))

# %%
print(np.matmul(genMatrix(translateOp,make_basis(L)),createH(L)) - np.matmul(createH(L),genMatrix(translateOp,make_basis(L))))
print(np.matmul(np.transpose(genMatrix(translateOp,make_basis(L))),genMatrix(translateOp,make_basis(L))))


# %%
print(np.matmul(np.transpose(genMatrix(translateOp,make_basis(L))),np.matmul(genMatrix(translateOp,make_basis(L)),createH(L))))
print(createH(L))

# %%
import matplotlib.pyplot as plt

L = 3

print(np.linalg.eigvals(genMatrix(translateOp,make_basis(L))))

x = [z.real for z in np.linalg.eigvals(genMatrix(translateOp,make_basis(L)))]
y = [z.imag for z in np.linalg.eigvals(genMatrix(translateOp,make_basis(L)))]

circle = plt.Circle((0, 0), 1, color='k',fill = False)
fig, ax = plt.subplots()
ax.add_patch(circle)
ax.scatter(x, y)
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.grid()

fig.show()

# %%
def translate(state):
    state2 = tuple(list(state[1:])+[state[0]])
    parity = sum(state[1:])
    if state[0]==1 and parity%2!=0:
        return state2, 1 # neglet the sign
    else:
        return state2, 1

# %%
import sympy
sympy.Matrix(np.matmul(genMatrix(translateOp,make_basis(L)),createH(L)) - np.matmul(createH(L),genMatrix(translateOp,make_basis(L))))


# %%
def translate(state):
    state2 = tuple(list(state[1:])+[state[0]])
    parity = sum(state[1:])
    if state[0]==1 and parity%2!=0:
        return state2, -1
    else:
        return state2, 1

# %% [markdown]
# ## Exercise 11
# 
# 

# %%
def make_spin_basis(L):
    basis = []
    for s in itertools.product((0,1), repeat=2*L):
        basis.append(s)
    return basis

# %%
L = 4

make_spin_basis(L)

# %%
def spin_density(state, site):
    return (state[site] - 0.5)

def cdag(state,site):
    if state[site]==1:
        return None
    else:
        retstate = list(state)
        retstate[site]=1
        retstate=tuple(retstate)
        nSpinfermions = sum(state[:site])
        
        if nSpinfermions%2==0:
            return (retstate, 1)
        else:
            return (retstate, -1)

def c(state,site):
    if state[site]==0:
        return None
    else:
        retstate = list(state)
        retstate[site]=0
        retstate=tuple(retstate)
        nfermions = sum(state[:site])
        
        if nfermions%2==0:
            return (retstate, 1)
        else:
            return (retstate, -1)
        

def create_spin_H(L):
    basis = make_spin_basis(L) #returns basis which state is 2*L long
    dim=len(basis)
    H=np.zeros((dim,dim))
    
    for row,s in enumerate(basis):
        diagmatel = 0.0
        for i in range(L*2):
            diagmatel+= spin_density(s,2*i)*spin_density(s,(2*i+1)%(L*2))
        H[row,row]=diagmatel
        
        for i in range((L*2)):
            ret = c(s,(i+1)%L)
            if ret!=None:
                s2,sign = ret
                
                ret2 = cdag(s2,i)
                if ret2!=None:
                    s3,sign2=ret2
                    
                    col=basis.index(s3)
                    H[row,col]=sign*sign2*hopping
                    H[col,row]=sign*sign2*hopping
    return H
        

# %%
import itertools as it
import numpy as np

def genBasis(L): #L number of fermions
    tmp = it.product([0,1], repeat = 2*L) # generate object that itterates over all combinations
    return np.asarray([i for i in tmp]) # assemble the list

# %%
L = 3

print(genBasis(L))

# %%
def spin_density(state, site):
    return (state[site] - 0.5)

def c(state, coef, site): #state is copy
    if (state[site] == 0):
        return np.zeros(len(state)), 0
    else:
        state[site] = 0
        if (np.sum(state[:site])%2 == 0):
            return state , 1*coef
        else:
            return state , -1*coef
    
def cdag(state, coef, site):
    if (state[site] == 1):
        return np.zeros(len(state)), 0
    else:
        state[site] = 1
        if (np.sum(state[:site])%2 == 0):
            return state , 1*coef
        else:
            return state , -1*coef

def hubbardHamiltonian(statei,statej, L,t,U): #spin up at even indicies spin down at odd
    hDiag = 0
    if ((np.array_equal(statei,statej))):
        hDiag = U * np.sum([spin_density(statei, 2*i)*spin_density(statei, (2*i +1)%(2*L)) for i in range(L)])
    hOfdiag = 0
    for s in range(L):
        for sigm in [0,1]:
            state1, coef1 = cdag(*c(np.copy(statei),1,2*s + sigm),(2*(s+1) + sigm)%(2*L))
            state2, coef2 = c(*cdag(np.copy(statei),1,2*s + sigm),(2*(s+1) + sigm)%(2*L))
            if np.array_equal(state1, statej):
                hOfdiag += coef1 * (-t)
            if np.array_equal(state2, statej):
                hOfdiag += coef2 * (-t)
    return hDiag + hOfdiag

def calcMatrix(Basis, L, t,U):# calculate the basis by calculating the hamiltonian with all combination from the basis
    Matrix = np.empty((len(Basis),len(Basis)))
    for i, j in it.product(range(len(Basis)), repeat=2):
        Matrix[i,j] = hubbardHamiltonian(Basis[i], Basis[j], L, t, U)
    return Matrix

def transtionOP(state):
    newState = np.append(state[2:],state[:2], axis = 0)
    parity = sum(state[2:])
    sign1 = (parity%2)*(-1)*state[0]
    sign2 = (parity%2)*(-1)*state[1]
    if (sign1 == sign2):
        sign = 1
    else:
        sign = -1
    return newState, sign

def translateOp(statei, statej, params = None):
    if np.array_equal(statei,transtionOP(statej)[0]):
        return transtionOP(statej)[1]
    else:
        return 0

def genMatrix(operator, basis):
    Matrix = np.empty((len(basis), len(basis)))
    for i, j in it.product(range(len(basis)), repeat=2):
        Matrix[i,j] = operator(basis[i],basis[j])
    return Matrix


def genFamily(state, operator, L): # generate the family of a state with an operator
    """generate the family of a state with an operator"""
    family = np.asarray([[state, 0, 1]], dtype = object)
    for r in range(1,L): #not 2*L
        family = np.append(family, [[operator(family[-1][0])[0],r,operator(family[-1][0])[1] * family[-1][2]]], axis=0)
    return family

def getRepresentativ(state, operator, L): # the representativ is the first entry in the sortet family
    tmpa =genFamily(state,operator, L)[:,0]
    tmp = []
    for i in tmpa:
        tmp.append(i)
    tmp = np.asanyarray(tmp,dtype = object)
    for i in range(len(tmp[0])): # over the length of the states
       tmp = tmp[tmp[:,i].argsort()]
    return tmp[0]

def get_shift_sign_ofstate(state,operator,L):
    rep = getRepresentativ(state,operator,L)
    s = rep
    sign=1
    shift=0
    for x in range(L):
        # print("equ", s, state)
        if np.array_equal(s, state):
            return shift, sign
        else:
            s,sgn = operator(s)
            sign*=sgn
            shift+=1
    if np.array_equal(s, state):
        return shift, sign


def get_norm(state, nk,operator, L ):
    k=2.*np.pi*nk/L
    fam = genFamily(state,operator,L)
    
    fam = tuple(fam)

    different_states = set()
    for s,r,sign in fam:
        different_states.add(tuple(s))
    
    prefactors = {s:0.0 for s in different_states}
    
    for s, r, sign in fam:
        prefactors[tuple(s)] += sign*np.exp(1.j*k*r)
    
    norm = 0.0
    for s in prefactors:
        norm += np.abs( prefactors[s] )**2
        
    return np.sqrt(norm)

def calcHam(L,operator):
    hamils = []
    dimsum = 0
    for nk in range(L):
        k=2.*np.pi*nk/L
    
        basis = genBasis(L)

        sector_reps={}
        for s in basis:
            rep = getRepresentativ(s,operator,L)
            #fam = create_family(rep)
            norm=get_norm(rep,nk, operator, L)
            if norm>1e-6: 
                if tuple(rep) not in sector_reps:
                    sector_reps[tuple(rep)]=norm
        secdim = len(sector_reps.keys())
        print(nk, secdim)
        dimsum+=secdim
        
        sector_basis=list(sector_reps.keys())
        
        
        # here, we have the sector reps and norms
        Hk = np.zeros((secdim, secdim), dtype=np.complex128)
        
        for row, rep in enumerate(sector_basis):
            matel=0.0
            
            Hk[row,row] = np.sum([spin_density(rep, 2*i)*spin_density(rep, (2*i +1)%(2*L)) for i in range(L)])
            
            #for i in range(L):
            #    matel+= # density(rep,i)*density(rep,(i+1)%L)
            # Hk[row,row]=matel
            
            for sigma in [0,1]:
                for l in range(L):
                    lp1=((l+1)*2 + sigma)%(L*2)
                    
                    for l1,l2 in [(2*l + sigma,lp1), (lp1, 2*l + sigma)]:
                        if rep[l1]==1 and rep[l2]==0:
                            s1, sgn1 = c(np.asarray(rep),1,l1)
                            s2, sgn2 = cdag(s1,1,l2)

                            

                            rep2 = getRepresentativ(s2,operator,L)
                            
                            if (rep2 == sector_basis).all(1).any():
                                col = np.argwhere(rep2 == sector_basis) #sector_basis.index(rep2)
                                repshift, repsign = get_shift_sign_ofstate(s2,operator,L)
                                
                                norm1 = sector_reps[tuple(rep)]
                                norm2 = sector_reps[tuple(rep2)]
                                                    
                                    
                                matel = norm2/norm1 * hopping * sgn1 * sgn2 * repsign * np.exp(1.j*k*repshift)
                                
                                Hk[row,col] += matel
        hamils.append(Hk)
    return hamils
    
    
    kspec = np.linalg.eigvalsh(Hk)
    # specs.extend(kspec)

# %%
testindex = 6

a = genBasis(L)
print(c(np.copy(a[testindex]), 1,1))
print(a[testindex], len(a))

print(transtionOP(a[testindex]))

# %%
#genFamily(a[testindex],transtionOP,L)

print(genFamily(a[testindex],transtionOP,L), "RESULT")
print(getRepresentativ(a[testindex],transtionOP,L))
print(get_shift_sign_ofstate(a[testindex +13],transtionOP,L))

# %%
print(get_norm(a[testindex],3,transtionOP,L))

# %%
print(hubbardHamiltonian(a[testindex],a[testindex],L,1,1))

print(hubbardHamiltonian(a[testindex],a[testindex + 4],L,1,1))

# %%
import sympy
#sympy.Matrix(calcMatrix(genBasis(L),L,1,1))
#sympy.Matrix(genMatrix(translateOp,genBasis(L)))

# %%
print(np.matmul(genMatrix(translateOp,genBasis(L)),calcMatrix(genBasis(L),L,1,1)) - np.matmul(calcMatrix(genBasis(L),L,1,1),genMatrix(translateOp,genBasis(L))))
print(np.matmul(np.transpose(genMatrix(translateOp,genBasis(L))),genMatrix(translateOp,genBasis(L))))

# %%
dimsum = 0
for nk in range(L):
    sector_reps={}
    for s in genBasis(L):
        rep = getRepresentativ(s,transtionOP,L)
        #fam = genFamily(rep)
        norm=get_norm(rep,nk,transtionOP,L)
        if norm>1e-6: 
            if tuple(rep) not in sector_reps:
                sector_reps[tuple(rep)]=norm
    secdim = len(sector_reps.keys())
    print(nk, secdim)
    dimsum+=secdim
print("total ", dimsum, " should be ", 4**L)

# %%
print(L)
print(calcHam(L,transtionOP))


