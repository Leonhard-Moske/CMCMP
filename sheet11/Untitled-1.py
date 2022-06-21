# %%
import numpy as np

def convert2mps(x,chi,L):
    us = []
    lambdas = []
    vtmp = x
    vshape = (1,2**(L))
    for i in range(1,L):
        u , lam, vtmp = np.linalg.svd(np.reshape(vtmp, (int(vshape[0]*2), int(vshape[1]/2))), full_matrices = False)
        vshape = np.shape(vtmp)
        ushape = np.shape(u)
        us.append(np.reshape(u,(int(ushape[0]/2), 2, ushape[1])))
        lambdas.append(lam)
        print(np.shape(u), np.shape(lam), np.shape(vtmp))
    us.append(np.reshape(vtmp,(int(vshape[0]), 2, vshape[1]/2)))
    lambdas.append([1]*2)
    return us, lambdas

# %%
L = 6
x = np.random.randint(0, L, 2**L)
print(convert2mps(x,4, L)[0])


