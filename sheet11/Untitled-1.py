# %% [markdown]
# # Sheet 11 Leonhard Moske
# ## Exercise 30

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
        lam[chi:] = 0
        lambdas.append(lam)
        # print(np.shape(u), np.shape(lam), np.shape(vtmp))
    us.append(np.reshape(vtmp,(int(vshape[0]), 2, 1)))
    lambdas.append([1])
    As = []
    for i in range(L):
        As.append(us[i] * lambdas[i])
    return As



# %%
L = 6
x = np.random.randint(0,16,2**L)
print(x)
# print(np.shape(convert2mps(x, 4, L)[0][1]))
for A in convert2mps(x, 3, L):
    print(np.shape(A))

# %% [markdown]
# ## Exercise 31

# %%
def convert2vetcor(mps, L):
    c = mps[0]
    for i in range(1,L):
        c = np.tensordot(c, mps[i], axes = [1 + i, 0])
    return np.reshape(np.squeeze(np.squeeze(c,axis=0),axis = -1), 2**L)

def overlap(x1, x2):
    return np.sum(x1 * x2)

# %%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(111, projection='3d')

Ls = []
chis = []
over = []


for L in range(6, 18, 1):
    for chi in range(2,9):
        x = np.random.normal(0,1,2**L)
        x /= np.linalg.norm(x)
        xtilde = convert2vetcor(convert2mps(x, 2**chi, L),L)

        Ls.append(L)
        chis.append(2**chi)
        over.append(overlap(x,xtilde))

ax.scatter(Ls,chis, over, s = 100, cmap=cm.coolwarm)

ax.set_xlabel("L")
ax.set_ylabel("chi")
ax.set_zlabel("overlap")
for angle in range(0, 360):
    ax.view_init(5, angle)
    plt.draw()
    plt.pause(.001)
# plt.show()


