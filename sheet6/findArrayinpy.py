import numpy as np

a = np.array([[0,1],[2,3],[5,4],[4,3]])
print(a)

b = np.array([2,3])
c = np.array([4,3])

print(b)

print(b in a)

print(np.where(b == a))
print(np.where(c == a))

print(c == a)
print(np.any(c == a, axis = 1))
print(np.all(c == a, axis = 1))
print(np.where(np.all(c == a, axis = 1)))
print(np.where(np.all(b == a, axis = 1)))