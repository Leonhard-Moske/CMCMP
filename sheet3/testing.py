import numpy as np

list_np_arrays = np.array([[1., 1.], [1., 2.]])
array_to_check = np.array([1., 2.])

is_in_list = np.any(np.all(array_to_check == list_np_arrays, axis=1))

print(is_in_list)

array_to_check2 = np.array([1., 2.])
list_np_arrays = np.array([1., 1.])
array_to_check = np.array([1., 2.])

np.equal(array_to_check,array_to_check2)
print(np.equal(array_to_check,array_to_check2))