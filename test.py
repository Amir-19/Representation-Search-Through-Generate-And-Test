import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation


# A = np.array([1, 7, 9, 2, 0.1, 17, 17, 1.5])
# A=A.reshape(A.shape[0],1)
# k = 3
# b = np.zeros(A.shape[0])
# idx = np.argpartition(A[:-1], k, axis=0)
# b[idx[:k]] = 1
# c = np.zeros(A.shape[0])
# np.putmask(A, b, [-33, -44, -55])
# print(A)


# A = np.array([1, 7, 9, 2, 0.1, 17, 17, 1.5])
# B = np.greater_equal(A, 2).astype(int).reshape(8,1)
# print(B)

# a = np.array([1, 2, 0, 0]).reshape(4,1)
# b = np.array([0, 1, 0, 1]).reshape(4,1)
# print(np.multiply(a,b))



v = np.random.choice([-1, 1], (3, 5), p=[0.5, 0.5])
b = np.array([0, 0, 0, 1, 1]).reshape(5,1)
replace = np.random.choice([-1, 1], (3, 2), p=[0.5, 0.5])

print(v)
print(replace)
# np.putmask(v[:,], b, replace)
wheremask = np.argwhere(b==1)[:,0]
v[:,wheremask] = replace
print(v)