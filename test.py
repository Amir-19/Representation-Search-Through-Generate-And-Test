import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation


A = np.array([1, 7, 9, 2, 0.1, 17, 17, 1.5])
A=A.reshape(A.shape[0],1)
k = 3
b = np.zeros(A.shape[0])
idx = np.argpartition(A[:-1], k, axis=0)
b[idx[:k]] = 1
print(b)
