import numpy as np


a = np.asarray([[1,2,3],[2,2,2]])
b = np.asarray([2,4,2])

c = np.count_nonzero(a == 2, axis=1).reshape([2,1])
print(c)