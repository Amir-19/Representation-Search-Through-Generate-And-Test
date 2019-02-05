import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation

a=np.asarray([1,2,3])
print(a.shape)
b=np.asarray([[1],[2],[3]])
print(b.shape)
print(a+b)