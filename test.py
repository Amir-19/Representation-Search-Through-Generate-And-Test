import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation

a=np.random.choice([-1, 1], (3, 5), p=[0.5, 0.5])
print(a)
a[0,:] = [1,2, 3, 4, 5]
print(a)