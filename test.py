import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation
data_gen = DataGenerator(20,20,0.6,0,1)
rep = ExpandedRandomRepresentation(20,1000000,0.6)
for i in range(1000000):
    x, y = data_gen.get_sample()
    delta = rep.update_weights(x, y)
    print(delta*delta)