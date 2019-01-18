import numpy as np
from expanded_random_representation import ExpandedRandomRepresentation

ERR_data_gen = ExpandedRandomRepresentation(20, 20, 0.6, weights_mode="data_gen")
x = np.random.randint(2, size=20).reshape([20,1])
y = ERR_data_gen.calculate_output(x)
print(y)