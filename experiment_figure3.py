import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation

error_interval = 10000
total_steps = 1000000
# total_steps = 5

data_gen = DataGenerator(20, 20, 0.6, 0, 1, 0, 1)

S1000 = ExpandedRandomRepresentation(20, 10000, 0.6, gen_test=True)
F1000 = ExpandedRandomRepresentation(20, 10000, 0.6, gen_test=False)

error_sum_S1000 = 0
error_sum_F1000 = 0


for i in range(total_steps):
    x, y = data_gen.get_sample()

    deltaS1000 = S1000.update_weights(x, y)
    deltaF1000 = F1000.update_weights(x, y)

    error_sum_S1000 += (deltaS1000 * deltaS1000)
    error_sum_F1000 += (deltaF1000 * deltaF1000)

    if (i + 1) % error_interval == 0:
        print(i + 1, error_sum_S1000 / error_interval, error_sum_F1000 / error_interval)
        error_sum_S1000 = 0
        error_sum_F1000 = 0
