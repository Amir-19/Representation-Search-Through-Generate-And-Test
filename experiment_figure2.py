import numpy as np
from data_generator import DataGenerator
from expanded_random_representation import ExpandedRandomRepresentation

error_interval = 1000
total_steps = 100000

data_gen = DataGenerator(20, 20, 0.6, 0, 1, 0, 1)

F100 = ExpandedRandomRepresentation(20, 100, 0.6)
F300 = ExpandedRandomRepresentation(20, 300, 0.6)
F1000 = ExpandedRandomRepresentation(20, 1000, 0.6)
F10000 = ExpandedRandomRepresentation(20, 10000, 0.6)
F100000 = ExpandedRandomRepresentation(20, 100000, 0.6)
F1000000 = ExpandedRandomRepresentation(20, 1000000, 0.6)

error_sum_100 = 0
error_sum_300 = 0
error_sum_1000 = 0
error_sum_10000 = 0
error_sum_100000 = 0
error_sum_1000000 = 0

for i in range(total_steps):
    x, y = data_gen.get_sample()

    delta100 = F100.update_weights(x, y)
    delta300 = F300.update_weights(x, y)
    delta1000 = F1000.update_weights(x, y)
    delta10000 = F10000.update_weights(x, y)
    delta100000 = F100000.update_weights(x, y)
    delta1000000 = F1000000.update_weights(x, y)

    error_sum_100 += (delta100*delta100)
    error_sum_300 += (delta300 * delta300)
    error_sum_1000 += (delta1000 * delta1000)
    error_sum_10000 += (delta10000 * delta10000)
    error_sum_100000 += (delta100000 * delta100000)
    error_sum_1000000 += (delta1000000 * delta1000000)

    if (i+1) % error_interval == 0:
        print(i+1, error_sum_100/error_interval, error_sum_300/error_interval, error_sum_1000/error_interval,
              error_sum_10000/error_interval, error_sum_100000/error_interval, error_sum_1000000/error_interval)
        error_sum_100 = 0
        error_sum_300 = 0
        error_sum_1000 = 0
        error_sum_10000 = 0
        error_sum_100000 = 0
        error_sum_1000000 = 0
