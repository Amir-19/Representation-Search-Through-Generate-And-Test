import numpy as np
from expanded_random_representation import ExpandedRandomRepresentation


class DataGenerator:

    def __init__(self, m, num_ltu, beta_ltu, mu, sigma):

        self.m = m
        self.num_ltu = num_ltu
        self.beta_ltu = beta_ltu
        self.mu = mu
        self.sigma = sigma
        self.ERR_data_gen = ExpandedRandomRepresentation(m, num_ltu, beta_ltu, weights_mode="data_gen")

    def get_sample(self):
        x = np.random.randint(2, size=self.m)
        y = self.ERR_data_gen.calculate_output(x) + np.random.normal(self.mu, self.sigma, 1)[0]
        return x, y
