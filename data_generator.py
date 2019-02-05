import numpy as np
from expanded_random_representation import ExpandedRandomRepresentation


class DataGenerator:

    def __init__(self, m, num_ltu, beta_ltu, mu_epsilon, sigma_epsilon, mu_weight, sigma_weight):

        self.m = m
        self.num_ltu = num_ltu
        self.beta_ltu = beta_ltu
        self.mu_epsilon = mu_epsilon
        self.sigma_epsilon = sigma_epsilon
        self.mu_weight = mu_weight
        self.sigma_weight = sigma_weight
        self.err_data_gen = ExpandedRandomRepresentation(m, num_ltu, beta_ltu)
        self.err_data_gen.w = np.random.normal(self.mu_weight, self.sigma_weight, (self.num_ltu + 1, 1))

    def get_sample(self):

        x = np.random.randint(2, size=(self.m, 1))
        y = self.err_data_gen.calculate_output(x) + np.random.normal(self.mu_epsilon, self.sigma_epsilon, 1)[0]
        return x, y
