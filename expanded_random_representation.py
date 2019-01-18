import numpy as np


class ExpandedRandomRepresentation:

    def __init__(self, m, n, beta, weights_mode="zero", data_gen_dis_mu=0, data_gen_dis_sigma=1):

        self.m = m
        self.n = n
        self.beta = beta
        self.v = np.random.choice([-1, 1], m * n, p=[0.5, 0.5]).reshape([m, n])
        if weights_mode == "zero":
            self.w = np.zeros([n+1, 1])
        elif weights_mode == "data_gen":
            self.w = np.random.normal(data_gen_dis_mu, data_gen_dis_sigma, n+1).reshape([n+1, 1])
        self.s = np.count_nonzero(self.v == -1, axis=1).reshape([n, 1])
        self.threshold = np.ones([n, 1])*(self.m * self.beta) - self.s
        self.step_size = None
        self.sample_number = 0

    def calculate_output(self, x):

        f = np.dot(x.T, self.v)  # expansion
        f = np.greater(f, self.threshold.T).astype(int)  # LTUs output
        f = np.append(f,[1])  # adding the bias unit to the f
        y = np.dot(self.w.T, f)  # map
        return y

    def update_weights(self, x, true_y):
        pass
