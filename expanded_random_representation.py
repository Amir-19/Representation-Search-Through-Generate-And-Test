import numpy as np


class ExpandedRandomRepresentation:

    def __init__(self, m, n, beta, gamma=0.1, weights_mode="zero", data_gen_dis_mu=0, data_gen_dis_sigma=1):

        self.m = m
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self.v = np.random.choice([-1, 1], m * n, p=[0.5, 0.5]).reshape([m, n])
        if weights_mode == "zero":
            self.w = np.zeros([n+1, 1])
        elif weights_mode == "data_gen":
            self.w = np.random.normal(data_gen_dis_mu, data_gen_dis_sigma, n+1).reshape([n+1, 1])
        self.s = np.count_nonzero(self.v == -1, axis=0).reshape([n, 1])
        self.threshold = np.ones([n, 1])*(self.m * self.beta) - self.s
        self.k = 0
        self.step_size = 1
        self.f = None
        self.lambdak = 0

    def calculate_output(self, x):

        f = np.dot(x.T, self.v)  # expansion
        f = np.greater(f, self.threshold.T).astype(int)  # LTUs output
        self.f = np.append(f, [1]).reshape([self.n+1, 1])  # adding the bias unit to the f
        y = np.dot(self.w.T, self.f)  # map
        return y

    def update_weights(self, x, true_y):
        self.k += 1
        est_y = self.calculate_output(x)
        delta = true_y - est_y
        feature_norm = np.dot(self.f.T,self.f)[0]
        self.lambdak = (self.lambdak * (self.k - 1) + feature_norm) / self.k
        self.step_size = self.gamma / self.lambdak
        self.w = self.w + self.step_size * delta * self.f
        return delta
