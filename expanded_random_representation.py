import numpy as np


class ExpandedRandomRepresentation:

    def __init__(self, m, n, beta):

        self.m = m
        self.n = n
        self.beta = beta
        self.v = np.random.choice([-1, 1], m * n, p=[0.5, 0.5]).reshape([n, m])
        self.w = np.zeros([n+1,1])
        self.s = np.count_nonzero(self.v == -1, axis=1).reshape([n,1])
        self.threshold = np.ones([n,1])*(self.m * self.beta) - self.s
        self.step_size = None

    def calculate_output(self, x):

        f = np.dot(self.v, x)  # expansion
        f = np.greater(f, self.threshold).astype(int)  # LTUs output
        f.extend([1])  # adding the bias unit to the f
        y = np.dot(self.w, f)  # map
        return y

    def update_weights(self, x, true_y):
        pass
