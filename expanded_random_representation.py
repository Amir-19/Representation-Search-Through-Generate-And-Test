import numpy as np


class ExpandedRandomRepresentation:

    def __init__(self, m, n, beta=0.6, gamma=0.1, rho=200.0):
        """

        :param m: input size (in bits)
        :param n: number of hidden LTU units
        :param beta: the proportion of the bits that have to match the prototype
        :param gamma: effective step size
        :param rho: replacement rate for generation of new features
        """
        self.m = m
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self.v = np.random.choice([-1, 1], (m, n), p=[0.5, 0.5])
        self.smin = np.count_nonzero(self.v == -1, axis=0) * -1
        self.theta = self.smin + (self.beta * self.m)
        self.w = np.zeros((n + 1, 1))
        self.f = None
        self.k = 0
        self.lambdak = 0
        self.step_size = 1
        self.age = np.zeros((n + 1, 1))
        self.rho = rho

    def calculate_output(self, x):
        """

        :param x: input as a vector of bits
        :return: the output of ERR
        """
        assert x.shape == (self.m,) or x.shape == (self.m, 1)
        f = np.dot(x.T, self.v)
        f = np.greater_equal(f, self.theta).astype(int)
        '''
            NOTE: it is very important to notice that the indexes are changed now  
            as the bias LTU unit is not connected to v. so the i node in in v is i+1 node in w    
            TODO: fix this and put bias unit at the last index 
        '''
        self.f = np.append([1], f)
        y = np.dot(self.f, self.w)[0]
        return y

    def update_weights(self, x, y_target):
        """

        :param x: input as a vector of bits
        :param y_target: target output for the input x
        :return: the prediction error
        """
        self.k += 1
        y_est = self.calculate_output(x)
        delta = y_target - y_est
        feature_norm = np.dot(self.f.T, self.f)
        self.lambdak = (self.lambdak * (self.k - 1) + feature_norm) / self.k
        self.step_size = self.gamma / self.lambdak
        temp = self.step_size * delta * self.f
        self.w = np.add(self.w, temp.reshape((self.n + 1, 1)))
        return delta

    def update_weights_gentest(self, x, y_target):
        """

        :param x: input as a vector of bits
        :param y_target: target output for the input x
        :return: the prediction error
        """
        self.k += 1
        y_est = self.calculate_output(x)
        delta = y_target - y_est
        feature_norm = np.dot(self.f.T, self.f)
        self.lambdak = (self.lambdak * (self.k - 1) + feature_norm) / self.k
        self.step_size = self.gamma / self.lambdak
        temp = self.step_size * delta * self.f
        self.w = np.add(self.w, temp.reshape((self.n + 1, 1)))
        return delta
