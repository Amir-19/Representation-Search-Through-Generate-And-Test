import numpy as np


class ExpandedRandomRepresentation:

    def __init__(self, m, n, beta=0.6, gamma=0.1, gen_test=False, rho=0.005, maturity_threshold=2):
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
        self.lambda_k = 0
        self.step_size = 1
        self.gen_test = gen_test
        self.age = np.zeros((n + 1, 1))
        self.rho = rho
        self.maturity_threshold = maturity_threshold
        self.reg_num = int(np.floor(self.rho * self.n))

    def calculate_output(self, x):
        """

        :param x: input as a vector of bits
        :return: the output of ERR
        """
        assert x.shape == (self.m,) or x.shape == (self.m, 1)
        f = np.dot(x.T, self.v)
        f = np.greater_equal(f, self.theta).astype(int)
        # the bias unit is added here at the last index of vector f
        self.f = np.append(f, [1])
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
        self.lambda_k = (self.lambda_k * (self.k - 1) + feature_norm) / self.k
        self.step_size = self.gamma / self.lambda_k
        temp = self.step_size * delta * self.f
        self.w = np.add(self.w, temp.reshape((self.n + 1, 1)))

        if self.gen_test:
            self.age += 1
            # evaluate the current features and create a mask for the features that need to be regenerated
            generate_mask = np.zeros((self.n + 1, 1))
            idx = np.argpartition(self.w[:-1], self.reg_num, axis=0) # find the least weighted features
            generate_mask[idx[:self.reg_num]] = 1
            # apply the age mask based on the maturity threshold
            age_mask = np.greater_equal(self.age, self.maturity_threshold).astype(int).reshape(self.n + 1, 1)
            generate_mask = np.multiply(generate_mask, age_mask)
            # we generate new features and replace the old ones where generate mask is 1
            gen_num = int(np.dot(generate_mask.T,generate_mask)[0][0])
            generate_mask_idx = np.argwhere(generate_mask == 1)[:, 0]
            if gen_num > 0:
                generated_partial_v = np.random.choice([-1, 1], (self.m, gen_num), p=[0.5, 0.5])
                self.v[:, generate_mask_idx] = generated_partial_v
                # self.age[generate_mask] = 0
                np.putmask(self.age, generate_mask, [0])
                # self.w[generate_mask] = 0.0
                np.putmask(self.w, generate_mask, [0])
                # note: super important, i need to exclude not mature features first and then look for bad features
                # not afterward. now it's kindda not taking full advantage of gen-test ...
        return delta
