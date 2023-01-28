import numpy as np
import random
import matplotlib.pyplot as plt
from line_profiler_pycharm import profile
from graph_computer import block_Q_creator





class quadratic_obj():

    def __init__(self, alpha=5, n=1, p=1, epsilon=1, sparsity = 0.1, rng = 0):
        # initialize problem data and possibly solution
        # n: num blocks
        # p: size block
        # sparsity: approximately, size of largest clique for each block
        # alpha: regularization coefficient for max clique formulation
        # epsilon: perturbation to be added to block diagonal
        self.name = 'quadratic'
        self.rng = rng
        self.n = n
        self.p = p
        self.size = self.n * self.p
        size = self.size
        self.sparsity = sparsity
        self.epsilon = epsilon / p
        self.alpha = alpha
        self.Q = block_Q_creator(n, p, sparsity, alpha=alpha, epsilon=epsilon, rng=rng)
        self.b = np.zeros(size)
        self.c = 0
        self.num_reset = self.p
        self.counter = 0
        self.stage = 0

    def recompute(self, rng):
        self.Q = block_Q_creator(self.n, self.p, self.sparsity, alpha=self.alpha, epsilon=self.epsilon * self.p,
                                 rng=rng)

    def objective(self, x, xold, block =[]):
        assert(self.stage == 0)
        self.stage = 1
        if len(block) != 1 or (self.counter % self.num_reset == 0):
            x.Qx = np.dot(self.Q, x.x.T)
        else:
            block_curr = block[0]
            left_block = block_curr * self.p
            right_block = (block_curr + 1) * self.p
            Qx_delta = self.Q[:, left_block: right_block] @ (x.x[left_block:right_block] - xold.x[left_block:right_block])
            x.Qx = xold.Qx + Qx_delta
        self.counter = self.counter + 1
        return np.dot(x.x, x.Qx.T) + np.dot(self.b, x.x.T) + self.c

    def gradient(self, x, block = []):
        assert(self.stage == 1)
        self.stage = 0
        gradtot = 2 * x.Qx.T + self.b
        return {i: gradtot[i*self.p:(i + 1) *self.p] for i in range(self.n)}