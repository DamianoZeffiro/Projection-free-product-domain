import numpy as np
from scipy.optimize import minimize_scalar
import scipy.special


def Q_graph_creator(n, p, alpha, rng):
    np.random.seed(rng)
    Q = np.random.random((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                Q[i, i] = alpha
            elif i < j:
                if Q[i, j] < p:
                    Q[i, j] = 1
                    Q[j, i] = 1
                else:
                    Q[i, j] = 0
                    Q[j, i] = 0
    return Q


def block_Q_creator(num_blocks, n, s, alpha = 0.5, rng = 0, epsilon = 0):
    np.random.seed(rng)
    y = np.random.exponential(scale=1.0, size=num_blocks)
    y = y / np.sum(y)
    y = y * num_blocks
    size_tot = num_blocks * n
    Q_tot = np.zeros((size_tot, size_tot))
    p = p_calculator(n, s)
    for j in range(num_blocks):
        seed_curr = j + rng
        Q_curr = Q_graph_creator(n, p, alpha, seed_curr)
        Q_tot[j*n:(j+1)*n, j*n:(j+1)*n] = Q_curr * y[j]
    Q2 = np.random.randn(size_tot, size_tot)
    Q2 = Q2 + Q2.T
    Q_tot = Q_tot + epsilon * Q2
    return -Q_tot


def p_calculator(n, s):
    size_c = int(np.round(n * s))
    p = (1/ scipy.special.binom(n, size_c)**(2/(size_c * (size_c - 1))))
    return p
