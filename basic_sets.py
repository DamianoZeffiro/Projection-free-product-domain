import numpy as np
import random
import matplotlib.pyplot as plt
from line_profiler_pycharm import profile


class product_simplex_el():

    def __init__(self, num_blocks, size_block, rs = 1, obj_class =None):
        self.n = num_blocks
        self.p = size_block
        self.dim = self.n * self.p
        self.x = []
        self.Qx = np.nan
        for j in range(self.n):
            np.random.seed(rs + j)
            y = np.random.exponential(scale=1.0, size=self.p)
            y = y / np.sum(y)
            self.x = self.x + list(y)
        self.x = np.array(self.x)


class product_simplex():

    def __init__(self, num_blocks, size_block, dir_type='fw', round_par=10**(-8)):
        self.p = size_block
        self.n = num_blocks
        self.dir_type = dir_type
        self.round_par = round_par
        self.name = 'p_simplex'
        self.num_blocks = num_blocks

    def change_dir(self, dir_type):
        self.dir_type = dir_type

    def get_block(self, x, num):
        return x.x[num * self.p: self.p * (num + 1)]

    def update_block(self, x, xnew, num):
        x.x[num * self.p: (num + 1) * self.p] = xnew

    def compute_descent_dir(self, x, g, M):
        dict_d = {}
        dict_alpha_max = {}
        dict_gap = {}
        dict_fw_gap = {}
        dict_bool_fw = {}
        dict_v = {}
        for j in M:
            d, alpha_max, gap, fw_gap, bool_fw, ind_v = self.compute_descent_dir_block(self.get_block(x, j), g[j], j)
            dict_d[j] = d
            dict_alpha_max[j] = alpha_max
            dict_gap[j] = gap
            dict_fw_gap[j] = fw_gap
            dict_bool_fw[j] = bool_fw
            dict_v[j] = ind_v
        return dict_d, dict_alpha_max, dict_gap, dict_fw_gap, dict_bool_fw, dict_v

    def lin_comb(self, x1, x2, alpha1, alpha2):
        return alpha1 * x1.x + alpha2*x2.x

    def scal_prod(self, g, deltax):
        return np.sum([np.dot(g[i], deltax[self.p * i: self.p*(i + 1)].T) for i in range(self.n)])

    def scal_prod_block(self, a, b, num_block):
        return np.dot(a, b.T)

    def compute_descent_dir_block(self, x, g, num):
        d, alpha_max, gap, fw_gap, bool_fw, ind_v = getattr(self, self.dir_type)(x, g, num)
        return d, alpha_max, gap, fw_gap, bool_fw, ind_v

    def compute_step(self, x, d, alpha, alpha_max, bool_fw, v_index=np.nan):
        if np.isnan(np.sum(v_index)) or (alpha < alpha_max) or (alpha_max == 0):
            return x + alpha * d
        elif bool_fw:
            xp = np.zeros(np.shape(x))
            xp[v_index] = 1
            return xp
        else:
            xp = x + alpha * d
            xp[v_index]=0
            return xp

    def fw_direction(self, x, g, num_block):
        fw_index = np.argmin(g)
        d_fw = - x.copy()
        d_fw[fw_index] = 1 + d_fw[fw_index]
        alpha_max = 1
        fw_gap = -np.dot(d_fw, g.T)
        return d_fw, alpha_max, fw_gap, fw_gap, True, fw_index

    def aw_direction(self, x, g, num_block):
        afw_components = np.argwhere(x > 0)
        afw_index_grad = np.argmax([g[i] for i in afw_components])
        afw_index = afw_components[afw_index_grad][0]
        d_aw = x.copy()
        d_aw[afw_index] = d_aw[afw_index] - 1
        if d_aw[afw_index] < -self.round_par:
            alpha_max = -x[afw_index]/d_aw[afw_index]
            aw_gap = - np.dot(d_aw, g.T)
        else:
            alpha_max = 0
            aw_gap = 0
        return d_aw, alpha_max, aw_gap, afw_index

    @profile
    def afw_direction(self, x, g, num_block):
        d_fw, alpha_max_fw, fw_gap, fw_gap, bool_fw, fw_index = self.fw_direction(x, g, num_block)
        d_aw, alpha_max_aw, aw_gap, afw_index = self.aw_direction(x, g, num_block)
        if fw_gap >= aw_gap:
            d = d_fw
            alpha_max = alpha_max_fw
            gap = fw_gap
            bool_fw = True
            ind_v = fw_index
        else:
            d = d_aw
            alpha_max = alpha_max_aw
            gap = aw_gap
            bool_fw = False
            ind_v = afw_index
        return d, alpha_max, gap, fw_gap, bool_fw, ind_v