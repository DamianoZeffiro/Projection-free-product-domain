import numpy as np
import random
import matplotlib.pyplot as plt
from line_profiler_pycharm import profile
from objectives import *
from basic_sets import *
from utilities import *
import copy
from ssc import *
import time

itcurr = 0

np.seterr(all='raise')

@profile
def general_pd(c_obj, constraint_set, p_params, x_start, type_m='randomized'):
    global itcurr
    # first order projection free method for product domains
    # c_obj: objective class
    # constraint_set: constraint_set class
    # p_params: problem parameters
    # x_start: starting point
    # type_m: block selection strategy. Possible values: 'randomized', 'GS' (Gauss-Southwell), 'parallel',
    #         'parallel_fwv' (parallel frank wolf variant, as in the paper)
    # bool_ssc: boolean indicator to decide whether or not to use the SSC
    fw_gap_vec = []
    fun_vec = []
    sparsity_vec = []
    num_grad_vec = []
    L = p_params.L
    verbosity = p_params.verbosity
    gamma = p_params.gamma
    x = copy.deepcopy(x_start)
    fx = c_obj.objective(x, x)
    fun_vec.append(fx)
    num_grad_vec.append(0)
    sparsity_vec.append(constraint_set.n * constraint_set.p)
    num_grad_comp = 0
    k = 0
    sparsity_curr, sparsity_curr_blocks = sparsity_counter(x, class_obj=c_obj)
    time_start = time.time()
    while True:
        itcurr = itcurr + 1
        M = block_selection(type_m, constraint_set)
        grad = c_obj.gradient(x, block=M)
        if p_params.stopping_crit == 'max_block_updates':
            if type_m == 'GS':
                num_grad_comp = num_grad_comp + 1
            else:
                num_grad_comp = num_grad_comp + len(M)
        elif p_params.stopping_crit == 'max_grad':
            num_grad_comp = num_grad_comp + len(M)
        elif p_params.stopping_crit == 'max_time':
            num_grad_comp = time.time() - time_start
        else:
            num_grad_comp = k
        if verbosity:
            print(type_m + ' ' + constraint_set.dir_type + ': ' + str(num_grad_comp) + ' iterations')
        dict_d, dict_alpha_max, dict_gap, dict_fw_gap, dict_bool_fw, dict_indv = constraint_set.compute_descent_dir(x, grad, M)
        for i in M:
            assert(dict_gap[i] >= 0 and dict_alpha_max[i] >= 0)
        fw_gap_estimate = np.sum(dict_fw_gap[i] for i in dict_fw_gap)
        if verbosity:
            print('fw gap is ' + str(fw_gap_estimate) + '\n' + 'Lipschitz estimate is ' + str(L) + '\n' +
                  'nonzero components of x are ' + str(sparsity_curr))
            pass
        if p_params.stopping_crit in ['max_grad', 'max_epochs', 'max_block_updates']:
            if num_grad_comp > p_params.bound:
                break
        elif p_params.stopping_crit == 'max_time':
            if time.time() - time_start > p_params.bound:
                break
        sparsity_vec.append(sparsity_curr)
        fw_gap_vec.append(fw_gap_estimate)
        fun_vec.append(fx)
        num_grad_vec.append(num_grad_comp)
        gap = np.sum([dict_gap[M[i]] for i, vali in enumerate(M)])
        if gap > 0.0 and fw_gap_estimate > 10**(-7):
            x, fx, L, alpha_curr, Mdef = parallel_ssc(x, grad, dict_d, dict_alpha_max, dict_indv, dict_bool_fw, fx, L, c_obj,
                                                constraint_set, gamma, pd=True, type_m=type_m, p_params=p_params)
            sparsity_curr, sparsity_curr_blocks = sparsity_counter(x, sparsity_old = sparsity_curr_blocks,
                                                                   list_blocks = Mdef, class_obj = c_obj)
        else:
            c_obj.stage = 1
        k = k + 1
    if verbosity and ('p_simplex' in c_obj.name):
        print(np.linalg.norm(x.x - c_obj.xstar))
    return x, np.array(fw_gap_vec), np.array(fun_vec), np.array(sparsity_vec), np.array(num_grad_vec)

