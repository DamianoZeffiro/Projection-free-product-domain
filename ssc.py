from utilities import *
num_ssc = 0



def ssc(x, grad, d, alpha_max, indv, bool_fw, L, constraint_set, num_block, p_params=problem_params(), c_obj = None):
    global num_ssc
    # implementation of the SSC procedure for product domains
    # x: point
    # grad: gradient
    # fx: objective value
    # L: Lipschitz constant estimate
    # d: descent direction
    # dict_alpha_max: maximal stepsize dictionary along
    # c_obj: objective class
    # c_constraint_set: constraint set class
    # gamma: armijo stepsize parameter
    # pd: product domain boolean
    # fun_reshape: function to reshape x from dictionary to vector
    # p_params: problem parameters
    # type_m: block selection strategy
    barx = x.copy()
    deltax = x - barx
    gdeltax = 0
    while True:
        a = L * constraint_set.scal_prod_block(d, d, num_block)
        b = 2 * L * constraint_set.scal_prod_block(d, deltax, num_block) + \
            constraint_set.scal_prod_block(grad, d, num_block)
        c = L * constraint_set.scal_prod_block(deltax, deltax, num_block) + gdeltax
        Delta = b**2 - 4*a*c
        if Delta >= 0:
            beta = second_degree_solver(a, b, c)
        else:
            beta = 0
        alpha = min([beta, alpha_max])
        x = constraint_set.compute_step(x, d, alpha, alpha_max, bool_fw, indv)
        deltax = x - barx
        gdeltax = constraint_set.scal_prod_block(grad,deltax, num_block)
        if beta <= alpha_max:
            break
        d, alpha_max, gap, fw_gap, bool_fw, indv = constraint_set.compute_descent_dir_block(
            x, grad, num_block)
        if fw_gap < p_params.tr_fw:
            break
    return x, L, 0

@profile
def parallel_ssc(x, grad, dict_d, dict_alpha_max, dict_indv, dict_bfw, fx, L, c_obj, constraint_set, gamma,
                                       pd=True, type_m='randomized', p_params=None):
    # implementation of the SSC procedure for product domains
    # x: point
    # grad: gradient
    # fx: objective value
    # L: Lipschitz constant estimate
    # d: descent direction
    # dict_alpha_max: maximal stepsize dictionary along
    # c_obj: objective class
    # c_constraint_set: constraint set class
    # gamma: armijo stepsize parameter
    # pd: product domain boolean
    # fun_reshape: function to reshape x from dictionary to vector
    # p_params: problem parameters
    # type_m: block selection strategy
    barx = copy.deepcopy(x)
    list_scores = [0 for j in range(c_obj.n)]
    dict_updates = {}
    for i in dict_d:
        xcurr = constraint_set.get_block(x, i)
        x_partial, L, alpha_curr = ssc(xcurr, grad[i], dict_d[i], dict_alpha_max[i], dict_indv[i], dict_bfw[i], L,
                                                  constraint_set, i,  p_params=p_params, c_obj = c_obj)
        if type_m == 'GS':
            list_scores[i] = np.dot(xcurr - x_partial, grad[i].T)
            dict_updates[i] = x_partial
        else:
            constraint_set.update_block(x, x_partial, i)
    if type_m == 'GS':
        h = np.argmax(list_scores)
        list_indices = [h]
        constraint_set.update_block(x, dict_updates[h], h)
    else:
        list_indices = [j for j in dict_d]
    f_plus = c_obj.objective(x, barx, list_indices)
    deltax = constraint_set.lin_comb(x, barx, 1, -1)
    gdeltax = constraint_set.scal_prod(grad, deltax)
    f_comp = fx + gdeltax * gamma
    if (f_plus > f_comp) and L <= p_params.Lmax:
        L = 2 * L
        return barx, fx, L, 0, list_indices
    elif L > p_params.Lmax:
        print('warning - large Lipschitz constant')
        return barx, fx, L/2, 0, list_indices
    return x, f_plus, L, 0, list_indices