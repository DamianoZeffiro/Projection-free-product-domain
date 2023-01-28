from utilities import *
import time
import matplotlib.pyplot as plt
import pickle

dict_x_label = {'max_grad':{'function':'gradients', 'sparsity':'gradients', 'solution':'local searches'},
                'max_epochs':{'function':'epochs', 'sparsity':'epochs'}, 'max_block_updates':
                    {'function':'block updates', 'sparsity': 'block updates'}, 'max_time':
                    {'function':'time', 'sparsity': 'time'}}
dict_y_label = {'function':'error estimate', 'sparsity':'l0 norm', 'solution': 'error estimate'}




def main_comparison(objective, constraint_set, list_dict_strategies,
                    p_params = problem_params(1, 10**(-3), 0.1, verbosity=True),
                    num_objectives=1, num_restarts = 1, run_algo=True, global_opt=0,
                    step=0, xticks=2):
    list_dict_plots = [{} for j in list_dict_strategies]
    list_times = [0 for j in list_dict_strategies]
    num_iter = num_objectives * num_restarts
    suptitle = constraint_set.name
    filename = p_params.stopping_crit + '_dict_plots_' + str(objective.n) + '_' + str(objective.p)
    if global_opt == 1:
        filename = filename + '_sol'
    if run_algo:
        f_curr = 0
        for numj, j in enumerate(list_dict_strategies):
            list_dict_strategies[numj]['f_best'] = np.zeros(num_restarts)
        for it in range(num_iter):
            objective_curr = int(np.floor(it/num_restarts))
            it_restart = it % num_restarts
            if it_restart == 0:
                objective.recompute(objective_curr)
            x_start = product_simplex_el(num_blocks=objective.n, size_block=objective.p,
                                         rs = (num_iter + it) * objective.size, obj_class=objective)
            for numj, j in enumerate(list_dict_strategies):
                if constraint_set.name == 'p_simplex' and (it % num_restarts != 0) and global_opt == 1:
                    x_start.x = constraint_set.lin_comb(j['x_start'], x_start, 1 - step, step)
                dict_curr = list_dict_plots[numj]
                dir_type = j['dir_type']
                constraint_set.change_dir(dir_type)
                t_start = time.time()
                print(j)
                x, fw_gap_vec, fun_vec, sparsity_vec, num_grad_vec = general_pd(objective, constraint_set,
                                                                                          p_params, x_start,
                                                                                          type_m=j['strategy'])
                if it_restart == 0 or (fun_vec[-1] < list_dict_strategies[numj]['f_best'][objective_curr]):
                    list_dict_strategies[numj]['f_best'][objective_curr] = fun_vec[-1]
                    list_dict_strategies[numj]['x_start'] = x
                t_tot = time.time() - t_start
                list_times[numj] += t_tot
                if p_params.stopping_crit == 'max_time':
                    fun_vec, new_time_vec = time_aggregator(num_grad_vec, fun_vec, 0.01, p_params.bound + 0.2)
                    sparsity_vec, new_time_vec = time_aggregator(num_grad_vec, sparsity_vec, 0.01, p_params.bound + 0.2)
                    fw_gap_vec, new_time_vec = time_aggregator(num_grad_vec, fw_gap_vec, 0.01, p_params.bound + 0.2)
                    num_grad_vec = new_time_vec
                if it == 0:
                    dict_curr['function'] = np.zeros((num_iter, len(fun_vec)))
                    dict_curr['sparsity'] = np.zeros((num_iter, len(fun_vec)))
                    dict_curr['solution'] = np.zeros((num_objectives, num_restarts + 1))
                if it_restart == 0:
                    dict_curr['solution'][objective_curr, it_restart] = fun_vec[0]
                dict_curr['solution'][objective_curr, it_restart + 1] = list_dict_strategies[numj]['f_best'][objective_curr]
                dict_curr['function'][it, :] = fun_vec
                dict_curr['sparsity'][it, :] = sparsity_vec
                if it == num_iter - 1:
                    strategy_type = j['strategy']
                    dict_curr['title'] = '_'.join(list(dict.fromkeys(constraint_set.dir_type))) + '_' + strategy_type
                    suptitle = suptitle + '_' + 'S' + str(numj) + '_' + j['strategy'][0:min([3, len(j['strategy'])])]
                    dict_curr['time'] = num_grad_vec
                    dict_curr['iteration_per_pass'] = constraint_set.num_blocks/(num_grad_vec[1] - num_grad_vec[0])
        for j, numj in enumerate(list_dict_strategies):
            print('strategy ' + str(j) + ': ' + str(list_times[j]))
        with open(filename, 'wb') as handle:
            pickle.dump(list_dict_plots, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        list_dict_plots = pickle.load(open(filename, 'rb'))
    list_dict_names = compute_plots_labels(list_dict_strategies)
    if global_opt == 0:
        for type_plot in ['function', 'sparsity']:
            x_label = dict_x_label[p_params.stopping_crit][type_plot]
            y_label = dict_y_label[type_plot]
            comparison_plotter_std(list_dict_plots, x_label, y_label, type_plot, objective,
                               p_params.stopping_crit, num_restarts, num_objectives, list_dict_names, xticks = xticks)
    else:
        for type_plot in ['solution']:
            x_label = dict_x_label[p_params.stopping_crit][type_plot]
            y_label = dict_y_label[type_plot]
            comparison_plotter_std(list_dict_plots, x_label, y_label, type_plot, objective,
                               p_params.stopping_crit, num_restarts, num_objectives, list_dict_names, xticks=xticks,
                                   precision=1)
