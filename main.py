import numpy as np
import random
import matplotlib.pyplot as plt
from line_profiler_pycharm import profile
import pandas as pd
from objectives import *
from basic_sets import *
from utilities import *
from algorithms import *
from main_functions import *

def prod_simplex_opt(row, run_algo = True):
    # run the product domain algorithm with a certain set of strategies for
    # quadratic objectives constrained to product of simplices
    # num_blocks: m (see the notation in the article)
    # size: n
    # epsilon: noise coefficient for the random matrix
    # sparsity: s (see (77))
    # df_strategies: dataframe with columns [dir_type, strategy]. Dir type refers to the direction used in the blocks,
    # strategy to the strategy used in the product domain (parallel, GS, randomized)
    # num_objectives: number of objectives to generate randomly
    # num_restarts: number of random restarts (for the multistart strategy) or number of cycles (for the baisin hopping
    # streategy (for each objective)
    # rs: random seed to determine the starting points
    # global_opt: set to 1 to run the baisin hopping strategy, any other value for the multistart
    # step: step parameter used in the baisin hopping strategy
    # xticks: interval in the restart/optimality gap for the baisin hopping strategy
    df_strategies = pd.read_excel(name_xls, sheet_name='strategies_p_simplex')
    df_inputs = pd.read_excel(name_xls, sheet_name='inputs_p_simplex')
    df_inputs.loc[0, :] = df_inputs.loc[row, :]
    size_blocks = df_inputs.loc[0, 'size']
    num_blocks = df_inputs.loc[0, 'num_blocks']
    list_dict_strategies = df_strategies.to_dict('records')
    objective = quadratic_obj(alpha=df_inputs.loc[0, 'alpha'], n=num_blocks, p=size_blocks,
                              epsilon=df_inputs.loc[0, 'epsilon'], sparsity=df_inputs.loc[0, 'sparsity'])
    constraint_set = product_simplex(num_blocks, size_blocks)
    # ALGORITHM PARAMETERS
    # stopping_crit: in [max_grad, max_time, max_block_updates, max_epochs]
    # Lmax: upper bound on Lipschitz constant estimates
    # bound: bound to be used in stopping criterion
    # gamma: coefficient in sufficient decrease condition to be used in the adaptive estimate of Lipschitz constant
    # verbosity: TRUE to print info during the algo
    # tr_fw: threshold to determine stationarity within the SSC
    params_curr = compute_params(df_inputs)
    main_comparison(objective, constraint_set, list_dict_strategies,
                    p_params=params_curr,
                    num_objectives=df_inputs.loc[0, 'num_objectives'], num_restarts = df_inputs.loc[0, 'num_restarts'],
                    run_algo=run_algo, global_opt=df_inputs.loc[0, 'global_opt'], step=df_inputs.loc[0, 'step'],
                    xticks=df_inputs.loc[0, 'xticks'])


name_sheet = 'inputs_p_simplex'
# set run_algo to False if you have already run the code and just need the plots
run_algo = True
name_xls = 'file_inputs.xlsx'

df_inputs = pd.read_excel(name_xls, sheet_name=name_sheet)
num_rows = len(df_inputs.index)
# run algo for each row of input in the excel file
for j in range(num_rows):
    prod_simplex_opt(j, run_algo)