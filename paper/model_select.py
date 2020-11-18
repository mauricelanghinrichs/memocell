
import memocell as me
import numpy as np
import itertools
import pickle

### list of models (just one here)
# def make_model(steps_d, steps_l):
#     name = 'net_' + str(steps_d) + '_' + str(steps_l)
#
#     topology = [
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd_ma', 'type': 'S -> E', 'reaction_steps': steps_d},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': steps_l}
#         ]
#
#     setup = {'initial_values': {'M_t': 1.0, 'A_t': 0.0}, 'theta_bounds': {'d_ma': (0.0, 0.15), 'la_a': (0.0, 0.15)}, 'variables': {'M_t': ('M_t', ), 'A_t': ('A_t', )}}
#
#     return (name, topology, setup)
#
# models = [make_model(steps_d, steps_l) for steps_d in range(1, 11) for steps_l in range(1, 11)]
# print(len(models))

# alternative list of models (minimal model, different steps)
# def make_net(n_d, n_l):
#     name = 'min_' + 'd_' + str(n_d) + '_l_' + str(n_l)
#     topology = [
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': n_d},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': n_l}
#         ]
#
#     net = me.Network(name)
#     net.structure(topology)
#     return net
#
# nets = [make_net(n_d, n_l) for n_d in range(1, 6) for n_l in range(1, 21)] # range(1, 6) and range(1, 21) # range(3, 8) # [50, 60, 70, 80]
# variables = [{'M_t': ('M_t', ), 'A_t': ('A_t', )}]*len(nets)
# initial_values = [{'M_t': 1.0, 'A_t': 0.0}]*len(nets)
# theta_bounds = [{'d': (0.0, 0.15), 'l': (0.0, 0.15)}]*len(nets)
# fit_mean_only = False
# print(len(nets))

### alternative list of models
# steps_list = [2, 4, 6, 8, 10, 14, 18] # [2, 8, 14, 20, 26]
# sparse_steps_list = list(itertools.combinations_with_replacement(steps_list, 3))
# models = list()
# for step_i, step_j, step_h in sparse_steps_list:
#         for step_l in steps_list:
#                 topology = [
#                         {'start': 'N_t', 'end': 'P1_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(step_i/2)},
#                         {'start': 'P1_t', 'end': 'A_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(step_i/2)},
#
#                         {'start': 'N_t', 'end': 'P2_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(step_j/2)},
#                         {'start': 'P2_t', 'end': 'A_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(step_j/2)},
#
#                         {'start': 'N_t', 'end': 'P3_t', 'rate_symbol': 'alpha_T', 'type': 'S -> E', 'reaction_steps': int(step_h/2)},
#                         {'start': 'P3_t', 'end': 'A_t', 'rate_symbol': 'alpha_T', 'type': 'S -> E', 'reaction_steps': int(step_h/2)},
#
#                         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'lambda', 'type': 'S -> S + S', 'reaction_steps': step_l}
#                         ]
#                 setting = {'initial_values': {'N_t': 1.0, 'A_t': 0.0, 'P1_t': 0.0, 'P2_t': 0.0, 'P3_t': 0.0}, 'variables': {'M_t': ('N_t', 'P1_t', 'P2_t', 'P3_t'), 'A_t': ('A_t', )}, 'theta_bounds': {'alpha_F': (0.0, 0.15), 'alpha_S': (0.0, 0.15), 'alpha_T': (0.0, 0.15), 'lambda': (0.0, 0.15)}}
#
#                 models.append([(f'net_alphaF{step_i}_alphaS{step_j}_alphaT{step_h}_lambda{step_l}'), (topology), (setting)])
# # print(models)
# print(len(models))

### alternative list of models
# steps_list = [10, 20, 40, 60] # [2, 8, 14, 20, 26]
# models = list()
# for step_l in steps_list:
#         topology = [
#                 {'start': 'N_t', 'end': 'P1_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(6/2)},
#                 {'start': 'P1_t', 'end': 'A_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(6/2)},
#
#                 {'start': 'N_t', 'end': 'P2_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(8/2)},
#                 {'start': 'P2_t', 'end': 'A_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(8/2)},
#
#                 {'start': 'N_t', 'end': 'P3_t', 'rate_symbol': 'alpha_T', 'type': 'S -> E', 'reaction_steps': int(14/2)},
#                 {'start': 'P3_t', 'end': 'A_t', 'rate_symbol': 'alpha_T', 'type': 'S -> E', 'reaction_steps': int(14/2)},
#
#                 {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'lambda', 'type': 'S -> S + S', 'reaction_steps': step_l}
#                 ]
#         setting = {'initial_values': {'N_t': 1.0, 'A_t': 0.0, 'P1_t': 0.0, 'P2_t': 0.0, 'P3_t': 0.0}, 'variables': {'M_t': ('N_t', 'P1_t', 'P2_t', 'P3_t'), 'A_t': ('A_t', )}, 'theta_bounds': {'alpha_F': (0.0, 0.15), 'alpha_S': (0.0, 0.15), 'alpha_T': (0.0, 0.15), 'lambda': (0.0, 0.15)}}
#
#         models.append([(f'net_alphaF{6}_alphaS{8}_alphaT{14}_lambda{step_l}'), (topology), (setting)])
# # print(models)
# print(len(models))

# ### alternative list of models
# def make_net(steps_d, steps_l):
#     name = 'net_' + str(steps_d) + '_' + str(steps_l)
#     topology = [
#         {'start': 'N_t', 'end': 'A_t', 'rate_symbol': 'd_na', 'type': 'S -> E', 'reaction_steps': steps_d},
#         {'start': 'N_t', 'end': 'I_t', 'rate_symbol': 'd_ni', 'type': 'S -> E', 'reaction_steps': steps_d},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': steps_l}
#         ]
#
#     net = me.Network(name)
#     net.structure(topology)
#     return net
#
# steps_list = [1, 2, 4, 6, 8, 10, 14, 18] # [2, 4, 6, 8, 10, 14, 18]
# nets = [make_net(steps_d, steps_l) for steps_d in steps_list for steps_l in steps_list]
# variables = [{'M_t': ('N_t', 'I_t'), 'A_t': ('A_t', )}]*len(nets)
# initial_values = [{'N_t': 1.0, 'I_t': 0.0, 'A_t': 0.0}]*len(nets)
# theta_bounds = [{'d_na': (0.0, 0.15), 'd_ni': (0.0, 0.15), 'la_a': (0.0, 0.15)}]*len(nets)
# print(len(nets))

### alternative list of models
# def make_model(steps_d, steps_i, steps_l):
#     name = 'net_' + str(steps_d) + '_' + str(steps_l)
#
#     topology = [
#         {'start': 'N_t', 'end': 'A_t', 'rate_symbol': 'd_na', 'type': 'S -> E', 'reaction_steps': steps_d},
#         {'start': 'N_t', 'end': 'I_t', 'rate_symbol': 'd_ni', 'type': 'S -> E', 'reaction_steps': steps_d},
#         {'start': 'I_t', 'end': 'A_t', 'rate_symbol': 'd_ia', 'type': 'S -> E', 'reaction_steps': steps_i},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': steps_l}
#         ]
#
#     setup = {'initial_values': {'N_t': 1.0, 'I_t': 0.0, 'A_t': 0.0}, 'theta_bounds': {'d_na': (0.0, 0.15), 'd_ni': (0.0, 0.15), 'la_a': (0.0, 0.15)}, 'variables': {'M_t': ('N_t', 'I_t'), 'A_t': ('A_t', )}}
#
#     return (name, topology, setup)
#
# steps_list = [2, 4, 6, 8, 10, 14, 18]
# models = [make_model(steps_d, steps_l) for steps_d in steps_list for steps_l in steps_list]
# print(len(models))

# ### alternative list of models (par2)
# steps_list = [2, 4, 6, 8, 10, 14, 18] # [2, 8, 14, 20, 26]
# sparse_steps_list = list(itertools.combinations_with_replacement(steps_list, 2))
# models = list()
# for step_i, step_j in sparse_steps_list:
#         for step_l in steps_list:
#                 topology = [
#                         {'start': 'N_t', 'end': 'P1_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(step_i/2)},
#                         {'start': 'P1_t', 'end': 'A_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(step_i/2)},
#
#                         {'start': 'N_t', 'end': 'P2_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(step_j/2)},
#                         {'start': 'P2_t', 'end': 'A_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(step_j/2)},
#
#                         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'lambda', 'type': 'S -> S + S', 'reaction_steps': step_l}
#                         ]
#                 setting = {'initial_values': {'N_t': 1.0, 'A_t': 0.0, 'P1_t': 0.0, 'P2_t': 0.0}, 'variables': {'M_t': ('N_t', 'P1_t', 'P2_t'), 'A_t': ('A_t', )}, 'theta_bounds': {'alpha_F': (0.0, 0.15), 'alpha_S': (0.0, 0.15), 'lambda': (0.0, 0.15)}}
#
#                 models.append([(f'net_alphaF{step_i}_alphaS{step_j}_lambda{step_l}'), (topology), (setting)])
# # print(models)
# print(len(models))

# ### 2 parallel activation (new multiedge version); fit_mean_only True and False
# def make_net(n_d1, n_d2, n_l1):
#     name = 'par2_d_' + str(n_d1) + '_' + str(n_d2) + '_l_' + str(n_l1)
#     topology = [
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd1', 'type': 'S -> E', 'reaction_steps': n_d1},
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': n_d2},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'l1', 'type': 'S -> S + S', 'reaction_steps': n_l1},
#         ]
#
#     net = me.Network(name)
#     net.structure(topology)
#     return net
#
# n_d_list = [1, 2, 4, 6, 8, 10, 14, 18]
# n_d_sparse = list(itertools.combinations_with_replacement(n_d_list, 2))
# n_l_list = n_d_list
# nets = [make_net(n_d1, n_d2, n_l1) for n_d1, n_d2 in n_d_sparse
#                                          for n_l1 in n_l_list]
# variables = [{'M_t': ('M_t', ), 'A_t': ('A_t', )}]*len(nets)
# initial_values = [{'M_t': 1.0, 'A_t': 0.0}]*len(nets)
# theta_bounds = [{'d1': (0.0, 0.15), 'd2': (0.0, 0.15), 'l1': (0.0, 0.15)}]*len(nets)
# fit_mean_only = True # True or False
# print(len(nets))

### 3 parallel activation (new multiedge version); fit_mean_only True and False
# def make_net(n_d1, n_d2, n_d3, n_l1):
#     name = 'par3_d_' + str(n_d1) + '_' + str(n_d2) + '_' + str(n_d3) + '_l_' + str(n_l1)
#     topology = [
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd1', 'type': 'S -> E', 'reaction_steps': n_d1},
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': n_d2},
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd3', 'type': 'S -> E', 'reaction_steps': n_d3},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'l1', 'type': 'S -> S + S', 'reaction_steps': n_l1},
#         ]
#
#     net = me.Network(name)
#     net.structure(topology)
#     return net
#
# n_d_list = [2, 4, 6, 8, 10, 14, 18] # [1, 2, 4, 6, 8, 10, 14, 18]
# n_d_sparse = list(itertools.combinations_with_replacement(n_d_list, 3))
# n_l_list = n_d_list
# nets = [make_net(n_d1, n_d2, n_d3, n_l1) for n_d1, n_d2, n_d3 in n_d_sparse
#                                          for n_l1 in n_l_list]
# variables = [{'M_t': ('M_t', ), 'A_t': ('A_t', )}]*len(nets)
# initial_values = [{'M_t': 1.0, 'A_t': 0.0}]*len(nets)
# theta_bounds = [{'d1': (0.0, 0.15), 'd2': (0.0, 0.15), 'd3': (0.0, 0.15), 'l1': (0.0, 0.15)}]*len(nets)
# fit_mean_only = False # True or False
# print(len(nets))

### models with 2 parallel activation and 2 parallel division
# def make_net(n_d1, n_d2, n_l1, n_l2):
#     name = 'net_d_' + str(n_d1) + '_' + str(n_d2) + '_l_' + str(n_l1) + '_' + str(n_l2)
#     topology = [
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd1', 'type': 'S -> E', 'reaction_steps': n_d1},
#         {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': n_d2},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'l1', 'type': 'S -> S + S', 'reaction_steps': n_l1},
#         {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'l2', 'type': 'S -> S + S', 'reaction_steps': n_l2}
#         ]
#
#     net = me.Network(name)
#     net.structure(topology)
#     return net
#
# n_d_list = [1, 2, 4, 6, 8, 10] # [2, 4, 6, 8, 10, 14, 18]
# n_d_sparse = list(itertools.combinations_with_replacement(n_d_list, 2))
# n_l_list = [1, 2, 4, 6, 8, 10, 14, 18] # [2, 4, 6, 8, 10, 14, 18]
# n_l_sparse = list(itertools.combinations_with_replacement(n_l_list, 2))
# nets = [make_net(n_d1, n_d2, n_l1, n_l2) for n_d1, n_d2 in n_d_sparse
#                                          for n_l1, n_l2 in n_l_sparse]
# variables = [{'M_t': ('M_t', ), 'A_t': ('A_t', )}]*len(nets)
# initial_values = [{'M_t': 1.0, 'A_t': 0.0}]*len(nets)
# theta_bounds = [{'d1': (0.0, 0.15), 'd2': (0.0, 0.15), 'l1': (0.0, 0.15), 'l2': (0.0, 0.15)}]*len(nets)
# print(len(nets))

### in silico par2 paper test
def make_net(steps_d4, steps_d2, steps_l):
    name = 'par2_d_' + str(steps_d4) + '_' + str(steps_d2) + '_l_' + str(steps_l)

    t = [
    {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': steps_d4},
    {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': steps_d2},
    {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': steps_l}
    ]

    net = me.Network(name)
    net.structure(t)

    return net

nets = [make_net(steps_d4, steps_d2, steps_l) for steps_d4 in range(1, 6)
                                                  for steps_d2 in range(1, 6)
                                                  for steps_l in range(1, 6)]
variables = [{'X_t': ('X_t', ), 'Y_t': ('Y_t', )}]*len(nets)
initial_values = [{'X_t': 1, 'Y_t': 0}]*len(nets)
theta_bounds = [{'d4': (0.0, 0.15), 'd2': (0.0, 0.15), 'l': (0.0, 0.15)}]*len(nets)
fit_mean_only = False
print(len(nets))

### load the data
# count_data_cd44_manual_counting_18_01_14_filtered_thin4
data = pickle.load(open('in_silico_files/in_silico_data_cd44_par2_paper_test_d4_d2_l3.pickle', 'rb'))
print(data.data_name)

### input for selection
# d = {
# # model set
# 'model_set': models,
#
# # data/model settings
# 'data': data,
# 'mean_only': False, # True or False
#
# # nested sampling settings
# 'nlive':                    1000, # 250 # 1000
# 'tolerance':                0.01, # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
# 'bound':                    'multi',
# 'sample':                   'unif'
# }

### computation, result is a list of Estimation class instances
# res = me.select_models(d)
res = me.select_models(nets, variables, initial_values,
                    theta_bounds, data,
                    fit_mean_only=fit_mean_only,
                    nlive=1000, tolerance=0.01)

### save estimation with pickle
with open('in_silico_files/estimation_in_silico_data_cd44_par2_paper_test_d4_d2_l3.pickle', 'wb') as file_: # in_silico_estimation
    pickle.dump(res, file_)
