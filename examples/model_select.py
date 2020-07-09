
import memo_py as me
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

## alternative list of models (minimal model, different steps)
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
# models = [make_model(steps_d, steps_l) for steps_d in range(3, 8) for steps_l in range(1, 21)] # range(1, 6) and range(1, 21) # [50, 60, 70, 80]
# print(len(models))

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
steps_list = [10, 20, 40, 60] # [2, 8, 14, 20, 26]
models = list()
for step_l in steps_list:
        topology = [
                {'start': 'N_t', 'end': 'P1_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(6/2)},
                {'start': 'P1_t', 'end': 'A_t', 'rate_symbol': 'alpha_F', 'type': 'S -> E', 'reaction_steps': int(6/2)},

                {'start': 'N_t', 'end': 'P2_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(8/2)},
                {'start': 'P2_t', 'end': 'A_t', 'rate_symbol': 'alpha_S', 'type': 'S -> E', 'reaction_steps': int(8/2)},

                {'start': 'N_t', 'end': 'P3_t', 'rate_symbol': 'alpha_T', 'type': 'S -> E', 'reaction_steps': int(14/2)},
                {'start': 'P3_t', 'end': 'A_t', 'rate_symbol': 'alpha_T', 'type': 'S -> E', 'reaction_steps': int(14/2)},

                {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'lambda', 'type': 'S -> S + S', 'reaction_steps': step_l}
                ]
        setting = {'initial_values': {'N_t': 1.0, 'A_t': 0.0, 'P1_t': 0.0, 'P2_t': 0.0, 'P3_t': 0.0}, 'variables': {'M_t': ('N_t', 'P1_t', 'P2_t', 'P3_t'), 'A_t': ('A_t', )}, 'theta_bounds': {'alpha_F': (0.0, 0.15), 'alpha_S': (0.0, 0.15), 'alpha_T': (0.0, 0.15), 'lambda': (0.0, 0.15)}}

        models.append([(f'net_alphaF{6}_alphaS{8}_alphaT{14}_lambda{step_l}'), (topology), (setting)])
# print(models)
print(len(models))

### alternative list of models
# def make_model(steps_d, steps_l):
#     name = 'net_' + str(steps_d) + '_' + str(steps_l)
#
#     topology = [
#         {'start': 'N_t', 'end': 'A_t', 'rate_symbol': 'd_na', 'type': 'S -> E', 'reaction_steps': steps_d},
#         {'start': 'N_t', 'end': 'I_t', 'rate_symbol': 'd_ni', 'type': 'S -> E', 'reaction_steps': steps_d},
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

### load the data
data = pickle.load(open('count_data_cd44_manual_counting_18_01_14_filtered_48h.pickle', 'rb'))
print(data.data_name)

### input for selection
d = {
# model set
'model_set': models,

# data/model settings
'data': data,
'mean_only': False, # True or False

# nested sampling settings
'nlive':                    1000, # 250 # 1000
'tolerance':                0.01, # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
'bound':                    'multi',
'sample':                   'unif'
}

### computation, result is a list of Estimation class instances
res = me.select_models(d)

### save estimation with pickle
with open('estimation_count_data_cd44_filtered_48h_par3_div.pickle', 'wb') as file_: # in_silico_estimation
    pickle.dump(res, file_)
