
import memo_py as me
import numpy as np
import itertools
import pickle

### create model set
def def_model(n_dxy, n_dyz, n_lx, n_ly, n_lz):
    initial_values = {'X_t': 1, 'Y_t': 0, 'Z_t': 0}
    theta_bounds = {'dxy': (0.0, 1.0), 'dyz': (0.0, 1.0), 'lx': (0.0, 1.0), 'ly': (0.0, 1.0), 'lz': (0.0, 1.0)}
    variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', ), 'Z_t': ('Z_t', )}

    name = f'net{n_dxy}{n_dyz}{n_lx}{n_ly}{n_lz}'
    t = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'dxy', 'type': 'S -> E', 'reaction_steps': n_dxy},
        {'start': 'Y_t', 'end': 'Z_t', 'rate_symbol': 'dyz', 'type': 'S -> E', 'reaction_steps': n_dyz},
        {'start': 'X_t', 'end': 'X_t', 'rate_symbol': 'lx', 'type': 'S -> S + S', 'reaction_steps': n_lx},
        {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'ly', 'type': 'S -> S + S', 'reaction_steps': n_ly},
        {'start': 'Z_t', 'end': 'Z_t', 'rate_symbol': 'lz', 'type': 'S -> S + S', 'reaction_steps': n_lz},
        ]
    s = {'initial_values': initial_values,
         'theta_bounds': theta_bounds,
         'variables': variables}
    return (name, t, s)

# models = [def_model(4,2,5,1,3), def_model(3,2,5,1,3), def_model(5,2,5,1,3),
#              def_model(4,1,5,1,3), def_model(4,3,5,1,3),
#              def_model(4,1,4,1,3), def_model(4,3,6,1,3),
#              def_model(4,1,5,2,3),
#              def_model(4,1,5,1,2), def_model(4,3,5,1,4)]

models = [def_model(*p) for p in itertools.product([1,2,3,4,5], repeat=5)]
print(len(models))

### load the data
data = pickle.load(open('data_pathway_topology_n1000_t11.pickle', 'rb'))
print(data.data_name)

d = {
# model set
'model_set': models,

# data/model settings
'data': data,
'mean_only': False, # True or False

# nested sampling settings
'nlive':                    1000, # 250 # 1000
'tolerance':                0.01, # 0.1 # 0.05 # 0.01
'bound':                    'multi',
'sample':                   'unif'
}

### computation, result is a list of Estimation class instances
res = me.select_models(d)

### save estimation with pickle
with open('estimation_data_pathway_topology_n1000_t11_toptrue3125.pickle', 'wb') as file_: # in_silico_estimation
    pickle.dump(res, file_)
