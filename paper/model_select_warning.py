
import memo_py as me
import numpy as np

### create data from 5-steps model
t = [
    {'start': 'X_t', 'end': 'X_t',
     'rate_symbol': 'l',
     'type': 'S -> S + S', 'reaction_steps': 5}
    ]

net = me.Network('net_div_g5')
net.structure(t)

num_iter = 100
initial_values = {'X_t': 1}
theta_values = {'l': 0.22}
time_values = np.array([0.0, 10.0])
variables = {'X_t': ('X_t', )}

sim = me.Simulation(net)
res_list = list()

for __ in range(num_iter):
    res_list.append(sim.simulate('gillespie', initial_values, theta_values, time_values, variables)[1])

sims = np.array(res_list)

data = me.Data('data_test_select_models')
data.load(['X_t',], time_values, sims, bootstrap_samples=10000, basic_sigma=1/num_iter)

# overwrite with known values (from notebook ex_docs_tests)
data.data_mean = np.array([[[1.         , 3.73      ]],
                           [[0.01       , 0.15406761]]])
data.data_variance = np.array([[[0.         , 2.36070707]],
                               [[0.01       , 0.32208202]]])


t2 = [{'start': 'X_t', 'end': 'X_t',
     'rate_symbol': 'l',
     'type': 'S -> S + S',
     'reaction_steps': 2}]

t5 = [{'start': 'X_t', 'end': 'X_t',
     'rate_symbol': 'l',
     'type': 'S -> S + S',
     'reaction_steps': 5}]

t15 = [{'start': 'X_t', 'end': 'X_t',
     'rate_symbol': 'l',
     'type': 'S -> S + S',
     'reaction_steps': 15}]

s = {'initial_values': {'X_t': 1.0},
     'theta_bounds': {'l': (0.0, 1.0)},
     'variables': {'X_t': ('X_t', )}}

models = [('m2', t2, s), ('m5', t5, s), ('m15', t15, s)]
print(len(models))
print(models)

# est_res = me.selection.select_models(models, data, parallel=True) # False (usually here)
# print([est.bay_est_log_evidence for est in est_res])
#
# est_res = me.selection.select_models(models, data, parallel=False) # False (usually here)
# print([est.bay_est_log_evidence for est in est_res])

# models_check = [('m15', t15, s)]
# est_check = me.selection.select_models(models_check, data)



net = me.Network('net_check')
net.structure(t15)

initial_values = {'X_t': 1}
theta_values = {'l': 0.26}
time_values = np.linspace(0.0, 10.0, num=200, endpoint=True)
variables = {'X_t': ('X_t', )}

sim = me.Simulation(net)

res = sim.simulate('moments', initial_values, theta_values,
             time_values, variables, estimation_mode=True) # False
print(res)
print(sim.sim_moments.moment_eqs_template_str)

from numba import jit
from scipy.integrate import odeint

def moment_pass(sim):
    str_for_exec = sim.sim_moments.moment_eqs_template_str
    exec(str_for_exec)
    return eval('_moment_eqs_template')

moment_system = moment_pass(sim) # sim.sim_moments.moment_system
init = sim.sim_moments.moment_initial_values
time_arr = time_values
theta = np.array([theta_values['l']])
sol = odeint(moment_system, init, time_arr, args=(theta, ))
print(sol)
