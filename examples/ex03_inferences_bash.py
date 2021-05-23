
### a sample script to run memocell estimations
### in the terminal/bash/shell etc.
### run this via 'python ex03_inferences_bash.py'
### for explanations/info, see jupyter notebook ex03_inferences.ipynb

import memocell as me
import numpy as np
import pickle
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()

    # construct networks
    def make_net(n):
        net = me.Network(f'net_div_erl{n}')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                     'rate_symbol': 'l',
                     'type': 'S -> S + S',
                     'reaction_steps': n}])
        return net

    nets = [make_net(n) for n in [1, 5, 10, 15, 20, 25, 30]]
    variables = [{'X_t': ('X_t', )}]*len(nets)
    initial_values_types = ['synchronous']*len(nets)
    initial_values = [{('X_t',): 1.0,
                       ('X_t', 'X_t'): 0.0}
                     ]*len(nets)
    theta_bounds = [{'l': (0.0, 0.5)}]*len(nets)
    time_values = [np.linspace(0.0, 10.0, num=11)]*len(nets)
    print('nets', len(nets))

    # load data
    data_10 = pickle.load(open('data_cell_div_t10.pickle', 'rb'))
    print(data_10.data_name)

    # run estimation
    est_res = me.selection.select_models(nets, variables,
                            initial_values_types, initial_values,
                            theta_bounds, data_10, sim_mean_only=False,
                            time_values=time_values, parallel=True)

    # save estimation with pickle
    with open('me_est_run_data_10_div_nets_7.pickle', 'wb') as file_:
        pickle.dump(est_res, file_)

    # afterwards you can load and then analyse the estimation
    # results via (inside jupyter notebook / lab or wherever)
    # est_res = pickle.load(open('me_est_run_data_10_div_nets_7.pickle', 'rb'))
