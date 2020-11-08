
### package imports
import memo_py as me
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import scipy.stats as stats
from scipy import integrate
import pymc3

### color settings
# https://www.molecularecologist.com/2020/04/simple-tools-for-mastering-color-in-scientific-figures/
geodataviz = sns.color_palette(["#FF1F5B", "#00CD6C", "#009ADE", "#AF58BA", "#FFC61E", "#F28522", "#A0B1BA"])
sns.palplot(geodataviz)
plt.show()
# sns.palplot(sns.color_palette("Set2"))
# sns.palplot(sns.color_palette("viridis"))

# define colors
grey = geodataviz[-1]; sns.palplot(grey); plt.show()
cnaive = geodataviz[2]; sns.palplot(cnaive); plt.show()
cactiv = geodataviz[1]; sns.palplot(cactiv); plt.show()
cwact = geodataviz[0]; sns.palplot(cwact); plt.show() # 2, 3
cwdiv = geodataviz[4]; sns.palplot(cwdiv); plt.show()
ctopx = geodataviz[4]; sns.palplot(ctopx); plt.show()
ctopy = geodataviz[0]; sns.palplot(ctopy); plt.show()
ctopz = geodataviz[2]; sns.palplot(ctopz); plt.show()

### figure settings
# plt.rcParams.update({'figure.autolayout': True})
# plt.rcParams.update({'figure.figsize': (1.5, 1)}) # can be adapted for each plot
# plt.rcParams.update({'figure.figsize': (8, 5)})
plt.rcParams.update({'font.size': 6})
plt.rcParams['font.family'] = 'Helvetica Neue'
# plt.rcParams['font.weight'] = 'medium'

# temp disabled
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica Neue'
plt.rcParams['mathtext.it'] = 'Helvetica Neue:italic'

plt.rcParams['axes.linewidth'] = 0.3
plt.rcParams['xtick.major.width'] = 0.4
plt.rcParams['xtick.minor.width'] = 0.3
plt.rcParams['ytick.major.width'] = 0.4
plt.rcParams['ytick.minor.width'] = 0.3

plt.rcParams.update( {# 'legend.fontsize': 20,
          'legend.handlelength': 1.0/2.0})

### general functions
### waiting time functions
def waiting_time_distr_samples_act(x, res, samples, time_max, mprior):
    model_probs = me.selection.compute_model_probabilities(res, mprior=mprior)

    # for checking set a counter
    model_type_counts = np.zeros((4,))

    # set general x values
    y = np.zeros((x.shape[0], samples))

    for i in range(samples):
        # get random model according to p(M|D)
        model_rand = np.random.choice(range(len(model_probs)), p=model_probs, replace=True)
        est = res[model_rand]

        # decide whether phase type or markov/erlang model
        # par3 model
        if 'par3_' in est.net.net_name:
#         if 'alphaT' in est.net.net_name:
            model_type_counts[3] += 1
#             print('par3')
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # read out steps and rates
            n_d1 = est.net.net_modules[0]['module_steps']
            n_d2 = est.net.net_modules[1]['module_steps']
            n_d3 = est.net.net_modules[2]['module_steps']
            d1 = theta[0]
            d2 = theta[1]
            d3 = theta[2]

            # calculate channel probabilities
            d_total_paths = n_d1 * d1 + n_d2 * d2 + n_d3 * d3
            d1_path = (n_d1 * d1)/d_total_paths
            d2_path = (n_d2 * d2)/d_total_paths
            d3_path = (n_d3 * d3)/d_total_paths

            # calculate different gamma pdfs
            d1_scale = 1.0/(n_d1 * d1)
            d2_scale = 1.0/(n_d2 * d2)
            d3_scale = 1.0/(n_d3 * d3)

            d1_gamma = stats.gamma.pdf(x, a=n_d1, scale=d1_scale)
            d2_gamma = stats.gamma.pdf(x, a=n_d2, scale=d2_scale)
            d3_gamma = stats.gamma.pdf(x, a=n_d3, scale=d3_scale)

            # compose total gamma over channels
            gamma = d1_path * d1_gamma + d2_path * d2_gamma + d3_path * d3_gamma

#             # read out steps for the first half of the channels (by construction)
#             alpha_F_steps = est.net.net_modules[0]['module_steps']
#             alpha_S_steps = est.net.net_modules[2]['module_steps']
#             alpha_T_steps = est.net.net_modules[4]['module_steps']

#             # read out rates for the first half of the channels (by construction)
#             alpha_F_rate = theta[0]
#             alpha_S_rate = theta[1]
#             alpha_T_rate = theta[2]

#             # calculate channel probabilities
#             alpha_total_paths = alpha_F_steps * alpha_F_rate + alpha_S_steps * alpha_S_rate + alpha_T_steps * alpha_T_rate
#             alpha_F_path = (alpha_F_steps * alpha_F_rate)/alpha_total_paths
#             alpha_S_path = (alpha_S_steps * alpha_S_rate)/alpha_total_paths
#             alpha_T_path = (alpha_T_steps * alpha_T_rate)/alpha_total_paths

#             # calculate different gamma pdfs
#             alpha_F_shape = alpha_F_steps * 2
#             alpha_F_scale = 1.0/(0.5*alpha_F_rate*alpha_F_shape)

#             alpha_S_shape = alpha_S_steps * 2
#             alpha_S_scale = 1.0/(0.5*alpha_S_rate*alpha_S_shape)

#             alpha_T_shape = alpha_T_steps * 2
#             alpha_T_scale = 1.0/(0.5*alpha_T_rate*alpha_T_shape)

#             alpha_F_gamma = stats.gamma.pdf(x, a=alpha_F_shape, scale=alpha_F_scale)
#             alpha_S_gamma = stats.gamma.pdf(x, a=alpha_S_shape, scale=alpha_S_scale)
#             alpha_T_gamma = stats.gamma.pdf(x, a=alpha_T_shape, scale=alpha_T_scale)

#             # compose total gamma over channels
#             gamma = alpha_F_path * alpha_F_gamma + alpha_S_path * alpha_S_gamma + alpha_T_path * alpha_T_gamma
        # par2 model
        elif 'par2_' in est.net.net_name:
            model_type_counts[2] += 1
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # read out steps and rates
            n_d1 = est.net.net_modules[0]['module_steps']
            n_d2 = est.net.net_modules[1]['module_steps']
            d1 = theta[0]
            d2 = theta[1]

            # calculate channel probabilities
            d_total_paths = n_d1 * d1 + n_d2 * d2
            d1_path = (n_d1 * d1)/d_total_paths
            d2_path = (n_d2 * d2)/d_total_paths

            # calculate different gamma pdfs
            d1_scale = 1.0/(n_d1 * d1)
            d2_scale = 1.0/(n_d2 * d2)

            d1_gamma = stats.gamma.pdf(x, a=n_d1, scale=d1_scale)
            d2_gamma = stats.gamma.pdf(x, a=n_d2, scale=d2_scale)

            # compose total gamma over channels
            gamma = d1_path * d1_gamma + d2_path * d2_gamma

        # par1+i model (identified by d_ni symbolic rate)
        elif 'd_ni' in est.net.net_rates_identifier.values():
            model_type_counts[1] += 1
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # read out steps
            d_na_steps = est.net.net_modules[0]['module_steps']

            # read out rate
            d_na_rate = theta[0]

            # calculate gamma pdf
            shape = d_na_steps
            scale = 1.0/(d_na_rate*d_na_steps)

            gamma = stats.gamma.pdf(x, a=shape, scale=scale)

        # markov/erlang model
        else:
            model_type_counts[0] += 1
#             print('markov/erlang')
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # compute Gamma parameters
            ### activation reaction ###
            theta_shape = est.net.net_modules[0]['module_steps']
            theta_scale = 1.0/(theta[0]*theta_shape)

            gamma = stats.gamma.pdf(x, a=theta_shape, scale=theta_scale)
        y[:, i] = gamma
    print(model_type_counts)
    return y

def waiting_time_distr_samples_div(x, res, samples, time_max, mprior):
    model_probs = me.selection.compute_model_probabilities(res, mprior=mprior)

    # set general x values
    y = np.zeros((x.shape[0], samples))

    # for checking set a counter
    model_type_counts = np.zeros((4,))

    for i in range(samples):
        # get random model according to p(M|D)
        model_rand = np.random.choice(range(len(model_probs)), p=model_probs, replace=True)
        est = res[model_rand]

        # decide whether phase type or markov/erlang model
        # par 3 model
        if 'par3_' in est.net.net_name:
#         if 'alphaT' in est.net.net_name:
            model_type_counts[3] += 1
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # compute Gamma parameters
            ### division reaction ###
            theta_shape = est.net.net_modules[3]['module_steps']
            theta_scale = 1.0/(theta[3]*theta_shape)

            gamma = stats.gamma.pdf(x, a=theta_shape, scale=theta_scale)

        # par 2 model
        elif 'par2_' in est.net.net_name:
            model_type_counts[2] += 1
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # compute Gamma parameters
            ### division reaction ###
            theta_shape = est.net.net_modules[2]['module_steps']
            theta_scale = 1.0/(theta[2]*theta_shape)

            gamma = stats.gamma.pdf(x, a=theta_shape, scale=theta_scale)

        # par1+i model (identified by d_ni symbolic rate)
        elif 'd_ni' in est.net.net_rates_identifier.values():
            model_type_counts[1] += 1
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # compute Gamma parameters
            ### division reaction ###
            theta_shape = est.net.net_modules[2]['module_steps']
            theta_scale = 1.0/(theta[2]*theta_shape)

            gamma = stats.gamma.pdf(x, a=theta_shape, scale=theta_scale)

        # markov/erlang model
        else:
            model_type_counts[0] += 1
            # get random theta according to p(theta|M, D)
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]

            # compute Gamma parameters
            ### division reaction ###
            theta_shape = est.net.net_modules[1]['module_steps']
            theta_scale = 1.0/(theta[1]*theta_shape)

            gamma = stats.gamma.pdf(x, a=theta_shape, scale=theta_scale)
        y[:, i] = gamma

    print(model_type_counts)
    return y

### cell hist and event functions
def hist_cell_counts_samples(res, celltypeind, time_point, sample_n, count_max, sim_n, mprior):
    model_probs = me.selection.compute_model_probabilities(res, mprior=mprior)

    # set general x values
    y = np.zeros((count_max, sample_n))

    # for checking set a counter
    model_type_counts = np.zeros((4,))

    for i in range(sample_n):
        print(i)
        # get random model according to p(M|D)
        model_rand = np.random.choice(range(len(model_probs)), p=model_probs, replace=True)
        est = res[model_rand]

        # decide whether phase type or markov/erlang model
        # par 3 model
        if 'par3_' in est.net.net_name:
            model_type_counts[3] += 1
            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            d1, d2, d3, l = theta
            theta_values = {'d1': d1, 'd2': d2,
                                'd3': d3, 'l1': l}

            # sim settings for par3 model
            time_values = est.data.data_time_values
            initial_values = {'M_t': 1, 'A_t': 0}
            variables = {'M_t': ('M_t', ), 'A_t': ('A_t', )}

            sim = me.Simulation(est.net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

            sample = np.zeros((count_max))
            for j in range(y.shape[0]):
                sample[j] = np.sum(in_silico_counts[:, celltypeind, time_point]==j)

        # par 2 model
        elif 'par2_' in est.net.net_name:
            model_type_counts[2] += 1

            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            d1, d2, l1 = theta
            theta_values = {'d1': d1, 'd2': d2, 'l1': l1}

            # sim settings for par2 model
            time_values = est.data.data_time_values
            initial_values = {'M_t': 1, 'A_t': 0}
            variables = {'M_t': ('M_t', ), 'A_t': ('A_t', )}

            sim = me.Simulation(est.net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

            sample = np.zeros((count_max))
            for j in range(y.shape[0]):
                sample[j] = np.sum(in_silico_counts[:, celltypeind, time_point]==j)

        # par1+i model (identified by d_ni symbolic rate)
        elif 'd_ni' in est.net.net_rates_identifier.values():
            model_type_counts[1] += 1

            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            dna, dni, l = theta
            theta_values = {'d_na': dna, 'd_ni': dni, 'la_a': l}

            # sim settings for par3 model
            time_values = est.data.data_time_values
            initial_values = {'N_t': 1, 'I_t': 0, 'A_t': 0}
            variables = {'M_t': ('N_t', 'I_t'), 'A_t': ('A_t', )}

            # different memopy versions
            # (sim = me.Simulation(est.net)) usually works
            dna_steps = est.net.net_modules[0]['module_steps']
            dni_steps = est.net.net_modules[1]['module_steps']
            l_steps = est.net.net_modules[2]['module_steps']

            t = [
            {'start': 'N_t', 'end': 'A_t', 'rate_symbol': 'd_na', 'type': 'S -> E', 'reaction_steps': dna_steps},
            {'start': 'N_t', 'end': 'I_t', 'rate_symbol': 'd_ni', 'type': 'S -> E', 'reaction_steps': dni_steps},
            {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': l_steps}
            ]

            net = me.Network(est.net.net_name)
            net.structure(t)

            sim = me.Simulation(net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

            sample = np.zeros((count_max))
            for j in range(y.shape[0]):
                sample[j] = np.sum(in_silico_counts[:, celltypeind, time_point]==j)

        # markov/erlang model
        else:
            model_type_counts[0] += 1
            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            a1, l = theta
            theta_values = {'d_ma': a1, 'la_a': l}

            # sim settings for par3 model
            time_values = est.data.data_time_values
            initial_values = {'M_t': 1, 'A_t': 0}
            variables = {'M_t': ('M_t', ), 'A_t': ('A_t', )}

            # different memopy versions
            # (sim = me.Simulation(est.net)) usually works
            d_steps = est.net.net_modules[0]['module_steps']
            l_steps = est.net.net_modules[1]['module_steps']

            t = [
            {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd_ma', 'type': 'S -> E', 'reaction_steps': d_steps},
            {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': l_steps}
            ]

            net = me.Network(est.net.net_name)
            net.structure(t)

            sim = me.Simulation(net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

            sample = np.zeros((count_max))
            for j in range(y.shape[0]):
                sample[j] = np.sum(in_silico_counts[:, celltypeind, time_point]==j)

        y[:, i] = sample
    print(model_type_counts)
    return y

def event_percentage(events):
    return 100.0*sum([1.0 if e[0] else 0.0 for e in events])/len(events)

def event_model_samples(res, sample_n, sim_n, mprior):
    model_probs = me.selection.compute_model_probabilities(res, mprior=mprior)

    e_act_model_samples = np.zeros(sample_n)
    e_div1_model_samples = np.zeros(sample_n)
    e_div2_model_samples = np.zeros(sample_n)
    e_div3_model_samples = np.zeros(sample_n)

    time_values = res[0].data.data_time_values
    data_variables = ['A_t', 'M_t']

    # for checking set a counter
    model_type_counts = np.zeros((4,))

    for i in range(sample_n):
        print(i)
        # get random model according to p(M|D)
        model_rand = np.random.choice(range(len(model_probs)), p=model_probs, replace=True)
        est = res[model_rand]

        # decide whether phase type or markov/erlang model
        # par 3 model
        if 'par3_' in est.net.net_name:
            model_type_counts[3] += 1
            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            d1, d2, d3, l = theta
            theta_values = {'d1': d1, 'd2': d2,
                                'd3': d3, 'l1': l}

            # sim settings for par3 model
            time_values = est.data.data_time_values
            initial_values = {'M_t': 1, 'A_t': 0}
            variables = {'M_t': ('M_t', ), 'A_t': ('A_t', )}

            sim = me.Simulation(est.net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

        # par 2 model
        elif 'par2_' in est.net.net_name:
            model_type_counts[2] += 1

            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            d1, d2, l1 = theta
            theta_values = {'d1': d1, 'd2': d2, 'l1': l1}

            # sim settings for par2 model
            time_values = est.data.data_time_values
            initial_values = {'M_t': 1, 'A_t': 0}
            variables = {'M_t': ('M_t', ), 'A_t': ('A_t', )}

            sim = me.Simulation(est.net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

        # par1+i model (identified by d_ni symbolic rate)
        elif 'd_ni' in est.net.net_rates_identifier.values():
            model_type_counts[1] += 1

            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            dna, dni, l = theta
            theta_values = {'d_na': dna, 'd_ni': dni, 'la_a': l}

            # sim settings for par3 model
            time_values = est.data.data_time_values
            initial_values = {'N_t': 1, 'I_t': 0, 'A_t': 0}
            variables = {'M_t': ('N_t', 'I_t'), 'A_t': ('A_t', )}

            # different memopy versions
            # (sim = me.Simulation(est.net)) usually works
            dna_steps = est.net.net_modules[0]['module_steps']
            dni_steps = est.net.net_modules[1]['module_steps']
            l_steps = est.net.net_modules[2]['module_steps']

            t = [
            {'start': 'N_t', 'end': 'A_t', 'rate_symbol': 'd_na', 'type': 'S -> E', 'reaction_steps': dna_steps},
            {'start': 'N_t', 'end': 'I_t', 'rate_symbol': 'd_ni', 'type': 'S -> E', 'reaction_steps': dni_steps},
            {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': l_steps}
            ]

            net = me.Network(est.net.net_name)
            net.structure(t)

            sim = me.Simulation(net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

        # markov/erlang model
        else:
            model_type_counts[0] += 1
            # get random theta for model
            inds = np.array(range(0, est.bay_est_samples_weighted.shape[0]))
            theta_ind = np.random.choice(inds, replace=True)
            theta = est.bay_est_samples_weighted[theta_ind, :]
            a1, l = theta
            theta_values = {'d_ma': a1, 'la_a': l}

            # sim settings for par3 model
            time_values = est.data.data_time_values
            initial_values = {'M_t': 1, 'A_t': 0}
            variables = {'M_t': ('M_t', ), 'A_t': ('A_t', )}

            # different memopy versions
            # (sim = me.Simulation(est.net)) usually works
            d_steps = est.net.net_modules[0]['module_steps']
            l_steps = est.net.net_modules[1]['module_steps']

            t = [
            {'start': 'M_t', 'end': 'A_t', 'rate_symbol': 'd_ma', 'type': 'S -> E', 'reaction_steps': d_steps},
            {'start': 'A_t', 'end': 'A_t', 'rate_symbol': 'la_a', 'type': 'S -> S + S', 'reaction_steps': l_steps}
            ]

            net = me.Network(est.net.net_name)
            net.structure(t)

            sim = me.Simulation(net)
            res_list = list()

            for __ in range(sim_n):
                res_list.append(sim.simulate('gillespie', variables, initial_values, theta_values, time_values)[1])
            in_silico_counts = np.array(res_list)

        data_btstrp = me.Data('data_btstrp')
        data_btstrp.load(data_variables, time_values, in_silico_counts, bootstrap_samples=2)
        data_btstrp.events_find_all()

        e_act_model_samples[i] = event_percentage(data_btstrp.event_all_first_cell_type_conversion)
        e_div1_model_samples[i] = event_percentage(data_btstrp.event_all_first_cell_count_increase_after_cell_type_conversion)
        e_div2_model_samples[i] = event_percentage(data_btstrp.event_all_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion)
        e_div3_model_samples[i] = event_percentage(data_btstrp.event_all_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion)

    print(model_type_counts)
    return (e_act_model_samples, e_div1_model_samples,
            e_div2_model_samples, e_div3_model_samples)
