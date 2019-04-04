

"""This script handles the parameter and evidence estimation of a given
set of networks/models. Use the top-level function 'select_models' to run
this script; 'select_models' then calls 'net_estimation' automatically."""

# TODO: user input validation?

# import modules of this packages
from .network import Network
from .data import Data
from .estimation import Estimation

# import python modules
import numpy as np
from multiprocessing import Pool


def select_models(input_dict, multiprocessing=True):
    """docstring for ."""
    ### this is the top-level function of this script to handle the set of
    ### networks/models for parameter and evidence estimation;
    ### for each network (in a parallelised loop), net_estimation function is called

    # load information that is the same for all models
    # mcmc information is combined to a new dict
    d_data = input_dict['data']
    d_mean_only = input_dict['mean_only']
    d_mcmc_setup = {
        'burn_in_steps':    input_dict['burn_in_steps'],
        'sampling_steps':   input_dict['sampling_steps'],
        'num_temps':        input_dict['num_temps'],
        'num_walkers':      input_dict['num_walkers']
    }

    # load information of the set of models
    d_model_set = input_dict['model_set']

    # create input variable 'input_var' (in net_estimation fct) that is stored in
    # 'pool_inputs' for the parallelised loop over the networks
    pool_inputs = list()
    for model in d_model_set:
        # load model information
        m_name = model[0]
        m_topology = model[1]
        m_setup = model[2]

        # add 'mean_only' information to m_setup
        m_setup['mean_only'] = d_mean_only

        pool_inputs.append((m_name,
                            m_topology,
                            m_setup,
                            d_data,
                            d_mcmc_setup))


    # if __name__ == '__main__': # TODO: is this needed somewhere?
    # parallelised version
    if multiprocessing:
        # create a pool for multiprocessing to run the estimation for the models
        # this automatically searches for the maximal possible computer cores to use
        pool = Pool()

        # in parallelised loop, run for each network (item in pool_inputs) the net_estimation funtion
        # 'results' receives the original order
        results = pool.map(net_estimation, pool_inputs)

    # unparallelised version
    # NOTE: turning off multiprocessing might facilitate debugging
    else:
        results = list()
        for input_var in pool_inputs:
            results.append(net_estimation(input_var))
    return results


def net_estimation(input_var):
    """docstring for ."""
    ### this function handles the parameter and evidence estimation of a
    ### single model/network as specified by input_var

    # read out input_var
    (m_name, # name of the network (as string)
    m_topology, # topology/structure of the network
    m_setup, # initial_values for nodes, theta_bounds for parameters, mean_only boolean
    d_data, # data that is tried to fit by the model
    d_mcmc_setup) = input_var # settings for Bayesian inference framework (Markov Chain Monte Carlo)

    # specify the model as an instance of the Network class
    net = Network(m_name)
    net.structure(m_topology)

    # conduct the estimation via the Estimation class
    est = Estimation(net, d_data)
    est.estimate(m_setup, d_mcmc_setup)

    # reset the eval() function 'moment_system' to prevent pickling error
    # 'reset' is just a placeholder string to indicate the reset
    est.net_simulation.sim_moments.moment_system = 'reset'

    # return the instance 'est' of the Estimation class
    # 'est' can be read out to obtain the estimation results
    print(f'{m_name} done')
    return est


def dots_w_bars_evidence(estimation_instances, settings):
    """docstring for ."""

    y_arr_err = np.zeros((len(estimation_instances), 3))
    x_ticks = list()
    attributes = dict()

    for i, est_i in enumerate(estimation_instances):
        log_evid = est_i.bay_est_log_evidence
        log_evid_err = est_i.bay_est_log_evidence_error

        y_arr_err[i, :] = np.array([log_evid, log_evid_err, log_evid_err])

        est_setting = settings[est_i.net.net_name]
        attributes[i] = (est_setting['label'], est_setting['color'])
        x_ticks.append(est_setting['label'])

    return y_arr_err, x_ticks, attributes
