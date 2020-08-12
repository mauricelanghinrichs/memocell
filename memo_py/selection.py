

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
from tqdm.autonotebook import tqdm

import warnings

def select_models(input_dict, parallel={'do': True, 'num_processes': None}):
    """docstring for ."""
    ### this is the top-level function of this script to handle the set of
    ### networks/models for parameter and evidence estimation;
    ### for each network (in a parallelised loop), net_estimation function is called

    # load information that is the same for all models
    # mcmc information is combined to a new dict
    d_data = input_dict['data']
    d_mean_only = input_dict['mean_only']
    d_mcmc_setup = {
        'nlive':    input_dict['nlive'],
        'tolerance':   input_dict['tolerance'],
        'bound':        input_dict['bound'],
        'sample':      input_dict['sample']
    }

    # load information of the set of models
    d_model_set = input_dict['model_set']

    # # load progress bar for current environment (jupyter or terminal)
    # if input_dict['progress_bar_env']=='jupyter':
    #     tqdm_version = tqdm.notebook.tqdm_notebook
    # elif input_dict['progress_bar_env']=='terminal':
    #     tqdm_version = tqdm.std.tqdm


    # create input variable 'input_var' (in net_estimation fct) that is stored in
    # 'pool_inputs' for the parallelised loop over the networks
    pool_inputs = list()
    for i, model in enumerate(d_model_set):
        # load model information
        m_name = model[0]
        m_topology = model[1]
        m_setup = model[2]

        # add 'mean_only' information to m_setup
        m_setup['mean_only'] = d_mean_only

        # pass a model iteration count
        m_iter = i

        pool_inputs.append((m_name,
                            m_topology,
                            m_setup,
                            m_iter,
                            d_data,
                            d_mcmc_setup))


    # if __name__ == '__main__': # TODO: is this needed somewhere?
    # parallelised version
    if parallel['do']:
        # read out number of processes (None if mp.cpu_count() should be used)
        num_processes = parallel['num_processes']

        with Pool(processes=num_processes) as p:

            results = list(tqdm(p.imap(net_estimation, pool_inputs), total=len(pool_inputs)))

        # # for progress bars
        # freeze_support()
        #
        # # create a pool for multiprocessing to run the estimation for the models
        # # this automatically searches for the maximal possible computer cores to use
        # pool = Pool(processes=num_processes,
        #             # for progress bars
        #             initargs=(RLock(),), initializer=tqdm.set_lock)
        #
        # # in parallelised loop, run for each network (item in pool_inputs) the net_estimation funtion
        # # 'results' receives the original order
        # results = pool.map(net_estimation, pool_inputs)
        #
        # # for the correct spacing of progress bars
        # print('\n' * (len(pool_inputs) + 1))

    # unparallelised version
    # NOTE: turning off multiprocessing might facilitate debugging
    else:
        results = list()
        for input_var in tqdm(pool_inputs):
            results.append(net_estimation(input_var))

        # # for the correct spacing of progress bars
        # print('\n' * (len(pool_inputs) + 1))
    return results


def net_estimation(input_var):
    """docstring for ."""
    ### this function handles the parameter and evidence estimation of a
    ### single model/network as specified by input_var

    # read out input_var
    (m_name, # name of the network (as string)
    m_topology, # topology/structure of the network
    m_setup, # initial_values for nodes, theta_bounds for parameters, mean_only boolean
    m_iter, # integer i denoting the i-th model in the set of models
    d_data, # data that is tried to fit by the model
    d_mcmc_setup) = input_var # settings for Bayesian inference framework (Markov Chain Monte Carlo)

    # specify the model as an instance of the Network class
    net = Network(m_name)
    net.structure(m_topology)

    # conduct the estimation via the Estimation class
    est_name = 'est_' + m_name
    est = Estimation(est_name, net, d_data, est_iter=m_iter)
    est.estimate(m_setup, d_mcmc_setup)

    # reset the eval() function 'moment_system' to prevent pickling error
    # 'reset' is just a placeholder string to indicate the reset
    est.net_simulation.sim_moments.moment_system = 'reset'
    est.bay_nested_sampler = 'reset'

    # return the instance 'est' of the Estimation class
    # 'est' can be read out to obtain the estimation results
    return est


def compute_model_probabilities(estimation_instances):
    """docstring for ."""
    ### compute probability distribution p(M | D) (probabilities of the models M
    ### given the data D), assuming uniform model prior; based on Bayes theorem
    ### p(M | D) = p(D | M) * p(M) / p(D), with model prior p(M)=1/n (n being
    ### the number of models); NOTE: a model probability depends on the set
    ### of tested models, while a model evidence p(D | M) is independent of
    ### the set of other tested models

    # see Goodnotes (bayes_model_distr); an alternative calculation
    # can be based on the Bayes factors (with respect to the best model)

    # read out evidence list
    logevids = np.array([est.bay_est_log_evidence for est in estimation_instances])

    # log model prior (log of model number)
    # assuming uniform model prior
    logmprior = - np.log(logevids.shape[0])

    # calculate normalising factor p(D)
    logpdata = np.log(np.sum(np.exp(logevids))) + logmprior

    # calculate model probabilities
    probs = np.exp(logevids + logmprior - logpdata)

    # check if probabilities sum to 1.0 (within default tolerances),
    # otherwise raise warning
    if not np.isclose(np.array([1.0]), np.sum(probs))[0]:
        warnings.warn('Probabilities do not sum to one.')

    return probs


def compute_model_bayes_factors(estimation_instances):
    """docstring for ."""
    ### compute the Bayes factors K's with respect to the overall best model
    ### from estimation_instances following formula K = p(D | M1) / p(D | M2),
    ### where M1 is the best model, M2 a second model and D the data

    # read out evidence list
    logevids = np.array([est.bay_est_log_evidence for est in estimation_instances])

    # get best logevid
    logevidbest = np.max(logevids)

    # calculate bayes factors
    bayesf = np.exp(logevidbest - logevids)
    return bayesf


### for plotting routines
def _dots_w_bars_evidence(estimation_instances, settings):
    """Private plotting helper method."""

    y_arr_err = np.zeros((len(estimation_instances), 3))
    x_ticks = list()
    attributes = dict()

    for i, est_i in enumerate(estimation_instances):
        log_evid = est_i.bay_est_log_evidence
        log_evid_err = est_i.bay_est_log_evidence_error

        y_arr_err[i, :] = np.array([log_evid, log_evid_err, log_evid_err])

        est_setting = settings[est_i.est_name]
        attributes[i] = (est_setting['label'], est_setting['color'])
        x_ticks.append(est_setting['label'])

    return y_arr_err, x_ticks, attributes


def _dots_wo_bars_likelihood_max(estimation_instances, settings):
    """Private plotting helper method."""

    y_arr_err = np.zeros((len(estimation_instances), 3))
    x_ticks = list()
    attributes = dict()

    for i, est_i in enumerate(estimation_instances):
        log_l_max = est_i.bay_est_log_likelihood_max

        y_arr_err[i, :] = np.array([log_l_max, None, None])

        est_setting = settings[est_i.est_name]
        attributes[i] = (est_setting['label'], est_setting['color'])
        x_ticks.append(est_setting['label'])

    return y_arr_err, x_ticks, attributes


def _dots_wo_bars_bic(estimation_instances, settings):
    """Private plotting helper method."""

    y_arr_err = np.zeros((len(estimation_instances), 3))
    x_ticks = list()
    attributes = dict()

    for i, est_i in enumerate(estimation_instances):
        bic = est_i.bay_est_bayesian_information_criterion

        y_arr_err[i, :] = np.array([bic, None, None])

        est_setting = settings[est_i.est_name]
        attributes[i] = (est_setting['label'], est_setting['color'])
        x_ticks.append(est_setting['label'])

    return y_arr_err, x_ticks, attributes


def _dots_wo_bars_evidence_from_bic(estimation_instances, settings):
    """Private plotting helper method."""

    y_arr_err = np.zeros((len(estimation_instances), 3))
    x_ticks = list()
    attributes = dict()

    for i, est_i in enumerate(estimation_instances):
        log_evid_from_bic = est_i.bay_est_log_evidence_from_bic

        y_arr_err[i, :] = np.array([log_evid_from_bic, None, None])

        est_setting = settings[est_i.est_name]
        attributes[i] = (est_setting['label'], est_setting['color'])
        x_ticks.append(est_setting['label'])

    return y_arr_err, x_ticks, attributes
###
