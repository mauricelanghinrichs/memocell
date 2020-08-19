

"""The selection module provides methods to run the statistical inference for a
set of models on given data (model selection and parameter estimation). Use the
top-level function `select_models`; `select_models` then calls required helper
functions (such as `net_estimation`) automatically."""

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

def select_models(models, data, mean_only=False,
                            nlive=1000, tolerance=0.01,
                            bound='multi', sample='unif',
                            parallel=True, processes=None):
    """Main function of the selection module for statistical inference of `models`
    `M` given `data` `D` (model selection and parameter estimation).
    Main results are model estimates, in terms of evidence values `p(D | M)`,
    and parameter estimates `p(θ | M, D)` for each model. The resulting output
    can be further processed to obtain model probabilities `p(M | D)` and model
    Bayes factors by the `compute_model_probabilities`
    and `compute_model_bayes_factors` functions, respectively (see there).

    Parameters
    ----------
    models : list of tuple
        A list of models to run the statistical inference with. A specific input
        format is required. Each tuple defines a model for the inference with
        `(model name, model topology, sampling setup)`, see example below.
    data : memo_py.Data.data
        A memo_py data object used in the statistcal inference.
    mean_only : bool, optional
        If the inference shall be based on the first moment (means) only,
        specify `mean_only=True`. If the inference shall be based on information
        from the first and second moments, specify `mean_only=False` (default).
    nlive : int, optional
        Number of live points used for the nested sampling; default is `1000`.
        Passed to `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_'s
        top-level `NestedSampler`; see there for more info.
    tolerance : float, optional
        Tolerance to define the stopping criterion for the nested sampling;
        default is `0.01`.
        Passed to `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_'s
        `dlogz` argument in the `run_nested` method; see there for more info.
    bound : str, optional
        Method used to approximately bound the prior for the nested sampling;
        default is `'multi'`.
        Passed to `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_'s
        top-level `NestedSampler`; see there for more info.
    sample : str, optional
        Method used to sample uniformly within the likelihood constraint for
        the nested sampling; default is `'unif'`.
        Passed to `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_'s
        top-level `NestedSampler`; see there for more info.
    parallel : bool, optional
        Run `select_models` in parallel based on Python's `multiprocessing`
        module with `parallel=True` (default); the number of parallel processes can be
        specified with the `processes` argument. Use `parallel=False`
        to run `select_models` sequentially.
    processes : None or int, optional
        If `parallel=True`, the number of parallel processes used for multiprocessing
        can be specified here. If `parallel=True` and `processes=None` (default)
        the available number of processes will be determined automatically.
        If `parallel=False`, the `processes` argument will be ignored.

    Returns
    -------
    est_res : list of memo_py.estimation.Estimation
        A list of memo_py estimation objects for the models.

    Examples
    --------
    >>> # given some memo_py data object, models are defined as follows
    >>> # (here a simple model with symmetric division of one cell type)
    >>> t2 = [{'start': 'X_t', 'end': 'X_t',
    >>>        'rate_symbol': 'l',
    >>>        'type': 'S -> S + S',
    >>>        'reaction_steps': 2}]
    >>> t5 = [{'start': 'X_t', 'end': 'X_t',
    >>>        'rate_symbol': 'l',
    >>>        'type': 'S -> S + S',
    >>>        'reaction_steps': 5}]
    >>> s = {'initial_values': {'X_t': 1.0},
    >>>      'theta_bounds': {'l': (0.0, 1.0)},
    >>>      'variables': {'X_t': ('X_t', )}}
    >>> models = [('model2', t2, s), ('model5', t5, s)]
    >>> # then the inference is started with
    >>> est_res = me.selection.select_models(models, data)
    """
    ### this is the top-level function of this script to handle the set of
    ### networks/models for parameter and evidence estimation;
    ### for each network (in a parallelised loop), net_estimation function is called

    # load information that is the same for all models
    # mcmc information is combined to a new dict
    d_data = data
    d_mean_only = mean_only
    d_mcmc_setup = {
        'nlive':    nlive,
        'tolerance':   tolerance,
        'bound':        bound,
        'sample':      sample
    }

    # load information of the set of models
    d_model_set = models

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
    if parallel:
        # read out number of processes (None if mp.cpu_count() should be used)
        num_processes = processes

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
    """Helper function to handle the parameter and evidence estimation of a
    single model as specified by `input_var`.

    `Note`: Please use the top-level `select_models` function for
    all inference tasks; if you wish to estimate a single model only, you can
    just apply `select_models` on a one-model list or use `estimate` from the
    memo_py estimation class.

    Parameters
    ----------
    input_var : tuple
        Internal tuple structure with various information passed
        over from `select_models` function.

    Returns
    -------
    est : memo_py.estimation.Estimation
        A memo_py estimation object.
    """
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


def compute_model_probabilities(estimation_instances, mprior=None):
    """Compute the posterior probability distribution `p(M | D)`
    (probabilities of the models `M` given the data `D`) for a list
    of estimated models.
    This function is a wrapper of the
    `compute_model_probabilities_from_log_evidences`
    function; see there for more info.

    Parameters
    ----------
    estimation_instances : list of memo_py.estimation.Estimation
        A list of memo_py estimation objects for the models;
        for example obtained by the `select_models` function.
    mprior : None or 1d numpy.ndarray, optional
        Array of prior model probabilities; if `mprior=None` (default) an uniform model
        prior will be used: `p(M)=1/n` where `n` is the total number of models.
        If a custom prior is specified,
        it has to have the same length `n` as `estimation_instances`.

    Returns
    -------
    probs : 1d numpy.ndarray
        Array of posterior model probabilities `p(M | D)`.

    Examples
    --------
    >>> # estimation_instances for example by est_res = me.select_models(...)
    >>> # with estimated log evidences = [4.1, 1.8, 4.4, -1.6]
    >>> # (see compute_model_probabilities_from_log_evidences for more examples)
    >>> me.selection.compute_model_probabilities_from_log_evidences(logevids)
    array([0.40758705, 0.04086421, 0.55018497, 0.00136377])
    """
    ### wrapper for compute_model_probabilities_from_log_evidences
    ### on estimation_instances from selection results; optional
    ### specification of model priors (otherwise (default) a uniform
    ### model prior is used)

    # read out log evidence list
    logevids = np.array([est.bay_est_log_evidence for est in estimation_instances])

    # compute model probabilities and return
    return compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)


def compute_model_probabilities_from_log_evidences(logevids, mprior=None):
    """Compute the posterior probability distribution `p(M | D)`
    (probabilities of the models `M` given the data `D`),
    assuming uniform model prior (`mprior=None`, default) or
    assuming a specified custom model prior `mprior`.
    Computation is based on the Bayes theorem
    :math:`p(M | D) = \\frac{p(D | M) \\cdot p(M)}{p(D)}`
    where `p(M)` is the model prior and `p(D | M)` are the evidence
    values obtained from the estimation.

    `Note`: While evidence values are model-instrinsic (do not depend on the
    considered set of models), posterior model probabilities depend on the context of
    the overall considered set of models. A 'high' model probability might not mean
    much, if all models are bad or if the model space is not exhaustive.

    Parameters
    ----------
    logevids : 1d numpy.ndarray
        Array of logarithmic evidence values of the models.
    mprior : None or 1d numpy.ndarray, optional
        Array of prior model probabilities; if `mprior=None` (default) an uniform model
        prior will be used: `p(M)=1/n` where `n` is the total number of models.
        If a custom prior is specified,
        it has to have the same length `n` as `logevids`.

    Returns
    -------
    probs : 1d numpy.ndarray
        Array of posterior model probabilities `p(M | D)`.

    Examples
    --------
    >>> # calculation of model probabilities; note that they sum to 1
    >>> logevids = np.array([4.1, 1.8, 4.4, -1.6])
    >>> probs = me.selection.compute_model_probabilities_from_log_evidences(logevids)
    >>> probs
    array([0.40758705, 0.04086421, 0.55018497, 0.00136377])
    >>> np.sum(probs)
    1.0

    >>> # calculation of model probabilities with an uniform model prior
    >>> # this is the default option (thus we get the same result)
    >>> mprior = np.array([0.25, 0.25, 0.25, 0.25])
    >>> me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
    array([0.40758705, 0.04086421, 0.55018497, 0.00136377])

    >>> # calculation of model probabilities with a non-uniform model prior
    >>> # (here, the first two models are given a bit more prior probability)
    >>> mprior = np.array([0.3, 0.3, 0.2, 0.2])
    >>> me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
    array([0.49940188, 0.05006945, 0.44941468, 0.00111399])
    """
    ### compute probability distribution p(M | D) (probabilities of the models M
    ### given the data D), assuming uniform model prior (mprior=None) or assuming
    ### a specified model prior; based on Bayes theorem
    ### p(M | D) = p(D | M) * p(M) / p(D), e.g. with uniform model prior
    ### p(M)=1/n (n being the number of models);
    ### NOTE: a model probability depends on the set
    ### of tested models, while a model evidence p(D | M) is independent of
    ### the set of other tested models

    # see Goodnotes (bayes_model_distr); an alternative calculation
    # can be based on the Bayes factors (with respect to the best model); an old
    # version can be seen below; the current version (also in bayes_model_distr)
    # uses the log-sum-exp trick to prevent overflow issue in logpdata calculation;
    # version can accept model-specific priors (not only general uniform prior)
    # (is actually quite similar to an alternative version based on the bayes factors)
    ### OLD
    # # log model prior (log of model number)
    # # assuming uniform model prior
    # logmprior = - np.log(logevids.shape[0])
    #
    # # calculate normalising factor p(D)
    # logpdata = np.log(np.sum(np.exp(logevids))) + logmprior
    #
    # # calculate model probabilities
    # probs = np.exp(logevids + logmprior - logpdata)
    ### OLD

    if mprior is None:
        # set a uniform model prior 1/n (with n the number of models, default)
        # in log-space, this is -log(n)
        logmprior = np.full(logevids.shape, -np.log(logevids.shape[0]))
    else:
        ### use the specific mprior if provided, convert to log-space
        # check if mprior sums to 1.0 (within default tolerances),
        # otherwise raise warning
        if not np.isclose(np.array([1.0]), np.sum(mprior))[0]:
            warnings.warn('Model prior probabilities do not sum to one.')
        # check if mprior has same shape as logevids
        if logevids.shape==mprior.shape:
            logmprior = np.log(mprior)
        else:
            raise ValueError('Shape mismatch between logevids and mprior.')

    # calculate aggregate of evidences and model prior in log-space (an array)
    # (this is log(p(D|M_i) * p(M_i)) for all models M_i)
    logevidmprior = logevids + logmprior

    # get maximal value to use log-sum-exp trick (a number)
    logmax = np.max(logevidmprior)

    # calculate normalising factor p(D) in log-space
    # using log-sum-exp trick (a number)
    # equivalent to: logpdata = log(sum(exp(logevidmprior)))
    logpdata = logmax + np.log(np.sum(np.exp(logevidmprior - logmax)))

    # calculate model probabilities (an array)
    probs = np.exp(logevidmprior - logpdata)

    # check if probabilities sum to 1.0 (within default tolerances),
    # otherwise raise warning
    if not np.isclose(np.array([1.0]), np.sum(probs))[0]:
        warnings.warn('Probabilities do not sum to one.')

    return probs


def compute_model_bayes_factors(estimation_instances):
    """Compute Bayes factors for a list of estimated models. This function is
    a wrapper of the `compute_model_bayes_factors_from_log_evidences`
    function; see there for more info.

    Parameters
    ----------
    estimation_instances : list of memo_py.estimation.Estimation
        A list of memo_py estimation objects for the models;
        for example obtained by the `select_models` function.

    Returns
    -------
    bayesf : 1d numpy.ndarray
        Array of Bayes factors of the estimated models.

    Examples
    --------
    >>> # estimation_instances for example by est_res = me.select_models(...)
    >>> # with estimated log evidences = [4.1, 1.8, 4.4, -1.6]
    >>> me.selection.compute_model_bayes_factors(est_res)
    array([  1.34985881,  13.46373804,   1.        , 403.42879349])
    """
    ### wrapper for compute_model_bayes_factors_from_log_evidences
    ### on estimation_instances from selection results

    # read out log evidence list
    logevids = np.array([est.bay_est_log_evidence for est in estimation_instances])

    # compute Bayes factors and return
    return compute_model_bayes_factors_from_log_evidences(logevids)


def compute_model_bayes_factors_from_log_evidences(logevids):
    """Compute Bayes factors from an array of logarithmic evidences. A Bayes
    factor `K` for a model `M` is computed by an evidence ratio as
    :math:`K = p(D | M_b) / p(D | M)` where `D` is the data and `M`:sub:`b`
    is the overall best model of the given evidences.
    Bayes factors can be used to classify models into different levels of support
    given by the data (e.g., see further info `here <https://en.wikipedia.org/wiki/Bayes_factor>`_).

    `Note`: While evidence values are model-instrinsic (do not depend on the
    considered set of models), a model Bayes factor depends on the context of
    the overall considered set of models. A 'good' Bayes factor might not mean
    much, if all models are bad or if the model space is not exhaustive.

    Parameters
    ----------
    logevids : 1d numpy.ndarray
        Array of logarithmic evidence values of the models.

    Returns
    -------
    bayesf : 1d numpy.ndarray
        Array of Bayes factors of the models.

    Examples
    --------
    >>> logevids = np.array([4.1, 1.8, 4.4, -1.6])
    >>> me.selection.compute_model_bayes_factors_from_log_evidences(logevids)
    array([  1.34985881,  13.46373804,   1.        , 403.42879349])
    """
    ### compute the Bayes factors K's with respect to the overall best model
    ### from log evidences following formula K = p(D | M1) / p(D | M2),
    ### where M1 is the best model, M2 a second model and D the data

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
