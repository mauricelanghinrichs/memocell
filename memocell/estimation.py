
"""
The estimation module contains the Estimation class for statistical inference
of models given data based on nested sampling.
"""

from .network import Network
from .data import Data
from .simulation import Simulation

from dynesty import NestedSampler
from dynesty import utils as dyfunc

import numpy as np
from tqdm import tqdm
import warnings

# TODO: user input validation?

class Estimation(object):
    """Class for statistical inference of a model given data.

    Main method is `estimate` which requires model and data input and then
    computes model and parameter estimates based on nested sampling.
    For the nested sampling the
    `dynesty <https://dynesty.readthedocs.io/en/latest/quickstart.html>`_
    package is used. For a typical use case,
    the `estimate` method is the only method to call. `estimate` is a wrapper method
    for the other class methods, so more documentation for the computations
    in the background and also how the class attributes are obtained can be found in
    the respective individual methods.

    Parameters
    ----------
    est_name : str
        A name for the estimation object.
    network : memocell.network.Network
        A memocell network object.
    data : memocell.data.Data
        A memocell data object.
    est_iter : None or int, optional
        Number to indicate iteration of a set of estimations; default is `None`.

    Returns
    -------
    est : memocell.estimation.Estimation
        Initialised memocell estimation object. Typically, continue with the
        `est.estimate` method to run the actual estimation.

    Examples
    --------
    >>> import memocell as me
    >>> import numpy as np
    >>> # initialise some data and a network
    >>> data = me.Data('my_data')
    >>> # data.load(...)
    >>> net = me.Network('my_net')
    >>> # net.structure(...)
    >>> # an estimation can then look like this
    >>> variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
    >>> init_val_type = 'synchronous'
    >>> initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
    >>>     ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
    >>> theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
    >>> est = me.Estimation('my_est', net, data)
    >>> est.estimate(variables, init_val_type, initial_values, theta_bounds)
    """

    def __init__(self, est_name, network, data, est_iter=None):

        # set the name of the estimation object and iteration number (optional)
        self.est_name = est_name
        self.est_iter = est_iter

        # validate network input (has to be instance of Network class) and instantiate
        self.net = self._validate_network_input(network)

        # validate data input (has to be instance of Data class) and instantiate
        # other objects for data input
        self.data = self._validate_data_input(data)
        self.data_time_values = None
        self.data_mean_values = None
        self.data_var_values = None
        self.data_cov_values = None
        self.data_num_values = None

        ### network related settings
        # instantiate object for rate parameter bounds (theta bounds)
        self.net_theta_bounds = None

        # instantiate object for initial values of network states (main nodes)
        self.net_initial_values = None

        # instantiate time values for fitting and index readouts
        self.net_time_values = None
        self.net_time_values_dense = None
        self.net_time_ind = None

        # set simulation type of the network to 'moments'
        self.net_simulation_type = 'moments'

        # instantiate object for an instance of the simulation class for the network
        self.net_simulation = None

        # instantiate objects for simulation and fit mean only modes
        # (first moment only or first and second moments computation)
        self.net_simulation_sim_mean_only = None
        self.net_simulation_fit_mean_only = None

        # instantiate object to store best-fit simulation
        # (best-fit computed by median parameter values of their 1d marginals)
        self.net_simulation_bestfit = None
        self.net_simulation_bestfit_exists = False

        # instantiate object to store best-fit simulation with credible band
        # (cred band best-fit computed by median over simulation posterior samples)
        self.net_simulation_credible_band = None
        self.net_simulation_credible_band_bestfit = None
        self.net_simulation_credible_band_exists = False
        ###

        ### bayesian (bay) inference related settings
        # instantiate object for the fixed log-likelihood norm
        self.bay_log_likelihood_norm = None

        # instantiate objects for Markov chain Monte Carlo Nested Sampling settings
        self.bay_nested_nlive = None
        self.bay_nested_tolerance = None
        self.bay_nested_bound = None
        self.bay_nested_ndims = None
        self.bay_nested_sample = None
        self.bay_nested_sampler = None
        self.bay_nested_sampler_res = None

        # instantiate objects to assign estimation results
        self.bay_est_samples = None
        self.bay_est_samples_weighted = None
        self.bay_est_weights = None
        self.bay_est_params_cred = None
        self.bay_est_params_median = None
        self.bay_est_params_log_likelihood_max = None

        # estimation results for model selection
        self.bay_est_log_evidence = None
        self.bay_est_log_evidence_error = None
        self.bay_est_log_likelihood_max = None
        self.bay_est_bayesian_information_criterion = None # short BIC
        self.bay_est_log_evidence_from_bic = None
        ###


        # ### TODO: adapt
        # # - change comments
        #
        # # information about the reactions
        # self.reac_modules = None
        # self.reac_ranges = None
        # self.reac_types = None
        #
        # # information for the prior
        # self.log_prior_supp = None
        #
        # # mcmc settings
        # self.mcmc_burn_in_steps = None
        # self.mcmc_sampling_steps = None
        # self.mcmc_num_temps = None
        # self.mcmc_num_walkers = None
        # self.mcmc_num_dim = None
        # self.mcmc_num_threads = None
        # self.mcmc_initial_params = None
        # self.mcmc_sampler = None
        # ###

    def estimate(self, variables, initial_values_type, initial_values,
                            theta_bounds, time_values=None,
                            sim_mean_only=False, fit_mean_only=False,
                            nlive=1000, tolerance=0.01,
                            bound='multi', sample='unif',):
        """Main method of the estimation class. This method computes model
        estimates and `θ` parameter estimates of a specified model `M` and given data
        `D` based on nested sampling; main results are the model estimate as logarithmic
        evidence value `p(D | M)` and the parameter posterior distribution
        `p(θ | M, D)`.

        These and other results of the estimation run can be accessed
        through various class attributes, automatically computed by the
        other methods of this class (see more info at the respective methods).
        For an estimation object `est`, the
        logarithmic model evidence can be accessed at `est.bay_est_log_evidence`
        and the estimated parameter posterior at `est.bay_est_samples_weighted`.
        Specific point estimates of the parameters are at `est.bay_est_params_median`
        and 95% credible intervals are at `est.bay_est_params_cred`.

        If multiple models shall be compared, please use the `select_models` function
        of the selection module, which is a wrapper of the `estimate` method for
        multiple models. The selection module also contains further model
        comparison measures such as the model posterior distribution `p(M | D)` and
        Bayes factors.

        Parameters
        ----------
        variables : dict
            Information for simulation variables with
            `key:value=simulation variable:tuple of network main nodes`
            dictionary pairs. The simulation variables have to correspond
            to the data variables. The tuple of network main nodes can be used to sum
            up multiple network nodes to one simulation variable.
        initial_values_type : str
            Initial value type to specify the multinomial distribution scheme
            of observable variables to the hidden variables (`'synchronous'` or
            `'uniform'`).
        initial_values : dict
            Initial values for the moments of network main nodes for the
            simulations during the estimation. Means are specified as
            `key:value=(node, ):initial value (float)`,
            variances are specified as `key:value=(node, node):initial value (float)`
            and covariances are specified as `key:value=(node1, node2):initial value (float)`
            dictionary pairs.
        theta_bounds : dict
            Uniform prior bounds of the parameters as
            `key:value=parameter:tuple of (lower bound, upper bound)` dictionary pairs.
        time_values : None or 1d numpy.ndarray, optional
            Time values to simulate the model with. If `None` (default), the time values
            of the data will be used (`data.data_time_values`). If specified, `time_values`
            has to contain at least all time values of the data, but can have more.
        sim_mean_only : bool, optional
            If the model simulations shall be computed for the first moment (means)
            only, specify `sim_mean_only=True`. If the model simulations shall be
            computed for the first and second moments,
            specify `sim_mean_only=False` (default). If `sim_mean_only=True`,
            `fit_mean_only` is overwritten with `True` in any case (when
            higher order moments are not computed, they cannot be fitted).
        fit_mean_only : bool, optional
            If the inference shall be based on the first moment (means) only,
            specify `fit_mean_only=True`. If the inference shall be
            based on information from the first and second moments,
            specify `fit_mean_only=False` (default). If `sim_mean_only=True`,
            `fit_mean_only` cannot be `False` and will be overwritten with `True`
            (when higher order moments are not computed, they cannot be fitted).
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

        Returns
        -------
        None

        Examples
        --------
        >>> # net is a memocell network object
        >>> # data is a memocell data object
        >>> # with this, an estimation can look like this
        >>> variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        >>> init_val_type = 'synchronous'
        >>> initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
        >>>     ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        >>> theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        >>> est = me.Estimation('my_est', net, data)
        >>> est.estimate(variables, init_val_type, initial_values, theta_bounds)
        """

        # # for progress bar
        # total_sampling_steps = 1
        # with tqdm(total=total_sampling_steps, desc='{est: <{width}}'.format(est=self.est_name, width=16), position=self.est_iter+1) as pbar:

        # initialise estimation
        # (set up network, simulation and sampling properties)
        self.initialise_estimation(variables, initial_values_type, initial_values,
                                    theta_bounds, time_values, sim_mean_only, fit_mean_only,
                                    nlive, tolerance, bound, sample)
        # self.initialise_estimation(network_setup, mcmc_setup)

        # execute the sampling for the estimation of parameters and model evidence
        self.run_estimation()

        # print(f"""results:\n
        # \t theta confidence: {self.bay_est_params_conf}\n
        # \t theta medians: {self.bay_est_params_median}\n
        # \t log evidence: {self.bay_est_log_evidence}\n
        # \t log evidence error: {self.bay_est_log_evidence_error}""")

        # # update progress bar (there is just not done or done (0/1 or 1/1 steps))
        # pbar.update(1)


    def initialise_estimation(self, variables, initial_values_type, initial_values,
                                theta_bounds, time_values, sim_mean_only, fit_mean_only,
                                nlive, tolerance, bound, sample):
        """Initialise and prepare an estimation.

        Helper function for the `estimate` method, arguments are passed over from
        there (see there for more info).
        """
        ### initialise network related settings
        # validate theta bounds user input and assign to numpy array object
        self._validate_theta_bounds_input(self.net.net_rates_identifier, theta_bounds)
        self.net_theta_bounds = self.initialise_net_theta_bounds(self.net.net_theta_symbolic, self.net.net_rates_identifier, theta_bounds)

        # validate initial values user input and assign to self
        # (further processing is done in the called simulation class methods)
        self._validate_initial_values_input(self.net_simulation_type,
                                    initial_values_type, initial_values)
        self.net_initial_values = initial_values

        # set the sim (/moments) and fit mean only modes (True or False)
        # we cannot fit higher moments if they are not computed, thus whenever
        # sim_mean_only==True we overwrite fit_mean_only=True in any case
        self.net_simulation_sim_mean_only = sim_mean_only
        self.net_simulation_fit_mean_only = True if sim_mean_only else fit_mean_only

        ### initialise the simulation for the network
        # generate an instance of the Simulation class
        self.net_simulation = Simulation(self.net)

        # prepare the simulation in 'moments' type
        self.net_simulation.prepare_simulation(self.net_simulation_type, variables,
                                    initial_values_type, None,
                                    self.net_initial_values, sim_mean_only)

        # the initial values for the moments have to be computed explicitly
        self.net_simulation.sim_moments.moment_initial_values = self.net_simulation.sim_moments.process_initial_values(
                                        self.net_initial_values,
                                        initial_values_type)

        ### initialise time values (fitting and data)
        # read out time values of the data
        self.data_time_values = self.data.data_time_values

        # validate user input for time_values (None or array)
        # and if array, check if it contains all data time values
        self._validate_time_values_input(time_values, self.data_time_values)

        # initialise fit time values and time indices for model readout
        (self.net_time_values,
        self.net_time_values_dense,
        self.net_time_ind) = self.initialise_time_values(time_values, self.data_time_values)

        ### initialise data
        # check mapping between simulation variables and data variables
        self._validate_simulation_and_data_variables_mapping(self.net_simulation.sim_variables,
                                                        self.data.data_variables)

        (self.data_mean_values,
        self.data_var_values,
        self.data_cov_values) = self.match_data_to_network(self.net_simulation.sim_variables_order,
                                                        self.net_simulation.sim_variables_identifier,
                                                        self.data.data_mean,
                                                        self.data.data_variance,
                                                        self.data.data_covariance,
                                                        self.data.data_mean_order,
                                                        self.data.data_variance_order,
                                                        self.data.data_covariance_order)

        # depending on fit mean only mode, get the number of summary data points
        # for data, the fit_mean_only mode is relevant as it determines
        # the actual data the models have contact with
        if self.net_simulation_fit_mean_only:
            self.data_num_values = self.data.data_num_values_mean_only
        else:
            self.data_num_values = self.data.data_num_values

        ### initialise bayesian inference related settings
        # # computation of the log prior value for theta's that are on the support of the prior
        # self.bay_log_prior_supp = self.compute_bayes_log_prior_value_on_support(self.net_theta_bounds)

        # compute the model-independent term of the log_likelihood
        # (similary here, net_simulation_fit_mean_only is used)
        self.bay_log_likelihood_norm = self.compute_log_likelihood_norm(
                                    self.data_mean_values,
                                    self.data_var_values,
                                    self.data_cov_values,
                                    self.net_simulation_fit_mean_only)

        # self.bay_mcmc_burn_in_steps = mcmc_setup['burn_in_steps'] # sampling steps per walker
        # self.bay_mcmc_sampling_steps = mcmc_setup['sampling_steps'] # sampling steps per walker
        # self.bay_mcmc_num_temps = mcmc_setup['num_temps'] # number of temperatures, e.g. 5, 10 or 20
        # self.bay_mcmc_num_walkers = mcmc_setup['num_walkers'] # number of walkers, e.g. 100 or 200
        self.bay_nested_nlive = nlive
        self.bay_nested_tolerance = tolerance
        self.bay_nested_bound = bound
        self.bay_nested_sample = sample
        self.bay_nested_ndims = len(self.net.net_theta_symbolic) # number of dimension for estimation (= number of rate parameters (theta))

        # define the sampler used for the estimation of parameters and model evidence
        self.bay_nested_sampler = NestedSampler(self.log_likelihood, self.prior_transform,
                                                self.bay_nested_ndims, bound=self.bay_nested_bound,
                                                sample=self.bay_nested_sample, nlive=self.bay_nested_nlive,
                                    logl_args=((self.net_simulation.sim_moments.moment_initial_values,
                                                self.net_time_values, self.net_time_ind,
                                                self.data_mean_values, self.data_var_values,
                                                self.data_cov_values)))


    def run_estimation(self):
        """Run the estimation based on nested sampling.

        Helper function for the `estimate` method, which handles the initialisation
        of the estimation before running it (see there for more info).
        """

        # run the dynesty sampler
        # NOTE: the very first iteration can take a while since preparations
        # for the moment calculations have to be done
        self.bay_nested_sampler.run_nested(dlogz=self.bay_nested_tolerance,
                                            print_progress=False) # dynesty progress bar

        # get sampler result
        self.bay_nested_sampler_res = self.bay_nested_sampler.results

        # obtain posterior parameter samples from reweighting
        self.bay_est_samples, self.bay_est_samples_weighted, self.bay_est_weights = self.get_posterior_samples(self.bay_nested_sampler_res)

        # set last theta values as maximal log likelihood parameters
        self.bay_est_params_log_likelihood_max = self.bay_est_samples[-1, :]

        # assess median and credible bounds for parameters
        self.bay_est_params_cred = self.get_credible_interval(self.bay_est_samples_weighted)
        self.bay_est_params_median = np.array([self.bay_est_params_cred[i][0] for i in range(len(self.bay_est_params_cred))])

        # obtain log evidence values with associated error
        self.bay_est_log_evidence, self.bay_est_log_evidence_error = self.get_model_evidence(self.bay_nested_sampler_res)

        # compute alternative measures for model selection
        self.bay_est_log_likelihood_max = self.get_maximal_log_likelihood(self.bay_nested_sampler_res)
        self.bay_est_bayesian_information_criterion = self.compute_bayesian_information_criterion(
                                                                    self.data_num_values,
                                                                    self.bay_nested_ndims,
                                                                    self.bay_est_log_likelihood_max)
        self.bay_est_log_evidence_from_bic = self.compute_log_evidence_from_bic(self.bay_est_bayesian_information_criterion)


    @staticmethod
    def get_posterior_samples(sampler_result):
        """Obtain samples for the parameters `θ` according to the estimated posterior
        `p(θ | M, D)`, where `M` is the corresponding model and `D` is the data. Parameter
        samples from nested sampling have to be weighted to represent the posterior, so
        either use the weighted samples directly (`samples_weighted`) or use the original
        samples and the weights to weight them manually (`samples` and `weights`).

        `Note`: After running a memocell estimation there is no need to run this
        method, one can simply access the output at `est.bay_est_samples`,
        `est.bay_est_samples_weighted` and `est.bay_est_weights`
        for the estimation instance `est`.

        Parameters
        ----------
        sampler_result : dynesty.results.Results
            Nested sampling result of a memocell estimation. Typically available at
            `est.bay_nested_sampler_res`.

        Returns
        -------
        samples : numpy.ndarray
            Unweighted parameter samples from nested sampling with shape (`number of samples`,
            `number of parameters`). Typically available at `est.bay_est_samples`.
        samples_weighted : numpy.ndarray
            Weighted parameter posterior samples from nested sampling with shape (`number of samples`,
            `number of parameters`). Typically available at `est.bay_est_samples_weighted`.
        weights : numpy.ndarray
            Weights to weight unweighted samples to get posterior samples with shape
            (`number of samples`,). Typically available at `est.bay_est_weights`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.get_posterior_samples(est.bay_nested_sampler_res)
        (array([[0.14791189, 0.14857705],
            [0.12864247, 0.14920066],
            [0.13767801, 0.14860021],
            ...,
            [0.02801861, 0.07502377],
            [0.02801369, 0.07502103],
            [0.02802205, 0.07499796]]),
        array([[0.02426309, 0.07119228],
            [0.03200628, 0.06646134],
            [0.03220339, 0.06802025],
            ...,
            [0.02802894, 0.07496195],
            [0.02800815, 0.07502692],
            [0.02801369, 0.07502103]]),
        array([0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
            9.93638169e-06, 9.93651290e-06, 9.93662331e-06]))
        """

        # see methods text or goodnotes
        # theta samples from nested sampling are reweighted (not just
        # taking as uniform) to get the "true" posterior samples
        # a given sample theta_i has to obey p(theta_i)=(Li deltaXi)/Z=:wi
        # where the evidence Z=sum over all Li deltaXi
        # thus the code below follows:
        # with logwt_i=log(Li deltaXi), logZ[-1]=log(Z)
        samples = sampler_result.samples
        weights = np.exp(sampler_result.logwt - sampler_result.logz[-1])
        samples_weighted = dyfunc.resample_equal(samples, weights)
        return samples, samples_weighted, weights


    @staticmethod
    def get_credible_interval(samples):
        """Get the median value and a 95% credible interval for each parameter based on
        an estimated posterior distribution given by `samples`.
        Median is obtained as 50-th percentile and the interval bounds are obtained by
        2.5-th and 97.5-th percentiles, respectively.

        `Note`: After running a memocell estimation there is no need to run this
        method, one can simply access the medians and credible intervals for the parameters
        with `est.bay_est_params_cred` for the estimation instance `est`.

        Parameters
        ----------
        samples : numpy.ndarray
            Parameter posterior samples with shape (`number of samples`,
            `number of parameters`). A typical choice are the weighted parameter
            samples from nested sampling at `est.bay_est_samples_weighted`.

        Returns
        -------
        params_cred : tuple of tuple
            Median values and 95% credible interval for each parameter (with
            the inner tuple order `median`, `2.5-th perc`, `97.5-th perc`).
            Typically available at `est.bay_est_params_cred`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> samples = est.bay_est_samples_weighted
        >>> samples.shape
        (12962, 2)
        >>> est.get_credible_interval(samples)
        ((0.0280347396878894, 0.02594988652911552, 0.030144084514338445),
        (0.07470536719462752, 0.06919784056074933, 0.07955644971380103))
        >>> est.bay_est_params_cred
        ((0.0280347396878894, 0.02594988652911552, 0.030144084514338445),
        (0.07470536719462752, 0.06919784056074933, 0.07955644971380103))

        >>> # minimal example on concrete values
        >>> samples = np.array([[0.2, 3.4], [0.4, 3.2], [0.25, 3.65]])
        >>> samples.shape
        (3, 2)
        >>> est.get_credible_interval(samples)
        ((0.25, 0.2025, 0.3925), (3.4, 3.21, 3.6375))
        """

        # the 2.5th, 50th and 97.5th percentiles of parameter distributions are extracted
        # params_cred then contains the tuple (median (50th), lower bound (2.5th), upper bound (97.5th))
        # to provide a 95%-credible interval
        params_cred = tuple(map(lambda v: (v[1], v[0], v[2]),
                            zip(*np.percentile(samples, [2.5, 50, 97.5], axis=0))))
        return params_cred


    @staticmethod
    def get_model_evidence(sampler_result):
        """Obtain logarithmic evidence value and its error estimate from the
        nested sampling result.

        `Note`: After running a memocell estimation there is no need to run this
        method, one can simply access the logarithmic model evidence and its error
        with `est.bay_est_log_evidence` and `est.bay_est_log_evidence_error` for the
        estimation instance `est`.

        Parameters
        ----------
        sampler_result : dynesty.results.Results
            Nested sampling result of a memocell estimation. Typically available at
            `est.bay_nested_sampler_res`.

        Returns
        -------
        log_evid_dynesty : float
            Logarithmic evidence of the estimated model. Typically available at
            `est.bay_est_log_evidence`.
        log_evid_err_dynesty : float
            Error of the logarithmic evidence of the estimated model. Typically
            available at `est.bay_est_log_evidence_error`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.get_model_evidence(est.bay_nested_sampler_res)
        (28.139812540432732, 0.11225503808864087)
        >>> est.bay_est_log_evidence
        28.139812540432732
        >>> est.bay_est_log_evidence_error
        0.11225503808864087
        """

        # value of log evidence (logZ) (last entry of nested sampling results)
        log_evid_dynesty = sampler_result.logz[-1]

        # estimate of the statistical uncertainty on logZ
        log_evid_err_dynesty = sampler_result.logzerr[-1]

        return log_evid_dynesty, log_evid_err_dynesty


    @staticmethod
    def get_maximal_log_likelihood(sampler_result):
        """Obtain the maximal logarithmic likelihood value from the nested
        sampling result.

        `Note`: After running a memocell estimation there is no need to run this
        method, one can simply access the maximal log-likelihood
        with `est.bay_est_log_likelihood_max` for the estimation instance `est`.

        Parameters
        ----------
        sampler_result : dynesty.results.Results
            Nested sampling result of a memocell estimation. Typically available at
            `est.bay_nested_sampler_res`.

        Returns
        -------
        logl_max : float
            Maximal logarithmic likelihood value of the estimated model.
            Typically available at `est.bay_est_log_likelihood_max`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.get_maximal_log_likelihood(est.bay_nested_sampler_res)
        35.48531419345989
        >>> est.bay_est_log_likelihood_max
        35.48531419345989
        """

        # get the value of the maximal log likelihood as last entry of nested sampling results
        return sampler_result.logl[-1]


    @staticmethod
    def compute_bayesian_information_criterion(num_data, num_params, log_likelihood_max):
        """Compute the Bayesian information criterion (BIC). Calculation is based
        on :math:`\\mathrm{BIC} = k \\cdot \\mathrm{ln}(n) - 2 \\, \\mathrm{ln}(L_{max})`
        where :math:`k` is the number of parameters (`num_params`), :math:`n` is the number of
        data points (`num_data`) and :math:`\\mathrm{ln}(L_{max})` is the maximal
        log-likelihood value (`log_likelihood_max`).

        `Note`: After running a memocell estimation there is no need to run this
        method, one can simply access the BIC with `est.bay_est_bayesian_information_criterion`
        for the estimation instance `est`.

        Parameters
        ----------
        num_data : int or float
            Number of data points. Typically available at `est.data_num_values`.
        num_params : int or float
            Number of estimated parameters. Typically available at `est.bay_nested_ndims`.
        log_likelihood_max : float
            Maximal logarithmic likelihood value of the estimated model.
            Typically available at `est.bay_est_log_likelihood_max`.


        Returns
        -------
        bic : float
            Bayesian information criterion of the estimated model.
            Typically available at `est.bay_est_bayesian_information_criterion`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.compute_bayesian_information_criterion(
        >>>         est.data_num_values,
        >>>         est.bay_nested_ndims,
        >>>         est.bay_est_log_likelihood_max)
        -65.55452798471536
        >>> est.bay_est_bayesian_information_criterion
        -65.55452798471536
        >>> est.compute_bayesian_information_criterion(15, 2, -35.49)
        -65.56389959779558
        """

        # the BIC (bayesian_information_criterion) is defined as
        # BIC = ln(n) k - 2 ln(Lmax)
        # with n being the number of data points, k the number of estimated
        # parameters, Lmax the maximal likelihood and ln() the natural logarithm
        return np.log(num_data) * num_params - 2.0 * log_likelihood_max


    @staticmethod
    def compute_log_evidence_from_bic(bic):
        """Under certain assumptions one can approximate the logarithmic evidence
        value with :math:`\\mathrm{ln}(p(D | M)) \\approx -\\frac{1}{2} \\mathrm{BIC}` where
        :math:`M` is the model, :math:`D` is the data and :math:`\\mathrm{BIC}` is
        the Bayesian information criterion, see
        `BIC (wiki) <https://en.wikipedia.org/wiki/Bayesian_information_criterion>`_.

        `Note`: This calculation is more a consistency check and can be accessed
        with `est.bay_est_log_evidence_from_bic` after a memocell estimation for `est`.
        The more accurate value of the logarithmic evidence from the nested sampling
        should be preferred for serious tasks (at `est.bay_est_log_evidence`).

        Parameters
        ----------
        bic : float
            Bayesian information criterion of the estimated model.
            Typically available at `est.bay_est_bayesian_information_criterion`.


        Returns
        -------
        log_evidence_from_bic : float
            Logarithmic evidence of the estimated model, approximated from the BIC.
            Typically available at `est.log_evidence_from_bic`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.compute_log_evidence_from_bic(est.bay_est_bayesian_information_criterion)
        32.77726399235768
        >>> est.bay_est_log_evidence_from_bic
        32.77726399235768
        >>> # compare with the more accurate log evid from nested sampling
        >>> est.bay_est_log_evidence
        28.139812540432732
        """

        # under certain assumptions the log evidence might be approximated from
        # the BIC (bayesian_information_criterion) via evidence ≈ exp(-BIC / 2)
        return - 0.5 * bic


    def prior_transform(self, theta_unit):
        """Transform parameter values :math:`\\theta` from the unit hypercube form
        (as used in the nested sampling) to the original prior space.

        For uniform parameter priors (as generally used) this transformation is
        achieved with the respective lower and upper parameter bounds
        :math:`[b_l, b_u]` as
        :math:`\\theta_{\\mathrm{orig}} = \\theta_{\\mathrm{unit}} (b_u - b_l) + b_l`.
        Parameter bounds can be accessed with `est.net_theta_bounds`.

        Parameters
        ----------
        theta_unit : 1d numpy.ndarray
            Values for parameters :math:`\\theta` in unit hypercube space.

        Returns
        -------
        theta_orig : 1d numpy.ndarray
            Values for parameters :math:`\\theta` in original prior space.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.net_theta_bounds
        array([[0.  , 0.15],
               [0.  , 0.15]])
        >>> theta_unit = np.array([0.2, 0.5])
        >>> est.prior_transform(theta_unit)
        array([0.03 , 0.075])
        """

        # we receive theta here in the unit hypercube form
        # and have to transform it back into the true parametrisation

        # since we use uniform priors we have to do in principle:
        # theta_true = theta_unit * (upper_bound-lower_bound) + lower_bound

        # if the lower_bound is zero, we would simply have:
        # theta_true = theta_unit * upper_bound
        return theta_unit * (self.net_theta_bounds[:, 1] - self.net_theta_bounds[:, 0]) + self.net_theta_bounds[:, 0]


    # def log_prior(self, theta):
    #     """docstring for ."""
    #     # st = time.time()
    #
    #     # log_prior is based on a uniform prior distribution with finite support
    #     # on_support is a boolean; True if all parameters/theta's are on the support (prior > 0) else False (prior = 0)
    #     on_support = np.all(( self.net_theta_bounds[:, 0] <= theta ) & ( theta <= self.net_theta_bounds[:, 1] ))
    #
    #     # log_prior returns its log value > -infinity (if on_support) or -infinity (if not on_support)
    #     if on_support:
    #         # et = time.time()
    #         # print('log_prior (ms)', (et - st)*1000)
    #         return self.bay_log_prior_supp
    #     else:
    #         # et = time.time()
    #         # print('log_prior (ms)', (et - st)*1000)
    #         return -np.inf

    def log_likelihood(self, theta_values, moment_initial_values,
                                        time_values, time_ind,
                                        mean_data, var_data, cov_data):
        """Compute the logarithmic likelihood :math:`\\mathrm{ln}(\\mathcal{L(\\theta)}) =
        \\mathrm{ln}(p(D | \\theta, M))` for parameter values :math:`\\theta` of a given
        model :math:`M` and given data :math:`D`. This method is used in the nested
        sampling.

        The computation is based on the following formula. Under the assumption
        of :math:`r` independent and normally distributed errors, the likelihood
        function is given by :math:`\\mathcal{L(\\theta)} = p(D | \\theta, M) =
        \\prod_{i=1}^{r} f_{\\mu_i, \\sigma_i}(x_i)`, where

        - :math:`D = (x_1,\\,..., x_r)` are the data points,

        - :math:`\\Sigma = (\\sigma_1,\\,..., \\sigma_r)` are the data standard errors,

        - :math:`M_{\\theta} = (\\mu_1,\\,..., \\mu_r)` are the model evaluations and

        - :math:`f_{\\mu, \\sigma}(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}\\,\\mathrm{exp}\\big(-\\frac{1}{2} \\big( \\frac{x-\\mu}{\\sigma} \\big)^2\\big)` is the normal density.

        The log-likelihood is then given by :math:`\\mathrm{ln}(\\mathcal{L(\\theta)})
        =-\\tfrac{1}{2} \\sum_{i=1}^{r} \\big( \\frac{x_i - \\mu_i}{\\sigma_i}
        \\big)^2 \\,+\\, \\eta`, where :math:`\\eta` is the model-independent
        normalisation term :math:`\\eta` computed as :math:`\\eta = -\\tfrac{1}{2}
        \\sum_{i=1}^{r} \\mathrm{ln}(2 \\pi \\sigma_{i}^{2})`; also see at the
        `compute_log_likelihood_norm` method.

        Parameters
        ----------
        theta_values : 1d numpy.ndarray
            Values for parameters :math:`\\theta` in the model order (according to `net.net_theta_symbolic`
            via `net.net_rates_identifier`); passed to a moment simulation method.
        moment_initial_values : dict
            Initial values for all moments of the hidden network layer;
            passed to a moment simulation method. Typically available at
            `est.net_simulation.sim_moments.moment_initial_values`; order of the
            moments corresponds to
            `est.net_simulation.sim_moments.moment_order_hidden`.
        time_values : 1d numpy.ndarray
            Time values for which model simulations are solved;
            passed to a moment simulation method. After estimation
            initialisation available at `est.net_time_values`.
        time_ind : slice or tuple of int
            Indexing information to read out model simulations at the time points
            of the data to allow comparison. After estimation
            initialisation available at `est.net_time_ind`.
        mean_data : numpy.ndarray
            Data mean statistics and standard errors with shape
            (2, `number of means`, `number of time points`) that have been matched
            to the model order. `mean_data[0, :, :]` contains the statistics;
            `mean_data[1, :, :]` contains the standard errors. After estimation
            initialisation available at `est.data_mean_values`.
        var_data : numpy.ndarray
            Data variance statistics and standard
            errors with shape (2, `number of variances`, `number of time points`)
            that have been matched to the model order. `var_data[0, :, :]` contains
            the statistics; `var_data[1, :, :]` contains the standard errors.
            After estimation initialisation available at `est.data_var_values`.
        cov_data : numpy.ndarray
            Data covariance statistics and standard
            errors with shape (2, `number of covariances`, `number of time points`)
            that have been matched to the model order. `cov_data[0, :, :]` contains
            the statistics; `cov_data[1, :, :]` contains the standard errors.
            After estimation initialisation available at `est.data_cov_values`.

        Returns
        -------
        logl : numpy.float64
            Computed value of the logarithmic likelihood.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> theta_values = np.array([0.03, 0.07])
        >>> est.log_likelihood(theta_values,
        >>>            est.net_simulation.sim_moments.moment_initial_values,
        >>>            est.net_time_values, est.net_time_ind,
        >>>            est.data_mean_values, est.data_var_values,
        >>>            est.data_cov_values)
        32.823084036435795
        """

        # NOTE: in the bayesian framework employed here, the likelihood is the
        # probability of the data, given a model (structure) and model parameters;
        # the log_likelihood is theoretically based on iid normally distributed error values
        # (data = model + error); thus, effectively, the log_likelihood depends
        # on the squared differences between data and model weighted by
        # measurement uncertainties (see chi's below)

        # mean, variance (if specified), covariance (if specified) of the model
        # are generated by the simulation class by a moment-based approach
        mean_m, var_m, cov_m  = self.net_simulation.sim_moments.run_moment_ode_system(
                                            moment_initial_values,
                                            time_values, theta_values)

        # compute the value of the log_likelihood
        if self.net_simulation_fit_mean_only:
            # when only mean values are fitted (first moments only)
            chi_mean = np.sum( ((mean_data[0, :, :] - mean_m[:, time_ind])/(mean_data[1, :, :]))**2 )
            chi_var = 0.0
            chi_cov = 0.0
        else:
            # when first (mean) and second moments (i.e., variance and covariance) are fitted
            chi_mean = np.sum( ((mean_data[0, :, :] - mean_m[:, time_ind])/(mean_data[1, :, :]))**2 )
            chi_var = np.sum( ((var_data[0, :, :] - var_m[:, time_ind])/(var_data[1, :, :]))**2 )
            chi_cov = np.sum( ((cov_data[0, :, :] - cov_m[:, time_ind])/(cov_data[1, :, :]))**2 )

        return -0.5 * (chi_mean + chi_var + chi_cov) + self.bay_log_likelihood_norm


    @staticmethod
    def compute_log_likelihood_norm(mean_data, var_data, cov_data, fit_mean_only):
        """Compute the model-independent normalisation term of the logarithmic
        likelihood. This value can be computed once and then used for all
        subsequent evaluations of the log-likelihood.

        With data points :math:`D = (x_1,\\,..., x_r)` and data standard errors
        :math:`\\Sigma = (\\sigma_1,\\,..., \\sigma_r)` the normalisation term
        :math:`\\eta` is computed as :math:`\\eta = -\\tfrac{1}{2} \\sum_{i=1}^{r}
        \\mathrm{ln}(2 \\pi \\sigma_{i}^{2})`; also see more info at
        the `log_likelihood` method.

        `Note`: This method will be automatically called during estimation
        initialisation (`est.initialise_estimation`).

        Parameters
        ----------
        mean_data : numpy.ndarray
            Data mean statistics and standard errors with shape
            (2, `number of means`, `number of time points`) that have been matched
            to the model order. `mean_data[0, :, :]` contains the statistics;
            `mean_data[1, :, :]` contains the standard errors. After estimation
            initialisation available at `est.data_mean_values`.
        var_data : numpy.ndarray
            Data variance statistics and standard
            errors with shape (2, `number of variances`, `number of time points`)
            that have been matched to the model order. `var_data[0, :, :]` contains
            the statistics; `var_data[1, :, :]` contains the standard errors.
            After estimation initialisation available at `est.data_var_values`.
        cov_data : numpy.ndarray
            Data covariance statistics and standard
            errors with shape (2, `number of covariances`, `number of time points`)
            that have been matched to the model order. `cov_data[0, :, :]` contains
            the statistics; `cov_data[1, :, :]` contains the standard errors.
            After estimation initialisation available at `est.data_cov_values`.
        fit_mean_only : bool
            Calculate the normalisation for an estimation in `fit_mean_only=False`
            or `fit_mean_only=True` mode.

        Returns
        -------
        norm : numpy.float64
            Model-independent normalisation term of the logarithmic
            likelihood. Typically available at `est.bay_log_likelihood_norm`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.compute_log_likelihood_norm(est.data_mean_values,
        >>>                         est.data_var_values,
        >>>                         est.data_cov_values,
        >>>                         False)
        37.04057852140377
        >>> est.bay_log_likelihood_norm
        37.04057852140377

        >>> # example with concrete values
        >>> mean_data = np.array([[[1., 0.67, 0.37],
        >>>                        [0., 0.45, 1.74]],
        >>>                       [[0.01, 0.0469473, 0.04838822],
        >>>                        [0.01, 0.07188642, 0.1995514]]])
        >>> var_data = np.array([[[0., 0.22333333, 0.23545455],
        >>>                       [0., 0.51262626, 4.03272727]],
        >>>                      [[0.01, 0.01631605, 0.01293869],
        >>>                       [0.01, 0.08878719, 0.68612036]]])
        >>> cov_data = np.array([[[ 0., -0.30454545, -0.65030303]],
        >>>                      [[ 0.01, 0.0303608, 0.06645246]]])
        >>> est.compute_log_likelihood_norm(mean_data, var_data, cov_data, False)
        37.04057852140377
        >>> est.compute_log_likelihood_norm(mean_data, var_data, cov_data, True)
        14.028288976285737
        """

        # compute the model-independent term of the log_likelihood
        # this is a fixed value that can be computed once over the data standard errors
        if fit_mean_only:
            norm_mean = np.sum( np.log(2 * np.pi * (mean_data[1, :, :]**2)) )
            norm_var = 0.0
            norm_cov = 0.0
        else:
            norm_mean = np.sum( np.log(2 * np.pi * (mean_data[1, :, :]**2)) )
            norm_var = np.sum( np.log(2 * np.pi * (var_data[1, :, :]**2)) )
            norm_cov = np.sum( np.log(2 * np.pi * (cov_data[1, :, :]**2)) )

        return -0.5 * (norm_mean + norm_var + norm_cov)


    # @staticmethod
    # def compute_bayes_log_prior_value_on_support(net_theta_bounds):
    #     """docstring for ."""
    #
    #     # compute the normalisation constant for the uniform prior
    #     # it is the volume given by the product of interval ranges of the theta's
    #     prior_norm = np.prod(np.diff(net_theta_bounds, axis=1))
    #
    #     # compute the value of the log prior on the support
    #     bayes_log_prior_supp = np.log(1.0/prior_norm)
    #     return bayes_log_prior_supp

    # @staticmethod
    # def generate_bayes_mcmc_initial_theta(num_temps, num_walkers, num_theta, theta_bounds):
    #     """docstring for ."""
    #
    #     # preallocate an array for initial values for theta
    #     # (a value is required for each theta_i, each walker, each temperature)
    #     mcmc_initial_params = np.zeros((num_temps, num_walkers, num_theta))
    #
    #     # for each theta_i / parameter we sample uniformly within its respective bounds
    #     # (on a linear scale)
    #     # this ensures that the initial parameters start on the support of the prior
    #     # (meaning prior(theta) > 0 and log(prior(theta))>-inf)
    #     for i in range(num_theta):
    #         mcmc_initial_params[:, :, i] = np.random.uniform(low=theta_bounds[i, 0], high=theta_bounds[i, 1], size=(num_temps, num_walkers))
    #     return mcmc_initial_params

    @staticmethod
    def initialise_time_values(time_values, data_time_values):
        """Obtain time values that can be used for model simulations
        (`net_time_values`).

        If `time_values=None`, the `data_time_values` are used to obtain
        `net_time_values`. If `time_values` are specified explicitly as array,
        `net_time_values` is referenced to them. This method also returns a
        dense version (`net_time_values_dense`) and tuple of indices to read
        out model simulations for time points of the data.

        Before this method is used the input should be checked with
        `_validate_time_values_input`.

        Parameters
        ----------
        time_values : None or 1d numpy.ndarray
            Information for time values for model simulations. If `None`,
            `net_time_values` will be referenced to data time values
            (`data.data_time_values`). If specified as array,
            `net_time_values=time_values`. Note in this case that
            `time_values` has to contain at least all time
            values of the data, but can have more.
        data_time_values : 1d numpy.ndarray
            Time values of the data. Typically at `data.data_time_values` of a
            memocell data object `data`.

        Returns
        -------
        net_time_values : 1d numpy.ndarray
            Time values for the model simulations.
        net_time_values_dense : 1d numpy.ndarray
            A dense version of `net_time_values` with the same minimal and
            maximal values but 1000 equally spaced total values.
        net_time_ind : slice or tuple of int
            Index information that can be used for numpy array indexing to read
            out model simulations at the data time points (`data_time_values`).

        Examples
        --------
        >>> import memocell as me
        >>> time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> data_time_values = np.array([1.0, 3.0, 5.0])
        >>> net_time_values, __, net_time_ind = me.Estimation.initialise_time_values(time_values, data_time_values)
        >>> net_time_values
        np.array([0., 1., 2., 3., 4., 5., 6.])
        >>> net_time_ind
        (1, 3, 5)
        """
        # NOTE that time_values and data_time_values were already checked
        # at this stage by _validate_time_values_input method

        # if time_values are not given explicitly (default), use data time values
        # in this case, we have to take the complete slice later for simulation readout
        if time_values is None:
            net_time_values = data_time_values
            # this is identical to ':', i.e. a[:, :]==a[:, slice(None)]
            # and faster than tuple indixing for all elements
            net_time_ind = slice(None)

        # if time_values is not None but equal to data_time_values,
        # we catch this case to set net_time_ind = slice(None)
        # (for performance issues since tuple of all elements is slower)
        elif np.array_equal(time_values, data_time_values):
            net_time_values = time_values
            net_time_ind = slice(None)

        # if times_values is not None and not equal, they are specified as array
        else:
            net_time_values = time_values

            # get a tuple of indices (net_time_ind) that can be used
            # to access a simulation at the time points where we have data
            net_time_where = [list(np.where(net_time_values==val)[0])
                                            for val in data_time_values]
            # flatten net_time_where and convert list to tuple
            net_time_ind = tuple([item for sublist in net_time_where
                                            for item in sublist])

        # in any case, we define a dense version of net_time_values
        net_time_values_dense = np.linspace(np.min(net_time_values),
                                            np.max(net_time_values),
                                            endpoint=True, num=1000)

        return (net_time_values, net_time_values_dense, net_time_ind)

    @staticmethod
    def initialise_net_theta_bounds(theta_symbolic, theta_identifier, theta_bounds):
        """Initialise uniform prior bounds of parameters :math:`\\theta`.

        `Note`: This method will be automatically called during estimation
        initialisation (`est.initialise_estimation`).

        Parameters
        ----------
        theta_symbolic : list of str
            List of `theta` identifiers. Each identifier represents a
            parameter. Typically available at `est.net.net_theta_symbolic`.
        theta_identifier : dict
            Map between parameter and their theta identifiers with
            `key:value=theta identifier:parameter` pairs.
            Typically available at `est.net.net_rates_identifier`.
        theta_bounds : numpy.ndarray
            Uniform prior bounds for the parameters with
            `key:value=parameter:tuple of bounds` pairs.

        Returns
        -------
        net_theta_bounds : numpy.ndarray
            Bounds of the uniform parameter prior with shape (`number of parameters`, 2).
            Typically available at `est.net_theta_bounds`. Lower bounds are at
            `net_theta_bounds[:, 0]` and upper bounds are at `net_theta_bounds[:, 1]`.

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.net.net_theta_symbolic
        ['theta_0', 'theta_1']
        >>> est.net.net_rates_identifier
        {'theta_0': 'd', 'theta_1': 'l'}
        >>> theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        >>> est.initialise_net_theta_bounds(est.net.net_theta_symbolic,
        >>>                         est.net.net_rates_identifier, theta_bounds)
        array([[0.  , 0.15],
               [0.  , 0.15]])
        """

        # define the parameter intervals (by lower and upper bounds) for the uniform prior
        # theta symbolic list also defines the order of the theta's in net_theta_bounds
        net_theta_bounds = np.zeros((len(theta_symbolic), 2))

        for i, theta in enumerate(theta_symbolic):
            net_theta_bounds[i, :] = np.array(theta_bounds[theta_identifier[theta]])
        return net_theta_bounds

    @staticmethod
    def match_data_to_network(                          sim_variables_order,
                                                        sim_variables_identifier,
                                                        data_mean,
                                                        data_var,
                                                        data_cov,
                                                        data_mean_order,
                                                        data_variance_order,
                                                        data_covariance_order):
        """Simulation and data variables have to be one-to-one/bijectively
        mappable in general. This method then sorts the data in a way that is
        given by the order of the simulation variables to simplify model/data
        comparison in the estimation. Sorted/ordered data arrays are typically
        available at `est.data_mean_values`, `est.data_var_values` and
        `est.data_cov_values`.

        `Note`: This method will be automatically called during estimation
        initialisation (`est.initialise_estimation`).

        Parameters
        ----------
        sim_variables_order : list of list
            Simulation variable order in terms of the variable identifiers;
            first element contains the first moment order (means), the
            second element the second moments order (variances and covariances).
        sim_variables_identifier : dict
            Information of mapping between simulation variables
            and their identifiers.
        data_mean : numpy.ndarray
            Dynamic data mean statistics and standard
            errors with shape (2, `len(data_mean_order)`, `len(time_values)`).
            `mean_data[0, :, :]` contains the statistics;
            `mean_data[1, :, :]` contains the standard errors.
        data_var : numpy.ndarray
            Dynamic data variance statistics and standard
            errors with shape (2, `len(data_variance_order)`, `len(time_values)`).
            `var_data[0, :, :]` contains the statistics;
            `var_data[1, :, :]` contains the standard errors.
        data_cov : numpy.ndarray
            Dynamic data covariance statistics and standard
            errors with shape (2, `len(data_covariance_order)`, `len(time_values)`).
            `cov_data[0, :, :]` contains the statistics;
            `cov_data[1, :, :]` contains the standard errors.
        data_mean_order : list of dict
            Variable order for the data means.
        data_variance_order : list of dict
            Variable order for the data variances.
        data_covariance_order : list of dict
            Variable order for the data covariances.

        Returns
        -------
        data_mean_ordered : numpy.ndarray
            Dynamic data mean statistics and standard
            errors matched to the simulation variable order
            with shape (2, `len(data_mean_order)`, `len(time_values)`).
        data_var_ordered : numpy.ndarray
            Dynamic data variance statistics and standard
            errors matched to the simulation variable order
            with shape (2, `len(data_variance_order)`, `len(time_values)`).
        data_cov_ordered : numpy.ndarray
            Dynamic data covariance statistics and standard
            errors matched to the simulation variable order
            with shape (2, `len(data_covariance_order)`, `len(time_values)`).

        Examples
        --------
        >>> # est is a memocell estimation instance obtained by est.estimate(...)
        >>> est.net_simulation.sim_variables_order
        [[('V_0',), ('V_1',)], [('V_0', 'V_0'), ('V_0', 'V_1'), ('V_1', 'V_1')]]
        >>> est.net_simulation.sim_variables_identifier
        {'V_0': ('X_t', ('X_t',)), 'V_1': ('Y_t', ('Y_t',))}
        >>> est.data.data_mean_order
        [{'variables': 'X_t', 'summary_indices': 0, 'count_indices': (0,)},
         {'variables': 'Y_t', 'summary_indices': 1, 'count_indices': (1,)}]
        >>> est.data.data_variance_order
        [{'variables': ('X_t', 'X_t'), 'summary_indices': 0, 'count_indices': (0, 0)},
         {'variables': ('Y_t', 'Y_t'), 'summary_indices': 1, 'count_indices': (1, 1)}]
        >>> est.data.data_covariance_order
        [{'variables': ('X_t', 'Y_t'), 'summary_indices': 0, 'count_indices': (0, 1)}]
        >>> est.match_data_to_network(est.net_simulation.sim_variables_order,
        >>>                           est.net_simulation.sim_variables_identifier,
        >>>                           est.data.data_mean,
        >>>                           est.data.data_variance,
        >>>                           est.data.data_covariance,
        >>>                           est.data.data_mean_order,
        >>>                           est.data.data_variance_order,
        >>>                           est.data.data_covariance_order)
        (array([[[1.        , 0.67      , 0.37      ],
                 [0.        , 0.45      , 1.74      ]],
                [[0.01      , 0.0469473 , 0.04838822],
                 [0.01      , 0.07188642, 0.1995514 ]]]),
         array([[[0.        , 0.22333333, 0.23545455],
                 [0.        , 0.51262626, 4.03272727]],
                [[0.01      , 0.01631605, 0.01293869],
                 [0.01      , 0.08878719, 0.68612036]]]),
         array([[[ 0.        , -0.30454545, -0.65030303]],
                [[ 0.01      ,  0.0303608 ,  0.06645246]]]))
        """

        # preallocate numpy arrays for the ordered data that is of same shape as the original data
        # if there is some mean_only mode active, var and cov arrays will stay at zeros
        # cases (sim_mean_only determines the variables order):
        # 1) if sim_mean_only=True, fit_mean_only is forced to be True, so we dont
        #       catch var and cov data and also dont fit it
        # 2) if sim_mean_only=False, fit_mean_only can be False too (default case);
        #       var and cov data is catched and also fitted
        # 3) if sim_mean_only=False, fit_mean_only can be True; we want to see full
        #       model summary stats but only fit to the mean data; in this case we
        #       catch the data but it is then later ignored in the fit (TODO: check)

        # NOTE: if the data has only mean summary stats, data_var_ordered
        # and data_cov_ordered will have 0 size in the variable dimension, and
        # the for loops below will just run through without doing anything
        # (as data_variance_order and data_covariance_order are empty)

        data_mean_ordered = np.zeros(data_mean.shape)
        data_var_ordered = np.zeros(data_var.shape)
        data_cov_ordered = np.zeros(data_cov.shape)

        # read out the order of mean, variances and covariances in the model
        model_mean = [sim_variables_identifier[variable][0] for variable, in sim_variables_order[0]]
        model_var = [(sim_variables_identifier[variable1][0], sim_variables_identifier[variable2][0]) for variable1, variable2 in sim_variables_order[1] if variable1==variable2]
        model_cov = [(sim_variables_identifier[variable1][0], sim_variables_identifier[variable2][0]) for variable1, variable2 in sim_variables_order[1] if variable1!=variable2]

        # loop over the mean order of the model to sort data accordingly
        for i, model_inf in enumerate(model_mean):
            for data_inf in data_mean_order:
                if data_inf['variables']==model_inf:
                    data_mean_ordered[:, i, :] = data_mean[:, data_inf['summary_indices'], :]

        # loop over the var order of the model to sort data accordingly
        for i, model_inf in enumerate(model_var):
            for data_inf in data_variance_order:
                if data_inf['variables']==model_inf:
                    data_var_ordered[:, i, :] = data_var[:, data_inf['summary_indices'], :]

        # loop over the cov order of the model to sort data accordingly
        for i, model_inf in enumerate(model_cov):
            for data_inf in data_covariance_order:
                if set(data_inf['variables'])==set(model_inf): # set() used due to symmetry [Cov(A,B)=Cov(B,A)]
                    data_cov_ordered[:, i, :] = data_cov[:, data_inf['summary_indices'], :]

        return (data_mean_ordered, data_var_ordered, data_cov_ordered)


    def compute_bestfit_simulation(self):
        """Compute a simulation of the estimated model with 'best-fit' :math:`\\theta`
        parameters. The 50-th percentiles (i.e., medians) of the respective
        1d marginal posterior distributions are taken as best-fit parameter values.

        The simulation result is available at `est.net_simulation_bestfit` for an
        estimation instance `est`. This method is used as plotting helper function;
        see `plots.est_bestfit_mean_plot`, `plots.est_bestfit_variance_plot` and
        `plots.est_bestfit_covariance_plot` for more info.

        Returns
        -------
        None
        """

        # run a simulation with best-fit values (here we used median/50th percentile of
        # one-dimensional parameter densities)
        if self.net_simulation.sim_moments.moment_system=='reset':
            self.net_simulation.sim_moments.set_moment_eqs_from_template_after_reset()

        self.net_simulation_bestfit = self.net_simulation.sim_moments.run_moment_ode_system(
                                            self.net_simulation.sim_moments.moment_initial_values,
                                            self.net_time_values_dense,
                                            self.bay_est_params_median)
        self.net_simulation_bestfit_exists = True


    def compute_simulation_credible_band(self, num_sim_ensemble=5000):
        """Compute a simulation of the estimated model with 'best-fit' :math:`\\theta`
        parameters and 95% credible band. Credible band and best-fit simulation are
        computed as percentiles (2.5-th, 50-th and 97.5-th) from an ensemble of
        simulations with parameter values sampled from the complete parameter
        posterior.

        The best-fit simulation result is available at
        `est.net_simulation_credible_band_bestfit` for an estimation instance `est`;
        the 95% credible band at `est.net_simulation_credible_band`.
        This method is used as plotting helper function;
        see `plots.est_bestfit_mean_plot`, `plots.est_bestfit_variance_plot` and
        `plots.est_bestfit_covariance_plot` for more info.

        Parameters
        ----------
        num_sim_ensemble : int, optional
            Ensemble size (number of simulations) to compute percentiles from.

        Returns
        -------
        None
        """

        # TODO: has to be improved! store model simulations from mcmc and use here again!
        # NOTE: we would need the data blobs functionality which is currently
        # only available for the EnsembleSampler but unfortunately not for the PTSampler

        if self.net_simulation.sim_moments.moment_system=='reset':
            self.net_simulation.sim_moments.set_moment_eqs_from_template_after_reset()

        # take at least 5000 simulations to calculate credible bands
        num_sim_ensemble = max(num_sim_ensemble, 5000)

        # raise warning if posterior samples are less than drawn samples for sim_ensemble
        if self.bay_est_samples_weighted.shape[0]<=num_sim_ensemble:
            warnings.warn(f'There are less than {num_sim_ensemble} parameter posterior samples to compute model credible bands from. Consider increasing depth of MCMC sampling.')


        # recompute the model simulations for a random selection of posterior sample
        # i.e. we obtain different model trajectories according to the parameter posterior distribution
        inds = np.array(range(0, self.bay_est_samples_weighted.shape[0]))
        inds_random_selection = np.random.choice(inds, size=(num_sim_ensemble), replace=True)
        theta_ensemble = self.bay_est_samples_weighted[inds_random_selection, :]



        sim_ensemble = [self.net_simulation.sim_moments.run_moment_ode_system(
                                            self.net_simulation.sim_moments.moment_initial_values,
                                            self.net_time_values_dense,
                                            theta)
                                            for theta in theta_ensemble]

        # then we compute the statistic of the sampled trajectories (means, variances, covariances)
        # and the corresponding 2.5th and 97.5th percentiles for 95%-credible band (both for all time points)
        mean_samples = np.array([sim[0] for sim in sim_ensemble])
        mean_percentiles = np.percentile(mean_samples, (2.5, 50.0, 97.5), axis=0)
        mean_lower_bound = mean_percentiles[0, :, :]
        mean_bestfit_band = mean_percentiles[1, :, :]
        mean_upper_bound = mean_percentiles[2, :, :]

        var_samples = np.array([sim[1] for sim in sim_ensemble])
        var_percentiles = np.percentile(var_samples, (2.5, 50.0, 97.5), axis=0)
        var_lower_bound = var_percentiles[0, :, :]
        var_bestfit_band = var_percentiles[1, :, :]
        var_upper_bound = var_percentiles[2, :, :]

        cov_samples = np.array([sim[2] for sim in sim_ensemble])
        cov_percentiles = np.percentile(cov_samples, (2.5, 50.0, 97.5), axis=0)
        cov_lower_bound = cov_percentiles[0, :, :]
        cov_bestfit_band = cov_percentiles[1, :, :]
        cov_upper_bound = cov_percentiles[2, :, :]

        # store the information for the credible band
        # with structure:
        # list index 0 for mean vs. var vs. cov
        # list index 1 for lower vs. upper
        # then numpy array with shape=(number of means, vars or covs; #time_values)
        self.net_simulation_credible_band = [
        [mean_lower_bound, mean_upper_bound],
        [var_lower_bound, var_upper_bound],
        [cov_lower_bound, cov_upper_bound]
        ]

        self.net_simulation_credible_band_bestfit = (mean_bestfit_band,
                                                        var_bestfit_band,
                                                        cov_bestfit_band)

        self.net_simulation_credible_band_exists = True


    ### plotting helper functions
    def _dots_w_bars_parameters(self, settings):
        """Private plotting helper method."""

        y_arr_err = np.zeros((len(self.net.net_theta_symbolic), 3))
        x_ticks = list()
        attributes = dict()

        for i, theta_id in enumerate(self.net.net_theta_symbolic):
            (median, perc_2p5, perc_97p5) = self.bay_est_params_cred[i]
            y_arr_err[i, :] = np.array([median, median - perc_2p5, perc_97p5 - median])

            param_setting = settings[self.net.net_rates_identifier[theta_id]]
            attributes[i] = (param_setting['label'], param_setting['color'])
            x_ticks.append(param_setting['label'])

        return y_arr_err, x_ticks, attributes


    def _samples_corner_parameters(self, settings):
        """Private plotting helper method."""

        samples = self.bay_est_samples_weighted
        labels = [settings[self.net.net_rates_identifier[theta_id]]['label'] for theta_id in self.net.net_theta_symbolic]
        return samples, labels


    def _samples_chains_parameters(self):
        """Private plotting helper method."""

        return self.bay_est_samples, self.bay_nested_ndims


    def _samples_weighted_chains_parameters(self):
        """Private plotting helper method."""

        return self.bay_est_samples_weighted, self.bay_nested_ndims


    def _sampling_res_and_labels(self, settings):
        """Private plotting helper method."""

        labels = [settings[self.net.net_rates_identifier[theta_id]]['label'] for theta_id in self.net.net_theta_symbolic]
        return self.bay_nested_sampler_res, labels


    def _sampling_res_and_labels_and_priortransform(self, settings):
        """Private plotting helper method."""

        labels = [settings[self.net.net_rates_identifier[theta_id]]['label'] for theta_id in self.net.net_theta_symbolic]
        return self.bay_nested_sampler_res, labels, self.prior_transform


    def _line_evolv_bestfit_mean(self, settings):
        """Private plotting helper method."""

        if not self.net_simulation_bestfit_exists:
            self.compute_bestfit_simulation()
        mean_m, __, __  = self.net_simulation_bestfit

        sim_variables_order_mean = self.net_simulation.sim_variables_order[0]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr = self.net_time_values_dense
        y_arr = np.zeros((len(sim_variables_order_mean), len(x_arr)))
        attributes = dict()

        for i, (variable_id, ) in enumerate(sim_variables_order_mean):
            y_arr[i, :] = mean_m[i]

            variable_settings = settings[sim_variables_identifier[variable_id][0]]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes


    def _dots_w_bars_and_line_evolv_bestfit_mean_data(self, settings):
        """Private plotting helper method."""

        if not self.net_simulation_bestfit_exists:
            self.compute_bestfit_simulation()
        mean_m, __, __  = self.net_simulation_bestfit

        sim_variables_order_mean = self.net_simulation.sim_variables_order[0]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr_line = self.net_time_values_dense
        y_line = np.zeros((len(sim_variables_order_mean), len(x_arr_line)))

        x_arr_dots = self.data_time_values
        y_dots_err = np.zeros((len(self.data.data_mean_order), self.data.data_num_time_values, 2))

        attributes = dict()

        for i, (variable_id, ) in enumerate(sim_variables_order_mean):
            y_line[i, :] = mean_m[i]

            y_dots_err[i, :, 0] = self.data_mean_values[0, i, :] # mean statistic
            y_dots_err[i, :, 1] = self.data_mean_values[1, i, :] # standard error

            variable_settings = settings[sim_variables_identifier[variable_id][0]]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr_dots, x_arr_line, y_dots_err, y_line, attributes


    def _line_w_band_evolv_mean_credible(self, settings, num_sim_ensemble=5000):
        """Private plotting helper method."""

        # compute the best-fit simulation and credible bands in case they do not exist already
        if not self.net_simulation_credible_band_exists:
            self.compute_simulation_credible_band(num_sim_ensemble=num_sim_ensemble)

        # in the case of credible bands, best-fit is taken from the simulation samples
        mean_m, __, __  = self.net_simulation_credible_band_bestfit
        mean_band = self.net_simulation_credible_band[0]

        sim_variables_order_mean = self.net_simulation.sim_variables_order[0]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr = self.net_time_values_dense
        y_line = np.zeros((len(sim_variables_order_mean), len(x_arr)))
        y_lower = np.zeros((len(sim_variables_order_mean), len(x_arr)))
        y_upper = np.zeros((len(sim_variables_order_mean), len(x_arr)))
        attributes = dict()

        for i, (variable_id, ) in enumerate(sim_variables_order_mean):
            y_line[i, :] = mean_m[i]
            y_lower[i, :] = mean_band[0][i, :]
            y_upper[i, :] = mean_band[1][i, :]

            variable_settings = settings[sim_variables_identifier[variable_id][0]]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_line, y_lower, y_upper, attributes


    def _dots_w_bars_and_line_w_band_evolv_mean_credible(self, settings, num_sim_ensemble=5000):
        """Private plotting helper method."""

        # compute the best-fit simulation and credible bands in case they do not exist already
        if not self.net_simulation_credible_band_exists:
            self.compute_simulation_credible_band(num_sim_ensemble=num_sim_ensemble)

        # in the case of credible bands, best-fit is taken from the simulation samples
        mean_m, __, __  = self.net_simulation_credible_band_bestfit
        mean_band = self.net_simulation_credible_band[0]

        sim_variables_order_mean = self.net_simulation.sim_variables_order[0]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr_dots = self.data_time_values
        x_arr_line = self.net_time_values_dense
        y_dots_err = np.zeros((len(self.data.data_mean_order), self.data.data_num_time_values, 2))
        y_line = np.zeros((len(sim_variables_order_mean), len(self.net_time_values_dense)))
        y_lower = np.zeros((len(sim_variables_order_mean), len(self.net_time_values_dense)))
        y_upper = np.zeros((len(sim_variables_order_mean), len(self.net_time_values_dense)))
        attributes = dict()

        for i, (variable_id, ) in enumerate(sim_variables_order_mean):
            y_line[i, :] = mean_m[i]
            y_lower[i, :] = mean_band[0][i, :]
            y_upper[i, :] = mean_band[1][i, :]

            y_dots_err[i, :, 0] = self.data_mean_values[0, i, :] # mean statistic
            y_dots_err[i, :, 1] = self.data_mean_values[1, i, :] # standard error

            variable_settings = settings[sim_variables_identifier[variable_id][0]]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, attributes


    def _line_evolv_bestfit_variance(self, settings):
        """Private plotting helper method."""

        if not self.net_simulation_bestfit_exists:
            self.compute_bestfit_simulation()
        __, var_m, __  = self.net_simulation_bestfit

        sim_variables_order_var = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id==variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr = self.net_time_values_dense
        y_arr = np.zeros((len(sim_variables_order_var), len(x_arr)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_var):
            y_arr[i, :] = var_m[i]

            variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes


    def _dots_w_bars_and_line_evolv_bestfit_variance_data(self, settings):
        """Private plotting helper method."""

        if not self.net_simulation_bestfit_exists:
            self.compute_bestfit_simulation()
        __, var_m, __  = self.net_simulation_bestfit

        sim_variables_order_var = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id==variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr_line = self.net_time_values_dense
        y_line = np.zeros((len(sim_variables_order_var), len(x_arr_line)))

        x_arr_dots = self.data_time_values
        y_dots_err = np.zeros((len(self.data.data_variance_order), self.data.data_num_time_values, 2))

        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_var):
            y_line[i, :] = var_m[i]

            # skip data, if there is no chance to get higher moments
            if not self.data.data_mean_exists_only:
                y_dots_err[i, :, 0] = self.data_var_values[0, i, :] # var statistic
                y_dots_err[i, :, 1] = self.data_var_values[1, i, :] # standard error

            variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr_dots, x_arr_line, y_dots_err, y_line, attributes


    def _line_w_band_evolv_variance_credible(self, settings, num_sim_ensemble=5000):
        """Private plotting helper method."""

        # compute the best-fit simulation and credible bands in case they do not exist already
        if not self.net_simulation_credible_band_exists:
            self.compute_simulation_credible_band(num_sim_ensemble=num_sim_ensemble)

        # in the case of credible bands, best-fit is taken from the simulation samples
        __, var_m, __  = self.net_simulation_credible_band_bestfit
        var_band = self.net_simulation_credible_band[1]

        sim_variables_order_var = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id==variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr = self.net_time_values_dense
        y_line = np.zeros((len(sim_variables_order_var), len(x_arr)))
        y_lower = np.zeros((len(sim_variables_order_var), len(x_arr)))
        y_upper = np.zeros((len(sim_variables_order_var), len(x_arr)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_var):
            y_line[i, :] = var_m[i]
            y_lower[i, :] = var_band[0][i, :]
            y_upper[i, :] = var_band[1][i, :]

            variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_line, y_lower, y_upper, attributes


    def _dots_w_bars_and_line_w_band_evolv_variance_credible(self, settings, num_sim_ensemble=5000):
        """Private plotting helper method."""

        # compute the best-fit simulation and credible bands in case they do not exist already
        if not self.net_simulation_credible_band_exists:
            self.compute_simulation_credible_band(num_sim_ensemble=num_sim_ensemble)

        # in the case of credible bands, best-fit is taken from the simulation samples
        __, var_m, __  = self.net_simulation_credible_band_bestfit
        var_band = self.net_simulation_credible_band[1]

        sim_variables_order_var = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id==variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr_dots = self.data_time_values
        x_arr_line = self.net_time_values_dense
        y_dots_err = np.zeros((len(self.data.data_variance_order), self.data.data_num_time_values, 2))
        y_line = np.zeros((len(sim_variables_order_var), len(self.net_time_values_dense)))
        y_lower = np.zeros((len(sim_variables_order_var), len(self.net_time_values_dense)))
        y_upper = np.zeros((len(sim_variables_order_var), len(self.net_time_values_dense)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_var):
            y_line[i, :] = var_m[i]
            y_lower[i, :] = var_band[0][i, :]
            y_upper[i, :] = var_band[1][i, :]

            # skip data, if there is no chance to get higher moments
            if not self.data.data_mean_exists_only:
                y_dots_err[i, :, 0] = self.data_var_values[0, i, :] # var statistic
                y_dots_err[i, :, 1] = self.data_var_values[1, i, :] # standard error

            variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, attributes


    def _line_evolv_bestfit_covariance(self, settings):
        """Private plotting helper method."""

        if not self.net_simulation_bestfit_exists:
            self.compute_bestfit_simulation()
        __, __, cov_m  = self.net_simulation_bestfit

        sim_variables_order_cov = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id!=variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr = self.net_time_values_dense
        y_arr = np.zeros((len(sim_variables_order_cov), len(x_arr)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_cov):
            y_arr[i, :] = cov_m[i]

            try:
                variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            except:
                variable_settings = settings[(sim_variables_identifier[variable2_id][0], sim_variables_identifier[variable1_id][0])]

            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes


    def _dots_w_bars_and_line_evolv_bestfit_covariance_data(self, settings):
        """Private plotting helper method."""

        if not self.net_simulation_bestfit_exists:
            self.compute_bestfit_simulation()
        __, __, cov_m  = self.net_simulation_bestfit

        sim_variables_order_cov = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id!=variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr_line = self.net_time_values_dense
        y_line = np.zeros((len(sim_variables_order_cov), len(x_arr_line)))

        x_arr_dots = self.data_time_values
        y_dots_err = np.zeros((len(self.data.data_covariance_order), self.data.data_num_time_values, 2))

        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_cov):
            y_line[i, :] = cov_m[i]

            # skip data, if there is no chance to get higher moments
            if not self.data.data_mean_exists_only:
                y_dots_err[i, :, 0] = self.data_cov_values[0, i, :] # cov statistic
                y_dots_err[i, :, 1] = self.data_cov_values[1, i, :] # standard error

            try:
                variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            except:
                variable_settings = settings[(sim_variables_identifier[variable2_id][0], sim_variables_identifier[variable1_id][0])]

            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr_dots, x_arr_line, y_dots_err, y_line, attributes


    def _line_w_band_evolv_covariance_credible(self, settings, num_sim_ensemble=5000):
        """Private plotting helper method."""

        # compute the best-fit simulation and credible bands in case they do not exist already
        if not self.net_simulation_credible_band_exists:
            self.compute_simulation_credible_band(num_sim_ensemble=num_sim_ensemble)

        # in the case of credible bands, best-fit is taken from the simulation samples
        __, __, cov_m  = self.net_simulation_credible_band_bestfit
        cov_band = self.net_simulation_credible_band[2]

        sim_variables_order_cov = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id!=variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr = self.net_time_values_dense
        y_line = np.zeros((len(sim_variables_order_cov), len(x_arr)))
        y_lower = np.zeros((len(sim_variables_order_cov), len(x_arr)))
        y_upper = np.zeros((len(sim_variables_order_cov), len(x_arr)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_cov):
            y_line[i, :] = cov_m[i]
            y_lower[i, :] = cov_band[0][i, :]
            y_upper[i, :] = cov_band[1][i, :]

            try:
                variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            except:
                variable_settings = settings[(sim_variables_identifier[variable2_id][0], sim_variables_identifier[variable1_id][0])]

            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_line, y_lower, y_upper, attributes


    def _dots_w_bars_and_line_w_band_evolv_covariance_credible(self, settings, num_sim_ensemble=5000):
        """Private plotting helper method."""

        # compute the best-fit simulation and credible bands in case they do not exist already
        if not self.net_simulation_credible_band_exists:
            self.compute_simulation_credible_band(num_sim_ensemble=num_sim_ensemble)

        # in the case of credible bands, best-fit is taken from the simulation samples
        __, __, cov_m  = self.net_simulation_credible_band_bestfit
        cov_band = self.net_simulation_credible_band[2]

        sim_variables_order_cov = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.net_simulation.sim_variables_order[1] if variable1_id!=variable2_id]
        sim_variables_identifier = self.net_simulation.sim_variables_identifier

        x_arr_dots = self.data_time_values
        x_arr_line = self.net_time_values_dense
        y_dots_err = np.zeros((len(self.data.data_covariance_order), self.data.data_num_time_values, 2))
        y_line = np.zeros((len(sim_variables_order_cov), len(self.net_time_values_dense)))
        y_lower = np.zeros((len(sim_variables_order_cov), len(self.net_time_values_dense)))
        y_upper = np.zeros((len(sim_variables_order_cov), len(self.net_time_values_dense)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_cov):
            y_line[i, :] = cov_m[i]
            y_lower[i, :] = cov_band[0][i, :]
            y_upper[i, :] = cov_band[1][i, :]

            # skip data, if there is no chance to get higher moments
            if not self.data.data_mean_exists_only:
                y_dots_err[i, :, 0] = self.data_cov_values[0, i, :] # cov statistic
                y_dots_err[i, :, 1] = self.data_cov_values[1, i, :] # standard error

            try:
                variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            except:
                variable_settings = settings[(sim_variables_identifier[variable2_id][0], sim_variables_identifier[variable1_id][0])]

            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr_dots, x_arr_line, y_dots_err, y_line, y_lower, y_upper, attributes
    ###

    @staticmethod
    def _validate_simulation_and_data_variables_mapping(sim_variables,
                                                        data_variables):
        """Private validation method."""
        # NOTE: by previous checks in simulation and data classes sim_variables
        # and data_variables both should have unique elements only

        # we dont really care about variables of the model (main nodes),
        # but simulation and data variables have to match (otherwise one
        # should just define a different data object, if not all data variables
        # can be described; if we have more sim variables then actually available
        # data, one should define different models) -> so we need a bijective
        # mapping between the two

        # the model in terms of underlying modelled variables (main nodes) can of
        # course have more detailed variables; so a sim variable can be a sum of
        # multiple main nodes or some main nodes might not be grapped by sim
        # variables at all (TODO: this should be all possible, check there)

        # check for one-to-one (bijective mapping)
        # can use sets due to uniqueness
        if not set(sim_variables)==set(data_variables):
            raise ValueError('Simulation and data variables have to be one-to-one/bijectively mappable.')

    @staticmethod
    def _validate_network_input(network):
        """Private validation method."""

        # check for instance of Network class
        if isinstance(network, Network):
            pass
        else:
            raise TypeError('Instance of Network class expected.')
        return network

    @staticmethod
    def _validate_data_input(data):
        """Private validation method."""

        # check for instance of Data class
        if isinstance(data, Data):
            pass
        else:
            raise TypeError('Instance of Data class expected.')
        return data

    @staticmethod
    def _validate_initial_values_input(simulation_type, initial_values_type,
                                    initial_moments):
        """Private validation method."""
        # NOTE: further checks will be done via simulation classes

        # initial_values_type should be either 'synchronous' or 'uniform'
        if initial_values_type=='synchronous' or initial_values_type=='uniform':
            pass
        elif not isinstance(initial_values_type, str):
            raise TypeError('Initial values type is not a string.')
        else:
            raise ValueError('Unknown initial values type: \'synchronous\' or \'uniform\' are expected.')

        # simulation_type has to be moments for estimation
        if not simulation_type=='moments':
            raise ValueError('Simulation type has to be \'moments\' for estimations.')
        if not isinstance(initial_moments, dict):
            raise TypeError('Dictionary expected for initial values (moments).')

    @staticmethod
    def _validate_theta_bounds_input(net_rates_identifier, theta_bounds):
        """Private validation method."""

        # check for correct user input for the rate parameters (theta)
        if isinstance(theta_bounds, dict):
            if set(net_rates_identifier.values()) == set(theta_bounds.keys()):
                if all(isinstance(bounds, tuple) for bounds in theta_bounds.values()) and all(len(bounds)==2 for bounds in theta_bounds.values()):
                    if all(isinstance(val, float) for bounds in theta_bounds.values() for val in bounds):
                        pass
                    else:
                        raise ValueError('Bounds of rate parameters (theta bounds) are expected to provide a tuple of two float values for each parameter.')
                else:
                    raise ValueError('Bounds of rate parameters (theta bounds) are expected to provide a tuple of two float values for each parameter.')
            else:
                raise ValueError('Bounds of rate parameters (theta bounds) are expected to provide a set of keys identical to the symbolic network parameters (theta).')
        else:
            raise TypeError('Bounds of rate parameters (theta bounds) are expected to be provided as a dictionary.')

    @staticmethod
    def _validate_time_values_input(time_values, data_time_values):
        """Private validation method."""

        # check for correct user input for the time_values
        # data_time_values are checked in data class (no need here)
        if isinstance(time_values, np.ndarray) or isinstance(time_values, type(None)):
            pass
        else:
            raise TypeError('Times values are expected to be None or a numpy array.')

        if isinstance(time_values, np.ndarray):
            if time_values.ndim == 1:
                pass
            else:
                raise ValueError('Times values are expected to be provided as a numpy array with shape \'(n, )\' with n being the number of values.')

        # if time_values is specified, check if each data time point
        # can be found in time_values (needed for index readout later)
        # (NOTE: it seems that this np function does not check for 1d-ness)
        if isinstance(time_values, np.ndarray):
            if not np.in1d(data_time_values, time_values).all():
                raise ValueError('Time values have to contain all data time values.')


    ##### TODO: old/adapt
    # def setup_prior(self, reac_ranges_list):
    #     # define the parameter intervals for the uniform prior
    #     reac_ranges = np.zeros((3, 2))
    #     reac_ranges[0, :] = np.array(reac_ranges_list[0])
    #     reac_ranges[1, :] = np.array(reac_ranges_list[1])
    #     reac_ranges[2, :] = np.array(reac_ranges_list[2])
    #     self.reac_ranges = reac_ranges
    #
    #     # find occurrences of each reaction type (this works since 1=True, 0=False)
    #     reac_occ_div = np.count_nonzero(self.reac_types==0)
    #     reac_occ_diff = np.count_nonzero(self.reac_types==1)
    #     reac_occ_death = np.count_nonzero(self.reac_types==2)
    #
    #     # compute the normalisation constant for the uniform prior
    #     prior_norm = ((self.reac_ranges[0, 1] - self.reac_ranges[0, 0])**reac_occ_div
    #                         * (self.reac_ranges[1, 1] - self.reac_ranges[1, 0])**reac_occ_diff
    #                         * (self.reac_ranges[2, 1] - self.reac_ranges[2, 0])**reac_occ_death)
    #
    #     # compute the value of the log prior on the support
    #     self.log_prior_supp = np.log(1.0/prior_norm)

    # normalisation is defined via reaction type specific range and how often each type occurs
    # on_support is a boolean; True if all parameters are on the support (prior > 0) else False (prior = 0)
    # def log_prior(self, theta):
    #     reac_type_div = theta[self.reac_types==0]
    #     reac_type_diff = theta[self.reac_types==1]
    #     reac_type_death = theta[self.reac_types==2]
    #
    #     bool_reac_type_div = (reac_type_div > self.reac_ranges[0, 0]) & (reac_type_div <= self.reac_ranges[0, 1])
    #     bool_reac_type_diff = (reac_type_diff > self.reac_ranges[1, 0]) & (reac_type_diff <= self.reac_ranges[1, 1])
    #     bool_reac_type_death = (reac_type_death > self.reac_ranges[2, 0]) & (reac_type_death <= self.reac_ranges[2, 1])
    #
    #     on_support = np.all(np.concatenate((bool_reac_type_div, bool_reac_type_diff, bool_reac_type_death)))
    #     if on_support:
    #         return self.log_prior_supp
    #     else:
    #         return -np.inf
    #
    # def setup_likelihood(self, moment_order, moment_order_main):
    #     # obtain the number of means, variances and covariances of/between the network nodes (num_... variables)
    #     # obtain indices which allow a correct reading of the solution array of odeint (..._ind variables)
    #     (self.num_means, self.mean_ind, self.num_vars, self.var_ind_intra, self.var_ind_inter,
    #         self.num_covs, self.cov_ind) = self.get_indices_for_solution_readout(moment_order, moment_order_main)
    #
    # def log_likelihood(self, theta, init, time_arr, mean_data, var_data, cov_data):
    #     mean_m, var_m, cov_m = self.forward_pass(init, time_arr, theta, self.num_time_points)
    #
    #     chi_mean = np.sum( ((mean_data[0, :, :] - mean_m)/(mean_data[1, :, :]))**2  + np.log(2 * np.pi * (mean_data[1, :, :]**2)) )
    #     chi_var = np.sum( ((var_data[0, :, :] - var_m)/(var_data[1, :, :]))**2  + np.log(2 * np.pi * (var_data[1, :, :]**2)) )
    #     chi_cov = np.sum( ((cov_data[0, :, :] - cov_m)/(cov_data[1, :, :]))**2  + np.log(2 * np.pi * (cov_data[1, :, :]**2)) )
    #     return -0.5 * (chi_mean + chi_var + chi_cov)
    # #####
    #
    # ##### TODO: old / adapt
    # ### two main functions
    # def init_mcmc(self, moments_init, synchro, burn_in_steps=1000, sampling_steps=500, num_temps=5, num_walkers=200, num_threads=1):
    #     # add mcmc settings to self
    #     self.mcmc_burn_in_steps = burn_in_steps # sampling steps per walker
    #     self.mcmc_sampling_steps = sampling_steps # sampling steps per walker
    #     self.mcmc_num_temps = num_temps # 10, 20, 5
    #     self.mcmc_num_walkers = num_walkers # 100, 200
    #     self.mcmc_num_params = len(self.sym_params)
    #     self.mcmc_num_threads = num_threads # threads used for parallisation, NOTE: not yet implemented for this class
    #     self.synchronized = synchro
    #
    #     # load the initial conditions for moments needed (moments_init can contain more moments)
    #     # moment_order_sorted = list()
    #     # print(self.moment_order)
    #     # print(moments_init)
    #     # for moment in self.moment_order:
    #     #     if len(moment)==1:
    #     #         moment_order_sorted.append(moment)
    #     #     if len(moment)==2:
    #     #         if moment[0]<=moment[1]:
    #     #             moment_order_sorted.append(moment)
    #     #         elif moment[0]>moment[1]:
    #     #             moment_order_sorted.append((moment[1], moment[0]))
    #     # self.mcmc_init_cond = [moments_init[moment] for moment in moment_order_sorted]
    #
    #     # original code
    #     self.mcmc_init_cond = self.get_init_moments_list(moments_init, self.moment_order, self.synchronized)
    #     # all_posible_init_cond = self.get_init_moments_list(moments_init, self.moment_order)
    #     # random_num = np.random.randint(all_posible_init_cond.shape[1])
    #     # self.mcmc_init_cond = list(all_posible_init_cond[:,random_num])
    #
    #
    #     # NOTE: maybe add the prior args also here and not using self in the function (might be faster...)
    #     self.mcmc_sampler = PTSampler(self.mcmc_num_temps, self.mcmc_num_walkers, self.mcmc_num_params,
    #                                     self.log_likelihood, self.log_prior,
    #                                     loglargs=(self.mcmc_init_cond , self.model_time_array, self.data_mean, self.data_var, self.data_cov),
    #                                     threads=self.mcmc_num_threads)
    #
    #     # sample initial parameter values according to their bounds of their reaction type
    #     # reaction types are division, differentiation or death rates; indexing in this order (0, 1, 2)
    #     self.mcmc_initial_params = self.sample_init_params(self.mcmc_num_temps, self.mcmc_num_walkers, self.mcmc_num_params, self.reac_types, self.reac_ranges)
    #
    #     # save the init_mcmc settings
    #     self.save_init_mcmc_variables(self.mcmc_burn_in_steps, self.mcmc_sampling_steps,
    #                                 self.mcmc_num_temps, self.mcmc_num_walkers,
    #                                 self.mcmc_num_params, self.mcmc_num_threads,
    #                                 self.mcmc_init_cond, self.mcmc_initial_params,
    #                                 self.mcmc_path, self.run_name)
    #
    # def run_mcmc(self):
    #     mcmc_sampler = self.mcmc_sampler
    #     # burn in a few steps
    #     # print('burn in ...')
    #
    #     for p, lnprob, lnlike in mcmc_sampler.sample(self.mcmc_initial_params, iterations=self.mcmc_burn_in_steps):
    #         pass
    #     mcmc_sampler.reset()
    #
    #     # actual sampling
    #     # the last (p, lnprob, lnlike) values from the burn in are used here for the start
    #     # print('sampling ...')
    #     for p, lnprob, lnlike in mcmc_sampler.sample(p, lnprob0=lnprob,
    #                                                lnlike0=lnlike,
    #                                                iterations=self.mcmc_sampling_steps, thin=1):
    #         pass
    #     self.mcmc_sampler = mcmc_sampler
    #
    #     # print('finalise ...')
    #
    #     self.save_samples(self.mcmc_sampler, self.mcmc_path, self.run_name)
    #
    #     # the samples used for parameter estimation are at standard temperature = 1
    #     # i.e. beta=1/temperature=1, i.e. index = 0 (see following print command)
    #     # print(self.mcmc_sampler.betas)
    #     self.samples_temp1 = self.mcmc_sampler.chain[0, :, :, :].reshape(self.mcmc_sampling_steps * self.mcmc_num_walkers, self.mcmc_num_params)
    #     # this temperature and all the others are used to estimatate the evidence by an interpolated thermodynamic integral (see emcee docs)
    #
    #     # assess confidence bounds for parameters and plot it
    #     self.params_conf = self.get_and_plot_confidence_bounds(self.samples_temp1, self.sym_params_latex, self.mcmc_path, self.run_name)
    #     self.params_est = np.array([self.params_conf[i][0] for i in range(len(self.params_conf))])
    #     self.save_params(self.sym_params, self.params_conf, self.params_est, self.mcmc_path, self.run_name)
    #
    #     # create corner plot for the posterior parameter distributions
    #     self.create_corner_plot(self.samples_temp1, self.sym_params_latex, self.mcmc_path, self.run_name)
    #
    #     # plot mcmc chains for different temperatures and parameters
    #     # the walker dimension is flattenend, i.e. the chains from all walkers are concatenated
    #     self.plot_chains(self.mcmc_sampler, self.mcmc_num_temps, self.mcmc_sampling_steps, self.mcmc_num_walkers, self.sym_params, self.sym_params_latex, self.mcmc_path, self.run_name)
    #
    #     # compute model predictions (different plots with/without data)
    #     # NOTE: that this can be implemented much faster; see comments in self.compute_and_plot_model_predictions()
    #     self.compute_and_plot_model_predictions(self.forward_pass, self.mcmc_init_cond, self.model_time_array,
    #                                             self.samples_temp1, self.params_est, self.pred_path, self.run_name,
    #                                             self.model_color_code, (self.data_mean, self.data_var, self.data_cov))
    #
    #     # calculate evidence and plot evidence with error
    #     self.log_evidence, self.log_evidence_error = self.compute_and_plot_model_evidence(self.mcmc_sampler)
    #     self.save_evidence(self.log_evidence, self.log_evidence_error, self.mcmc_num_temps, self.mcmc_path, self.run_name)
    #
    #
    # ### helper functions
    # def sample_init_params(self, num_temps, num_walkers, num_params, reac_types, reac_ranges):
    #     mcmc_initial_params = np.zeros((num_temps, num_walkers, num_params))
    #
    #     # for each parameter (which has a specified reaction type) we sample uniform within the bounds
    #     # this ensures that the initial parameters start on the support of the prior (prior > 0)
    #     for i in range(num_params):
    #         reac_type = reac_types[i]
    #         mcmc_initial_params[:, :, i] = np.random.uniform(low=reac_ranges[reac_type, 0], high=reac_ranges[reac_type, 1], size=(num_temps, num_walkers))
    #     return mcmc_initial_params
    #
    # #####
