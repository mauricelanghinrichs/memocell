
from .network import Network
from .data import Data
from .simulation import Simulation

from emcee import PTSampler
import numpy as np

# TODO: user input validation?

class Estimation(object):
    """docstring for ."""

    def __init__(self, network, data):

        # validate network input (has to be instance of Network class) and instantiate
        self.net = self.validate_network_input(network)

        # validate data input (has to be instance of Data class) and instantiate
        # other objects for data input
        self.data = self.validate_data_input(data)
        self.data_time_values = None
        self.data_mean_values = None
        self.data_var_values = None
        self.data_cov_values = None

        # ###### INTERMEDIATE SOLUTION TO TEST THIS CLASS
        # # function to load and return data, specified with the input variables
        # # data has shape = (type, variables, time_points), where type can be a statistic or standard error of that statistic
        # # function returns the tuple (mean_data, var_data, cov_data)
        # def load_data(str_type):
        #     if str_type=='18_01_14_data_cd44_28_mp':
        #         ### NOTE: specify if a basic sigma should be added
        #         add_basic_sigma = True
        #         ###
        #
        #         # mlanghinrichs or mauricelanghinrichs
        #         top_folder = '/Users/mauricelanghinrichs/Documents/Studium/MSc/09_hiwi_hoefer/01_project/memo_py/examples/numpy_data_cd44_28_mp'
        #         mean_data = np.load(top_folder + '/mean_data_reduced_single.npy')
        #         var_data = np.load(top_folder + '/var_data_reduced_single.npy')
        #         cov_data = np.load(top_folder + '/cov_data_reduced_single.npy')
        #
        #         # add basic sigma
        #         # only set those to basic sigma which have a zero value
        #         if add_basic_sigma:
        #             basic_sigma = 0.001
        #             mean_data[1, mean_data[1, :, :]==0] += basic_sigma
        #             var_data[1, var_data[1, :, :]==0] += basic_sigma
        #             cov_data[1, cov_data[1, :, :]==0] += basic_sigma
        #
        #         return (mean_data, var_data, cov_data)
        #
        # data = load_data('18_01_14_data_cd44_28_mp')
        # self.data_time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        # self.data_mean_values = data[0]
        # self.data_var_values = data[1]
        # self.data_cov_values = data[2]
        #
        # print(self.data_mean_values)
        # ######

        ### network related settings
        # instantiate object for rate parameter bounds (theta bounds)
        self.net_theta_bounds = None

        # instantiate object for initial values of network states
        self.net_initial_values = None

        # set simulation type of the network to 'moments'
        self.net_simulation_type = 'moments'

        # instantiate object for an instance of the simulation class for the network
        self.net_simulation = None

        # instantiate object for first moment only or first and second moments computation
        self.net_simulation_mean_only = None

        # initialise bool to pass estimate_mode=True to simulation methods
        self.net_simulation_estimate_mode = True
        ###

        ### bayesian (bay) inference related settings
        # instantiate objects for mcmc (Markov chain Monte Carlo) settings
        self.bay_mcmc_burn_in_steps = None
        self.bay_mcmc_sampling_steps = None
        self.bay_mcmc_num_temps = None
        self.bay_mcmc_num_walkers = None
        self.bay_mcmc_num_dim = None
        self.bay_mcmc_num_threads = None
        self.bay_mcmc_initial_theta = None
        self.bay_mcmc_sampler = None

        # instantiate object for information on the prior
        # (value of logarithmic prior on its support)
        self.bay_log_prior_supp = None

        # instantiate objects to assign estimation results
        self.bay_est_samples_temp1 = None
        self.bay_est_params_conf = None
        self.bay_est_params_median = None
        self.bay_est_log_evidence = None
        self.bay_est_log_evidence_error = None
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

    def estimate(self, network_setup, mcmc_setup):
        """docstring for ."""

        # initialise estimation
        # (set up network, simulation and sampling properties)
        print('initialise estimation ...')
        self.initialise_estimation(network_setup, mcmc_setup)

        # execute the sampling for the estimation of parameters and model evidence
        print('run estimation ...')
        self.run_estimation()

        print(f"""results:\n
        \t theta confidence: {self.bay_est_params_conf}\n
        \t theta medians: {self.bay_est_params_median}\n
        \t log evidence: {self.bay_est_log_evidence}\n
        \t log evidence error: {self.bay_est_log_evidence_error}""")



    def run_estimation(self):
        """docstring for ."""
        counter = 0

        # NOTE: the very first burn in step can take a while since preparations for the
        # moment calculations have to be done
        bay_mcmc_sampler = self.bay_mcmc_sampler
        # burn in a few steps
        print('   burn in ...')
        for p, lnprob, lnlike in bay_mcmc_sampler.sample(self.bay_mcmc_initial_theta, iterations=self.bay_mcmc_burn_in_steps):
            counter += 1
            print(counter)
            pass
        bay_mcmc_sampler.reset()

        # actual sampling
        # the last (p, lnprob, lnlike) values from the burn in are used here for the start
        print('   sampling ...')
        for p, lnprob, lnlike in bay_mcmc_sampler.sample(p, lnprob0=lnprob,
                                                   lnlike0=lnlike,
                                                   iterations=self.bay_mcmc_sampling_steps, thin=1):
            counter += 1
            print(counter)
            pass

        self.bay_mcmc_sampler = bay_mcmc_sampler

        print('   finalise ...')
        # the samples used for parameter estimation are at standard temperature = 1
        # i.e. beta=1/temperature=1, i.e. index = 0 (see following print command)
        # print(self.mcmc_sampler.betas)
        self.bay_est_samples_temp1 = self.bay_mcmc_sampler.chain[0, :, :, :].reshape(self.bay_mcmc_sampling_steps * self.bay_mcmc_num_walkers, self.bay_mcmc_num_dim)
        # this temperature and all the others are used to estimatate the evidence by an interpolated thermodynamic integral (see emcee docs)

        # assess confidence bounds for parameters
        self.bay_est_params_conf = self.get_confidence_bounds(self.bay_est_samples_temp1)
        self.bay_est_params_median = np.array([self.bay_est_params_conf[i][0] for i in range(len(self.bay_est_params_conf))])

        # calculate evidence and plot evidence with error
        self.bay_est_log_evidence, self.bay_est_log_evidence_error = self.compute_model_evidence(self.bay_mcmc_sampler)


    def get_confidence_bounds(self, samples_at_temperature1):
        """docstring for ."""

        # the 5th, 50th and 95th percentiles of parameter distributions are extracted
        # params_conf then contains the tuple (median (50th), lower bound (5th), upper bound (95th))
        params_conf = tuple(map(lambda v: (v[1], v[0], v[2]), zip(*np.percentile(samples_at_temperature1, [5, 50, 95], axis=0))))
        return params_conf

    def compute_model_evidence(self, sampler):
        """docstring for ."""

        # estimation of the logarithmic evidence and an according error
        log_evid, log_evid_err = sampler.thermodynamic_integration_log_evidence()

        return log_evid, log_evid_err

    def initialise_estimation(self, network_setup, mcmc_setup):
        """docstring for ."""
        ### initialise network related settings
        # validate theta bounds user input and assign to numpy array object
        self.validate_theta_bounds_input(self.net.net_rates_identifier, network_setup['theta_bounds'])
        self.net_theta_bounds = self.initialise_net_theta_bounds(self.net.net_theta_symbolic, self.net.net_rates_identifier, network_setup['theta_bounds'])

        # validate initial values user input and assign to self
        # (further processing is done in the called simulation class methods)
        self.validate_initial_values_input(self.net.net_nodes_identifier, self.net_simulation_type, network_setup['initial_values'])
        self.net_initial_values = network_setup['initial_values']

        # set the mean only mode (True or False)
        self.net_simulation_mean_only = network_setup['mean_only']

        ### initialise the simulation for the network
        # generate an instance of the Simulation class
        self.net_simulation = Simulation(self.net)

        # prepare the moment
        self.net_simulation.sim_moments.prepare_moment_simulation(mean_only=self.net_simulation_mean_only, estimate_mode=self.net_simulation_estimate_mode)


        ### initialise data
        self.data_time_values = self.data.data_time_values

        (self.data_mean_values,
        self.data_var_values,
        self.data_cov_values) = self.match_data_to_network(self.net_simulation.sim_moments.moment_order_main,
                                                        self.net.net_nodes_identifier,
                                                        self.data.data_mean,
                                                        self.data.data_variance,
                                                        self.data.data_covariance,
                                                        self.data.data_mean_order,
                                                        self.data.data_variance_order,
                                                        self.data.data_covariance_order)

        ### initialise bayesian inference related settings
        # computation of the log prior value for theta's that are on the support of the prior
        self.bay_log_prior_supp = self.compute_bayes_log_prior_value_on_support(self.net_theta_bounds)

        self.bay_mcmc_burn_in_steps = mcmc_setup['burn_in_steps'] # sampling steps per walker
        self.bay_mcmc_sampling_steps = mcmc_setup['sampling_steps'] # sampling steps per walker
        self.bay_mcmc_num_temps = mcmc_setup['num_temps'] # number of temperatures, e.g. 5, 10 or 20
        self.bay_mcmc_num_walkers = mcmc_setup['num_walkers'] # number of walkers, e.g. 100 or 200
        self.bay_mcmc_num_dim = len(self.net.net_theta_symbolic) # number of dimension for estimation (= number of rate parameters (theta))
        self.bay_mcmc_num_threads = 1 # threads used for parallisation, NOTE: not yet implemented for this class

        # sample uniformly within the bounds for each parameter to obtain
        # initial parameter values for each walker at each temperature
        self.bay_mcmc_initial_theta = self.generate_bayes_mcmc_initial_theta(self.bay_mcmc_num_temps,
                                                                            self.bay_mcmc_num_walkers,
                                                                            self.bay_mcmc_num_dim,
                                                                            self.net_theta_bounds)
        # define the sampler used for the estimation of parameters and model evidence
        # the PTSampler from the emcee package is used here
        # it also allows estimation of model evidence by 'thermodynamic integration'
        # (see package documentation there)
        self.bay_mcmc_sampler = PTSampler(self.bay_mcmc_num_temps, self.bay_mcmc_num_walkers, self.bay_mcmc_num_dim,
                                        self.log_likelihood, self.log_prior,
                                        loglargs=(self.net_initial_values, self.data_time_values, self.data_mean_values, self.data_var_values, self.data_cov_values),
                                        threads=self.bay_mcmc_num_threads)


    def log_prior(self, theta):
        """docstring for ."""

        # log_prior is based on a uniform prior distribution with finite support
        # on_support is a boolean; True if all parameters/theta's are on the support (prior > 0) else False (prior = 0)
        on_support = np.all(( self.net_theta_bounds[:, 0] <= theta ) & ( theta <= self.net_theta_bounds[:, 1] ))

        # log_prior returns its log value > -infinity (if on_support) or -infinity (if not on_support)
        if on_support:
            return self.bay_log_prior_supp
        else:
            return -np.inf


    def log_likelihood(self, theta_values, initial_values, time_values, mean_data, var_data, cov_data):
        """docstring for ."""

        # NOTE: in the bayesian framework employed here the likelihood is the
        # probability of the data, given a model (structure) and model parameters;
        # the log_likelihood is theoretically based on iid normally distributed error values
        # (data = model + error); thus, effectively, the log_likelihood depends
        # on the squared differences between data and model weighted by
        # measurement uncertainties (see chi's below)

        # mean, variance (if specified), covariance (if specified) of the model
        # are generated by the simulation class by a moment-based approach
        mean_m, var_m, cov_m  = self.net_simulation.simulate(self.net_simulation_type, initial_values,
                                                                        theta_values, time_values,
                                                                        moment_mean_only=self.net_simulation_mean_only,
                                                                        estimate_mode=self.net_simulation_estimate_mode)

        # compute the value of the log_likelihood
        if self.net_simulation_mean_only:
            # when only mean values are fitted (first moments only)
            chi_mean = np.sum( ((mean_data[0, :, :] - mean_m)/(mean_data[1, :, :]))**2  + np.log(2 * np.pi * (mean_data[1, :, :]**2)) )
            chi_var = 0.0
            chi_cov = 0.0
        else:
            # when first (mean) and second moments (i.e., variance and covariance) are fitted
            chi_mean = np.sum( ((mean_data[0, :, :] - mean_m)/(mean_data[1, :, :]))**2  + np.log(2 * np.pi * (mean_data[1, :, :]**2)) )
            chi_var = np.sum( ((var_data[0, :, :] - var_m)/(var_data[1, :, :]))**2  + np.log(2 * np.pi * (var_data[1, :, :]**2)) )
            chi_cov = np.sum( ((cov_data[0, :, :] - cov_m)/(cov_data[1, :, :]))**2  + np.log(2 * np.pi * (cov_data[1, :, :]**2)) )
        return -0.5 * (chi_mean + chi_var + chi_cov)

    @staticmethod
    def compute_bayes_log_prior_value_on_support(net_theta_bounds):
        """docstring for ."""

        # compute the normalisation constant for the uniform prior
        # it is the volume given by the product of interval ranges of the theta's
        prior_norm = np.prod(np.diff(net_theta_bounds, axis=1))

        # compute the value of the log prior on the support
        bayes_log_prior_supp = np.log(1.0/prior_norm)
        return bayes_log_prior_supp

    @staticmethod
    def generate_bayes_mcmc_initial_theta(num_temps, num_walkers, num_theta, theta_bounds):
        """docstring for ."""

        # preallocate an array for initial values for theta
        # (a value is required for each theta_i, each walker, each temperature)
        mcmc_initial_params = np.zeros((num_temps, num_walkers, num_theta))

        # for each theta_i / parameter we sample uniformly within its respective bounds
        # (on a linear scale)
        # this ensures that the initial parameters start on the support of the prior
        # (meaning prior(theta) > 0 and log(prior(theta))>-inf)
        for i in range(num_theta):
            mcmc_initial_params[:, :, i] = np.random.uniform(low=theta_bounds[i, 0], high=theta_bounds[i, 1], size=(num_temps, num_walkers))
        return mcmc_initial_params

    @staticmethod
    def initialise_net_theta_bounds(theta_symbolic, theta_identifier, theta_bounds):
        """docstring for ."""

        # define the parameter intervals (by lower and upper bounds) for the uniform prior
        # theta symbolic list also defines the order of the theta's in net_theta_bounds
        net_theta_bounds = np.zeros((len(theta_symbolic), 2))

        for i, theta in enumerate(theta_symbolic):
            net_theta_bounds[i, :] = np.array(theta_bounds[theta_identifier[theta]])
        return net_theta_bounds

    @staticmethod
    def match_data_to_network(                          moment_order_main,
                                                        net_nodes_identifier,
                                                        data_mean,
                                                        data_var,
                                                        data_cov,
                                                        data_mean_order,
                                                        data_variance_order,
                                                        data_covariance_order):
        """docstring for ."""

        # preallocate numpy arrays for the ordered data that is of same shape as the original data
        data_mean_ordered = np.zeros(data_mean.shape)
        data_var_ordered = np.zeros(data_var.shape)
        data_cov_ordered = np.zeros(data_cov.shape)

        # read out the order of mean, variances and covariances in the model
        model_mean = [net_nodes_identifier[node] for node, in moment_order_main[0]]
        model_var = [(net_nodes_identifier[node1], net_nodes_identifier[node2]) for node1, node2 in moment_order_main[1] if node1==node2]
        model_cov = [(net_nodes_identifier[node1], net_nodes_identifier[node2]) for node1, node2 in moment_order_main[1] if node1!=node2]

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

    ### plotting helper functions
    def dots_w_bars_parameters(self, settings):
        """docstring for ."""

        y_arr_err = np.zeros((len(self.net.net_theta_symbolic), 3))
        x_ticks = list()
        attributes = dict()

        for i, theta_id in enumerate(self.net.net_theta_symbolic):
            (median, perc_5, perc_95) = self.bay_est_params_conf[i]
            y_arr_err[i, :] = np.array([median, median - perc_5, perc_95 - median])

            param_setting = settings[self.net.net_rates_identifier[theta_id]]
            attributes[i] = (param_setting['label'], param_setting['color'])
            x_ticks.append(param_setting['label'])

        return y_arr_err, x_ticks, attributes


    def samples_corner_parameters(self, settings, temperature_ind=0):
        """docstring for ."""

        samples = self.bay_mcmc_sampler.chain[temperature_ind, :, :, :].reshape(self.bay_mcmc_sampling_steps * self.bay_mcmc_num_walkers, self.bay_mcmc_num_dim)
        labels = [settings[self.net.net_rates_identifier[theta_id]]['label'] for theta_id in self.net.net_theta_symbolic]
        return samples, labels

    def samples_chains_parameters(self):
        """docstring for ."""

        return self.bay_mcmc_sampler, self.bay_mcmc_num_temps, self.bay_mcmc_sampling_steps, self.bay_mcmc_num_walkers, self.bay_mcmc_num_dim
    ###

    @staticmethod
    def validate_network_input(network):
        """docstring for ."""

        # check for instance of Network class
        if isinstance(network, Network):
            pass
        else:
            raise TypeError('Instance of Network class expected.')
        return network

    @staticmethod
    def validate_data_input(data):
        """docstring for ."""

        # check for instance of Data class
        if isinstance(data, Data):
            pass
        else:
            raise TypeError('Instance of Data class expected.')
        return data

    @staticmethod
    def validate_initial_values_input(net_nodes_identifier, simulation_type, initial_values):
        """docstring for ."""

        # check for correct user input for the initial values
        if isinstance(initial_values, dict):
            if set(net_nodes_identifier.values()) - set(['env']) == set(initial_values.keys()):
                if ((simulation_type=='gillespie' and all(isinstance(val, int) for val in initial_values.values())) or
                    (simulation_type=='moments' and all(isinstance(val, float) for val in initial_values.values()))):
                    pass
                else:
                    raise ValueError('Initial values are expected to provide integer or float values for Gillespie or Moment simulations, respectively.')
            else:
                raise ValueError('Initial values are expected to provide a set of keys identical to the nodes of the main network.')
        else:
            raise TypeError('Initial values are expected to be provided as a dictionary.')

    @staticmethod
    def validate_theta_bounds_input(net_rates_identifier, theta_bounds):
        """docstring for ."""

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
