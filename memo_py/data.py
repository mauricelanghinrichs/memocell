
import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize

class Data(object):
    """docstring for ."""

    def __init__(self, data_name):

        # initialise the name of this data set
        self.data_name = data_name

        # instantiate an object for the nodes/variables of this data
        self.data_variables = None
        self.data_num_variables = None

        # instantiate an object for time points of this data
        self.data_time_values = None
        self.data_num_time_values = None

        #  instantiate objects for the order of variables in mean, variance and covariance data
        self.data_mean_order = None
        self.data_variance_order = None
        self.data_covariance_order = None

        # instantiate an object for the type of this data
        self.data_type = None

        # instantiate objects for the data summary statistics
        self.data_mean = None
        self.data_variance = None
        self.data_covariance = None

        # instantiate object for the number of data points in summary statistics
        self.data_num_values = None
        self.data_num_values_mean_only = None # in case data is used in mean_only mode

        # instantiate object for the count data
        # only used, if data_type == 'counts'
        self.data_counts = None

        # instantiate object for the number of bootstrapping samples to
        # convert count data to summary statistics
        self.data_bootstrap_samples = None

        # instantiate object for a basic_sigma value to handle zero-valued standard errors
        self.data_basic_sigma = None

        # instantiate objects for data events analysis
        self.event_all_first_change_from_inital_conditions = None
        self.event_all_first_cell_count_increase = None
        self.event_all_first_cell_type_conversion = None
        self.event_all_first_cell_count_increase_after_cell_type_conversion = None
        self.event_all_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion = None
        self.event_all_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion = None

        # instantiate objects for Gamma fit of binned waiting time distribution
        self.gamma_fit_bins = None
        self.gamma_fit_bin_inds_sample = None
        self.gamma_fit_bin_inds_all_occ = None
        self.gamma_fit_theta_init = None
        self.gamma_fit_result = None
        self.gamma_fit_theta = None # parameters for the gamma distribution, shape 'a' and scale


    def load(self, data_input):
        """docstring for ."""

        # validate the user's data_input
        self.validate_data_input(data_input)

        # initialise information from data_input
        self.data_variables = data_input['variables']
        self.data_type = data_input['data_type']
        self.data_time_values = data_input['time_values']
        self.data_basic_sigma = data_input['basic_sigma']

        # obtain the number of variables and time_values, respectively
        self.data_num_variables = len(self.data_variables)
        self.data_num_time_values = self.data_time_values.shape[0]

        # create indexing order for the data based on the data_variables list
        (self.data_mean_order,
        self.data_variance_order,
        self.data_covariance_order) = self.create_data_variable_order(self.data_variables)

        # dependent on data_type, load data as summary statistics or count data
        if self.data_type=='summary':
            self.data_mean = data_input['mean_data']
            self.data_variance = data_input['var_data']
            self.data_covariance = data_input['cov_data']

        # in case of count data, bootstrapping is used to compute the summary statistics
        elif self.data_type=='counts':
            self.data_counts = data_input['count_data']
            self.data_bootstrap_samples = data_input['bootstrap_samples']

            (self.data_mean,
            self.data_variance,
            self.data_covariance) = self.bootstrap_count_data_to_summary_stats(
                                                                self.data_num_time_values,
                                                                self.data_mean_order,
                                                                self.data_variance_order,
                                                                self.data_covariance_order,
                                                                self.data_counts,
                                                                self.data_bootstrap_samples)

        # add a basic_sigma value to all standard errors of the summary statistics that are lower than basic_sigma
        self.data_mean = self.introduce_basic_sigma(self.data_basic_sigma, self.data_mean)
        self.data_variance = self.introduce_basic_sigma(self.data_basic_sigma, self.data_variance)
        self.data_covariance = self.introduce_basic_sigma(self.data_basic_sigma, self.data_covariance)

        # obtain the number of summary data points
        self.data_num_values, self.data_num_values_mean_only = self.get_number_data_points(
                                                                        self.data_mean,
                                                                        self.data_variance,
                                                                        self.data_covariance)


    @staticmethod
    def create_data_variable_order(data_variables):
        """docstring for ."""

        # order of mean and variance indices just matches the data_variables order
        data_mean_order = [{'variables': var,  'summary_indices': i, 'count_indices': (i, )}
                                        for i, var in enumerate(data_variables)]
        data_variance_order = [{'variables': (var, var),  'summary_indices': i, 'count_indices': (i, i)}
                                        for i, var in enumerate(data_variables)]

        # data_covariance_order is ordered with a priority of the smaller index
        # (i.e., what comes first in data_variables)
        data_covariance_order = list()
        count = 0
        for i, var1 in enumerate(data_variables):
            for j, var2 in enumerate(data_variables):
                if i<j:
                    data_covariance_order.append({'variables': (var1, var2),  'summary_indices': count, 'count_indices': (i, j)})
                    count += 1

        return data_mean_order, data_variance_order, data_covariance_order


    def bootstrap_count_data_to_summary_stats(self,
                                                data_num_time_values,
                                                data_mean_order,
                                                data_variance_order,
                                                data_covariance_order,
                                                count_data,
                                                bootstrap_samples):
        """docstring for ."""

        # preallocate numpy arrays for summary statistics
        # the first axis has dimension two; to save statistic and standard error of that statistic
        data_mean = np.zeros((2, len(data_mean_order), data_num_time_values))
        data_var = np.zeros((2, len(data_variance_order), data_num_time_values))
        data_cov = np.zeros((2, len(data_covariance_order), data_num_time_values))

        # calculate mean statistics
        for variable_ind in range(len(data_mean_order)):
            for time_value_ind in range(data_num_time_values):
                data_mean[:, variable_ind, time_value_ind] = self.bootstrapping_mean(count_data[:, variable_ind, time_value_ind], bootstrap_samples)

        # calculate variance statistics
        for variable_ind in range(len(data_variance_order)):
            for time_value_ind in range(data_num_time_values):
                data_var[:, variable_ind, time_value_ind] = self.bootstrapping_variance(count_data[:, variable_ind, time_value_ind], bootstrap_samples)

        # calculate covariance statistics
        for variable_ind in range(len(data_covariance_order)):
            count_indices = data_covariance_order[variable_ind]['count_indices']
            for time_value_ind in range(data_num_time_values):
                data_cov[:, variable_ind, time_value_ind] = self.bootstrapping_covariance(
                                                                count_data[:, count_indices[0], time_value_ind],
                                                                count_data[:, count_indices[1], time_value_ind],
                                                                bootstrap_samples)

        # ###
        # plus_plus_cells = data[0, :, :]
        # plus_minus_cells = data[1, :, :]
        # minus_plus_cells = data[2, :, :]
        # minus_minus_cells = data[3, :, :]
        #
        # mean_plus_plus_cells = np.mean(plus_plus_cells, axis=0)
        # mean_plus_minus_cells = np.mean(plus_minus_cells, axis=0)
        # mean_minus_plus_cells = np.mean(minus_plus_cells, axis=0)
        # mean_minus_minus_cells = np.mean(minus_minus_cells, axis=0)
        #
        # means = [mean_plus_plus_cells, mean_plus_minus_cells, mean_minus_plus_cells, mean_minus_minus_cells]
        #
        # var_plus_plus_cells = np.var(plus_plus_cells, axis=0, ddof=0)
        # var_plus_minus_cells = np.var(plus_minus_cells, axis=0, ddof=0)
        # var_minus_plus_cells = np.var(minus_plus_cells, axis=0, ddof=0)
        # var_minus_minus_cells = np.var(minus_minus_cells, axis=0, ddof=0)
        #
        # CV_or_var = [var_plus_plus_cells, var_plus_minus_cells, var_minus_plus_cells, var_minus_minus_cells]
        #
        # timepoints = range(plus_plus_cells.shape[1])
        # cov_plus_plus_vs_plus_minus = np.array([np.cov(plus_plus_cells[:, timepoint], plus_minus_cells[:, timepoint], ddof=0)[1,0] for timepoint in timepoints])
        # cov_plus_plus_vs_minus_plus = np.array([np.cov(plus_plus_cells[:, timepoint], minus_plus_cells[:, timepoint], ddof=0)[1,0] for timepoint in timepoints])
        # cov_plus_plus_vs_minus_minus = np.array([np.cov(plus_plus_cells[:, timepoint], minus_minus_cells[:, timepoint], ddof=0)[1,0] for timepoint in timepoints])
        # cov_plus_minus_vs_minus_plus = np.array([np.cov(plus_minus_cells[:, timepoint], minus_plus_cells[:, timepoint], ddof=0)[1,0] for timepoint in timepoints])
        # cov_plus_minus_vs_minus_minus = np.array([np.cov(plus_minus_cells[:, timepoint], minus_minus_cells[:, timepoint], ddof=0)[1,0] for timepoint in timepoints])
        # cov_minus_plus_vs_minus_minus = np.array([np.cov(minus_plus_cells[:, timepoint], minus_minus_cells[:, timepoint], ddof=0)[1,0] for timepoint in timepoints])
        #
        # corr_or_cov = [cov_plus_plus_vs_plus_minus, cov_plus_plus_vs_minus_plus, cov_plus_plus_vs_minus_minus, cov_plus_minus_vs_minus_plus, cov_plus_minus_vs_minus_minus, cov_minus_plus_vs_minus_minus]
        # ###

        return (data_mean, data_var, data_cov)


    @staticmethod
    def bootstrapping_mean(sample, num_resamples):
        """docstring for ."""
        ### sample should be a one-dimensional, flat array

        ### calculate the statistic of the sample
        stat_sample = np.mean(sample)

        ### bootstrap the standard error of the sample statistic (se_stat_sample)
        # draw random number from sample with replacement
        resamples = np.random.choice(sample, size=(num_resamples, *sample.shape), replace=True)

        # calculate the statistic for each resample
        stat_resamples = np.mean(resamples, axis=1)

        # compute the standard error as standard deviation over all statistic resamples
        se_stat_sample = np.std(stat_resamples, ddof=1)

        return (stat_sample, se_stat_sample)


    @staticmethod
    def bootstrapping_variance(sample, num_resamples):
        """docstring for ."""
        ### sample should be a one-dimensional, flat array

        ### calculate the statistic of the sample
        stat_sample = np.var(sample, ddof=1)

        ### bootstrap the standard error of the sample statistic (se_stat_sample)
        # draw random number from sample with replacement
        resamples = np.random.choice(sample, size=(num_resamples, *sample.shape), replace=True)

        # calculate the statistic for each resample
        stat_resamples = np.var(resamples, axis=1, ddof=1)

        # compute the standard error as standard deviation over all statistic resamples
        se_stat_sample = np.std(stat_resamples, ddof=1)

        return (stat_sample, se_stat_sample)


    @staticmethod
    def bootstrapping_covariance(sample1, sample2, num_resamples):
        """docstring for ."""

        sample = np.array([sample1, sample2])

        ### calculate the statistic of the sample
        stat_sample = np.cov(sample[0, :], sample[1, :], ddof=1)[1,0]

        ### bootstrap the standard error of the sample statistic (se_stat_sample)
        # draw random number from sample with replacement
        resample_inds = [np.random.randint(sample.shape[1], size=sample.shape[1]) for i in range(num_resamples)]
        resamples = np.array([sample[:, ind] for ind in resample_inds])

        # calculate the statistic for each resample
        stat_resamples = np.zeros((num_resamples,))
        for resample_ind in range(num_resamples):
            stat_resamples[resample_ind] = np.cov(resamples[resample_ind, 0, :], resamples[resample_ind, 1, :], ddof=1)[0, 1]

        # compute the standard error as standard deviation over all statistic resamples
        se_stat_sample = np.std(stat_resamples, ddof=1)

        return (stat_sample, se_stat_sample)


    @staticmethod
    def introduce_basic_sigma(basic_sigma, data):
        """docstring for ."""

        ### this function handles zero-valued or almost zero-valued standard errors by
        ### introducing a basic_sigma standard error value for those entries;
        ### this is required to weight model/data difference by the standard errors
        ### (prevent division by zero or almost-zero values)

        # replaces all standard errors (data[1, :, :]) that are below the
        # basic_sigma value by basic_sigma;
        # this ensures that for each standard error holds: standard error >= basic_sigma
        data[1, data[1, :, :] <= basic_sigma] = basic_sigma
        return data


    @staticmethod
    def get_number_data_points(data_mean, data_var, data_cov):
        """docstring for ."""

        # calculate the number of data points along their last two dimensions
        # number of variables * number of time points
        data_points_mean = int(data_mean.shape[1] * data_mean.shape[2])
        data_points_var = int(data_var.shape[1] * data_var.shape[2])
        data_points_cov = int(data_cov.shape[1] * data_cov.shape[2])

        return data_points_mean + data_points_var + data_points_cov, data_points_mean


    def events_find_all(self):
        """docstring for ."""

        self.event_all_first_change_from_inital_conditions = [
                self.event_find_first_change_from_inital_conditions(self.data_counts[trace_ind, :, :], self.data_time_values)
                for trace_ind in range(self.data_counts.shape[0])
        ]

        self.event_all_first_cell_count_increase = [
                self.event_find_first_cell_count_increase(self.data_counts[trace_ind, :, :], self.data_time_values)
                for trace_ind in range(self.data_counts.shape[0])
        ]

        self.event_all_first_cell_type_conversion = [
                self.event_find_first_cell_type_conversion(self.data_counts[trace_ind, :, :], self.data_time_values)
                for trace_ind in range(self.data_counts.shape[0])
        ]

        self.event_all_first_cell_count_increase_after_cell_type_conversion = [
                self.event_find_first_cell_count_increase_after_cell_type_conversion(self.data_counts[trace_ind, :, :], self.data_time_values)
                for trace_ind in range(self.data_counts.shape[0])
        ]

        self.event_all_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion = [
                self.event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(self.data_counts[trace_ind, :, :], self.data_time_values)
                for trace_ind in range(self.data_counts.shape[0])
        ]

        self.event_all_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion = [
                self.event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion(self.data_counts[trace_ind, :, :], self.data_time_values)
                for trace_ind in range(self.data_counts.shape[0])
        ]

    ### single-well event functions
    # well event functions search for the first occurance of an event
    # return True or False whether event is found, if True with an event waiting time tau else tau=None

    # IMPORTANT NOTE: the cell count increase functions treat an increase by two cells as the same event
    # it will also look for 'any increase of cell numbers between two time points'
    # THIS MEANS in particular that the 'second increase' event will not yield tau=0 but looks for the third cell increase
    # working with the 'first', 'second', 'third' increase functions is thus not strictly representative of actual cell numbers

    # NOTE: we have currently no event function for backwards conversion (there is one well for this)

    def event_find_first_change_from_inital_conditions(self, well_trace, time_values):
        """docstring for ."""

        # initial setting that event did not happen
        event_bool = False
        event_tau = None

        # get initial conditions
        init_cond = well_trace[:, 0]

        # loop over well_trace and get waiting time tau if event happens
        for time_ind in range(well_trace.shape[1]):
            if np.any(well_trace[:, time_ind]!=init_cond):
                event_bool = True
                event_tau = time_values[time_ind]
                break

        return event_bool, event_tau


    def event_find_first_cell_count_increase(self, well_trace, time_values):
        """docstring for ."""

        ### here we check for the first increase in TOTAL cell numbers
        ### (not an increase in any individual cell type population)

        # initial setting that event did not happen
        event_bool = False
        event_tau = None

        # first, get the sum of all cell types (along axis=0)
        well_trace_sum = np.sum(well_trace, axis=0)

        # get initial conditions
        init_cond_sum = well_trace_sum[0]

        # loop over well_trace and get waiting time tau if event happens
        for time_ind in range(well_trace_sum.shape[0]):
            if well_trace_sum[time_ind] > init_cond_sum:
                event_bool = True
                event_tau = time_values[time_ind]
                break

        return event_bool, event_tau


    def event_find_first_cell_type_conversion(self, well_trace, time_values):
        """docstring for ."""

        ### here we check for any event with a change in state space
        ### but maintenance of the total cell numbers
        ### (since we exclude cell death from happening)

        # initial setting that event did not happen
        event_bool = False
        event_tau = None

        # also get the sum of all cell types (along axis=0)
        well_trace_sum = np.sum(well_trace, axis=0)

        # loop over well_trace and get waiting time tau if event happens
        for time_ind in range(1, well_trace.shape[1]):
            if (np.any(well_trace[:, time_ind]!=well_trace[:, time_ind - 1])
                and well_trace_sum[time_ind]==well_trace_sum[time_ind - 1]):

                event_bool = True
                event_tau = time_values[time_ind]
                break

        return event_bool, event_tau


    def event_find_first_cell_count_increase_after_cell_type_conversion(self, well_trace, time_values, diff=True):
        """docstring for ."""

        ### this event is checked by the sequential use of the event
        ### functions 'conversion' and 'cell count increase'

        # initial setting that event did not happen
        event_bool = False
        event_tau = None

        # first, check for cell type conversion
        event_bool_conv, event_tau_conv = self.event_find_first_cell_type_conversion(well_trace, time_values)

        # in case of conversion proceed with the following
        if event_bool_conv:

            # get time index of conversion
            time_ind_conv = np.nonzero(time_values == event_tau_conv)[0][0]

            # shorten the well_trace and time_values starting with conv event
            well_trace_shortened = well_trace[:, time_ind_conv:]
            time_values_shortened = time_values[time_ind_conv:]

            # now check for cell count increase of the shortened trace
            # (yielding the overall event information)
            event_bool, event_tau = self.event_find_first_cell_count_increase(well_trace_shortened, time_values_shortened)

            # if difference shall be computed, we are interested in the waiting starting with the conditional event, thus
            if event_bool and diff:
                event_tau = event_tau - event_tau_conv

        return event_bool, event_tau


    def event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(self, well_trace, time_values, diff=True):
        """docstring for ."""

        # initial setting that event did not happen
        event_bool = False
        event_tau = None

        # check if the conditional event happened (first cell count increase after conversion)
        event_bool_conditional, event_tau_conditional = self.event_find_first_cell_count_increase_after_cell_type_conversion(well_trace, time_values, diff=False)

        # in case that the conditional event happened, proceed with:
        if event_bool_conditional:

            # get time index of conditional event
            time_ind_conditional = np.nonzero(time_values == event_tau_conditional)[0][0]

            # shorten the well_trace and time_values starting with conditional event
            well_trace_shortened = well_trace[:, time_ind_conditional:]
            time_values_shortened = time_values[time_ind_conditional:]

            # now check for cell count increase of the shortened trace
            # (yielding the overall event information)
            event_bool, event_tau = self.event_find_first_cell_count_increase(well_trace_shortened, time_values_shortened)

            # if difference shall be computed, we are interested in the waiting starting with the conditional event, thus
            if event_bool and diff:
                event_tau = event_tau - event_tau_conditional

        return event_bool, event_tau


    def event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion(self, well_trace, time_values, diff=True):
        """docstring for ."""

        # initial setting that event did not happen
        event_bool = False
        event_tau = None

        # check if the conditional event happened (first and second cell count increase after conversion)
        event_bool_conditional, event_tau_conditional = self.event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(well_trace, time_values, diff=False)

        # in case that the conditional event happened, proceed with:
        if event_bool_conditional:

            # get time index of conditional event
            time_ind_conditional = np.nonzero(time_values == event_tau_conditional)[0][0]

            # shorten the well_trace and time_values starting with conditional event
            well_trace_shortened = well_trace[:, time_ind_conditional:]
            time_values_shortened = time_values[time_ind_conditional:]

            # now check for cell count increase of the shortened trace
            # (yielding the overall event information)
            event_bool, event_tau = self.event_find_first_cell_count_increase(well_trace_shortened, time_values_shortened)

            # if difference shall be computed, we are interested in the waiting starting with the conditional event, thus
            if event_bool and diff:
                event_tau = event_tau - event_tau_conditional

        return event_bool, event_tau
    ###

    ### methods for fitting binned waiting times with the Gamma distribution
    def gamma_fit_binned_waiting_times(self, waiting_times_arr):
        """docstring for ."""

        ### IDEA: for a given Gamma distribution the probability to find a
        ### drawn waiting time within an interval (a, b] is given by the cumulative
        ### Gamma distribution;
        ### we can calculate this probability for all bins individually and can
        ### obtain a likelihood of a measured binned distribution by comparing the
        ### bin probabilities with the measured bin frequencies;
        ### this is a multinomial likelihood

        # given times_values = [0, 2, ..., 54],
        # bins are defined as (-inf, 0], (0, 2], ..., (52, 54], (54, inf)
        # 29 bins in total
        # variable bins is = [-inf, 0, 2, ..., 52, 54, inf] (len = 30)
        self.gamma_fit_bins = np.concatenate(([-np.inf], self.data_time_values, [np.inf]))
        # print(len(bins))
        # print(bins)

        # bin indices are then as follows
        # 0: (-inf, 0], 1: (0, 2], ..., 27: (52, 54], 28: (54, inf)
        self.gamma_fit_bin_inds_sample = np.digitize(waiting_times_arr, self.gamma_fit_bins, right=True) - 1
        # print(len(bin_inds_sample))
        # print(bin_inds_sample)

        # all bin indices then go from 0 to 28
        bin_inds_all = np.arange(len(self.gamma_fit_bins) - 1)
        # print(len(bin_inds_all))
        # print(bin_inds_all)

        # count the occurences of data points in each bin
        self.gamma_fit_bin_inds_all_occ = np.array([np.count_nonzero(self.gamma_fit_bin_inds_sample==bin_ind) for bin_ind in bin_inds_all])
        # print(np.sum(bin_inds_all_occ))
        # print(len(bin_inds_all_occ))
        # print(bin_inds_all_occ)

        # compute a rough estimation of theta as an initial theta for the optimisation
        var_init = np.var(waiting_times_arr)
        mean_init = np.mean(waiting_times_arr)
        self.gamma_fit_theta_init = [mean_init**2/var_init, var_init/mean_init]
        # print(self.gamma_fit_theta_init)

        # optimise the multinomial log likelihood to find theta
        self.gamma_fit_result = optimize.minimize(self.negative_multinomial_log_likelihood,
                                                    self.gamma_fit_theta_init,
                                                    method='L-BFGS-B',
                                                    bounds=[(0, None)]*len(self.gamma_fit_theta_init))
        self.gamma_fit_theta = self.gamma_fit_result['x']


    def compute_bin_probabilities(self, theta):
        """docstring for ."""

        # the probability to be in bin (a, b] are given by prob(theta) = Gamma_cdf_theta(b) - Gamma_cdf_theta(a)
        # F(time_values[1:]) - F(time_values[:-1]) achieves bin-wise calculation of
        # bin probs by prob(theta) = F(b) - F(a), with F Gamma CDF for a given theta
        bins_probs = stats.gamma.cdf(self.gamma_fit_bins[1:], a=theta[0], loc=0, scale=theta[1]) - stats.gamma.cdf(self.gamma_fit_bins[:-1], a=theta[0], loc=0, scale=theta[1])
        return bins_probs


    def negative_multinomial_log_likelihood(self, theta):
        """docstring for ."""

        # calculate the bin probabilities for a given theta = (shape, scale)
        bin_probs = self.compute_bin_probabilities(theta)

        # use a multinomial model to compute the log likelihood of observing the data (counts in each bin) for the given bins probs
        log_likelihood = stats.multinomial.logpmf(self.gamma_fit_bin_inds_all_occ, n=np.sum(self.gamma_fit_bin_inds_all_occ), p=bin_probs)
        return - log_likelihood
    ###

    ### plotting helper functions
    def event_percentages(self, settings):
        """docstring for ."""

        y_list_err = list()
        x_ticks = list()
        attributes = dict()

        for i, event_dict in enumerate(settings):
            event_results = event_dict['event']
            event_perc = 100.0 * (sum([event_bool for event_bool, __ in event_results])/float(len(event_results)))
            y_list_err.append([event_perc])

            x_ticks.append(event_dict['label'])

            attributes[i] = (None, event_dict['color'])

        y_arr_err = np.array(y_list_err)
        return y_arr_err, x_ticks, attributes

    def scatter_at_time_point(self, variable1, variable2, time_ind, settings):
        """docstring for ."""

        var_ind_x = self.data_variables.index(variable1)
        var_ind_y = self.data_variables.index(variable2)

        x_arr = self.data_counts[:, var_ind_x, time_ind]
        y_arr = self.data_counts[:, var_ind_y, time_ind]

        attributes = dict()
        attributes['color'] = settings['color']
        attributes['opacity'] = settings['opacity']
        attributes['label'] = settings['label']

        return x_arr, y_arr, attributes

    def histogram_continuous_event_waiting_times(self, event_results, settings):
        """docstring for ."""

        bar_attributes = dict()
        bar_list = list()

        tau_list = [event_tau for event_bool, event_tau in event_results if event_bool]
        bar_arr = np.array(tau_list).reshape(len(tau_list), 1)

        bar_attributes[0] = {   'label': settings['label'],
                                'color': settings['color'],
                                'opacity': settings['opacity'],
                                'edges': self.data_time_values,
                                'interval_type': '(]'
                                }

        return bar_arr, bar_attributes

    def histogram_continuous_event_waiting_times_w_gamma_fit(self, event_results, settings):
        """docstring for ."""

        bar_attributes = dict()
        bar_list = list()

        tau_list = [event_tau for event_bool, event_tau in event_results if event_bool]
        bar_arr = np.array(tau_list).reshape(len(tau_list), 1)

        bar_attributes[0] = {   'label': settings['label'],
                                'color': settings['color'],
                                'opacity': settings['opacity'],
                                'edges': self.data_time_values,
                                'interval_type': '(]'
                                }

        def gamma_fit_func(x_line_arr, bar_arr):
            # compute Gamma distr. fit
            self.gamma_fit_binned_waiting_times(bar_arr)
            gamma_fit_shape, gamma_fit_scale = self.gamma_fit_theta
            print('gamma_fit_shape: ', gamma_fit_shape, '\n',
                    'gamma_fit_scale: ', gamma_fit_scale)

            y_line_arr = stats.gamma.pdf(x_line_arr, a=gamma_fit_shape, loc=0.0, scale=gamma_fit_scale)

            # # KS test for if data follows Gamma distr.
            # ks_stat, pval = stats.kstest(bar_arr, 'gamma', args=(fit_alpha, fit_loc, fit_beta))

            return x_line_arr, y_line_arr, f'$\Gamma$($n$={round(gamma_fit_shape, 1)}, $θ$={round(gamma_fit_shape * gamma_fit_scale, 1)})', settings['gamma_color'] # , KS $p$-value {round(pval, 2)}

        return bar_arr, bar_attributes, gamma_fit_func

    def histogram_discrete_cell_counts_at_time_point(self, time_ind, settings):
        """docstring for ."""

        bar_arr = self.data_counts[:, :, time_ind]
        bar_attributes = dict()

        for i, var in enumerate(self.data_variables):
            bar_attributes[i] = {   'label': settings[var]['label'],
                                    'color': settings[var]['color'],
                                    'opacity': settings[var]['opacity']}

        return bar_arr, bar_attributes

    def dots_w_bars_evolv_mean(self, settings):
        """docstring for ."""

        x_arr = self.data_time_values
        y_arr = np.zeros((len(self.data_mean_order), self.data_num_time_values, 2))
        attributes = dict()

        for var_inf in self.data_mean_order:
            variable = var_inf['variables']
            i = var_inf['summary_indices']

            y_arr[i, :, 0] = self.data_mean[0, i, :] # mean statistic
            y_arr[i, :, 1] = self.data_mean[1, i, :] # standard error

            node_settings = settings[variable]
            attributes[i] = (node_settings['label'], node_settings['color'])

        return x_arr, y_arr, attributes

    def dots_w_bars_evolv_variance(self, settings):
        """docstring for ."""

        x_arr = self.data_time_values
        y_arr = np.zeros((len(self.data_variance_order), self.data_num_time_values, 2))
        attributes = dict()

        for var_inf in self.data_variance_order:
            variable = var_inf['variables']
            i = var_inf['summary_indices']

            y_arr[i, :, 0] = self.data_variance[0, i, :] # var statistic
            y_arr[i, :, 1] = self.data_variance[1, i, :] # standard error

            node_settings = settings[variable]
            attributes[i] = (node_settings['label'], node_settings['color'])

        return x_arr, y_arr, attributes

    def dots_w_bars_evolv_covariance(self, settings):
        """docstring for ."""

        x_arr = self.data_time_values
        y_arr = np.zeros((len(self.data_covariance_order), self.data_num_time_values, 2))
        attributes = dict()

        for var_inf in self.data_covariance_order:
            variable = var_inf['variables']
            i = var_inf['summary_indices']

            y_arr[i, :, 0] = self.data_covariance[0, i, :] # cov statistic
            y_arr[i, :, 1] = self.data_covariance[1, i, :] # standard error

            try:
                node_settings = settings[variable]
            except:
                node_settings = settings[(variable[1], variable[0])]

            attributes[i] = (node_settings['label'], node_settings['color'])

        return x_arr, y_arr, attributes
    ###

    @staticmethod
    def validate_data_input(data_input):
        """docstring for ."""
        # TODO: there could be more validation: check the shapes of all data-related inputs

        # check general data_input dictionary structure
        if isinstance(data_input, dict):
            if set(data_input.keys()) == set(['variables', 'data_type', 'time_values', 'mean_data', 'var_data', 'cov_data', 'count_data', 'bootstrap_samples', 'basic_sigma']):
                pass
            else:
                raise ValueError('Method load() expects a dictionary for the data_input with keys \'variables\', \'data_type\', \'time_values\', \'mean_data\', \'var_data\', \'cov_data\', \'count_data\', \'bootstrap_samples\' and \'basic_sigma\' (as strings).')
        else:
            raise TypeError('Method load() expects a dictionary as variable for the data_input.')

        # check data_input variables
        if isinstance(data_input['variables'], list):
            if all(isinstance(var, str) for var in data_input['variables']):
                pass
            else:
                raise TypeError('List with string items expected as value of key \'variables\' of dictionary data_input.')
        else:
            raise TypeError('List expected as value of key \'variables\' of dictionary data_input.')

        # check data_input type
        if isinstance(data_input['data_type'], str):
            if data_input['data_type']=='summary' or data_input['data_type']=='counts':
                pass
            else:
                raise ValueError('String \'summary\' or \'counts\' expected as value of key \'data_type\' of dictionary data_input.')
        else:
            raise TypeError('String expected as value of key \'data_type\' of dictionary data_input.')

        # check data_input time values
        if isinstance(data_input['time_values'], np.ndarray):
            pass
        else:
            raise TypeError('Numpy array expected as value of key \'time_values\' of dictionary data_input.')

        # check data_input mean_data
        if isinstance(data_input['mean_data'], np.ndarray):
            pass
        else:
            raise TypeError('Numpy array expected as value of key \'mean_data\' of dictionary data_input.')

        # check data_input var_data
        if isinstance(data_input['var_data'], np.ndarray):
            pass
        else:
            raise TypeError('Numpy array expected as value of key \'var_data\' of dictionary data_input.')

        # check data_input cov_data
        if isinstance(data_input['cov_data'], np.ndarray):
            pass
        else:
            raise TypeError('Numpy array expected as value of key \'cov_data\' of dictionary data_input.')

        # check data_input count_data
        if isinstance(data_input['count_data'], np.ndarray):
            pass
        else:
            raise TypeError('Numpy array expected as value of key \'count_data\' of dictionary data_input.')

        # check data_input bootstrap_samples
        if isinstance(data_input['bootstrap_samples'], int):
            pass
        else:
            raise TypeError('Integer expected as value of key \'bootstrap_samples\' of dictionary data_input.')

        # check data_input basic_sigma
        if isinstance(data_input['basic_sigma'], float):
            pass
        else:
            raise TypeError('Float value expected as value of key \'basic_sigma\' of dictionary data_input.')



    # ##### TODO: old / adapt
    # """
    # PART V: load the data and match it to the model
    #
    # --- information ---
    # - data has to have shape = (#types, #variables, #time_points), e.g. = (2, 4, 72)
    # where type can be a statistic or standard error of that statistic
    #
    # - data has to have the order in the variables as specified in the following
    # mean: CD27plus_CD62Lplus, CD27plus_CD62Lminus, CD27minus_CD62Lminus, CD27minus_CD62Lplus
    # variance: CD27plus_CD62Lplus, CD27plus_CD62Lminus, CD27minus_CD62Lminus, CD27minus_CD62Lplus
    # covariance: +|+ vs +|-, +|+ vs -|-, +|+ vs -|+, +|- vs -|-, +|- vs -|+, -|- vs -|+
    # """
    #
    # ### main function
    # def load_data(self, data_model_relation, mean_data, var_data, cov_data, save=True, plot=True):
    #     # resort data, so that model variables (nodes) match the data variables
    #     # the order of the model is given by self.moment_order
    #     data_model_relation = self.data_rel_to_str(data_model_relation)
    #     self.data_mean, self.model_names_mean = self.match_data_to_model_mean((2, self.num_means, self.num_time_points), self.moment_order_main, data_model_relation, mean_data)
    #     self.data_var, self.model_names_var = self.match_data_to_model_var((2, self.num_vars, self.num_time_points), self.moment_order_main, data_model_relation, var_data)
    #     self.data_cov, self.model_names_cov = self.match_data_to_model_cov((2, self.num_covs, self.num_time_points), self.moment_order_main, data_model_relation, cov_data)
    #
    #     # as a preparation for plots (data or predictions) define a fixed color code for the nodes
    #     self.model_color_code = self.set_fixed_color_code_for_nodes(self.model_names_mean, self.model_names_var, self.model_names_cov)
    #
    #     # save information on data (and data-model-relation) if save==True
    #     if save:
    #         self.save_data_variables(data_model_relation, (self.model_names_mean, self.model_names_var, self.model_names_cov), self.data_path, self.run_name)
    #
    #     # plot the data used for fitting (and only this data; might be smaller than the whole data set) if plot==True
    #     if plot:
    #         self.plot_data(self.model_time_array, self.data_mean, self.data_var, self.data_cov, self.model_color_code, self.data_path, self.run_name)
    #
    # ### helper functions
    # # NOTE:
    # # the model gives the structure
    # # the model captures the data it needs, not the other way round
    #
    # def data_rel_to_str(self, rel):
    #     new_rel = list()
    #     for tup in rel:
    #         new_rel.append((str(tup[0]), tup[1]))
    #     return new_rel
    #
    # def match_data_to_model_mean(self, shape, moment_order, rel, mean_data):
    #     # create an array with zeros
    #     mean_data_sort = np.zeros(shape)
    #     model_names_mean = list()
    #
    #     count = 0
    #     for tup in moment_order:
    #         if len(tup)==1: # this assures that we have the mean of a node
    #             for node, data_variable in rel:
    #                 if node==tup[0]:
    #                     # for the CD27/CD62L data sets
    #                     if data_variable=='CD27plus_CD62Lplus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 0, :]
    #                         model_names_mean.append('CD27+ CD62L+')
    #                         count += 1
    #                     elif data_variable=='CD27plus_CD62Lminus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 1, :]
    #                         model_names_mean.append('CD27+ CD62L-')
    #                         count += 1
    #                     elif data_variable=='CD27minus_CD62Lminus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 2, :]
    #                         model_names_mean.append('CD27- CD62L-')
    #                         count += 1
    #                     elif data_variable=='CD27minus_CD62Lplus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 3, :]
    #                         model_names_mean.append('CD27- CD62L+')
    #                         count += 1
    #
    #                     # for the CD44 only data sets
    #                     elif data_variable=='CD44plus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 0, :]
    #                         model_names_mean.append('CD44+')
    #                         count += 1
    #                     elif data_variable=='CD44minus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 1, :]
    #                         model_names_mean.append('CD44-')
    #                         count += 1
    #
    #                     # for the CD62L only data sets
    #                     elif data_variable=='CD62Lplus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 0, :]
    #                         model_names_mean.append('CD62L+')
    #                         count += 1
    #                     elif data_variable=='CD62Lminus':
    #                         mean_data_sort[:, count, :] = mean_data[:, 1, :]
    #                         model_names_mean.append('CD62L-')
    #                         count += 1
    #
    #                     # TODO: for the CD44/CD62L data sets
    #                     # here
    #
    #     return mean_data_sort, model_names_mean
    #
    # def match_data_to_model_var(self, shape, moment_order, rel, var_data):
    #     # create an array with zeros
    #     var_data_sort = np.zeros(shape)
    #     model_names_var = list()
    #
    #     count = 0
    #     for tup in moment_order:
    #         if len(tup)==2: # this assures that we have the variance of a node
    #             if tup[0]==tup[1]: # and this
    #                 for node, data_variable in rel:
    #                     if node==tup[0]:
    #                         # for the CD27/CD62L data sets
    #                         if data_variable=='CD27plus_CD62Lplus':
    #                             var_data_sort[:, count, :] = var_data[:, 0, :]
    #                             model_names_var.append('CD27+ CD62L+')
    #                             count += 1
    #                         elif data_variable=='CD27plus_CD62Lminus':
    #                             var_data_sort[:, count, :] = var_data[:, 1, :]
    #                             model_names_var.append('CD27+ CD62L-')
    #                             count += 1
    #                         elif data_variable=='CD27minus_CD62Lminus':
    #                             var_data_sort[:, count, :] = var_data[:, 2, :]
    #                             model_names_var.append('CD27- CD62L-')
    #                             count += 1
    #                         elif data_variable=='CD27minus_CD62Lplus':
    #                             var_data_sort[:, count, :] = var_data[:, 3, :]
    #                             model_names_var.append('CD27- CD62L+')
    #                             count += 1
    #
    #                         # for the CD44 only data sets
    #                         elif data_variable=='CD44plus':
    #                             var_data_sort[:, count, :] = var_data[:, 0, :]
    #                             model_names_var.append('CD44+')
    #                             count += 1
    #                         elif data_variable=='CD44minus':
    #                             var_data_sort[:, count, :] = var_data[:, 1, :]
    #                             model_names_var.append('CD44-')
    #                             count += 1
    #
    #                         # for the CD62L only data sets
    #                         elif data_variable=='CD62Lplus':
    #                             var_data_sort[:, count, :] = var_data[:, 0, :]
    #                             model_names_var.append('CD62L+')
    #                             count += 1
    #                         elif data_variable=='CD62Lminus':
    #                             var_data_sort[:, count, :] = var_data[:, 1, :]
    #                             model_names_var.append('CD62L-')
    #                             count += 1
    #
    #                         # TODO: for the CD44/CD62L data sets
    #                         # here
    #
    #     return var_data_sort, model_names_var
    #
    # def match_data_to_model_cov(self, shape, moment_order, rel, cov_data):
    #     # create an array with zeros
    #     cov_data_sort = np.zeros(shape)
    #     model_names_cov = list()
    #
    #     count = 0
    #     for tup in moment_order:
    #         if len(tup)==2: # this assures that we have the covariance of a node
    #             if tup[0]!=tup[1]: # and this
    #                 for node1, dv1 in rel:
    #                     for node2, dv2 in rel:
    #                         if node1==tup[0] and node2==tup[1]:
    #                             # for the CD27/CD62L data sets
    #                             # +|+ vs +|-
    #                             if (dv1=='CD27plus_CD62Lplus' and dv2=='CD27plus_CD62Lminus') or (dv1=='CD27plus_CD62Lminus' and dv2=='CD27plus_CD62Lplus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 0, :]
    #                                 model_names_cov.append('+|+ vs +|-')
    #                                 count += 1
    #                             # +|+ vs -|-
    #                             elif (dv1=='CD27plus_CD62Lplus' and dv2=='CD27minus_CD62Lminus') or (dv1=='CD27minus_CD62Lminus' and dv2=='CD27plus_CD62Lplus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 1, :]
    #                                 model_names_cov.append('+|+ vs -|-')
    #                                 count += 1
    #                             # +|+ vs -|+
    #                             elif (dv1=='CD27plus_CD62Lplus' and dv2=='CD27minus_CD62Lplus') or (dv1=='CD27minus_CD62Lplus' and dv2=='CD27plus_CD62Lplus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 2, :]
    #                                 model_names_cov.append('+|+ vs -|+')
    #                                 count += 1
    #                             # +|- vs -|-
    #                             elif (dv1=='CD27plus_CD62Lminus' and dv2=='CD27minus_CD62Lminus') or (dv1=='CD27minus_CD62Lminus' and dv2=='CD27plus_CD62Lminus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 3, :]
    #                                 model_names_cov.append('+|- vs -|-')
    #                                 count += 1
    #                             # +|- vs -|+
    #                             elif (dv1=='CD27plus_CD62Lminus' and dv2=='CD27minus_CD62Lplus') or (dv1=='CD27minus_CD62Lplus' and dv2=='CD27plus_CD62Lminus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 4, :]
    #                                 model_names_cov.append('+|- vs -|+')
    #                                 count += 1
    #                             # -|- vs -|+
    #                             elif (dv1=='CD27minus_CD62Lminus' and dv2=='CD27minus_CD62Lplus') or (dv1=='CD27minus_CD62Lplus' and dv2=='CD27minus_CD62Lminus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 5, :]
    #                                 model_names_cov.append('-|- vs -|+')
    #                                 count += 1
    #
    #                             # for the CD44 only data sets
    #                             elif (dv1=='CD44plus' and dv2=='CD44minus') or (dv1=='CD44minus' and dv2=='CD44plus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 0, :]
    #                                 model_names_cov.append('CD44+ vs CD44-')
    #                                 count += 1
    #
    #                             # for the CD62L only data sets
    #                             elif (dv1=='CD62Lplus' and dv2=='CD62Lminus') or (dv1=='CD62Lminus' and dv2=='CD62Lplus'):
    #                                 cov_data_sort[:, count, :] = cov_data[:, 0, :]
    #                                 model_names_cov.append('CD62L+ vs CD62L-')
    #                                 count += 1
    #
    #                             # TODO: for the CD44/CD62L data sets
    #                             # here
    #
    #     return cov_data_sort, model_names_cov
    #
    # def set_fixed_color_code_for_nodes(self, model_names_mean, model_names_var, model_names_cov):
    #     # create list of tuples with the name of the mean/var/cov and a corresponding fixed color for plots later
    #     model_names_color_mean = list()
    #     model_names_color_var = list()
    #     model_names_color_cov = list()
    #
    #     # means
    #     for name in model_names_mean:
    #         if name=='CD27+ CD62L+':
    #             color = 'gold'
    #             marker = 'o'
    #         elif name=='CD27+ CD62L-':
    #             color = 'green'
    #             marker = 'v'
    #         elif name=='CD27- CD62L-':
    #             color = 'gray'
    #             marker = 'X'
    #         elif name=='CD27- CD62L+':
    #             color = 'red'
    #             marker = 's'
    #         elif name=='CD44+':
    #             color = 'limegreen'
    #             marker = 'o'
    #         elif name=='CD44-':
    #             color = 'darkgreen'
    #             marker = 's'
    #         elif name=='CD62L+':
    #             color = 'red'
    #             marker = 'o'
    #         elif name=='CD62L-':
    #             color = 'darkred'
    #             marker = 's'
    #         model_names_color_mean.append((name, color, marker))
    #
    #     # variances
    #     for name in model_names_var:
    #         if name=='CD27+ CD62L+':
    #             color = 'gold'
    #             marker = 'o'
    #         elif name=='CD27+ CD62L-':
    #             color = 'green'
    #             marker = 'v'
    #         elif name=='CD27- CD62L-':
    #             color = 'gray'
    #             marker = 'X'
    #         elif name=='CD27- CD62L+':
    #             color = 'red'
    #             marker = 's'
    #         elif name=='CD44+':
    #             color = 'limegreen'
    #             marker = 'o'
    #         elif name=='CD44-':
    #             color = 'darkgreen'
    #             marker = 's'
    #         elif name=='CD62L+':
    #             color = 'red'
    #             marker = 'o'
    #         elif name=='CD62L-':
    #             color = 'darkred'
    #             marker = 's'
    #         model_names_color_var.append((name, color, marker))
    #
    #     # covariances
    #     for name in model_names_cov:
    #         if name=='+|+ vs +|-':
    #             color = 'dodgerblue'
    #         elif name=='+|+ vs -|-':
    #             color = 'limegreen'
    #         elif name=='+|+ vs -|+':
    #             color = 'darkorange'
    #         elif name=='+|- vs -|-':
    #             color = 'yellowgreen'
    #         elif name=='+|- vs -|+':
    #             color = 'tomato'
    #         elif name=='-|- vs -|+':
    #             color = 'deepskyblue'
    #         elif name=='CD44+ vs CD44-':
    #             color = 'deepskyblue'
    #         elif name=='CD62L+ vs CD62L-':
    #             color = 'deepskyblue'
    #         model_names_color_cov.append((name, color, '.'))
    #
    #     return (model_names_color_mean, model_names_color_var, model_names_color_cov)
    # #####
