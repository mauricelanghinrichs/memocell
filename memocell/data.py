
"""
The data module contains the Data class to handle and load data which subsequently
can be used for statistical inference.
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize

class Data(object):
    """Class to handle and load data.

    Main method is `load` which will create a ready-to-use
    data instance with dynamic summary statistics
    that can subsequently be used for the statistical inference; for the
    typical use case this is the only method to call. `load` is a wrapper method
    for most of the other class methods, so more documentation can be found in the
    respective individual methods. Further functionalities of this class
    include stochastic event analysis and Gamma/Erlang fitting to waiting time distributions.

    Parameters
    ----------
    data_name : str
        A name for the data object.

    Returns
    -------
    data : memocell.data.Data
        Initialised memocell data object. Typically, continue with the `data.load`
        method to add the actual data information.

    Examples
    --------
    >>> # initialise data
    >>> import memocell as me
    >>> import numpy as np
    >>> data = me.Data('my_data')
    >>> # use load() to fill it
    >>> # data.load(...)
    """

    def __init__(self, data_name):

        # initialise the name of this data set
        self.data_name = data_name

        # instantiate an object for the nodes/variables of this data
        self.data_variables = None
        self.data_num_variables = None

        # instantiate an object for time points of this data
        self.data_time_values = None
        self.data_num_time_values = None

        # instantiate object to indicate whether this data is limited
        # to mean summary statistics
        self.data_mean_exists_only = None

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


    def load(self, variables, time_values, count_data, data_type='counts',
                    mean_data=None, var_data=None, cov_data=None,
                    bootstrap_samples=10000, basic_sigma=0.0):
        """Main method of the data class. Method will load data either from dynamic
        count data (`data_type='counts'`, default) or from already summarised data as
        mean, variance and covariance statistics (`data_type='summary'`). Depending on the
        selected data type, different inputs are used: 1) `variables`, `time_values`
        and `basic_sigma` in both cases 2) `data_type='counts'` additionally uses
        `count_data` and `bootstrap_samples` 3) `data_type='summary'` additionally uses
        `mean_data`, `var_data` and `cov_data` (to load `mean_data` only is also supported).
        `load` defines many data attributes, the resulting summary statistics
        particularly are accessible via `data.data_mean`,
        `data.data_variance` and `data.data_covariance`. For more information on how
        the order of the summary statistics corresponds to `variables`, see
        method `data.create_data_variable_order`.


        Parameters
        ----------
        variables : list of str
            A list of strings for the data variables.
        time_values : 1d numpy.ndarray
            Time values corresponding to time dimension of `count_data` (in case of
            `data_type='counts'`, default) or of `mean_data`, `var_data` and `cov_data` (in case of
            `data_type='summary'`).
        count_data : numpy.ndarray or None
            Required input for `data_type='counts'` (default); data object will be
            computed based on `count_data` to get summary statistics `data_mean`,
            `data_variance` and `data_covariance` including standard errors
            by bootstrapping.
            Shape of `count_data` has to be (`n`, `m`, `t`), with the
            number of repeats `n`, the number of variables `m`, the number of time points `t`.
            Order of the variables should match with `variables`.
        data_type : str, optional
            String to define the mode how to create the data object;
            either `'counts'` (default) or `'summary'`.
        mean_data : numpy.ndarray or None, optional
            Required input for `data_type='summary'`; `mean_data` will be directly
            loaded into `data.data_mean`.
            `mean_data` contains the dynamic mean statistics and standard
            errors with shape (2, `len(data_mean_order)`, `len(time_values)`).
            `mean_data[0, :, :]` contains the statistics;
            `mean_data[1, :, :]` contains the standard errors.
        var_data : numpy.ndarray or None, optional
            Optional input for `data_type='summary'`; `var_data` will be directly
            loaded into `data.data_variance`.
            `var_data` contains the dynamic variance statistics and standard
            errors with shape (2, `len(data_variance_order)`, `len(time_values)`).
            `var_data[0, :, :]` contains the statistics;
            `var_data[1, :, :]` contains the standard errors.
        cov_data : numpy.ndarray or None, optional
            Optional input for `data_type='summary'`; `cov_data` will be directly
            loaded into `data.data_covariance`.
            `cov_data` contains the dynamic covariance statistics and standard
            errors with shape (2, `len(data_covariance_order)`, `len(time_values)`).
            `cov_data[0, :, :]` contains the statistics;
            `cov_data[1, :, :]` contains the standard errors.
        bootstrap_samples : int, optional
            Integer to set the number of bootstrap samples (in case of
            `data_type='counts'`, default). Number between
            `bootstrap_samples=10000` (default) and `bootstrap_samples=100000`
            is typically sufficient. For more information, see
            method `data.bootstrap_count_data_to_summary_stats` and methods therein.
        basic_sigma : float, optional
            Non-negative float value for `basic_sigma`. For more information, see
            method `data.introduce_basic_sigma`.

        Returns
        -------
        None


        Examples
        --------
        >>> import memocell as me
        >>> import numpy as np
        >>> data = me.Data('my_data')
        >>> variables = ['A', 'B']
        >>> time_values = np.linspace(0.0, 4.0, num=5)
        >>> count_data = np.array([[[0.0, 0.0, 2.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0, 0.0]],
        >>>                        [[0.0, 1.0, 2.0, 4.0, 4.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
        >>>                        [[0.0, 1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
        >>>                        [[0.0, 0.0, 0.0, 2.0, 4.0], [1.0, 0.0, 0.0, 0.0, 0.0]]])
        >>> data.load(variables, time_values, count_data)
        >>> data.data_mean
        [[[0.         0.5        1.25       3.         4.        ]
          [1.         0.75       0.25       0.25       0.        ]]
         [[0.         0.25096378 0.41313568 0.50182694 0.        ]
          [0.         0.21847682 0.21654396 0.21624184 0.        ]]]
        >>> data.data_variance
        [[[0.         0.33333333 0.91666667 1.33333333 0.        ]
          [0.         0.25       0.25       0.25       0.        ]]
         [[0.         0.10247419 0.39239737 0.406103   0.        ]
          [0.         0.13197367 0.13279328 0.13272021 0.        ]]]
        >>> data.data_covariance
        [[[ 0.          0.16666667  0.25       -0.33333333  0.        ]]
         [[ 0.          0.11379549  0.18195518  0.22722941  0.        ]]]
        """

        # validate the user's data_input
        self._validate_data_input(variables, time_values, count_data, data_type,
                        mean_data, var_data, cov_data,
                        bootstrap_samples, basic_sigma)

        # initialise information from data_input
        self.data_variables = variables
        self.data_type = data_type
        self.data_time_values = time_values
        self.data_basic_sigma = basic_sigma

        # obtain the number of variables and time_values, respectively
        self.data_num_variables = len(self.data_variables)
        self.data_num_time_values = self.data_time_values.shape[0]

        # find out whether data has mean summary stats only
        # (no count data and also no explicit var and cov data, only mean)
        self.data_mean_exists_only = self.process_mean_exist_only(
                                                data_type, var_data, cov_data)

        # create indexing order for the data based on the data_variables list
        (self.data_mean_order,
        self.data_variance_order,
        self.data_covariance_order) = self.create_data_variable_order(self.data_variables,
                                                                    self.data_mean_exists_only)

        # convert none-type data info to an empty number array with one 0 length axis
        count_data, mean_data, var_data, cov_data = self.convert_none_data_to_empty_array(
                                        count_data, mean_data, var_data, cov_data,
                                        self.data_num_variables, self.data_num_time_values)

        # dependent on data_type, load data as summary statistics or count data
        if self.data_type=='summary':
            self._validate_shape_summary(self.data_mean_order, self.data_variance_order,
                                        self.data_covariance_order, self.data_num_time_values,
                                        mean_data, var_data, cov_data)
            self.data_mean = mean_data
            self.data_variance = var_data
            self.data_covariance = cov_data

            # also initialise empty count data array
            self.data_counts = count_data

        # in case of count data, bootstrapping is used to compute the summary statistics
        elif self.data_type=='counts':
            self._validate_shape_counts(self.data_num_variables, self.data_num_time_values, count_data)
            self.data_counts = count_data
            self.data_bootstrap_samples = bootstrap_samples

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
    def process_mean_exist_only(data_type, var_data, cov_data):
        """Initialise the `data_mean_exists_only` data attribute. The
        `data_mean_exists_only` attribute indicates whether a data object contains
        potentially higher summary moments (variance and covariance) or the first
        moments only (means).

        Parameters
        ----------
        data_type : str
            Type of the data object; either `'counts'` or `'summary'`.
        var_data : numpy.ndarray or None
            Dynamic variance statistics used in `data_type='summary'` mode.
        cov_data : numpy.ndarray or None
            Dynamic covariance statistics used in `data_type='summary'` mode.

        Returns
        -------
        data_mean_exists_only : bool
            Bool to indicate whether data contains mean information only or also
            higher moments (variance and covariance). Typically available at
            `data.data_mean_exists_only` for a data object `data`.

        Examples
        --------
        >>> me.Data.process_mean_exist_only('counts', None, None)
        False
        >>> me.Data.process_mean_exist_only('summary', None, None)
        True
        >>> # with some data arrays for variance and covariance (!=None)
        >>> me.Data.process_mean_exist_only('summary', var_data, cov_data)
        False
        """

        if data_type=='counts':
            data_mean_exists_only = False
        elif data_type=='summary':
            # for data_mean_exists_only=True we expect var and cov to be None/empty
            if var_data is None:
                var_bool = True
            elif var_data.size==0:
                var_bool = True
            else:
                var_bool = False

            if cov_data is None:
                cov_bool = True
            elif cov_data.size==0:
                cov_bool = True
            else:
                cov_bool = False

            data_mean_exists_only = True if (var_bool and cov_bool) else False
        return data_mean_exists_only


    @staticmethod
    def convert_none_data_to_empty_array(count_data, mean_data, var_data, cov_data,
                                    num_variables, num_time_values):
        """Convert `None`-type data input into empty numpy arrays.

        Returned data arrays have a zero-sized data repeats or variable order
        dimension but are otherwise shaped as needed for further internal
        computations.

        Parameters
        ----------
        count_data : numpy.ndarray or None
            Dynamic count data used in `data_type='counts'` mode. If `None`,
            an empty array will be constructed with shape `(0, number of variables,
            number of time points)`;
            if numpy array already, it will be left unchanged.
        mean_data : numpy.ndarray or None
            Dynamic mean statistics used in `data_type='summary'` mode. If `None`,
            an empty array will be constructed with shape `(2, 0, number of time points)`;
            if numpy array already, it will be left unchanged.
        var_data : numpy.ndarray or None
            Dynamic variance statistics used in `data_type='summary'` mode. If `None`,
            an empty array will be constructed with shape `(2, 0, number of time points)`;
            if numpy array already, it will be left unchanged.
        cov_data : numpy.ndarray or None
            Dynamic covariance statistics used in `data_type='summary'` mode. If `None`,
            an empty array will be constructed with shape `(2, 0, number of time points)`;
            if numpy array already, it will be left unchanged.
        num_variables : int
            Number of data variables.
        num_time_values : int
            Number of time points.

        Returns
        -------
        count_data : numpy.ndarray
            Dynamic count data used in `data_type='counts'` mode, possibly an
            empty array.
        mean_data : numpy.ndarray
            Dynamic mean statistics used in `data_type='summary'` mode,
            possibly an empty array.
        var_data : numpy.ndarray
            Dynamic variance statistics used in `data_type='summary'` mode,
            possibly an empty array.
        cov_data : numpy.ndarray
            Dynamic covariance statistics used in `data_type='summary'` mode,
            possibly an empty array.
        """

        if count_data is None:
            count_data = np.empty((0, num_variables, num_time_values))

        if mean_data is None:
            mean_data = np.empty((2, 0, num_time_values))

        if var_data is None:
            var_data = np.empty((2, 0, num_time_values))

        if cov_data is None:
            cov_data = np.empty((2, 0, num_time_values))

        return count_data, mean_data, var_data, cov_data


    @staticmethod
    def create_data_variable_order(data_variables, data_mean_exists_only):
        """Creates objects to define the order of data variables for mean,
        variance and covariance. Mean and variance order follows the order of the
        input; covariances are ordered with a priority for what comes first in the
        input.

        Parameters
        ----------
        data_variables : list of str
            A list of strings for the data variables.
        data_mean_exists_only : bool
            Bool to indicate whether data contains mean information only or also
            higher moments (variance and covariance). If `False`, variable
            order will be computed for mean, variance and covariance; if `True`,
            variance and covariance order will be empty.

        Returns
        -------
        data_mean_order : list of dict
            Variable order for the data means.
        data_variance_order : list of dict
            Variable order for the data variances.
        data_covariance_order : list of dict
            Variable order for the data covariances.

        Examples
        --------
        >>> mean_only = False
        >>> me.Data.create_data_variable_order(['A', 'B', 'C'], mean_only)
        ([{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)},
          {'variables': 'C', 'summary_indices': 2, 'count_indices': (2,)}],
         [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
          {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)},
          {'variables': ('C', 'C'), 'summary_indices': 2, 'count_indices': (2, 2)}],
         [{'variables': ('A', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)},
          {'variables': ('A', 'C'), 'summary_indices': 1, 'count_indices': (0, 2)},
          {'variables': ('B', 'C'), 'summary_indices': 2, 'count_indices': (1, 2)}])
        """

        # order of mean and variance indices just matches the data_variables order
        data_mean_order = [{'variables': var,  'summary_indices': i, 'count_indices': (i, )}
                                        for i, var in enumerate(data_variables)]

        if data_mean_exists_only:
            data_variance_order = list()
        else:
            data_variance_order = [{'variables': (var, var),  'summary_indices': i, 'count_indices': (i, i)}
                                        for i, var in enumerate(data_variables)]

        # data_covariance_order is ordered with a priority of the smaller index
        # (i.e., what comes first in data_variables)
        data_covariance_order = list()
        if not data_mean_exists_only:
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
        """Bootstraps dynamic count data to mean, variance and covariance statistics
        over time with associated standard errors.

        Parameters
        ----------
        data_num_time_values : int
            Number of time values; typical usage with `data_num_time_values=data.data_num_time_values`
            of a memocell data object.
        data_mean_order : list of dict
            Data mean order of a memocell data object; typical usage with
            `data_mean_order=data.data_mean_order`.
        data_variance_order : list of dict
            Data variance order of a memocell data object; typical usage with
            `data_variance_order=data.data_variance_order`.
        data_covariance_order : list of dict
            Data covariance order of a memocell data object; typical usage with
            `data_covariance_order=data.data_covariance_order`.
        count_data : numpy.ndarray
            Numpy array of the count data to bootstrap with shape (`n`, `m`, `t`); the
            number of repeats `n`, the number of variables `m`, the number of time points `t`.
            Typical usage with `count_data=data.data_counts` of a memocell data object.
            Order of the variables should match with `data_mean_order`,
            `data_variance_order` and `data_covariance_order`. Time points `t` should be
            equal to `data_num_time_values`.
        bootstrap_samples : int
            Integer to set the number of bootstrap samples. Typically `bootstrap_samples=100000`
            is sufficient; typical usage with `data.data_bootstrap_samples`.

        Returns
        -------
        data_mean : numpy.ndarray
            Numpy array of the bootstrapped dynamic mean statistics and standard
            errors with shape (2, `len(data_mean_order)`, `data_num_time_values`).
            `data_mean[0, :, :]` contains the statistics;
            `data_mean[1, :, :]` contains the standard errors.
        data_var : numpy.ndarray
            Numpy array of the bootstrapped dynamic variance statistics and standard
            errors with shape (2, `len(data_variance_order)`, `data_num_time_values`).
            `data_var[0, :, :]` contains the statistics;
            `data_var[1, :, :]` contains the standard errors.
        data_cov : numpy.ndarray
            Numpy array of the bootstrapped dynamic covariance statistics and standard
            errors with shape (2, `len(data_covariance_order)`, `data_num_time_values`).
            `data_cov[0, :, :]` contains the statistics;
            `data_cov[1, :, :]` contains the standard errors.

        """

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
        """Compute mean and associated standard error of the mean of a given
        1-dimensional sample. Standard error is obtained by bootstrapping.

        Parameters
        ----------
        sample : 1d numpy.ndarray
            Sample used to compute mean and standard error of the mean.
        num_resamples : int
            Integer to set the number of bootstrap samples. Typically `num_resamples=100000`
            is sufficient.

        Returns
        -------
        stat_sample : numpy.float64
            Mean of the sample.
        se_stat_sample : numpy.float64
            Standard error of the mean of the sample obtained by bootstrapping.

        Examples
        --------
        >>> me.Data.bootstrapping_mean(np.array([1.0, 2.0, 3.0]), 100000)
        (2.0, 0.4705758644290084)
        """

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
        """Compute variance and associated standard error of the variance of a given
        1-dimensional sample. Standard error is obtained by bootstrapping.

        Parameters
        ----------
        sample : 1d numpy.ndarray
            Sample used to compute variance and standard error of the variance.
        num_resamples : int
            Integer to set the number of bootstrap samples. Typically `num_resamples=100000`
            is sufficient.

        Returns
        -------
        stat_sample : numpy.float64
            Variance of the sample (ddof=1).
        se_stat_sample : numpy.float64
            Standard error of the variance of the sample obtained by bootstrapping.

        Examples
        --------
        >>> me.Data.bootstrapping_variance(np.array([1.0, 2.0, 3.0]), 100000)
        (1.0, 0.4707057737121149)
        """

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
        """Compute covariance and associated standard error of the covariance of two given
        1-dimensional samples. Standard error is obtained by bootstrapping.

        Parameters
        ----------
        sample1 : 1d numpy.ndarray
            Sample used to compute covariance and standard error of the covariance.
            Same length as `sample2`.
        sample2 : 1d numpy.ndarray
            Sample used to compute covariance and standard error of the covariance.
            Same length as `sample1`.
        num_resamples : int
            Integer to set the number of bootstrap samples. Typically `num_resamples=100000`
            is sufficient.

        Returns
        -------
        stat_sample : numpy.float64
            Covariance of the sample (ddof=1).
        se_stat_sample : numpy.float64
            Standard error of the covariance of the sample obtained by bootstrapping.

        Examples
        --------
        >>> me.Data.bootstrapping_covariance(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), 10000)
        (-1.0, 0.47186781968816127)
        """

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
        """Handles zero-valued or almost zero-valued standard errors by
        introducing a `basic_sigma` standard error value for those entries;
        this is required for numerical stability of the likelihood calculation
        and allows to remove too strong weights on certain data points. Zero or almost-zero
        standard errors can occur when bootstrapping data features that are not
        adequately represented by the actual data sample used in the bootstrap.
        `Important note:` In such a situation a `basic_sigma` can be introduced but the
        specific value has to be chosen very carefully, as its choice can strongly
        influence model selection results. One can test different choices on artificial
        `in silico` data; or use guidelines based on
        `additive smoothing <https://en.wikipedia.org/wiki/Additive_smoothing>`_ and
        `the rule of succession <https://en.wikipedia.org/wiki/Rule_of_succession>`_.

        Parameters
        ----------
        basic_sigma : float
            Non-negative float value for `basic_sigma`.
        data : numpy.ndarray
            Data to introduce a `basic_sigma` to; more specifically, the standard
            errors (in `data[1, :, :]`) will be replaced by `basic_sigma` if they are
            smaller than `basic_sigma`.

        Returns
        -------
        data : numpy.ndarray
            Returns data with standard errors larger or equal to `basic_sigma`.

        Examples
        --------
        >>> data = np.array([[[0.        , 0.02272727, 2.93181818],
        >>>                   [1.        , 0.97727273, 0.15909091]],
        >>>                  [[0.        , 0.01998622, 0.40653029],
        >>>                   [0.        , 0.02294546, 0.05969529]]])
        >>> me.Data.introduce_basic_sigma(0.01, data)
        array([[[0.        , 0.02272727, 2.93181818],
                [1.        , 0.97727273, 0.15909091]],
                [[0.01      , 0.01998622, 0.40653029],
                [0.01      , 0.02294546, 0.05969529]]])
        """

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
        """Get the number of total data points `n` used in the inference.
        This is the number of data points of mean, variance and covariance statistics
        (default inference) or the number of mean data points only (mean-only modes).

        Parameters
        ----------
        data_mean : numpy.ndarray
            Numpy array of the dynamic mean statistics and standard
            errors with shape (2, `len(data_mean_order)`, `data_num_time_values`).
        data_var : numpy.ndarray
            Numpy array of the dynamic variance statistics and standard
            errors with shape (2, `len(data_variance_order)`, `data_num_time_values`).
        data_cov : numpy.ndarray
            Numpy array of the dynamic covariance statistics and standard
            errors with shape (2, `len(data_covariance_order)`, `data_num_time_values`).

        Returns
        -------
        data_num_values : int
            Number of data points of mean, variance and covariance statistics.
        data_num_values_mean_only : int
            Number of data points of mean statistics.
        """

        # calculate the number of data points along their last two dimensions
        # number of variables * number of time points
        data_points_mean = int(data_mean.shape[1] * data_mean.shape[2])
        data_points_var = int(data_var.shape[1] * data_var.shape[2])
        data_points_cov = int(data_cov.shape[1] * data_cov.shape[2])

        return data_points_mean + data_points_var + data_points_cov, data_points_mean


    def events_find_all(self):
        """Wrapper method to call a set of event functions on dynamic count data in `data.data_counts`.
        Results are accessible via data event attributes; for example
        `data.event_all_first_cell_type_conversion`."""

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
        """Find events of first change from initial conditions (state at first
        time point).

        Parameters
        ----------
        well_trace : numpy.ndarray
            Dynamic variable counts of one well / one stochastic realisation with shape
            (`number of variables`, `len(time_values)`).
        time_values : 1d numpy.ndarray
            Time values corresponding to variable counts (`len(time_values)` should
            match `well_trace.shape[1]`).

        Returns
        -------
        event_bool : bool
            `True` if an event occurs in this `well_trace`, `False` otherwise.
        event_tau : float or None
            If `event_bool=True`, `event_tau` provides the waiting time when the
            event occured (according to `time_values`); `None` otherwise.

        Examples
        --------
        >>> me.Data.event_find_first_change_from_inital_conditions(data,
        >>>                              np.array([[0.0, 0.0, 1.0, 2.0],
        >>>                                        [0.0, 0.0, 0.0, 1.0]]),
        >>>                              np.array([0.0, 1.0, 2.0, 3.0]))
        (True, 2.0)
        """

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
        """Find events of first cell count increase; calculation based on total cell
        count compared to initial condition (state at first time point).

        Parameters
        ----------
        well_trace : numpy.ndarray
            Dynamic variable counts of one well / one stochastic realisation with shape
            (`number of variables`, `len(time_values)`).
        time_values : 1d numpy.ndarray
            Time values corresponding to variable counts (`len(time_values)` should
            match `well_trace.shape[1]`).

        Returns
        -------
        event_bool : bool
            `True` if an event occurs in this `well_trace`, `False` otherwise.
        event_tau : float or None
            If `event_bool=True`, `event_tau` provides the waiting time when the
            event occured (according to `time_values`); `None` otherwise.

        Examples
        --------
        >>> me.Data.event_find_first_cell_count_increase(data,
        >>>                         np.array([[4.0, 4.0, 4.0, 4.0],
        >>>                                   [1.0, 1.0, 2.0, 3.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0]))
        (True, 2.0)
        """

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
        """Find events of first cell type conversion (e.g., differentiation
        between different cell types / count variables); calculation looks for
        change of cell state despite maintenance of total cell counts.

        Parameters
        ----------
        well_trace : numpy.ndarray
            Dynamic variable counts of one well / one stochastic realisation with shape
            (`number of variables`, `len(time_values)`).
        time_values : 1d numpy.ndarray
            Time values corresponding to variable counts (`len(time_values)` should
            match `well_trace.shape[1]`).

        Returns
        -------
        event_bool : bool
            `True` if an event occurs in this `well_trace`, `False` otherwise.
        event_tau : float or None
            If `event_bool=True`, `event_tau` provides the waiting time when the
            event occured (according to `time_values`); `None` otherwise.

        Examples
        --------
        >>> me.Data.event_find_first_cell_type_conversion(data,
        >>>                         np.array([[4.0, 4.0, 3.0, 3.0],
        >>>                                   [1.0, 1.0, 2.0, 2.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0]))
        (True, 2.0)
        """

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
        """Find events of first cell count increase after cell type conversion.
        Calculation looks first for the conditioned event (cell type conversion,
        based on `event_find_first_cell_type_conversion` method).
        If this event exists, the `well_trace` and `time_values` are shortened and
        then searched for a first cell count increase (based on
        `event_find_first_cell_count_increase` method).

        Parameters
        ----------
        well_trace : numpy.ndarray
            Dynamic variable counts of one well / one stochastic realisation with shape
            (`number of variables`, `len(time_values)`).
        time_values : 1d numpy.ndarray
            Time values corresponding to variable counts (`len(time_values)` should
            match `well_trace.shape[1]`).
        diff : bool
            If `diff=True`, `event_tau` will provide the waiting time starting
            at the conditioned event; otherwise the waiting time is provided starting
            at the first time point.

        Returns
        -------
        event_bool : bool
            `True` if an event occurs in this `well_trace`, `False` otherwise.
        event_tau : float or None
            If `event_bool=True`, `event_tau` provides the waiting time when the
            event occured (according to `time_values`); `None` otherwise.

        Examples
        --------
        >>> me.Data.event_find_first_cell_count_increase_after_cell_type_conversion(data,
        >>>                         np.array([[4.0, 3.0, 3.0, 4.0],
        >>>                                   [1.0, 2.0, 2.0, 2.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0]))
        (True, 2.0)

        >>> me.Data.event_find_first_cell_count_increase_after_cell_type_conversion(data,
        >>>                         np.array([[4.0, 3.0, 3.0, 4.0],
        >>>                                   [1.0, 2.0, 2.0, 2.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0]), diff=False)
        (True, 3.0)
        """

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
        """Find events of a second cell count increase after a first cell count increase
        and (even more before that) a cell type conversion.
        Calculation looks first for the conditioned event (first increase and cell type conversion,
        based on `event_find_first_cell_count_increase_after_cell_type_conversion` method).
        If this event exists, the `well_trace` and `time_values` are shortened and
        then searched for another "first" cell count increase (based on
        `event_find_first_cell_count_increase` method).

        Parameters
        ----------
        well_trace : numpy.ndarray
            Dynamic variable counts of one well / one stochastic realisation with shape
            (`number of variables`, `len(time_values)`).
        time_values : 1d numpy.ndarray
            Time values corresponding to variable counts (`len(time_values)` should
            match `well_trace.shape[1]`).
        diff : bool
            If `diff=True`, `event_tau` will provide the waiting time starting
            at the conditioned event; otherwise the waiting time is provided starting
            at the first time point.

        Returns
        -------
        event_bool : bool
            `True` if an event occurs in this `well_trace`, `False` otherwise.
        event_tau : float or None
            If `event_bool=True`, `event_tau` provides the waiting time when the
            event occured (according to `time_values`); `None` otherwise.

        Examples
        --------
        >>> me.Data.event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(
        >>>                         data,
        >>>                         np.array([[4.0, 3.0, 3.0, 4.0, 5.0],
        >>>                                   [1.0, 2.0, 2.0, 2.0, 2.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
        (True, 1.0)

        >>> me.Data.event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(
        >>>                         data,
        >>>                         np.array([[4.0, 3.0, 3.0, 4.0, 5.0],
        >>>                                   [1.0, 2.0, 2.0, 2.0, 2.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0, 4.0]), diff=False)
        (True, 4.0)
        """

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
        """Find events of a third cell count increase after a first and second cell count increase
        and (even more before that) a cell type conversion.
        Calculation looks first for the conditioned event (first and second increase and cell type conversion,
        based on `event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion` method).
        If this event exists, the `well_trace` and `time_values` are shortened and
        then searched for another "first" cell count increase (based on
        `event_find_first_cell_count_increase` method).

        Parameters
        ----------
        well_trace : numpy.ndarray
            Dynamic variable counts of one well / one stochastic realisation with shape
            (`number of variables`, `len(time_values)`).
        time_values : 1d numpy.ndarray
            Time values corresponding to variable counts (`len(time_values)` should
            match `well_trace.shape[1]`).
        diff : bool
            If `diff=True`, `event_tau` will provide the waiting time starting
            at the conditioned event; otherwise the waiting time is provided starting
            at the first time point.

        Returns
        -------
        event_bool : bool
            `True` if an event occurs in this `well_trace`, `False` otherwise.
        event_tau : float or None
            If `event_bool=True`, `event_tau` provides the waiting time when the
            event occured (according to `time_values`); `None` otherwise.

        Examples
        --------
        >>> me.Data.event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion(
        >>>                         data,
        >>>                         np.array([[4.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        >>>                                   [1.0, 2.0, 2.0, 2.0, 2.0, 3.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
        (True, 1.0)

        >>> me.Data.event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion(
        >>>                         data,
        >>>                         np.array([[4.0, 3.0, 3.0, 4.0, 5.0, 5.0],
        >>>                                   [1.0, 2.0, 2.0, 2.0, 2.0, 3.0]]),
        >>>                         np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), diff=False)
        (True, 5.0)
        """

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
        """Fit a Gamma distribution with parameter `theta` (`theta[0]` shape parameter
        and `theta[1]` scale parameter) to a data sample of waiting times (`waiting_times_arr`).
        Fitting is based on binning the waiting time data first, so this method works
        equally well for continuous and discrete waiting times (when continuous
        observation is experimentally not accessible). The bin edges are given by
        `[-np.inf, data.data_time_values, np.inf]` where individual bins are
        left-open, right-closed intervals of the bins edges (of form `(a, b]`). See utility
        methods `data.gamma_compute_bin_probabilities` and
        `data.gamma_negative_multinomial_log_likelihood` for more information of code
        specifics.

        Parameters
        ----------
        waiting_times_arr : 1d numpy.ndarray or list of float or int
            Continuous or discrete waiting times to fit the Gamma distribution to.

        Returns
        -------
        theta : 1d numpy.ndarray
            Result of the fit of a `Gamma(theta)` distribution to the data of
            `waiting_times_arr`; with `theta[0]` shape parameter
            and `theta[1]` scale parameter of the `Gamma` distribution.

        Examples
        --------
        >>> data = me.Data('data_init')
        >>> data.data_time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> theta = [4.0, 0.5]
        >>> waiting_times_arr = np.random.gamma(theta[0], theta[1], 100000)
        >>> data.gamma_fit_binned_waiting_times(waiting_times_arr)
        >>> data.gamma_fit_theta
        array([4.01966125, 0.49769529])
        """

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
        self.gamma_fit_result = optimize.minimize(self.gamma_negative_multinomial_log_likelihood,
                                                    self.gamma_fit_theta_init,
                                                    method='L-BFGS-B',
                                                    bounds=[(0, None)]*len(self.gamma_fit_theta_init))
        self.gamma_fit_theta = self.gamma_fit_result['x']


    def gamma_compute_bin_probabilities(self, theta):
        """Utility method for fitting Gamma/Erlang distribution to waiting time
        histogram. This method computes bin probabilities for bins in `data.gamma_fit_bins`
        for a given `theta`, where `theta[0]` and `theta[1]` are the shape and scale
        parameter of the Gamma distribution, respectively. More specifically, the
        probability of a continuous waiting time `tau` to be in the bin `(a, b]` is calculated
        by `p(tau in (a, b]) = F_theta(b) - F_theta(a)`, with `F_theta` being the
        cumulative density function of the Gamma distribution for parameters `theta`.

        Parameters
        ----------
        theta : list of float
            Shape (`theta[0]`) and scale (`theta[1]`) parameter for the Gamma distribution.

        Returns
        -------
        bins_probs : 1d numpy.ndarray
            Probabilities to find a continuous waiting time following a Gamma(`theta`)
            distribution in the bins given by `data.gamma_fit_bins`.

        Examples
        --------
        >>> data = me.Data('data_init')
        >>> data_time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        >>> data.gamma_fit_bins = np.concatenate(([-np.inf], data_time_values, [np.inf]))
        >>> data.gamma_compute_bin_probabilities([4.0, 0.5])
        np.array([0.        , 0.14287654, 0.42365334, 0.28226624, 0.10882377,
                  0.03204406, 0.01033605])
        """

        # the probability to be in bin (a, b] are given by prob(theta) = Gamma_cdf_theta(b) - Gamma_cdf_theta(a)
        # F(time_values[1:]) - F(time_values[:-1]) achieves bin-wise calculation of
        # bin probs by prob(theta) = F(b) - F(a), with F Gamma CDF for a given theta
        bins_probs = stats.gamma.cdf(self.gamma_fit_bins[1:], a=theta[0], loc=0, scale=theta[1]) - stats.gamma.cdf(self.gamma_fit_bins[:-1], a=theta[0], loc=0, scale=theta[1])
        return bins_probs


    def gamma_negative_multinomial_log_likelihood(self, theta):
        """Utility method for fitting Gamma/Erlang distribution to waiting time
        histogram. This method computes the negative logarithmic likelihood to see
        a certain waiting time histogram (in the form of
        `data.gamma_fit_bin_inds_all_occ`) for given bin probabilities based on a
        Gamma waiting time distribution with parameters `theta`. Likelihood calculation
        is based on a multinomial model.

        Parameters
        ----------
        theta : list of float
            Shape (`theta[0]`) and scale (`theta[1]`) parameter for the Gamma distribution.

        Returns
        -------
        neg_log_likelihood : float
            Negative logarithmic likelihood `p(D | Gamma(theta))` to see waiting time
            histogram data `D` given waiting time distribution `Gamma(theta)`.
        """

        # calculate the bin probabilities for a given theta = (shape, scale)
        bin_probs = self.gamma_compute_bin_probabilities(theta)

        # use a multinomial model to compute the log likelihood of observing the data (counts in each bin) for the given bins probs
        log_likelihood = stats.multinomial.logpmf(self.gamma_fit_bin_inds_all_occ, n=np.sum(self.gamma_fit_bin_inds_all_occ), p=bin_probs)
        return -log_likelihood
    ###

    ### plotting helper functions
    
    # NOTE: maybe reactivate later
    # def _event_percentages(self, settings):
    #     """Private plotting helper method."""
    #
    #     y_list_err = list()
    #     x_ticks = list()
    #     attributes = dict()
    #
    #     for i, event_dict in enumerate(settings):
    #         event_results = event_dict['event']
    #         event_perc = 100.0 * (sum([event_bool for event_bool, __ in event_results])/float(len(event_results)))
    #         y_list_err.append([event_perc])
    #
    #         x_ticks.append(event_dict['label'])
    #
    #         attributes[i] = (None, event_dict['color'])
    #
    #     y_arr_err = np.array(y_list_err)
    #     return y_arr_err, x_ticks, attributes

    def _scatter_at_time_point(self, variable1, variable2, time_ind, settings):
        """Private plotting helper method."""

        var_ind_x = self.data_variables.index(variable1)
        var_ind_y = self.data_variables.index(variable2)

        x_arr = self.data_counts[:, var_ind_x, time_ind]
        y_arr = self.data_counts[:, var_ind_y, time_ind]

        attributes = dict()
        attributes['color'] = settings['color']
        attributes['opacity'] = settings['opacity']
        attributes['label'] = settings['label']

        return x_arr, y_arr, attributes

    def _histogram_continuous_event_waiting_times(self, event_results, settings):
        """Private plotting helper method."""

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

    def _histogram_continuous_event_waiting_times_w_gamma_fit(self, event_results, settings):
        """Private plotting helper method."""

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

            return x_line_arr, y_line_arr, r'$\Gamma$' + f'($n$={round(gamma_fit_shape, 1)}, $θ$={round(gamma_fit_shape * gamma_fit_scale, 1)})', settings['gamma_color'] # , KS $p$-value {round(pval, 2)}

        return bar_arr, bar_attributes, gamma_fit_func

    def _histogram_discrete_cell_counts_at_time_point(self, time_ind, settings):
        """Private plotting helper method."""

        bar_arr = self.data_counts[:, :, time_ind]
        bar_attributes = dict()

        for i, var in enumerate(self.data_variables):
            bar_attributes[i] = {   'label': settings[var]['label'],
                                    'color': settings[var]['color'],
                                    'opacity': settings[var]['opacity']}

        return bar_arr, bar_attributes

    def _dots_w_bars_evolv_mean(self, settings):
        """Private plotting helper method."""

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

    def _dots_w_bars_evolv_variance(self, settings):
        """Private plotting helper method."""

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

    def _dots_w_bars_evolv_covariance(self, settings):
        """Private plotting helper method."""

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
    def _validate_shape_summary(mean_order, var_order, cov_order, num_time_values,
                                    mean_data, var_data, cov_data):
        """Private validation method."""
        # data and standard error dimensions required
        if 2==mean_data.shape[0] and 2==var_data.shape[0] and 2==cov_data.shape[0]:
            pass
        else:
            raise ValueError('Unfit dimensions of summary data.')

        if (num_time_values==mean_data.shape[2] and num_time_values==var_data.shape[2]
                        and num_time_values==cov_data.shape[2]):
            pass
        else:
            raise ValueError('Dimension mismatch between time_values and summary data.')

        # even in summary stats mode we allow zero variable order dimension for
        # var and cov data (in the case one wants to pass mean summary stats only)
        if (len(mean_order)==mean_data.shape[1] and
                    (len(var_order)==var_data.shape[1] or 0==var_data.shape[1]) and
                    (len(cov_order)==cov_data.shape[1] or 0==cov_data.shape[1])):
            pass
        else:
            raise ValueError('Dimension mismatch between data variables and summary data.')

    @staticmethod
    def _validate_shape_counts(num_variables, num_time_values, count_data):
        """Private validation method."""
        if num_variables==count_data.shape[1]:
            pass
        else:
            raise ValueError('Dimension mismatch between data variables and count_data.')

        if num_time_values==count_data.shape[2]:
            pass
        else:
            raise ValueError('Dimension mismatch between time_values and count_data.')

    @staticmethod
    def _validate_data_input(variables, time_values, count_data, data_type,
                    mean_data, var_data, cov_data,
                    bootstrap_samples, basic_sigma):
        """Private validation method."""
        # TODO: there could be more validation: check the shapes of all data-related inputs

        # check data input variables
        if isinstance(variables, list):
            if all(isinstance(var, str) for var in variables):
                pass
            else:
                raise TypeError('List with string items expected for variables.')
        else:
            raise TypeError('List expected for variables.')

        # check uniqueness of variables
        if len(variables) > len(set(variables)):
            raise ValueError('Data variables have to be unique.')

        # check data input type
        if isinstance(data_type, str):
            if data_type=='summary' or data_type=='counts':
                pass
            else:
                raise ValueError('String \'summary\' or \'counts\' expected for data_type.')
        else:
            raise TypeError('String expected for data_type.')

        # check data input time values
        if isinstance(time_values, np.ndarray):
            if time_values.ndim == 1:
                pass
            else:
                raise ValueError('Times values are expected to be provided as a numpy array with shape \'(n, )\' with n being the number of values.')
        else:
            raise TypeError('Numpy array expected for time_values.')

        # check data input mean_data
        if isinstance(mean_data, np.ndarray) or isinstance(mean_data, type(None)):
            pass
        else:
            raise TypeError('Numpy array or None expected for mean_data.')

        # check data input var_data
        if isinstance(var_data, np.ndarray) or isinstance(var_data, type(None)):
            pass
        else:
            raise TypeError('Numpy array or None expected for var_data.')

        # check data input cov_data
        if isinstance(cov_data, np.ndarray) or isinstance(cov_data, type(None)):
            pass
        else:
            raise TypeError('Numpy array or None expected for cov_data.')

        # check data input count_data
        if isinstance(count_data, np.ndarray) or isinstance(count_data, type(None)):
            pass
        else:
            raise TypeError('Numpy array or None expected for count_data.')

        # check data input bootstrap_samples
        if isinstance(bootstrap_samples, int):
            pass
        else:
            raise TypeError('Integer expected for bootstrap_samples.')

        # check data input basic_sigma
        if isinstance(basic_sigma, float):
            pass
        else:
            raise TypeError('Float value expected for basic_sigma.')

        # check if some data is there, possible options
        # - count data
        # - mean var and cov data
        # - mean data only
        if data_type=='counts':
            if isinstance(count_data, np.ndarray):
                pass
            else:
                raise ValueError('In counts data type, count_data is expected as numpy array.')
        elif data_type=='summary':
            if isinstance(mean_data, np.ndarray):
                pass
            else:
                raise ValueError('In summary data type, at least mean_data is expected as numpy array.')


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
