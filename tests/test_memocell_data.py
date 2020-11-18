
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import memocell as me
import numpy as np

class TestDataClass(object):
    ### tests for create_data_variable_order()
    def test_create_data_variable_order_mean_only_false(self):
        mean_only = False
        assert(me.Data.create_data_variable_order(['A', 'B', 'C'], mean_only) == (
                            [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)},
                          {'variables': 'C', 'summary_indices': 2, 'count_indices': (2,)}],
                         [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
                          {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)},
                          {'variables': ('C', 'C'), 'summary_indices': 2, 'count_indices': (2, 2)}],
                         [{'variables': ('A', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)},
                          {'variables': ('A', 'C'), 'summary_indices': 1, 'count_indices': (0, 2)},
                          {'variables': ('B', 'C'), 'summary_indices': 2, 'count_indices': (1, 2)}]))

    def test_create_data_variable_order_mean_only_true(self):
        mean_only = True
        assert(me.Data.create_data_variable_order(['A', 'B', 'C'], mean_only) == (
                            [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)},
                          {'variables': 'C', 'summary_indices': 2, 'count_indices': (2,)}],
                         [],
                         []))

    def test_create_data_variable_order_no_alphabetical_order(self):
        mean_only = False
        assert(me.Data.create_data_variable_order(['C', 'B', 'A'], mean_only) == (
                        [{'variables': 'C', 'summary_indices': 0, 'count_indices': (0,)},
                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)},
                          {'variables': 'A', 'summary_indices': 2, 'count_indices': (2,)}],
                         [{'variables': ('C', 'C'), 'summary_indices': 0, 'count_indices': (0, 0)},
                          {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)},
                          {'variables': ('A', 'A'), 'summary_indices': 2, 'count_indices': (2, 2)}],
                         [{'variables': ('C', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)},
                          {'variables': ('C', 'A'), 'summary_indices': 1, 'count_indices': (0, 2)},
                          {'variables': ('B', 'A'), 'summary_indices': 2, 'count_indices': (1, 2)}]))

    def test_create_data_variable_order_no_validation_here(self):
        mean_only = False
        assert(me.Data.create_data_variable_order(['A', 'A'], mean_only) == (
                        [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                          {'variables': 'A', 'summary_indices': 1, 'count_indices': (1,)}],
                         [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
                          {'variables': ('A', 'A'), 'summary_indices': 1, 'count_indices': (1, 1)}],
                         [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 1)}]))

    ### tests for process_mean_exist_only
    def test_process_mean_exist_only_counts(self):
        assert(False == me.Data.process_mean_exist_only('counts', None, None))

    def test_process_mean_exist_only_summary_mean_only(self):
        assert(True == me.Data.process_mean_exist_only('summary', None, None))

    def test_process_mean_exist_only_summary_mean_only_via_empty(self):
        var_data = np.empty((2, 0, 3)) # some fake data
        cov_data = np.empty((2, 0, 3)) # some fake data
        assert(True == me.Data.process_mean_exist_only('summary', var_data, cov_data))

    def test_process_mean_exist_only_summary_mean_only_mixed_1(self):
        var_data = np.empty((2, 0, 3)) # some fake data
        assert(True == me.Data.process_mean_exist_only('summary', var_data, None))

    def test_process_mean_exist_only_summary_mean_only_mixed_2(self):
        cov_data = np.empty((2, 0, 3)) # some fake data
        assert(True == me.Data.process_mean_exist_only('summary', None, cov_data))

    def test_process_mean_exist_only_counts_var_and_cov(self):
        var_data = np.empty((2, 2, 3)) # some fake data
        cov_data = np.empty((2, 1, 3)) # some fake data
        assert(False == me.Data.process_mean_exist_only('summary', var_data, cov_data))

    def test_process_mean_exist_only_counts_var_only(self):
        var_data = np.empty((2, 2, 3)) # some fake data
        assert(False == me.Data.process_mean_exist_only('summary', var_data, None))

    def test_process_mean_exist_only_counts_cov_only(self):
        cov_data = np.empty((2, 1, 3)) # some fake data
        assert(False == me.Data.process_mean_exist_only('summary', None, cov_data))

    ### tests for convert_none_data_to_empty_array
    def test_convert_none_data_to_empty_array_none_data(self):
        count_data = None
        mean_data = None
        var_data = None
        cov_data = None
        num_variables = 2
        num_time_values = 3

        res_counts, res_mean, res_var, res_cov = me.Data.convert_none_data_to_empty_array(
                                count_data, mean_data,
                                var_data, cov_data,
                                num_variables, num_time_values)

        sol_counts = np.empty((0, 2, 3))
        sol_mean = np.empty((2, 0, 3))
        sol_var = np.empty((2, 0, 3))
        sol_cov = np.empty((2, 0, 3))
        np.testing.assert_allclose(sol_counts, res_counts)
        np.testing.assert_allclose(sol_mean, res_mean)
        np.testing.assert_allclose(sol_var, res_var)
        np.testing.assert_allclose(sol_cov, res_cov)

    def test_convert_none_data_to_empty_array_random_data(self):
        # create some random fake data
        # with 4 wells, 2 variables, 3 time points
        sol_counts = np.random.rand(4, 2, 3)
        sol_mean = np.random.rand(2, 2, 3)
        sol_var = np.random.rand(2, 2, 3)
        sol_cov = np.random.rand(2, 1, 3)
        num_variables = 2
        num_time_values = 3

        res_counts, res_mean, res_var, res_cov = me.Data.convert_none_data_to_empty_array(
                                sol_counts, sol_mean,
                                sol_var, sol_cov,
                                num_variables, num_time_values)

        np.testing.assert_allclose(sol_counts, res_counts)
        np.testing.assert_allclose(sol_mean, res_mean)
        np.testing.assert_allclose(sol_var, res_var)
        np.testing.assert_allclose(sol_cov, res_cov)

    ### tests for bootstrapping methods
    def test_bootstrapping_mean(self):
        stat_sample, se_stat_sample = me.Data.bootstrapping_mean(np.array([1.0, 2.0, 3.0]), 100000)
        assert((stat_sample, round(se_stat_sample, 1)) == (2.0, 0.5))

    def test_bootstrapping_variance(self):
        stat_sample, se_stat_sample = me.Data.bootstrapping_variance(np.array([1.0, 2.0, 3.0]), 100000)
        assert((stat_sample, round(se_stat_sample, 1)) == (1.0, 0.5))

    def test_bootstrapping_covariance(self):
        stat_sample, se_stat_sample = me.Data.bootstrapping_covariance(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), 10000)
        assert((stat_sample, round(se_stat_sample, 1)) == (-1.0, 0.5))

    def test_bootstrap_count_data_to_summary_stats_shape(self):
        count_data = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                             1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3.],
                                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                                           [[0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
                                             1., 1., 1., 1., 1., 2., 2., 3., 4., 4., 5., 5.],
                                            [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

        data_name = 'test_data'
        data = me.Data(data_name)
        data.load(['A', 'B'], np.linspace(0.0, 54.0, num=28, endpoint=True), count_data, bootstrap_samples=10)

        data_mean, data_var, data_cov = me.Data.bootstrap_count_data_to_summary_stats(
                                                data,
                                                data.data_num_time_values,
                                                data.data_mean_order,
                                                data.data_variance_order,
                                                data.data_covariance_order,
                                                data.data_counts,
                                                data.data_bootstrap_samples)

        assert((data_mean.shape, data_var.shape, data_cov.shape) == ((2, 2, 28), (2, 2, 28), (2, 1, 28)))


    def test_bootstrap_count_data_to_summary_stats_stat_values(self):
        count_data = np.array([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
                                             1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3.],
                                            [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],

                                           [[0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
                                             1., 1., 1., 1., 1., 2., 2., 3., 4., 4., 5., 5.],
                                            [1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
                                             0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]])

        data_name = 'test_data'
        data = me.Data(data_name)
        data.load(['A', 'B'], np.linspace(0.0, 54.0, num=28, endpoint=True), count_data, bootstrap_samples=10)

        data_mean, data_var, data_cov = me.Data.bootstrap_count_data_to_summary_stats(
                                                data,
                                                data.data_num_time_values,
                                                data.data_mean_order,
                                                data.data_variance_order,
                                                data.data_covariance_order,
                                                data.data_counts,
                                                data.data_bootstrap_samples)
        assert(np.all(data_mean[0, :, :] == np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 0.5,
                                            0.5, 0.5, 1. , 1. , 1. , 1. , 1. , 1. , 2. , 2. , 2.5, 3. , 3. ,
                                            4. , 4. ],
                                           [1. , 1. , 1. , 1. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0.5,
                                            0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                                            0. , 0. ]])) == True)
        assert(np.all(data_var[0, :, :] == np.array([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 0.5,
                                            0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 2. , 2. ,
                                            2. , 2. ],
                                           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0.5, 0.5, 0.5,
                                            0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,
                                            0. , 0. ]])) == True)
        assert(np.all(data_cov[0, :, :] == np.array([[ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.5, -0.5, -0.5,
                                            -0.5, -0.5, -0.5, -0.5,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
                                             0. ,  0. ,  0. ,  0. ,  0. ,  0. ]])) == True)

    ### test basic_sigma method
    def test_introduce_basic_sigma(self):
        data = np.array([[[0.        , 0.        , 0.        , 0.        , 0.02272727,
                         0.04545455, 0.09090909, 0.15909091, 0.18181818, 0.31818182,
                         0.45454545, 0.61363636, 0.75      , 0.81818182, 1.02272727,
                         1.31818182, 1.5       , 1.79545455, 2.25      , 2.61363636,
                         2.93181818, 3.38636364, 4.13636364, 4.75      , 5.59090909,
                         6.47727273, 7.40909091, 8.47727273],
                        [1.        , 1.        , 1.        , 1.        , 0.97727273,
                         0.95454545, 0.90909091, 0.86363636, 0.84090909, 0.72727273,
                         0.65909091, 0.56818182, 0.5       , 0.43181818, 0.34090909,
                         0.31818182, 0.25      , 0.22727273, 0.22727273, 0.18181818,
                         0.15909091, 0.13636364, 0.11363636, 0.09090909, 0.09090909,
                         0.09090909, 0.09090909, 0.09090909]],

                       [[0.        , 0.        , 0.        , 0.        , 0.02217504,
                         0.03127734, 0.04332831, 0.06380841, 0.06626216, 0.08355682,
                         0.10488916, 0.12202523, 0.13359648, 0.12848238, 0.14533142,
                         0.18722643, 0.19360286, 0.24160564, 0.29076081, 0.34158666,
                         0.39142742, 0.40884032, 0.51906845, 0.59882769, 0.6827803 ,
                         0.82338009, 0.94878574, 1.06604142],
                        [0.        , 0.        , 0.        , 0.        , 0.02248729,
                         0.03151183, 0.04341279, 0.05226862, 0.05494241, 0.06746095,
                         0.07182489, 0.07404752, 0.07514181, 0.07523978, 0.07086163,
                         0.0707864 , 0.0651271 , 0.06331053, 0.06288894, 0.05833507,
                         0.0543081 , 0.05187501, 0.04779238, 0.04355323, 0.04319857,
                         0.0426004 , 0.04327476, 0.0436796 ]]])
        data_bs = np.array([[[0.        , 0.        , 0.        , 0.        , 0.02272727,
                         0.04545455, 0.09090909, 0.15909091, 0.18181818, 0.31818182,
                         0.45454545, 0.61363636, 0.75      , 0.81818182, 1.02272727,
                         1.31818182, 1.5       , 1.79545455, 2.25      , 2.61363636,
                         2.93181818, 3.38636364, 4.13636364, 4.75      , 5.59090909,
                         6.47727273, 7.40909091, 8.47727273],
                        [1.        , 1.        , 1.        , 1.        , 0.97727273,
                         0.95454545, 0.90909091, 0.86363636, 0.84090909, 0.72727273,
                         0.65909091, 0.56818182, 0.5       , 0.43181818, 0.34090909,
                         0.31818182, 0.25      , 0.22727273, 0.22727273, 0.18181818,
                         0.15909091, 0.13636364, 0.11363636, 0.09090909, 0.09090909,
                         0.09090909, 0.09090909, 0.09090909]],

                       [[0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.10488916, 0.12202523, 0.13359648, 0.12848238, 0.14533142,
                         0.18722643, 0.19360286, 0.24160564, 0.29076081, 0.34158666,
                         0.39142742, 0.40884032, 0.51906845, 0.59882769, 0.6827803 ,
                         0.82338009, 0.94878574, 1.06604142],
                        [0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.1       , 0.1       , 0.1       , 0.1       , 0.1       ,
                         0.1       , 0.1       , 0.1       ]]])
        assert(np.all(data_bs == me.Data.introduce_basic_sigma(0.1, data))==True)

    ### test event methods
    def test_event_find_first_change_from_inital_conditions_1(self):
        data = me.Data('data_init')
        assert((True, 2.0) == me.Data.event_find_first_change_from_inital_conditions(data,
                                np.array([[0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))
    def test_event_find_first_change_from_inital_conditions_2(self):
        data = me.Data('data_init')
        assert((False, None) == me.Data.event_find_first_change_from_inital_conditions(data,
                                np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))
    def test_event_find_first_change_from_inital_conditions_3(self):
        data = me.Data('data_init')
        assert((True, 1.0) == me.Data.event_find_first_change_from_inital_conditions(data,
                                np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))

    def test_event_find_first_cell_count_increase_1(self):
        data = me.Data('data_init')
        assert((False, None) == me.Data.event_find_first_cell_count_increase(data,
                                np.array([[0.0, 0.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0], [1.0, 1.0, 1.0, 1.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))
    def test_event_find_first_cell_count_increase_2(self):
        data = me.Data('data_init')
        assert((True, 1.0) == me.Data.event_find_first_cell_count_increase(data,
                                np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 2.0, 3.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))
    def test_event_find_first_cell_count_increase_3(self):
        data = me.Data('data_init')
        assert((True, 2.0) == me.Data.event_find_first_cell_count_increase(data,
                                np.array([[4.0, 4.0, 4.0, 4.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 2.0, 3.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))

    def test_event_find_first_cell_type_conversion_1(self):
        data = me.Data('data_init')
        assert((True, 3.0) == me.Data.event_find_first_cell_type_conversion(data,
                                np.array([[4.0, 4.0, 4.0, 3.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
                                np.array([0.0, 1.0, 2.0, 3.0])))

    def test_event_find_first_cell_count_increase_after_cell_type_conversion_1(self):
        data = me.Data('data_init')
        assert((False, None) == me.Data.event_find_first_cell_count_increase_after_cell_type_conversion(data,
                       np.array([[4.0, 4.0, 3.0, 3.0],
                                [1.0, 1.0, 2.0, 2.0]]),
                       np.array([0.0, 1.0, 2.0, 3.0])))
    def test_event_find_first_cell_count_increase_after_cell_type_conversion_2(self):
        data = me.Data('data_init')
        assert((True, 2.0) == me.Data.event_find_first_cell_count_increase_after_cell_type_conversion(data,
                       np.array([[4.0, 3.0, 3.0, 3.0],
                                [1.0, 2.0, 2.0, 3.0]]),
                       np.array([0.0, 1.0, 2.0, 3.0]), diff=True))
    def test_event_find_first_cell_count_increase_after_cell_type_conversion_3(self):
        data = me.Data('data_init')
        assert((True, 3.0) == me.Data.event_find_first_cell_count_increase_after_cell_type_conversion(data,
                       np.array([[4.0, 3.0, 3.0, 3.0],
                                [1.0, 2.0, 2.0, 3.0]]),
                       np.array([0.0, 1.0, 2.0, 3.0]), diff=False))
    def test_event_find_first_cell_count_increase_after_cell_type_conversion_4(self):
        data = me.Data('data_init')
        assert((False, None) == me.Data.event_find_first_cell_count_increase_after_cell_type_conversion(data,
                       np.array([[4.0, 4.0, 5.0, 6.0],
                                [1.0, 1.0, 1.0, 1.0]]),
                       np.array([0.0, 1.0, 2.0, 3.0])))

    def test_event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion_1(self):
        data = me.Data('data_init')
        assert((True, 1.0) == me.Data.event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(
                        data,
                        np.array([[4.0, 3.0, 3.0, 4.0, 5.0],
                                  [1.0, 2.0, 2.0, 2.0, 2.0]]),
                        np.array([0.0, 1.0, 2.0, 3.0, 4.0]), diff=True))
    def test_event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion_2(self):
        data = me.Data('data_init')
        assert((True, 4.0) == me.Data.event_find_second_cell_count_increase_after_first_cell_count_increase_after_cell_type_conversion(
                        data,
                        np.array([[4.0, 3.0, 3.0, 4.0, 5.0],
                                  [1.0, 2.0, 2.0, 2.0, 2.0]]),
                        np.array([0.0, 1.0, 2.0, 3.0, 4.0]), diff=False))

    def test_event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion_1(self):
        data = me.Data('data_init')
        assert((True, 1.0) == me.Data.event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion(
                            data,
                        np.array([[4.0, 3.0, 3.0, 4.0, 5.0, 5.0],
                                  [1.0, 2.0, 2.0, 2.0, 2.0, 3.0]]),
                        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), diff=True))
    def test_event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion_2(self):
        data = me.Data('data_init')
        assert((True, 5.0) == me.Data.event_find_third_cell_count_increase_after_first_and_second_cell_count_increase_after_cell_type_conversion(
                            data,
                        np.array([[4.0, 3.0, 3.0, 4.0, 5.0, 5.0],
                                  [1.0, 2.0, 2.0, 2.0, 2.0, 3.0]]),
                        np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]), diff=False))

    ### test gamma histogram fitting
    def test_gamma_compute_bin_probabilities_sum(self):
        data = me.Data('data_init')
        data_time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        data.gamma_fit_bins = np.concatenate(([-np.inf], data_time_values, [np.inf]))
        assert(0.9999 < sum(data.gamma_compute_bin_probabilities([4.0, 0.5])) < 1.0001)
    def test_gamma_compute_bin_probabilities_values(self):
        data = me.Data('data_init')
        data_time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        data.gamma_fit_bins = np.concatenate(([-np.inf], data_time_values, [np.inf]))
        res = np.array([0.        , 0.14287654, 0.42365334, 0.28226624, 0.10882377, 0.03204406, 0.01033605])
        lower_res = res - 0.0001
        uppper_res = res + 0.0001
        assert(np.all([np.all(lower_res < data.gamma_compute_bin_probabilities([4.0, 0.5])),
                       np.all(data.gamma_compute_bin_probabilities([4.0, 0.5]) < uppper_res)]) == True)
    def test_check_bin_digitalisation(self):
        data = me.Data('data_init')
        data_time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        data.gamma_fit_bins = np.concatenate(([-np.inf], data_time_values, [np.inf]))
        assert(np.all(np.array([0, 1, 1, 2, 2, 6]) == np.digitize([0.0, 0.1, 1.0, 1.8, 2.0, 5.2], data.gamma_fit_bins, right=True) - 1) == True)
    def test_gamma_fit_binned_waiting_times(self):
        data = me.Data('data_init')
        data.data_time_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        theta = [4.0, 0.5]
        waiting_times_arr = np.random.gamma(theta[0], theta[1], 100000)
        data.gamma_fit_binned_waiting_times(waiting_times_arr)
        theta_fit = data.gamma_fit_theta
        assert(3.8 < theta_fit[0] < 4.2)
        assert(0.4 < theta_fit[1] < 0.6)

    ### test load method
    # @pytest.mark.slow
    def test_load_count_data(self):
        variables = ['A', 'B']
        time_values = np.linspace(0.0, 4.0, num=5)
        count_data = np.array([[[0.0, 0.0, 2.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0, 0.0]],
                             [[0.0, 1.0, 2.0, 4.0, 4.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                             [[0.0, 1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0, 2.0, 4.0], [1.0, 0.0, 0.0, 0.0, 0.0]]])
        data = me.Data('data_init')
        data.load(variables, time_values, count_data)
        sol_mean = np.array([[[0.,         0.5,        1.25,       3.,         4.        ],
                              [1.,         0.75,       0.25,       0.25,       0.        ]],
                             [[0.,         0.25096378, 0.41313568, 0.50182694, 0.        ],
                              [0.,         0.21847682, 0.21654396, 0.21624184, 0.        ]]])
        sol_var = np.array([[[0.,         0.33333333, 0.91666667, 1.33333333, 0.,        ],
                      [0.,         0.25,       0.25,       0.25,       0.,        ]],
                     [[0.,         0.10247419, 0.39239737, 0.406103,   0.,        ],
                      [0.,         0.13197367, 0.13279328, 0.13272021, 0.,        ]]])
        sol_cov = np.array([[[ 0.,          0.16666667,  0.25,       -0.33333333,  0.        ]],
                             [[ 0.,          0.11379549,  0.18195518,  0.22722941,  0.        ]]])
        np.testing.assert_allclose(sol_mean, data.data_mean, rtol=0.1)
        np.testing.assert_allclose(sol_var, data.data_variance, rtol=0.1)
        np.testing.assert_allclose(sol_cov, data.data_covariance, rtol=0.1)

        assert(data.data_mean_exists_only == False)
        assert(data.data_num_variables == 2)
        assert(data.data_num_time_values == 5)
        assert(data.data_mean_order == [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)}])
        assert(data.data_variance_order == [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
                                             {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)}])
        assert(data.data_covariance_order == [{'variables': ('A', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)}])
        assert(data.data_type == 'counts')
        assert(data.data_num_values == 25)
        assert(data.data_num_values_mean_only == 10)

    def test_load_summary_data(self):
        variables = ['A', 'B']
        time_values = np.linspace(0.0, 4.0, num=5)
        sol_mean = np.array([[[0.,         0.5,        1.25,       3.,         4.        ],
                              [1.,         0.75,       0.25,       0.25,       0.        ]],
                             [[0.,         0.25096378, 0.41313568, 0.50182694, 0.        ],
                              [0.,         0.21847682, 0.21654396, 0.21624184, 0.        ]]])
        sol_var = np.array([[[0.,         0.33333333, 0.91666667, 1.33333333, 0.,        ],
                      [0.,         0.25,       0.25,       0.25,       0.,        ]],
                     [[0.,         0.10247419, 0.39239737, 0.406103,   0.,        ],
                      [0.,         0.13197367, 0.13279328, 0.13272021, 0.,        ]]])
        sol_cov = np.array([[[ 0.,          0.16666667,  0.25,       -0.33333333,  0.        ]],
                             [[ 0.,          0.11379549,  0.18195518,  0.22722941,  0.        ]]])
        data = me.Data('data_init')
        data.load(variables, time_values, None, data_type='summary',
                    mean_data=sol_mean, var_data=sol_var, cov_data=sol_cov)
        np.testing.assert_allclose(sol_mean, data.data_mean)
        np.testing.assert_allclose(sol_var, data.data_variance)
        np.testing.assert_allclose(sol_cov, data.data_covariance)

        assert(data.data_mean_exists_only == False)
        assert(data.data_num_variables == 2)
        assert(data.data_num_time_values == 5)
        assert(data.data_mean_order == [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)}])
        assert(data.data_variance_order == [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
                                             {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)}])
        assert(data.data_covariance_order == [{'variables': ('A', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)}])
        assert(data.data_type == 'summary')
        assert(data.data_num_values == 25)
        assert(data.data_num_values_mean_only == 10)

    def test_load_summary_data_mean_only_1(self):
        variables = ['A', 'B']
        time_values = np.linspace(0.0, 4.0, num=5)
        sol_mean = np.array([[[0.,         0.5,        1.25,       3.,         4.        ],
                              [1.,         0.75,       0.25,       0.25,       0.        ]],
                             [[0.,         0.25096378, 0.41313568, 0.50182694, 0.        ],
                              [0.,         0.21847682, 0.21654396, 0.21624184, 0.        ]]])
        sol_var = np.empty((2, 0, 5))
        sol_cov = np.empty((2, 0, 5))
        data = me.Data('data_init')
        data.load(variables, time_values, None, data_type='summary',
                        mean_data=sol_mean, var_data=sol_var, cov_data=sol_cov)
        np.testing.assert_allclose(sol_mean, data.data_mean)
        np.testing.assert_allclose(sol_var, data.data_variance)
        np.testing.assert_allclose(sol_cov, data.data_covariance)

        assert(data.data_mean_exists_only == True)
        assert(data.data_num_variables == 2)
        assert(data.data_num_time_values == 5)
        assert(data.data_mean_order == [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)}])
        assert(data.data_variance_order == [])
        assert(data.data_covariance_order == [])
        assert(data.data_type == 'summary')
        assert(data.data_num_values == 10)
        assert(data.data_num_values_mean_only == 10)

    def test_load_summary_data_mean_only_2(self):
        variables = ['A', 'B']
        time_values = np.linspace(0.0, 4.0, num=5)
        sol_mean = np.array([[[0.,         0.5,        1.25,       3.,         4.        ],
                              [1.,         0.75,       0.25,       0.25,       0.        ]],
                             [[0.,         0.25096378, 0.41313568, 0.50182694, 0.        ],
                              [0.,         0.21847682, 0.21654396, 0.21624184, 0.        ]]])
        sol_var = np.empty((2, 0, 5))
        sol_cov = np.empty((2, 0, 5))
        data = me.Data('data_init')
        data.load(variables, time_values, None, data_type='summary', mean_data=sol_mean)
        np.testing.assert_allclose(sol_mean, data.data_mean)
        np.testing.assert_allclose(sol_var, data.data_variance)
        np.testing.assert_allclose(sol_cov, data.data_covariance)

        assert(data.data_mean_exists_only == True)
        assert(data.data_num_variables == 2)
        assert(data.data_num_time_values == 5)
        assert(data.data_mean_order == [{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                                          {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)}])
        assert(data.data_variance_order == [])
        assert(data.data_covariance_order == [])
        assert(data.data_type == 'summary')
        assert(data.data_num_values == 10)
        assert(data.data_num_values_mean_only == 10)
