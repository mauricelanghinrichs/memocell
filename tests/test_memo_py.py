# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memo_py.py"
# or after pip installation $py.test --pyargs memo_py$

import memo_py as me
import numpy as np

# TODO: adapt expected output in those tests
class TestNetworkClass(object):
    def test_one(self):
        assert(1 == 1)

    # def test_network_structure_input_one_reaction_nodes(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_main.nodes()) == ['X_t', 'Y_t'])
    #
    # def test_network_structure_input_one_reaction_edges(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_main.edges()) == [('X_t', 'Y_t')])
    #
    # def test_network_structure_input_one_reaction_edges_with_data(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_main.edges(data=True)) == [('X_t', 'Y_t', {'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1})])
    #
    # def test_network_structure_input_two_reactions_node_identifier(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
    #     {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(net.create_node_identifiers(net.net_main) == {'X_t': 'Z_0', 'Y_t': 'Z_1', 'env': 'Z_env'})
    #
    # def test_network_structure_input_two_reactions_rate_identifier(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
    #     {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(net.create_rate_identifiers(net.net_main) == {'k_xy': 'theta_0'})
    #
    # def test_network_structure_input_two_reactions_modules(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
    #     {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(net.net_modules == [
    #         {'module': 'module_0', 'start-end': ('X_t', 'Y_t'), 'start-end_ident': ('Z_0', 'Z_1'), 'sym_rate': 'k_xy', 'sym_rate_ident': 'theta_0', 'type': 'S -> E', 'module_steps': 1},
    #         {'module': 'module_1', 'start-end': ('Y_t', 'env'), 'start-end_ident': ('Z_1', 'Z_env'), 'sym_rate': 'k_xy', 'sym_rate_ident': 'theta_0', 'type': 'S ->', 'module_steps': 1}
    #         ])
    #
    # def test_hidden_network_structure_input_one_reaction_nodes(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_hidden.nodes()) == ['X_t', 'Y_t'])
    #
    # def test_hidden_network_structure_input_one_reaction_edges(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_hidden.edges()) == [('X_t', 'Y_t')])
    #
    # def test_hidden_network_structure_input_one_reaction_edges_with_data(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_hidden.edges(data=True)) == [('X_t', 'Y_t', {'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1})])
    #
    # def test_hidden_network_structure_input_two_reactions_nodes(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
    #     {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_hidden.nodes()) == ['X_t', 'Y_t'])
    #
    # def test_hidden_network_structure_input_two_reactions_edges(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
    #     {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_hidden.edges()) == [('X_t', 'Y_t')])
    #
    # def test_hidden_network_structure_input_two_reactions_edges_with_data(self):
    #     network_structure_input = [
    #     {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
    #     {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
    #     ]
    #     net = me.Network('net_name')
    #     net.structure(network_structure_input)
    #     assert(list(net.net_hidden.edges(data=True)) == [('X_t', 'Y_t', {'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1})])


class TestDataClass(object):
    ### tests for create_data_variable_order()
    def test_create_data_variable_order_1(self):
        assert(me.Data.create_data_variable_order(['A', 'B', 'C']) == ([{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                                                                      {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)},
                                                                      {'variables': 'C', 'summary_indices': 2, 'count_indices': (2,)}],
                                                                     [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
                                                                      {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)},
                                                                      {'variables': ('C', 'C'), 'summary_indices': 2, 'count_indices': (2, 2)}],
                                                                     [{'variables': ('A', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)},
                                                                      {'variables': ('A', 'C'), 'summary_indices': 1, 'count_indices': (0, 2)},
                                                                      {'variables': ('B', 'C'), 'summary_indices': 2, 'count_indices': (1, 2)}]))

    def test_create_data_variable_order_no_alphabetical_order(self):
        assert(me.Data.create_data_variable_order(['C', 'B', 'A']) == ([{'variables': 'C', 'summary_indices': 0, 'count_indices': (0,)},
                                                                      {'variables': 'B', 'summary_indices': 1, 'count_indices': (1,)},
                                                                      {'variables': 'A', 'summary_indices': 2, 'count_indices': (2,)}],
                                                                     [{'variables': ('C', 'C'), 'summary_indices': 0, 'count_indices': (0, 0)},
                                                                      {'variables': ('B', 'B'), 'summary_indices': 1, 'count_indices': (1, 1)},
                                                                      {'variables': ('A', 'A'), 'summary_indices': 2, 'count_indices': (2, 2)}],
                                                                     [{'variables': ('C', 'B'), 'summary_indices': 0, 'count_indices': (0, 1)},
                                                                      {'variables': ('C', 'A'), 'summary_indices': 1, 'count_indices': (0, 2)},
                                                                      {'variables': ('B', 'A'), 'summary_indices': 2, 'count_indices': (1, 2)}]))

    def test_create_data_variable_order_no_validation_here(self):
        assert(me.Data.create_data_variable_order(['A', 'A']) == ([{'variables': 'A', 'summary_indices': 0, 'count_indices': (0,)},
                                                                      {'variables': 'A', 'summary_indices': 1, 'count_indices': (1,)}],
                                                                     [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 0)},
                                                                      {'variables': ('A', 'A'), 'summary_indices': 1, 'count_indices': (1, 1)}],
                                                                     [{'variables': ('A', 'A'), 'summary_indices': 0, 'count_indices': (0, 1)}]))
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

    def test_load(self):
        data = me.Data('data_init')
        variables = ['A', 'B']
        time_values = np.linspace(0.0, 4.0, num=5)
        count_data = np.array([[[0.0, 0.0, 2.0, 2.0, 4.0], [1.0, 1.0, 1.0, 1.0, 0.0]],
                             [[0.0, 1.0, 2.0, 4.0, 4.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                             [[0.0, 1.0, 1.0, 4.0, 4.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0, 2.0, 4.0], [1.0, 0.0, 0.0, 0.0, 0.0]]])
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


class TestSelectionModule(object):
    def test_bayes_factor_calculation(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        res = me.selection.compute_model_bayes_factors_from_log_evidences(logevids)
        sol = np.array([  1.34985881,  13.46373804,   1.        , 403.42879349])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_uniform_prior_default(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids)
        sol = np.array([0.40758705, 0.04086421, 0.55018497, 0.00136377])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_uniform_prior_default_sum(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids)
        np.testing.assert_allclose(1.0, np.sum(res), rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_uniform_prior_explicit(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        mprior = np.array([0.25, 0.25, 0.25, 0.25])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
        sol = np.array([0.40758705, 0.04086421, 0.55018497, 0.00136377])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_non_uniform_prior(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        mprior = np.array([0.3, 0.3, 0.2, 0.2])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
        sol = np.array([0.49940188, 0.05006945, 0.44941468, 0.00111399])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_non_uniform_prior_sum(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        mprior = np.array([0.3, 0.3, 0.2, 0.2])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
        np.testing.assert_allclose(1.0, np.sum(res), rtol=1e-06, atol=1e-06)
