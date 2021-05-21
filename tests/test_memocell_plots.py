
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import pytest
import memocell as me
import numpy as np
import matplotlib.pyplot as plt

# NOTE: for the plots module we do simple 'run-through' tests which would at least
# signal some bugs..

class TestPlotsModule(object):
    ### plots methods for networks
    def test_plots_net_main(self):
        net = me.Network('net_div_l2')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                'rate_symbol': 'l',
                'type': 'S -> S + S', 'reaction_steps': 2}])
        me.plots.net_main_plot(net, show=False)
        plt.close('all')

    def test_plots_net_hidden(self):
        net = me.Network('net_div_l2')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                'rate_symbol': 'l',
                'type': 'S -> S + S', 'reaction_steps': 2}])
        me.plots.net_hidden_plot(net, show=False)
        plt.close('all')

    ### plot methods for data
    @pytest.fixture(scope='class')
    def data_setup(self):
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
        return data

    def test_plots_data_mean_plot(self, data_setup):
        me.plots.data_mean_plot(data_setup, show=False)
        plt.close('all')

    def test_plots_data_var_plot(self, data_setup):
        me.plots.data_variance_plot(data_setup, show=False)
        plt.close('all')

    def test_plots_data_cov_plot(self, data_setup):
        me.plots.data_covariance_plot(data_setup, show=False)
        plt.close('all')

    def test_plots_data_var_scatter(self, data_setup):
        me.plots.data_variable_scatter_plot(data_setup, -1, 'A', 'B', show=False)
        plt.close('all')

    def test_plots_data_var_hist(self, data_setup):
        me.plots.data_hist_variables_plot(data_setup, -1, show=False)
        plt.close('all')

    def test_plots_data_waiting_time_hist(self, data_setup):
        data_setup.events_find_all()
        me.plots.data_hist_waiting_times_plot(data_setup, data_setup.event_all_first_cell_type_conversion, show=False)
        plt.close('all')

    def test_plots_data_waiting_time_hist_no_gamma(self, data_setup):
        data_setup.events_find_all()
        me.plots.data_hist_waiting_times_plot(data_setup, data_setup.event_all_first_cell_type_conversion, gamma_fit=False, show=False)
        plt.close('all')

    ### plot methods for moment and gillespie simulations
    @pytest.fixture(scope='class')
    def sim_l2_moments_setup(self):
        net = me.Network('net_div_l2')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 2}])
        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('X_t','X_t'): 0.0}
        theta_values = {'l': 0.15}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)
        return sim

    def test_plots_sim_mean_plot(self, sim_l2_moments_setup):
        me.plots.sim_mean_plot(sim_l2_moments_setup, show=False)
        plt.close('all')

    def test_plots_sim_var_plot(self, sim_l2_moments_setup):
        me.plots.sim_variance_plot(sim_l2_moments_setup, show=False)
        plt.close('all')

    def test_plots_sim_cov_plot(self, sim_l2_moments_setup):
        me.plots.sim_covariance_plot(sim_l2_moments_setup, show=False)
        plt.close('all')

    @pytest.fixture(scope='class')
    def sim_l2_gillespie_setup(self):
        net = me.Network('net_div_l2')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 2}])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.15}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        num_iter = 1
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])
        return sim

    def test_plots_sim_counts_plot(self, sim_l2_gillespie_setup):
        me.plots.sim_counts_plot(sim_l2_gillespie_setup, show=False)
        plt.close('all')

    ### plots methods for estimation and selection
    @pytest.fixture(scope='class')
    def est_sel_setup(self):
        net = me.Network('net_div_g5')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 5}])

        num_iter = 5
        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.22}
        time_values = np.array([0.0, 10.0])
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                                initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, sims, bootstrap_samples=5, basic_sigma=0.01)

        # overwrite with known values (from a num_iter=100 simulation)
        data.data_mean = np.array([[[1.         , 3.73      ]],
                                   [[0.01       , 0.15406761]]])
        data.data_variance = np.array([[[0.         , 2.36070707]],
                                       [[0.01       , 0.32208202]]])

        ### define model(s) for selection
        net2 = me.Network('net_div_g2')
        net2.structure([{'start': 'X_t', 'end': 'X_t',
                         'rate_symbol': 'l',
                         'type': 'S -> S + S',
                         'reaction_steps': 2}])

        # important note: theta_bounds are reduced here to
        # prevent odeint warning at high steps
        networks = [net2]
        variables = [{'X_t': ('X_t', )}]*1
        initial_values_types = ['synchronous']*1
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*1
        theta_bounds = [{'l': (0.0, 0.3)}]*1

        ### run selection (sequentially)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
                                            theta_bounds, data, parallel=False)
        return est_res

    @pytest.mark.slow
    def test_plots_sel_plot_evidence(self, est_sel_setup):
        me.plots.selection_plot(est_sel_setup, est_type='evidence', show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_sel_plot_likelihood(self, est_sel_setup):
        me.plots.selection_plot(est_sel_setup, est_type='likelihood', show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_sel_plot_bic(self, est_sel_setup):
        me.plots.selection_plot(est_sel_setup, est_type='bic', show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_sel_plot_evidence_from_bic(self, est_sel_setup):
        me.plots.selection_plot(est_sel_setup, est_type='evidence_from_bic', show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_mean_plot(self, est_sel_setup):
        me.plots.est_bestfit_mean_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_var_plot(self, est_sel_setup):
        me.plots.est_bestfit_variance_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_cov_plot(self, est_sel_setup):
        me.plots.est_bestfit_covariance_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_mean_plot_no_band(self, est_sel_setup):
        me.plots.est_bestfit_mean_plot(est_sel_setup[0], cred=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_var_plot_no_band(self, est_sel_setup):
        me.plots.est_bestfit_variance_plot(est_sel_setup[0], cred=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_cov_plot_no_band(self, est_sel_setup):
        me.plots.est_bestfit_covariance_plot(est_sel_setup[0], cred=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_mean_plot_no_data(self, est_sel_setup):
        me.plots.est_bestfit_mean_plot(est_sel_setup[0], data=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_var_plot_no_data(self, est_sel_setup):
        me.plots.est_bestfit_variance_plot(est_sel_setup[0], data=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_cov_plot_no_data(self, est_sel_setup):
        me.plots.est_bestfit_covariance_plot(est_sel_setup[0], data=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_mean_plot_no_band_no_data(self, est_sel_setup):
        me.plots.est_bestfit_mean_plot(est_sel_setup[0], data=False, cred=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_var_plot_no_band_no_data(self, est_sel_setup):
        me.plots.est_bestfit_variance_plot(est_sel_setup[0], data=False, cred=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_cov_plot_no_band_no_data(self, est_sel_setup):
        me.plots.est_bestfit_covariance_plot(est_sel_setup[0], data=False, cred=False, show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_chains_plot(self, est_sel_setup):
        me.plots.est_chains_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_corner_kernel_plot(self, est_sel_setup):
        me.plots.est_corner_kernel_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_corner_plot(self, est_sel_setup):
        me.plots.est_corner_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_parameter_plot(self, est_sel_setup):
        me.plots.est_parameter_plot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_run_plot(self, est_sel_setup):
        me.plots.est_runplot(est_sel_setup[0], show=False)
        plt.close('all')

    @pytest.mark.slow
    def test_plots_est_trace_plot(self, est_sel_setup):
        me.plots.est_traceplot(est_sel_setup[0], show=False)
        plt.close('all')

    # est_corner_weight_plot needs 2d (or more) posterior
    @pytest.mark.slow
    def test_plots_est_corner_weight_plot(self):
        net = me.Network('net_div_g5')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 5}])

        num_iter = 5
        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.22}
        time_values = np.array([0.0, 10.0])
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                                initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, sims, bootstrap_samples=5, basic_sigma=0.01)

        # overwrite with known values (from a num_iter=100 simulation)
        data.data_mean = np.array([[[1.         , 3.73      ]],
                                   [[0.01       , 0.15406761]]])
        data.data_variance = np.array([[[0.         , 2.36070707]],
                                       [[0.01       , 0.32208202]]])

        ### define model(s) for selection
        net2 = me.Network('net_div_g2')
        net2.structure([{'start': 'X_t', 'end': 'X_t',
                         'rate_symbol': 'l',
                         'type': 'S -> S + S',
                         'reaction_steps': 2},
                         {'start': 'X_t', 'end': 'env',
                          'rate_symbol': 'd',
                          'type': 'S ->',
                          'reaction_steps': 1}])

        # important note: theta_bounds are reduced here to
        # prevent odeint warning at high steps
        networks = [net2]
        variables = [{'X_t': ('X_t', )}]*1
        initial_values_types = ['synchronous']*1
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*1
        theta_bounds = [{'l': (0.0, 0.3), 'd': (0.0, 0.1)}]*1

        ### run selection (sequentially)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
                                            theta_bounds, data, parallel=False)

        me.plots.est_corner_weight_plot(est_res[0], show=False)
        plt.close('all')
