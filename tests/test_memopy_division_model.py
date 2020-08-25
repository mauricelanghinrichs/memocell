
import pytest
import memo_py as me
import numpy as np

class TestSelectionDivisionModel(object):
    @pytest.mark.slow
    def test_division_model_sequential(self):
        ### create data from 5-steps model
        net = me.Network('net_div_g5')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 5}])

        num_iter = 100
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.22}
        time_values = np.array([0.0, 10.0])
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', initial_values, theta_values, time_values, variables)[1])

        sims = np.array(res_list)

        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, sims, bootstrap_samples=10000, basic_sigma=1/num_iter)

        # overwrite with known values (from a num_iter=100 simulation)
        data.data_mean = np.array([[[1.         , 3.73      ]],
                                   [[0.01       , 0.15406761]]])
        data.data_variance = np.array([[[0.         , 2.36070707]],
                                       [[0.01       , 0.32208202]]])

        ### define models for selection
        net2 = me.Network('net_div_g2')
        net2.structure([{'start': 'X_t', 'end': 'X_t',
                         'rate_symbol': 'l',
                         'type': 'S -> S + S',
                         'reaction_steps': 2}])

        net5 = me.Network('net_div_g5')
        net5.structure([{'start': 'X_t', 'end': 'X_t',
                         'rate_symbol': 'l',
                         'type': 'S -> S + S',
                         'reaction_steps': 5}])

        net15 = me.Network('net_div_g15')
        net15.structure([{'start': 'X_t', 'end': 'X_t',
                          'rate_symbol': 'l',
                          'type': 'S -> S + S',
                          'reaction_steps': 15}])

        # important note: theta_bounds are reduced here to
        # prevent odeint warning at high steps
        networks = [net2, net5, net15]
        variables = [{'X_t': ('X_t', )}]*3
        initial_values = [{'X_t': 1.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (sequentially)
        est_res = me.selection.select_models(networks, variables, initial_values,
                                            theta_bounds, data, parallel=False)

        ### assert log evidence values
        evid_res = np.array([est.bay_est_log_evidence for est in est_res])
        # expected solution
        evid_summary = np.array([
            [-10.629916697804973, 4.7325886118420755, -5.956029975676918],
            [-10.707752281691006, 4.648483811563277, -5.904935097433867],
            [-10.660052905403797, 4.786992306046756, -5.865460195018101],
            [-10.560456631689906, 4.712198906183986, -6.024845876715448],
            [-10.67465188224791, 4.739031922471972, -5.832238436663494],
            [-10.688286571898711, 4.796078612214922, -5.866083545090796],
            [-10.674939299757629, 4.763798403008956, -5.869504378082593],
            [-10.706803542779763, 4.758241853964541, -5.921262356541525],
            [-10.55770560556853, 4.770112513127923, -5.9405505492747315],
            [-10.660075345658038, 4.832557590020761, -5.869471476946568]
        ])
        evid_sol = np.mean(evid_summary, axis=0)
        evid_tol = np.max(np.std(evid_summary, axis=0)*6)

        np.testing.assert_allclose(evid_sol, evid_res, rtol=evid_tol, atol=evid_tol)

        ### assert maximal log likelihood value
        logl_max_res = np.array([est.bay_est_log_likelihood_max for est in est_res])
        logl_max_sol = np.array([-6.655917426977538, 8.536409735755383, -2.583752295846338])
        np.testing.assert_allclose(logl_max_sol, logl_max_res, rtol=1.0, atol=1.0)

        ### assert estimated parameter value (95% credible interval)
        theta_cred_res = np.array([est.bay_est_params_cred for est in est_res])
        theta_cred_sol = [[[0.15403995, 0.14659241, 0.16079126]],
                             [[0.21181266, 0.20239213, 0.22030757]],
                             [[0.23227569, 0.21840278, 0.24577387]]]
        np.testing.assert_allclose(theta_cred_sol, theta_cred_res, rtol=0.02, atol=0.02)

    @pytest.mark.slow
    def test_division_model_parallel(self):
        ### create data from 5-steps model
        t = [
            {'start': 'X_t', 'end': 'X_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 5}
            ]

        net = me.Network('net_div_g5')
        net.structure(t)

        num_iter = 100
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.22}
        time_values = np.array([0.0, 10.0])
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', initial_values, theta_values, time_values, variables)[1])

        sims = np.array(res_list)

        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, sims, bootstrap_samples=10000, basic_sigma=1/num_iter)

        # overwrite with known values (from a num_iter=100 simulation)
        data.data_mean = np.array([[[1.         , 3.73      ]],
                                   [[0.01       , 0.15406761]]])
        data.data_variance = np.array([[[0.         , 2.36070707]],
                                       [[0.01       , 0.32208202]]])

        ### define models for selection
        net2 = me.Network('net_div_g2')
        net2.structure([{'start': 'X_t', 'end': 'X_t',
                         'rate_symbol': 'l',
                         'type': 'S -> S + S',
                         'reaction_steps': 2}])

        net5 = me.Network('net_div_g5')
        net5.structure([{'start': 'X_t', 'end': 'X_t',
                         'rate_symbol': 'l',
                         'type': 'S -> S + S',
                         'reaction_steps': 5}])

        net15 = me.Network('net_div_g15')
        net15.structure([{'start': 'X_t', 'end': 'X_t',
                          'rate_symbol': 'l',
                          'type': 'S -> S + S',
                          'reaction_steps': 15}])

        # important note: theta_bounds are reduced here to
        # prevent odeint warning at high steps
        networks = [net2, net5, net15]
        variables = [{'X_t': ('X_t', )}]*3
        initial_values = [{'X_t': 1.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables, initial_values,
                                            theta_bounds, data, parallel=True)

        ### assert log evidence values
        evid_res = np.array([est.bay_est_log_evidence for est in est_res])
        # expected solution
        evid_summary = np.array([
            [-10.629916697804973, 4.7325886118420755, -5.956029975676918],
            [-10.707752281691006, 4.648483811563277, -5.904935097433867],
            [-10.660052905403797, 4.786992306046756, -5.865460195018101],
            [-10.560456631689906, 4.712198906183986, -6.024845876715448],
            [-10.67465188224791, 4.739031922471972, -5.832238436663494],
            [-10.688286571898711, 4.796078612214922, -5.866083545090796],
            [-10.674939299757629, 4.763798403008956, -5.869504378082593],
            [-10.706803542779763, 4.758241853964541, -5.921262356541525],
            [-10.55770560556853, 4.770112513127923, -5.9405505492747315],
            [-10.660075345658038, 4.832557590020761, -5.869471476946568]
        ])
        evid_sol = np.mean(evid_summary, axis=0)
        evid_tol = np.max(np.std(evid_summary, axis=0)*6)

        np.testing.assert_allclose(evid_sol, evid_res, rtol=evid_tol, atol=evid_tol)

        ### assert maximal log likelihood value
        logl_max_res = np.array([est.bay_est_log_likelihood_max for est in est_res])
        logl_max_sol = np.array([-6.655917426977538, 8.536409735755383, -2.583752295846338])
        np.testing.assert_allclose(logl_max_sol, logl_max_res, rtol=1.0, atol=1.0)

        ### assert estimated parameter value (95% credible interval)
        theta_cred_res = np.array([est.bay_est_params_cred for est in est_res])
        theta_cred_sol = [[[0.15403995, 0.14659241, 0.16079126]],
                             [[0.21181266, 0.20239213, 0.22030757]],
                             [[0.23227569, 0.21840278, 0.24577387]]]
        np.testing.assert_allclose(theta_cred_sol, theta_cred_res, rtol=0.02, atol=0.02)
