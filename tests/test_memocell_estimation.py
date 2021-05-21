
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import pytest
import memocell as me
import numpy as np

class TestEstimationClass(object):
    ### test on 1st and 2nd moments first, fit_mean_only later
    @pytest.fixture(scope='class')
    def simple_est_setup(self):
        # see jupyter notebook ex_docs_tests for more info and plots
        ### define network
        net = me.Network('net_min_2_4')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ])

        ### create data with known values
        num_iter = 100
        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'d': 0.03, 'l': 0.07}
        time_values = np.array([0.0, 20.0, 40.0])
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                            initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        data = me.Data('data_test_est_min_2_4')
        data.load(['X_t', 'Y_t'], time_values, sims,
                  bootstrap_samples=10000, basic_sigma=1/num_iter)

        # overwrite with fixed values (from a num_iter = 100 simulation)
        data.data_mean = np.array([[[1.         ,0.67       ,0.37      ],
                                  [0.         ,0.45       ,1.74      ]],
                                 [[0.01       ,0.0469473  ,0.04838822],
                                  [0.01       ,0.07188642 ,0.1995514 ]]])
        data.data_variance = np.array([[[0.         ,0.22333333 ,0.23545455],
                                      [0.         ,0.51262626 ,4.03272727]],
                                     [[0.01       ,0.01631605 ,0.01293869],
                                      [0.01       ,0.08878719 ,0.68612036]]])
        data.data_covariance = np.array([[[ 0.         ,-0.30454545 ,-0.65030303]],
                                     [[ 0.01        ,0.0303608   ,0.06645246]]])

        ### run estimation and return object
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        time_values = None
        sim_mean_only = False
        fit_mean_only = False

        nlive = 1000 # 250 # 1000
        tolerance = 0.01 # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
        bound = 'multi'
        sample = 'unif'

        est = me.Estimation('est_min_2_4', net, data)
        est.estimate(variables, initial_values_type, initial_values,
                            theta_bounds, time_values, sim_mean_only, fit_mean_only,
                            nlive, tolerance, bound, sample)
        return est

    @pytest.fixture(scope='class')
    def simple_est_setup_summary(self):
        # see jupyter notebook ex_docs_tests for more info and plots
        ### define network
        net = me.Network('net_min_2_4')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ])

        ### create data with known values
        time_values = np.array([0.0, 20.0, 40.0])
        data_mean = np.array([[[1.         ,0.67       ,0.37      ],
                                  [0.         ,0.45       ,1.74      ]],
                                 [[0.01       ,0.0469473  ,0.04838822],
                                  [0.01       ,0.07188642 ,0.1995514 ]]])
        data_variance = np.array([[[0.         ,0.22333333 ,0.23545455],
                                      [0.         ,0.51262626 ,4.03272727]],
                                     [[0.01       ,0.01631605 ,0.01293869],
                                      [0.01       ,0.08878719 ,0.68612036]]])
        data_covariance = np.array([[[ 0.         ,-0.30454545 ,-0.65030303]],
                                     [[ 0.01        ,0.0303608   ,0.06645246]]])
        data = me.Data('data_test_est_min_2_4')
        data.load(['X_t', 'Y_t'], time_values, None, 'summary',
                mean_data=data_mean, var_data=data_variance, cov_data=data_covariance)


        ### run estimation and return object
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        time_values = None
        sim_mean_only = False
        fit_mean_only = False

        nlive = 1000 # 250 # 1000
        tolerance = 0.01 # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
        bound = 'multi'
        sample = 'unif'

        est = me.Estimation('est_min_2_4', net, data)
        est.estimate(variables, initial_values_type, initial_values,
                            theta_bounds, time_values, sim_mean_only, fit_mean_only,
                            nlive, tolerance, bound, sample)
        return est

    @pytest.mark.slow
    def test_log_evid_of_simple_minimal_model_1(self, simple_est_setup):
        np.testing.assert_allclose(28.2, simple_est_setup.bay_est_log_evidence,
                                    rtol=1.0, atol=1.0)

    @pytest.mark.slow
    def test_log_evid_of_simple_minimal_model_2(self, simple_est_setup_summary):
        np.testing.assert_allclose(28.2, simple_est_setup_summary.bay_est_log_evidence,
                                    rtol=1.0, atol=1.0)

    @pytest.mark.slow
    def test_max_log_likelihood_of_simple_minimal_model_1(self, simple_est_setup):
        np.testing.assert_allclose(35.5, simple_est_setup.bay_est_log_likelihood_max,
                                    rtol=1.0, atol=1.0)

    @pytest.mark.slow
    def test_max_log_likelihood_of_simple_minimal_model_2(self, simple_est_setup_summary):
        np.testing.assert_allclose(35.5, simple_est_setup_summary.bay_est_log_likelihood_max,
                                    rtol=1.0, atol=1.0)

    @pytest.mark.slow
    def test_theta_credible_interval_of_simple_minimal_model_1(self, simple_est_setup):
        sol_bay_est_params_cred = np.array([[[0.02803474, 0.02594989, 0.03014408],
                                              [0.07470537, 0.06919784, 0.07955645]]])
        np.testing.assert_allclose(sol_bay_est_params_cred,
                                    np.array([simple_est_setup.bay_est_params_cred]),
                                    rtol=0.002, atol=0.002)

    @pytest.mark.slow
    def test_theta_credible_interval_of_simple_minimal_model_2(self, simple_est_setup_summary):
        sol_bay_est_params_cred = np.array([[[0.02803474, 0.02594989, 0.03014408],
                                              [0.07470537, 0.06919784, 0.07955645]]])
        np.testing.assert_allclose(sol_bay_est_params_cred,
                                    np.array([simple_est_setup_summary.bay_est_params_cred]),
                                    rtol=0.002, atol=0.002)

    @pytest.mark.slow
    def test_get_model_evidence_from_sampler_res(self, simple_est_setup):
        sampler_result = simple_est_setup.bay_nested_sampler_res
        logZ, logzerr = simple_est_setup.get_model_evidence(sampler_result)
        np.testing.assert_allclose(28.2, logZ, rtol=1.0, atol=1.0)
        np.testing.assert_allclose(0.1, logzerr, rtol=0.1, atol=0.1)

    @pytest.mark.slow
    def test_get_maximal_log_likelihood_from_sampler_res(self, simple_est_setup):
        sampler_result = simple_est_setup.bay_nested_sampler_res
        logl_max = simple_est_setup.get_maximal_log_likelihood(sampler_result)
        np.testing.assert_allclose(35.5, logl_max, rtol=1.0, atol=1.0)

    @pytest.mark.slow
    def test_compute_bayesian_information_criterion_from_sampler_res(self, simple_est_setup):
        bic = simple_est_setup.compute_bayesian_information_criterion(
                simple_est_setup.data_num_values,
                simple_est_setup.bay_nested_ndims,
                simple_est_setup.bay_est_log_likelihood_max)
        np.testing.assert_allclose(-65.5, bic, rtol=1.0, atol=1.0)

    def test_compute_bayesian_information_criterion(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)
        bic = est.compute_bayesian_information_criterion(15, 2, 35.49)
        np.testing.assert_allclose(-65.56389959779558, bic)
        bic = est.compute_bayesian_information_criterion(15.0, 2, 35.49)
        np.testing.assert_allclose(-65.56389959779558, bic)
        bic = est.compute_bayesian_information_criterion(15, 2.0, 35.49)
        np.testing.assert_allclose(-65.56389959779558, bic)
        bic = est.compute_bayesian_information_criterion(15.0, 2.0, 35.49)
        np.testing.assert_allclose(-65.56389959779558, bic)

    @pytest.mark.slow
    def test_compute_log_evidence_from_bic_from_sampler_res(self, simple_est_setup):
        log_evid_from_bic = simple_est_setup.compute_log_evidence_from_bic(
                simple_est_setup.bay_est_bayesian_information_criterion)
        np.testing.assert_allclose(32.8, log_evid_from_bic, rtol=1.0, atol=1.0)

    def test_compute_log_evidence_from_bic(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)
        log_evid_from_bic = est.compute_log_evidence_from_bic(-65.5)
        np.testing.assert_allclose(32.75, log_evid_from_bic)

    @pytest.mark.slow
    def test_get_posterior_samples_from_sampler_res(self, simple_est_setup):
        s, sw, w = simple_est_setup.get_posterior_samples(
                simple_est_setup.bay_nested_sampler_res)
        # s and sw shape should be (nsamples, nparams)
        assert(2 == s.shape[1])
        assert(2 == sw.shape[1])

    def test_get_credible_interval(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)
        samples = np.array([[0.2, 3.4], [0.4, 3.2], [0.25, 3.65]])
        params_cred_res = np.array(est.get_credible_interval(samples))
        params_cred_sol = np.array([[0.25  , 0.2025, 0.3925],
                                 [3.4   , 3.21  , 3.6375]])
        np.testing.assert_allclose(params_cred_sol, params_cred_res)

    def test_compute_log_likelihood_norm_mean_only_false(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)
        mean_data = np.array([[[1., 0.67, 0.37],
                               [0., 0.45, 1.74]],
                              [[0.01, 0.0469473, 0.04838822],
                               [0.01, 0.07188642, 0.1995514]]])
        var_data = np.array([[[0., 0.22333333, 0.23545455],
                              [0., 0.51262626, 4.03272727]],
                             [[0.01, 0.01631605, 0.01293869],
                              [0.01, 0.08878719, 0.68612036]]])
        cov_data = np.array([[[ 0., -0.30454545, -0.65030303]],
                             [[ 0.01, 0.0303608, 0.06645246]]])
        norm_res = est.compute_log_likelihood_norm(mean_data, var_data, cov_data, False)
        norm_sol = 37.04057852140377
        np.testing.assert_allclose(norm_sol, norm_res)

    def test_compute_log_likelihood_norm_mean_only_true(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)
        mean_data = np.array([[[1., 0.67, 0.37],
                               [0., 0.45, 1.74]],
                              [[0.01, 0.0469473, 0.04838822],
                               [0.01, 0.07188642, 0.1995514]]])
        var_data = np.array([[[0., 0.22333333, 0.23545455],
                              [0., 0.51262626, 4.03272727]],
                             [[0.01, 0.01631605, 0.01293869],
                              [0.01, 0.08878719, 0.68612036]]])
        cov_data = np.array([[[ 0., -0.30454545, -0.65030303]],
                             [[ 0.01, 0.0303608, 0.06645246]]])
        norm_res = est.compute_log_likelihood_norm(mean_data, var_data, cov_data, True)
        norm_sol = 14.028288976285737
        np.testing.assert_allclose(norm_sol, norm_res)

    @pytest.mark.slow
    def test_compute_log_likelihood_norm_mean_only_false_from_sampler_res(self, simple_est_setup):
        norm_res = simple_est_setup.bay_log_likelihood_norm
        norm_sol = 37.04057852140377
        np.testing.assert_allclose(norm_sol, norm_res)

    @pytest.mark.slow
    def test_log_likelihood_from_sampler_res(self, simple_est_setup):
        theta_values = np.array([0.03, 0.07])
        logl_res = simple_est_setup.log_likelihood(theta_values,
                   simple_est_setup.net_simulation.sim_moments.moment_initial_values,
                   simple_est_setup.net_time_values, simple_est_setup.net_time_ind,
                   simple_est_setup.data_mean_values, simple_est_setup.data_var_values,
                   simple_est_setup.data_cov_values)
        logl_sol = 32.823084036435795
        np.testing.assert_allclose(logl_sol, logl_res)

        theta_values = np.array([0.028, 0.075])
        logl_res = simple_est_setup.log_likelihood(theta_values,
                   simple_est_setup.net_simulation.sim_moments.moment_initial_values,
                   simple_est_setup.net_time_values, simple_est_setup.net_time_ind,
                   simple_est_setup.data_mean_values, simple_est_setup.data_var_values,
                   simple_est_setup.data_cov_values)
        logl_sol = 35.485136238014185
        np.testing.assert_allclose(logl_sol, logl_res)

    @pytest.mark.slow
    def test_prior_transform_from_sampler_res(self, simple_est_setup):
        theta_unit = np.array([0.03/0.15, 0.075/0.15])
        theta_orig_res = simple_est_setup.prior_transform(theta_unit)
        theta_orig_sol = np.array([0.03 , 0.075])
        np.testing.assert_allclose(theta_orig_sol, theta_orig_res)

    def test_prior_transform(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)

        est.net_theta_bounds = np.array([[0.  , 0.15],
                                         [0.  , 0.15]])
        theta_unit = np.array([0.03/0.15, 0.075/0.15])
        theta_orig_res = est.prior_transform(theta_unit)
        theta_orig_sol = np.array([0.03 , 0.075])
        np.testing.assert_allclose(theta_orig_sol, theta_orig_res)

        est.net_theta_bounds = np.array([[0.5  , 2.0],
                                         [0.1  , 4.0]])
        theta_unit = np.array([0.8, 0.2])
        theta_orig_res = est.prior_transform(theta_unit)
        theta_orig_sol = np.array([0.8 * (2.0 - 0.5) + 0.5,
                                   0.2 * (4.0 - 0.1) + 0.1])
        np.testing.assert_allclose(theta_orig_sol, theta_orig_res)

    def test_initialise_net_theta_bounds(self):
        net = me.Network('net_test')
        data = me.Data('data_test')
        est = me.Estimation('est_test', net, data)
        net_theta_bounds_res = est.initialise_net_theta_bounds(
                                        ['theta_0', 'theta_1', 'theta_2'],
                                        {'theta_2':'p3', 'theta_0':'p1', 'theta_1':'p2'},
                                        {'p2': (0.0, 0.1), 'p3': (0.0, 0.2), 'p1': (0.0, 0.3)})
        net_theta_bounds_sol = np.array([[0. , 0.3],
                                         [0. , 0.1],
                                         [0. , 0.2]])
        np.testing.assert_allclose(net_theta_bounds_sol, net_theta_bounds_res)

    @pytest.mark.slow
    def test_match_data_to_network_from_sampler_res(self, simple_est_setup):

        # check normal order of simulation variable identifiers
        assert({'V_0': ('X_t', ('X_t',)), 'V_1': ('Y_t', ('Y_t',))}==
                    simple_est_setup.net_simulation.sim_variables_identifier)
        data_order_res_1 = simple_est_setup.match_data_to_network(
                    simple_est_setup.net_simulation.sim_variables_order,
                    simple_est_setup.net_simulation.sim_variables_identifier,
                    simple_est_setup.data.data_mean,
                    simple_est_setup.data.data_variance,
                    simple_est_setup.data.data_covariance,
                    simple_est_setup.data.data_mean_order,
                    simple_est_setup.data.data_variance_order,
                    simple_est_setup.data.data_covariance_order)
        data_order_sol_1 = (
                    np.array([[[1., 0.67, 0.37],
                               [0., 0.45, 1.74]],
                              [[0.01, 0.0469473, 0.04838822],
                               [0.01, 0.07188642, 0.1995514]]]),
                    np.array([[[0., 0.22333333, 0.23545455],
                              [0., 0.51262626, 4.03272727]],
                             [[0.01, 0.01631605, 0.01293869],
                              [0.01, 0.08878719, 0.68612036]]]),
                    np.array([[[ 0., -0.30454545, -0.65030303]],
                             [[ 0.01, 0.0303608, 0.06645246]]])
                             )
        np.testing.assert_allclose(data_order_sol_1[0], data_order_res_1[0])
        np.testing.assert_allclose(data_order_sol_1[1], data_order_res_1[1])
        np.testing.assert_allclose(data_order_sol_1[2], data_order_res_1[2])

        # switch around order
        data_order_res_2 = simple_est_setup.match_data_to_network(
                    simple_est_setup.net_simulation.sim_variables_order,
                    {'V_0': ('Y_t', ('Y_t',)), 'V_1': ('X_t', ('X_t',))},
                    simple_est_setup.data.data_mean,
                    simple_est_setup.data.data_variance,
                    simple_est_setup.data.data_covariance,
                    simple_est_setup.data.data_mean_order,
                    simple_est_setup.data.data_variance_order,
                    simple_est_setup.data.data_covariance_order)
        data_order_sol_2 = (
                    np.array([[[0.        , 0.45      , 1.74      ],
                             [1.        , 0.67      , 0.37      ]],
                            [[0.01      , 0.07188642, 0.1995514 ],
                             [0.01      , 0.0469473 , 0.04838822]]]),
                     np.array([[[0.        , 0.51262626, 4.03272727],
                             [0.        , 0.22333333, 0.23545455]],
                            [[0.01      , 0.08878719, 0.68612036],
                             [0.01      , 0.01631605, 0.01293869]]]),
                     np.array([[[ 0.        , -0.30454545, -0.65030303]],
                            [[ 0.01      ,  0.0303608 ,  0.06645246]]])
                            )

        np.testing.assert_allclose(data_order_sol_2[0], data_order_res_2[0])
        np.testing.assert_allclose(data_order_sol_2[1], data_order_res_2[1])
        np.testing.assert_allclose(data_order_sol_2[2], data_order_res_2[2])

    ### mean only tests
    @pytest.fixture(scope='class')
    def simple_est_setup_mean_only(self):
        # see jupyter notebook ex_docs_tests for more info and plots
        ### define network
        net = me.Network('net_min_2_4')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ])

        ### create data with known values
        num_iter = 100
        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'d': 0.03, 'l': 0.07}
        time_values = np.array([0.0, 20.0, 40.0])
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                            initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        data = me.Data('data_test_est_min_2_4')
        data.load(['X_t', 'Y_t'], time_values, sims,
                  bootstrap_samples=10000, basic_sigma=1/num_iter)

        # overwrite with fixed values (from a num_iter = 100 simulation)
        data.data_mean = np.array([[[1.         ,0.67       ,0.37      ],
                                  [0.         ,0.45       ,1.74      ]],
                                 [[0.01       ,0.0469473  ,0.04838822],
                                  [0.01       ,0.07188642 ,0.1995514 ]]])
        data.data_variance = np.array([[[0.         ,0.22333333 ,0.23545455],
                                      [0.         ,0.51262626 ,4.03272727]],
                                     [[0.01       ,0.01631605 ,0.01293869],
                                      [0.01       ,0.08878719 ,0.68612036]]])
        data.data_covariance = np.array([[[ 0.         ,-0.30454545 ,-0.65030303]],
                                     [[ 0.01        ,0.0303608   ,0.06645246]]])

        ### run estimation and return object
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        time_values = None
        sim_mean_only = True
        fit_mean_only = True

        nlive = 1000 # 250 # 1000
        tolerance = 0.01 # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
        bound = 'multi'
        sample = 'unif'

        est = me.Estimation('est_min_2_4', net, data)
        est.estimate(variables, initial_values_type, initial_values,
                            theta_bounds, time_values, sim_mean_only, fit_mean_only,
                            nlive, tolerance, bound, sample)
        return est

    @pytest.fixture(scope='class')
    def simple_est_setup_mean_only_summary(self):
        # see jupyter notebook ex_docs_tests for more info and plots
        ### define network
        net = me.Network('net_min_2_4')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ])

        ### create data with known values
        time_values = np.array([0.0, 20.0, 40.0])
        data_mean = np.array([[[1.         ,0.67       ,0.37      ],
                                  [0.         ,0.45       ,1.74      ]],
                                 [[0.01       ,0.0469473  ,0.04838822],
                                  [0.01       ,0.07188642 ,0.1995514 ]]])
        data_variance = np.array([[[0.         ,0.22333333 ,0.23545455],
                                      [0.         ,0.51262626 ,4.03272727]],
                                     [[0.01       ,0.01631605 ,0.01293869],
                                      [0.01       ,0.08878719 ,0.68612036]]])
        data_covariance = np.array([[[ 0.         ,-0.30454545 ,-0.65030303]],
                                     [[ 0.01        ,0.0303608   ,0.06645246]]])
        data = me.Data('data_test_est_min_2_4')
        data.load(['X_t', 'Y_t'], time_values, None, 'summary',
                  mean_data=data_mean, var_data=data_variance, cov_data=data_covariance)

        ### run estimation and return object
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        time_values = None
        sim_mean_only = True
        fit_mean_only = True

        nlive = 1000 # 250 # 1000
        tolerance = 0.01 # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
        bound = 'multi'
        sample = 'unif'

        est = me.Estimation('est_min_2_4', net, data)
        est.estimate(variables, initial_values_type, initial_values,
                            theta_bounds, time_values, sim_mean_only, fit_mean_only,
                            nlive, tolerance, bound, sample)
        return est

    @pytest.fixture(scope='class')
    def simple_est_setup_mean_only_summary_data_mean_only(self):
        # see jupyter notebook ex_docs_tests for more info and plots
        ### define network
        net = me.Network('net_min_2_4')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ])

        ### create data with known values
        time_values = np.array([0.0, 20.0, 40.0])
        data_mean = np.array([[[1.         ,0.67       ,0.37      ],
                                  [0.         ,0.45       ,1.74      ]],
                                 [[0.01       ,0.0469473  ,0.04838822],
                                  [0.01       ,0.07188642 ,0.1995514 ]]])
        data = me.Data('data_test_est_min_2_4')
        data.load(['X_t', 'Y_t'], time_values, None, 'summary',
                  mean_data=data_mean)

        ### run estimation and return object
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_bounds = {'d': (0.0, 0.15), 'l': (0.0, 0.15)}
        time_values = None
        sim_mean_only = True
        fit_mean_only = True

        nlive = 1000 # 250 # 1000
        tolerance = 0.01 # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
        bound = 'multi'
        sample = 'unif'

        est = me.Estimation('est_min_2_4', net, data)
        est.estimate(variables, initial_values_type, initial_values,
                            theta_bounds, time_values, sim_mean_only, fit_mean_only,
                            nlive, tolerance, bound, sample)
        return est

    @pytest.mark.slow
    def test_log_evid_of_simple_minimal_model_mean_only_1(self, simple_est_setup_mean_only):
        log_evid_summary = np.array([7.7605709408748975, 7.78125756882485, 7.918132885953295,
                                    7.8897310484135135, 8.007149819432435, 7.908905626137309,
                                    7.813419052227798, 7.893391782255274, 7.8661928300681865,
                                    7.7371076942168235])
        log_evid_sol = np.mean(log_evid_summary)
        log_evid_tol = np.std(log_evid_summary)*6
        np.testing.assert_allclose(log_evid_sol,
                                    simple_est_setup_mean_only.bay_est_log_evidence,
                                    rtol=log_evid_tol, atol=log_evid_tol)

    @pytest.mark.slow
    def test_log_evid_of_simple_minimal_model_mean_only_2(self, simple_est_setup_mean_only_summary):
        log_evid_summary = np.array([7.7605709408748975, 7.78125756882485, 7.918132885953295,
                                    7.8897310484135135, 8.007149819432435, 7.908905626137309,
                                    7.813419052227798, 7.893391782255274, 7.8661928300681865,
                                    7.7371076942168235])
        log_evid_sol = np.mean(log_evid_summary)
        log_evid_tol = np.std(log_evid_summary)*6
        np.testing.assert_allclose(log_evid_sol,
                                    simple_est_setup_mean_only_summary.bay_est_log_evidence,
                                    rtol=log_evid_tol, atol=log_evid_tol)

    @pytest.mark.slow
    def test_log_evid_of_simple_minimal_model_mean_only_3(self, simple_est_setup_mean_only_summary_data_mean_only):
        log_evid_summary = np.array([7.7605709408748975, 7.78125756882485, 7.918132885953295,
                                    7.8897310484135135, 8.007149819432435, 7.908905626137309,
                                    7.813419052227798, 7.893391782255274, 7.8661928300681865,
                                    7.7371076942168235])
        log_evid_sol = np.mean(log_evid_summary)
        log_evid_tol = np.std(log_evid_summary)*6
        np.testing.assert_allclose(log_evid_sol,
                                    simple_est_setup_mean_only_summary_data_mean_only.bay_est_log_evidence,
                                    rtol=log_evid_tol, atol=log_evid_tol)

    @pytest.mark.slow
    def test_max_log_likelihood_of_simple_minimal_model_mean_only_1(self, simple_est_setup_mean_only):
        logl_max_summary = np.array([13.608863858268542, 13.608877226458901, 13.608847553454728,
                                    13.608877529200766, 13.60886643722303, 13.608873842971521,
                                    13.608861237001252, 13.608871370412565, 13.608861291328083,
                                    13.608867280974184])
        logl_max_sol = np.mean(logl_max_summary)
        logl_max_tol = np.std(logl_max_summary)*6
        np.testing.assert_allclose(logl_max_sol,
                                    simple_est_setup_mean_only.bay_est_log_likelihood_max,
                                    rtol=logl_max_tol, atol=logl_max_tol)

    @pytest.mark.slow
    def test_max_log_likelihood_of_simple_minimal_model_mean_only_2(self, simple_est_setup_mean_only_summary):
        logl_max_summary = np.array([13.608863858268542, 13.608877226458901, 13.608847553454728,
                                    13.608877529200766, 13.60886643722303, 13.608873842971521,
                                    13.608861237001252, 13.608871370412565, 13.608861291328083,
                                    13.608867280974184])
        logl_max_sol = np.mean(logl_max_summary)
        logl_max_tol = np.std(logl_max_summary)*6
        np.testing.assert_allclose(logl_max_sol,
                                    simple_est_setup_mean_only_summary.bay_est_log_likelihood_max,
                                    rtol=logl_max_tol, atol=logl_max_tol)

    @pytest.mark.slow
    def test_max_log_likelihood_of_simple_minimal_model_mean_only_3(self, simple_est_setup_mean_only_summary_data_mean_only):
        logl_max_summary = np.array([13.608863858268542, 13.608877226458901, 13.608847553454728,
                                    13.608877529200766, 13.60886643722303, 13.608873842971521,
                                    13.608861237001252, 13.608871370412565, 13.608861291328083,
                                    13.608867280974184])
        logl_max_sol = np.mean(logl_max_summary)
        logl_max_tol = np.std(logl_max_summary)*6
        np.testing.assert_allclose(logl_max_sol,
                                    simple_est_setup_mean_only_summary_data_mean_only.bay_est_log_likelihood_max,
                                    rtol=logl_max_tol, atol=logl_max_tol)

    @pytest.mark.slow
    def test_theta_credible_interval_of_simple_minimal_model_mean_only_1(self, simple_est_setup_mean_only):
        params_cred_summary = np.array([
                                        [[[0.02837977816466184, 0.02475317184603701, 0.032252371853851],
                                              [0.07347594766652865, 0.056945734692037, 0.08750586475951816]]],
                                        [[[0.028306399126361383, 0.024806844409570726, 0.03231374332054859],
                                              [0.07320951510718421, 0.05693863384948446, 0.08717210242134305]]],
                                        [[[0.02827899953673494, 0.024841594040143064, 0.032267882689003914],
                                              [0.0730942610197843, 0.05625009927327231, 0.08742397351019225]]],
                                        [[[0.02838163006364825, 0.024865759994829033, 0.03212986797543949],
                                              [0.07331954689754412, 0.05729540812526361, 0.0870374377933814]]],
                                        [[[0.028341976327621917, 0.024789528169831564, 0.032102435029126096],
                                              [0.07308260556778846, 0.05703471572401049, 0.08762153845829682]]],
                                        [[[0.02828908188331957, 0.024786375546703175, 0.032176919891949554],
                                              [0.07319853179674726, 0.05698164846042654, 0.08755942244446942]]],
                                        [[[0.028365004575820335, 0.024737675660559674, 0.03239593960086084],
                                              [0.07302648932606505, 0.05727973219749373, 0.08780361683668651]]],
                                        [[[0.028276876599934067, 0.024764474498227603, 0.03219409470556767],
                                              [0.07339656793664723, 0.056642380205776596, 0.08745371415790436]]],
                                        [[[0.02834698614659556, 0.024844365293418918, 0.032263830905032855],
                                              [0.07313390392614322, 0.057244039923494534, 0.08798062276751904]]],
                                        [[[0.02830009724678409, 0.024665930762921694, 0.032320966472588676],
                                              [0.0734295735250799, 0.05659526164492768, 0.08776299471142669]]]
                                        ])
        params_cred_sol = np.mean(params_cred_summary, axis=0)
        params_cred_tol = np.max(np.std(params_cred_summary, axis=0)*6)
        np.testing.assert_allclose(params_cred_sol,
                                    np.array([simple_est_setup_mean_only.bay_est_params_cred]),
                                    rtol=params_cred_tol, atol=params_cred_tol)

    @pytest.mark.slow
    def test_theta_credible_interval_of_simple_minimal_model_mean_only_2(self, simple_est_setup_mean_only_summary):
        params_cred_summary = np.array([
                                        [[[0.02837977816466184, 0.02475317184603701, 0.032252371853851],
                                              [0.07347594766652865, 0.056945734692037, 0.08750586475951816]]],
                                        [[[0.028306399126361383, 0.024806844409570726, 0.03231374332054859],
                                              [0.07320951510718421, 0.05693863384948446, 0.08717210242134305]]],
                                        [[[0.02827899953673494, 0.024841594040143064, 0.032267882689003914],
                                              [0.0730942610197843, 0.05625009927327231, 0.08742397351019225]]],
                                        [[[0.02838163006364825, 0.024865759994829033, 0.03212986797543949],
                                              [0.07331954689754412, 0.05729540812526361, 0.0870374377933814]]],
                                        [[[0.028341976327621917, 0.024789528169831564, 0.032102435029126096],
                                              [0.07308260556778846, 0.05703471572401049, 0.08762153845829682]]],
                                        [[[0.02828908188331957, 0.024786375546703175, 0.032176919891949554],
                                              [0.07319853179674726, 0.05698164846042654, 0.08755942244446942]]],
                                        [[[0.028365004575820335, 0.024737675660559674, 0.03239593960086084],
                                              [0.07302648932606505, 0.05727973219749373, 0.08780361683668651]]],
                                        [[[0.028276876599934067, 0.024764474498227603, 0.03219409470556767],
                                              [0.07339656793664723, 0.056642380205776596, 0.08745371415790436]]],
                                        [[[0.02834698614659556, 0.024844365293418918, 0.032263830905032855],
                                              [0.07313390392614322, 0.057244039923494534, 0.08798062276751904]]],
                                        [[[0.02830009724678409, 0.024665930762921694, 0.032320966472588676],
                                              [0.0734295735250799, 0.05659526164492768, 0.08776299471142669]]]
                                        ])
        params_cred_sol = np.mean(params_cred_summary, axis=0)
        params_cred_tol = np.max(np.std(params_cred_summary, axis=0)*6)
        np.testing.assert_allclose(params_cred_sol,
                                    np.array([simple_est_setup_mean_only_summary.bay_est_params_cred]),
                                    rtol=params_cred_tol, atol=params_cred_tol)

    @pytest.mark.slow
    def test_theta_credible_interval_of_simple_minimal_model_mean_only_3(self, simple_est_setup_mean_only_summary_data_mean_only):
        params_cred_summary = np.array([
                                        [[[0.02837977816466184, 0.02475317184603701, 0.032252371853851],
                                              [0.07347594766652865, 0.056945734692037, 0.08750586475951816]]],
                                        [[[0.028306399126361383, 0.024806844409570726, 0.03231374332054859],
                                              [0.07320951510718421, 0.05693863384948446, 0.08717210242134305]]],
                                        [[[0.02827899953673494, 0.024841594040143064, 0.032267882689003914],
                                              [0.0730942610197843, 0.05625009927327231, 0.08742397351019225]]],
                                        [[[0.02838163006364825, 0.024865759994829033, 0.03212986797543949],
                                              [0.07331954689754412, 0.05729540812526361, 0.0870374377933814]]],
                                        [[[0.028341976327621917, 0.024789528169831564, 0.032102435029126096],
                                              [0.07308260556778846, 0.05703471572401049, 0.08762153845829682]]],
                                        [[[0.02828908188331957, 0.024786375546703175, 0.032176919891949554],
                                              [0.07319853179674726, 0.05698164846042654, 0.08755942244446942]]],
                                        [[[0.028365004575820335, 0.024737675660559674, 0.03239593960086084],
                                              [0.07302648932606505, 0.05727973219749373, 0.08780361683668651]]],
                                        [[[0.028276876599934067, 0.024764474498227603, 0.03219409470556767],
                                              [0.07339656793664723, 0.056642380205776596, 0.08745371415790436]]],
                                        [[[0.02834698614659556, 0.024844365293418918, 0.032263830905032855],
                                              [0.07313390392614322, 0.057244039923494534, 0.08798062276751904]]],
                                        [[[0.02830009724678409, 0.024665930762921694, 0.032320966472588676],
                                              [0.0734295735250799, 0.05659526164492768, 0.08776299471142669]]]
                                        ])
        params_cred_sol = np.mean(params_cred_summary, axis=0)
        params_cred_tol = np.max(np.std(params_cred_summary, axis=0)*6)
        np.testing.assert_allclose(params_cred_sol,
                                    np.array([simple_est_setup_mean_only_summary_data_mean_only.bay_est_params_cred]),
                                    rtol=params_cred_tol, atol=params_cred_tol)

    @pytest.mark.slow
    def test_match_data_to_network_from_sampler_res_mean_only_1(self, simple_est_setup_mean_only):

        # check normal order of simulation variable identifiers
        assert({'V_0': ('X_t', ('X_t',)), 'V_1': ('Y_t', ('Y_t',))}==
                    simple_est_setup_mean_only.net_simulation.sim_variables_identifier)
        data_order_res_1 = simple_est_setup_mean_only.match_data_to_network(
                    simple_est_setup_mean_only.net_simulation.sim_variables_order,
                    simple_est_setup_mean_only.net_simulation.sim_variables_identifier,
                    simple_est_setup_mean_only.data.data_mean,
                    simple_est_setup_mean_only.data.data_variance,
                    simple_est_setup_mean_only.data.data_covariance,
                    simple_est_setup_mean_only.data.data_mean_order,
                    simple_est_setup_mean_only.data.data_variance_order,
                    simple_est_setup_mean_only.data.data_covariance_order)
        data_order_sol_1 = (
                    np.array([[[1., 0.67, 0.37],
                               [0., 0.45, 1.74]],
                              [[0.01, 0.0469473, 0.04838822],
                               [0.01, 0.07188642, 0.1995514]]]),
                    np.zeros((2, 2, 3)),
                    np.zeros((2, 1, 3)))
        np.testing.assert_allclose(data_order_sol_1[0], data_order_res_1[0])
        np.testing.assert_allclose(data_order_sol_1[1], data_order_res_1[1])
        np.testing.assert_allclose(data_order_sol_1[2], data_order_res_1[2])

    @pytest.mark.slow
    def test_match_data_to_network_from_sampler_res_mean_only_2(self, simple_est_setup_mean_only_summary):

        # check normal order of simulation variable identifiers
        assert({'V_0': ('X_t', ('X_t',)), 'V_1': ('Y_t', ('Y_t',))}==
                    simple_est_setup_mean_only_summary.net_simulation.sim_variables_identifier)
        data_order_res_1 = simple_est_setup_mean_only_summary.match_data_to_network(
                    simple_est_setup_mean_only_summary.net_simulation.sim_variables_order,
                    simple_est_setup_mean_only_summary.net_simulation.sim_variables_identifier,
                    simple_est_setup_mean_only_summary.data.data_mean,
                    simple_est_setup_mean_only_summary.data.data_variance,
                    simple_est_setup_mean_only_summary.data.data_covariance,
                    simple_est_setup_mean_only_summary.data.data_mean_order,
                    simple_est_setup_mean_only_summary.data.data_variance_order,
                    simple_est_setup_mean_only_summary.data.data_covariance_order)
        data_order_sol_1 = (
                    np.array([[[1., 0.67, 0.37],
                               [0., 0.45, 1.74]],
                              [[0.01, 0.0469473, 0.04838822],
                               [0.01, 0.07188642, 0.1995514]]]),
                    np.zeros((2, 2, 3)),
                    np.zeros((2, 1, 3)))
        np.testing.assert_allclose(data_order_sol_1[0], data_order_res_1[0])
        np.testing.assert_allclose(data_order_sol_1[1], data_order_res_1[1])
        np.testing.assert_allclose(data_order_sol_1[2], data_order_res_1[2])

    @pytest.mark.slow
    def test_match_data_to_network_from_sampler_res_mean_only_3(self, simple_est_setup_mean_only_summary_data_mean_only):

        # check normal order of simulation variable identifiers
        assert({'V_0': ('X_t', ('X_t',)), 'V_1': ('Y_t', ('Y_t',))}==
                    simple_est_setup_mean_only_summary_data_mean_only.net_simulation.sim_variables_identifier)
        data_order_res_1 = simple_est_setup_mean_only_summary_data_mean_only.match_data_to_network(
                    simple_est_setup_mean_only_summary_data_mean_only.net_simulation.sim_variables_order,
                    simple_est_setup_mean_only_summary_data_mean_only.net_simulation.sim_variables_identifier,
                    simple_est_setup_mean_only_summary_data_mean_only.data.data_mean,
                    simple_est_setup_mean_only_summary_data_mean_only.data.data_variance,
                    simple_est_setup_mean_only_summary_data_mean_only.data.data_covariance,
                    simple_est_setup_mean_only_summary_data_mean_only.data.data_mean_order,
                    simple_est_setup_mean_only_summary_data_mean_only.data.data_variance_order,
                    simple_est_setup_mean_only_summary_data_mean_only.data.data_covariance_order)
        data_order_sol_1 = (
                    np.array([[[1., 0.67, 0.37],
                               [0., 0.45, 1.74]],
                              [[0.01, 0.0469473, 0.04838822],
                               [0.01, 0.07188642, 0.1995514]]]),
                    np.empty((2, 0, 3)),
                    np.empty((2, 0, 3)))
        np.testing.assert_allclose(data_order_sol_1[0], data_order_res_1[0])
        np.testing.assert_allclose(data_order_sol_1[1], data_order_res_1[1])
        np.testing.assert_allclose(data_order_sol_1[2], data_order_res_1[2])

    def test_initialise_time_values_different(self):
        time_values = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        data_time_values = np.array([1.0, 3.0, 5.0])
        net_time_values, __, net_time_ind = me.Estimation.initialise_time_values(time_values, data_time_values)
        np.testing.assert_allclose(net_time_values, np.array([0., 1., 2., 3., 4., 5., 6.]))
        assert net_time_ind==(1, 3, 5)

    def test_initialise_time_values_none(self):
        time_values = None
        data_time_values = np.array([1.0, 3.0, 5.0])
        net_time_values, __, net_time_ind = me.Estimation.initialise_time_values(time_values, data_time_values)
        np.testing.assert_allclose(net_time_values, np.array([1.0, 3.0, 5.0]))
        assert net_time_ind==slice(None)

    def test_initialise_time_values_equal(self):
        time_values = np.array([1.0, 3.0, 5.0])
        data_time_values = np.array([1.0, 3.0, 5.0])
        net_time_values, __, net_time_ind = me.Estimation.initialise_time_values(time_values, data_time_values)
        np.testing.assert_allclose(net_time_values, np.array([1.0, 3.0, 5.0]))
        assert net_time_ind==slice(None)
