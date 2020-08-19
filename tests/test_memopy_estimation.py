
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memopy_[...].py"
# or after pip installation $py.test --pyargs memo_py$

import pytest
import memo_py as me
import numpy as np

class TestEstimationClass(object):
    @pytest.fixture()
    def simple_est_setup(self):
        # see jupyter notebook ex_docs_tests for more info and plots
        ### define network
        t = [
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ]

        net = me.Network('net_min_2_4')
        net.structure(t)

        ### create data with known values
        num_iter = 100
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'d': 0.03, 'l': 0.07}
        time_values = np.array([0.0, 20.0, 40.0])
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        res_list = list()

        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', initial_values, theta_values, time_values, variables)[1])

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
        network_setup = {
            'initial_values': {'X_t': 1, 'Y_t': 0},
            'theta_bounds': {'d': (0.0, 0.15), 'l': (0.0, 0.15)},
            'mean_only': False,
            'variables': {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}
        }

        mcmc_setup = {
            'nlive':                    1000, # 250 # 1000
            'tolerance':                0.01, # 0.1 (COARSE) # 0.05 # 0.01 (NORMAL)
            'bound':                    'multi',
            'sample':                   'unif'
        }

        est = me.Estimation('est_min_2_4', net, data)
        est.estimate(network_setup, mcmc_setup)
        return est

    @pytest.mark.slow
    def test_log_evid_of_simple_minimal_model(self, simple_est_setup):
        np.testing.assert_allclose(28.2, simple_est_setup.bay_est_log_evidence,
                                    rtol=1.0, atol=1.0)
    @pytest.mark.slow
    def test_max_log_likelihood_of_simple_minimal_model(self, simple_est_setup):
        np.testing.assert_allclose(35.5, simple_est_setup.bay_est_log_likelihood_max,
                                    rtol=1.0, atol=1.0)

    @pytest.mark.slow
    def test_theta_credible_interval_of_simple_minimal_model(self, simple_est_setup):
        sol_bay_est_params_conf = np.array([[[0.02803474, 0.02594989, 0.03014408],
                                              [0.07470537, 0.06919784, 0.07955645]]])
        np.testing.assert_allclose(sol_bay_est_params_conf,
                                    np.array([simple_est_setup.bay_est_params_conf]),
                                    rtol=0.002, atol=0.002)

    @pytest.mark.slow
    def test_get_model_evidence_from_sampler_res(self, simple_est_setup):
        sampler_result = simple_est_setup.bay_nested_sampler_res
        logZ, logzerr = simple_est_setup.get_model_evidence(sampler_result)
        np.testing.assert_allclose(28.2, logZ, rtol=1.0, atol=1.0)
        np.testing.assert_allclose(0.1, logzerr, rtol=0.1, atol=0.1)
