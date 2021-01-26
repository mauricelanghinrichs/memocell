
import pytest
import memocell as me
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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (sequentially)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
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

    @pytest.mark.slow
    def test_division_model_parallel_summary_data(self):
        ### create data from 5-steps model

        time_values = np.array([0.0, 10.0])
        mean_data = np.array([[[1.         , 3.73      ]],
                              [[0.01       , 0.15406761]]])
        var_data = np.array([[[0.         , 2.36070707]],
                             [[0.01       , 0.32208202]]])
        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, None, data_type='summary',
                        mean_data=mean_data, var_data=var_data)

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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
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

    @pytest.mark.slow
    def test_division_model_parallel_mean_only(self):
        ### create data from 5-steps model
        t = [
            {'start': 'X_t', 'end': 'X_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 5}
            ]

        net = me.Network('net_div_g5')
        net.structure(t)

        num_iter = 100
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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
                                            theta_bounds, data, sim_mean_only=True,
                                            fit_mean_only=True, parallel=True)

        ### assert log evidence values
        evid_res = np.array([est.bay_est_log_evidence for est in est_res])
        # expected solution
        evid_summary = np.array([
            [0.9419435322851446, 1.0106982945064, 1.1012240890675635],
            [0.9566246018872815, 1.0030070861382694, 1.185808337623029],
            [0.9907574912748793, 1.047671927169712, 1.19272285522771],
            [0.9344626373613755, 1.0150202141481144, 1.1356276897852824],
            [0.9957857383128643, 1.1147029967812334, 1.2813572453964095],
            [0.9224870435679013, 1.135816900454853, 1.2377276605741654],
            [0.9894778490215606, 1.0941277563587202, 1.1494814071976407],
            [0.9845703717052129, 1.2037659025849716, 1.1818061600496066],
            [0.9512176652730777, 1.0724397747692442, 1.2583570008429867],
            [0.8883984291598804, 1.003479199497773, 1.1824954558159297]
        ])
        evid_sol = np.mean(evid_summary, axis=0)
        evid_tol = np.max(np.std(evid_summary, axis=0)*6)

        np.testing.assert_allclose(evid_sol, evid_res, rtol=evid_tol, atol=evid_tol)

        ### assert maximal log likelihood value
        logl_max_res = np.array([est.bay_est_log_likelihood_max for est in est_res])
        # expected solution
        logl_max_summary = np.array([
            [4.637656866490975, 4.63765686651381, 4.637656866427406],
            [4.637656866313304, 4.63765686645764, 4.637656866472604],
            [4.637656866511199, 4.637656866434893, 4.637656866503798],
            [4.637656866478411, 4.637656866496112, 4.637656866472503],
            [4.637656866481246, 4.637656866487981, 4.637656866364033],
            [4.63765686651618, 4.637656866167647, 4.637656866000495],
            [4.6376568664946385, 4.637656866516615, 4.637656865509905],
            [4.637656866397452, 4.63765686646518, 4.637656866516921],
            [4.637656866513331, 4.637656866516135, 4.637656866481746],
            [4.6376568665054085, 4.637656866400874, 4.63765686645308]
        ])
        logl_max_sol = np.mean(logl_max_summary, axis=0)
        logl_max_tol = np.max(np.std(logl_max_summary, axis=0)*6)
        np.testing.assert_allclose(logl_max_sol, logl_max_res, rtol=logl_max_tol, atol=logl_max_tol)

        ### assert estimated parameter value (95% credible interval)
        theta_cred_res = np.array([est.bay_est_params_cred for est in est_res])
        # expected solution
        theta_cred_summary = np.array([
                        [[[0.17767525971322726, 0.16766484301297668, 0.1871859065530188]],
                            [[0.2116378072993456, 0.19980784768361473, 0.22221378166000505]],
                            [[0.22706308238730954, 0.21426688997969887, 0.23898897505750633]]],
                        [[[0.17783574075135636, 0.16745234255033054, 0.18709129910629485]],
                            [[0.21159566312843459, 0.20025740650972587, 0.22225069777450657]],
                            [[0.22685131261479294, 0.2148889188053539, 0.23873886327930333]]],
                        [[[0.17772495817619227, 0.16792866692656294, 0.18716373871632497]],
                            [[0.21147737507817208, 0.20013138837338232, 0.2221836017996362]],
                            [[0.22712942097758385, 0.21459786012153031, 0.239019411969675]]],
                        [[[0.17765310186185357, 0.16737710709424608, 0.18699853276265388]],
                            [[0.21152367704878855, 0.20017333735288512, 0.22195840272665873]],
                            [[0.22739299020772027, 0.21494870304871327, 0.23871190646445717]]],
                        [[[0.17774890954479852, 0.16785378992605757, 0.18692744095053854]],
                            [[0.2117422771562304, 0.20033115893162734, 0.221744778476685]],
                            [[0.22700636347057088, 0.21483175457317827, 0.23850802940459553]]],
                        [[[0.17770323548026393, 0.16762338985697867, 0.18722030610380444]],
                            [[0.21148199985918062, 0.20004025740164794, 0.22246417034066013]],
                            [[0.22726641468296005, 0.21508239943597135, 0.23893200778722384]]],
                        [[[0.17774638376606106, 0.16741064926107696, 0.18714861450676257]],
                            [[0.211685981167947, 0.20044959549495348, 0.2218134914679289]],
                            [[0.2272913418178502, 0.21488828222344508, 0.23918800766586332]]],
                        [[[0.17786567511829948, 0.16774372529113826, 0.18716669314766138]],
                            [[0.21176955766928585, 0.2003380716254387, 0.2220132491656906]],
                            [[0.2267721834898168, 0.2144859118131004, 0.23877758912985064]]],
                        [[[0.17774477002988257, 0.16777800470913823, 0.186737513643131]],
                            [[0.21140631049740677, 0.20034762799106276, 0.22201850676261048]],
                            [[0.22699475212678594, 0.21520075443250175, 0.23875173560623902]]],
                        [[[0.17768763634149626, 0.16748127097311355, 0.18726644765855052]],
                            [[0.21158086725786138, 0.20022479235379922, 0.22237544704503034]],
                            [[0.22692686604795786, 0.21503615490679384, 0.2388629470323908]]]])
        theta_cred_sol = np.mean(theta_cred_summary, axis=0)
        theta_cred_tol = np.max(np.std(theta_cred_summary, axis=0)*6)
        np.testing.assert_allclose(theta_cred_sol, theta_cred_res, rtol=theta_cred_tol, atol=theta_cred_tol)

    @pytest.mark.slow
    def test_division_model_parallel_fit_mean_only(self):
        ### create data from 5-steps model
        t = [
            {'start': 'X_t', 'end': 'X_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 5}
            ]

        net = me.Network('net_div_g5')
        net.structure(t)

        num_iter = 100
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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
                                            theta_bounds, data, sim_mean_only=False,
                                            fit_mean_only=True, parallel=True)

        ### assert log evidence values
        evid_res = np.array([est.bay_est_log_evidence for est in est_res])
        # expected solution
        evid_summary = np.array([
            [0.9419435322851446, 1.0106982945064, 1.1012240890675635],
            [0.9566246018872815, 1.0030070861382694, 1.185808337623029],
            [0.9907574912748793, 1.047671927169712, 1.19272285522771],
            [0.9344626373613755, 1.0150202141481144, 1.1356276897852824],
            [0.9957857383128643, 1.1147029967812334, 1.2813572453964095],
            [0.9224870435679013, 1.135816900454853, 1.2377276605741654],
            [0.9894778490215606, 1.0941277563587202, 1.1494814071976407],
            [0.9845703717052129, 1.2037659025849716, 1.1818061600496066],
            [0.9512176652730777, 1.0724397747692442, 1.2583570008429867],
            [0.8883984291598804, 1.003479199497773, 1.1824954558159297]
        ])
        evid_sol = np.mean(evid_summary, axis=0)
        evid_tol = np.max(np.std(evid_summary, axis=0)*6)

        np.testing.assert_allclose(evid_sol, evid_res, rtol=evid_tol, atol=evid_tol)

        ### assert maximal log likelihood value
        logl_max_res = np.array([est.bay_est_log_likelihood_max for est in est_res])
        # expected solution
        logl_max_summary = np.array([
            [4.637656866490975, 4.63765686651381, 4.637656866427406],
            [4.637656866313304, 4.63765686645764, 4.637656866472604],
            [4.637656866511199, 4.637656866434893, 4.637656866503798],
            [4.637656866478411, 4.637656866496112, 4.637656866472503],
            [4.637656866481246, 4.637656866487981, 4.637656866364033],
            [4.63765686651618, 4.637656866167647, 4.637656866000495],
            [4.6376568664946385, 4.637656866516615, 4.637656865509905],
            [4.637656866397452, 4.63765686646518, 4.637656866516921],
            [4.637656866513331, 4.637656866516135, 4.637656866481746],
            [4.6376568665054085, 4.637656866400874, 4.63765686645308]
        ])
        logl_max_sol = np.mean(logl_max_summary, axis=0)
        logl_max_tol = np.max(np.std(logl_max_summary, axis=0)*6)
        np.testing.assert_allclose(logl_max_sol, logl_max_res, rtol=logl_max_tol, atol=logl_max_tol)

        ### assert estimated parameter value (95% credible interval)
        theta_cred_res = np.array([est.bay_est_params_cred for est in est_res])
        # expected solution
        theta_cred_summary = np.array([
                        [[[0.17767525971322726, 0.16766484301297668, 0.1871859065530188]],
                            [[0.2116378072993456, 0.19980784768361473, 0.22221378166000505]],
                            [[0.22706308238730954, 0.21426688997969887, 0.23898897505750633]]],
                        [[[0.17783574075135636, 0.16745234255033054, 0.18709129910629485]],
                            [[0.21159566312843459, 0.20025740650972587, 0.22225069777450657]],
                            [[0.22685131261479294, 0.2148889188053539, 0.23873886327930333]]],
                        [[[0.17772495817619227, 0.16792866692656294, 0.18716373871632497]],
                            [[0.21147737507817208, 0.20013138837338232, 0.2221836017996362]],
                            [[0.22712942097758385, 0.21459786012153031, 0.239019411969675]]],
                        [[[0.17765310186185357, 0.16737710709424608, 0.18699853276265388]],
                            [[0.21152367704878855, 0.20017333735288512, 0.22195840272665873]],
                            [[0.22739299020772027, 0.21494870304871327, 0.23871190646445717]]],
                        [[[0.17774890954479852, 0.16785378992605757, 0.18692744095053854]],
                            [[0.2117422771562304, 0.20033115893162734, 0.221744778476685]],
                            [[0.22700636347057088, 0.21483175457317827, 0.23850802940459553]]],
                        [[[0.17770323548026393, 0.16762338985697867, 0.18722030610380444]],
                            [[0.21148199985918062, 0.20004025740164794, 0.22246417034066013]],
                            [[0.22726641468296005, 0.21508239943597135, 0.23893200778722384]]],
                        [[[0.17774638376606106, 0.16741064926107696, 0.18714861450676257]],
                            [[0.211685981167947, 0.20044959549495348, 0.2218134914679289]],
                            [[0.2272913418178502, 0.21488828222344508, 0.23918800766586332]]],
                        [[[0.17786567511829948, 0.16774372529113826, 0.18716669314766138]],
                            [[0.21176955766928585, 0.2003380716254387, 0.2220132491656906]],
                            [[0.2267721834898168, 0.2144859118131004, 0.23877758912985064]]],
                        [[[0.17774477002988257, 0.16777800470913823, 0.186737513643131]],
                            [[0.21140631049740677, 0.20034762799106276, 0.22201850676261048]],
                            [[0.22699475212678594, 0.21520075443250175, 0.23875173560623902]]],
                        [[[0.17768763634149626, 0.16748127097311355, 0.18726644765855052]],
                            [[0.21158086725786138, 0.20022479235379922, 0.22237544704503034]],
                            [[0.22692686604795786, 0.21503615490679384, 0.2388629470323908]]]])
        theta_cred_sol = np.mean(theta_cred_summary, axis=0)
        theta_cred_tol = np.max(np.std(theta_cred_summary, axis=0)*6)
        np.testing.assert_allclose(theta_cred_sol, theta_cred_res, rtol=theta_cred_tol, atol=theta_cred_tol)

    @pytest.mark.slow
    def test_division_model_parallel_mean_only_summary(self):
        ### create data from 5-steps model
        time_values = np.array([0.0, 10.0])
        data_mean = np.array([[[1.         , 3.73      ]],
                              [[0.01       , 0.15406761]]])
        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, None, 'summary',
                    mean_data=data_mean)


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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
                                            theta_bounds, data, sim_mean_only=True,
                                            fit_mean_only=True, parallel=True)

        ### assert log evidence values
        evid_res = np.array([est.bay_est_log_evidence for est in est_res])
        # expected solution
        evid_summary = np.array([
            [0.9419435322851446, 1.0106982945064, 1.1012240890675635],
            [0.9566246018872815, 1.0030070861382694, 1.185808337623029],
            [0.9907574912748793, 1.047671927169712, 1.19272285522771],
            [0.9344626373613755, 1.0150202141481144, 1.1356276897852824],
            [0.9957857383128643, 1.1147029967812334, 1.2813572453964095],
            [0.9224870435679013, 1.135816900454853, 1.2377276605741654],
            [0.9894778490215606, 1.0941277563587202, 1.1494814071976407],
            [0.9845703717052129, 1.2037659025849716, 1.1818061600496066],
            [0.9512176652730777, 1.0724397747692442, 1.2583570008429867],
            [0.8883984291598804, 1.003479199497773, 1.1824954558159297]
        ])
        evid_sol = np.mean(evid_summary, axis=0)
        evid_tol = np.max(np.std(evid_summary, axis=0)*6)

        np.testing.assert_allclose(evid_sol, evid_res, rtol=evid_tol, atol=evid_tol)

        ### assert maximal log likelihood value
        logl_max_res = np.array([est.bay_est_log_likelihood_max for est in est_res])
        # expected solution
        logl_max_summary = np.array([
            [4.637656866490975, 4.63765686651381, 4.637656866427406],
            [4.637656866313304, 4.63765686645764, 4.637656866472604],
            [4.637656866511199, 4.637656866434893, 4.637656866503798],
            [4.637656866478411, 4.637656866496112, 4.637656866472503],
            [4.637656866481246, 4.637656866487981, 4.637656866364033],
            [4.63765686651618, 4.637656866167647, 4.637656866000495],
            [4.6376568664946385, 4.637656866516615, 4.637656865509905],
            [4.637656866397452, 4.63765686646518, 4.637656866516921],
            [4.637656866513331, 4.637656866516135, 4.637656866481746],
            [4.6376568665054085, 4.637656866400874, 4.63765686645308]
        ])
        logl_max_sol = np.mean(logl_max_summary, axis=0)
        logl_max_tol = np.max(np.std(logl_max_summary, axis=0)*6)
        np.testing.assert_allclose(logl_max_sol, logl_max_res, rtol=logl_max_tol, atol=logl_max_tol)

        ### assert estimated parameter value (95% credible interval)
        theta_cred_res = np.array([est.bay_est_params_cred for est in est_res])
        # expected solution
        theta_cred_summary = np.array([
                        [[[0.17767525971322726, 0.16766484301297668, 0.1871859065530188]],
                            [[0.2116378072993456, 0.19980784768361473, 0.22221378166000505]],
                            [[0.22706308238730954, 0.21426688997969887, 0.23898897505750633]]],
                        [[[0.17783574075135636, 0.16745234255033054, 0.18709129910629485]],
                            [[0.21159566312843459, 0.20025740650972587, 0.22225069777450657]],
                            [[0.22685131261479294, 0.2148889188053539, 0.23873886327930333]]],
                        [[[0.17772495817619227, 0.16792866692656294, 0.18716373871632497]],
                            [[0.21147737507817208, 0.20013138837338232, 0.2221836017996362]],
                            [[0.22712942097758385, 0.21459786012153031, 0.239019411969675]]],
                        [[[0.17765310186185357, 0.16737710709424608, 0.18699853276265388]],
                            [[0.21152367704878855, 0.20017333735288512, 0.22195840272665873]],
                            [[0.22739299020772027, 0.21494870304871327, 0.23871190646445717]]],
                        [[[0.17774890954479852, 0.16785378992605757, 0.18692744095053854]],
                            [[0.2117422771562304, 0.20033115893162734, 0.221744778476685]],
                            [[0.22700636347057088, 0.21483175457317827, 0.23850802940459553]]],
                        [[[0.17770323548026393, 0.16762338985697867, 0.18722030610380444]],
                            [[0.21148199985918062, 0.20004025740164794, 0.22246417034066013]],
                            [[0.22726641468296005, 0.21508239943597135, 0.23893200778722384]]],
                        [[[0.17774638376606106, 0.16741064926107696, 0.18714861450676257]],
                            [[0.211685981167947, 0.20044959549495348, 0.2218134914679289]],
                            [[0.2272913418178502, 0.21488828222344508, 0.23918800766586332]]],
                        [[[0.17786567511829948, 0.16774372529113826, 0.18716669314766138]],
                            [[0.21176955766928585, 0.2003380716254387, 0.2220132491656906]],
                            [[0.2267721834898168, 0.2144859118131004, 0.23877758912985064]]],
                        [[[0.17774477002988257, 0.16777800470913823, 0.186737513643131]],
                            [[0.21140631049740677, 0.20034762799106276, 0.22201850676261048]],
                            [[0.22699475212678594, 0.21520075443250175, 0.23875173560623902]]],
                        [[[0.17768763634149626, 0.16748127097311355, 0.18726644765855052]],
                            [[0.21158086725786138, 0.20022479235379922, 0.22237544704503034]],
                            [[0.22692686604795786, 0.21503615490679384, 0.2388629470323908]]]])
        theta_cred_sol = np.mean(theta_cred_summary, axis=0)
        theta_cred_tol = np.max(np.std(theta_cred_summary, axis=0)*6)
        np.testing.assert_allclose(theta_cred_sol, theta_cred_res, rtol=theta_cred_tol, atol=theta_cred_tol)

    @pytest.mark.slow
    def test_division_model_parallel_fit_mean_only_summary(self):
        ### create data from 5-steps model
        time_values = np.array([0.0, 10.0])
        data_mean = np.array([[[1.         , 3.73      ]],
                              [[0.01       , 0.15406761]]])

        data = me.Data('data_test_select_models')
        data.load(['X_t',], time_values, None, 'summary',
                    mean_data=data_mean)

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
        initial_values_types = ['synchronous']*3
        initial_values = [{('X_t',): 1.0, ('X_t', 'X_t'): 0.0}]*3
        theta_bounds = [{'l': (0.0, 0.5)}]*3

        ### run selection (in parallel)
        est_res = me.selection.select_models(networks, variables,
                                            initial_values_types, initial_values,
                                            theta_bounds, data, sim_mean_only=False,
                                            fit_mean_only=True, parallel=True)

        ### assert log evidence values
        evid_res = np.array([est.bay_est_log_evidence for est in est_res])
        # expected solution
        evid_summary = np.array([
            [0.9419435322851446, 1.0106982945064, 1.1012240890675635],
            [0.9566246018872815, 1.0030070861382694, 1.185808337623029],
            [0.9907574912748793, 1.047671927169712, 1.19272285522771],
            [0.9344626373613755, 1.0150202141481144, 1.1356276897852824],
            [0.9957857383128643, 1.1147029967812334, 1.2813572453964095],
            [0.9224870435679013, 1.135816900454853, 1.2377276605741654],
            [0.9894778490215606, 1.0941277563587202, 1.1494814071976407],
            [0.9845703717052129, 1.2037659025849716, 1.1818061600496066],
            [0.9512176652730777, 1.0724397747692442, 1.2583570008429867],
            [0.8883984291598804, 1.003479199497773, 1.1824954558159297]
        ])
        evid_sol = np.mean(evid_summary, axis=0)
        evid_tol = np.max(np.std(evid_summary, axis=0)*6)

        np.testing.assert_allclose(evid_sol, evid_res, rtol=evid_tol, atol=evid_tol)

        ### assert maximal log likelihood value
        logl_max_res = np.array([est.bay_est_log_likelihood_max for est in est_res])
        # expected solution
        logl_max_summary = np.array([
            [4.637656866490975, 4.63765686651381, 4.637656866427406],
            [4.637656866313304, 4.63765686645764, 4.637656866472604],
            [4.637656866511199, 4.637656866434893, 4.637656866503798],
            [4.637656866478411, 4.637656866496112, 4.637656866472503],
            [4.637656866481246, 4.637656866487981, 4.637656866364033],
            [4.63765686651618, 4.637656866167647, 4.637656866000495],
            [4.6376568664946385, 4.637656866516615, 4.637656865509905],
            [4.637656866397452, 4.63765686646518, 4.637656866516921],
            [4.637656866513331, 4.637656866516135, 4.637656866481746],
            [4.6376568665054085, 4.637656866400874, 4.63765686645308]
        ])
        logl_max_sol = np.mean(logl_max_summary, axis=0)
        logl_max_tol = np.max(np.std(logl_max_summary, axis=0)*6)
        np.testing.assert_allclose(logl_max_sol, logl_max_res, rtol=logl_max_tol, atol=logl_max_tol)

        ### assert estimated parameter value (95% credible interval)
        theta_cred_res = np.array([est.bay_est_params_cred for est in est_res])
        # expected solution
        theta_cred_summary = np.array([
                        [[[0.17767525971322726, 0.16766484301297668, 0.1871859065530188]],
                            [[0.2116378072993456, 0.19980784768361473, 0.22221378166000505]],
                            [[0.22706308238730954, 0.21426688997969887, 0.23898897505750633]]],
                        [[[0.17783574075135636, 0.16745234255033054, 0.18709129910629485]],
                            [[0.21159566312843459, 0.20025740650972587, 0.22225069777450657]],
                            [[0.22685131261479294, 0.2148889188053539, 0.23873886327930333]]],
                        [[[0.17772495817619227, 0.16792866692656294, 0.18716373871632497]],
                            [[0.21147737507817208, 0.20013138837338232, 0.2221836017996362]],
                            [[0.22712942097758385, 0.21459786012153031, 0.239019411969675]]],
                        [[[0.17765310186185357, 0.16737710709424608, 0.18699853276265388]],
                            [[0.21152367704878855, 0.20017333735288512, 0.22195840272665873]],
                            [[0.22739299020772027, 0.21494870304871327, 0.23871190646445717]]],
                        [[[0.17774890954479852, 0.16785378992605757, 0.18692744095053854]],
                            [[0.2117422771562304, 0.20033115893162734, 0.221744778476685]],
                            [[0.22700636347057088, 0.21483175457317827, 0.23850802940459553]]],
                        [[[0.17770323548026393, 0.16762338985697867, 0.18722030610380444]],
                            [[0.21148199985918062, 0.20004025740164794, 0.22246417034066013]],
                            [[0.22726641468296005, 0.21508239943597135, 0.23893200778722384]]],
                        [[[0.17774638376606106, 0.16741064926107696, 0.18714861450676257]],
                            [[0.211685981167947, 0.20044959549495348, 0.2218134914679289]],
                            [[0.2272913418178502, 0.21488828222344508, 0.23918800766586332]]],
                        [[[0.17786567511829948, 0.16774372529113826, 0.18716669314766138]],
                            [[0.21176955766928585, 0.2003380716254387, 0.2220132491656906]],
                            [[0.2267721834898168, 0.2144859118131004, 0.23877758912985064]]],
                        [[[0.17774477002988257, 0.16777800470913823, 0.186737513643131]],
                            [[0.21140631049740677, 0.20034762799106276, 0.22201850676261048]],
                            [[0.22699475212678594, 0.21520075443250175, 0.23875173560623902]]],
                        [[[0.17768763634149626, 0.16748127097311355, 0.18726644765855052]],
                            [[0.21158086725786138, 0.20022479235379922, 0.22237544704503034]],
                            [[0.22692686604795786, 0.21503615490679384, 0.2388629470323908]]]])
        theta_cred_sol = np.mean(theta_cred_summary, axis=0)
        theta_cred_tol = np.max(np.std(theta_cred_summary, axis=0)*6)
        np.testing.assert_allclose(theta_cred_sol, theta_cred_res, rtol=theta_cred_tol, atol=theta_cred_tol)
