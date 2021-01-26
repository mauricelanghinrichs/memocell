
import pytest
import memocell as me
import numpy as np
import copy

class TestEstimationClass(object):
    ###### tests with mean, var and cov
    ### division model l2
    def test_division_model_l2_moments(self):
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

        sim_sol_mean = np.array([[1.        , 1.03747108, 1.12877492, 1.25584829, 1.41121682,
                                 1.59270775, 1.80090981, 2.03796736, 2.30702475, 2.61198952,
                                 2.9574545 ]])
        sim_sol_var = np.array([[0.        , 0.0371514 , 0.12666012, 0.25345033, 0.41852285,
                                 0.63196069, 0.90975564, 1.27312791, 1.74922638, 2.37265998,
                                 3.18767674]])
        sim_sol_cov = np.empty(shape=(0, 11), dtype=np.float64)
        np.testing.assert_allclose(sim_sol_mean, sim_res_mom[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim_res_mom[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim_res_mom[2], rtol=1e-06, atol=1e-06)

    def test_division_model_l2_gillespie(self):
        net = me.Network('net_div_l2')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 2}])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.15}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 1.03739, 1.13064, 1.25668, 1.40981, 1.5916 , 1.80055,
                                2.03637, 2.30674, 2.61072, 2.95297]])
        sim_sol_var = np.array([[0.        , 0.03709236, 0.12869448, 0.25419792, 0.41848995,
                                0.63293577, 0.91199882, 1.2775    , 1.75406811, 2.37076479,
                                3.18641004]])
        np.testing.assert_allclose(sim_sol_mean, np.mean(sims, axis=0),
                                                            rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, np.var(sims, axis=0, ddof=1),
                                                            rtol=0.2, atol=0.2)

    ### division model l5
    def test_division_model_l5_moments(self):
        net = me.Network('net_div_l5')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 5}])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('X_t','X_t'): 0.0}
        theta_values = {'l': 0.22}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[1.        , 1.00543582, 1.07269814, 1.2418072 , 1.47882701,
                                 1.75244286, 2.06375747, 2.42784846, 2.85837618, 3.36664248,
                                 3.96524759]])
        sim_sol_var = np.array([[0.        , 0.00540733, 0.06781916, 0.1923667 , 0.31379163,
                                 0.43719485, 0.61987894, 0.90288174, 1.30969425, 1.87625342,
                                 2.66684227]])
        sim_sol_cov = np.empty(shape=(0, 11), dtype=np.float64)
        np.testing.assert_allclose(sim_sol_mean, sim_res_mom[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim_res_mom[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim_res_mom[2], rtol=1e-06, atol=1e-06)

    def test_division_model_l5_gillespie(self):
        net = me.Network('net_div_l5')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 5}])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.22}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 1.00512, 1.07264, 1.23965, 1.47588, 1.74758, 2.05933,
                                2.42631, 2.85557, 3.36449, 3.96022]])
        sim_sol_var = np.array([[0.        , 0.00509384, 0.06764411, 0.19057978, 0.31048133,
                                0.43212846, 0.61251608, 0.8915387 , 1.30052298, 1.87737581,
                                2.66210417]])
        np.testing.assert_allclose(sim_sol_mean, np.mean(sims, axis=0),
                                                            rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, np.var(sims, axis=0, ddof=1),
                                                            rtol=0.2, atol=0.2)

    ### division model l15
    @pytest.mark.slow
    def test_division_model_l15_moments(self):
        net = me.Network('net_div_l15')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 15}])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('X_t','X_t'): 0.0}
        theta_values = {'l': 0.23}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[1.        , 1.0000036 , 1.00504242, 1.1029673 , 1.40858873,
                                 1.74539991, 1.9836416 , 2.2599425 , 2.69956532, 3.24078993,
                                 3.79706324]])
        sim_sol_var = np.array([[0.00000000e+00, 3.59865514e-06, 5.01699099e-03, 9.23670549e-02,
                                 2.42073295e-01, 2.03605526e-01, 1.56235900e-01, 3.31027260e-01,
                                 6.27340021e-01, 7.62571027e-01, 8.43506503e-01]])
        sim_sol_cov = np.empty(shape=(0, 11), dtype=np.float64)
        np.testing.assert_allclose(sim_sol_mean, sim_res_mom[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim_res_mom[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim_res_mom[2], rtol=1e-06, atol=1e-06)

    @pytest.mark.slow
    def test_division_model_l15_gillespie(self):
        net = me.Network('net_div_l15')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 15}])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1}
        theta_values = {'l': 0.23}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 1.     , 1.00493, 1.10301, 1.40927, 1.74484, 1.98433,
                                2.26114, 2.69884, 3.2382 , 3.79592]])
        sim_sol_var = np.array([[0.        , 0.        , 0.00490574, 0.09239986, 0.24225049,
                                0.20399541, 0.15758603, 0.33282923, 0.62928895, 0.76288839,
                                0.8444798 ]])
        np.testing.assert_allclose(sim_sol_mean, np.mean(sims, axis=0),
                                                            rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, np.var(sims, axis=0, ddof=1),
                                                            rtol=0.2, atol=0.2)

    ### minimal model d4 l3
    def test_minimal_model_d4_l3_moments(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[1.00000000e+00, 9.99661300e-01, 9.95786822e-01, 9.83366954e-01,
                                 9.58874286e-01, 9.21186515e-01, 8.71262892e-01, 8.11430693e-01,
                                 7.44676766e-01, 6.74094967e-01, 6.02519716e-01, 5.32323178e-01,
                                 4.65336116e-01, 4.02852123e-01, 3.45681548e-01, 2.94229914e-01,
                                 2.48583556e-01, 2.08591717e-01, 1.73939147e-01, 1.44206591e-01,
                                 1.18918774e-01, 9.75807212e-02, 7.97039295e-02, 6.48240663e-02,
                                 5.25120773e-02, 4.23801158e-02, 3.40838378e-02, 2.73219929e-02],
                                [0.00000000e+00, 3.38772857e-04, 4.22009262e-03, 1.67215423e-02,
                                 4.16241791e-02, 8.06083680e-02, 1.33617836e-01, 1.99525682e-01,
                                 2.76739218e-01, 3.63634007e-01, 4.58818192e-01, 5.61265730e-01,
                                 6.70361616e-01, 7.85894491e-01, 9.08021977e-01, 1.03722529e+00,
                                 1.17426313e+00, 1.32013046e+00, 1.47602480e+00, 1.64332093e+00,
                                 1.82355402e+00, 2.01841042e+00, 2.22972532e+00, 2.45948640e+00,
                                 2.70984247e+00, 2.98311663e+00, 3.28182300e+00, 3.60868691e+00]])
        sim_sol_var = np.array([[0.00000000e+00, 3.38585676e-04, 4.19542729e-03, 1.63563879e-02,
                                 3.94343898e-02, 7.26019193e-02, 1.12163865e-01, 1.53010923e-01,
                                 1.90133280e-01, 2.19690943e-01, 2.39489708e-01, 2.48955212e-01,
                                 2.48798415e-01, 2.40562290e-01, 2.26185816e-01, 2.07658672e-01,
                                 1.86789772e-01, 1.65081213e-01, 1.43684320e-01, 1.23411050e-01,
                                 1.04777099e-01, 8.80587240e-02, 7.33512132e-02, 6.06219068e-02,
                                 4.97545591e-02, 4.05840416e-02, 3.29221298e-02, 2.65755016e-02],
                                [0.00000000e+00, 3.38803034e-04, 4.21612622e-03, 1.66195131e-02,
                                 4.08962765e-02, 7.77541268e-02, 1.25774835e-01, 1.82506706e-01,
                                 2.45524937e-01, 3.13171241e-01, 3.84932265e-01, 4.61544157e-01,
                                 5.44942233e-01, 6.38155787e-01, 7.45214674e-01, 8.71104465e-01,
                                 1.02178677e+00, 1.20429010e+00, 1.42687214e+00, 1.69925365e+00,
                                 2.03292608e+00, 2.44153841e+00, 2.94137389e+00, 3.55192500e+00,
                                 4.29658892e+00, 5.20350641e+00, 6.30656621e+00, 7.64661662e+00]])
        sim_sol_cov = np.array([[ 0.        , -0.00033866, -0.00420231, -0.01644341, -0.03991236,
                                 -0.07425534, -0.11641626, -0.16190126, -0.20608127, -0.24512385,
                                 -0.27644701, -0.29877476, -0.31194347, -0.31659926, -0.31388644,
                                 -0.30518271, -0.2919025 , -0.27536828, -0.25673849, -0.23697771,
                                 -0.21685481, -0.19695794, -0.17771787, -0.15943391, -0.14229946,
                                 -0.12642483, -0.11185712, -0.09859652]])

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    @pytest.mark.slow
    def test_minimal_model_d4_l3_gillespie(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)
        sim_res_cov = np.array([np.cov(sims[:, 0, i], sims[:, 1, i], ddof=1)[1,0]
                                                    for i in range(sims.shape[2])])

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.00000e+00, 9.99530e-01, 9.95480e-01, 9.83160e-01, 9.58880e-01,
                                9.21070e-01, 8.71110e-01, 8.11450e-01, 7.47190e-01, 6.77130e-01,
                                6.04910e-01, 5.35250e-01, 4.67990e-01, 4.04650e-01, 3.47950e-01,
                                2.96460e-01, 2.51030e-01, 2.11430e-01, 1.75970e-01, 1.45980e-01,
                                1.21450e-01, 9.95900e-02, 8.10600e-02, 6.65400e-02, 5.43100e-02,
                                4.37800e-02, 3.51900e-02, 2.80700e-02],
                               [0.00000e+00, 4.70000e-04, 4.53000e-03, 1.69600e-02, 4.16900e-02,
                                8.06800e-02, 1.33540e-01, 1.99480e-01, 2.74680e-01, 3.61030e-01,
                                4.56120e-01, 5.58050e-01, 6.67250e-01, 7.82160e-01, 9.04820e-01,
                                1.03354e+00, 1.16850e+00, 1.31146e+00, 1.46749e+00, 1.63472e+00,
                                1.81542e+00, 2.00863e+00, 2.21956e+00, 2.44913e+00, 2.69783e+00,
                                2.97141e+00, 3.26897e+00, 3.59363e+00]])
        sim_sol_var = np.array([[0.00000000e+00, 4.69783798e-04, 4.49961460e-03, 1.65565800e-02,
                                3.94295399e-02, 7.27007821e-02, 1.12278491e-01, 1.53000428e-01,
                                1.88898993e-01, 2.18627149e-01, 2.38996282e-01, 2.48759925e-01,
                                2.48977850e-01, 2.40910787e-01, 2.26883066e-01, 2.08573554e-01,
                                1.88015819e-01, 1.66729022e-01, 1.45006009e-01, 1.24671086e-01,
                                1.06700965e-01, 8.96727286e-02, 7.44900213e-02, 6.21130495e-02,
                                5.13609375e-02, 4.18637302e-02, 3.39520034e-02, 2.72823479e-02],
                               [0.00000000e+00, 4.69783798e-04, 4.52952440e-03, 1.69125275e-02,
                                4.10923548e-02, 7.76715143e-02, 1.25248321e-01, 1.82409554e-01,
                                2.45593354e-01, 3.13370473e-01, 3.83918385e-01, 4.61594813e-01,
                                5.45852896e-01, 6.37712112e-01, 7.49148259e-01, 8.74603814e-01,
                                1.02163797e+00, 1.20002467e+00, 1.41959730e+00, 1.69802750e+00,
                                2.03763060e+00, 2.43713989e+00, 2.93098272e+00, 3.55528780e+00,
                                4.30866638e+00, 5.20978471e+00, 6.29272807e+00, 7.63592978e+00]])
        sim_sol_cov = np.array([ 0.        , -0.00046978, -0.00450957, -0.01667456, -0.03997611,
                               -0.07431267, -0.11632919, -0.16186966, -0.2052402 , -0.24446669,
                               -0.27591431, -0.29869925, -0.31226945, -0.31650421, -0.31483527,
                               -0.30640633, -0.29333149, -0.27728476, -0.2582368 , -0.23863881,
                               -0.22048496, -0.20004146, -0.17991933, -0.16296674, -0.14652061,
                               -0.13008963, -0.1150362 , -0.1008742 ])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_cov, sim_res_cov, rtol=0.2, atol=0.2)

    @pytest.mark.slow
    def test_minimal_model_d4_l3_gillespie_independent(self):
        # add the summary stats here of an independent gillespie simulation
        # (see jupyter notebooks gillespie_test_1 and gillespie_test_2)

        # order of the variables of the hidden layer
        # X centric, X diff 0, X diff 1, X diff 2, Y centric, Y div 0, Y div 1
        initial_values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        d = 0.04
        l = 0.06
        num_iter = 10000

        # save result as external simulation results
        sims = gill_indep_test_1(initial_values, time_values, d, l, num_iter)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)
        sim_res_cov = np.array([np.cov(sims[:, 0, i], sims[:, 1, i], ddof=1)[1,0]
                                                    for i in range(sims.shape[2])])

        sim_sol_mean = np.array([[1.00000e+00, 9.99650e-01, 9.95750e-01, 9.83370e-01, 9.58770e-01,
                                9.20350e-01, 8.71090e-01, 8.11320e-01, 7.44530e-01, 6.74820e-01,
                                6.02140e-01, 5.32020e-01, 4.65220e-01, 4.02320e-01, 3.45360e-01,
                                2.94870e-01, 2.48850e-01, 2.09330e-01, 1.73650e-01, 1.43200e-01,
                                1.18660e-01, 9.80300e-02, 8.02000e-02, 6.51500e-02, 5.28200e-02,
                                4.25100e-02, 3.41700e-02, 2.79700e-02],
                               [0.00000e+00, 3.50000e-04, 4.25000e-03, 1.67500e-02, 4.18300e-02,
                                8.15000e-02, 1.33830e-01, 1.99760e-01, 2.77570e-01, 3.63560e-01,
                                4.59590e-01, 5.62800e-01, 6.71670e-01, 7.88180e-01, 9.11180e-01,
                                1.04066e+00, 1.17833e+00, 1.32300e+00, 1.48030e+00, 1.64735e+00,
                                1.82632e+00, 2.01993e+00, 2.22942e+00, 2.46117e+00, 2.71453e+00,
                                2.98650e+00, 3.28881e+00, 3.61411e+00]])
        sim_sol_var = np.array([[0.00000000e+00, 3.49880999e-04, 4.23197982e-03, 1.63536066e-02,
                                3.95304824e-02, 7.33066106e-02, 1.12293335e-01, 1.53081388e-01,
                                1.90206981e-01, 2.19440162e-01, 2.39569816e-01, 2.48977209e-01,
                                2.48792840e-01, 2.40461022e-01, 2.26088731e-01, 2.07923762e-01,
                                1.86925547e-01, 1.65512606e-01, 1.43497112e-01, 1.22694987e-01,
                                1.04580850e-01, 8.84210033e-02, 7.37686977e-02, 6.09060866e-02,
                                5.00305479e-02, 4.07033069e-02, 3.30027411e-02, 2.71879510e-02],
                               [0.00000000e+00, 3.49880999e-04, 4.23197982e-03, 1.67096046e-02,
                                4.13006641e-02, 7.86585366e-02, 1.26000791e-01, 1.82957772e-01,
                                2.47507370e-01, 3.14787274e-01, 3.85850890e-01, 4.63920799e-01,
                                5.49894910e-01, 6.42418712e-01, 7.52358531e-01, 8.80135566e-01,
                                1.03315874e+00, 1.21644316e+00, 1.44056632e+00, 1.71260510e+00,
                                2.04427570e+00, 2.45859738e+00, 2.95611602e+00, 3.58072804e+00,
                                4.35016038e+00, 5.24087016e+00, 6.38366262e+00, 7.75105642e+00]])
        sim_sol_cov = np.array([ 0.        , -0.00034988, -0.00423198, -0.01647161, -0.04010575,
                               -0.07500928, -0.11657914, -0.1620709 , -0.20666126, -0.24534001,
                               -0.27674029, -0.29942385, -0.31247744, -0.31710375, -0.31468827,
                               -0.30686248, -0.29323035, -0.27694636, -0.25705667, -0.23590288,
                               -0.2167133 , -0.19801572, -0.17880127, -0.16034683, -0.14338291,
                               -0.12695738, -0.11237976, -0.10108767])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_cov, sim_res_cov, rtol=0.2, atol=0.2)

    ### parallel2 model df2 ds4 l3
    def test_parallel2_model_df2_ds4_l3_moments(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'S_t', 'end': 'P1_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': int(4/2)},
            {'start': 'P1_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': int(4/2)},

            {'start': 'S_t', 'end': 'P2_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': int(2/2)},
            {'start': 'P2_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': int(2/2)},

            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('S_t',): 1.0, ('Y_t',): 0.0, ('P1_t',): 0.0, ('P2_t',): 0.0,
                        ('S_t','S_t'): 0.0, ('Y_t','Y_t'): 0.0, ('P1_t','P1_t'): 0.0, ('P2_t','P2_t'): 0.0,
                        ('S_t','Y_t'): 0.0, ('S_t','P1_t'): 0.0, ('S_t','P2_t'): 0.0, ('Y_t','P1_t'): 0.0,
                        ('Y_t','P2_t'): 0.0, ('P1_t','P2_t'): 0.0}
        theta_values = {'l': 0.06, 'd4': 0.06, 'd2': 0.08}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('S_t', 'P1_t', 'P2_t'), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        res = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[1.        , 0.9892332 , 0.9628573 , 0.92641326, 0.88291527,
                                 0.83425905, 0.78189005, 0.72708038, 0.671013  , 0.61478239,
                                 0.55936821, 0.50560865, 0.45418393, 0.40561213, 0.36025565,
                                 0.31833546, 0.27994993, 0.24509568, 0.21368863, 0.18558362,
                                 0.16059206, 0.13849701, 0.11906575, 0.10205975, 0.08724246,
                                 0.07438504, 0.06327037, 0.05369577],
                                [0.        , 0.01077441, 0.03733088, 0.07470416, 0.1208133 ,
                                 0.1748674 , 0.23655813, 0.30571417, 0.38219578, 0.46589589,
                                 0.55677665, 0.6549082 , 0.76049766, 0.8739066 , 0.99565908,
                                 1.12644342, 1.2671107 , 1.41867221, 1.58229732, 1.75931293,
                                 1.95120503, 2.15962273, 2.38638491, 2.63348958, 2.90312602,
                                 3.1976896 , 3.51979935, 3.87231827]])
        sim_sol_var = np.array([[0.00000000e+00, 1.06508802e-02, 3.57631223e-02, 6.81717355e-02,
                                 1.03375893e-01, 1.38270887e-01, 1.70537997e-01, 1.98434501e-01,
                                 2.20754555e-01, 2.36825004e-01, 2.46475416e-01, 2.49968543e-01,
                                 2.47900888e-01, 2.41090929e-01, 2.30471516e-01, 2.16997996e-01,
                                 2.01577965e-01, 1.85023789e-01, 1.68025800e-01, 1.51142341e-01,
                                 1.34802248e-01, 1.19315590e-01, 1.04889097e-01, 9.16435597e-02,
                                 7.96312172e-02, 6.88519022e-02, 5.92672286e-02, 5.08125332e-02],
                                [0.00000000e+00, 1.06735467e-02, 3.63144252e-02, 7.13734183e-02,
                                 1.13789945e-01, 1.63071913e-01, 2.19273854e-01, 2.82664130e-01,
                                 3.53752593e-01, 4.33448130e-01, 5.23227776e-01, 6.25276607e-01,
                                 7.42598855e-01, 8.79117068e-01, 1.03977907e+00, 1.23068991e+00,
                                 1.45928261e+00, 1.73453877e+00, 2.06726949e+00, 2.47046781e+00,
                                 2.95974360e+00, 3.55385666e+00, 4.27536741e+00, 5.15142343e+00,
                                 6.21471343e+00, 7.50461864e+00, 9.06860280e+00, 1.09638889e+01]])
        sim_sol_cov = np.array([[ 0.        , -0.01065841, -0.03594431, -0.06920692, -0.10666791,
                                 -0.14588471, -0.18496245, -0.22227878, -0.25645833, -0.28642458,
                                 -0.31144316, -0.33112725, -0.34540582, -0.35446711, -0.35869181,
                                 -0.35858689, -0.35472755, -0.34771044, -0.33811895, -0.32649966,
                                 -0.31334803, -0.2991013 , -0.28413671, -0.2687733 , -0.25327587,
                                 -0.23786026, -0.222699  , -0.20792711]])

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    def test_parallel2_model_df2_ds4_l3_moments_multiedge(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        # differentiation theta values have to be half now
        theta_values = {'l': 0.06, 'd4': 0.03, 'd2': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        res = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[1.        , 0.9892332 , 0.9628573 , 0.92641326, 0.88291527,
                                 0.83425905, 0.78189005, 0.72708038, 0.671013  , 0.61478239,
                                 0.55936821, 0.50560865, 0.45418393, 0.40561213, 0.36025565,
                                 0.31833546, 0.27994993, 0.24509568, 0.21368863, 0.18558362,
                                 0.16059206, 0.13849701, 0.11906575, 0.10205975, 0.08724246,
                                 0.07438504, 0.06327037, 0.05369577],
                                [0.        , 0.01077441, 0.03733088, 0.07470416, 0.1208133 ,
                                 0.1748674 , 0.23655813, 0.30571417, 0.38219578, 0.46589589,
                                 0.55677665, 0.6549082 , 0.76049766, 0.8739066 , 0.99565908,
                                 1.12644342, 1.2671107 , 1.41867221, 1.58229732, 1.75931293,
                                 1.95120503, 2.15962273, 2.38638491, 2.63348958, 2.90312602,
                                 3.1976896 , 3.51979935, 3.87231827]])
        sim_sol_var = np.array([[0.00000000e+00, 1.06508802e-02, 3.57631223e-02, 6.81717355e-02,
                                 1.03375893e-01, 1.38270887e-01, 1.70537997e-01, 1.98434501e-01,
                                 2.20754555e-01, 2.36825004e-01, 2.46475416e-01, 2.49968543e-01,
                                 2.47900888e-01, 2.41090929e-01, 2.30471516e-01, 2.16997996e-01,
                                 2.01577965e-01, 1.85023789e-01, 1.68025800e-01, 1.51142341e-01,
                                 1.34802248e-01, 1.19315590e-01, 1.04889097e-01, 9.16435597e-02,
                                 7.96312172e-02, 6.88519022e-02, 5.92672286e-02, 5.08125332e-02],
                                [0.00000000e+00, 1.06735467e-02, 3.63144252e-02, 7.13734183e-02,
                                 1.13789945e-01, 1.63071913e-01, 2.19273854e-01, 2.82664130e-01,
                                 3.53752593e-01, 4.33448130e-01, 5.23227776e-01, 6.25276607e-01,
                                 7.42598855e-01, 8.79117068e-01, 1.03977907e+00, 1.23068991e+00,
                                 1.45928261e+00, 1.73453877e+00, 2.06726949e+00, 2.47046781e+00,
                                 2.95974360e+00, 3.55385666e+00, 4.27536741e+00, 5.15142343e+00,
                                 6.21471343e+00, 7.50461864e+00, 9.06860280e+00, 1.09638889e+01]])
        sim_sol_cov = np.array([[ 0.        , -0.01065841, -0.03594431, -0.06920692, -0.10666791,
                                 -0.14588471, -0.18496245, -0.22227878, -0.25645833, -0.28642458,
                                 -0.31144316, -0.33112725, -0.34540582, -0.35446711, -0.35869181,
                                 -0.35858689, -0.35472755, -0.34771044, -0.33811895, -0.32649966,
                                 -0.31334803, -0.2991013 , -0.28413671, -0.2687733 , -0.25327587,
                                 -0.23786026, -0.222699  , -0.20792711]])

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    @pytest.mark.slow
    def test_parallel2_model_df2_ds4_l3_gillespie(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'S_t', 'end': 'P1_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': int(4/2)},
            {'start': 'P1_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': int(4/2)},

            {'start': 'S_t', 'end': 'P2_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': int(2/2)},
            {'start': 'P2_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': int(2/2)},

            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {'S_t': 1, 'Y_t': 0, 'P1_t': 0, 'P2_t': 0}
        theta_values = {'l': 0.06, 'd4': 0.06, 'd2': 0.08}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('S_t', 'P1_t', 'P2_t'), 'Y_t': ('Y_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)
        sim_res_cov = np.array([np.cov(sims[:, 0, i], sims[:, 1, i], ddof=1)[1,0]
                                                    for i in range(sims.shape[2])])

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 0.98917, 0.96263, 0.92617, 0.88359, 0.83522, 0.78319,
                                0.72803, 0.67007, 0.61486, 0.55918, 0.50559, 0.45419, 0.40623,
                                0.36079, 0.31911, 0.28094, 0.24543, 0.21371, 0.18521, 0.15994,
                                0.1384 , 0.11941, 0.10197, 0.08769, 0.07461, 0.06377, 0.05413],
                               [0.     , 0.01084, 0.03752, 0.07478, 0.12034, 0.17392, 0.23561,
                                0.30488, 0.38269, 0.46588, 0.55629, 0.65378, 0.75971, 0.87289,
                                0.99413, 1.1231 , 1.26451, 1.41578, 1.58057, 1.75827, 1.94899,
                                2.15551, 2.38079, 2.62727, 2.89559, 3.19119, 3.51406, 3.86838]])
        sim_sol_var = np.array([[0.00000000e+00, 1.07128182e-02, 3.59738428e-02, 6.83798149e-02,
                                1.02859740e-01, 1.37628928e-01, 1.69805122e-01, 1.98004299e-01,
                                2.21078406e-01, 2.36809548e-01, 2.46500193e-01, 2.49971252e-01,
                                2.47903923e-01, 2.41209599e-01, 2.30622882e-01, 2.17280981e-01,
                                2.02014737e-01, 1.85195967e-01, 1.68039716e-01, 1.50908765e-01,
                                1.34360540e-01, 1.19246632e-01, 1.05152303e-01, 9.15730348e-02,
                                8.00012639e-02, 6.90440383e-02, 5.97039841e-02, 5.12004551e-02],
                               [0.00000000e+00, 1.07426018e-02, 3.64126137e-02, 7.11286629e-02,
                                1.13819423e-01, 1.62513459e-01, 2.19480123e-01, 2.82591012e-01,
                                3.53281897e-01, 4.33220158e-01, 5.20336639e-01, 6.22157933e-01,
                                7.40698123e-01, 8.81641864e-01, 1.04132596e+00, 1.22763867e+00,
                                1.45927905e+00, 1.72986429e+00, 2.05994907e+00, 2.46300124e+00,
                                2.94169740e+00, 3.53208196e+00, 4.25209150e+00, 5.10897344e+00,
                                6.18169037e+00, 7.47975118e+00, 9.04491277e+00, 1.09330855e+01]])
        sim_sol_cov = np.array([ 0.        , -0.01072271, -0.03611824, -0.06925969, -0.10633228,
                               -0.14526292, -0.18452924, -0.22196401, -0.25643165, -0.28645384,
                               -0.31106935, -0.33054794, -0.34505614, -0.35459765, -0.35867575,
                               -0.35839602, -0.35525499, -0.34747836, -0.33778699, -0.32565244,
                               -0.31172458, -0.29832557, -0.28429298, -0.2679054 , -0.25391683,
                               -0.23809707, -0.22409385, -0.2093975 ])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_cov, sim_res_cov, rtol=0.2, atol=0.2)

    @pytest.mark.slow
    def test_parallel2_model_df2_ds4_l3_gillespie_multiedge(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        # differentiation theta values have to be half now
        theta_values = {'l': 0.06, 'd4': 0.03, 'd2': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)
        sim_res_cov = np.array([np.cov(sims[:, 0, i], sims[:, 1, i], ddof=1)[1,0]
                                                    for i in range(sims.shape[2])])

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 0.98917, 0.96263, 0.92617, 0.88359, 0.83522, 0.78319,
                                0.72803, 0.67007, 0.61486, 0.55918, 0.50559, 0.45419, 0.40623,
                                0.36079, 0.31911, 0.28094, 0.24543, 0.21371, 0.18521, 0.15994,
                                0.1384 , 0.11941, 0.10197, 0.08769, 0.07461, 0.06377, 0.05413],
                               [0.     , 0.01084, 0.03752, 0.07478, 0.12034, 0.17392, 0.23561,
                                0.30488, 0.38269, 0.46588, 0.55629, 0.65378, 0.75971, 0.87289,
                                0.99413, 1.1231 , 1.26451, 1.41578, 1.58057, 1.75827, 1.94899,
                                2.15551, 2.38079, 2.62727, 2.89559, 3.19119, 3.51406, 3.86838]])
        sim_sol_var = np.array([[0.00000000e+00, 1.07128182e-02, 3.59738428e-02, 6.83798149e-02,
                                1.02859740e-01, 1.37628928e-01, 1.69805122e-01, 1.98004299e-01,
                                2.21078406e-01, 2.36809548e-01, 2.46500193e-01, 2.49971252e-01,
                                2.47903923e-01, 2.41209599e-01, 2.30622882e-01, 2.17280981e-01,
                                2.02014737e-01, 1.85195967e-01, 1.68039716e-01, 1.50908765e-01,
                                1.34360540e-01, 1.19246632e-01, 1.05152303e-01, 9.15730348e-02,
                                8.00012639e-02, 6.90440383e-02, 5.97039841e-02, 5.12004551e-02],
                               [0.00000000e+00, 1.07426018e-02, 3.64126137e-02, 7.11286629e-02,
                                1.13819423e-01, 1.62513459e-01, 2.19480123e-01, 2.82591012e-01,
                                3.53281897e-01, 4.33220158e-01, 5.20336639e-01, 6.22157933e-01,
                                7.40698123e-01, 8.81641864e-01, 1.04132596e+00, 1.22763867e+00,
                                1.45927905e+00, 1.72986429e+00, 2.05994907e+00, 2.46300124e+00,
                                2.94169740e+00, 3.53208196e+00, 4.25209150e+00, 5.10897344e+00,
                                6.18169037e+00, 7.47975118e+00, 9.04491277e+00, 1.09330855e+01]])
        sim_sol_cov = np.array([ 0.        , -0.01072271, -0.03611824, -0.06925969, -0.10633228,
                               -0.14526292, -0.18452924, -0.22196401, -0.25643165, -0.28645384,
                               -0.31106935, -0.33054794, -0.34505614, -0.35459765, -0.35867575,
                               -0.35839602, -0.35525499, -0.34747836, -0.33778699, -0.32565244,
                               -0.31172458, -0.29832557, -0.28429298, -0.2679054 , -0.25391683,
                               -0.23809707, -0.22409385, -0.2093975 ])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_cov, sim_res_cov, rtol=0.2, atol=0.2)

    @pytest.mark.slow
    def test_parallel2_model_df2_ds4_l3_gillespie_independent(self):
        # add the summary stats here of an independent gillespie simulation
        # (see jupyter notebooks gillespie_test_1 and gillespie_test_2)

        # order of the variables of the hidden layer
        # 0,1,2,3,4: X centric, X diff4 0, X diff4 1, X diff4 2, X diff2 0,
        # 5,6,7: Y centric, Y div 0, Y div 1
        initial_values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        d4 = 0.06
        d2 = 0.08
        l = 0.06

        num_iter = 10000
        sims = gill_indep_test_2(initial_values, time_values, d4, d2, l, num_iter)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)
        sim_res_cov = np.array([np.cov(sims[:, 0, i], sims[:, 1, i], ddof=1)[1,0]
                                                    for i in range(sims.shape[2])])

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 0.98943, 0.9631 , 0.92714, 0.88312, 0.83479, 0.78267,
                                0.72796, 0.67315, 0.61471, 0.5597 , 0.50621, 0.45568, 0.40713,
                                0.36222, 0.31951, 0.28133, 0.24632, 0.21476, 0.18607, 0.161  ,
                                0.13809, 0.11862, 0.10166, 0.08677, 0.07401, 0.06294, 0.05356],
                               [0.     , 0.0106 , 0.0371 , 0.07394, 0.12049, 0.17407, 0.23557,
                                0.30505, 0.3802 , 0.46596, 0.55587, 0.65428, 0.75853, 0.87224,
                                0.9934 , 1.12505, 1.26381, 1.41601, 1.58016, 1.7573 , 1.94785,
                                2.15775, 2.38401, 2.62926, 2.89853, 3.19214, 3.51472, 3.86759]])
        sim_sol_var = np.array([[0.00000000e+00, 1.04583797e-02, 3.55387454e-02, 6.75520959e-02,
                                1.03220098e-01, 1.37917035e-01, 1.70099372e-01, 1.98036219e-01,
                                2.20021278e-01, 2.36843984e-01, 2.46438374e-01, 2.49963936e-01,
                                2.48038218e-01, 2.41377577e-01, 2.31018982e-01, 2.17425534e-01,
                                2.02185453e-01, 1.85648314e-01, 1.68639829e-01, 1.51449470e-01,
                                1.35080351e-01, 1.19022342e-01, 1.04550341e-01, 9.13261577e-02,
                                7.92417595e-02, 6.85332052e-02, 5.89791462e-02, 5.06918333e-02],
                               [0.00000000e+00, 1.05477455e-02, 3.61239512e-02, 7.06735831e-02,
                                1.13273293e-01, 1.62051256e-01, 2.18438959e-01, 2.83377331e-01,
                                3.53771498e-01, 4.34305621e-01, 5.23163775e-01, 6.26783949e-01,
                                7.43369673e-01, 8.84046223e-01, 1.04654691e+00, 1.23784488e+00,
                                1.45946888e+00, 1.73526303e+00, 2.06791505e+00, 2.48022151e+00,
                                2.96636004e+00, 3.56040054e+00, 4.26600898e+00, 5.13526321e+00,
                                6.18845572e+00, 7.46423686e+00, 9.04141374e+00, 1.09308469e+01]])
        sim_sol_cov = np.array([ 0.        , -0.01048806, -0.03573137, -0.06855342, -0.10640819,
                               -0.14531335, -0.18437542, -0.22206642, -0.25593419, -0.28643314,
                               -0.31112355, -0.33120639, -0.34565041, -0.35511862, -0.35983295,
                               -0.35946832, -0.35555122, -0.34879507, -0.33935856, -0.32698408,
                               -0.31360699, -0.29796668, -0.28279409, -0.26729324, -0.25150796,
                               -0.23625264, -0.22121869, -0.20715019])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_cov, sim_res_cov, rtol=0.2, atol=0.2)

    @pytest.mark.slow
    def test_parallel2_model_df2_ds4_l3_gillespie_independent_multiedge(self):
        # add the summary stats here of an independent gillespie simulation
        # (see jupyter notebooks gillespie_test_1 and gillespie_test_2)

        # order of the variables of the hidden layer
        # 0,1,2,3,4: X centric, X diff4 0, X diff4 1, X diff4 2, X diff2 0,
        # 5,6,7: Y centric, Y div 0, Y div 1
        initial_values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        d4 = 0.03
        d2 = 0.04
        l = 0.06

        num_iter = 10000
        sims = gill_indep_test_2_multiedge(initial_values, time_values, d4, d2, l, num_iter)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)
        sim_res_cov = np.array([np.cov(sims[:, 0, i], sims[:, 1, i], ddof=1)[1,0]
                                                    for i in range(sims.shape[2])])

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.     , 0.98943, 0.9631 , 0.92714, 0.88312, 0.83479, 0.78267,
                                0.72796, 0.67315, 0.61471, 0.5597 , 0.50621, 0.45568, 0.40713,
                                0.36222, 0.31951, 0.28133, 0.24632, 0.21476, 0.18607, 0.161  ,
                                0.13809, 0.11862, 0.10166, 0.08677, 0.07401, 0.06294, 0.05356],
                               [0.     , 0.0106 , 0.0371 , 0.07394, 0.12049, 0.17407, 0.23557,
                                0.30505, 0.3802 , 0.46596, 0.55587, 0.65428, 0.75853, 0.87224,
                                0.9934 , 1.12505, 1.26381, 1.41601, 1.58016, 1.7573 , 1.94785,
                                2.15775, 2.38401, 2.62926, 2.89853, 3.19214, 3.51472, 3.86759]])
        sim_sol_var = np.array([[0.00000000e+00, 1.04583797e-02, 3.55387454e-02, 6.75520959e-02,
                                1.03220098e-01, 1.37917035e-01, 1.70099372e-01, 1.98036219e-01,
                                2.20021278e-01, 2.36843984e-01, 2.46438374e-01, 2.49963936e-01,
                                2.48038218e-01, 2.41377577e-01, 2.31018982e-01, 2.17425534e-01,
                                2.02185453e-01, 1.85648314e-01, 1.68639829e-01, 1.51449470e-01,
                                1.35080351e-01, 1.19022342e-01, 1.04550341e-01, 9.13261577e-02,
                                7.92417595e-02, 6.85332052e-02, 5.89791462e-02, 5.06918333e-02],
                               [0.00000000e+00, 1.05477455e-02, 3.61239512e-02, 7.06735831e-02,
                                1.13273293e-01, 1.62051256e-01, 2.18438959e-01, 2.83377331e-01,
                                3.53771498e-01, 4.34305621e-01, 5.23163775e-01, 6.26783949e-01,
                                7.43369673e-01, 8.84046223e-01, 1.04654691e+00, 1.23784488e+00,
                                1.45946888e+00, 1.73526303e+00, 2.06791505e+00, 2.48022151e+00,
                                2.96636004e+00, 3.56040054e+00, 4.26600898e+00, 5.13526321e+00,
                                6.18845572e+00, 7.46423686e+00, 9.04141374e+00, 1.09308469e+01]])
        sim_sol_cov = np.array([ 0.        , -0.01048806, -0.03573137, -0.06855342, -0.10640819,
                               -0.14531335, -0.18437542, -0.22206642, -0.25593419, -0.28643314,
                               -0.31112355, -0.33120639, -0.34565041, -0.35511862, -0.35983295,
                               -0.35946832, -0.35555122, -0.34879507, -0.33935856, -0.32698408,
                               -0.31360699, -0.29796668, -0.28279409, -0.26729324, -0.25150796,
                               -0.23625264, -0.22121869, -0.20715019])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_cov, sim_res_cov, rtol=0.2, atol=0.2)

    ### tests with sparse and sum simulation variables
    ### with minimal model d4 l3
    def test_minimal_model_d4_l3_moments_sparse_variables_X(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        res = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[1.00000000e+00, 9.99661300e-01, 9.95786822e-01, 9.83366954e-01,
                                 9.58874286e-01, 9.21186515e-01, 8.71262892e-01, 8.11430693e-01,
                                 7.44676766e-01, 6.74094967e-01, 6.02519716e-01, 5.32323178e-01,
                                 4.65336116e-01, 4.02852123e-01, 3.45681548e-01, 2.94229914e-01,
                                 2.48583556e-01, 2.08591717e-01, 1.73939147e-01, 1.44206591e-01,
                                 1.18918774e-01, 9.75807212e-02, 7.97039295e-02, 6.48240663e-02,
                                 5.25120773e-02, 4.23801158e-02, 3.40838378e-02, 2.73219929e-02]])
        sim_sol_var = np.array([[0.00000000e+00, 3.38585676e-04, 4.19542729e-03, 1.63563879e-02,
                                 3.94343898e-02, 7.26019193e-02, 1.12163865e-01, 1.53010923e-01,
                                 1.90133280e-01, 2.19690943e-01, 2.39489708e-01, 2.48955212e-01,
                                 2.48798415e-01, 2.40562290e-01, 2.26185816e-01, 2.07658672e-01,
                                 1.86789772e-01, 1.65081213e-01, 1.43684320e-01, 1.23411050e-01,
                                 1.04777099e-01, 8.80587240e-02, 7.33512132e-02, 6.06219068e-02,
                                 4.97545591e-02, 4.05840416e-02, 3.29221298e-02, 2.65755016e-02]])
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    @pytest.mark.slow
    def test_minimal_model_d4_l3_gillespie_sparse_variables_X(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[1.00000e+00, 9.99530e-01, 9.95480e-01, 9.83160e-01, 9.58880e-01,
                                9.21070e-01, 8.71110e-01, 8.11450e-01, 7.47190e-01, 6.77130e-01,
                                6.04910e-01, 5.35250e-01, 4.67990e-01, 4.04650e-01, 3.47950e-01,
                                2.96460e-01, 2.51030e-01, 2.11430e-01, 1.75970e-01, 1.45980e-01,
                                1.21450e-01, 9.95900e-02, 8.10600e-02, 6.65400e-02, 5.43100e-02,
                                4.37800e-02, 3.51900e-02, 2.80700e-02]])
        sim_sol_var = np.array([[0.00000000e+00, 4.69783798e-04, 4.49961460e-03, 1.65565800e-02,
                                3.94295399e-02, 7.27007821e-02, 1.12278491e-01, 1.53000428e-01,
                                1.88898993e-01, 2.18627149e-01, 2.38996282e-01, 2.48759925e-01,
                                2.48977850e-01, 2.40910787e-01, 2.26883066e-01, 2.08573554e-01,
                                1.88015819e-01, 1.66729022e-01, 1.45006009e-01, 1.24671086e-01,
                                1.06700965e-01, 8.96727286e-02, 7.44900213e-02, 6.21130495e-02,
                                5.13609375e-02, 4.18637302e-02, 3.39520034e-02, 2.72823479e-02]])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)

    def test_minimal_model_d4_l3_moments_sparse_variables_Y(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        res = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        sim_sol_mean = np.array([[0.00000000e+00, 3.38772857e-04, 4.22009262e-03, 1.67215423e-02,
                                 4.16241791e-02, 8.06083680e-02, 1.33617836e-01, 1.99525682e-01,
                                 2.76739218e-01, 3.63634007e-01, 4.58818192e-01, 5.61265730e-01,
                                 6.70361616e-01, 7.85894491e-01, 9.08021977e-01, 1.03722529e+00,
                                 1.17426313e+00, 1.32013046e+00, 1.47602480e+00, 1.64332093e+00,
                                 1.82355402e+00, 2.01841042e+00, 2.22972532e+00, 2.45948640e+00,
                                 2.70984247e+00, 2.98311663e+00, 3.28182300e+00, 3.60868691e+00]])
        sim_sol_var = np.array([[0.00000000e+00, 3.38803034e-04, 4.21612622e-03, 1.66195131e-02,
                                 4.08962765e-02, 7.77541268e-02, 1.25774835e-01, 1.82506706e-01,
                                 2.45524937e-01, 3.13171241e-01, 3.84932265e-01, 4.61544157e-01,
                                 5.44942233e-01, 6.38155787e-01, 7.45214674e-01, 8.71104465e-01,
                                 1.02178677e+00, 1.20429010e+00, 1.42687214e+00, 1.69925365e+00,
                                 2.03292608e+00, 2.44153841e+00, 2.94137389e+00, 3.55192500e+00,
                                 4.29658892e+00, 5.20350641e+00, 6.30656621e+00, 7.64661662e+00]])
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    @pytest.mark.slow
    def test_minimal_model_d4_l3_gillespie_sparse_variables_Y(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'Y_t': ('Y_t', )}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)

        # the solutions have been computed with num_iter = 100000
        sim_sol_mean = np.array([[0.00000e+00, 4.70000e-04, 4.53000e-03, 1.69600e-02, 4.16900e-02,
                                8.06800e-02, 1.33540e-01, 1.99480e-01, 2.74680e-01, 3.61030e-01,
                                4.56120e-01, 5.58050e-01, 6.67250e-01, 7.82160e-01, 9.04820e-01,
                                1.03354e+00, 1.16850e+00, 1.31146e+00, 1.46749e+00, 1.63472e+00,
                                1.81542e+00, 2.00863e+00, 2.21956e+00, 2.44913e+00, 2.69783e+00,
                                2.97141e+00, 3.26897e+00, 3.59363e+00]])
        sim_sol_var = np.array([[0.00000000e+00, 4.69783798e-04, 4.52952440e-03, 1.69125275e-02,
                                4.10923548e-02, 7.76715143e-02, 1.25248321e-01, 1.82409554e-01,
                                2.45593354e-01, 3.13370473e-01, 3.83918385e-01, 4.61594813e-01,
                                5.45852896e-01, 6.37712112e-01, 7.49148259e-01, 8.74603814e-01,
                                1.02163797e+00, 1.20002467e+00, 1.41959730e+00, 1.69802750e+00,
                                2.03763060e+00, 2.43713989e+00, 2.93098272e+00, 3.55528780e+00,
                                4.30866638e+00, 5.20978471e+00, 6.29272807e+00, 7.63592978e+00]])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)

    def test_minimal_model_d4_l3_moments_variables_sum_X_Y(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0,
                        ('X_t','X_t'): 0.0, ('Y_t','Y_t'): 0.0, ('X_t','Y_t'): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'T_t': ('Y_t', 'X_t')}

        sim = me.Simulation(net)
        res = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values)

        # NOTE that these solution can also be obtained with the X, Y
        # simuation variables from before and then calculate
        # E(T) = E(X) + E(Y), Var(T) = Var(X) + Var(Y) + 2 Cov(X, Y)
        # (which gives the same result as below)
        sim_sol_mean = np.array([[1.        , 1.00000007, 1.00000691, 1.0000885 , 1.00049847,
                             1.00179488, 1.00488073, 1.01095637, 1.02141598, 1.03772897,
                             1.06133791, 1.09358891, 1.13569773, 1.18874661, 1.25370353,
                             1.3314552 , 1.42284668, 1.52872218, 1.64996395, 1.78752752,
                             1.9424728 , 2.11599114, 2.30942925, 2.52431046, 2.76235455,
                             3.02549675, 3.31590684, 3.6360089 ]])
        sim_sol_var = np.array([[0.00000000e+00, 7.24812210e-08, 6.92827298e-06, 8.90767729e-05,
                             5.05956224e-04, 1.84536272e-03, 5.10617417e-03, 1.17151055e-02,
                             2.34956851e-02, 4.26144757e-02, 7.15279595e-02, 1.12949855e-01,
                             1.69853707e-01, 2.45519549e-01, 3.43627603e-01, 4.68397723e-01,
                             6.24771534e-01, 8.18634751e-01, 1.05707948e+00, 1.34870928e+00,
                             1.70399356e+00, 2.13568124e+00, 2.65928936e+00, 3.29367908e+00,
                             4.06174456e+00, 4.99124079e+00, 6.11577409e+00, 7.47599908e+00]])
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    @pytest.mark.slow
    def test_minimal_model_d4_l3_gillespie_variables_sum_X_Y(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {'X_t': 1, 'Y_t': 0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'T_t': ('Y_t', 'X_t')}

        num_iter = 10000
        sim = me.Simulation(net)
        res_list = list()
        for __ in range(num_iter):
            res_list.append(sim.simulate('gillespie', variables, theta_values, time_values,
                                        initial_values_type, initial_gillespie=initial_values)[1])

        sims = np.array(res_list)
        sim_res_mean =  np.mean(sims, axis=0)
        sim_res_var = np.var(sims, axis=0, ddof=1)

        # moment solutions used here
        sim_sol_mean = np.array([[1.        , 1.00000007, 1.00000691, 1.0000885 , 1.00049847,
                             1.00179488, 1.00488073, 1.01095637, 1.02141598, 1.03772897,
                             1.06133791, 1.09358891, 1.13569773, 1.18874661, 1.25370353,
                             1.3314552 , 1.42284668, 1.52872218, 1.64996395, 1.78752752,
                             1.9424728 , 2.11599114, 2.30942925, 2.52431046, 2.76235455,
                             3.02549675, 3.31590684, 3.6360089 ]])
        sim_sol_var = np.array([[0.00000000e+00, 7.24812210e-08, 6.92827298e-06, 8.90767729e-05,
                             5.05956224e-04, 1.84536272e-03, 5.10617417e-03, 1.17151055e-02,
                             2.34956851e-02, 4.26144757e-02, 7.15279595e-02, 1.12949855e-01,
                             1.69853707e-01, 2.45519549e-01, 3.43627603e-01, 4.68397723e-01,
                             6.24771534e-01, 8.18634751e-01, 1.05707948e+00, 1.34870928e+00,
                             1.70399356e+00, 2.13568124e+00, 2.65928936e+00, 3.29367908e+00,
                             4.06174456e+00, 4.99124079e+00, 6.11577409e+00, 7.47599908e+00]])

        np.testing.assert_allclose(sim_sol_mean, sim_res_mean, rtol=0.2, atol=0.2)
        np.testing.assert_allclose(sim_sol_var, sim_res_var, rtol=0.2, atol=0.2)

    ###### tests with mean only
    # only moment simulations need to be covered since gillespie produces complete counts
    ### mean only division model l2
    def test_division_model_l2_moments_mean_only(self):
        net = me.Network('net_div_l2')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 2}])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0}
        theta_values = {'l': 0.15}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.        , 1.03747108, 1.12877492, 1.25584829, 1.41121682,
                                 1.59270775, 1.80090981, 2.03796736, 2.30702475, 2.61198952,
                                 2.9574545 ]])
        sim_sol_var = np.empty(shape=(0, 11), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 11), dtype=np.float64)
        np.testing.assert_allclose(sim_sol_mean, sim_res_mom[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim_res_mom[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim_res_mom[2], rtol=1e-06, atol=1e-06)

    ### mean only division model l5
    def test_division_model_l5_moments_mean_only(self):
        net = me.Network('net_div_l5')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 5}])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0}
        theta_values = {'l': 0.22}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.        , 1.00543582, 1.07269814, 1.2418072 , 1.47882701,
                                 1.75244286, 2.06375747, 2.42784846, 2.85837618, 3.36664248,
                                 3.96524759]])
        sim_sol_var = np.empty(shape=(0, 11), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 11), dtype=np.float64)
        np.testing.assert_allclose(sim_sol_mean, sim_res_mom[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim_res_mom[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim_res_mom[2], rtol=1e-06, atol=1e-06)

    ### mean only division model l15
    def test_division_model_l15_moments_mean_only(self):
        net = me.Network('net_div_l15')
        net.structure([{'start': 'X_t', 'end': 'X_t',
                        'rate_symbol': 'l',
                        'type': 'S -> S + S', 'reaction_steps': 15}])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0}
        theta_values = {'l': 0.23}
        time_values = np.linspace(0.0, 10.0, endpoint=True, num=11)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.        , 1.0000036 , 1.00504242, 1.1029673 , 1.40858873,
                                 1.74539991, 1.9836416 , 2.2599425 , 2.69956532, 3.24078993,
                                 3.79706324]])
        sim_sol_var = np.empty(shape=(0, 11), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 11), dtype=np.float64)
        np.testing.assert_allclose(sim_sol_mean, sim_res_mom[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim_res_mom[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim_res_mom[2], rtol=1e-06, atol=1e-06)

    ### mean only minimal model d4 l3
    def test_minimal_model_d4_l3_moments_mean_only(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.00000000e+00, 9.99661300e-01, 9.95786822e-01, 9.83366954e-01,
                                 9.58874286e-01, 9.21186515e-01, 8.71262892e-01, 8.11430693e-01,
                                 7.44676766e-01, 6.74094967e-01, 6.02519716e-01, 5.32323178e-01,
                                 4.65336116e-01, 4.02852123e-01, 3.45681548e-01, 2.94229914e-01,
                                 2.48583556e-01, 2.08591717e-01, 1.73939147e-01, 1.44206591e-01,
                                 1.18918774e-01, 9.75807212e-02, 7.97039295e-02, 6.48240663e-02,
                                 5.25120773e-02, 4.23801158e-02, 3.40838378e-02, 2.73219929e-02],
                                [0.00000000e+00, 3.38772857e-04, 4.22009262e-03, 1.67215423e-02,
                                 4.16241791e-02, 8.06083680e-02, 1.33617836e-01, 1.99525682e-01,
                                 2.76739218e-01, 3.63634007e-01, 4.58818192e-01, 5.61265730e-01,
                                 6.70361616e-01, 7.85894491e-01, 9.08021977e-01, 1.03722529e+00,
                                 1.17426313e+00, 1.32013046e+00, 1.47602480e+00, 1.64332093e+00,
                                 1.82355402e+00, 2.01841042e+00, 2.22972532e+00, 2.45948640e+00,
                                 2.70984247e+00, 2.98311663e+00, 3.28182300e+00, 3.60868691e+00]])
        sim_sol_var = np.empty(shape=(0, 28), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    ### mean only parallel2 model df2 ds4 l3
    def test_parallel2_model_df2_ds4_l3_moments_mean_only(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'S_t', 'end': 'P1_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': int(4/2)},
            {'start': 'P1_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': int(4/2)},

            {'start': 'S_t', 'end': 'P2_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': int(2/2)},
            {'start': 'P2_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': int(2/2)},

            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('S_t',): 1.0, ('Y_t',): 0.0, ('P1_t',): 0.0, ('P2_t',): 0.0}
        theta_values = {'l': 0.06, 'd4': 0.06, 'd2': 0.08}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('S_t', 'P1_t', 'P2_t'), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.        , 0.9892332 , 0.9628573 , 0.92641326, 0.88291527,
                                 0.83425905, 0.78189005, 0.72708038, 0.671013  , 0.61478239,
                                 0.55936821, 0.50560865, 0.45418393, 0.40561213, 0.36025565,
                                 0.31833546, 0.27994993, 0.24509568, 0.21368863, 0.18558362,
                                 0.16059206, 0.13849701, 0.11906575, 0.10205975, 0.08724246,
                                 0.07438504, 0.06327037, 0.05369577],
                                [0.        , 0.01077441, 0.03733088, 0.07470416, 0.1208133 ,
                                 0.1748674 , 0.23655813, 0.30571417, 0.38219578, 0.46589589,
                                 0.55677665, 0.6549082 , 0.76049766, 0.8739066 , 0.99565908,
                                 1.12644342, 1.2671107 , 1.41867221, 1.58229732, 1.75931293,
                                 1.95120503, 2.15962273, 2.38638491, 2.63348958, 2.90312602,
                                 3.1976896 , 3.51979935, 3.87231827]])
        sim_sol_var = np.empty(shape=(0, 28), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    def test_parallel2_model_df2_ds4_l3_moments_mean_only_multiedge(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        # differentiation theta values have to be half now
        theta_values = {'l': 0.06, 'd4': 0.03, 'd2': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', ), 'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.        , 0.9892332 , 0.9628573 , 0.92641326, 0.88291527,
                                 0.83425905, 0.78189005, 0.72708038, 0.671013  , 0.61478239,
                                 0.55936821, 0.50560865, 0.45418393, 0.40561213, 0.36025565,
                                 0.31833546, 0.27994993, 0.24509568, 0.21368863, 0.18558362,
                                 0.16059206, 0.13849701, 0.11906575, 0.10205975, 0.08724246,
                                 0.07438504, 0.06327037, 0.05369577],
                                [0.        , 0.01077441, 0.03733088, 0.07470416, 0.1208133 ,
                                 0.1748674 , 0.23655813, 0.30571417, 0.38219578, 0.46589589,
                                 0.55677665, 0.6549082 , 0.76049766, 0.8739066 , 0.99565908,
                                 1.12644342, 1.2671107 , 1.41867221, 1.58229732, 1.75931293,
                                 1.95120503, 2.15962273, 2.38638491, 2.63348958, 2.90312602,
                                 3.1976896 , 3.51979935, 3.87231827]])
        sim_sol_var = np.empty(shape=(0, 28), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    ### mean only tests with sparse and sum simulation variables
    ### with minimal model d4 l3
    def test_minimal_model_d4_l3_moments_sparse_variables_X_mean_only(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'X_t': ('X_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[1.00000000e+00, 9.99661300e-01, 9.95786822e-01, 9.83366954e-01,
                                 9.58874286e-01, 9.21186515e-01, 8.71262892e-01, 8.11430693e-01,
                                 7.44676766e-01, 6.74094967e-01, 6.02519716e-01, 5.32323178e-01,
                                 4.65336116e-01, 4.02852123e-01, 3.45681548e-01, 2.94229914e-01,
                                 2.48583556e-01, 2.08591717e-01, 1.73939147e-01, 1.44206591e-01,
                                 1.18918774e-01, 9.75807212e-02, 7.97039295e-02, 6.48240663e-02,
                                 5.25120773e-02, 4.23801158e-02, 3.40838378e-02, 2.73219929e-02]])
        sim_sol_var = np.empty(shape=(0, 28), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    def test_minimal_model_d4_l3_moments_sparse_variables_Y_mean_only(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'Y_t': ('Y_t', )}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        sim_sol_mean = np.array([[0.00000000e+00, 3.38772857e-04, 4.22009262e-03, 1.67215423e-02,
                                 4.16241791e-02, 8.06083680e-02, 1.33617836e-01, 1.99525682e-01,
                                 2.76739218e-01, 3.63634007e-01, 4.58818192e-01, 5.61265730e-01,
                                 6.70361616e-01, 7.85894491e-01, 9.08021977e-01, 1.03722529e+00,
                                 1.17426313e+00, 1.32013046e+00, 1.47602480e+00, 1.64332093e+00,
                                 1.82355402e+00, 2.01841042e+00, 2.22972532e+00, 2.45948640e+00,
                                 2.70984247e+00, 2.98311663e+00, 3.28182300e+00, 3.60868691e+00]])
        sim_sol_var = np.empty(shape=(0, 28), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)

    def test_minimal_model_d4_l3_moments_variables_sum_X_Y_mean_only(self):
        net = me.Network('net_min')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        initial_values_type = 'synchronous'
        initial_values = {('X_t',): 1.0, ('Y_t',): 0.0}
        theta_values = {'l': 0.06, 'd': 0.04}
        time_values = np.linspace(0.0, 54.0, num=28, endpoint=True)
        variables = {'T_t': ('Y_t', 'X_t')}

        sim = me.Simulation(net)
        sim_res_mom = sim.simulate('moments', variables, theta_values, time_values,
                                initial_values_type, initial_moments=initial_values,
                                sim_mean_only=True)

        # NOTE that these solution can also be obtained with the X, Y
        # simuation variables from before and then calculate
        # E(T) = E(X) + E(Y), Var(T) = Var(X) + Var(Y) + 2 Cov(X, Y)
        # (which gives the same result as below)
        sim_sol_mean = np.array([[1.        , 1.00000007, 1.00000691, 1.0000885 , 1.00049847,
                             1.00179488, 1.00488073, 1.01095637, 1.02141598, 1.03772897,
                             1.06133791, 1.09358891, 1.13569773, 1.18874661, 1.25370353,
                             1.3314552 , 1.42284668, 1.52872218, 1.64996395, 1.78752752,
                             1.9424728 , 2.11599114, 2.30942925, 2.52431046, 2.76235455,
                             3.02549675, 3.31590684, 3.6360089 ]])
        sim_sol_var = np.empty(shape=(0, 28), dtype=np.float64)
        sim_sol_cov = np.empty(shape=(0, 28), dtype=np.float64)

        np.testing.assert_allclose(sim_sol_mean, sim.sim_moments_res[0], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_var, sim.sim_moments_res[1], rtol=1e-06, atol=1e-06)
        np.testing.assert_allclose(sim_sol_cov, sim.sim_moments_res[2], rtol=1e-06, atol=1e-06)


### helper function for testing
# function taken from gillespie_test_1 notebook
def gill_indep_test_1(initial_state, time_values, d, l, num_iter):
    # initialisation
    res = np.zeros((num_iter, 7, time_values.shape[0]))

    for i in range(num_iter):
        time_current = time_values[0]
        time_max = time_values[-1]
        cell_state = np.zeros(7)
        cell_state = copy.deepcopy(initial_state)

#         print('iter: ', i)
#         print('time_current: ', time_current)
#         print('time_max: ', time_max)
#         print('cell_state: ', cell_state)

        while time_current <= time_max:
            # reaction propensities
            reac_props = np.array([
                4.0 * d * cell_state[0], # d from Xcentric to X0
                4.0 * d * cell_state[1], # d from X0 to X1
                4.0 * d * cell_state[2], # d from X1 to X2
                4.0 * d * cell_state[3], # d from X2 to Ycentric
                3.0 * l * cell_state[4], # l from Ycentric to Y0
                3.0 * l * cell_state[5], # l from Y0 to Y1
                3.0 * l * cell_state[6], # l from Y1 to Ycentric
            ])
#             print('\n while start')
#             print('reac_props: ', reac_props)

            # draw exponential random time for next reaction
            total_prop = np.sum(reac_props)
            dt = np.random.exponential(1.0/total_prop)

#             print('total_prop: ', total_prop)
#             print('dt: ', dt)

            # save cell state results for relevant time points
            for ind in np.where((time_values >= time_current)
                                & (time_values < time_current + dt))[0]:
                res[i, :, ind] = copy.deepcopy(cell_state).reshape((1,7))

#             print('res: ', res)

            # draw which reaction takes place
            reac_probs = reac_props/np.sum(reac_props)
            reac_rand = np.random.choice(reac_props.shape[0], p=reac_probs)

#             print('reac_probs: ', reac_probs)
#             print('reac_rand: ', reac_rand)

            # update cell state according to selected reaction
            # differentiation
            if reac_rand==0:
                cell_state[0] += -1.0
                cell_state[1] += +1.0
            elif reac_rand==1:
                cell_state[1] += -1.0
                cell_state[2] += +1.0
            elif reac_rand==2:
                cell_state[2] += -1.0
                cell_state[3] += +1.0
            elif reac_rand==3:
                cell_state[3] += -1.0
                cell_state[4] += +1.0

            # division
            elif reac_rand==4:
                cell_state[4] += -1.0
                cell_state[5] += +1.0
            elif reac_rand==5:
                cell_state[5] += -1.0
                cell_state[6] += +1.0
            elif reac_rand==6:
                cell_state[6] += -1.0
                cell_state[4] += +2.0
            else:
                print('error')

            # update current time by delta t
            time_current += dt

#             print('cell_state: ', cell_state)
#             print('time_current: ', time_current)
#             print('\n')

    # sum hidden states to get observable layer
    res_obs = np.zeros((num_iter, 2, time_values.shape[0]))
    res_obs[:, 0, :] = np.sum(res[:, (0,1,2,3), :], axis=1)
    res_obs[:, 1, :] = np.sum(res[:, (4,5,6), :], axis=1)
    return res_obs

# function taken from gillespie_test_2 notebook
def gill_indep_test_2(initial_state, time_values, d4, d2, l, num_iter):
    # initialisation
    res = np.zeros((num_iter, 8, time_values.shape[0]))

    for i in range(num_iter):
        time_current = time_values[0]
        time_max = time_values[-1]
        cell_state = np.zeros(8)
        cell_state = copy.deepcopy(initial_state)

#         print('iter: ', i)
#         print('time_current: ', time_current)
#         print('time_max: ', time_max)
#         print('cell_state: ', cell_state)

        while time_current <= time_max:
            # reaction propensities
            reac_props = np.array([
                # NOTE: division by 2 in the diff channels is only due to
                # current workaround for multigraphs (with P intermediate nodes)

                # differentiation d4 channel
                4.0/2 * d4 * cell_state[0], # d from Xcentric to X0
                4.0/2 * d4 * cell_state[1], # d from X0 to X1
                4.0/2 * d4 * cell_state[2], # d from X1 to X2
                4.0/2 * d4 * cell_state[3], # d from X2 to Ycentric

                # differentiation d2 channel
                2.0/2 * d2 * cell_state[0], # d from Xcentric to X0
                2.0/2 * d2 * cell_state[4], # d from X0 to X1

                # division
                3.0 * l * cell_state[5], # l from Ycentric to Y0
                3.0 * l * cell_state[6], # l from Y0 to Y1
                3.0 * l * cell_state[7], # l from Y1 to Ycentric
            ])
#             print('\n while start')
#             print('reac_props: ', reac_props)

            # draw exponential random time for next reaction
            total_prop = np.sum(reac_props)
            dt = np.random.exponential(1.0/total_prop)

#             print('total_prop: ', total_prop)
#             print('dt: ', dt)

            # save cell state results for relevant time points
            for ind in np.where((time_values >= time_current)
                                & (time_values < time_current + dt))[0]:
                res[i, :, ind] = copy.deepcopy(cell_state).reshape((1,8))

#             print('res: ', res)

            # draw which reaction takes place
            reac_probs = reac_props/np.sum(reac_props)
            reac_rand = np.random.choice(reac_props.shape[0], p=reac_probs)

#             print('reac_probs: ', reac_probs)
#             print('reac_rand: ', reac_rand)

            # update cell state according to selected reaction
            # differentiation d4 channel
            if reac_rand==0:
                cell_state[0] += -1.0
                cell_state[1] += +1.0
            elif reac_rand==1:
                cell_state[1] += -1.0
                cell_state[2] += +1.0
            elif reac_rand==2:
                cell_state[2] += -1.0
                cell_state[3] += +1.0
            elif reac_rand==3:
                cell_state[3] += -1.0
                cell_state[5] += +1.0

            # differentiation d2 channel
            elif reac_rand==4:
                cell_state[0] += -1.0
                cell_state[4] += +1.0
            elif reac_rand==5:
                cell_state[4] += -1.0
                cell_state[5] += +1.0

            # division
            elif reac_rand==6:
                cell_state[5] += -1.0
                cell_state[6] += +1.0
            elif reac_rand==7:
                cell_state[6] += -1.0
                cell_state[7] += +1.0
            elif reac_rand==8:
                cell_state[7] += -1.0
                cell_state[5] += +2.0
            else:
                print('error')

            # update current time by delta t
            time_current += dt

#             print('cell_state: ', cell_state)
#             print('time_current: ', time_current)
#             print('\n')

    # sum hidden states to get observable layer
    res_obs = np.zeros((num_iter, 2, time_values.shape[0]))
    res_obs[:, 0, :] = np.sum(res[:, (0,1,2,3,4), :], axis=1)
    res_obs[:, 1, :] = np.sum(res[:, (5,6,7), :], axis=1)
    return res_obs

# function taken from gillespie_test_2 notebook
def gill_indep_test_2_multiedge(initial_state, time_values, d4, d2, l, num_iter):
    # initialisation
    res = np.zeros((num_iter, 8, time_values.shape[0]))

    for i in range(num_iter):
        time_current = time_values[0]
        time_max = time_values[-1]
        cell_state = np.zeros(8)
        cell_state = copy.deepcopy(initial_state)

#         print('iter: ', i)
#         print('time_current: ', time_current)
#         print('time_max: ', time_max)
#         print('cell_state: ', cell_state)

        while time_current <= time_max:
            # reaction propensities
            reac_props = np.array([
                # NOTE: division by 2 in the diff channels is only due to
                # current workaround for multigraphs (with P intermediate nodes)

                # differentiation d4 channel
                4.0 * d4 * cell_state[0], # d from Xcentric to X0
                4.0 * d4 * cell_state[1], # d from X0 to X1
                4.0 * d4 * cell_state[2], # d from X1 to X2
                4.0 * d4 * cell_state[3], # d from X2 to Ycentric

                # differentiation d2 channel
                2.0 * d2 * cell_state[0], # d from Xcentric to X0
                2.0 * d2 * cell_state[4], # d from X0 to X1

                # division
                3.0 * l * cell_state[5], # l from Ycentric to Y0
                3.0 * l * cell_state[6], # l from Y0 to Y1
                3.0 * l * cell_state[7], # l from Y1 to Ycentric
            ])
#             print('\n while start')
#             print('reac_props: ', reac_props)

            # draw exponential random time for next reaction
            total_prop = np.sum(reac_props)
            dt = np.random.exponential(1.0/total_prop)

#             print('total_prop: ', total_prop)
#             print('dt: ', dt)

            # save cell state results for relevant time points
            for ind in np.where((time_values >= time_current)
                                & (time_values < time_current + dt))[0]:
                res[i, :, ind] = copy.deepcopy(cell_state).reshape((1,8))

#             print('res: ', res)

            # draw which reaction takes place
            reac_probs = reac_props/np.sum(reac_props)
            reac_rand = np.random.choice(reac_props.shape[0], p=reac_probs)

#             print('reac_probs: ', reac_probs)
#             print('reac_rand: ', reac_rand)

            # update cell state according to selected reaction
            # differentiation d4 channel
            if reac_rand==0:
                cell_state[0] += -1.0
                cell_state[1] += +1.0
            elif reac_rand==1:
                cell_state[1] += -1.0
                cell_state[2] += +1.0
            elif reac_rand==2:
                cell_state[2] += -1.0
                cell_state[3] += +1.0
            elif reac_rand==3:
                cell_state[3] += -1.0
                cell_state[5] += +1.0

            # differentiation d2 channel
            elif reac_rand==4:
                cell_state[0] += -1.0
                cell_state[4] += +1.0
            elif reac_rand==5:
                cell_state[4] += -1.0
                cell_state[5] += +1.0

            # division
            elif reac_rand==6:
                cell_state[5] += -1.0
                cell_state[6] += +1.0
            elif reac_rand==7:
                cell_state[6] += -1.0
                cell_state[7] += +1.0
            elif reac_rand==8:
                cell_state[7] += -1.0
                cell_state[5] += +2.0
            else:
                print('error')

            # update current time by delta t
            time_current += dt

#             print('cell_state: ', cell_state)
#             print('time_current: ', time_current)
#             print('\n')

    # sum hidden states to get observable layer
    res_obs = np.zeros((num_iter, 2, time_values.shape[0]))
    res_obs[:, 0, :] = np.sum(res[:, (0,1,2,3,4), :], axis=1)
    res_obs[:, 1, :] = np.sum(res[:, (5,6,7), :], axis=1)
    return res_obs
