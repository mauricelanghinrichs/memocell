
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import pytest
import memocell as me
import numpy as np

class TestNetworkClass(object):
    def test_net_diff_markov(self):
        t = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network('net_diff_exp')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_diff_exp')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_1'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric', 'Z_1__centric'])
        assert(net.net_modules == [{'module': 'module_0',
                                      'start-end': ('X_t', 'Y_t'),
                                      'start-end_ident': ('Z_0', 'Z_1'),
                                      'sym_rate': 'k_xy',
                                      'sym_rate_ident': 'theta_0',
                                      'type': 'S -> E',
                                      'module_steps': 1}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                      'Z_1',
                      {'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'k_xy',
                       'module_identifier': 'module_0',
                       'module_type': 'S -> E',
                       'module_steps': 1})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                     'Z_1__centric',
                     {'edge_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                      'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                      'module_start_end_identifier': ('Z_0', 'Z_1'),
                      'module_start_end': ('X_t', 'Y_t'),
                      'edge_rate_symbol_identifier': '1.0 * theta_0',
                      'edge_rate_symbol': '1.0 * k_xy',
                      'module_rate_symbol_identifier': 'theta_0',
                      'module_rate_symbol': 'k_xy',
                      'module_identifier': 'module_0',
                      'edge_type': 'S -> E',
                      'module_type': 'S -> E',
                      'module_steps': 1})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_1': 'Y_t'})
        assert(net.net_rates_identifier == {'theta_0': 'k_xy'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_diff_erl3(self):
        t = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 3}
        ]
        net = me.Network('net_diff_erl3')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_diff_erl3')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_1'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric', 'Z_0__module_0__0', 'Z_0__module_0__1', 'Z_1__centric'])
        assert(net.net_modules == [{'module': 'module_0',
                          'start-end': ('X_t', 'Y_t'),
                          'start-end_ident': ('Z_0', 'Z_1'),
                          'sym_rate': 'k_xy',
                          'sym_rate_ident': 'theta_0',
                          'type': 'S -> E',
                          'module_steps': 3}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 3, 'Z_1': 1})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                      'Z_1',
                      {'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'k_xy',
                       'module_identifier': 'module_0',
                       'module_type': 'S -> E',
                       'module_steps': 3})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                      'Z_0__module_0__0',
                      {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * k_xy',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'k_xy',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> E',
                       'module_steps': 3}),
                     ('Z_0__module_0__0',
                      'Z_0__module_0__1',
                      {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_0__module_0__1'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * k_xy',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'k_xy',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> E',
                       'module_steps': 3}),
                     ('Z_0__module_0__1',
                      'Z_1__centric',
                      {'edge_start_end_identifier': ('Z_0__module_0__1', 'Z_1__centric'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * k_xy',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'k_xy',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> E',
                       'module_steps': 3})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_1': 'Y_t'})
        assert(net.net_rates_identifier == {'theta_0': 'k_xy'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_sym_div_erl3(self):
        t = [
        {'start': 'X_t', 'end': 'X_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
        ]
        net = me.Network('net_sym_div_erl3')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_sym_div_erl3')
        assert(list(net.net_main.nodes()) == ['Z_0'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric', 'Z_0__module_0__0', 'Z_0__module_0__1'])
        assert(net.net_modules == [{'module': 'module_0',
                      'start-end': ('X_t', 'X_t'),
                      'start-end_ident': ('Z_0', 'Z_0'),
                      'sym_rate': 'l',
                      'sym_rate_ident': 'theta_0',
                      'type': 'S -> S + S',
                      'module_steps': 3}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 3})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                      'Z_0',
                      {'module_start_end_identifier': ('Z_0', 'Z_0'),
                       'module_start_end': ('X_t', 'X_t'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'module_type': 'S -> S + S',
                       'module_steps': 3})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                      'Z_0__module_0__0',
                      {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_0__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_0'),
                       'module_start_end': ('X_t', 'X_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * l',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> S + S',
                       'module_steps': 3}),
                     ('Z_0__module_0__0',
                      'Z_0__module_0__1',
                      {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_0__module_0__1'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_0__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_0'),
                       'module_start_end': ('X_t', 'X_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * l',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> S + S',
                       'module_steps': 3}),
                     ('Z_0__module_0__1',
                      'Z_0__centric',
                      {'edge_start_end_identifier': ('Z_0__module_0__1', 'Z_0__centric'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_0__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_0'),
                       'module_start_end': ('X_t', 'X_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * l',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E + E',
                       'module_type': 'S -> S + S',
                       'module_steps': 3})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t'})
        assert(net.net_rates_identifier == {'theta_0': 'l'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_asym_div_erl3(self):
        t = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + E', 'reaction_steps': 3}
        ]
        net = me.Network('net_asym_div_erl3')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_asym_div_erl3')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_1'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric', 'Z_0__module_0__0', 'Z_0__module_0__1', 'Z_1__centric'])
        assert(net.net_modules == [{'module': 'module_0',
                      'start-end': ('X_t', 'Y_t'),
                      'start-end_ident': ('Z_0', 'Z_1'),
                      'sym_rate': 'l',
                      'sym_rate_ident': 'theta_0',
                      'type': 'S -> S + E',
                      'module_steps': 3}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 3, 'Z_1': 1})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                      'Z_1',
                      {'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'module_type': 'S -> S + E',
                       'module_steps': 3})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                      'Z_0__module_0__0',
                      {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * l',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> S + E',
                       'module_steps': 3}),
                     ('Z_0__module_0__0',
                      'Z_0__module_0__1',
                      {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_0__module_0__1'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * l',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> S + E',
                       'module_steps': 3}),
                     ('Z_0__module_0__1',
                      'Z_1__centric',
                      {'edge_start_end_identifier': ('Z_0__module_0__1', 'Z_1__centric'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * l',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'l',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E1 + E2',
                       'module_type': 'S -> S + E',
                       'module_steps': 3})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_1': 'Y_t'})
        assert(net.net_rates_identifier == {'theta_0': 'l'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_diff_div_erl3(self):
        t = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'dl', 'type': 'S -> E + E', 'reaction_steps': 3}
        ]
        net = me.Network('net_diff_div_erl3')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_diff_div_erl3')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_1'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric', 'Z_0__module_0__0', 'Z_0__module_0__1', 'Z_1__centric'])
        assert(net.net_modules == [{'module': 'module_0',
                      'start-end': ('X_t', 'Y_t'),
                      'start-end_ident': ('Z_0', 'Z_1'),
                      'sym_rate': 'dl',
                      'sym_rate_ident': 'theta_0',
                      'type': 'S -> E + E',
                      'module_steps': 3}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 3, 'Z_1': 1})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                      'Z_1',
                      {'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dl',
                       'module_identifier': 'module_0',
                       'module_type': 'S -> E + E',
                       'module_steps': 3})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                      'Z_0__module_0__0',
                      {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * dl',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dl',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> E + E',
                       'module_steps': 3}),
                     ('Z_0__module_0__0',
                      'Z_0__module_0__1',
                      {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_0__module_0__1'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * dl',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dl',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S -> E + E',
                       'module_steps': 3}),
                     ('Z_0__module_0__1',
                      'Z_1__centric',
                      {'edge_start_end_identifier': ('Z_0__module_0__1', 'Z_1__centric'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_1'),
                       'module_start_end': ('X_t', 'Y_t'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * dl',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dl',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E + E',
                       'module_type': 'S -> E + E',
                       'module_steps': 3})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_1': 'Y_t'})
        assert(net.net_rates_identifier == {'theta_0': 'dl'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_influx_markov(self):
        t = [
        {'start': 'env', 'end': 'Y_t', 'rate_symbol': 'din', 'type': '-> E', 'reaction_steps': 1}
        ]
        net = me.Network('net_influx_exp')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_influx_exp')
        assert(list(net.net_main.nodes()) == ['Z_env', 'Z_0'])
        assert(list(net.net_hidden.nodes()) == ['Z_env__centric', 'Z_0__centric'])
        assert(net.net_modules == [{'module': 'module_0',
                      'start-end': ('env', 'Y_t'),
                      'start-end_ident': ('Z_env', 'Z_0'),
                      'sym_rate': 'din',
                      'sym_rate_ident': 'theta_0',
                      'type': '-> E',
                      'module_steps': 1}])
        assert(net.net_main_node_numbers == {'Z_env': 1, 'Z_0': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 1, 'Z_0': 1})
        assert(list(net.net_main.edges(data=True)) == [('Z_env',
                      'Z_0',
                      {'module_start_end_identifier': ('Z_env', 'Z_0'),
                       'module_start_end': ('env', 'Y_t'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'din',
                       'module_identifier': 'module_0',
                       'module_type': '-> E',
                       'module_steps': 1})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_env__centric',
                      'Z_0__centric',
                      {'edge_start_end_identifier': ('Z_env__centric', 'Z_0__centric'),
                       'edge_centric_start_end_identifier': ('Z_env__centric', 'Z_0__centric'),
                       'module_start_end_identifier': ('Z_env', 'Z_0'),
                       'module_start_end': ('env', 'Y_t'),
                       'edge_rate_symbol_identifier': '1.0 * theta_0',
                       'edge_rate_symbol': '1.0 * din',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'din',
                       'module_identifier': 'module_0',
                       'edge_type': '-> E',
                       'module_type': '-> E',
                       'module_steps': 1})])
        assert(net.net_nodes_identifier == {'Z_0': 'Y_t', 'Z_env': 'env'})
        assert(net.net_rates_identifier == {'theta_0': 'din'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_efflux_erl3(self):
        t = [
        {'start': 'X_t', 'end': 'env', 'rate_symbol': 'dout', 'type': 'S ->', 'reaction_steps': 3}
        ]
        net = me.Network('net_efflux_erl3')
        net.structure(t)

        # check all created attributes for the network
        assert(net.net_name == 'net_efflux_erl3')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_env'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric', 'Z_0__module_0__0', 'Z_0__module_0__1', 'Z_env__centric'])
        assert(net.net_modules == [{'module': 'module_0',
                      'start-end': ('X_t', 'env'),
                      'start-end_ident': ('Z_0', 'Z_env'),
                      'sym_rate': 'dout',
                      'sym_rate_ident': 'theta_0',
                      'type': 'S ->',
                      'module_steps': 3}])
        assert(net.net_main_node_numbers == {'Z_env': 1, 'Z_0': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 1, 'Z_0': 3})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                      'Z_env',
                      {'module_start_end_identifier': ('Z_0', 'Z_env'),
                       'module_start_end': ('X_t', 'env'),
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dout',
                       'module_identifier': 'module_0',
                       'module_type': 'S ->',
                       'module_steps': 3})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                      'Z_0__module_0__0',
                      {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_env__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_env'),
                       'module_start_end': ('X_t', 'env'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * dout',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dout',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S ->',
                       'module_steps': 3}),
                     ('Z_0__module_0__0',
                      'Z_0__module_0__1',
                      {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_0__module_0__1'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_env__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_env'),
                       'module_start_end': ('X_t', 'env'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * dout',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dout',
                       'module_identifier': 'module_0',
                       'edge_type': 'S -> E',
                       'module_type': 'S ->',
                       'module_steps': 3}),
                     ('Z_0__module_0__1',
                      'Z_env__centric',
                      {'edge_start_end_identifier': ('Z_0__module_0__1', 'Z_env__centric'),
                       'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_env__centric'),
                       'module_start_end_identifier': ('Z_0', 'Z_env'),
                       'module_start_end': ('X_t', 'env'),
                       'edge_rate_symbol_identifier': '3.0 * theta_0',
                       'edge_rate_symbol': '3.0 * dout',
                       'module_rate_symbol_identifier': 'theta_0',
                       'module_rate_symbol': 'dout',
                       'module_identifier': 'module_0',
                       'edge_type': 'S ->',
                       'module_type': 'S ->',
                       'module_steps': 3})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_env': 'env'})
        assert(net.net_rates_identifier == {'theta_0': 'dout'})
        assert(net.net_theta_symbolic == ['theta_0'])

    def test_net_min_2_4(self):
        net = me.Network('net_min_2_4')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t',
             'rate_symbol': 'd',
             'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t',
             'rate_symbol': 'l',
             'type': 'S -> S + S', 'reaction_steps': 4}
            ])

        # check all created attributes for the network
        assert(net.net_name == 'net_min_2_4')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_1'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric',
                                             'Z_0__module_0__0',
                                             'Z_1__centric',
                                             'Z_1__module_1__0',
                                             'Z_1__module_1__1',
                                             'Z_1__module_1__2'])
        assert(net.net_modules == [{'module': 'module_0',
                              'start-end': ('X_t', 'Y_t'),
                              'start-end_ident': ('Z_0', 'Z_1'),
                              'sym_rate': 'd',
                              'sym_rate_ident': 'theta_0',
                              'type': 'S -> E',
                              'module_steps': 2},
                             {'module': 'module_1',
                              'start-end': ('Y_t', 'Y_t'),
                              'start-end_ident': ('Z_1', 'Z_1'),
                              'sym_rate': 'l',
                              'sym_rate_ident': 'theta_1',
                              'type': 'S -> S + S',
                              'module_steps': 4}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 2, 'Z_1': 4})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                              'Z_1',
                              {'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'module_rate_symbol_identifier': 'theta_0',
                               'module_rate_symbol': 'd',
                               'module_identifier': 'module_0',
                               'module_type': 'S -> E',
                               'module_steps': 2}),
                             ('Z_1',
                              'Z_1',
                              {'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_1',
                               'module_type': 'S -> S + S',
                               'module_steps': 4})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                              'Z_0__module_0__0',
                              {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '2.0 * theta_0',
                               'edge_rate_symbol': '2.0 * d',
                               'module_rate_symbol_identifier': 'theta_0',
                               'module_rate_symbol': 'd',
                               'module_identifier': 'module_0',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 2}),
                             ('Z_0__module_0__0',
                              'Z_1__centric',
                              {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_1__centric'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '2.0 * theta_0',
                               'edge_rate_symbol': '2.0 * d',
                               'module_rate_symbol_identifier': 'theta_0',
                               'module_rate_symbol': 'd',
                               'module_identifier': 'module_0',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 2}),
                             ('Z_1__centric',
                              'Z_1__module_1__0',
                              {'edge_start_end_identifier': ('Z_1__centric', 'Z_1__module_1__0'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * l',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_1',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> S + S',
                               'module_steps': 4}),
                             ('Z_1__module_1__0',
                              'Z_1__module_1__1',
                              {'edge_start_end_identifier': ('Z_1__module_1__0', 'Z_1__module_1__1'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * l',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_1',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> S + S',
                               'module_steps': 4}),
                             ('Z_1__module_1__1',
                              'Z_1__module_1__2',
                              {'edge_start_end_identifier': ('Z_1__module_1__1', 'Z_1__module_1__2'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * l',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_1',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> S + S',
                               'module_steps': 4}),
                             ('Z_1__module_1__2',
                              'Z_1__centric',
                              {'edge_start_end_identifier': ('Z_1__module_1__2', 'Z_1__centric'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * l',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_1',
                               'edge_type': 'S -> E + E',
                               'module_type': 'S -> S + S',
                               'module_steps': 4})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_1': 'Y_t'})
        assert(net.net_rates_identifier == {'theta_0': 'd', 'theta_1': 'l'})
        assert(net.net_theta_symbolic == ['theta_0', 'theta_1'])

    def test_net_par2(self):
        net = me.Network('net_par2')
        net.structure([
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd4', 'type': 'S -> E', 'reaction_steps': 4},
            {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'd2', 'type': 'S -> E', 'reaction_steps': 2},
            {'start': 'Y_t', 'end': 'Y_t', 'rate_symbol': 'l', 'type': 'S -> S + S', 'reaction_steps': 3}
            ])

        # check all created attributes for the network
        assert(net.net_name == 'net_par2')
        assert(list(net.net_main.nodes()) == ['Z_0', 'Z_1'])
        assert(list(net.net_hidden.nodes()) == ['Z_0__centric',
                                             'Z_0__module_0__0',
                                             'Z_0__module_0__1',
                                             'Z_0__module_0__2',
                                             'Z_1__centric',
                                             'Z_0__module_1__0',
                                             'Z_1__module_2__0',
                                             'Z_1__module_2__1'])
        assert(net.net_modules == [{'module': 'module_0',
                              'start-end': ('X_t', 'Y_t'),
                              'start-end_ident': ('Z_0', 'Z_1'),
                              'sym_rate': 'd4',
                              'sym_rate_ident': 'theta_1',
                              'type': 'S -> E',
                              'module_steps': 4},
                             {'module': 'module_1',
                              'start-end': ('X_t', 'Y_t'),
                              'start-end_ident': ('Z_0', 'Z_1'),
                              'sym_rate': 'd2',
                              'sym_rate_ident': 'theta_0',
                              'type': 'S -> E',
                              'module_steps': 2},
                             {'module': 'module_2',
                              'start-end': ('Y_t', 'Y_t'),
                              'start-end_ident': ('Z_1', 'Z_1'),
                              'sym_rate': 'l',
                              'sym_rate_ident': 'theta_2',
                              'type': 'S -> S + S',
                              'module_steps': 3}])
        assert(net.net_main_node_numbers == {'Z_env': 0, 'Z_0': 1, 'Z_1': 1})
        assert(net.net_hidden_node_numbers == {'Z_env': 0, 'Z_0': 5, 'Z_1': 3})
        assert(list(net.net_main.edges(data=True)) == [('Z_0',
                              'Z_1',
                              {'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'd4',
                               'module_identifier': 'module_0',
                               'module_type': 'S -> E',
                               'module_steps': 4}),
                             ('Z_0',
                              'Z_1',
                              {'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'module_rate_symbol_identifier': 'theta_0',
                               'module_rate_symbol': 'd2',
                               'module_identifier': 'module_1',
                               'module_type': 'S -> E',
                               'module_steps': 2}),
                             ('Z_1',
                              'Z_1',
                              {'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'module_rate_symbol_identifier': 'theta_2',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_2',
                               'module_type': 'S -> S + S',
                               'module_steps': 3})])
        assert(list(net.net_hidden.edges(data=True)) == [('Z_0__centric',
                              'Z_0__module_0__0',
                              {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_0__0'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * d4',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'd4',
                               'module_identifier': 'module_0',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 4}),
                             ('Z_0__centric',
                              'Z_0__module_1__0',
                              {'edge_start_end_identifier': ('Z_0__centric', 'Z_0__module_1__0'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '2.0 * theta_0',
                               'edge_rate_symbol': '2.0 * d2',
                               'module_rate_symbol_identifier': 'theta_0',
                               'module_rate_symbol': 'd2',
                               'module_identifier': 'module_1',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 2}),
                             ('Z_0__module_0__0',
                              'Z_0__module_0__1',
                              {'edge_start_end_identifier': ('Z_0__module_0__0', 'Z_0__module_0__1'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * d4',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'd4',
                               'module_identifier': 'module_0',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 4}),
                             ('Z_0__module_0__1',
                              'Z_0__module_0__2',
                              {'edge_start_end_identifier': ('Z_0__module_0__1', 'Z_0__module_0__2'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * d4',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'd4',
                               'module_identifier': 'module_0',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 4}),
                             ('Z_0__module_0__2',
                              'Z_1__centric',
                              {'edge_start_end_identifier': ('Z_0__module_0__2', 'Z_1__centric'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '4.0 * theta_1',
                               'edge_rate_symbol': '4.0 * d4',
                               'module_rate_symbol_identifier': 'theta_1',
                               'module_rate_symbol': 'd4',
                               'module_identifier': 'module_0',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 4}),
                             ('Z_1__centric',
                              'Z_1__module_2__0',
                              {'edge_start_end_identifier': ('Z_1__centric', 'Z_1__module_2__0'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '3.0 * theta_2',
                               'edge_rate_symbol': '3.0 * l',
                               'module_rate_symbol_identifier': 'theta_2',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_2',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> S + S',
                               'module_steps': 3}),
                             ('Z_0__module_1__0',
                              'Z_1__centric',
                              {'edge_start_end_identifier': ('Z_0__module_1__0', 'Z_1__centric'),
                               'edge_centric_start_end_identifier': ('Z_0__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_0', 'Z_1'),
                               'module_start_end': ('X_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '2.0 * theta_0',
                               'edge_rate_symbol': '2.0 * d2',
                               'module_rate_symbol_identifier': 'theta_0',
                               'module_rate_symbol': 'd2',
                               'module_identifier': 'module_1',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> E',
                               'module_steps': 2}),
                             ('Z_1__module_2__0',
                              'Z_1__module_2__1',
                              {'edge_start_end_identifier': ('Z_1__module_2__0', 'Z_1__module_2__1'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '3.0 * theta_2',
                               'edge_rate_symbol': '3.0 * l',
                               'module_rate_symbol_identifier': 'theta_2',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_2',
                               'edge_type': 'S -> E',
                               'module_type': 'S -> S + S',
                               'module_steps': 3}),
                             ('Z_1__module_2__1',
                              'Z_1__centric',
                              {'edge_start_end_identifier': ('Z_1__module_2__1', 'Z_1__centric'),
                               'edge_centric_start_end_identifier': ('Z_1__centric', 'Z_1__centric'),
                               'module_start_end_identifier': ('Z_1', 'Z_1'),
                               'module_start_end': ('Y_t', 'Y_t'),
                               'edge_rate_symbol_identifier': '3.0 * theta_2',
                               'edge_rate_symbol': '3.0 * l',
                               'module_rate_symbol_identifier': 'theta_2',
                               'module_rate_symbol': 'l',
                               'module_identifier': 'module_2',
                               'edge_type': 'S -> E + E',
                               'module_type': 'S -> S + S',
                               'module_steps': 3})])
        assert(net.net_nodes_identifier == {'Z_0': 'X_t', 'Z_1': 'Y_t'})
        assert(net.net_rates_identifier == {'theta_0': 'd2', 'theta_1': 'd4', 'theta_2': 'l'})
        assert(net.net_theta_symbolic == ['theta_0', 'theta_1', 'theta_2'])
