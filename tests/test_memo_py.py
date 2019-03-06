# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memo_py.py"
# or after pip installation $py.test --pyargs memo_py$

import memo_py as me

# TODO: adapt expected output in those tests
class TestNetworkClass(object):
    def test_network_structure_input_one_reaction_nodes(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_main.nodes()) == ['X_t', 'Y_t'])

    def test_network_structure_input_one_reaction_edges(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_main.edges()) == [('X_t', 'Y_t')])

    def test_network_structure_input_one_reaction_edges_with_data(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_main.edges(data=True)) == [('X_t', 'Y_t', {'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1})])

    def test_network_structure_input_two_reactions_node_identifier(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
        {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(net.create_node_identifiers(net.net_main) == {'X_t': 'Z_0', 'Y_t': 'Z_1', 'env': 'Z_env'})

    def test_network_structure_input_two_reactions_rate_identifier(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
        {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(net.create_rate_identifiers(net.net_main) == {'k_xy': 'theta_0'})

    def test_network_structure_input_two_reactions_modules(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
        {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(net.net_modules == [
            {'module': 'module_0', 'start-end': ('X_t', 'Y_t'), 'start-end_ident': ('Z_0', 'Z_1'), 'sym_rate': 'k_xy', 'sym_rate_ident': 'theta_0', 'type': 'S -> E', 'module_steps': 1},
            {'module': 'module_1', 'start-end': ('Y_t', 'env'), 'start-end_ident': ('Z_1', 'Z_env'), 'sym_rate': 'k_xy', 'sym_rate_ident': 'theta_0', 'type': 'S ->', 'module_steps': 1}
            ])

    def test_hidden_network_structure_input_one_reaction_nodes(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_hidden.nodes()) == ['X_t', 'Y_t'])

    def test_hidden_network_structure_input_one_reaction_edges(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_hidden.edges()) == [('X_t', 'Y_t')])

    def test_hidden_network_structure_input_one_reaction_edges_with_data(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1}
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_hidden.edges(data=True)) == [('X_t', 'Y_t', {'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1})])

    def test_hidden_network_structure_input_two_reactions_nodes(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
        {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_hidden.nodes()) == ['X_t', 'Y_t'])

    def test_hidden_network_structure_input_two_reactions_edges(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
        {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_hidden.edges()) == [('X_t', 'Y_t')])

    def test_hidden_network_structure_input_two_reactions_edges_with_data(self):
        network_structure_input = [
        {'start': 'X_t', 'end': 'Y_t', 'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1},
        {'start': 'Y_t', 'end': 'env', 'rate_symbol': 'k_xy', 'type': 'S ->', 'reaction_steps': 1},
        ]
        net = me.Network()
        net.structure(network_structure_input)
        assert(list(net.net_hidden.edges(data=True)) == [('X_t', 'Y_t', {'rate_symbol': 'k_xy', 'type': 'S -> E', 'reaction_steps': 1})])


class TestTwo(object):
    def test_one(self):
        assert(1 == 1)
