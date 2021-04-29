
"""
The network module contains the Network class for the symbolic handling of
network topologies and waiting time structures, to be used for downstream
simulation and estimation/selection methods.
"""

import networkx as nx
from copy import deepcopy
import warnings

class Network(object):
    """Class for defining a symbolic network with main and hidden layer.

    Main method is `structure` which requires a specific `net_structure` input.
    For a typical use case, the `structure` method is the only method to call;
    `structure` is a wrapper method
    for the other class methods, see there for more documentation.

    Parameters
    ----------
    net_name : str
        A name for the network object.

    Returns
    -------
    net : memocell.network.Network
        Initialised memocell network object. Typically, continue with the
        `net.structure` method to further specify the network.

    Examples
    --------
    >>> import memocell as me
    >>> import numpy as np
    >>> # network with symmetric division reaction
    >>> # of a 5-step Erlang waiting time and mean
    >>> # division time 1/l
    >>> t = [{'start': 'X_t', 'end': 'X_t',
    >>>  'rate_symbol': 'l',
    >>>  'type': 'S -> S + S', 'reaction_steps': 5}]
    >>> net = me.Network('net_div_erl5')
    >>> net.structure(t)
    """

    def __init__(self, net_name):
        # set a name for a Network class instance
        self._validate_net_name_input(net_name)
        self.net_name = net_name

        # initialise the main network as networkx MultiDiGraph
        self.net_main = nx.MultiDiGraph()

        # initialise the hidden network as networkx MultiDiGraph
        self.net_hidden = nx.MultiDiGraph()

        # initialise the network modules as list
        self.net_modules = list()

        # initialise identifier dict for nodes and rates
        self.net_nodes_identifier = dict()
        self.net_rates_identifier = dict()

        # initialise a list to store the identifiers of the symbolic rates ('theta_<int>')
        self.net_theta_symbolic = list()

        # initialise lists to store nodes (and pairs of nodes) in an ordered, linear structure
        self.net_main_node_order = list()
        self.net_hidden_node_order = list()

        # initialise dicts to store number of hidden nodes for each main node
        # (as dict, using node identifiers; including env node)
        self.net_main_node_numbers = dict()
        self.net_hidden_node_numbers = dict()


    # validate user input and define the network structure
    def structure(self, net_structure):
        """Main method of the network class to define the network structure
        with main and hidden layer. Updates a network object in place,
        that can then be passed on to simulation or estimation methods.

        `Note`: The basic reaction module specifies an Erlang-type waiting time (`reaction_steps`
        as shape parameter and `1/rate_symbol` as mean waiting time).
        The standard Markov model is obtained when `reaction_steps` is
        set to `1` for all reactions of the network.
        More advanced phase-type waiting times can be generated
        by specifying two or more Erlang-type channels in parallel (define two or more
        reactions of the same `type` between the same `start` and `end` main nodes).
        Alternatively/in addition, one can use simulation variables
        (in `simulate` and `estimate` methods) to sum up a set of main nodes to
        a quantity of interest; this allows to specify arbitrary transition schemes.

        `Note`: `net.structure` fills many attributes of a `net` instance that can be
        accessed afterwards. A lot of information is also contained in the edges of the
        networkx MultiDiGraph objects (main and hidden layer), accessible via
        `net.net_main.edges(data=True)` and `net.net_hidden.edges(data=True)`.

        `Note`: The reaction types below are interpreted in the context of cell
        differentiation pathways; of course, these reactions (and MemoCell as a whole)
        can be equally applied to molecular reactions etc.

        Parameters
        ----------
        net_structure : list of dict
            Network structure specifying the topology of interlinked reaction modules of
            different `types` and their waiting times.
            Each reaction is specified as a `dict` in the `list`, connecting main nodes
            with `start` and `end`, and identified with a symbolic `rate_symbol`
            and the `reaction_steps` on the hidden Markov layer
            (where `reaction_steps` is the shape parameter and `1/rate` is the
            mean waiting time for isolated Erlang-type reactions).
            Available reaction types are

            - `'S -> E'` (cell differentiation),

            - `'S -> S + S'` (symmetric self-renewing division),

            - `'S -> S + E'` (asymmetric division),

            - `'S -> E + E'` (symmetric differentiating division),

            - `'S ->'` (efflux or cell death) and

            - `'-> E'` (influx or birth),

            where `S` corresponds to the `start` node and `E` corresponds to the
            `end` node. If a reaction does not have `S` or `E` (influx and efflux reactions),
            use `'env'` (the general environment node) for `start` and `end` node, respectively.
            If multiple reactions shall share the same reaction rate,
            use the same `rate_symbol`.

        Returns
        -------
        None

        Examples
        --------
        >>> import memocell as me
        >>> import numpy as np
        >>> # network with symmetric division reaction
        >>> # of a 5-step Erlang waiting time and mean
        >>> # division time 1/l
        >>> t = [{'start': 'X_t', 'end': 'X_t',
        >>>  'rate_symbol': 'l',
        >>>  'type': 'S -> S + S', 'reaction_steps': 5}]
        >>> net = me.Network('net_div_erl5')
        >>> net.structure(t)
        """

        # validate the user input
        self._validate_net_structure(net_structure)

        # create the network modules
        self.net_modules, self.net_nodes_identifier, self.net_rates_identifier = self.create_net_modules_and_identifiers(net_structure)

        # create a list ordering the symbolic rates of a network
        self.net_theta_symbolic = self.create_theta_order(self.net_modules)

        # define the main network structure
        self.net_main = self.structure_net_main(nx.MultiDiGraph(), self.net_modules)

        # define the hidden network structure
        self.net_hidden = self.structure_net_hidden(nx.MultiDiGraph(), self.net_modules)

        # create a list ordering the nodes and pairs of nodes of the main and hidden networks
        self.net_main_node_order = self.create_node_order(self.net_main.nodes())
        self.net_hidden_node_order = self.create_node_order(self.net_hidden.nodes())

        # count number of main and hidden nodes for each node identifier, including env
        self.net_main_node_numbers = self.count_net_main_nodes(self.net_main.nodes())
        self.net_hidden_node_numbers = self.count_net_hidden_nodes(self.net_hidden.nodes())


    def create_net_modules_and_identifiers(self, net_structure):
        """Creates network modules and identifiers for nodes and rates.

        `Note`: After defining a network via the `net.structure` method,
        one can simply access the output at `net.net_modules`,
        `net.net_nodes_identifier` and `net.net_rates_identifier`
        for the network instance `net`.

        Parameters
        ----------
        net_structure : list of dict
            Network structure specifying the topology and waiting times on the hidden
            Markov layer. See `structure` method for more information.

        Returns
        -------
        net_modules : list of dict
            Reaction modules of a network with identifier.
        net_nodes_identifier : dict
            `Z`-Identifier for the main nodes.
        net_rates_identifier : dict
            `theta`-Identifier for the symbolic rate parameters.

        Examples
        --------
        >>> import memocell as me
        >>> # network with symmetric division reaction
        >>> # of a 5-step Erlang waiting time and mean
        >>> # division time 1/l
        >>> t = [{'start': 'X_t', 'end': 'X_t',
        >>>  'rate_symbol': 'l',
        >>>  'type': 'S -> S + S', 'reaction_steps': 5}]
        >>> net = me.Network('net_div_erl5')
        >>> net.create_net_modules_and_identifiers(t)
        ([{'module': 'module_0',
           'start-end': ('X_t', 'X_t'),
           'start-end_ident': ('Z_0', 'Z_0'),
           'sym_rate': 'l',
           'sym_rate_ident': 'theta_0',
           'type': 'S -> S + S',
           'module_steps': 5}],
         {'Z_0': 'X_t'},
         {'theta_0': 'l'})
         """

        # create neutral identifiers for rate_symbol's and nodes
        ident_rates = self.create_rate_identifiers(net_structure)
        ident_nodes = self.create_node_identifiers(net_structure)

        # the above dictionaries are inverted here
        ident_rates_inv = {rate: rate_id for rate_id, rate in ident_rates.items()}
        ident_nodes_inv = {node: node_id for node_id, node in ident_nodes.items()}

        # iterate over edges to define a module
        # a "module" is the collection of edges in a hidden network between two main/center nodes
        net_modules = list()
        for i, edge in enumerate(net_structure):
            module = {
            'module': f'module_{i}',
            'start-end': (edge['start'], edge['end']),
            'start-end_ident': (ident_nodes_inv[edge['start']], ident_nodes_inv[edge['end']]),
            'sym_rate': edge['rate_symbol'],
            'sym_rate_ident': ident_rates_inv[edge['rate_symbol']],
            'type': edge['type'],
            'module_steps': edge['reaction_steps']
            }

            net_modules.append(module)

        return net_modules, ident_nodes, ident_rates


    def structure_net_main(self, net, net_modules):
        """Updates a networkx MultiDiGraph for the main (observable) layer
        of a memocell network.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_main` for the network instance `net`.

        Parameters
        ----------
        net : networkx.classes.multidigraph.MultiDiGraph
            An (empty) initialised networkx MultiDiGraph object.
        net_modules : list of dict
            Reaction modules of a network with identifier. As generated by the
            `create_net_modules_and_identifiers` method and available at
            `net.net_modules`.

        Returns
        -------
        net_main : networkx.classes.multidigraph.MultiDiGraph
            The main (observable) layer of a memocell network.
            Typically available at `net.net_main`.
        """

        # copy net to create net_main
        net_main = net.copy()

        # assign modules as edges to net_main
        for module in net_modules:
            net_main.add_edge(module['start-end_ident'][0],
                        module['start-end_ident'][1],
                        module_start_end_identifier = module['start-end_ident'],
                        module_start_end = module['start-end'],
                        module_rate_symbol_identifier = module['sym_rate_ident'],
                        module_rate_symbol = module['sym_rate'],
                        # module_rate_numeric = None, # NOTE: add this really here?
                        module_identifier = module['module'],
                        module_type = module['type'],
                        module_steps = module['module_steps']
                        )
        return net_main

    def structure_net_hidden(self, net, net_modules):
        """Updates a networkx MultiDiGraph for the hidden (Markov) layer
        of a memocell network.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_hidden` for the network instance `net`.

        Parameters
        ----------
        net : networkx.classes.multidigraph.MultiDiGraph
            An (empty) initialised networkx MultiDiGraph object.
        net_modules : list of dict
            Reaction modules of a network with identifier. As generated by the
            `create_net_modules_and_identifiers` method and available at
            `net.net_modules`.

        Returns
        -------
        net_hidden : networkx.classes.multidigraph.MultiDiGraph
            The hidden (Markov) layer of a memocell network.
            Typically available at `net.net_hidden`.
        """

        # copy net to create net_hidden
        net_hidden = net.copy()

        # TODO: implement
        # assign all reactions of a module as edges to net_hidden
        for module in net_modules:
            start_main_node = module['start-end_ident'][0]
            end_main_node = module['start-end_ident'][1]
            module_ident = module['module']
            module_steps = module['module_steps']
            module_type = module['type']

            # obtain a list of the sequential nodes part of a module
            module_nodes = self.create_module_nodes(start_main_node, end_main_node, module_ident, module_steps)

            # obtain a list of the sequential reaction types part of a module
            reaction_types = self.create_module_reaction_types(module_type, module_steps)

            ### assign the edges to net_hidden
            # iterate over all pairs of consecutive nodes in module_nodes
            # associated with the reaction type of the edge linking the two nodes
            for first_node, second_node, reaction_type in zip(module_nodes, module_nodes[1:], reaction_types):

                net_hidden.add_edge(first_node,
                            second_node,
                            edge_start_end_identifier = (first_node, second_node),
                            edge_centric_start_end_identifier = (module_nodes[0], module_nodes[-1]),
                            module_start_end_identifier = module['start-end_ident'],
                            module_start_end = module['start-end'],
                            edge_rate_symbol_identifier = f'{float(module_steps)} * ' + module['sym_rate_ident'],
                            edge_rate_symbol = f'{float(module_steps)} * ' + module['sym_rate'],
                            module_rate_symbol_identifier = module['sym_rate_ident'],
                            module_rate_symbol = module['sym_rate'],
                            # edge_rate_numeric = None, # NOTE: add this really here?
                            # module_rate_numeric = None, # NOTE: add this really here?
                            module_identifier = module['module'],
                            edge_type = reaction_type,
                            module_type = module['type'],
                            module_steps = module['module_steps']
                            )

        # NOTE: for S->S+E module, a S->E1+E2 reaction type is required for the last substep (add, see written notes!)

        return net_hidden

    @staticmethod
    def create_theta_order(net_modules):
        """Creates an order of the symbolic rate parameters in their `theta`-identifier form.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_theta_symbolic`.

        Examples
        --------
        >>> # after defining some network by
        >>> # net.structure(...)
        >>> net.net_theta_symbolic
        ['theta_0', 'theta_1']
        """

        # read out all symbolic rates of network modules (their theta identifier),
        # return sorted list without duplicates
        return sorted(set([module['sym_rate_ident'] for module in net_modules]))


    @staticmethod
    def create_node_order(network_nodes):
        """Creates an order of the symbolic nodes in the main or hidden layer
        in their `Z`-identifier form; includes univariate order and order of pairs.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_main_node_order` and `net.net_hidden_node_order`.
        These attributes are later used to define the moment and variable order
        for moment and Gillespie simulations, respectively.

        Examples
        --------
        >>> # after defining some network by
        >>> # net.structure(...)
        >>> net.net_main_node_order
        [[('Z_0',), ('Z_1',)], [('Z_0', 'Z_0'), ('Z_0', 'Z_1'), ('Z_1', 'Z_1')]]
        >>> net.net_hidden_node_order
        [[('Z_0__centric',), ('Z_0__module_0__0',), ('Z_1__centric',)],
         [('Z_0__centric', 'Z_0__centric'),
          ('Z_0__centric', 'Z_0__module_0__0'),
          ('Z_0__centric', 'Z_1__centric'),
          ('Z_0__module_0__0', 'Z_0__module_0__0'),
          ('Z_0__module_0__0', 'Z_1__centric'),
          ('Z_1__centric', 'Z_1__centric')]]
        """

        net_node_order = list()

        # the network nodes are sorted() to have the same deterministic sequence of nodes
        # for any identical set of nodes
        network_nodes = sorted(network_nodes)

        # an order for each node in a network
        # e.g., used to define the order of the first moments
        net_node_order.append([(node, ) for node in network_nodes])

        # an order for all pairs of node (symmetric pairs are only added once)
        # ['node_0', 'node_1'] would give [('node_0', 'node_0'), ('node_0', 'node_1'), ('node_1', 'node_1')]
        # e.g., used to define the order of the second moments
        net_node_order.append([(network_nodes[i], network_nodes[j])
                                for i in range(len(network_nodes))
                                for j in range(len(network_nodes))
                                if i<=j])

        return net_node_order

    @staticmethod
    def count_net_main_nodes(net_main_nodes):
        """Counts the number of main nodes for all main node identifiers;
        identifier for the environment node is always included.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_main_node_numbers`.

        Examples
        --------
        >>> # after defining some network by
        >>> # net.structure(...)
        >>> net.net_main_node_numbers
        {'Z_env': 0, 'Z_0': 1, 'Z_1': 1}
        """

        # initialise and add Z_env in any case with zero counts
        net_main_node_numbers = dict()
        net_main_node_numbers['Z_env'] = 0

        # loop over nodes and add a count if a variable was seen
        # .get allows you to specify a default value if the key does not exist
        # (from stackoverflow)
        for node in net_main_nodes:
            net_main_node_numbers[node] = net_main_node_numbers.get(node, 0) + 1

        return net_main_node_numbers

    @staticmethod
    def count_net_hidden_nodes(net_hidden_nodes):
        """Counts the number of hidden nodes for all main node identifiers;
        identifier for the environment node is always included.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_hidden_node_numbers`.

        Examples
        --------
        >>> # after defining some network by
        >>> # net.structure(...)
        >>> net.net_hidden_node_numbers
        {'Z_env': 0, 'Z_0': 2, 'Z_1': 1}
        """

        # initialise and add Z_env in any case with zero counts
        net_hidden_node_numbers = dict()
        net_hidden_node_numbers['Z_env'] = 0

        # loop over nodes and add a count if a variable was seen
        # .get allows you to specify a default value if the key does not exist
        # (from stackoverflow)
        for node in net_hidden_nodes:
            # extract the main node identifier
            node_main = node.split('__')[0]
            net_hidden_node_numbers[node_main] = net_hidden_node_numbers.get(node_main, 0) + 1

        return net_hidden_node_numbers

    @staticmethod
    def create_module_nodes(start_main_node, end_main_node, module_ident, module_steps):
        """Creates the hidden layer nodes in `Z`-identifier form for an
        Erlang-type reaction/module channel (with `module_steps` as shape parameter).

        `Note`: There is no need to run this manually, this method is run automatically
        during the main method `structure` (as part of `structure_net_hidden`).

        Examples
        --------
        >>> net = me.Network('my_net')
        >>> net.create_module_nodes('Z_1', 'Z_2', 'module_2', 5)
        ['Z_1__centric',
         'Z_1__module_2__0',
         'Z_1__module_2__1',
         'Z_1__module_2__2',
         'Z_1__module_2__3',
         'Z_2__centric']
        """

        # create the intermediate_nodes if they exist (i.e. if steps > 1)
        intermediate_nodes = list()
        if module_steps > 1:
            for i in range(module_steps - 1):
                # notate intermediate nodes as
                # <node in net_main they belong to>_<module identifier>_<number of intermediate node in that module>
                intermediate_nodes.append(f'{start_main_node}__{module_ident}__{i}')

        # add up all nodes (including start and end node) to a list
        # if intermediate_nodes are empty, there will be a direct edge between start and end node
        module_nodes = [f'{start_main_node}__centric'] + intermediate_nodes + [f'{end_main_node}__centric']
        return module_nodes

    @staticmethod
    def create_module_reaction_types(module_type, module_steps):
        """Creates the hidden layer reaction types (Markovian reaction with
        exponential waiting times) for an Erlang-type reaction/module channel
        of type `module_type` (on the non-Markovian observable layer;
        `module_steps` as Erlang shape parameter).

        `Note`: For Erlang-type reactions typically all but the last transition are
        unobservable transitions, as they do not affect the variable state
        on the observable layer; only the last
        transition into the absorbing state realises the desired state change
        on the observable layer.

        `Note`: There is no need to run this manually, this method is run automatically
        during the main method `structure` (as part of `structure_net_hidden`).

        Examples
        --------
        >>> net = me.Network('my_net')
        >>> net.create_module_reaction_types('S -> S + S', 5)
        ['S -> E', 'S -> E', 'S -> E', 'S -> E', 'S -> E + E']
        """

        # ask for the supported module reaction type and set the list of the
        # reac_types for net_hidden

        # each reaction is of type 'S -> E'
        if module_type == 'S -> E':
            reaction_types = ['S -> E'] * module_steps

        # the first reaction is type '-> E', remaining (if steps>1) are 'S -> E'
        elif module_type == '-> E':
            reaction_types = ['-> E'] + ['S -> E'] * (module_steps - 1)

        # last is 'S ->', remaining (if steps>1) prior edges are 'S -> E'
        elif module_type == 'S ->':
            reaction_types = ['S -> E'] * (module_steps - 1) + ['S ->']

        # if steps==1 then 'S -> S + E', else last is 'S -> E1 + E2' with remaining prior edges 'S -> E'
        # (E1 and E2 indicate that end nodes are different)
        elif module_type == 'S -> S + E':
            reaction_types = reaction_types = ['S -> E'] * (module_steps - 1) + ['S -> E1 + E2'] if module_steps>=2 else ['S -> S + E']

        # if steps==1 then 'S -> S + S', else last is 'S -> E + E' with remaining prior edges 'S -> E'
        elif module_type == 'S -> S + S':
            reaction_types = ['S -> E'] * (module_steps - 1) + ['S -> E + E'] if module_steps>=2 else ['S -> S + S']

        # last is 'S -> E + E', remaining (if steps>1) prior edges are 'S -> E'
        elif module_type == 'S -> E + E':
            reaction_types = ['S -> E'] * (module_steps - 1) + ['S -> E + E']

        return reaction_types

    @staticmethod
    def create_rate_identifiers(net_structure):
        """Creates the unique `theta`-identifiers and mapping for the rate parameters.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_rates_identifier`.

        Examples
        --------
        >>> # after defining some network by
        >>> # net.structure(...)
        >>> net.net_rates_identifier
        {'theta_0': 'd_xy', 'theta_1': 'l_y'}
        """

        # get all rate_symbol's from a network, remove duplicates and sort strings
        rates_sorted = sorted(set([module['rate_symbol'] for module in net_structure]))

        # create a list of neutral rate identifiers 'theta_<integer>'
        ident_rates_list = [f'theta_{i}' for i in range(len(rates_sorted))]

        # return a dictionary with rate_symbol's as values for rate identifiers as keys
        return dict(zip(ident_rates_list, rates_sorted))

    @staticmethod
    def create_node_identifiers(net_structure):
        """Creates the unique `Z`-identifiers and mapping for the main nodes.

        `Note`: After defining a network via the `net.structure` method,
        there is no need to run this. One can simply access the output at
        `net.net_nodes_identifier`.

        Examples
        --------
        >>> # after defining some network by
        >>> # net.structure(...)
        >>> net.net_nodes_identifier
        {'Z_0': 'X_t', 'Z_1': 'Y_t'}
        """

        # get all nodes from a network and sort strings
        nodes_sorted = sorted(set([module['start'] for module in net_structure] +
                                    [module['end'] for module in net_structure]))

        # create a list of neutral node identifiers 'Z_<integer>'
        # 'env' (environment) is treated separately
        count = 0
        ident_nodes_list = list()
        for node in nodes_sorted:
            if node != 'env':
                ident_nodes_list.append(f'Z_{count}')
                count += 1
            else:
                ident_nodes_list.append('Z_env')

        # return a dictionary with nodes as values for node identifiers as keys
        return dict(zip(ident_nodes_list, nodes_sorted))


    ### plotting helper function
    def _draw_main_network_graph(self, node_settings, edge_settings):
        """Private plotting helper method."""
        # NOTE: see git before June 2020 for old graphviz code here

        # set layout_engine for main networks
        layout_engine = 'dot' # 'dot', 'neato', 'circo'

        # deepcopy network to have a copy for creating the graphviz network
        net_main_graphviz = deepcopy(self.net_main)

        ### plotting via networkx
        # # instantiate lists for adding information
        # node_labels = dict()
        # node_colors = list()
        # node_sizes = 300 # networkx default
        #
        # edge_labels = dict()
        # edge_colors = list()
        ###

        ### plotting via graphviz
        net_main_graphviz.graph['graph'] = {'pad': 0.5, 'nodesep': 0.3}
        net_main_graphviz.graph['node'] = {'shape': 'circle'}
        ###

        # node settings
        for node in net_main_graphviz.nodes():
            ### plotting via networkx
            # node_labels[node] = node_settings[self.net_nodes_identifier[node]]['label']
            # node_colors.append(node_settings[self.net_nodes_identifier[node]]['color'])
            ###

            ### plotting via graphviz
            net_main_graphviz.nodes[node]['color'] = 'white'
            net_main_graphviz.nodes[node]['style'] = 'filled'
            net_main_graphviz.nodes[node]['fixedsize'] = 'true'
            # net_main_graphviz.nodes[node]['margin'] = 0.01 # if fixedsize is 'false', this might be interesting
            net_main_graphviz.nodes[node]['height'] = 0.4
            net_main_graphviz.nodes[node]['fillcolor'] = node_settings[self.net_nodes_identifier[node]]['color']
            net_main_graphviz.nodes[node]['fontsize'] = 11
            net_main_graphviz.nodes[node]['fontname'] = 'Helvetica'
            net_main_graphviz.nodes[node]['label'] = node_settings[self.net_nodes_identifier[node]]['label'] # f'<<I>{node_label}</I>>' # html based italic
            ###

        # edge settings
        for node1_id, node2_id, key, edge_inf in net_main_graphviz.edges(data=True, keys=True):
            edge = (node1_id, node2_id, key)

            sym_rate = edge_inf['module_rate_symbol']
            edge_label = edge_settings[sym_rate]['label']
            edge_color = edge_settings[sym_rate]['color']
            module_steps = edge_inf['module_steps']

            ### plotting via networkx
            # edge_labels[edge] = edge_label
            # if edge_color!=None:
            #     edge_colors.append(edge_color)
            # else:
            #     edge_colors.append('black')
            ###

            ### plotting via graphviz
            net_main_graphviz.edges[edge]['label'] = f'<  ({edge_label}, {module_steps})  >'
            net_main_graphviz.edges[edge]['fontsize'] = 11
            net_main_graphviz.edges[edge]['fontname'] = 'Helvetica'

            if edge_color!=None:
                net_main_graphviz.edges[edge]['color'] = edge_color
            else:
                net_main_graphviz.edges[edge]['color'] = 'black'
            ###

        return (net_main_graphviz, layout_engine)

    def _draw_hidden_network_graph(self, node_settings, edge_settings):
        """Private plotting helper method."""
        # NOTE: see git before June 2020 for old graphviz code here

        # set layout_engine for hidden networks
        layout_engine = 'neato' # 'dot', 'neato', 'circo'

        # deepcopy network to have a copy for creating the graphviz network
        net_hidden_graphviz = deepcopy(self.net_hidden)

        ### plotting via networkx
        # # instantiate lists for adding information
        # node_labels = dict()
        # node_colors = list()
        # node_sizes = list()
        #
        # edge_labels = dict()
        # edge_colors = list()
        ###

        ### plotting via graphviz
        net_hidden_graphviz.graph['graph'] = {'pad': 0.5, 'nodesep': 0.1}
        net_hidden_graphviz.graph['node'] = {'shape': 'circle'}
        ###

        # node settings
        for node in net_hidden_graphviz.nodes():

            node_main = node.split('__')[0]
            node_centric = node.split('__')[1]=='centric'

            ### plotting via networkx
            # node_colors.append(node_settings[self.net_nodes_identifier[node_main]]['color'])
            # if node_centric:
            #     node_labels[node] = node_settings[self.net_nodes_identifier[node_main]]['label']
            #     node_sizes.append(300) # networkx default
            # else:
            #     node_labels[node] = node.split('__')[2]
            #     node_sizes.append(200) # networkx default
            ###

            ### plotting graphviz
            net_hidden_graphviz.nodes[node]['color'] = 'white'
            net_hidden_graphviz.nodes[node]['style'] = 'filled'
            net_hidden_graphviz.nodes[node]['fixedsize'] = 'true'
            # net_hidden_graphviz.nodes[node]['margin'] = 0.01 # if fixedsize is 'false', this might be interesting
            net_hidden_graphviz.nodes[node]['fillcolor'] = node_settings[self.net_nodes_identifier[node_main]]['color']
            net_hidden_graphviz.nodes[node]['fontsize'] = 10
            net_hidden_graphviz.nodes[node]['fontname'] = 'Helvetica'

            if node_centric:
                net_hidden_graphviz.nodes[node]['height'] = 0.3
                net_hidden_graphviz.nodes[node]['label'] = node_settings[self.net_nodes_identifier[node_main]]['label'] # f'<<I>{node_label}</I>>' # html based italic
            else:
                net_hidden_graphviz.nodes[node]['height'] = 0.2
                net_hidden_graphviz.nodes[node]['label'] = node.split('__')[2]
            ###

        # edge settings
        for node1_id, node2_id, key, edge_inf in net_hidden_graphviz.edges(data=True, keys=True):
            edge = (node1_id, node2_id, key)

            sym_rate = edge_inf['module_rate_symbol']
            module_steps = edge_inf['module_steps']
            edge_label = edge_settings[sym_rate]['label']
            edge_color = edge_settings[sym_rate]['color']

            ### plotting via networkx
            # if edge_label!='':
            #     edge_labels[edge] = f' {module_steps} {edge_label} ' # f'<<I>{edge_label}</I>>' # html based italic
            #
            # if edge_color!=None:
            #     edge_colors.append(edge_color)
            # else:
            #     edge_colors.append('black')
            ###

            ### plotting via graphviz
            net_hidden_graphviz.edges[edge]['weight'] = 4.0
            net_hidden_graphviz.edges[edge]['arrowsize'] = 0.5

            if edge_label!='':
                net_hidden_graphviz.edges[edge]['label'] = f'< {module_steps} {edge_label} >'
                net_hidden_graphviz.edges[edge]['fontsize'] = 10
                net_hidden_graphviz.edges[edge]['fontname'] = 'Helvetica'

            if edge_color!=None:
                net_hidden_graphviz.edges[edge]['color'] = edge_color
            else:
                net_hidden_graphviz.edges[edge]['color'] = 'black'
            ###

        return (net_hidden_graphviz, layout_engine)

    ###

    @staticmethod
    def _validate_net_name_input(net_name):
        """Private validation method."""

        # validate user input for network name
        if isinstance(net_name, str):
            pass
        else:
            raise TypeError('A string is expected for the network name.')

    # validate the user input to define the network structure
    @staticmethod
    def _validate_net_structure(net_structure):
        """Private validation method."""

        # network structure has to be defined by a list of dictionaries
        if isinstance(net_structure, list):
            pass
        else:
            raise TypeError('A list of dictionaries is expected.')

        if all(isinstance(item, dict) for item in net_structure):
            pass
        else:
            raise TypeError('A list of dictionaries is expected.')

        # check if all required dictionary keys and only them are present
        if all(set(item) == set(['start', 'end', 'rate_symbol', 'type', 'reaction_steps'])
                        for item in net_structure):
            pass
        else:
            raise ValueError('The keys \'start\', \'end\', \'rate_symbol\', \'type\' and \'reaction_steps\' are expected in the dictionary as strings.')

        # check the value of each key to have required types and values
        for item in net_structure:
            if isinstance(item['start'], str):
                pass
            else:
                raise TypeError('Value for key \'start\' is not a string.')

            if isinstance(item['end'], str):
                pass
            else:
                raise TypeError('Value for key \'end\' is not a string.')

            if isinstance(item['rate_symbol'], str):
                pass
            else:
                raise TypeError('Value for key \'rate_symbol\' is not a string.')

            if isinstance(item['type'], str):
                pass
            else:
                raise TypeError('Value for key \'type\' is not a string.')

            if item['type'] in ['S -> E','-> E', 'S ->', 'S -> S + E',
                                            'S -> S + S', 'S -> E + E']:
                pass
            else:
                raise ValueError('Value for key \'type\' is not one of the expected strings: \'S -> E\', \'-> E\', \'S ->\', \'S -> S + E\', \'S -> S + S\', \'S -> E + E\'.')

            if isinstance(item['reaction_steps'], int):
                pass
            else:
                raise TypeError('Value for key \'reaction_steps\' is not an integer.')

            if item['reaction_steps'] >= 1:
                pass
            else:
                raise ValueError('Value for key \'reaction_steps\' is not an integer >= 1.')

            # check if 'env' (environment) node is used as required
            # check if start and end node logic is used as required
            if item['type'] == 'S -> E':
                if item['start']!='env' and item['end']!='env' and item['start']!=item['end']:
                    pass
                else:
                    raise ValueError('Invalid reaction type logic for \'{2}\' with start node \'{0}\' and end node \'{1}\'.'.format(item['start'], item['end'], item['type']))
            elif item['type'] == '-> E':
                if item['start']=='env' and item['end']!='env':
                    if item['reaction_steps'] != 1:
                        warnings.warn('Warning: reaction type \'-> E\' not supported for \'reaction_steps\' different from 1.')
                else:
                    raise ValueError('Invalid reaction type logic for \'{2}\' with start node \'{0}\' and end node \'{1}\'.'.format(item['start'], item['end'], item['type']))
            elif item['type'] == 'S ->':
                if item['start']!='env' and item['end']=='env':
                    pass
                else:
                    raise ValueError('Invalid reaction type logic for \'{2}\' with start node \'{0}\' and end node \'{1}\'.'.format(item['start'], item['end'], item['type']))
            elif item['type'] == 'S -> S + E':
                if item['start']!='env' and item['end']!='env' and item['start']!=item['end']:
                    pass
                else:
                    raise ValueError('Invalid reaction type logic for \'{2}\' with start node \'{0}\' and end node \'{1}\'.'.format(item['start'], item['end'], item['type']))
            elif item['type'] == 'S -> S + S':
                if item['start']!='env' and item['end']!='env' and item['start']==item['end']:
                    pass
                else:
                    raise ValueError('Invalid reaction type logic for \'{2}\' with start node \'{0}\' and end node \'{1}\'.'.format(item['start'], item['end'], item['type']))
            elif item['type'] == 'S -> E + E':
                if item['start']!='env' and item['end']!='env' and item['start']!=item['end']:
                    pass
                else:
                    raise ValueError('Invalid reaction type logic for \'{2}\' with start node \'{0}\' and end node \'{1}\'.'.format(item['start'], item['end'], item['type']))
