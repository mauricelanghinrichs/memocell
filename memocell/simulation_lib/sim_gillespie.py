
"""
The simulation library contains the GillespieSim and MomentsSim helper classes
for stochastic and moment (mean, variance, covariance) simulations, respectively.
"""

import numpy as np

class GillespieSim(object):
    """Helper class for stochastic simulations.

    In the typical situation, use the top-level `Simulation` class with its
    main method `simulate` (`simulation_type='gillespie'`).
    The `GillespieSim` class and its methods are then called automatically.

    `Note`: The stochastic simulations in MemoCell are based on a
    Gillespie simulation (first-reaction method) on the hidden layer
    and produce correct stochastic realisations of the (possibly non-Markovian)
    stochastic process on the observable/main layer.
    """

    def __init__(self, net):

        # inherit the instance of the Network class
        self.net = net

        # create a lists of net_main and net_hidden nodes except
        # the environment node (without the tuple structure)
        self.net_main_node_order_without_env = [node for node,
                            in self.net.net_main_node_order[0] if node!='Z_env']
        # NOTE: net_hidden_node_order_without_env matches nodes_state order
        self.net_hidden_node_order_without_env = [node for node,
                            in self.net.net_hidden_node_order[0] if node!='Z_env__centric']

        # load the net_hidden information of all indiviual edges (reactions)
        # (keys=True provides unique identifiers for parallel multiedges)
        self.net_hidden_edges = sorted(self.net.net_hidden.edges(data=True, keys=True))

        # instantiate variable for preparation of the simulations
        self.sim_gill_propensities_eval = None
        self.sim_gill_reaction_update_exec = None
        self.sim_gill_reaction_update_str = None
        self.sim_gill_reaction_number = None

        # instantiate variables used in simulations
        self.sim_gill_theta_numeric_exec = None
        self.sim_gill_initial_values_main = None
        self.sim_gill_initial_values = None

        # instantiate objects for summation indices from hidden network
        # nodes to simulation variables
        self.summation_indices_nodes = None
        self.summation_indices_variables = None

        # initialise boolean to handle the preparation step that has to be
        # executed once before a simulation
        self.gillespie_preparation_exists = False

    def prepare_gillespie_simulation(self, variables_order, variables_identifier):
        """Prepares Gillespie simulation by creating symbolic attributes for the
        propensity and node state update schemes and the summation indices.
        Specifically `define_gill_fct`, `create_node_summation_indices` and
        `create_variables_summation_indices` downstream methods are called, see
        there for more info.
        """

        # trigger the preparation if it does not exist already
        if not self.gillespie_preparation_exists:
            # preparations to be able to run gillespie simulations
            # 1) an evaluable numpy array to calculate propensities (sim_gill_propensities_eval)
            # 2) an executable function to upate node states (sim_gill_reaction_update_exec)
            # 3) number of possible reactions (sim_gill_reaction_number)
            (self.sim_gill_propensities_eval,
                self.sim_gill_reaction_update_exec,
                self.sim_gill_reaction_update_str,
                self.sim_gill_reaction_number) = self.define_gill_fct(
                                self.net_hidden_node_order_without_env, self.net_hidden_edges,
                                self.create_propensities_update_str, self.create_node_state_update_str)

            self.summation_indices_nodes = self.create_node_summation_indices(self.net_main_node_order_without_env,
                                                        self.net_hidden_node_order_without_env)

            self.summation_indices_variables = self.create_variables_summation_indices(
                                                        variables_order,
                                                        variables_identifier,
                                                        self.net_main_node_order_without_env,
                                                        self.net.net_nodes_identifier)

            # once this function has run preparations are done
            self.gillespie_preparation_exists = True

    def gillespie_simulation(self, theta_values_order, time_values,
                                    initial_values_main, initial_values_type):
        """Top-level method in the GillespieSim class to produce one
        stochastic simulation for a memocell model.

        This method wraps downstream methods to update the user provided
        `theta` rate parameters and initial values, compute a stochastic
        simulation on the hidden layer (by `run_gillespie_first_reaction_method`),
        and sum up the hidden layer numbers to obtain observable/main node
        and simulation variable numbers.

        `Note`: In the typical situation, use the top-level `Simulation` class
        with its main method `simulate`; this method is then run automatically."""

        ### TODO: maybe use getter/setter attributes or similar to only rerun these
        ### lines when initial_values_order or theta_values_order have changed

        # only able to run a simulation, once preparations are done
        if self.gillespie_preparation_exists:
            # create an executable string setting the numerical values of the rates (as theta identifiers)
            self.sim_gill_theta_numeric_exec = self.create_theta_numeric_exec(self.net.net_theta_symbolic, theta_values_order)

            # process user given initial values to hidden nodes
            self.sim_gill_initial_values = self.process_initial_values(
                                                            initial_values_main,
                                                            initial_values_type)
            ###

            # run the actual gillespie algorithm (first reaction method)
            sim_gill_sol = self.run_gillespie_first_reaction_method(time_values,
                                                    self.sim_gill_initial_values, self.sim_gill_theta_numeric_exec,
                                                    self.sim_gill_propensities_eval, self.sim_gill_reaction_number,
                                                    self.sim_gill_reaction_update_exec)

            # interpolate the random times obtained in the simulation for a fixed time values
            sim_gill_sol_expl_time = self.exact_interpolation(sim_gill_sol, time_values)

            # add up all hidden nodes corresponding to their respective main node
            sim_gill_sol_expl_time_main = self.summation_main_nodes(
                                                        self.summation_indices_nodes,
                                                        sim_gill_sol_expl_time)

            # add main nodes to obtain results for simulation variables
            sim_gill_sol_expl_time_variables = self.summation_simulation_variables(
                                                        self.summation_indices_variables,
                                                        sim_gill_sol_expl_time_main)
            return sim_gill_sol_expl_time_variables

    # TODO: implement direct method or other algorithms that might be more efficient
    @staticmethod
    def run_gillespie_first_reaction_method(time_arr_expl, initial_values,
                    reac_rates_exec, prop_arr_eval, num_reacs, reac_event_fct):
        """Runs the Gillespie first-reaction method on the hidden
        Markov layer. Makes use of metaprogramming to be applicaple
        for any memocell network, see also `define_gill_fct` and related methods.

        `Note`: The current implementation is not yet designed for performance.
        Current main purpose is to obtain correct stochastic realisations of the
        process, in match with the exact moment simulations. Faster Gillespie
        methods exists and might be integrated in the future.
        """

        # initialise solution arrays
        t_current = time_arr_expl[0]
        maxtime = time_arr_expl[-1]
        time_array = np.array([t_current])
        nodes_array = initial_values

        # initialise kinetic rates
        exec(reac_rates_exec)

        # number of reactions and nodes
        num_nodes = nodes_array.shape[0]
        num_reacs = num_reacs

        # loop till end time point is reached
        while t_current < maxtime:

            nodes_state = nodes_array[:, -1].copy()
            propensities = eval(prop_arr_eval)

            # sample time point for each reaction event
            # a waiting time is infinity if the reaction cannot occur (propensity zero)
            waiting_times = np.full((num_reacs,), np.inf)
            waiting_times[propensities > 0.0] = np.random.exponential(scale=1/propensities[propensities > 0.0])

            # the reaction with the earliest time point happens (get its index)
            reac_ind = np.argmin(waiting_times)

            # update time
            t_current += waiting_times[reac_ind]

            # conduct the reaction event
            nodes_state = reac_event_fct(nodes_state, reac_ind)

            nodes_array = np.concatenate((nodes_array, np.array(nodes_state).reshape(num_nodes, 1)), axis=1)
            time_array = np.append(time_array, np.array([t_current]))

        return [time_array, nodes_array]

    @staticmethod
    def create_theta_numeric_exec(net_theta_symbolic, theta_values_order):
        """Creates an executable string to assign numerical values for
        `theta` rate parameters; used in `run_gillespie_first_reaction_method`
        and top-level `gillespie_simulation`.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='gillespie'`.
        Afterwards one can access the output at
        `sim.sim_gillespie.sim_gill_theta_numeric_exec`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> sim.sim_gillespie.sim_gill_theta_numeric_exec
        'theta_0, theta_1 = (0.04, 0.06)'
        """

        # create an executable string to set symbolic theta rates to
        # their respective numerical values
        theta_numeric_exec = ''

        # left-hand side (theta, ...) =
        for theta_symbol in net_theta_symbolic:
            theta_numeric_exec += theta_symbol + ', '

        theta_numeric_exec = theta_numeric_exec[:-2] + ' = ('

        # right-hand side = (0.1, ...)
        for theta_value in theta_values_order:
            theta_numeric_exec += str(theta_value) + ', '

        theta_numeric_exec = theta_numeric_exec[:-2] + ')'
        return theta_numeric_exec


    def process_initial_values(self, initial_values_main, initial_values_type):
        """Processes the user provided initial distribution (for the main
        node numbers) to obtain an initial distribution on the hidden layer,
        depending on the multinomial schemes `initial_values_type='synchronous'` or
        `initial_values_type='uniform'`.

        `Note`: `Synchronous` initial distribution type means that main node
        numbers are placed into the each main node's `'centric'` hidden layer node.
        `Uniform` initial distribution types means that main node numbers
        are distributed randomly (uniform) among all its hidden layer nodes.
        For this the respective helper methods `process_initial_values_synchronous`
        or `process_initial_values_uniform` are called.

        `Note`: The distribution types have their moment simulation equivalents,
        see there also for background theory on the employed multinomial
        sampling scheme.
        """

        if initial_values_type=='synchronous':
            initial_values_hidden = self.process_initial_values_synchronous(
                                        self.net_hidden_node_order_without_env,
                                        initial_values_main,
                                        self.net.net_nodes_identifier)

        elif initial_values_type=='uniform':
            initial_values_hidden = self.process_initial_values_uniform(
                                        self.net_main_node_order_without_env,
                                        self.net_hidden_node_order_without_env,
                                        initial_values_main,
                                        self.net.net_nodes_identifier,
                                        self.summation_indices_nodes)
        return initial_values_hidden

    @staticmethod
    def process_initial_values_synchronous(hidden_node_order, initial_values_main,
                                            net_nodes_identifier):
        """Helper method for `process_initial_values`; returns the hidden layer
        initial distribution under `'synchronous'` `initial_values_type`.
        """

        initial_values = [initial_values_main[net_nodes_identifier[node.split('__')[0]]]
                                if 'centric' in node else 0 for node in hidden_node_order]
        initial_values = np.array(initial_values).reshape((len(initial_values),1))
        # nodes_num = len(hidden_node_order)
        # initial_values = np.zeros((nodes_num, 1))
        #
        # for node_ind in range(nodes_num):
        #     if '_' not in hidden_node_order[node_ind]:
        #         initial_values[node_ind, 0] = initial_values_order[int(hidden_node_order[node_ind])]

        return initial_values

    @staticmethod
    def process_initial_values_uniform(main_node_order, hidden_node_order, initial_values_main,
                                        net_nodes_identifier, summation_indices_nodes):
        """Helper method for `process_initial_values`; returns the hidden layer
        initial distribution under `'uniform'` `initial_values_type`.
        """

        # maybe use: self.summation_indices_nodes = self.create_node_summation_indices
        # idea: loop over main nodes and extract the main initial value
        # for each main initial value, draw a random hidden node for the main node
        # and increment its index in initial_values by one
        # -> use summary_indices_nodes which should contain these indices already

        # instantiate initial values array
        initial_values = np.zeros((len(hidden_node_order), 1), dtype=int)

        # loop over main nodes via identifiers
        # (corresponds to order in summary_indices_nodes)
        for i, node in enumerate(main_node_order):
            # get initial values of main node
            init_val = initial_values_main[net_nodes_identifier[node]]
            # create the random indices of hidden nodes for the main node
            hidden_inds = np.random.choice(summation_indices_nodes[i], init_val)
            # loop over the indices and increment the hidden nodes
            for j in hidden_inds:
                initial_values[j, 0] += 1

        return initial_values

    @staticmethod
    def define_gill_fct(node_order, net_hidden_edges,
                    create_propensities_update_str, create_node_state_update_str):
        """Defines update schemes for the propensities and node states for the
        Gillespie simulation on the hidden layer.

        `Note`: Based on metaprogramming to construct schemes for any
        user provided network, using `eval()` and `exec()` methods. Uses
        `create_propensities_update_str` and `create_node_state_update_str`
        downstream methods; outputs are then used in `run_gillespie_first_reaction_method`
        and top-level `gillespie_simulation`.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='gillespie'`.
        Afterwards one can access the outputs at
        `sim_gill_propensities_eval`, `sim_gill_reaction_update_exec`,
        `sim_gill_reaction_update_str` and `sim_gill_reaction_number`.
        """

        # propensities function; this array can be used to compute propensities
        # and thus helps to determine the next reaction that takes place
        prop_arr_str = 'np.array([\n'

        # reaction event function; this function can be executed to update the
        # nodes_state given the reaction (reaction index, reac_ind) that took place
        reac_event_fct_str = 'def reac_event_fct(nodes_state, reac_ind):\n'

        # NOTE: delete following two lines?
        # # dynamic count of the reactions to get the reac_ind
        # reac_ind_count = 0

        # loop over the sorted edges of net_hidden representing the single reactions
        for reac_ind_count, edge in enumerate(net_hidden_edges):

            ### read out essential information from each reaction edge
            # 1) indices of start and end node for each reaction with respect to node_order
            # if the start or end node is 'Z_env__centric' (environment node) then return signal word 'env'
            try:
                start_node_ind = node_order.index(edge[3]['edge_start_end_identifier'][0])
            except:
                start_node_ind = 'env'

            try:
                end_node_ind = node_order.index(edge[3]['edge_start_end_identifier'][1])
            except:
                end_node_ind = 'env'

            # 2) rate of a reaction (including the factor of the step size); e.g., '2.0 * theta_0'
            reaction_rate_symbolic = edge[3]['edge_rate_symbol_identifier']

            # 3) the reaction type of this edge (not of the module)
            reaction_type = edge[3]['edge_type']

            # 4) the centric nodes connected by the module where this reaction is part of
            # (i.e., needed for S -> E1 + E2 reaction)
            # if 'Z_env__centric' is start or end node return 'env' signal word
            try:
                start_module_centric_node_ind = node_order.index(edge[3]['edge_centric_start_end_identifier'][0])

            except:
                start_module_centric_node_ind = 'env'

            try:
                end_module_centric_node_ind = node_order.index(edge[3]['edge_centric_start_end_identifier'][1])
            except:
                end_module_centric_node_ind = 'env'
            ###

            # add the propensity for this reaction to the propensitiy array
            prop_arr_str += create_propensities_update_str(reaction_rate_symbolic, start_node_ind)



            # add the node update for this reaction to the reac_event_fct_str
            reac_event_fct_str += create_node_state_update_str(reac_ind_count,
                                        start_node_ind, end_node_ind, reaction_type,
                                        start_module_centric_node_ind, end_module_centric_node_ind)

        # finish the executable strings
        prop_arr_str = prop_arr_str[:-2] + '\n])'
        reac_event_fct_str += '\treturn nodes_state'

        # print for visualisation
        # print(prop_arr_str)
        # print(reac_event_fct_str)

        # execute function to return it afterwards (and also return the string)
        exec(reac_event_fct_str)
        return prop_arr_str, eval('reac_event_fct'), reac_event_fct_str, reac_ind_count + 1

    @staticmethod
    def create_propensities_update_str(reaction_rate_symbolic, start_node_ind):
        """Helper method for `define_gill_fct`; returns propensity update scheme
        for a single hidden layer reaction."""

        if start_node_ind!='env':
            # standard first order reaction propensities
            return f'{reaction_rate_symbolic} * nodes_state[{start_node_ind}],\n'
        else:
            # standard zero order reaction propensities in case of 'env'
            return f'{reaction_rate_symbolic} * 1.0,\n'

    @staticmethod
    def create_node_state_update_str(reac_ind_count,
                                start_node_ind, end_node_ind, reaction_type,
                                start_module_centric_node_ind, end_module_centric_node_ind):
        """Helper method for `define_gill_fct`; returns node update scheme
        for a single hidden layer reaction."""

        # each reaction has its unique index
        add_to_reac_event_fct_str = '\tif reac_ind=={0}:\n'.format(reac_ind_count)

        # NOTE: we do not have to do case distinction for start/end_node_ind == 'env'
        # this is already checked in Network class

        # reaction type of the given reaction determine the nodes_state update step
        # simple linear conversion
        # e.g., cell differentiation
        if reaction_type == 'S -> E':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{start_node_ind}] -= 1\n'
            add_to_reac_event_fct_str += f'\t\tnodes_state[{end_node_ind}] += 1\n'
            return add_to_reac_event_fct_str

        # birth reaction (constant rate input)
        elif reaction_type == '-> E':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{end_node_ind}] += 1\n'
            return add_to_reac_event_fct_str

        # linear death reaction
        elif reaction_type == 'S ->':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{start_node_ind}] -= 1\n'
            return add_to_reac_event_fct_str

        # linear asymmetric production
        # e.g., asymmetric cell division
        elif reaction_type == 'S -> S + E':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{end_node_ind}] += 1\n'
            return add_to_reac_event_fct_str

        # linear (symmetric) production
        # e.g., exponential growth, symmetric cell division
        elif reaction_type == 'S -> S + S':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{start_node_ind}] += 1\n'
            return add_to_reac_event_fct_str

        # linear production upon conversion
        # e.g., cell division upon differentiation
        elif reaction_type == 'S -> E + E':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{start_node_ind}] -= 1\n'
            add_to_reac_event_fct_str += f'\t\tnodes_state[{end_node_ind}] += 2\n'
            return add_to_reac_event_fct_str

        # linear production upon conversion into distict products
        # NOTE: this reaction is a special requirement to realise a
        # module reaction such as S -> S + E on the net_hidden level, because
        # product S (E1) and product E (E2) are different and not identical on
        # substrate S
        # NOTE: E1 represents the centric node where the module started
        # E2 represents the centric node where the module ends (as in all others
        # reactions)
        elif reaction_type == 'S -> E1 + E2':
            add_to_reac_event_fct_str += f'\t\tnodes_state[{start_node_ind}] -= 1\n' # S
            add_to_reac_event_fct_str += f'\t\tnodes_state[{start_module_centric_node_ind}] += 1\n' # E1
            add_to_reac_event_fct_str += f'\t\tnodes_state[{end_node_ind}] += 1\n' # E2
            return add_to_reac_event_fct_str

    @staticmethod
    def exact_interpolation(simulation, time_array_explicit):
        """Reads out a stochastic `simulation` at explicitly
        given time values (`time_array_explicit`).

        `Note`: The Gillespie simulation provides a time array of random numbers
        at which the reactions happened; this method reads out the
        state of the system for an explicitly given time array of fixed values.
        This means: If `time_array_explicit` is too coarse, one might miss
        some reaction events and/or a reaction is seen much later than it
        actually occured; this behavior is desired, when the biological
        experiments are also restricted to read-outs at certain time points only.
        In contrast, one can choose a very dense time array to report all reaction
        events.

        Examples
        --------
        >>> import memocell as me
        >>> import numpy as np
        >>> # made-up stochastic simulation times and numbers
        >>> time_array_gill = np.array([0.00, 0.12, 4.67, 8.01, 10.00])
        >>> nodes_array_gill = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        >>> simulation = [time_array_gill, nodes_array_gill]
        >>> # explicit times to read-out the simulation
        >>> time_array_explicit = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        >>> me.simulation_lib.sim_gillespie.GillespieSim.exact_interpolation(simulation, time_array_explicit)
        [array([ 0.,  2.,  4.,  6.,  8., 10.]), array([[1., 2., 2., 3., 3., 5.]])]
        """
        # read out the simulation results
        time_array_gill = simulation[0]
        nodes_array_gill = simulation[1]

        # get indices to lookup the correct node state
        # works since argmax returns the first (!) True
        time_array_ind = np.array([np.argmax(time_point < time_array_gill) - 1 for time_point in time_array_explicit])
        nodes_array_explicit = nodes_array_gill[:, time_array_ind]

        return [time_array_explicit, nodes_array_explicit]

    @staticmethod
    def create_node_summation_indices(main_node_order, hidden_node_order):
        """Creates a list of tuples with hidden node indices that are needed to
        sum up each main node; index ordering as in
        `sim.sim_gillespie.net_main_node_order_without_env` and
        `sim.sim_gillespie.net_hidden_node_order_without_env` with
        `Z`-identifier `sim.net.net_nodes_identifier`.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='gillespie'`.
        Afterwards one can access the output at
        `sim.sim_gillespie.summation_indices_nodes`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> # e.g., the first four hidden nodes provide the first main node
        >>> sim.sim_gillespie.summation_indices_nodes
        [(0, 1, 2, 3), (4, 5, 6)]
        """

        # for each main node, find indices of corresponding hidden nodes
        sum_tuple_main = list()
        for main_node in main_node_order:
            tup_i = ()
            for node_ind in range(len(hidden_node_order)):

                # e.g., the following lines is True if 'Z_2__module_2__0'.split('__')[0] == 'Z_2'
                if hidden_node_order[node_ind].split('__')[0]==main_node:
                    tup_i += (node_ind, )
            sum_tuple_main.append(tup_i)

        return sum_tuple_main

    @staticmethod
    def summation_main_nodes(sum_tuple_main, sim_sol_expl_time):
        """Computes the stochastic simulation numbers for the main nodes
        from the Gillespie-simulated numbers of the hidden nodes; returns list of time values
        and simulated number of the main nodes (with order as in
        `sim.sim_gillespie.net_main_node_order_without_env`
        and `sim.net.net_nodes_identifier`).

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='gillespie'`.
        """

        # pre allocate results array (sim_sol_expl_time_main_nodes)
        num_time_points = sim_sol_expl_time[0].shape[0]
        sim_sol_expl_time_main_nodes = np.zeros((len(sum_tuple_main), num_time_points))

        # print(sum_tuple_main)
        # for each main node, add up corresponding hidden nodes to sim_sol_expl_time_main_nodes
        for main_node_ind in range(len(sum_tuple_main)):
            sim_sol_expl_time_main_nodes[main_node_ind, :] = np.sum(sim_sol_expl_time[1][sum_tuple_main[main_node_ind], :], axis=0)

        return [sim_sol_expl_time[0], sim_sol_expl_time_main_nodes]

    @staticmethod
    def create_variables_summation_indices(variables_order,
                                            variables_identifier,
                                            net_main_node_order_without_env,
                                            net_nodes_identifier):
        """Creates a list of tuples with main node indices that are needed to
        sum up each simulation variable; index ordering as in
        `sim.sim.sim_variables_order` and
        `sim.sim_gillespie.net_main_node_order_without_env` with
        `V`-identifier `sim.sim.sim_variables_identifier` and
        `Z`-identifier `sim.net.net_nodes_identifier`, respectively.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='gillespie'`.
        Afterwards one can access the output at
        `sim.sim_gillespie.summation_indices_variables`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> # e.g., main nodes and simulation variables are the same
        >>> sim.sim_gillespie.summation_indices_variables
        [(0,), (1,)]
        """

        # inverse the node identifier dictionary
        net_nodes_identifier_inv = {val: key for key, val in net_nodes_identifier.items()}

        # with this, create new dict with variables identifiers and node identifiers ('Z_<int>' nomenclature)
        # e.g., {'V_0': ('W_t', ('Y_t', 'X_t')), 'V_1': ('X_t', ('X_t',)), 'V_2': ('Y_t', ('Y_t',))}
        # becomes {'V_0': ('W_t', ('Z_0', 'Z_1')), 'V_1': ('X_t', ('Z_0',)), 'V_2': ('Y_t', ('Z_1',))}
        variables_node_identifier = dict()
        for key, value in variables_identifier.items():
            variable_nodes = value[1]
            variable_nodes_ident = tuple(sorted([net_nodes_identifier_inv[node] for node in variable_nodes]))
            variables_node_identifier[key] = (value[0], variable_nodes_ident)

        # loop over variable identifiers
        sum_tuple_variables = list()
        for var in variables_order[0]:
            tup_i = tuple()

            # obtain the nodes identifier for each variable
            variable_id = var[0]
            variable_nodes_id = variables_node_identifier[variable_id][1]

            # for a given node identifier, look up its index in the moment solution
            for node_id in variable_nodes_id:
                ind_node_id = net_main_node_order_without_env.index(node_id)
                tup_i += (ind_node_id, )

            sum_tuple_variables.append(tup_i)

        return sum_tuple_variables

    @staticmethod
    def summation_simulation_variables(sum_tuple_variables, sim_sol_expl_time_main_nodes):
        """Computes the stochastic simulation numbers for the simulation variables
        from the simulated numbers of the main nodes; returns list of time values
        and simulated number of the variables (with order as in
        `sim.sim.sim_variables_order` and `sim.sim.sim_variables_identifier`).

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='gillespie'`.
        """

        # pre allocate results array (sim_sol_expl_time_sim_variables)
        num_time_points = sim_sol_expl_time_main_nodes[0].shape[0]
        sim_sol_expl_time_sim_variables = np.zeros((len(sum_tuple_variables), num_time_points))

        # print(sum_tuple_variables)
        # for each simulation output variable, add up corresponding main nodes to sim_sol_expl_time_sim_variables
        for variable_ind in range(len(sum_tuple_variables)):
            sim_sol_expl_time_sim_variables[variable_ind, :] = np.sum(sim_sol_expl_time_main_nodes[1][sum_tuple_variables[variable_ind], :], axis=0)

        return [sim_sol_expl_time_main_nodes[0], sim_sol_expl_time_sim_variables]
