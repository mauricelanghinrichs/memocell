
import numpy as np

class GillespieSim(object):
    """docstring for ."""

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
        self.net_hidden_edges = sorted(self.net.net_hidden.edges(data=True))

        # instantiate variable for preparation of the simulations
        self.sim_gill_propensities_eval = None
        self.sim_gill_reaction_update_exec = None
        self.sim_gill_reaction_number = None

        # initialise boolean to handle the preparation step that has to be
        # executed once before a simulation
        self.gillespie_preparation_exists = False


    def prepare_gillespie_simulation(self):
        """docstring for ."""

        # preparations to be able to run gillespie simulations
        # 1) an evaluable numpy array to calculate propensities (sim_gill_propensities_eval)
        # 2) an executable function to upate node states (sim_gill_reaction_update_exec)
        # 3) number of possible reactions (sim_gill_reaction_number)
        (self.sim_gill_propensities_eval,
            self.sim_gill_reaction_update_exec,
            self.sim_gill_reaction_number) = self.define_gill_fct(
                            self.net_hidden_node_order_without_env, self.net_hidden_edges,
                            self.create_propensities_update_str, self.create_node_state_update_str)

        # once this function has run preparations are done
        self.gillespie_preparation_exists = True


    def gillespie_simulation(self, initial_values_dict, theta_values_order, time_values):
        """docstring for ."""

        ### TODO: maybe use getter/setter attributes or similar to only rerun these
        ### lines when initial_values_order or theta_values_order have changed

        # only able to run a simulation, once preparations are done
        if self.gillespie_preparation_exists:
            # create an executable string setting the numerical values of the rates (as theta identifiers)
            theta_numeric_exec = self.create_theta_numeric_exec(self.net.net_theta_symbolic, theta_values_order)

            # process user given initial values to hidden nodes
            initial_values = self.process_initial_values_order(self.net_hidden_node_order_without_env,
                                                                initial_values_dict,
                                                                self.net.net_nodes_identifier)
            ###

            # run the actual gillespie algorithm (first reaction method)
            sim_gill_sol = self.gill_first_reaction_method_sim_alg(time_values, initial_values, theta_numeric_exec,
                                                    self.sim_gill_propensities_eval, self.sim_gill_reaction_number,
                                                    self.sim_gill_reaction_update_exec)

            # interpolate the random times obtained in the simulation for a fixed time values
            sim_gill_sol_expl_time = self.exact_interpolation(sim_gill_sol, time_values)

            # add up all hidden nodes corresponding to their respective main node
            sim_gill_sol_expl_time_main = self.sum_up_nodes(self.net_main_node_order_without_env,
                                                        self.net_hidden_node_order_without_env,
                                                        sim_gill_sol_expl_time)
            return sim_gill_sol_expl_time_main

    # TODO: implement direct method or other algorithms that might be more efficient
    @staticmethod
    def gill_first_reaction_method_sim_alg(time_arr_expl, initial_values, reac_rates_exec, prop_arr_eval, num_reacs, reac_event_fct):
        """docstring for ."""

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
        """docstring for ."""

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

    @staticmethod
    def process_initial_values_order(hidden_node_order, initial_values_dict, net_nodes_identifier):
        """docstring for ."""

        initial_values = [initial_values_dict[net_nodes_identifier[node.split('__')[0]]]
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
    def define_gill_fct(node_order, net_hidden_edges,
                    create_propensities_update_str, create_node_state_update_str):
        """docstring for ."""

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
                start_node_ind = node_order.index(edge[2]['edge_start_end_identifier'][0])
            except:
                start_node_ind = 'env'

            try:
                end_node_ind = node_order.index(edge[2]['edge_start_end_identifier'][1])
            except:
                end_node_ind = 'env'

            # 2) rate of a reaction (including the factor of the step size); e.g., '2.0 * theta_0'
            reaction_rate_symbolic = edge[2]['edge_rate_symbol_identifier']

            # 3) the reaction type of this edge (not of the module)
            reaction_type = edge[2]['edge_type']

            # 4) the centric nodes connected by the module where this reaction is part of
            # (i.e., needed for S -> E1 + E2 reaction)
            # if 'Z_env__centric' is start or end node return 'env' signal word
            try:
                start_module_centric_node_ind = node_order.index(edge[2]['edge_centric_start_end_identifier'][0])

            except:
                start_module_centric_node_ind = 'env'

            try:
                end_module_centric_node_ind = node_order.index(edge[2]['edge_centric_start_end_identifier'][1])
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

        # execute function to return it afterwards
        exec(reac_event_fct_str)
        return prop_arr_str, eval('reac_event_fct'), reac_ind_count + 1


    @staticmethod
    def create_propensities_update_str(reaction_rate_symbolic, start_node_ind):
        """docstring for ."""

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
        """docstring for ."""

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
        """docstring for ."""
        # NOTE: the Gillespie simulation provides a time array of random numbers
        # at which a certain reaction happened; this function reads out the
        # state of the system for a explicitly given time array of fixed values
        # NOTE: if time_array_explicit is too coarse, one might of course miss
        # some reaction events and/or a reaction is seen much later than when it
        # actually occured; this behavior might be desired, when the natural process
        # under investigation is also observed at certain time points only

        # read out the simulation results
        time_array_gill = simulation[0]
        nodes_array_gill = simulation[1]

        # get indices to lookup the correct node state
        # works since argmax returns the first (!) True
        time_array_ind = np.array([np.argmax(time_point < time_array_gill) - 1 for time_point in time_array_explicit])
        nodes_array_explicit = nodes_array_gill[:, time_array_ind]

        return [time_array_explicit, nodes_array_explicit]

    @staticmethod
    def sum_up_nodes(main_node_order, hidden_node_order, sim_sol_expl_time):
        """docstring for ."""

        # print(main_node_order)
        # print(hidden_node_order)
        # pre allocate results array (sim_sol_expl_time_main_nodes)
        num_main_nodes = len(main_node_order)
        num_time_points = sim_sol_expl_time[0].shape[0]
        sim_sol_expl_time_main_nodes = np.zeros((num_main_nodes, num_time_points))

        # for each main node, find indices of corresponding hidden nodes
        sum_tuple_main = list()
        for main_node in main_node_order:
            tup_i = ()
            for node_ind in range(len(hidden_node_order)):

                # e.g., the following lines is True if 'Z_2__module_2__0'.split('__')[0] == 'Z_2'
                if hidden_node_order[node_ind].split('__')[0]==main_node:
                    tup_i += (node_ind, )
            sum_tuple_main.append(tup_i)

        # print(sum_tuple_main)
        # for each main node, add up corresponding hidden nodes to sim_sol_expl_time_main_nodes
        for main_node_ind in range(len(main_node_order)):
            sim_sol_expl_time_main_nodes[main_node_ind, :] = np.sum(sim_sol_expl_time[1][sum_tuple_main[main_node_ind], :], axis=0)

        return [sim_sol_expl_time[0], sim_sol_expl_time_main_nodes]
