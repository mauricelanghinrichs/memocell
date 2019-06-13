
# TODO: remove this when version check is no longer needed
import sympy
#

from sympy import var
from sympy import sympify
from sympy import Function
from sympy import diff

import itertools
import numpy as np
from scipy.integrate import odeint
from numba import jit

# TODO: delete after run time analysis
import time

class MomentsSim(object):
    """docstring for ."""
    def __init__(self, net):

        # inherit the instance of the Network class
        self.net = net

        # load the net_hidden information of all indiviual edges (reactions)
        self.net_hidden_edges = sorted(self.net.net_hidden.edges(data=True))

        # boolean to indicate if (first) or (first and second) order moments should be derived
        # specified through upper level class Simulation
        self.moment_mean_only = None

        # boolean to indicate whether simulation is called from estimation class or not
        # specified through upper level class Simulation
        self.moment_estimation_mode = None

        # initialise boolean to handle the preparation step that has to be
        # executed once before a simulation
        self.moments_preparation_exists = False

        # instantiate moment order lists
        self.moment_order_main = list()
        self.moment_order_hidden = list()

        # instantiate objects for auxiliary variables
        self.moment_aux_vars = list()
        self.moment_aux_vars_dict = dict()

        # instantiate object for initial values for the moments
        self.moment_initial_values = None
        self.moment_initial_values_exist = False

        # instantiate objects for string-replaceable symbolic parameters (theta notation)
        self.theta_replaceable = list()
        self.theta_replaceable_dict = dict()

        # numerical values for theta parameters (in order as in self.theta_replaceable or
        # self.net.net_theta_symbolic, together with self.net.net_rates_identifier)
        self.theta_numeric = list()

        # instantiate variables for the partial differential equation (pde)
        # and the differential equations for the moments
        self.moment_pde = None
        self.moment_eqs = None

        # instantiate objects the ordinary differential system (ode) for the moments
        self.moment_eqs_template_str = None
        self.moment_system = None

        # instantiate objects for storing the indices of the ode system/ moments
        # of the hidden Markov layer that belong to a certain main network moment
        self.moment_mean_ind = None
        self.moment_var_ind_intra = None
        self.moment_var_ind_inter = None
        self.moment_cov_ind = None

        # instantiate objects for the number of mean, variance and covariance equations
        # given the nodes in the main network and the kind of moment simulations
        # (first only (mean_only=True) or first and second moments)
        self.moment_num_means = None
        self.moment_num_vars = None
        self.moment_num_covs = None


    def prepare_moment_simulation(self, variables_order, variables_identifier, mean_only=False, estimate_mode=False):
        """docstring for ."""

        # set information for mean_only and estimation modes
        self.moment_mean_only = mean_only
        self.moment_estimation_mode = estimate_mode

        # derive an order of the moments
        self.moment_order_main = self.derive_moment_order_main(self.net.net_main_node_order, self.moment_mean_only)
        self.moment_order_hidden = self.derive_moment_order_hidden(self.net.net_hidden_node_order, self.moment_mean_only)

        # create a list of auxiliary variables for the moment approach
        # set of variables identical to the nodes in moment_order_hidden
        # e.g., 'Z_0__module_1__0' provides 'z_0__module_1__0_q' auxiliary variable
        # 'q' at the end is short for 'quit' and indicates the end of the string
        # (this is help for string replacement)
        self.moment_aux_vars = ['z' + node[1:] + '_q' for node, in self.moment_order_hidden[0]]
        # create a dictionary that links each node (key) to its auxiliary variable (value)
        self.moment_aux_vars_dict = dict(zip([node for node, in self.moment_order_hidden[0]], self.moment_aux_vars))

        # create a list and dictionary of theta identifiers with '_q' indicating end
        self.theta_replaceable = [theta + '_q' for theta in self.net.net_theta_symbolic]
        self.theta_replaceable_dict = dict(zip(self.net.net_theta_symbolic, self.theta_replaceable))

        # derive the partial differential equation of the probability generating function
        self.moment_pde = self.derive_moment_pde(self.net_hidden_edges, self.moment_aux_vars, self.moment_aux_vars_dict, self.theta_replaceable_dict)

        # derive differential equations for the moments (E(X), E(X (X-1)), E(X Y))
        self.moment_eqs = self.derive_moment_eqs(self.moment_pde, self.moment_order_hidden, self.moment_aux_vars, self.moment_aux_vars_dict, self.theta_replaceable)

        # for moments in the main network, collect the nodes of the hidden network for summation
        (self.moment_num_means, self.moment_mean_ind, self.moment_num_vars,
        self.moment_var_ind_intra, self.moment_var_ind_inter, self.moment_num_covs,
        self.moment_cov_ind) = self.get_indices_for_solution_readout(self.moment_order_main, self.moment_order_hidden)

        # setup an executable string for the simuation of the moment equations
        self.moment_system = self.setup_executable_moment_eqs_template(self.moment_eqs)

        # TODO: variables feature
        (self.variables_num_means, self.variables_mean_ind,
        self.variables_num_vars, self.variables_var_ind,
        self.variables_num_covs, self.variables_cov_ind) = self.get_indices_for_moment_readout(
                                                                        variables_order,
                                                                        variables_identifier,
                                                                        self.moment_order_main,
                                                                        self.net.net_nodes_identifier)

        # once this function has run preparations are done
        self.moments_preparation_exists = True

    def moment_simulation(self, initial_values_dict, theta_values_order, time_values):
        """docstring for ."""

        ### TODO: maybe use getter/setter attributes or similar to only rerun these
        ### lines when initial_values_order or theta_values_order have changed

        # check if preparation was executed
        if self.moments_preparation_exists:

            # if not in estimation_mode, process user given initial values to hidden nodes
            if not self.moment_estimation_mode:
                self.moment_initial_values = self.process_initial_values_order(self.moment_order_hidden,
                                                                    initial_values_dict,
                                                                    self.net.net_nodes_identifier,
                                                                    type='centric_mean_only')
                self.moment_initial_values_exist = True

            # if in estimation_mode, process user given initial values to hidden nodes only the first time
            else:
                if self.moment_initial_values_exist:
                    pass
                else:
                    # process user given initial values to hidden nodes
                    self.moment_initial_values = self.process_initial_values_order(self.moment_order_hidden,
                                                                        initial_values_dict,
                                                                        self.net.net_nodes_identifier,
                                                                        type='centric_mean_only')
                    self.moment_initial_values_exist = True

            # setting the numerical values of the rates (as theta identifiers and in symbolic theta order)
            self.theta_numeric = theta_values_order
            ###

            # simulate the network, given initial_values, time points and parameters (theta)
            return self.forward_pass(self.moment_initial_values, time_values, theta_values_order)

    @staticmethod
    def derive_moment_order_main(node_order, mean_only):
        """docstring for ."""

        # initialise moment_order, first index for mean moments, second index for second order moments
        moment_order = list(([], []))

        # mean (first order) moments E(X) (with expectation value E())
        # go through node_order, but leave out all 'Z_env' nodes and node tuples
        moment_order[0] = [(node, ) for node, in node_order[0] if not node=='Z_env']

        # second order moments E(X (X-1)) or E(X Y) (with expectation value E())
        if not mean_only:
            moment_order[1] = [(node_1, node_2) for node_1, node_2 in node_order[1]
                                if not (node_1=='Z_env' or node_2=='Z_env')]

        return moment_order

    @staticmethod
    def derive_moment_order_hidden(node_order, mean_only):
        """docstring for ."""

        # initialise moment_order, first index for mean moments, second index for second order moments
        moment_order = list(([], []))

        # mean (first order) moments E(X) (with expectation value E())
        # go through node_order, but leave out all 'Z_env' nodes and node tuples
        moment_order[0] = [(node, ) for node, in node_order[0] if not node=='Z_env__centric']

        # second order moments E(X (X-1)) or E(X Y) (with expectation value E())
        if not mean_only:
            moment_order[1] = [(node_1, node_2) for node_1, node_2 in node_order[1]
                                if not (node_1=='Z_env__centric' or node_2=='Z_env__centric')]

        return moment_order

    def derive_moment_pde(self, net_edges, z_aux_vars, z_aux_vars_dict, theta_repl_dict):
        """docstring for ."""

        # subsequently, add parts to the pde
        pde = ''

        for edge in net_edges:
            # read out the auxiliary variables for the start and end node each reaction
            # if a node is the environmental node ('Z_env__centric'), a constant is introduced (=1.0)
            z_start_node = z_aux_vars_dict[edge[0]] if edge[0]!='Z_env__centric' else '1.0'
            z_node_end = z_aux_vars_dict[edge[1]] if edge[1]!='Z_env__centric' else '1.0'

            # read out reaction type and reaction rate (symbolic form, accelerated by step size)
            reac_type = edge[2]['edge_type']
            # example for reaction_rate: '3.0 * theta_2_q' (if module has theta rate 'theta_2' and three reaction steps)
            reac_rate = edge[2]['edge_rate_symbol_identifier'].replace(edge[2]['module_rate_symbol_identifier'],
                                                                theta_repl_dict[edge[2]['module_rate_symbol_identifier']])

            # for the special case of an edge type of 'S -> E1 + E2' capture both end nodes
            # this edge type occurs for the last reaction of a 'S -> S + E' module
            if reac_type == 'S -> E1 + E2':
                # the end node which is the start (centric) node of the module
                z_node_end_1 = z_aux_vars_dict[edge[2]['edge_centric_start_end_identifier'][0]]

                # the end node which the start node is actually connected to
                z_node_end_2 = z_node_end

            # add all pde parts as string to the overall pde given the reaction type
            if reac_type == '-> E':
                pde += self.reac_type_to_end(z_start_node, z_node_end, reac_rate, z_aux_vars)
            elif reac_type == 'S ->':
                pde += self.reac_type_start_to(z_start_node, z_node_end, reac_rate, z_aux_vars)
            elif reac_type == 'S -> E':
                pde += self.reac_type_start_to_end(z_start_node, z_node_end, reac_rate, z_aux_vars)
            elif reac_type == 'S -> S + E':
                pde += self.reac_type_start_to_start_end(z_start_node, z_node_end, reac_rate, z_aux_vars)
            elif reac_type == 'S -> S + S':
                pde += self.reac_type_start_to_start_start(z_start_node, z_node_end, reac_rate, z_aux_vars)
            elif reac_type == 'S -> E + E':
                pde += self.reac_type_start_to_end_end(z_start_node, z_node_end, reac_rate, z_aux_vars)
            elif reac_type == 'S -> E1 + E2':
                pde += self.reac_type_start_to_end1_end2(z_start_node, z_node_end_1, z_node_end_2, reac_rate, z_aux_vars)

            # pde parts are summed up, hence ' + '
            if edge != net_edges[-1]:
                pde += ' + '

        return pde

    @staticmethod
    def derive_moment_eqs(moment_pde, moment_order_hidden, moment_aux_vars, moment_aux_vars_dict, theta_replaceables):
        """docstring for ."""

        # initialise sympy objects
        z_vars = var(' '.join(moment_aux_vars))
        # params = var(' '.join(['theta_{0}_'.format(key) for key in theta_order_dict.keys()]))
        PDE = sympify(moment_pde)

        # for replacement later
        z_vars_str = ', '.join(moment_aux_vars)

        ### first step: append unsubstituted moments to list by differentiation
        ### of the pde (using sympy methods)
        moment_eqs = list()

        # first order moments (means: E(X))
        moment_order_1st_vars = [moment_aux_vars_dict[node] for node, in moment_order_hidden[0]]

        # append first derivatives
        for z_var in moment_order_1st_vars:
            moment_eqs.append(diff(PDE, z_var))

        # second order moments ((i,i)-tuple: E(X(X-1)); (i,j)-tuple E(X Y) with i!=j)
        moment_order_2nd_vars = [(moment_aux_vars_dict[node1], moment_aux_vars_dict[node2]) for node1, node2 in moment_order_hidden[1]]

        # append second derivatives (using the already computed first derivatives)
        # NOTE: Schwarz theorem ensures that order of differentiation does not matter
        for z_var1, z_var2 in moment_order_2nd_vars:
            moment_eqs.append(diff(moment_eqs[moment_order_1st_vars.index(z_var1)], z_var2))

        ### second step: convert sympy object to string and conduct string
        ### substitution methods
        moment_eqs = [str(eq) for eq in moment_eqs]

        # create a list of tuples for replacement ((old str, new str))
        replace_tuples = list()

        # NOTE: that below our alpha-numerical ordering of z_vars coincides with sympy's ordering
        count_i = 0
        for z_var in moment_order_1st_vars:
            string_deriv = f'Derivative(F({z_vars_str}), {z_var})'
            string_subs = f'm_{count_i}_q'
            replace_tuples.append((string_deriv, string_subs))

            count_i += 1

        for z_var1, z_var2 in moment_order_2nd_vars:
            # TODO: check if all similar to old version and update sympy (>=1.3)
            if z_var1==z_var2:
                if sympy.__version__=='1.1.1':
                    print('hello 1.1.1')
                    string_deriv = f'Derivative(F({z_vars_str}), {z_var1}, {z_var2})'
                else:
                    string_deriv = f'Derivative(F({z_vars_str}), ({z_var1}, 2))'
            else:
                # the following sorted function is needed since auxiliary vars (with
                # additional 'q') can have a different tuple order than the tuple
                # order in moment_order_hidden (e.g. if one hidden node has index
                # >= 10 (two digits)), see jupyter notebook for sympy bug
                z_var1, z_var2 = tuple(sorted([z_var1, z_var2]))
                string_deriv = f'Derivative(F({z_vars_str}), {z_var1}, {z_var2})'
            string_subs = f'm_{count_i}_q'
            replace_tuples.append((string_deriv, string_subs))

            count_i += 1

        # replace remaining z variables by 1.0
        for z_var in moment_aux_vars:
            replace_tuples.append((z_var, '1.0'))

        # replace higher moment derivatives by a constant (these terms cancel anyway)
        # replace second order derivatives when there are no demanded second moments
        if len(moment_order_2nd_vars) == 0:
            inner_F = ', '.join(len(moment_aux_vars)*['1.0'])
            replace_tuples.append((f'Derivative(F({inner_F}), 1.0, 1.0)', 'const'))
        # else replace third order derivatives
        else:
            inner_F = ', '.join(len(moment_aux_vars)*['1.0'])

            # TODO: check if all similar to old version and update sympy (>=1.3)
            if sympy.__version__=='1.1.1':
                replace_tuples.append((f'Derivative(F({inner_F}), 1.0, 1.0, 1.0)', 'const'))
                print('hello 1.1.1')
            else:
                replace_tuples.append((f'Derivative(F({inner_F}), 1.0, 1.0, 1.0)', 'const'))
                replace_tuples.append((f'Derivative(F({inner_F}), (1.0, 2), 1.0)', 'const'))
                replace_tuples.append((f'Derivative(F({inner_F}), 1.0, (1.0, 2))', 'const'))
                replace_tuples.append((f'Derivative(F({inner_F}), (1.0, 3))', 'const'))

        # replace the plain probability generating function by one (since probabilities sum up to one)
        inner_F = ', '.join(len(moment_aux_vars)*['1.0'])
        replace_tuples.append((f'F({inner_F})', '1.0'))

        # now conduct substitution
        for i, eq in enumerate(moment_eqs):
            for tup in replace_tuples:
                eq = eq.replace(*tup)
            moment_eqs[i] = eq

        ### third and last step: sympify each eq, so that sympy can cancel terms
        ### and replace '_' in theta's and m's to have real brackets
        ### (we cannot use brackets in the first place due to sympy)
        # sympify with sympy leading to analytic term simplification
        for i, eq in enumerate(moment_eqs):
            moment_eqs[i] = str(sympify(eq))

        # create remaining string substitutions now to obtain evaluable brackets
        # create a list of tuples for replacement ((old str, new str))
        replace_tuples_2 = list()

        # replace moments and theta by bracket notation
        # e.g., 'm_12_q' becomes 'm[12]'
        for i in range(len(moment_order_1st_vars) + len(moment_order_2nd_vars)):
            replace_tuples_2.append((f'm_{i}_q', f'm[{i}]'))

        # e.g., 'theta_2_q' becomes 'theta[2]'
        for theta_repl in theta_replaceables:
            theta_num = theta_repl.split('_')[1]
            replace_tuples_2.append((theta_repl, f'theta[{theta_num}]'))

        # conduct substitution
        for i, eq in enumerate(moment_eqs):
            for tup in replace_tuples_2:
                eq = eq.replace(*tup)
            moment_eqs[i] = eq

        # print(moment_eqs)
        return moment_eqs

    @staticmethod
    def get_indices_for_solution_readout(moment_order_main, moment_order_hidden):
        """docstring for ."""

        # count the numbers of mean, var and covar moment equations for the main nodes
        # 'val' in the following are the tuples describing the moments,
        # i.e. first moment of node 'Z_0' is ('Z_0',), second moment between node 'Z_0' and 'Z_1' is
        # due to string sorting always ('Z_0', 'Z_1')
        mean_match = [val for val in moment_order_main[0] if len(val)==1]
        num_means = len(mean_match)
        var_match = [val for val in moment_order_main[1] if val[0]==val[1]]
        num_vars = len(var_match)
        cov_match = [val for val in moment_order_main[1] if val[0]!=val[1]]
        num_covs = len(cov_match)

        # print(mean_match)
        # print(num_means)
        # print(var_match)
        # print(num_vars)
        # print(cov_match)
        # print(num_covs)

        # cast the numpy arrays which store the index information
        # means are just the first moments (second axis dimension = 1)
        mean_ind = np.zeros((num_means, 1), dtype=object)

        # variances are composed of two or three moments (second axis dimension = 2 or 3)
        # intra (two moments) if it is a real variance (=self-covariance) of a node belonging to the set of a given main node
        # e.g.: Var(Z_0__module_1__0) = Cov(Z_0__module_1__0, Z_0__module_1__0)
        # inter (three moments) if it is an actual covariance of two different nodes, but both belonging to the same set of a given main node
        # e.g.: Cov(Z_0__module_1__0, Z_0__module_1__1)
        var_ind_intra = np.zeros((num_vars, 2), dtype=object)
        var_ind_inter = np.zeros((num_vars, 3), dtype=object)

        # covariances are composed of three moments (second axis dimension = 3)
        # NOTE: these are actual covariances of different nodes, since we do not
        # allow shared intermediate nodes (which would belong to two different main nodes)
        # e.g.: Cov(Z_0__module_1__0, Z_1__module_2__0)
        cov_ind = np.zeros((num_covs, 3), dtype=object)

        # set the mean indices
        mean_match_tup_list = list()
        for i in range(num_means):
            mean_match_tup = tuple()
            for ind, val in enumerate(moment_order_hidden[0]):
                    if mean_match[i][0]==val[0].split('__')[0]:
                        mean_match_tup += (ind, )

            mean_ind[i, 0] = mean_match_tup
            mean_match_tup_list.append(mean_match_tup)

        # set the variance indices
        for i in range(num_vars):
            var_intra_match_tup = ()

            var_inter_match_tup = ()
            var_inter_mean1_match_tup = ()
            var_inter_mean2_match_tup = ()
            for ind, val in enumerate(moment_order_hidden[1]):
                # ask for the set of nodes belonging to main node with index i
                if var_match[i][0]==val[0].split('__')[0] and var_match[i][1]==val[1].split('__')[0]:
                    # intra variances
                    if val[0]==val[1]:
                        # number of mean equations is added to the index (' + len(moment_order_hidden[0])')
                        var_intra_match_tup += (ind + len(moment_order_hidden[0]), )
                    # inter variances (an actual covariance)
                    elif val[0]!=val[1]:
                        # number of mean equations is added to the index (' + len(moment_order_hidden[0])')
                        var_inter_match_tup += (ind + len(moment_order_hidden[0]), )
                        # get the corresponding two means
                        for ind_2, val_2 in enumerate(moment_order_hidden[0]):
                            if val[0]==val_2[0]:
                                var_inter_mean1_match_tup += (ind_2, )
                            elif val[1]==val_2[0]:
                                var_inter_mean2_match_tup += (ind_2, )

            var_ind_intra[i, 0] = var_intra_match_tup
            var_ind_intra[i, 1] = mean_match_tup_list[i]

            var_ind_inter[i, 0] = var_inter_match_tup
            var_ind_inter[i, 1] = var_inter_mean1_match_tup
            var_ind_inter[i, 2] = var_inter_mean2_match_tup


        # set the covariance indices
        for i in range(num_covs):
            cov_match_tup = ()

            cov_mean1_match_tup = ()
            cov_mean2_match_tup = ()
            for ind, val in enumerate(moment_order_hidden[1]):
                # ask for covariances which are between the sets of the two main nodes
                if cov_match[i][0]==val[0].split('__')[0] and cov_match[i][1]==val[1].split('__')[0]:
                    if val[0]!=val[1]:
                        # number of mean equations is added to the index (' + len(moment_order_hidden[0])')
                        cov_match_tup += (ind + len(moment_order_hidden[0]), )
                        # get the corresponding two means
                        for ind_2, val_2 in enumerate(moment_order_hidden[0]):
                                if val[0]==val_2[0]:
                                    cov_mean1_match_tup += (ind_2, )
                                elif val[1]==val_2[0]:
                                    cov_mean2_match_tup += (ind_2, )

            cov_ind[i, 0] = cov_match_tup
            cov_ind[i, 1] = cov_mean1_match_tup
            cov_ind[i, 2] = cov_mean2_match_tup

        # print(mean_ind)
        #
        # print(var_ind_intra)
        # print(var_ind_inter)
        #
        # print(cov_ind)

        return num_means, mean_ind, num_vars, var_ind_intra, var_ind_inter, num_covs, cov_ind

    @staticmethod
    def get_indices_for_moment_readout(variables_order,
                                            variables_identifier,
                                            moment_order_main,
                                            net_nodes_identifier):
        """docstring for ."""

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

        # read out the order of means, vars, covs and their occurrences in
        # variable identifier notation ('V_<int>') from variables order
        variables_mean_match = [val for val in variables_order[0] if len(val)==1]
        variables_num_means = len(variables_mean_match)
        variables_var_match = [val for val in variables_order[1] if val[0]==val[1]]
        variables_num_vars = len(variables_var_match)
        variables_cov_match = [val for val in variables_order[1] if val[0]!=val[1]]
        variables_num_covs = len(variables_cov_match)

        # re-load the moment order of the moment solutions in node
        # identifier notation ('Z_<int>')
        moment_mean_match = [val for val in moment_order_main[0] if len(val)==1]
        moment_var_match = [val for val in moment_order_main[1] if val[0]==val[1]]
        moment_cov_match = [val for val in moment_order_main[1] if val[0]!=val[1]]

        # preallocate numpy array object to later assign tuples to sum over from moment solutions;
        # the second axis has two dimensions for var and cov (index=0 to obtain
        # node variances, index=1 to obtain node covariances)
        variables_mean_ind = np.zeros((variables_num_means, 1), dtype=object)
        variables_var_ind = np.zeros((variables_num_vars, 2), dtype=object)
        variables_cov_ind = np.zeros((variables_num_covs, 2), dtype=object)

        # set the mean indices; for a given variable V = Z0 + ... + Zn it
        # holds that E(V) = E(Z0) + ... + E(Zn), thus we collect all nodes
        # belonging to variable V here
        for ind_var, var in enumerate(variables_mean_match):
            mean_match_tup = tuple()

            # obtain the nodes identifier for each variable
            variable_id = var[0]
            variable_nodes_id = variables_node_identifier[variable_id][1]

            # for a given node identifier, look up its index in the moment solution
            for node_id in variable_nodes_id:
                ind_node_id = moment_mean_match.index((node_id, ))
                mean_match_tup += (ind_node_id, )

            variables_mean_ind[ind_var, 0] = mean_match_tup

        # set the variance indices; for a given variable V = Z0 + ... + Zn it
        # holds that Var(V) = Cov(V, V) = sum(i=0 to n, Var(Zi)) + sum(over (i,j) i!=j, Cov(Zi, Zj))
        for ind_var, var in enumerate(variables_var_match):
            var_intra_match_tup = () # for sum(i=0 to n, Var(Zi))
            var_inter_match_tup = () # for sum(over (i,j) i!=j, Cov(Zi, Zj))

            # obtain the nodes identifier for each variable
            variable_id = var[0] # == var[1]
            variable_nodes_id = variables_node_identifier[variable_id][1]

            # create a cartesian product to get all node_id's tuple combinations;
            # order is removed by sorted(), e.g. ('Z_1', 'Z_0') == ('Z_0', 'Z_1')
            # (since covariances are symmetric)
            product = [tuple(sorted(tup)) for tup in itertools.product(variable_nodes_id, variable_nodes_id)]

            # loop over tuple combinations and add variances and covariances, respectively
            for tup in product:
                # intra variances, actual variances
                if tup[0]==tup[1]:
                    var_intra_match_tup += (moment_var_match.index(tup), )

                # inter variances, actual covariances
                elif tup[0]!=tup[1]:
                    var_inter_match_tup += (moment_cov_match.index(tup), )

            # NOTE: indices can appear multiple times and numpy sum
            # will also add a certain axis multiple times accordingly
            variables_var_ind[ind_var, 0] = var_intra_match_tup
            variables_var_ind[ind_var, 1] = var_inter_match_tup

        # set the covariance indices; for given variable V1 = Z0 + ... + Zn and
        # V2 = W0 + ... + Wm it holds that
        # Cov(V1, V2) = sum(over (i, j), Cov(Zi, Wj))
        # = sum((i, j) with Zi==Wj, Var(Zi)) + sum((i, j) with Zi!=Wj, Cov(Zi, Wj))
        for ind_var, var in enumerate(variables_cov_match):
            cov_intra_match_tup = () # for sum((i, j) with Zi==Wj, Var(Zi))
            cov_inter_match_tup = () # for sum((i, j) with Zi!=Wj, Cov(Zi, Wj))

            # obtain the nodes identifier for each variable
            # here we have two variables V1 and V2 for a covariance Cov(V1, V2)
            variable_id_1 = var[0]
            variable_id_2 = var[1]
            variable_nodes_id_1 = variables_node_identifier[variable_id_1][1]
            variable_nodes_id_2 = variables_node_identifier[variable_id_2][1]

            # create a cartesian product to get tuple combinations between the
            # node identifiers between variables V1 and V2;
            # order is removed by sorted(), e.g. ('Z_1', 'Z_0') == ('Z_0', 'Z_1')
            # (since covariances are symmetric)
            product = [tuple(sorted(tup)) for tup in itertools.product(variable_nodes_id_1, variable_nodes_id_2)]

            for tup in product:
                # intra covariances, actual variances
                if tup[0]==tup[1]:
                    cov_intra_match_tup += (moment_var_match.index(tup), )

                # inter covariances, actual covariances
                elif tup[0]!=tup[1]:
                    cov_inter_match_tup += (moment_cov_match.index(tup), )

            # NOTE: indices can appear multiple times and numpy sum
            # will also add a certain axis multiple times accordingly
            variables_cov_ind[ind_var, 0] = cov_intra_match_tup
            variables_cov_ind[ind_var, 1] = cov_inter_match_tup

        return (variables_num_means, variables_mean_ind,
        variables_num_vars, variables_var_ind,
        variables_num_covs, variables_cov_ind)


    @staticmethod
    def process_initial_values_order(moment_order_hidden, initial_values_dict, net_nodes_identifier, type=None):
        """docstring for ."""

        # NOTE: there are many more types that could be imagined and implemented
        if type == 'centric_mean_only':
            # compute initial value for the hidden layer based on assumptions:
            # - user-given initial values set mean levels to start with on the main layer
            # - no variance of mean nodes, no covariance between main nodes
            # - mean levels in the hidden layer are focused solely on centric nodes
            # - no variance or covariance for hidden nodes

            # this processing type gives rise to the following initial values for hidden nodes
            # let x0 and y0 denote the respective mean levels on the main layer
            # E( X_centric ) = x0,
            # E( X_centric * (X_centric - 1) ) = x0 * x0 - x0,
            # E( X_centric * Y_centric ) = x0 * y0
            # and all remaining moments = 0

            # loop over moment_order for hidden net and find initial values as above
            init = list()

            # first moments
            for node, in moment_order_hidden[0]:
                # e.g., 'Z_0__module_1__0' or 'Z_0__centric'
                node_split = node.split('__')

                # get centric nodes and read out initial values (x0, y0, ...)
                # case: E( X_centric ) = x0
                if node_split[1] == 'centric':
                    node_id = node_split[0]
                    init_val = float(initial_values_dict[net_nodes_identifier[node_id]])

                else:
                    init_val = 0.0

                init.append(init_val)

            # second moments
            for node1, node2 in moment_order_hidden[1]:
                node1_split = node1.split('__')
                node2_split = node2.split('__')

                if node1_split[1] == 'centric' and node2_split[1] == 'centric':
                    node1_id = node1_split[0]
                    node2_id = node2_split[0]

                    # case: E( X_centric * (X_centric - 1) ) = x0 * x0 - x0
                    if node1_id == node2_id:
                        init_val1 = float(initial_values_dict[net_nodes_identifier[node1_id]])
                        init_val = init_val1 * init_val1 - init_val1

                    # case: E( X_centric * Y_centric ) = x0 * y0
                    else:
                        init_val1 = float(initial_values_dict[net_nodes_identifier[node1_id]])
                        init_val2 = float(initial_values_dict[net_nodes_identifier[node2_id]])
                        init_val = init_val1 * init_val2
                else:
                    init_val = 0.0

                init.append(init_val)

            return np.array(init)
        else:
            raise ValueError('Type \'centric_mean_only\' expected for processing initial values.')

    def setup_executable_moment_eqs_template(self, moment_eqs, use_jit=True):
        """docstring for ."""

        # print(moment_eqs)

        # NOTE:
        # 1. moment_system is a highly dynamic function (different networks have different ode equations)
        # 2. moment_system is the most evaluated function in this script, so it should be fast
        # => we therefore create the whole function with exec() methods and use just in time (jit) c compilation (use_jit=True)
        # after its creation, it serves like a static function which was specifically implemented for a given network

        # first function line
        if use_jit:
            # use jit to have a fast cython computation of the right hand side of the ode
            str_for_exec = '@jit(nopython=True)\ndef _moment_eqs_template(m, time, theta):\n'
        else:
            str_for_exec = 'def _moment_eqs_template(m, time, theta):\n'

        # ### OLD script
        # # lines with the moment equations in an odeint-suitable form, i.e. m0 = ...; m1 = ...; ...
        # for i, eq in enumerate(moment_eqs):
        #     str_for_exec += '\t' f'm{i} = ' + eq + '\n'
        #
        # # a line which returns the calculated m_i's, i.e. 'return m0, m1, m2, ...'
        # str_for_exec += '\treturn ' + ', '.join([f'm{i}' for i in range(len(moment_eqs))])
        # ###

        ### NEW script
        str_for_exec += '\treturn np.array([\n'
        for eq in moment_eqs:
            str_for_exec += '\t' + eq + ',\n'
        str_for_exec = str_for_exec[:-2] + '\n'
        str_for_exec += '\t])'
        ###

        # save this string to self
        self.moment_eqs_template_str = str_for_exec

        # this string is now executed for once and stored (via eval) as a function in this class
        # print(str_for_exec) # uncomment this for visualisation
        exec(str_for_exec)
        return eval('_moment_eqs_template')


    def set_moment_eqs_from_template_after_reset(self):
        """docstring for ."""

        # this string is now executed for once and stored (via eval) as a function in this class
        # print(str_for_exec) # uncomment this for visualisation
        exec(self.moment_eqs_template_str)

        if self.moment_system=='reset':
            self.moment_system = eval('_moment_eqs_template')
        else:
            print('Moment system was not in \'reset\' mode.')

    # the forward_pass triggers one integration of the ode system yielding a solution of the different moments over time
    # the solution depends on the initial condition (init) and the parameters (theta) of the ode system
    # afterwards the means (more precisely the expectation), variances and covariances are computed by using the appropriate moment solutions
    # NOTE: in some cases we use explicitly that np.sum([]) = 0 (on any empty array), e.g. when there are no inter variances
    def forward_pass(self, init, time_arr, theta):
        """docstring for ."""

        # number of time points
        num_time_points = len(time_arr)

        # st = time.time()

        # here python's scipy ode integrator is used
        sol = odeint(self.moment_system, init, time_arr, args=(theta, ))

        # et = time.time()
        # print('\tode_int (ms)', (et - st)*1000)

        # st = time.time()

        ### sum up hidden layer to main layer nodes
        # NOTE: the rules for summation follow preceding theoretical derivations
        # NOTE: idea: the self.mean_ind[i, 0] stuff now has to give tuples and then np.sum() over the higher dimensional array
        mean = np.zeros((self.moment_num_means, num_time_points))
        for i in range(self.moment_num_means):
            mean[i, :] = np.sum(sol[:, self.moment_mean_ind[i, 0]], axis=1)

        var_intra = np.zeros((self.moment_num_vars, num_time_points))
        var_inter = np.zeros((self.moment_num_vars, num_time_points))
        for i in range(self.moment_num_vars):
            var_intra[i, :] = np.sum(sol[:, self.moment_var_ind_intra[i, 0]] + sol[:, self.moment_var_ind_intra[i, 1]] - sol[:, self.moment_var_ind_intra[i, 1]]**2, axis=1)
            var_inter[i, :] = 2.0 * np.sum(sol[:, self.moment_var_ind_inter[i, 0]] - sol[:, self.moment_var_ind_inter[i, 1]] * sol[:, self.moment_var_ind_inter[i, 2]], axis=1)
        var = var_intra + var_inter

        cov = np.zeros((self.moment_num_covs, num_time_points))
        for i in range(self.moment_num_covs):
            cov[i, :] = np.sum(sol[:, self.moment_cov_ind[i, 0]] - sol[:, self.moment_cov_ind[i, 1]] * sol[:, self.moment_cov_ind[i, 2]], axis=1)
        ###

        ### sum up or reorder solution to obtain the simulation variables output
        # TODO: here
        variables_mean = np.zeros((self.variables_num_means, num_time_points))
        variables_var = np.zeros((self.variables_num_vars, num_time_points))
        variables_cov = np.zeros((self.variables_num_covs, num_time_points))

        for i in range(self.variables_num_means):
            variables_mean[i, :] = np.sum(mean[self.variables_mean_ind[i, 0], :], axis=0)

        for i in range(self.variables_num_vars):
            variables_var[i, :] = (np.sum(var[self.variables_var_ind[i, 0], :], axis=0) +
                                    np.sum(cov[self.variables_var_ind[i, 1], :], axis=0))

        for i in range(self.variables_num_covs):
            variables_cov[i, :] = (np.sum(var[self.variables_cov_ind[i, 0], :], axis=0) +
                                    np.sum(cov[self.variables_cov_ind[i, 1], :], axis=0))
        ###

        # et = time.time()
        # print('\tfw pass rest (ms)', (et - st)*1000)

        return variables_mean, variables_var, variables_cov

    ### helper functions for the derive_pde method
    @staticmethod
    def reac_type_to_end(z_start, z_end, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        return '{0} * ({2} - 1) * F({3})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to(z_start, z_end, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        return '{0} * (1 - {1}) * diff(F({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_end(z_start, z_end, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        return '{0} * ({2} - {1}) * diff(F({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_start_end(z_start, z_end, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        return '{0} * ({1} * {2} - {1}) * diff(F({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_start_start(z_start, z_end, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        return '{0} * ({1} * {1} - {1}) * diff(F({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_end_end(z_start, z_end, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        return '{0} * ({2} * {2} - {1}) * diff(F({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_end1_end2(z_start, z_end_1, z_end_2, rate, z_vars):
        """docstring for ."""

        # this formula is taken for granted
        # TODO: check this formula (have it on paper notes)
        return '{0} * ({2} * {3} - {1}) * diff(F({4}), {1})'.format(rate, z_start, z_end_1, z_end_2, ', '.join(z_vars))
    ###
