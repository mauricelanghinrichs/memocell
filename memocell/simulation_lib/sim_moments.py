
"""
The simulation library contains the GillespieSim and MomentsSim class
for stochastic and moment (mean, variance, covariance) simulations, respectively.
"""

from sympy import var
from sympy import sympify
from sympy import Function
from sympy import diff

import itertools
import numpy as np
from scipy.integrate import odeint
from numba import jit

# NOTE: for some of the docstrings one could add more formula, e.g.
# - cell type stochastic processes (main layer) as sum of hidden Markov processes
# - how mean, variance and covariance are then obtained by summation
# - basic definition of the probability generating function G and z
# - why derivatives of G in z are connected to moments
# - why derivatives of the PDE for G in z provide the ODE system for the moments

class MomentsSim(object):
    """Helper class for moment (mean, variance, covariance) simulations.

    In the typical situation, use the top-level `Simulation` class with its
    main method `simulate` (`simulation_type='moments'`).
    The `MomentsSim` class and its methods are then called automatically.

    `Note`: The moment simulations in MemoCell are obtained as solutions of a
    differential equation system for the moments of all hidden Markov layer
    variables. They are the exact counterpart to mean, variance and covariance
    statistics as computed approximately from a set of stochastic simulations.
    """
    def __init__(self, net):

        # inherit the instance of the Network class
        self.net = net

        # load the net_hidden information of all indiviual edges (reactions)
        # (keys=True provides unique identifiers for parallel multiedges)
        self.net_hidden_edges = sorted(self.net.net_hidden.edges(data=True, keys=True))

        # boolean to indicate if (first) or (first and second) order moments should be derived
        # specified through upper level class Simulation
        self.sim_mean_only = None

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
        self.moment_initial_values_main = None
        self.moment_initial_values = None

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

        # instantiate also the simulation variables (indices and numbers)
        self.variables_mean_ind = None
        self.variables_var_ind = None
        self.variables_cov_ind = None
        self.variables_num_means = None
        self.variables_num_vars = None
        self.variables_num_covs  = None

    def prepare_moment_simulation(self, variables_order, variables_identifier, mean_only):
        """Prepares the moment simulation by automatic symbolic derivation of
        the differential equations for the moments on the hidden Markov layer
        and the summation indices to assemble them to mean, variance and covariance
        solutions for the main/observable layer and simulation variables. See
        the called downstream methods for more info.
        """

        # trigger the preparation if it does not exist already
        if not self.moments_preparation_exists:
            # set information for mean_only
            self.sim_mean_only = mean_only

            # derive an order of the moments
            self.moment_order_main = self.derive_moment_order_main(self.net.net_main_node_order, self.sim_mean_only)
            self.moment_order_hidden = self.derive_moment_order_hidden(self.net.net_hidden_node_order, self.sim_mean_only)

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

            # variables feature
            (self.variables_num_means, self.variables_mean_ind,
            self.variables_num_vars, self.variables_var_ind,
            self.variables_num_covs, self.variables_cov_ind) = self.get_indices_for_moment_readout(
                                                                            variables_order,
                                                                            variables_identifier,
                                                                            self.moment_order_main,
                                                                            self.net.net_nodes_identifier)

            # once this function has run preparations are done
            self.moments_preparation_exists = True

    def moment_simulation(self, theta_values_order, time_values,
                                initial_values_main, initial_values_type):
        """Top-level method in the MomentsSim class to compute moment
        (mean, variance, covariance) simulations.

        This method wraps downstream methods to update the user provided
        `theta` rate parameters and initial values, compute the moment
        simulation on the hidden layer and sum up the hidden layer moments to
        obtain mean, variance and covariance solutions on the observable/main layer
        and for the simulation variables (by `run_moment_ode_system`).

        `Note`: In the typical situation, use the top-level `Simulation` class
        with its main method `simulate`; this method is then run automatically."""

        ### TODO: maybe use getter/setter attributes or similar to only rerun these
        ### lines when initial_values_order or theta_values_order have changed

        # check if preparation was executed
        if self.moments_preparation_exists:

            # process user given initial values to hidden nodes
            # NOTE: this happens every time, if moment_initial_values don't change
            # for many moment_simulation calls one should prepare them separately
            # and then use run_moment_ode_system directly
            self.moment_initial_values = self.process_initial_values(
                                                    initial_values_main,
                                                    initial_values_type)

            # setting the numerical values of the rates
            # (as theta identifiers and in symbolic theta order)
            self.theta_numeric = theta_values_order
            ###

            # simulate the network, given initial_values, time points and parameters (theta)
            return self.run_moment_ode_system(self.moment_initial_values, time_values, theta_values_order)

    def run_moment_ode_system(self, moment_initial_values, time_values, theta_values):
        """Integrates the differential equation system for the
        hidden layer moments and sums them up to obtain mean, variance and
        covariance solutions for the simulation variables (and
        main/observable layer nodes).

        `Note`: Based on the `moment_system` and summation indices,
        obtained by automatic symbolic derivation and metaprogramming for
        any MemoCell model (e.g., as in executed in `prepare_moment_simulation`);
        see downstream methods for more info. Integration itself is done
        numerically by scipy's `odeint` method.
        """

        # run_moment_ode_system triggers one integration of the ODE system
        # yielding a solution of the different moments over time
        # the solution depends on the initial condition (moment_initial_values)
        # and the parameters (theta_values) of the ode system
        # afterwards the means (more precisely the expectation), variances and
        # covariances are computed by using the appropriate moment solutions
        # NOTE: in some cases we use explicitly that np.sum([]) = 0 (on any empty array),
        # e.g. when there are no inter variances

        # number of time points
        num_time_points = len(time_values)

        # here python's scipy ode integrator is used
        sol = odeint(self.moment_system, moment_initial_values, time_values, args=(theta_values, ))

        ### sum up hidden layer to main layer nodes
        # NOTE: the rules for summation follow preceding theoretical derivations
        # NOTE: idea: the self.mean_ind[i, 0] stuff now has to give tuples
        # and then np.sum() over the higher dimensional array
        mean = np.zeros((self.moment_num_means, num_time_points))
        for i in range(self.moment_num_means):
            mean[i, :] = np.sum(sol[:, self.moment_mean_ind[i, 0]], axis=1)

        var_intra = np.zeros((self.moment_num_vars, num_time_points))
        var_inter = np.zeros((self.moment_num_vars, num_time_points))
        for i in range(self.moment_num_vars):
            var_intra[i, :] = np.sum(sol[:, self.moment_var_ind_intra[i, 0]]
                                    + sol[:, self.moment_var_ind_intra[i, 1]]
                                    - sol[:, self.moment_var_ind_intra[i, 1]]**2, axis=1)
            var_inter[i, :] = 2.0 * np.sum(sol[:, self.moment_var_ind_inter[i, 0]]
                                    - sol[:, self.moment_var_ind_inter[i, 1]] * sol[:, self.moment_var_ind_inter[i, 2]], axis=1)
        var = var_intra + var_inter

        cov = np.zeros((self.moment_num_covs, num_time_points))
        for i in range(self.moment_num_covs):
            cov[i, :] = np.sum(sol[:, self.moment_cov_ind[i, 0]]
                            - sol[:, self.moment_cov_ind[i, 1]] * sol[:, self.moment_cov_ind[i, 2]], axis=1)
        ###

        ### sum up or reorder solution to obtain the simulation variables output
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
        return variables_mean, variables_var, variables_cov


    def process_initial_values(self, initial_values_main, initial_values_type):
        """Processes the user provided initial values for the moments (mean,
        variance, covariance for the main nodes) to obtain the initial moments
        on the hidden layer, depending on the multinomial schemes
        `initial_values_type='synchronous'` or `initial_values_type='uniform'`.

        `Note`: `Synchronous` initial distribution type means that main node
        numbers are placed into the each main node's `'centric'` hidden layer node.
        `Uniform` initial distribution types means that main node numbers
        are distributed randomly (uniform) among all its hidden layer nodes.
        For this the respective helper methods `process_initial_values_synchronous`
        or `process_initial_values_uniform` are called.

        `Note`: The distribution types have their stochastic simulation equivalents,
        see there.

        `Note`: Below we summarise the theory for distributing the hidden layer
        from main layer moments + initial value type. In MemoCell the stochastic
        processes on the main/observable layer are the sum of their stochastic
        processes on the hidden Markov layer. For each cell type :math:`i` its
        stochastic cell numbers follow
        :math:`W^{(i)}_t = \\sum_{j \\in \\{1,...,u_i\\} } W^{(i,j)}_t`,
        where :math:`u_i` is the number of all hidden variables for that cell type.
        For the initial distribution (:math:`t=0`) we have :math:`N=W^{(i)}_0`
        (random) cells to distribute for each cell type and hence sample the
        hidden variables from a multinomial distribution, i.e.
        :math:`(..., W^{(i,j)}_0,...) \\sim \\mathrm{MultiNomial}(p_1,...,p_j,...,p_{u_i}; N)`,
        where the :math:`p_j` probabilities allow to encode any hidden layer
        distribution scheme. Using theorems of conditional and total expectation,
        variance and covariance one can then obtain the following relations,
        connecting the main/observable and the hidden layer:

        - The mean of the :math:`j`-th hidden variable of cell type :math:`i` is :math:`\\mathrm{E}(W^{(i,j)}_0) = p_j\\,\\mathrm{E}(W^{(i)}_0)`,

        - the variance of the :math:`j`-th hidden variable of cell type :math:`i` is :math:`\\mathrm{Var}(W^{(i,j)}_0) = p_j (1-p_j)\\,\\mathrm{E}(W^{(i)}_0) + p_j^2 \\,\\mathrm{Var}(W^{(i)}_0)`,

        - the covariance between the :math:`j`-th and :math:`l`-th hidden variables (:math:`j≠l`) of cell type :math:`i` is :math:`\\mathrm{Cov}(W^{(i,j)}_0, W^{(i,l)}_0) = - p_j p_l \\,\\mathrm{E}(W^{(i)}_0) + p_j p_l \\,\\mathrm{Var}(W^{(i)}_0)` and

        - the covariance between the :math:`j`-th hidden variable of cell type :math:`i` and the :math:`l`-th hidden variable of cell type :math:`k` (:math:`i≠k`) is :math:`\\mathrm{Cov}(W^{(i,j)}_0, W^{(k,l)}_0) = p_j p_l \\, \\mathrm{Cov}(W^{(i)}_0, W^{(k)}_0)`.

        As MemoCell works with (mixed/factorial) moments one readily rephrases
        the above relations and obtains

        - The mean remains :math:`\\mathrm{E}(W^{(i,j)}_0) = p_j\\,\\mathrm{E}(W^{(i)}_0)`,

        - the second factorial moment is :math:`\\mathrm{E}\\big(W^{(i,j)}_0 (W^{(i,j)}_0-1)\\big) = p_j^2\\,\\big(\\mathrm{Var}(W^{(i)}_0)+ \\mathrm{E}(W^{(i)}_0)^2 - \\mathrm{E}(W^{(i)}_0)\\big)`,

        - the second mixed moment within cell type :math:`i` (:math:`j≠l`) is :math:`\\mathrm{E}(W^{(i,j)}_0 W^{(i,l)}_0) = p_j p_l\\big(\\mathrm{Var}(W^{(i)}_0)+ \\mathrm{E}(W^{(i)}_0)^2 - \\mathrm{E}(W^{(i)}_0)\\big)` and

        - the second mixed moment for different cell types :math:`i≠k` is :math:`\\mathrm{E}(W^{(i,j)}_0 W^{(k,l)}_0) = p_j p_l\\big(\\mathrm{Cov}(W^{(i)}_0, W^{(k)}_0) + \\mathrm{E}(W^{(i)}_0) \\, \\mathrm{E}(W^{(k)}_0) \\big)`.

        These ideas allow to implement any distribution scheme for the hidden layer
        from observable information and the given multinomial type (:math:`p_j`
        parameters). Specifically, MemoCell currently implements a
        uniform and a synchronous type, i.e.

        - `uniform` initial value type: :math:`p_j=1/u_i` (for each cell type :math:`i`) and

        - `synchronous` initial value type: :math:`p_1=1` (`'centric'` node), else :math:`p_j=0`, :math:`j>1` (for each cell type).
        """
        # NOTE: more notes/tests in jupyter notebook (env_initial_values) and
        # derivation in written notes in goodnotes

        # NOTE: initial values processing was updated
        # for paper version, see stalled memo_py module

        # different initial values scheme on the hidden layer for the same
        # observable layer mean, variance and covariance statistics

        # idea for the distribution schemes:
        # loop over the hidden moments, each of which has to obtain an
        # initial value; we have to following moments types (examples):
        # ('Z_0__module_1__0', ) = E(X)
        # ('Z_0__centric', 'Z_0__module_1__0') = E(X Y)
        # ('Z_0__centric', 'Z_0__centric') = E(X(X-1))
        # so we can obtain the observable nodes with a string split method

        # initial_values_main contain the same tuples, but for the original
        # observable/main node names and the following types:
        # ('X_t') = E(X)
        # ('X_t', 'Y_t') = Cov(X, Y)
        # ('X_t', 'X_t') = Var(X)

        if initial_values_type=='synchronous':
            initial_values_hidden = self.process_initial_values_synchronous(
                                        self.moment_order_hidden,
                                        initial_values_main,
                                        self.net.net_nodes_identifier)

        elif initial_values_type=='uniform':
            initial_values_hidden = self.process_initial_values_uniform(
                                        self.moment_order_hidden,
                                        initial_values_main,
                                        self.net.net_nodes_identifier,
                                        self.net.net_hidden_node_numbers)

        return initial_values_hidden


    def process_initial_values_uniform(self, moment_order_hidden,
                                initial_values_main, net_nodes_identifier,
                                net_hidden_node_numbers):
        """Helper method for `process_initial_values` (see there also);
        returns the hidden layer initial moment values under `'uniform'`
        `initial_values_type`. Order of the moments follows
        `sim.sim_moments.moment_order_hidden` in their `Z`-identifier form.
        """

        # for uniform initial values we have p_j = 1/u_i, where u_i is the
        # number of all hidden nodes for a cell type / main node i, so each
        # observable cell is distributed uniformly among its hidden variables
        # (for multinomial distribution scheme on the hidden layer)

        # we can calculate the 1/u_i fractions before starting with the loop
        # we have net_hidden_node_numbers like {'Z_env': 1, 'Z_0': 2, 'Z_1': 3}
        pj_uniform = dict()
        for node_id in net_nodes_identifier.keys():
            if node_id!='Z_env':
                pj_uniform[node_id] = 1.0/float(net_hidden_node_numbers[node_id])

        # loop over moment_order for hidden net and find initial values as above
        init = list()

        # first moments
        for node, in moment_order_hidden[0]:
            # e.g., 'Z_0__module_1__0' or 'Z_0__centric'
            # split to access main node
            node_split = node.split('__')
            node_id = node_split[0]
            # read out mean via tuple notation, e.g. with key ('X_t',) for id 'Z_0'
            node_orig = net_nodes_identifier[node_id]
            mean_i = float(initial_values_main[(node_orig, )])
            # get pj value for the respective main node / cell type
            pj = pj_uniform[node_id]
            init_val = self.compute_initial_moment_first(pj, mean_i)
            init.append(init_val)

        # second moments
        for node1, node2 in moment_order_hidden[1]:
            # split to access main nodes
            node1_split = node1.split('__')
            node2_split = node2.split('__')
            node1_id = node1_split[0] # e.g., 'Z_0'
            node2_id = node2_split[0]
            node1_orig = net_nodes_identifier[node1_id] # e.g., 'X_t'
            node2_orig = net_nodes_identifier[node2_id]
            # three cases have to be distinguished
            # 1) same main node, same hidden node -> 2nd factorial moment
            # 2) same main node, different hidden nodes -> mixed 2nd for same i
            # 3) different main nodes -> mixed 2nd for different i,k cell types

            # case 1)
            if node1==node2:
                # read out mean and variance for main node
                mean_i = float(initial_values_main[(node1_orig, )])
                var_i = float(initial_values_main[(node1_orig, node1_orig)])
                # get pj value for the respective main node / cell type
                pj = pj_uniform[node1_id]
                init_val = self.compute_initial_moment_second_factorial(
                                                        pj, mean_i, var_i)
            # case 2)
            elif node1_id==node2_id:
                # read out mean and variance for main node
                mean_i = float(initial_values_main[(node1_orig, )])
                var_i = float(initial_values_main[(node1_orig, node1_orig)])
                # get pj=pl value for the respective main node / cell type
                pj = pj_uniform[node1_id]
                init_val = self.compute_initial_moment_second_mixed_ii(
                                                    pj, pj, mean_i, var_i)
            # case 3)
            else:
                # read out means and covariance for the two main nodes
                mean_i = float(initial_values_main[(node1_orig, )])
                mean_k = float(initial_values_main[(node2_orig, )])
                # user input is checked, so one of these will work (unique)
                try:
                    cov_ik = float(initial_values_main[(node1_orig, node2_orig)])
                except:
                    cov_ik = float(initial_values_main[(node2_orig, node1_orig)])
                # get pj and pl value for the respective main nodes / cell types
                pj = pj_uniform[node1_id]
                pl = pj_uniform[node2_id]
                init_val = self.compute_initial_moment_second_mixed_ik(
                                            pj, pl, mean_i, mean_k, cov_ik)
            init.append(init_val)
        return np.array(init)


    def process_initial_values_synchronous(self, moment_order_hidden,
                                initial_values_main, net_nodes_identifier):
        """Helper method for `process_initial_values` (see there also);
        returns the hidden layer initial moment values under `'synchronous'`
        `initial_values_type`. Order of the moments follows
        `sim.sim_moments.moment_order_hidden` in their `Z`-identifier form.
        """
        # for synchronous initial values we have p_1 = 1 (centric node), else 0
        # (for multinomial distribution scheme on the hidden layer)

        # loop over moment_order for hidden net and find initial values as above
        init = list()

        # first moments
        for node, in moment_order_hidden[0]:
            # e.g., 'Z_0__module_1__0' or 'Z_0__centric'
            node_split = node.split('__')
            # get centric nodes and read out mean value for cell type i
            if node_split[1] == 'centric':
                node_id = node_split[0]
                # read out mean via tuple notation, e.g. with key ('X_t',) for id 'Z_0'
                node_orig = net_nodes_identifier[node_id]
                mean_i = float(initial_values_main[(node_orig, )])
                # pj = 1.0
                init_val = self.compute_initial_moment_first(1.0, mean_i)

            else:
                # pj = 0.0 implies compute_initial_moment_first()=0.0
                init_val = 0.0
            init.append(init_val)

        # second moments
        for node1, node2 in moment_order_hidden[1]:
            node1_split = node1.split('__')
            node2_split = node2.split('__')
            # again, only centric nodes are interesting, otherwise pj=0 anyway;
            # as there is only one centric node per cell type we never have the
            # case of a non-zero mixed second moment within the same cell type;
            # we only have cases like (Z_0__centric, Z_0__centric) or (Z_0__centric, Z_1__centric)
            if node1_split[1] == 'centric' and node2_split[1] == 'centric':
                # get main nodes / cell types
                node1_id = node1_split[0]
                node2_id = node2_split[0]

                # case: E( X_centric * (X_centric - 1) )
                if node1_id == node2_id:
                    node_orig = net_nodes_identifier[node1_id]
                    mean_i = float(initial_values_main[(node_orig, )])
                    var_i = float(initial_values_main[(node_orig, node_orig)])
                    init_val = self.compute_initial_moment_second_factorial(1.0, mean_i, var_i)

                # case: E( X_centric * Y_centric )
                else:
                    node1_orig = net_nodes_identifier[node1_id]
                    node2_orig = net_nodes_identifier[node2_id]
                    mean_i = float(initial_values_main[(node1_orig, )])
                    mean_k = float(initial_values_main[(node2_orig, )])
                    # user input is checked, so one of these will work (unique)
                    try:
                        cov_ik = float(initial_values_main[(node1_orig, node2_orig)])
                    except:
                        cov_ik = float(initial_values_main[(node2_orig, node1_orig)])
                    init_val = self.compute_initial_moment_second_mixed_ik(
                                                1.0, 1.0, mean_i, mean_k, cov_ik)
            else:
                init_val = 0.0
            init.append(init_val)

        return np.array(init)

    @staticmethod
    def compute_initial_moment_first(p_j, mean_i):
        """Helper method for `process_initial_values` (see there and related);
        computes mean of a hidden variable with multinomial parameter `p_j`
        and main/observable mean `mean_i` for cell type `i`."""
        return p_j * mean_i

    @staticmethod
    def compute_initial_moment_second_factorial(p_j, mean_i, var_i):
        """Helper method for `process_initial_values` (see there and related);
        computes second factorial moment of a hidden variable with multinomial
        parameter `p_j` and main/observable mean `mean_i` and variance `var_i`
        for cell type `i`."""
        return p_j * p_j * (var_i + mean_i * mean_i - mean_i)

    @staticmethod
    def compute_initial_moment_second_mixed_ii(p_j, p_l, mean_i, var_i):
        """Helper method for `process_initial_values` (see there and related);
        computes second mixed moment of hidden variables with multinomial
        parameters `p_j`, `p_l` and main/observable mean `mean_i` and variance `var_i`
        of the same cell type `i`."""
        return p_j * p_l * (var_i + mean_i * mean_i - mean_i)

    @staticmethod
    def compute_initial_moment_second_mixed_ik(p_j, p_l, mean_i, mean_k, cov_ik):
        """Helper method for `process_initial_values` (see there and related);
        computes second mixed moment of hidden variables with multinomial
        parameters `p_j`, `p_l` and main/observable means `mean_i`, `mean_k`
        and covariance `cov_ik` for different cell types `i, k`."""
        return p_j * p_l * (cov_ik + mean_i * mean_k)

    @staticmethod
    def derive_moment_order_main(node_order, mean_only):
        """Derives the order of the moments for the main/observable nodes in
        their `Z`-identifier form. Contains two lists, with the first moments
        (means) and second moments (for variance, covariance), respectively;
        the second moments are left out if `mean_only=True`.

        `Note`: Based on `net.net_main_node_order`, with the difference that the
        environmental node is removed for the moments. Hence original names are also
        available via `sim.net.net_nodes_identifier`.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available via
        `sim_moments.moment_order_main`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> sim.sim_moments.moment_order_main
        [[('Z_0',), ('Z_1',)], [('Z_0', 'Z_0'), ('Z_0', 'Z_1'), ('Z_1', 'Z_1')]]
        """

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
        """Derives the order of the moments for the hidden nodes in
        their `Z`-identifier form. Contains two lists, with the first moments
        (means) and second moments (for variance, covariance), respectively;
        the second moments are left out if `mean_only=True`.

        `Note`: Based on `net.net_hidden_node_order`, with the difference that the
        environmental node is removed for the moments. Hence original names are also
        available via `sim.net.net_nodes_identifier`.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available via
        `sim_moments.moment_order_hidden`.

        `Note`: The order for the hidden moments defines the order of the initial
        values (`moment_initial_values`) and the differential equation system
        (`moment_eqs` and `moment_system`). The tuples below correspond to
        means `E(X)` (e.g., `('Z_0__centric',)`), second factorial moments
        `E(X(X-1))` (e.g., `('Z_0__centric', 'Z_0__centric')`) and second mixed
        moments `E(XY)` (e.g., `('Z_0__centric', 'Z_1__centric')`),
        respectively.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> sim.sim_moments.moment_order_hidden
        [[('Z_0__centric',), ('Z_1__centric',), ('Z_1__module_1__0',)],
         [('Z_0__centric', 'Z_0__centric'),
          ('Z_0__centric', 'Z_1__centric'),
          ('Z_0__centric', 'Z_1__module_1__0'),
          ('Z_1__centric', 'Z_1__centric'),
          ('Z_1__centric', 'Z_1__module_1__0'),
          ('Z_1__module_1__0', 'Z_1__module_1__0')]]
        """

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
        """Derives the partial differential equation (PDE) for the
        probability generating function `G`, providing a complete description
        of the stochastic process on the hidden Markov layer.

        `Note`: This method goes over all edges (`net_edges`) to accumulate
        the overall PDE from the single-reaction building blocks (see
        helper methods below). The PDE description is equivalent to
        (and can be derived from) the description in terms of the master
        equation. Taking derivatives for the auxiliary `z`-variables and applying
        the limit operator provide the differential equation system for the moments
        (see `derive_moment_eqs` method).

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available via
        `sim_moments.moment_pde`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> sim.sim_moments.moment_pde
        '1.0 * theta_0_q * (z_1__centric_q - z_0__centric_q) * diff(G(z_0__centric_q, z_1__centric_q), z_0__centric_q)'
        """

        # subsequently, add parts to the pde
        pde = ''

        for edge in net_edges:
            # read out the auxiliary variables for the start and end node each reaction
            # if a node is the environmental node ('Z_env__centric'), a constant is introduced (=1.0)
            z_start_node = z_aux_vars_dict[edge[0]] if edge[0]!='Z_env__centric' else '1.0'
            z_node_end = z_aux_vars_dict[edge[1]] if edge[1]!='Z_env__centric' else '1.0'

            # read out reaction type and reaction rate (symbolic form, accelerated by step size)
            reac_type = edge[3]['edge_type']
            # example for reaction_rate: '3.0 * theta_2_q' (if module has theta rate 'theta_2' and three reaction steps)
            reac_rate = edge[3]['edge_rate_symbol_identifier'].replace(edge[3]['module_rate_symbol_identifier'],
                                                                theta_repl_dict[edge[3]['module_rate_symbol_identifier']])

            # for the special case of an edge type of 'S -> E1 + E2' capture both end nodes
            # this edge type occurs for the last reaction of a 'S -> S + E' module
            if reac_type == 'S -> E1 + E2':
                # the end node which is the start (centric) node of the module
                z_node_end_1 = z_aux_vars_dict[edge[3]['edge_centric_start_end_identifier'][0]]

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
        """Derives the ordinary differential equation (ODE) system for the moments
        on the hidden Markov layer in its symbolic form.

        `Note`: This is applied theory surrounding the probability generating
        function `G` and Markov jump processes described by a PDE in `G` (or
        equivalent a master equation). Derivatives of the PDE (as in `moment_pde`)
        in the auxiliary variables `z` and application of the limit `z→1` lead to
        linear differential equations for the moments (mean/first moment and second
        factorial and mixed moments). Also, this ODE system is closed for the
        linear reaction types available in MemoCell, i.e. the resulting equations
        are exact. These operations are automatically conducted
        for any MemoCell model, making use of `sympy`; downstream, they are
        processed to a callable class method (`moment_system`), available for
        numerical integration (`run_moment_ode_system` and top-level
        `moment_simulation`).

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available via
        `sim_moments.moment_eqs`; the order of the equations corresponds to
        `sim_moments.moment_order_hidden`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> # theta rate parameters and moment vector m
        >>> sim.sim_moments.moment_eqs
        ['-1.0*m[0]*theta[0]',
         '1.0*m[0]*theta[0]',
         '-2.0*m[2]*theta[0]',
         '1.0*m[2]*theta[0] - 1.0*m[3]*theta[0]',
         '2.0*m[3]*theta[0]']
        """

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
            string_deriv = f'Derivative(G({z_vars_str}), {z_var})'
            string_subs = f'm_{count_i}_q'
            replace_tuples.append((string_deriv, string_subs))

            count_i += 1

        for z_var1, z_var2 in moment_order_2nd_vars:
            if z_var1==z_var2:
                string_deriv = f'Derivative(G({z_vars_str}), ({z_var1}, 2))'
            else:
                # the following sorted function is needed since auxiliary vars (with
                # additional 'q') can have a different tuple order than the tuple
                # order in moment_order_hidden (e.g. if one hidden node has index
                # >= 10 (two digits)), see jupyter notebook for sympy bug
                z_var1, z_var2 = tuple(sorted([z_var1, z_var2]))
                string_deriv = f'Derivative(G({z_vars_str}), {z_var1}, {z_var2})'
            string_subs = f'm_{count_i}_q'
            replace_tuples.append((string_deriv, string_subs))

            count_i += 1

        # replace remaining z variables by 1.0
        for z_var in moment_aux_vars:
            replace_tuples.append((z_var, '1.0'))

        # replace higher moment derivatives by a constant (these terms cancel anyway)
        # replace second order derivatives when there are no demanded second moments
        if len(moment_order_2nd_vars) == 0:
            inner_G = ', '.join(len(moment_aux_vars)*['1.0'])
            replace_tuples.append((f'Derivative(G({inner_G}), 1.0, 1.0)', 'const'))
            replace_tuples.append((f'Derivative(G({inner_G}), (1.0, 2))', 'const'))
        # else replace third order derivatives
        else:
            inner_G = ', '.join(len(moment_aux_vars)*['1.0'])

            replace_tuples.append((f'Derivative(G({inner_G}), 1.0, 1.0, 1.0)', 'const'))
            replace_tuples.append((f'Derivative(G({inner_G}), (1.0, 2), 1.0)', 'const'))
            replace_tuples.append((f'Derivative(G({inner_G}), 1.0, (1.0, 2))', 'const'))
            replace_tuples.append((f'Derivative(G({inner_G}), (1.0, 3))', 'const'))

        # replace the plain probability generating function by one (since probabilities sum up to one)
        inner_G = ', '.join(len(moment_aux_vars)*['1.0'])
        replace_tuples.append((f'G({inner_G})', '1.0'))

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
        """Creates array objects with indices for the hidden layer moments
        (first, second mixed and factorial) that allow to sum them up
        for solutions of mean, variance and covariance of the main/observable
        nodes.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available at `moment_mean_ind`,
        `moment_var_ind_intra`, `moment_var_ind_inter` and `moment_cov_ind` and
        used in `run_moment_ode_system` and top-level `moment_simulation`;
        index values correspond to `moment_order_hidden`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> sim.sim_moments.moment_mean_ind
        array([[(0,)],
        [(1, 2)]], dtype=object)
        >>> sim.sim_moments.moment_var_ind_intra
        array([[(3,), (0,)],
        [(6, 8), (1, 2)]], dtype=object)
        >>> sim.sim_moments.moment_var_ind_inter
        array([[(), (), ()],
        [(7,), (1,), (2,)]], dtype=object)
        >>> sim.sim_moments.moment_cov_ind
        array([[(4, 5), (0, 0), (1, 2)]], dtype=object)
        """

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
        """Creates array objects with indices for the mean, variance and
        covariance solutions on the main/observable layer that allow to sum them up
        for mean, variance and covariance solutions for the simulation variables.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available at `variables_mean_ind`,
        `variables_var_ind` and `variables_cov_ind` and
        used in `run_moment_ode_system` and top-level `moment_simulation`.

        Examples
        --------
        >>> # with a memocell simulation instance sim
        >>> sim.sim_moments.variables_mean_ind
        array([[(0,)],
        [(1,)]], dtype=object)
        >>> sim.sim_moments.variables_var_ind
        array([[(0,), ()],
        [(1,), ()]], dtype=object)
        >>> sim.sim_moments.variables_cov_ind
        array([[(), (0,)]], dtype=object)
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

    def setup_executable_moment_eqs_template(self, moment_eqs, use_jit=True):
        """
        Converts the user-specific `moment_eqs` (list of str) to a callable class
        method, applying the metaprogramming principle via `eval()` and `exec()`
        methods.

        `Note`: This method is automatically run during `sim.simulate` in
        `simulation_type='moments'` and during `estimate` and `select_models`
        methods. The output is typically available via the `moment_system` method.

        `Note`: Numba's `@jit` (just-in-time compilation) decorator is added (`default`)
        to allow fast computation of the differential moment equation system
        during simulation and estimation runs; the first `moment_system` call
        might then take a bit longer due to the compilation.
        """

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
        """Reevaluates the differential moment equation system when it is in `'reset'`
        mode. The `moment_system` is typically overwritten with `'reset'` when
        the `selection` module was used to be able to save and load objects
        with the `pickle` package.
        """

        # this string is now executed for once and stored (via eval) as a function in this class
        # print(str_for_exec) # uncomment this for visualisation
        exec(self.moment_eqs_template_str)

        if self.moment_system=='reset':
            self.moment_system = eval('_moment_eqs_template')
        else:
            print('Moment system was not in \'reset\' mode.')

    ### helper functions for the derive_pde method
    @staticmethod
    def reac_type_to_end(z_start, z_end, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'-> E'` reaction (e.g., cell influx or birth)
        on the hidden Markov layer.

        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`∅ → W^{(i,j)}` with the
        hidden variable :math:`W^{(i,j)}` in state :math:`w^{(i,j)}`
        and auxiliary variable :math:`z_{(i,j)}` we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, t) = \\lambda \\, p(w^{(i,j)}-1, t)
        - \\lambda \\, p(w^{(i,j)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (z_{(i,j)} - 1) \\, G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the transition rate.
        """

        # this formula is taken for granted
        return '{0} * ({2} - 1) * G({3})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to(z_start, z_end, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'S ->'` reaction (e.g., efflux or cell death)
        on the hidden Markov layer.

        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`W^{(i,j)} → ∅` with the
        hidden variable :math:`W^{(i,j)}` in state :math:`w^{(i,j)}`
        and auxiliary variable :math:`z_{(i,j)}` we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, t) = \\lambda \\, (w^{(i,j)}+1) \\, p(w^{(i,j)}+1, t)
        - \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (1 - z_{(i,j)}) \\, \\partial_{z_{(i,j)}} G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the single-cell transition rate.
        """

        # this formula is taken for granted
        return '{0} * (1 - {1}) * diff(G({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_end(z_start, z_end, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'S -> E'` reaction (e.g., cell
        differentiation or hidden transitions)
        on the hidden Markov layer.


        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`W^{(i,j)} → W^{(k,l)}` with
        different hidden variables :math:`W^{(i,j)}`, :math:`W^{(k,l)}` in states
        :math:`w^{(i,j)}`, :math:`w^{(k,l)}` and auxiliary variables :math:`z_{(i,j)}`,
        :math:`z_{(k,l)}`, respectively, we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, w^{(k,l)}, t) =
        \\lambda \\, (w^{(i,j)}+1) \\, p(w^{(i,j)}+1, w^{(k,l)}-1, t)
        - \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, w^{(k,l)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (z_{(k,l)} - z_{(i,j)}) \\, \\partial_{z_{(i,j)}} G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the single-cell transition rate. The reaction
        is hidden for :math:`i=k` (same cell type) and realises a differentiation event
        for :math:`i≠k` (different cell types).
        """

        # this formula is taken for granted
        return '{0} * ({2} - {1}) * diff(G({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_start_end(z_start, z_end, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'S -> S + E'` reaction (e.g., asymmetric
        cell division) on the hidden Markov layer.


        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`W^{(i,j)} → W^{(i,j)} + W^{(k,l)}` with
        different hidden variables :math:`W^{(i,j)}`, :math:`W^{(k,l)}` in states
        :math:`w^{(i,j)}`, :math:`w^{(k,l)}` and auxiliary variables :math:`z_{(i,j)}`,
        :math:`z_{(k,l)}`, respectively, we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, w^{(k,l)}, t) =
        \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, w^{(k,l)}-1, t)
        - \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, w^{(k,l)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (z_{(k,l)}\\,z_{(i,j)} - z_{(i,j)}) \\, \\partial_{z_{(i,j)}} G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the single-cell transition rate.
        """

        # this formula is taken for granted
        return '{0} * ({1} * {2} - {1}) * diff(G({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_start_start(z_start, z_end, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'S -> S + S'` reaction (e.g., symmetric
        self-renewing cell division) on the hidden Markov layer.


        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`W^{(i,j)} → W^{(i,j)} + W^{(i,j)}` with the
        hidden variable :math:`W^{(i,j)}` in state
        :math:`w^{(i,j)}` and auxiliary variable :math:`z_{(i,j)}`
        we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, t) =
        \\lambda \\, (w^{(i,j)}-1) \\, p(w^{(i,j)}-1, t)
        - \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (z_{(i,j)}^2 - z_{(i,j)}) \\, \\partial_{z_{(i,j)}} G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the single-cell transition rate.
        """

        # this formula is taken for granted
        return '{0} * ({1} * {1} - {1}) * diff(G({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_end_end(z_start, z_end, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'S -> E + E'` reaction (e.g., symmetric
        differentiating cell division) on the hidden Markov layer.


        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`W^{(i,j)} → W^{(k,l)} + W^{(k,l)}` with
        different hidden variables :math:`W^{(i,j)}`, :math:`W^{(k,l)}` in states
        :math:`w^{(i,j)}`, :math:`w^{(k,l)}` and auxiliary variables :math:`z_{(i,j)}`,
        :math:`z_{(k,l)}`, respectively, we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, w^{(k,l)}, t) =
        \\lambda \\, (w^{(i,j)}+1) \\, p(w^{(i,j)}+1, w^{(k,l)}-2, t)
        - \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, w^{(k,l)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (z_{(k,l)}^2 - z_{(i,j)}) \\, \\partial_{z_{(i,j)}} G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the single-cell transition rate.
        """

        # this formula is taken for granted
        return '{0} * ({2} * {2} - {1}) * diff(G({3}), {1})'.format(rate, z_start, z_end, ', '.join(z_vars))

    @staticmethod
    def reac_type_start_to_end1_end2(z_start, z_end_1, z_end_2, rate, z_vars):
        """Returns the PDE building block of the probability generating
        function :math:`G` for a `'S -> E1 + E2'` reaction (e.g., asymmetric
        differentiating cell division) on the hidden Markov layer.


        `Note`: The PDE building block can be derived from (and is equivalent
        to) the master equation for this hidden layer reaction.
        For the reaction :math:`W^{(i,j)} → W^{(k,l)} + W^{(r,s)}` with
        different hidden variables :math:`W^{(i,j)}`, :math:`W^{(k,l)}`, :math:`W^{(r,s)}` in states
        :math:`w^{(i,j)}`, :math:`w^{(k,l)}`, :math:`w^{(r,s)}` and auxiliary variables :math:`z_{(i,j)}`,
        :math:`z_{(k,l)}`, :math:`z_{(r,s)}`, respectively, we have the master equation

        :math:`\\partial_t \\, p(w^{(i,j)}, w^{(k,l)}, w^{(r,s)}, t) =
        \\lambda \\, (w^{(i,j)}+1) \\, p(w^{(i,j)}+1, w^{(k,l)}-1, w^{(r,s)}-1, t)
        - \\lambda \\, w^{(i,j)} \\, p(w^{(i,j)}, w^{(k,l)}, w^{(r,s)}, t)`

        which is equivalent to the PDE for :math:`G`

        :math:`\\partial_t \\, G(z,t) = \\lambda \\, (z_{(k,l)} \\, z_{(r,s)} - z_{(i,j)}) \\, \\partial_{z_{(i,j)}} G(z,t)`,

        where :math:`z` is representative for all auxiliary variables and
        :math:`\\lambda` is the single-cell transition rate.
        """

        # this formula is taken for granted
        # TODO: check this formula (have it on paper notes)
        return '{0} * ({2} * {3} - {1}) * diff(G({4}), {1})'.format(rate, z_start, z_end_1, z_end_2, ', '.join(z_vars))
    ###
