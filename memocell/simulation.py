
"""
copied here from estimation docs (still to check):
think about sparse simulation variabesl of a network (does this work already?
do I have to check for something?)
"""

from .network import Network
from .simulation_lib.sim_gillespie import GillespieSim
from .simulation_lib.sim_moments import MomentsSim
import numpy as np

class Simulation(object):
    """docstring for ."""
    def __init__(self, network):

        # validate network input (has to be instance of Network class) and instantiate
        self.net = self._validate_network_input(network)

        # initialise general simulation settings
        self.sim_type = None
        self.sim_initial_values_type = None

        # initialise instances of MomentsSim and GillespieSim simulation classes
        self.sim_moments = MomentsSim(self.net)
        self.sim_gillespie = GillespieSim(self.net)

        # instantiate object for the time values for simulations
        self.sim_time_values = None

        # instantiate objects for the (current) simulation output variables
        self.sim_variables = None
        self.sim_variables_identifier = None
        self.sim_variables_order = None

        # instantiate object to store results of latest moment and gillespie simulation
        self.sim_moments_res = None
        self.sim_gillespie_res = None

        # instantiate object to indicate mean only mode in case of moment simulations
        self.sim_mean_only = None


# TODO: need input
# - net_topol
# - reac_rates
# - init_cond
# - time_arr
#
# need processed objects (here or in network class)
# - main_node_order (now part of the network class)
# - node_order (now part of the network class)
# - reac_modules (probably best to use net_hidden ?)
# - sym_params (in the right order with values)
# - init_cond (in the right order with values)

    def simulate(self, simulation_type, simulation_variables,
                        theta_values, time_values,
                        initial_values_type, initial_gillespie=None,
                        initial_moments=None, sim_mean_only=False):
        """docstring for .

        sim_mean_only=False
        is only relevant for moments, will be set to true for gilespie to have
        only simple variables list
        """

        # prepare simulation with some validation and setting symbolic
        # information for sim_mean_only and simulation_variables
        # and trigger preparations in downstream moment or gillespie objects
        self.prepare_simulation(simulation_type, simulation_variables,
                                    initial_values_type, initial_gillespie,
                                    initial_moments, sim_mean_only)

        # validate and set time_values (at self.sim_time_values)
        self.prepare_time_values(time_values)

        # validate and read out theta parameters from dict to ordered array
        theta_values_order = self.prepare_theta_values(theta_values)

        # ask for simulation_type and run respective class methods (Moments or Gillespie)
        if simulation_type=='moments':
            # run, store and return a simulation
            self.sim_moments_res = self.sim_moments.moment_simulation(
                                            theta_values_order, self.sim_time_values,
                                            self.sim_moments.moment_initial_values_main,
                                            self.sim_initial_values_type)
            return self.sim_moments_res
        elif simulation_type=='gillespie':
            # NOTE: maybe add kwargs for automatic multiple simulations, N = ...
            # run, store and return a simulation
            self.sim_gillespie_res = self.sim_gillespie.gillespie_simulation(
                                            theta_values_order, self.sim_time_values,
                                            self.sim_gillespie.sim_gill_initial_values_main,
                                            self.sim_initial_values_type)
            return self.sim_gillespie_res

    def prepare_simulation(self, simulation_type, simulation_variables,
                                initial_values_type, initial_gillespie,
                                initial_moments, sim_mean_only):
        """docstring for ."""
        ### user input is checked and theta values (dict) are ordered
        # check user input for the simulation_type
        self._validate_simulation_type_input(simulation_type)
        self.sim_type = simulation_type

        # check user input for the initial values
        self._validate_initial_values_input(simulation_type, initial_values_type,
                                        initial_gillespie, initial_moments)
        self.sim_initial_values_type = initial_values_type

        if simulation_type=='moments':
            self._validate_initial_values_input_moments(initial_moments, sim_mean_only,
                                                self.net.net_main_node_order, self.net.net_nodes_identifier)
            self.sim_moments.moment_initial_values_main = initial_moments
        elif simulation_type=='gillespie':
            self._validate_initial_values_input_gillespie(initial_gillespie, self.net.net_nodes_identifier)
            self.sim_gillespie.sim_gill_initial_values_main = initial_gillespie

        ### some symbolic preparations
        # read out mean only mode (default is False meaning that first
        # and second moments are used then)
        self.sim_mean_only = self.process_sim_mean_only(self.sim_type, sim_mean_only)

        # prepare simulation variables (this method acts only upon change
        # of previous simulation variables (or first time) and will in this
        # case also reset moment and gillespie preparations to force re-run)
        self.prepare_simulation_variables(simulation_variables)

        ### depending on simulation type, trigger also preparations in the
        ### downstream MomentsSim or GillespieSim objects
        if simulation_type=='moments':
            # the first time a Moments simulation is run
            # (or in case of a reset), preparations are done
            self.sim_moments.prepare_moment_simulation(self.sim_variables_order,
                                                    self.sim_variables_identifier,
                                                    self.sim_mean_only)
        elif simulation_type=='gillespie':
            # the first time a Gillespie simulation is run
            # (or in case of a reset), preparations are done
            self.sim_gillespie.prepare_gillespie_simulation(self.sim_variables_order,
                                                            self.sim_variables_identifier)

    def prepare_time_values(self, time_values):
        """docstring for ."""
        # check user input for the time values
        self._validate_time_values_input(time_values)
        self.sim_time_values = time_values

    def prepare_theta_values(self, theta_values_dict):
        """docstring for ."""
        # check user input for the rate parameters (theta)
        self._validate_theta_values_input(self.net.net_rates_identifier, theta_values_dict)

        # read out theta_values (dictionaries) according to theta order
        return self.process_theta_values(theta_values_dict,
                                self.net.net_rates_identifier, self.net.net_theta_symbolic)

    @staticmethod
    def process_theta_values(theta_values_dict, net_rates_identifier, net_theta_symbolic):
        """docstring for ."""
        # read out theta_values_dict (dictionary) to an ordered array of
        # parameter value according to order in net_theta_symbolic
        return np.array([theta_values_dict[net_rates_identifier[rate_id]]
                                for rate_id in net_theta_symbolic])

    @staticmethod
    def process_sim_mean_only(simulation_type, mean_only):
        """docstring for ."""
        # is only relevant for moment simulation;
        # for gillespie only variables_order[0] is used thus
        # sim_mean_only has no effect but just for aesthetics we set
        # variables_order to only have normal variables (=
        # first moments) in this case
        return True if simulation_type=='gillespie' else mean_only

    def prepare_simulation_variables(self, simulation_variables):
        """docstring for ."""
        # upon change of simulation output variables, or when provided for the first time,
        # set up object for the order and identification of the simulation variables
        if self.sim_variables!=simulation_variables:

            # validate the user input information
            self.sim_variables = self._validate_simulation_variables_input(simulation_variables, self.net.net_nodes_identifier)

            # create unique identifiers for the simulation variables ('V_<integer>')
            self.sim_variables_identifier = self.create_variables_identifiers(self.sim_variables)

            # create an order of variable identifiers for a sequence (list index=0)
            # and sequence of unique pairs (list index=1)
            self.sim_variables_order = self.create_variables_order(self.sim_variables_identifier, self.sim_mean_only)

            # reset preparations for moments or gillespie simulation to force a re-run
            # upon change of simulation variables
            self.sim_moments.moments_preparation_exists = False
            self.sim_gillespie.gillespie_preparation_exists = False

    @staticmethod
    def create_variables_identifiers(variables):
        """docstring for ."""

        # get all user provided output simulation variables as (key, value) tuples
        # remove duplicates and sort tuples
        variables_sorted = sorted(set([(var, variables[var]) for var in variables.keys()]))

        # create a list of neutral rate identifiers 'V_<integer>' (V for variable)
        ident_variables_list = [f'V_{i}' for i in range(len(variables_sorted))]

        # return a dictionary with user provided variable tuples as values for variable identifiers as keys
        return dict(zip(ident_variables_list, variables_sorted))

    @staticmethod
    def create_variables_order(variables_identifier, mean_only):
        """docstring for ."""

        variable_order = list()

        # the variables (neutral identifiers) are sorted() to have the same
        # deterministic sequence of nodes for any identical set of variables
        variables_identifiers = sorted(variables_identifier.keys())

        # an order for each variable for a simulation output
        # e.g., used to define the order of the first moments
        variable_order.append([(var_id, ) for var_id in variables_identifiers])

        if not mean_only:
            # an order for all pairs of variables (symmetric pairs are only added once)
            # ['V_0', 'V_1'] would give [('V_0', 'V_0'), ('V_0', 'V_1'), ('V_1', 'V_1')]
            # e.g., used to define the order of the second moments
            variable_order.append([(variables_identifiers[i], variables_identifiers[j])
                                    for i in range(len(variables_identifiers))
                                    for j in range(len(variables_identifiers))
                                    if i<=j])
        else:
            variable_order.append([])

        return variable_order


    ### plotting helper functions
    def _line_evolv_mean(self, settings):
        """Private plotting helper method."""

        sim_variables_order_mean = self.sim_variables_order[0]
        sim_variables_identifier = self.sim_variables_identifier

        x_arr = self.sim_time_values
        y_arr = np.zeros((len(sim_variables_order_mean), len(self.sim_time_values)))
        attributes = dict()

        for i, (variable_id, ) in enumerate(sim_variables_order_mean):
            y_arr[i, :] = self.sim_moments_res[0][i]

            variable_settings = settings[sim_variables_identifier[variable_id][0]]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes


    def _line_evolv_variance(self, settings):
        """Private plotting helper method."""

        sim_variables_order_var = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.sim_variables_order[1] if variable1_id==variable2_id]
        sim_variables_identifier = self.sim_variables_identifier

        x_arr = self.sim_time_values
        y_arr = np.zeros((len(sim_variables_order_var), len(self.sim_time_values)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_var):
            y_arr[i, :] = self.sim_moments_res[1][i]

            variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes


    def _line_evolv_covariance(self, settings):
        """Private plotting helper method."""

        sim_variables_order_cov = [(variable1_id, variable2_id) for (variable1_id, variable2_id) in self.sim_variables_order[1] if variable1_id!=variable2_id]
        sim_variables_identifier = self.sim_variables_identifier

        x_arr = self.sim_time_values
        y_arr = np.zeros((len(sim_variables_order_cov), len(self.sim_time_values)))
        attributes = dict()

        for i, (variable1_id, variable2_id) in enumerate(sim_variables_order_cov):
            y_arr[i, :] = self.sim_moments_res[2][i]

            try:
                variable_settings = settings[(sim_variables_identifier[variable1_id][0], sim_variables_identifier[variable2_id][0])]
            except:
                variable_settings = settings[(sim_variables_identifier[variable2_id][0], sim_variables_identifier[variable1_id][0])]

            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes


    def _line_evolv_counts(self, settings):
        """Private plotting helper method."""

        sim_variables_order_main = [var for (var, ) in self.sim_variables_order[0]]
        sim_variables_identifier = self.sim_variables_identifier

        x_arr = self.sim_gillespie_res[0]
        y_arr = np.zeros((len(sim_variables_order_main), len(x_arr)))
        attributes = dict()

        for i, node_id in enumerate(sim_variables_order_main):
            y_arr[i, :] = self.sim_gillespie_res[1][i, :]

            variable_settings = settings[sim_variables_identifier[node_id][0]]
            attributes[i] = (variable_settings['label'], variable_settings['color'])

        return x_arr, y_arr, attributes
    ###

    # # NOTE: put in init?
    # def parametrise(self, rate_values, validate_rate_values=False):
    #     """docstring for ."""
    #
    #
    #
    #     # validate user input for parameters if desired
    #     if validate_rate_values:
    #         # TODO: implement
    #         self.validate_rate_values_input(parameters)
    #
    #
    #     pass
    #
    # # TODO: run_once or something like: (for subclasses)
    # # class S:
    # # def __init__(self):        # Initializer function for instance members
    # #     self.flag = True
    # #
    # # def myMethod(self):        # Actual method to be called
    # #     if self.flag:
    # #         ....
    # #         ....
    # #         self.flag = False
    #
    # @run_once
    # def initialise_moment_simulation(self):
    #     """docstring for ."""
    #
    #     # TODO: implement
    #
    #     # instantiate an instance of MomentsSim (Moments Simulation) class
    #     moments_sim = MomentsSim('hello')
    #
    #     print(moments_sim.arg)
    #
    #     self.moment_equations_exist = True
    #
    # @run_once
    # def initialise_gillespie_simulation(self):
    #     """docstring for ."""
    #
    #     # TODO: implement
    #
    #     self.gillespie_equations_exist = True
    #
    # # function decorator to handle that decorated functions are only executed once
    # # taken from stackoverflow "Efficient way of having a function only executed once in a loop"
    # # https://stackoverflow.com/questions/4103773/efficient-way-of-having-a-function-only-execute-once-in-a-loop
    # def run_once(f):
    #     def wrapper(*args, **kwargs):
    #         if not wrapper.has_run:
    #             wrapper.has_run = True
    #             return f(*args, **kwargs)
    #     wrapper.has_run = False
    #     return wrapper
    #
    # # TODO: delete this method if no longer used
    # def validate_and_instantiate_simulation_type(self, simulation_type):
    #     """docstring for ."""
    #
    #     # check for simulation type and if preparations for this simulation type are done
    #     if simulation_type=='moments' and self.moment_equations_exist:
    #         pass
    #     elif simulation_type=='moments' and not self.moment_equations_exist:
    #         self.initialise_moment_simulation()
    #     elif simulation_type=='gillespie' and self.gillespie_equations_exist:
    #         pass
    #     elif simulation_type=='gillespie' and not self.gillespie_equations_exist:
    #         self.initialise_gillespie_simulation()
    #     # if none of those above cases, check for user input
    #     elif not isinstance(simulation_type, str):
    #         raise TypeError('Simulation type is not a string.')
    #     else:
    #         raise ValueError('Unknown simulation type: \'moments\' or \'gillespie\' are expected.')

    @staticmethod
    def _validate_simulation_variables_input(variables, net_nodes_identifier):
        """Private validation method."""

        # validate variables user input
        if not isinstance(variables, dict):
            raise TypeError('A dictionary for the variables is expected.')

        if not all(isinstance(key, str) for key in variables.keys()):
            raise TypeError('Strings are expected as keys of the variables dictionary.')

        if not all(isinstance(val, tuple) for val in variables.values()):
            raise TypeError('Tuples are expected as values of the variables dictionary.')

        if not all(isinstance(string, str) for tup in variables.values() for string in tup):
            raise TypeError('Tuples of strings are expected as values of the variables dictionary.')

        net_nodes_without_env = set(net_nodes_identifier.values()) - set(['env'])
        if not all(set(val).issubset(net_nodes_without_env) for val in variables.values()):
            raise ValueError('Strings in the tuple as value of the variables dictionary have to be nodes of the network.')

        # check for uniqueness of simulation variables (addition Aug 2020)
        sim_variables = list(variables.keys())
        if len(sim_variables) > len(set(sim_variables)):
            raise ValueError('Simulation variables have to be unique.')
        return variables


    @staticmethod
    def _validate_simulation_type_input(simulation_type):
        """Private validation method."""

        # check for user input if
        if simulation_type=='moments' or simulation_type=='gillespie':
            pass
        elif not isinstance(simulation_type, str):
            raise TypeError('Simulation type is not a string.')
        else:
            raise ValueError('Unknown simulation type: \'moments\' or \'gillespie\' are expected.')


    @staticmethod
    def _validate_network_input(network):
        """Private validation method."""

        # check for instance of Network class
        if isinstance(network, Network):
            pass
        else:
            raise TypeError('Instance of Network class expected.')
        return network

    @staticmethod
    def _validate_initial_values_input(simulation_type, initial_values_type,
                                    initial_gillespie, initial_moments):
        """Private validation method."""
        # NOTE: some more checks specific to gillespie and moment sim below

        # initial_values_type should be either 'synchronous' or 'uniform'
        if initial_values_type=='synchronous' or initial_values_type=='uniform':
            pass
        elif not isinstance(initial_values_type, str):
            raise TypeError('Initial values type is not a string.')
        else:
            raise ValueError('Unknown initial values type: \'synchronous\' or \'uniform\' are expected.')

        # if simulation_type is gillespie, check if not None, but dict
        if simulation_type=='gillespie':
            if not isinstance(initial_gillespie, dict):
                raise TypeError('Dictionary expected for initial values (gillespie).')

        if simulation_type=='moments':
            if not isinstance(initial_moments, dict):
                raise TypeError('Dictionary expected for initial values (moments).')

    @staticmethod
    def _validate_initial_values_input_gillespie(initial_values, net_nodes_identifier):
        """Private validation method."""

        # check for correct user input for the initial values
        # (check for uniqueness not needed as dict have unique keys)
        if set(net_nodes_identifier.values()) - set(['env']) == set(initial_values.keys()):
            if all(isinstance(val, (np.integer, int)) for val in initial_values.values()):
                pass
            else:
                raise ValueError('Initial values are expected to provide integers for Gillespie simulations.')
        else:
            raise ValueError('Initial values are expected to provide a set of keys identical to the nodes of the main network.')

    @staticmethod
    def _validate_initial_values_input_moments(initial_moments, sim_mean_only,
                                        net_main_node_order, net_nodes_identifier):
        """Private validation method."""

        # check for correct user input for the initial values
        if all(isinstance(val, (np.floating, float)) for val in initial_moments.values()):
            pass
        else:
            raise ValueError('Initial values are expected to provide floats for moment simulations.')

        # 'moment_order' is not yet available at this validation check, so
        # we basically repeat the order computation here for the check

        # access all identifiers for mean or var/cov and convert to original names
        moments_mean = [(net_nodes_identifier[node],) for node, in net_main_node_order[0] if not node=='Z_env']
        moments_var_cov = [tuple(sorted((net_nodes_identifier[node_1], net_nodes_identifier[node_2])))
                            for node_1, node_2 in net_main_node_order[1]
                            if not (node_1=='Z_env' or node_2=='Z_env')]

        # also sort initial values input
        # (check for uniqueness not needed as dict have unique keys)
        initial_moments_keys_sorted = [tuple(sorted(m)) for m in initial_moments.keys()]

        # now it can happen that one specified a covariance with ('A', 'B') and ('B', A)
        if not len(initial_moments_keys_sorted)==len(set(initial_moments_keys_sorted)):
            raise ValueError('Non-unique initial values.')

        # depending on mean-only mode check if all initial moments (exp, var, cov)
        # are given as keys
        if sim_mean_only:
            if not set(moments_mean) == set(initial_moments_keys_sorted):
                raise ValueError('Initial values have to match with main network nodes and given mean-only mode.')
        else:
            if not set(moments_mean + moments_var_cov) == set(initial_moments_keys_sorted):
                raise ValueError('Initial values have to match with main network nodes and given mean-only mode.')

    @staticmethod
    def _validate_theta_values_input(net_rates_identifier, theta_values):
        """Private validation method."""

        # check for correct user input for the rate parameters (theta)
        if isinstance(theta_values, dict):
            if set(net_rates_identifier.values()) == set(theta_values.keys()):
                if all(isinstance(val, float) for val in theta_values.values()):
                    pass
                else:
                    raise ValueError('Rate parameters (theta) are expected to provide float values.')
            else:
                raise ValueError('Rate parameters (theta) are expected to provide a set of keys identical to the symbolic network parameters (theta).')
        else:
            raise TypeError('Rate parameters (theta) are expected to be provided as a dictionary.')

    @staticmethod
    def _validate_time_values_input(time_values):
        """Private validation method."""

        # check for correct user input for the time values
        if isinstance(time_values, np.ndarray):
            if time_values.ndim == 1:
                pass
            else:
                raise ValueError('Times values are expected to be provided as a numpy array with shape \'(n, )\' with n being the number of values.')
        else:
            raise TypeError('Times values are expected to be provided as a numpy array.')
