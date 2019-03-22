
from .network import Network
from .simulation_lib.sim_gillespie import GillespieSim
from .simulation_lib.sim_moments import MomentsSim
import numpy as np

class Simulation(object):
    """docstring for ."""
    def __init__(self, network):

        # validate network input (has to be instance of Network class) and instantiate
        self.net = self.validate_network_input(network)

        # initialise instances of MomentsSim and GillespieSim simulation classes
        self.sim_moments = MomentsSim(self.net)
        self.sim_gillespie = GillespieSim(self.net)

        # instantiate object for the time values for simulations
        self.sim_time_values = None

        # # create booleans to execute simulation preparations only once
        # self.sim_moments_prep_exist = False
        # self.sim_gillespie_prep_exist = False

        # instantiate object to store results of latest moment and gillespie simulation
        self.sim_moments_res = None
        self.sim_gillespie_res = None


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

    def simulate(self, simulation_type, initial_values, theta_values, time_values, estimate_mode=False, **kwargs):
        """docstring for ."""

        # if simulations are done on its own (estimate_mode=False)
        # user input has to be checked and theta values have to be ordered
        if not estimate_mode:
            # check user input for the simulation_type
            self.validate_simulation_type_input(simulation_type)

            # check user input for the initial values
            self.validate_initial_values_input(self.net.net_nodes_identifier, simulation_type, initial_values)

            # check user input for the rate parameters (theta)
            self.validate_theta_values_input(self.net.net_rates_identifier, theta_values)

            # check user input for the time values
            self.validate_time_values_input(time_values)
            self.sim_time_values = time_values

            # read out initial_values and theta_values (dictionaries) according to node or theta order
            # initial_values_order = [initial_values[self.net.net_nodes_identifier[node_id]]
            #                         for node_id,  in self.net.net_main_node_order[0] if node_id!='Z_env']
            theta_values_order = [theta_values[self.net.net_rates_identifier[rate_id]]
                                    for rate_id in self.net.net_theta_symbolic]

        # in the estimate_mode (call of simulation class from estimation class),
        # some steps can be skipped as speed up
        else:
            # pass time_values without validation
            self.sim_time_values = time_values

            # theta_values are already ordered in this case
            theta_values_order = theta_values


        # ask for simulation_type and run respective class methods (Moments or Gillespie)
        if simulation_type=='moments':

            # the first time a Moments simulation is run, preparations have to be done
            if not self.sim_moments.moments_preparation_exists:

                # read out kwargs to see if first moments / mean only shall be computed
                # else the first and second moments will be generated
                try:
                    moment_mean_only = kwargs['moment_mean_only'] if isinstance(kwargs['moment_mean_only'], bool) else False
                except:
                    moment_mean_only = False

                # actual preparatory computations
                self.sim_moments.prepare_moment_simulation(mean_only=moment_mean_only, estimate_mode=estimate_mode)

            # run, store and return a simulation
            self.sim_moments_res = self.sim_moments.moment_simulation(initial_values, theta_values_order, self.sim_time_values)
            return self.sim_moments_res

        elif simulation_type=='gillespie':

            # the first time a Gillespie simulation is run, preparations have to be done
            if not self.sim_gillespie.gillespie_preparation_exists:
                self.sim_gillespie.prepare_gillespie_simulation()

            # NOTE: maybe add kwargs for automatic multiple simulations, N = ...

            # run, store and return a simulation
            self.sim_gillespie_res = self.sim_gillespie.gillespie_simulation(initial_values, theta_values_order, self.sim_time_values)
            return self.sim_gillespie_res


    ### plotting helper functions
    def line_evolv_mean(self, settings):
        """docstring for ."""

        moment_order_main_mean = self.sim_moments.moment_order_main[0]
        net_nodes_identifier = self.net.net_nodes_identifier

        x_arr = self.sim_time_values
        y_arr = np.zeros((len(moment_order_main_mean), len(self.sim_time_values)))
        attributes = dict()

        for i, (node_id, ) in enumerate(moment_order_main_mean):
            y_arr[i, :] = self.sim_moments_res[0][i]

            node_settings = settings[net_nodes_identifier[node_id]]
            attributes[i] = (node_settings['label'], node_settings['color'])

        return x_arr, y_arr, attributes


    def line_evolv_variance(self, settings):
        """docstring for ."""

        moment_order_main_var = [(node1_id, node2_id) for (node1_id, node2_id) in self.sim_moments.moment_order_main[1] if node1_id==node2_id]
        net_nodes_identifier = self.net.net_nodes_identifier

        x_arr = self.sim_time_values
        y_arr = np.zeros((len(moment_order_main_var), len(self.sim_time_values)))
        attributes = dict()

        for i, (node1_id, node2_id) in enumerate(moment_order_main_var):
            y_arr[i, :] = self.sim_moments_res[1][i]

            node_settings = settings[(net_nodes_identifier[node1_id], net_nodes_identifier[node2_id])]
            attributes[i] = (node_settings['label'], node_settings['color'])

        return x_arr, y_arr, attributes


    def line_evolv_covariance(self, settings):
        """docstring for ."""

        moment_order_main_cov = [(node1_id, node2_id) for (node1_id, node2_id) in self.sim_moments.moment_order_main[1] if node1_id!=node2_id]
        net_nodes_identifier = self.net.net_nodes_identifier

        x_arr = self.sim_time_values
        y_arr = np.zeros((len(moment_order_main_cov), len(self.sim_time_values)))
        attributes = dict()

        for i, (node1_id, node2_id) in enumerate(moment_order_main_cov):
            y_arr[i, :] = self.sim_moments_res[2][i]

            try:
                node_settings = settings[(net_nodes_identifier[node1_id], net_nodes_identifier[node2_id])]
            except:
                node_settings = settings[(net_nodes_identifier[node2_id], net_nodes_identifier[node1_id])]

            attributes[i] = (node_settings['label'], node_settings['color'])

        return x_arr, y_arr, attributes


    def line_evolv_counts(self, settings):
        """docstring for ."""

        gillespie_order_main = self.sim_gillespie.net_main_node_order_without_env
        net_nodes_identifier = self.net.net_nodes_identifier

        x_arr = self.sim_gillespie_res[0]
        y_arr = np.zeros((len(gillespie_order_main), len(x_arr)))
        attributes = dict()

        for i, node_id in enumerate(gillespie_order_main):
            y_arr[i, :] = self.sim_gillespie_res[1][i, :]

            node_settings = settings[net_nodes_identifier[node_id]]
            attributes[i] = (node_settings['label'], node_settings['color'])

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
    def validate_simulation_type_input(simulation_type):
        """docstring for ."""

        # check for user input if
        if simulation_type=='moments' or simulation_type=='gillespie':
            pass
        elif not isinstance(simulation_type, str):
            raise TypeError('Simulation type is not a string.')
        else:
            raise ValueError('Unknown simulation type: \'moments\' or \'gillespie\' are expected.')


    @staticmethod
    def validate_network_input(network):
        """docstring for ."""

        # check for instance of Network class
        if isinstance(network, Network):
            pass
        else:
            raise TypeError('Instance of Network class expected.')
        return network

    @staticmethod
    def validate_initial_values_input(net_nodes_identifier, simulation_type, initial_values):
        """docstring for ."""

        # check for correct user input for the initial values
        if isinstance(initial_values, dict):
            if set(net_nodes_identifier.values()) - set(['env']) == set(initial_values.keys()):
                if ((simulation_type=='gillespie' and all(isinstance(val, int) for val in initial_values.values())) or
                    (simulation_type=='moments' and all(isinstance(val, float) for val in initial_values.values()))):
                    pass
                else:
                    raise ValueError('Initial values are expected to provide integer or float values for Gillespie or Moment simulations, respectively.')
            else:
                raise ValueError('Initial values are expected to provide a set of keys identical to the nodes of the main network.')
        else:
            raise TypeError('Initial values are expected to be provided as a dictionary.')

    @staticmethod
    def validate_theta_values_input(net_rates_identifier, theta_values):
        """docstring for ."""

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
    def validate_time_values_input(time_values):
        """docstring for ."""

        # check for correct user input for the time values
        if isinstance(time_values, np.ndarray):
            if time_values.ndim == 1:
                pass
            else:
                raise ValueError('Times values are expected to be provided as a numpy array with shape \'(n, )\' with n being the number of values.')
        else:
            raise TypeError('Times values are expected to be provided as a numpy array.')
