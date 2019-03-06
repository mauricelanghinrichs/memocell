
from .network import Network
from .simulation_lib.sim_gillespie import GillespieSim
from .simulation_lib.sim_moments import MomentsSim

class Simulation(object):
    """docstring for ."""
    def __init__(self, network):

        # validate network input (has to be instance of Network class) and instantiate
        self.net = self.validate_network_input(network)

        # initialise instances of MomentsSim and GillespieSim simulation classes
        self.sim_moments = MomentsSim(self.net)
        self.sim_gillespie = GillespieSim(self.net)

        # create booleans to execute simulation preparations only once
        self.sim_moments_prep_exist = False
        self.sim_gillespie_prep_exist = False


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

    def simulate(self, simulation_type, initial_values, theta_values, time_values, **kwargs):
        """docstring for ."""

        # check user input for the simulation_type
        self.validate_simulation_type_input(simulation_type)

        # read out initial_values and theta_values (dictionaries) according to node or theta order
        # initial_values_order = [initial_values[self.net.net_nodes_identifier[node_id]]
        #                         for node_id,  in self.net.net_main_node_order[0] if node_id!='Z_env']
        theta_values_order = [theta_values[self.net.net_rates_identifier[rate_id]]
                                for rate_id in self.net.net_theta_symbolic]

        # ask for simulation_type and run respective class methods (Moments or Gillespie)
        if simulation_type=='moments':

            # the first time a Moments simulation is run, preparations have to be done
            if not self.sim_moments_prep_exist:

                # read out kwargs to see if first moments / mean only shall be computed
                # else the first and second moments will be generated
                try:
                    self.sim_moments.moment_mean_only = kwargs['moment_mean_only'] if isinstance(kwargs['moment_mean_only'], bool) else False
                except:
                    self.sim_moments.moment_mean_only = False

                # actual preparatory computations
                self.sim_moments.prepare_moment_simulation()
                self.sim_moments_prep_exist = True



            # run and return a simulation
            return self.sim_moments.moment_simulation(initial_values, theta_values_order, time_values)

        elif simulation_type=='gillespie':

            # the first time a Gillespie simulation is run, preparations have to be done
            if not self.sim_gillespie_prep_exist:
                self.sim_gillespie.prepare_gillespie_simulation()
                self.sim_gillespie_prep_exist = True

            # NOTE: maybe add kwargs for automatic multiple simulations, N = ...

            # run and return a simulation
            return self.sim_gillespie.gillespie_simulation(initial_values, theta_values_order, time_values)


    #


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
    def validate_rate_values_input():
        """docstring for ."""
        # TODO: implement
        pass
