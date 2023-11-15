"""memocell - Bayesian inference of stochastic cellular processes with and without memory in Python."""

__version__ = '0.1.5'
__author__ = 'Maurice Langhinrichs <m.langhinrichs@icloud.com>'

# __all__ applies to importing with "from memocell import *"
__all__ = ['network', 'simulation', 'data', 'estimation',
            'selection', 'plots', 'utils']

# to be able to use "import memocell as me" and "me.Network()":
from .network import Network
from .simulation import Simulation
from .data import Data
from .estimation import Estimation
from .selection import select_models
from . import plots
from . import utils
