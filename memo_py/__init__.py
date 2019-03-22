"""memo_py - An exact modelling framework for stochastic processes with and
without memory in Python."""

__version__ = '0.1.0'
__author__ = 'Maurice Langhinrichs <m.langhinrichs@icloud.com>'

# __all__ applies to importing with "from memo_py import *"
__all__ = ['network', 'simulation', 'data', 'estimation', 'selection', 'plots']

# to be able to use "import memo_py as me" and "me.Network()":
from .network import Network
from .simulation import Simulation
from .data import Data
from .estimation import Estimation
from .selection import select_models
from .plots import Plots
