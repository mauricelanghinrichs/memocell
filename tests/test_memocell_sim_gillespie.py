
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import pytest
import memocell as me
import numpy as np

class TestGillespieSimClass(object):
    def test_exact_interpolation(self):
        time_array_gill = np.array([0.0, 0.12, 4.67, 8.01, 10.00])
        nodes_array_gill = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        simulation = [time_array_gill, nodes_array_gill]

        time_array_explicit = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
        res = [np.array([ 0.,  2.,  4.,  6.,  8., 10.]), np.array([[1., 2., 2., 3., 3., 5.]])]

        np.testing.assert_allclose(res[0], me.simulation_lib.sim_gillespie.GillespieSim.exact_interpolation(simulation, time_array_explicit)[0])
        np.testing.assert_allclose(res[1], me.simulation_lib.sim_gillespie.GillespieSim.exact_interpolation(simulation, time_array_explicit)[1])
