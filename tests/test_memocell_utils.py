
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import pytest
import memocell as me
import numpy as np

class TestUtilsModule(object):
    def test_utils_phase_type_from_erlang_1(self):
        a, S = me.utils.phase_type_from_erlang(0.2, 1)
        np.testing.assert_allclose(a, np.array([[1.]]))
        np.testing.assert_allclose(S, np.array([[-0.2]]))

    def test_utils_phase_type_from_erlang_1_pdf(self):
        a, S = me.utils.phase_type_from_erlang(0.2, 1)
        x = np.linspace(0.0, 20.0, num=11, endpoint=True)
        y = me.utils.phase_type_pdf(a, S, x)
        np.testing.assert_allclose(y, np.array([0.2       , 0.13406401, 0.08986579, 0.06023884, 0.0403793 ,
                                                   0.02706706, 0.01814359, 0.01216201, 0.00815244, 0.00546474,
                                                   0.00366313]),
                                                   rtol=1e-06, atol=1e-06)

    def test_utils_phase_type_from_erlang_5(self):
        a, S = me.utils.phase_type_from_erlang(0.2, 5)
        np.testing.assert_allclose(a, np.array([[1., 0., 0., 0., 0.]]))
        np.testing.assert_allclose(S, np.array([[-1.,  1.,  0.,  0.,  0.],
                                                 [ 0., -1.,  1.,  0.,  0.],
                                                 [ 0.,  0., -1.,  1.,  0.],
                                                 [ 0.,  0.,  0., -1.,  1.],
                                                 [ 0.,  0.,  0.,  0., -1.]]))

    def test_utils_phase_type_from_erlang_5_pdf(self):
        a, S = me.utils.phase_type_from_erlang(0.2, 5)
        x = np.linspace(0.0, 20.0, num=11, endpoint=True)
        y = me.utils.phase_type_pdf(a, S, x)
        np.testing.assert_allclose(y, np.array([0.00000000e+00, 9.02235222e-02, 1.95366815e-01, 1.33852618e-01,
                                                   5.72522885e-02, 1.89166374e-02, 5.30859947e-03, 1.33100030e-03,
                                                   3.07296050e-04, 6.66159314e-05, 1.37410241e-05]),
                                                   rtol=1e-06, atol=1e-06)

    def test_utils_phase_type_from_parallel_erlang2_exp(self):
        a, S = me.utils.phase_type_from_parallel_erlang2(0.2, 0.1, 1, 1)
        np.testing.assert_allclose(a, np.array([[1.]]))
        np.testing.assert_allclose(S, np.array([[-0.3]]))

    def test_utils_phase_type_from_parallel_erlang2_22(self):
        a, S = me.utils.phase_type_from_parallel_erlang2(0.2, 0.1, 2, 2)
        np.testing.assert_allclose(a, np.array([[1., 0., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.6,  0.4,  0.2],
                                                 [ 0. , -0.4,  0. ],
                                                 [ 0. ,  0. , -0.2]]))

    def test_utils_phase_type_from_parallel_erlang2_12(self):
        a, S = me.utils.phase_type_from_parallel_erlang2(0.2, 0.1, 1, 2)
        np.testing.assert_allclose(a, np.array([[1., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.4,  0.2],
                                                [ 0. , -0.2]]))

    def test_utils_phase_type_from_parallel_erlang2_21(self):
        a, S = me.utils.phase_type_from_parallel_erlang2(0.2, 0.1, 2, 1)
        np.testing.assert_allclose(a, np.array([[1., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.5,  0.4],
                                                [ 0. , -0.4]]))

    def test_utils_phase_type_from_parallel_erlang2(self):
        a, S = me.utils.phase_type_from_parallel_erlang2(0.2, 0.1, 5, 8)
        np.testing.assert_allclose(a, np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]))
        np.testing.assert_allclose(S, np.array([[-1.8,  1. ,  0. ,  0. ,  0. ,  0.8,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
         [ 0. , -1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0. , -1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0. , -1. ,  1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0. ,  0. , -1. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0. ,  0. ,  0. , -0.8,  0.8,  0. ,  0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.8,  0.8,  0. ,  0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.8,  0.8,  0. ,  0. ,  0. ],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.8,  0.8,  0. ,  0. ],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.8,  0.8,  0. ],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.8,  0.8],
         [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. , -0.8]]))

    def test_utils_phase_type_from_parallel_erlang2_pdf(self):
        a, S = me.utils.phase_type_from_parallel_erlang2(0.2, 0.1, 5, 8)
        x = np.linspace(0.0, 20.0, num=11, endpoint=True)
        y = me.utils.phase_type_pdf(a, S, x)
        np.testing.assert_allclose(y, np.array([0.        , 0.06827431, 0.13007684, 0.10767396, 0.07956391,
                                                   0.0543326 , 0.03245024, 0.01689615, 0.00782326, 0.00329188,
                                                   0.00128175]),
                                                   rtol=1e-06, atol=1e-06)

    def test_utils_phase_type_from_parallel_erlang3_exp(self):
        a, S = me.utils.phase_type_from_parallel_erlang3(0.2, 0.1, 0.05, 1, 1, 1)
        np.testing.assert_allclose(a, np.array([[1.]]))
        np.testing.assert_allclose(S, np.array([[-0.35]]))

    def test_utils_phase_type_from_parallel_erlang3_222(self):
        a, S = me.utils.phase_type_from_parallel_erlang3(0.2, 0.1, 0.05, 2, 2, 2)
        np.testing.assert_allclose(a, np.array([[1., 0., 0., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.7,  0.4,  0.2,  0.1],
                                                 [ 0. , -0.4,  0. ,  0. ],
                                                 [ 0. ,  0. , -0.2,  0. ],
                                                 [ 0. ,  0. ,  0. , -0.1]]))

    def test_utils_phase_type_from_parallel_erlang3_112(self):
        a, S = me.utils.phase_type_from_parallel_erlang3(0.2, 0.1, 0.05, 1, 1, 2)
        np.testing.assert_allclose(a, np.array([[1., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.4,  0.1],
                                                [ 0. , -0.1]]))

    def test_utils_phase_type_from_parallel_erlang3_221(self):
        a, S = me.utils.phase_type_from_parallel_erlang3(0.2, 0.1, 0.05, 2, 2, 1)
        np.testing.assert_allclose(a, np.array([[1., 0., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.65,  0.4 ,  0.2 ],
                                                 [ 0.  , -0.4 ,  0.  ],
                                                 [ 0.  ,  0.  , -0.2 ]]))

    def test_utils_phase_type_from_parallel_erlang3(self):
        a, S = me.utils.phase_type_from_parallel_erlang3(0.2, 0.1, 0.05, 2, 4, 3)
        np.testing.assert_allclose(a, np.array([[1., 0., 0., 0., 0., 0., 0.]]))
        np.testing.assert_allclose(S, np.array([[-0.95,  0.4 ,  0.4 ,  0.  ,  0.  ,  0.15,  0.  ],
                                                 [ 0.  , -0.4 ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
                                                 [ 0.  ,  0.  , -0.4 ,  0.4 ,  0.  ,  0.  ,  0.  ],
                                                 [ 0.  ,  0.  ,  0.  , -0.4 ,  0.4 ,  0.  ,  0.  ],
                                                 [ 0.  ,  0.  ,  0.  ,  0.  , -0.4 ,  0.  ,  0.  ],
                                                 [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.15,  0.15],
                                                 [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , -0.15]]))

    def test_utils_phase_type_from_parallel_erlang3_pdf(self):
        a, S = me.utils.phase_type_from_parallel_erlang3(0.2, 0.1, 0.05, 2, 4, 3)
        x = np.linspace(0.0, 20.0, num=11, endpoint=True)
        y = me.utils.phase_type_pdf(a, S, x)
        np.testing.assert_allclose(y, np.array([0.        , 0.10223638, 0.09316866, 0.07696751, 0.0596036 ,
                                                   0.04347545, 0.03037457, 0.02071724, 0.01403341, 0.00957331,
                                                   0.0066425 ]),
                                                   rtol=1e-06, atol=1e-06)
