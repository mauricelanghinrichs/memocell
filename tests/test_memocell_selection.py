
# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import pytest
import memocell as me
import numpy as np

class TestSelectionModule(object):
    def test_bayes_factor_calculation(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        res = me.selection.compute_model_bayes_factors_from_log_evidences(logevids)
        sol = np.array([  1.34985881,  13.46373804,   1.        , 403.42879349])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_uniform_prior_default(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids)
        sol = np.array([0.40758705, 0.04086421, 0.55018497, 0.00136377])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_uniform_prior_default_sum(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids)
        np.testing.assert_allclose(1.0, np.sum(res), rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_uniform_prior_explicit(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        mprior = np.array([0.25, 0.25, 0.25, 0.25])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
        sol = np.array([0.40758705, 0.04086421, 0.55018497, 0.00136377])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_non_uniform_prior(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        mprior = np.array([0.3, 0.3, 0.2, 0.2])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
        sol = np.array([0.49940188, 0.05006945, 0.44941468, 0.00111399])
        np.testing.assert_allclose(sol, res, rtol=1e-06, atol=1e-06)

    def test_model_probs_calculation_non_uniform_prior_sum(self):
        logevids = np.array([4.1, 1.8, 4.4, -1.6])
        mprior = np.array([0.3, 0.3, 0.2, 0.2])
        res = me.selection.compute_model_probabilities_from_log_evidences(logevids, mprior=mprior)
        np.testing.assert_allclose(1.0, np.sum(res), rtol=1e-06, atol=1e-06)
