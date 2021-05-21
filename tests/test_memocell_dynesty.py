
# NOTE: in this script, more expensive tests are established
# that can be skipped if a quick test run is desired only

# for package testing with pytest call
# in upper directory "$ python setup.py pytest"
# or in this directory "$ py.test test_memocell_[...].py"
# or after pip installation $py.test --pyargs memocell$

import numpy as np
import pytest
from dynesty import NestedSampler
from dynesty.utils import resample_equal

class TestDynestyEvidence(object):
    @pytest.fixture(scope='class')
    def sampler_setup(self):
        # see jupyter notebook memopy_testing_evidence_analytical (env_working)
        # for background, reference and visualisations of this test
        sigma = 0.001

        def loglikelihood(theta):
            r2 = np.sum(theta**2)
            logl = - r2 / (2 * sigma**2)
            return logl

        def logprior_transform_2_sphere(hypercube):
            # transforms a 2-dimenional vector of uniform[0, 1] random variables
            # into a two-dimensional uniform vector of the 2-sphere interior
            u = hypercube[0]
            v = hypercube[1]

            r = u**0.5  # sqrt function
            theta = 2* np.pi * v

            x = r*np.cos(theta)
            y = r*np.sin(theta)
            return np.array([x, y])

        # theoretical results
        # logevid_analytical = -13.122363377404328
        # logl_max_analytical = 0.0
        # theta means = 0.0, 0.0

        nlive = 1000      # number of live points
        bound = 'multi'   # use MutliNest algorithm for bounds
        ndims = 2         # two parameters
        sample = 'unif'   # unif or random walk sampling or ...
        tol = 0.01        # the stopping criterion

        sampler = NestedSampler(loglikelihood, logprior_transform_2_sphere, ndims,
                                bound=bound, sample=sample, nlive=nlive)
        sampler.run_nested(dlogz=tol, print_progress=False) # don't output progress bar
        return sampler

    def test_evidence_calc_analytical(self, sampler_setup):
        logevid_analytical = -13.122363377404328
        res = sampler_setup.results # get results dictionary from sampler
        logZdynesty = res.logz[-1]        # value of logZ
        logZerrdynesty = res.logzerr[-1]
        np.testing.assert_allclose(logevid_analytical, logZdynesty,
                                    rtol=1.0, atol=1.0)

    def test_max_likelihood_analytical(self, sampler_setup):
        logl_max_analytical = 0.0
        res = sampler_setup.results # get results dictionary from sampler
        logl_max = res.logl[-1]
        np.testing.assert_allclose(logl_max_analytical, logl_max,
                                    rtol=1.0, atol=0.001)

    def test_max_likelihood_max_equal_last_point(self, sampler_setup):
        res = sampler_setup.results # get results dictionary from sampler
        logl_max = res.logl[-1]
        np.testing.assert_allclose(np.max(res.logl), logl_max)

    def test_posterior_samples(self, sampler_setup):
        res = sampler_setup.results # get results dictionary from sampler
        # draw posterior samples
        weights = np.exp(res['logwt'] - res['logz'][-1])
        samples_dynesty = resample_equal(res.samples, weights)
        np.testing.assert_allclose(np.array([0.0, 0.0]),
                        np.mean(samples_dynesty, axis=0),
                        rtol=1.0, atol=0.0001)
