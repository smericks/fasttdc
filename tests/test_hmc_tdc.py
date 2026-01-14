import unittest
import numpy as np
import jax.numpy as jnp
import sys
import jax_cosmo
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/fasttdc_dev/fasttdc/')
import hmc_tdc_likelihood

class HMCLikelihoodTests(unittest.TestCase):

    def setUp(self):
        
        self.stuff = None

    def test_likelihood_eval(self):

        # start with some Gaussian time-delay measurement
        # take a ground truth lens from the catalog (a quad)
        td_truth = [-8.521770967161610,	-10.576512575533200,	-19.904090992057400]

        # 2-day meas. error, centered on ground truth
        mu_td_meas = [-8.5,	-10.5,	-19.9]
        cov_td_meas = np.diag([4,4,4])