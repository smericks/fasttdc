import unittest
import numpy as np
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler

class TDCSamplerTests(unittest.TestCase):

    def setUp(self):

        self.td_measured = [
            [1,np.nan,np.nan],
            [2,3,5],
            [4,np.nan,np.nan]
        ]

        self.td_cov = [
            [[0.2,0.,0.],[0.,np.nan,0.],[0.,0.,np.nan]],
            [[0.2,0.,0.],[0.,0.3,0.],[0.,0.,0.4]],
            [[0.2,0.,0.],[0.,np.nan,0.],[0.,0.,np.nan]]
        ]

        self.fpd_pred_samples = [
            [[1.2,1.1,1.3,0.9,1.3]],
            [
                [2.2,1.8,1.9,2.1,1.8],
                [2.9,2.7,3.3,3.4,3.1],
                [6.,5.1,4.,4.5,5.5]
            ],
            [[4.1,4.2,4.3,4.4,4.5]]
        ]

    def test_preprocessing(self):

        (td_measured_padded,fpd_samples_padded,td_likelihood_prefactors,
            td_likelihood_prec) = tdc_sampler.preprocess_td_measured(
            self.td_measured,self.td_cov,self.fpd_pred_samples)
        
        # TODO: ok now write some assert statements here