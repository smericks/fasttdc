import unittest
import numpy as np
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler
from scipy.stats import norm,multivariate_normal

class TDCSamplerTests(unittest.TestCase):

    def setUp(self):

        self.td_measured = [
            [100,np.nan,np.nan],
            [300,400,700],
        ]

        self.td_cov = [
            [[10.,0.,0.],[0.,np.nan,0.],[0.,0.,np.nan]],
            [[20.,0.,0.],[0.,22.,0.],[0.,0.,24.]],
        ]

        self.fpd_pred_samples = [
            [[1.2,1.1,1.3,0.9,1.3]],
            [
                [2.2,1.8,1.9,2.1,1.8],
                [2.9,2.7,3.3,3.4,3.1],
                [6.,5.1,4.,4.5,5.5]
            ],
        ]

        self.gamma_pred_samples = [
            [2.,2.1,2.05,2.07,1.99],
            [1.8,1.75,1.9,1.85,1.9]
        ]

    def test_preprocessing(self):

        (td_measured_padded,fpd_samples_padded,td_likelihood_prefactors,
            td_likelihood_prec) = tdc_sampler.preprocess_td_measured(
            self.td_measured,self.td_cov,self.fpd_pred_samples)

        # assert that shape of fpd_pred_samples is (num_lenses,num_fpd_samps,3) 
        #   i.e. (2,5,3)
        fpd_shape = np.shape(fpd_samples_padded)
        for i,s in enumerate([2,5,3]):
            self.assertEqual(fpd_shape[i],s)

    def test_tdclikelihood(self):

        # make TDCLikelihood object
        z_lens = [0.5,0.6]
        z_src = [1.2,1.3]
        my_tdc = tdc_sampler.TDCLikelihood(self.td_measured,self.td_cov,
            z_lens,z_src,self.fpd_pred_samples,self.gamma_pred_samples)
        my_w0wa_tdc = tdc_sampler.TDCLikelihood(self.td_measured,self.td_cov,
            z_lens,z_src,self.fpd_pred_samples,self.gamma_pred_samples,
            cosmo_model='w0waCDM')
        
        # FUNCTION 1: td_log_likelihood_per_samp
        td_pred_samples = my_tdc.fpd_samples_padded*1.2
        likelihood_per_samp = my_tdc.td_log_likelihood_per_samp(td_pred_samples)
        # test that shape of output is (num_lenses,num_fpd_samples)
        for i,s in enumerate([2,5]):
            self.assertEqual(np.shape(likelihood_per_samp)[i],s)

        # FUNCTION 2: td_pred_from_fpd_pred(hyperparameters)
        # h0,mu(gamma_lens),sigma(gamma_lens)
        hyperparameters = [70.,2.0,0.1]
        td_predicted = my_tdc.td_pred_from_fpd_pred(hyperparameters)

        # test that shape of output is (num_lenses,num_fpd_samples,3)
        for i,s in enumerate([2,5,3]):
            self.assertEqual(np.shape(td_predicted)[i],s)

        # test that the padding of zeros is retained for doubles
        self.assertEqual(td_predicted[0][0][1],0)

        # FUNCTION 3: full_log_likelihood(hyperparameters)
        # make population mu/sigma(gamma_lens) same as the prior so the rw.
        # factor is just 1

        def likelihood_test_case(my_tdc,hyperparameters):
        
            log_likelihood = my_tdc.full_log_likelihood(hyperparameters)

            # TODO: should do the math & compare with what this outputs
            td_pred_samples = my_tdc.td_pred_from_fpd_pred(hyperparameters)
            
            # lens 1 (the double)
            lens1_likelihood = 0
            for f in range(0,5):
                my_pred = td_pred_samples[0][f][0]
                exponent = -0.5*(my_pred - self.td_measured[0][0])**2 / self.td_cov[0][0][0]
                prefactor = 1/np.sqrt(2*np.pi) * 1/np.sqrt(self.td_cov[0][0][0])
                lens1_likelihood += prefactor * np.exp(exponent)

            lens1_likelihood = lens1_likelihood/5
                
            # lens 2 (the quad)
            lens2_likelihood = 0
            for f in range(0,5):
                my_pred = np.asarray(td_pred_samples[1][f])
                x_minus_mu = (my_pred-np.asarray(self.td_measured[1]))
                prec_mat = np.linalg.inv(self.td_cov[1])
                exponent = -0.5 * np.matmul(x_minus_mu,np.matmul(prec_mat,x_minus_mu))
                prefactor = (1/(2*np.pi))**(3/2) * 1/np.sqrt(np.linalg.det(self.td_cov[1]))
                lens2_likelihood += prefactor*np.exp(exponent)

            lens2_likelihood /= 5

            combined_log_likelihood = np.log(lens1_likelihood) + np.log(lens2_likelihood)
            
            self.assertAlmostEqual(combined_log_likelihood,log_likelihood)

        # LCDM case
        likelihood_test_case(my_tdc,hyperparameters = [70.,2.0,0.2])
        # w0waCDM case
        likelihood_test_case(my_w0wa_tdc,hyperparameters=[70,-1.,0.,2.0,0.2])