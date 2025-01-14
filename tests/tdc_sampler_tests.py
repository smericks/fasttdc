import unittest
import numpy as np
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler
from scipy.stats import norm,multivariate_normal,uniform

class TDCSamplerTests(unittest.TestCase):

    def setUp(self):

        self.td_measured = np.asarray([
            [100,0.,0.],
            [300,400,700],
        ])

        self.td_prec = np.asarray([
            [[1/10.,0.,0.],[0.,0.,0.],[0.,0.,0.]],
            [[1/20.,0.,0.],[0.,1/22.,0.],[0.,0.,1/24.]], #1/sigma^2
        ])

        # log( (1/(2pi)^k/2) * 1/sqrt(det(Sigma)) ) - doubles: k=1,det(Sigma)=det([sigma^2]) - quads: k=3, det(Sigma)=det(Sigma)
        self.td_prefactors = np.asarray([
            np.log((1/2*np.pi)**(0.5) / np.sqrt(10.)), # double
            np.log((1/2*np.pi)**(1.5) / np.sqrt(20.*22.*24.)) #quad
        ])

        self.fpd_pred_samples = np.asarray([
            [[1.2,0,0],
             [1.1,0,0],
             [1.3,0,0],
             [0.9,0,0],
             [1.3,0,0]],
            [
                [2.2,2.9,6.],
                [1.8,2.7,5.1],
                [1.9,3.3,4.],
                [2.1,3.4,4.5],
                [1.8,3.1,5.5]
            ],
        ])

        self.gamma_pred_samples = np.asarray([
            [2.,2.1,2.05,2.07,1.99],
            [1.8,1.75,1.9,1.85,1.9]
        ])

    def test_tdclikelihood(self):

        # make TDCLikelihood object
        z_lens = [0.5,0.6]
        z_src = [1.2,1.3]
        my_tdc = tdc_sampler.TDCLikelihood(
            self.td_measured,self.td_prec,
            self.td_prefactors,
            self.fpd_pred_samples,self.gamma_pred_samples,
            z_lens,z_src)
        my_w0wa_tdc = tdc_sampler.TDCLikelihood(self.td_measured,self.td_prec,
            self.td_prefactors,
            self.fpd_pred_samples,self.gamma_pred_samples,
            z_lens,z_src,
            cosmo_model='w0waCDM')
        
        # FUNCTION 1: td_log_likelihood_per_samp
        td_pred_samples = my_tdc.fpd_samples_padded*1.2
        likelihood_per_samp = my_tdc.td_log_likelihood_per_samp(td_pred_samples)
        # test that shape of output is (num_lenses,num_fpd_samples)
        for i,s in enumerate([2,5]):
            self.assertEqual(np.shape(likelihood_per_samp)[i],s)

        # FUNCTION 2: td_pred_from_fpd_pred(hyperparameters)
        # h0,mu(gamma_lens),sigma(gamma_lens)
        hyperparameters = [70.,0.3,2.0,0.1]
        td_predicted = my_tdc.td_pred_from_fpd_pred(hyperparameters)

        # test that shape of output is (num_lenses,num_fpd_samples,3)
        for i,s in enumerate([2,5,3]):
            self.assertEqual(np.shape(td_predicted)[i],s)

        # test that the padding of zeros is retained for doubles
        self.assertEqual(td_predicted[0][0][1],0)

        # FUNCTION 3: full_log_likelihood(hyperparameters)
        # make population mu/sigma(gamma_lens) same as the prior so the rw.
        # factor is just 1

        # TODO: update test case to include reweighting for gamma_lens!!!!
        # THIS SHOULD ALSO INCLUDE REWEIGHTING FROM INFERRED MU(GAMMA_LENS) AND SIGMA(GAMMA_LENS)
        def likelihood_test_case(my_tdc,hyperparameters):
        
            log_likelihood = my_tdc.full_log_likelihood(hyperparameters)

            # TODO: should do the math & compare with what this outputs
            td_pred_samples = my_tdc.td_pred_from_fpd_pred(hyperparameters)
            prior_gamma_model = uniform(loc=1.,scale=2.)
            proposed_gamma_model = norm(loc=hyperparameters[-2],scale=hyperparameters[-1])

            # lens 1 (the double)
            lens1_likelihood = 0
            for f in range(0,5):
                my_pred = td_pred_samples[0][f][0]
                exponent = -0.5*(my_pred - self.td_measured[0][0])**2 *self.td_prec[0][0][0]
                log_prefactor = self.td_prefactors[0]
                gamma_samp = self.gamma_pred_samples[0][f]
                rw_factor = proposed_gamma_model.logpdf(gamma_samp) - prior_gamma_model.logpdf(gamma_samp)
                lens1_log_likelihood = log_prefactor + exponent + rw_factor
                lens1_likelihood += np.exp(lens1_log_likelihood)

            lens1_likelihood = lens1_likelihood/5
                
            # lens 2 (the quad)
            lens2_likelihood = 0
            for f in range(0,5):
                my_pred = np.asarray(td_pred_samples[1][f])
                x_minus_mu = (my_pred-np.asarray(self.td_measured[1]))
                prec_mat = self.td_prec[1]
                exponent = -0.5 * np.matmul(x_minus_mu,np.matmul(prec_mat,x_minus_mu))
                log_prefactor = self.td_prefactors[1]
                gamma_samp = self.gamma_pred_samples[1][f]
                rw_factor = proposed_gamma_model.logpdf(gamma_samp) - prior_gamma_model.logpdf(gamma_samp)
                lens2_log_likelihood = log_prefactor + exponent + rw_factor
                lens2_likelihood += np.exp(lens2_log_likelihood)

            lens2_likelihood /= 5

            combined_log_likelihood = np.log(lens1_likelihood) + np.log(lens2_likelihood)
            
            self.assertAlmostEqual(combined_log_likelihood,log_likelihood)

        # LCDM case
        likelihood_test_case(my_tdc,hyperparameters = [70.,0.3,2.0,0.2])
        # w0waCDM case
        likelihood_test_case(my_w0wa_tdc,hyperparameters=[70,0.3,-1.,0.,2.0,0.2])