import unittest
import numpy as np
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler
from scipy.stats import norm,multivariate_normal,uniform

class TDCSamplerTests(unittest.TestCase):

    def setUp(self):

        self.td_measured_dbls = np.asarray([
            [100],
        ])
        self.td_measured_quads = np.asarray([
            [300,400,700],
        ])

        self.td_prec_dbls = np.asarray([
            [[1/10.]], #1/sigma^2
        ])
        self.td_prec_quads = np.asarray([
            [[1/20.,0.,0.],[0.,1/22.,0.],[0.,0.,1/24.]], #1/sigma^2
        ])

        self.fpd_pred_samples_dbls = np.asarray([
            [[1.2],
             [1.1],
             [1.3],
             [0.9],
             [1.3]]
        ])
        self.fpd_pred_samples_quads = np.asarray([
            [
                [2.2,2.9,6.],
                [1.8,2.7,5.1],
                [1.9,3.3,4.],
                [2.1,3.4,4.5],
                [1.8,3.1,5.5]
            ]
        ])

        self.gamma_pred_samples_dbls = np.asarray([
            [2.,2.1,2.05,2.07,1.99]
        ])

        self.gamma_pred_samples_quads = np.asarray([
            [1.8,1.75,1.9,1.85,1.9]
        ])

        # kinematics
        self.sigma_v_measured = np.asarray([
            [130.]
        ])

        self.sigma_v_likelihood_prec = np.asarray([
            [[1/25.]]
        ])

        self.kin_pred_samples = np.asarray([
            [[120.],[140.],[150.],[100.],[110.]]
        ])

    def test_tdclikelihood(self):

        # make TDCLikelihood object
        z_lens = [0.5,0.6]
        z_src = [1.2,1.3]
        dbl_lklhd = tdc_sampler.TDCLikelihood(
            self.td_measured_dbls,self.td_prec_dbls,
            self.fpd_pred_samples_dbls,self.gamma_pred_samples_dbls,
            z_lens=[0.5],z_src=[1.2])
        quad_lklhd = tdc_sampler.TDCLikelihood(
            self.td_measured_quads,self.td_prec_quads,
            self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
            z_lens=[0.6],z_src=[1.3])
        dbl_lklhd_w0wa = tdc_sampler.TDCLikelihood(
            self.td_measured_dbls,self.td_prec_dbls,
            self.fpd_pred_samples_dbls,self.gamma_pred_samples_dbls,
            z_lens=[0.5],z_src=[1.2],cosmo_model='w0waCDM')
        quad_lklhd_w0wa = tdc_sampler.TDCLikelihood(
            self.td_measured_quads,self.td_prec_quads,
            self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
            z_lens=[0.6],z_src=[1.3],cosmo_model='w0waCDM')
        
        # FUNCTION 1: td_log_likelihood_per_samp
        td_pred_samples = dbl_lklhd.fpd_samples*1.2
        likelihood_per_samp = dbl_lklhd.td_log_likelihood_per_samp(td_pred_samples)
        # test that shape of output is (num_lenses,num_fpd_samples)
        for i,s in enumerate([1,5]):
            self.assertEqual(np.shape(likelihood_per_samp)[i],s)

        # FUNCTION 2: td_pred_from_fpd_pred(hyperparameters)
        # h0,Omega_M,mu(gamma_lens),sigma(gamma_lens)
        hyperparameters = [70.,0.3,2.0,0.1]
        proposed_cosmo = dbl_lklhd.construct_proposed_cosmo(hyperparameters)
        td_predicted_dbls = dbl_lklhd.td_pred_from_fpd_pred(proposed_cosmo)

        # test that shape of output is (num_lenses,num_fpd_samples,1) for dbls
        for i,s in enumerate([1,5,1]):
            self.assertEqual(np.shape(td_predicted_dbls)[i],s)

        # check for last dim. (num_lenses,num_fpd_samples,3) for quads
        td_predicted_quads = quad_lklhd.td_pred_from_fpd_pred(proposed_cosmo)
        self.assertEqual(np.shape(td_predicted_quads)[2],3)


        # FUNCTION 3: full_log_likelihood(hyperparameters)

        # TODO: update test case to include reweighting for gamma_lens!!!!
        # THIS SHOULD ALSO INCLUDE REWEIGHTING FROM INFERRED MU(GAMMA_LENS) AND SIGMA(GAMMA_LENS)
        def likelihood_test_case(dbl_lklhd,quad_lklhd,hyperparameters):
            """
            Args:
                dbl_lklhd (tdc_sampler.TDCLikelihood):
                quad_lklhd (tdc_sampler.TDCLikelihood):
            """

            # NOTE: this prior model matches the default option in TDCLikelihood
            prior_gamma_model = uniform(loc=1.,scale=2.)
            proposed_gamma_model = norm(loc=hyperparameters[-2],scale=hyperparameters[-1])

            # lens 1 (the double)
            lens1_computed_ll = dbl_lklhd.full_log_likelihood(hyperparameters)

            proposed_cosmo = dbl_lklhd.construct_proposed_cosmo(hyperparameters)
            td_pred_samples = dbl_lklhd.td_pred_from_fpd_pred(proposed_cosmo)

            # lens 1 (the double)
            lens1_likelihood = 0
            for f in range(0,5):
                my_pred = td_pred_samples[0][f][0]
                exponent = (-0.5*(my_pred - self.td_measured_dbls[0][0])**2 * 
                    self.td_prec_dbls[0][0][0])
                log_prefactor = (np.log((1/(2*np.pi))**(0.5) / 
                    np.sqrt(10.)) )# NOTE: hardcoded
                gamma_samp = self.gamma_pred_samples_dbls[0][f]
                rw_factor = (proposed_gamma_model.logpdf(gamma_samp) - 
                    prior_gamma_model.logpdf(gamma_samp))
                lens1_log_likelihood = log_prefactor + exponent + rw_factor
                lens1_likelihood += np.exp(lens1_log_likelihood)

            lens1_likelihood /= 5

            self.assertAlmostEqual(lens1_computed_ll,np.log(lens1_likelihood))
                
            # lens 2 (the quad)
            lens2_computed_ll = quad_lklhd.full_log_likelihood(hyperparameters)

            # proposed_cosmo is still the same here...
            td_pred_samples = quad_lklhd.td_pred_from_fpd_pred(proposed_cosmo)

            lens2_likelihood = 0
            for f in range(0,5):
                my_pred = np.asarray(td_pred_samples[0][f])
                x_minus_mu = (my_pred-np.asarray(self.td_measured_quads[0]))
                prec_mat = self.td_prec_quads[0]
                exponent = -0.5 * np.matmul(x_minus_mu,np.matmul(prec_mat,x_minus_mu))
                log_prefactor = (np.log((1/(2*np.pi))**(1.5) / 
                    np.sqrt(20.*22.*24.))) # NOTE: hardcoded
                gamma_samp = self.gamma_pred_samples_quads[0][f]
                rw_factor = proposed_gamma_model.logpdf(gamma_samp) - prior_gamma_model.logpdf(gamma_samp)
                lens2_log_likelihood = log_prefactor + exponent + rw_factor
                lens2_likelihood += np.exp(lens2_log_likelihood)

            lens2_likelihood /= 5

            self.assertAlmostEqual(lens2_computed_ll,np.log(lens2_likelihood))

        # LCDM case
        likelihood_test_case(dbl_lklhd,quad_lklhd,hyperparameters = [70.,0.3,2.0,0.2])
        # w0waCDM case
        likelihood_test_case(dbl_lklhd_w0wa,quad_lklhd_w0wa,hyperparameters=[70,0.3,-1.,0.,2.0,0.2])


    def test_tdckinlikelihood(self):

        # initialize likelihood object
        quad_kin_lklhd = tdc_sampler.TDCKinLikelihood(
                    self.td_measured_quads,self.td_prec_quads,
                    self.sigma_v_measured,self.sigma_v_likelihood_prec,
                    self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
                    self.kin_pred_samples,
                    z_lens=[0.6],z_src=[1.3])
        
        # h0,Omega_M,mu(gamma_lens),sigma(gamma_lens)
        hyperparameters = [70.,0.3,2.0,0.1]
        # NOTE: this prior model matches the default option in TDCLikelihood
        prior_gamma_model = uniform(loc=1.,scale=2.)
        proposed_gamma_model = norm(loc=hyperparameters[-2],scale=hyperparameters[-1])

        # lens 1 (the quad)
        lens1_computed_ll = quad_kin_lklhd.full_log_likelihood(hyperparameters)

        # get model predictions
        proposed_cosmo = quad_kin_lklhd.construct_proposed_cosmo(hyperparameters)
        td_pred_samples = quad_kin_lklhd.td_pred_from_fpd_pred(proposed_cosmo)
        sigma_v_pred_samples = quad_kin_lklhd.sigma_v_pred_from_kin_pred(proposed_cosmo)

        # lens 1 (the quad)
        lens1_likelihood = 0
        for f in range(0,5):
            # time delay likelihood
            my_pred = np.asarray(td_pred_samples[0][f])
            x_minus_mu = (my_pred-np.asarray(self.td_measured_quads[0]))
            prec_mat = self.td_prec_quads[0]
            exponent_td = -0.5 * np.matmul(x_minus_mu,np.matmul(prec_mat,x_minus_mu))
            log_prefactor_td = (np.log((1/(2*np.pi))**(1.5) / 
                np.sqrt(20.*22.*24.))) # NOTE: hardcoded
            td_ll = exponent_td + log_prefactor_td

            # kinematic likelihood 
            sigma_v_pred = sigma_v_pred_samples[0][f][0]
            exponent_kin = (-0.5*(sigma_v_pred - self.sigma_v_measured[0][0])**2 * 
                    self.sigma_v_likelihood_prec[0][0][0])
            log_prefactor_kin = (np.log((1/(2*np.pi))**(0.5) / 
                np.sqrt(25.)) ) # NOTE: hardcoded

            kin_ll = exponent_kin + log_prefactor_kin
            # gamma_lens reweighting
            gamma_samp = self.gamma_pred_samples_quads[0][f]
            rw_factor = (proposed_gamma_model.logpdf(gamma_samp) - 
                prior_gamma_model.logpdf(gamma_samp))
            
            # the combination
            lens1_log_likelihood = td_ll + kin_ll + rw_factor
            lens1_likelihood += np.exp(lens1_log_likelihood)

        lens1_likelihood /= 5

        self.assertAlmostEqual(lens1_computed_ll,np.log(lens1_likelihood))
        

    def test_fast_tdc(self):

        # make TDCLikelihood object
        z_lens = [0.5,0.6]
        z_src = [1.2,1.3]
        my_tdc = tdc_sampler.TDCLikelihood(
            self.td_measured_dbls,self.td_prec_dbls,
            self.fpd_pred_samples_dbls,self.gamma_pred_samples_dbls,
            z_lens=[0.5],z_src=[1.2])
        
        def check_chain_moves(mcmc_chain):
            # loop over params, check that chain is moving
            for param_idx in range(0,mcmc_chain.shape[2]):
                # isolate a single walker
                single_chain = mcmc_chain[0,:,param_idx]
                # check if chain has moved away from starting point
                diff_from_initial = single_chain[1:] - single_chain[0]
                self.assertNotAlmostEqual(0.,np.sum(diff_from_initial) )


        # check if it works, test_chain dims are: (walkers,samples,params)
        test_chain = tdc_sampler.fast_TDC([my_tdc],num_emcee_samps=5,
            n_walkers=20)
        check_chain_moves(test_chain)


        # Combine doubles and quads, check it works...
        quads_tdc_lhood = tdc_sampler.TDCLikelihood(
            self.td_measured_quads,self.td_prec_quads,
            self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
            z_lens=[0.6],z_src=[1.3])
        
        # check if it works, test_chain dims are: (walkers,samples,params)
        test_chain = tdc_sampler.fast_TDC([my_tdc,quads_tdc_lhood],
            num_emcee_samps=5,n_walkers=20)
        check_chain_moves(test_chain)
