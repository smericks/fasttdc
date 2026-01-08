import unittest
import numpy as np
import jax.numpy as jnp
import sys
import jax_cosmo
from scipy.stats import norm,multivariate_normal,uniform
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/fasttdc_dev/fasttdc/')
import tdc_sampler
import Utils.tdc_utils as tdc_utils

class TDCSamplerTests(unittest.TestCase):

    def setUp(self):

        self.data_vector_dict_dbls = {
            'td_measured':np.asarray([[100],]),
            'td_likelihood_prec':np.asarray([[[1/10.]], ]), #1/sigma^2
            'td_likelihood_prefactors':None,
            'fpd_samples':np.asarray([
                [[1.2],
                [1.1],
                [1.3],
                [0.9],
                [1.3]]
            ]),
            'lens_param_samples':np.asarray([
                [ # theta_E, gamma1/2, gamma_lens, e1/2
                    [1., 0., 0., 2., 0., 0.],
                    [1., 0., 0., 2.1, 0., 0.],
                    [1., 0., 0., 2.05, 0., 0.],
                    [1., 0., 0., 2.07, 0., 0.],
                    [1., 0., 0., 1.99, 0., 0.]
                ]
            ]),
            'z_lens':np.asarray([0.5]),
            'z_src':np.asarray([1.2])
        }

        num_td = 1
        self.data_vector_dict_dbls['td_likelihood_prefactors'] = np.log( 
            (1/(2*np.pi)**(num_td/2)) / np.sqrt(np.linalg.det(
                np.linalg.inv(self.data_vector_dict_dbls['td_likelihood_prec']))) )

        self.data_vector_dict_quads = {
            'td_measured':np.asarray([[300,400,700],]),
            'td_likelihood_prec': np.asarray([
                [[1/20.,0.,0.],[0.,1/22.,0.],[0.,0.,1/24.]],]), #1/sigma^2
            'td_likelihood_prefactors':None,
            'fpd_samples':np.asarray([
                [
                    [2.2,2.9,6.],
                    [1.8,2.7,5.1],
                    [1.9,3.3,4.],
                    [2.1,3.4,4.5],
                    [1.8,3.1,5.5]
                ]
            ]),
            'lens_param_samples':np.asarray([
                [ # theta_E, gamma1/2, gamma_lens, e1/2
                    [1., 0., 0., 1.8, 0., 0.],
                    [1., 0., 0., 1.75, 0., 0.],
                    [1., 0., 0., 1.9, 0., 0.],
                    [1., 0., 0., 1.85, 0., 0.],
                    [1., 0., 0., 1.9, 0., 0.]
                ]
            ]),
            'z_lens':np.asarray([0.6]),
            'z_src':np.asarray([1.3])
        }


        num_td = 3
        self.data_vector_dict_quads['td_likelihood_prefactors'] = np.log( 
            (1/(2*np.pi)**(num_td/2)) / np.sqrt(np.linalg.det(
                np.linalg.inv(self.data_vector_dict_quads['td_likelihood_prec']))) )

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

        # ifu kinematics
        self.ifu_sigma_v_measured = np.asarray([
            [150., 155., 160.]
        ])

        self.ifu_sigma_v_likelihood_prec = np.asarray([
            [[1/25., 0., 0.],
             [0., 1/25., 0.],
             [0., 0., 1/25.]
            ]
        ])

        self.ifu_sigma_v_pred_samples = np.asarray([
            [[160., 160., 160.],
             [150., 140., 170.],
             [151., 156., 161.],
             [149., 154., 159.],
             [158., 155., 170.]]
        ])

        self.beta_ani_samples = np.asarray([
            [0.1,-0.1,0.05,-0.05,0.]
        ])

        self.kappa_ext_samples = np.asarray([
            [0.,0.01,-0.01,0.02,-0.02]
        ])



    def test_tdclikelihood(self):

        # make TDCLikelihood object
        z_lens = [0.5,0.6]
        z_src = [1.2,1.3]
        # TODO: fix all of these tests for the new formatting
        dbl_lklhd = tdc_sampler.TDCLikelihood(
            fpd_sample_shape=np.shape(self.data_vector_dict_dbls['fpd_samples']),
            cosmo_model='LCDM',use_gamma_info=False) # TODO: gamma_info on or off?
        
        quad_lklhd = tdc_sampler.TDCLikelihood(
            fpd_sample_shape=np.shape(self.data_vector_dict_quads['fpd_samples']),
            cosmo_model='LCDM',use_gamma_info=False)
        
        dbl_lklhd_w0wa = tdc_sampler.TDCLikelihood(
            fpd_sample_shape=np.shape(self.data_vector_dict_dbls['fpd_samples']),
            cosmo_model='w0waCDM',use_gamma_info=False)
        
        quad_lklhd_w0wa = tdc_sampler.TDCLikelihood(
            fpd_sample_shape=np.shape(self.data_vector_dict_quads['fpd_samples']),
            cosmo_model='w0waCDM',use_gamma_info=False)
        
        # FUNCTION 1: td_log_likelihood_per_samp
        td_pred_samples = self.data_vector_dict_dbls['fpd_samples']*1.2 # just random vals.
        likelihood_per_samp = dbl_lklhd.td_log_likelihood_per_samp(
            td_pred_samples,data_vector_dict=self.data_vector_dict_dbls)        
        # test that shape of output is (num_lenses,num_fpd_samples)
        for i,s in enumerate([1,5]):
            print('shape: ',np.shape(td_pred_samples))
            # testing the two shape dimensions one at a time...
            self.assertEqual(np.shape(likelihood_per_samp)[i],s)

        # FUNCTION 2: td_pred_from_fpd_pred(hyperparameters)
        # h0,Omega_M,mu(gamma_lens),sigma(gamma_lens)
        hyperparameters = [70.,0.3,2.0,0.1]
        proposed_cosmo = dbl_lklhd.construct_proposed_cosmo(hyperparameters)
        td_predicted_dbls = dbl_lklhd.td_pred_from_fpd_pred(proposed_cosmo,
            data_vector_dict=self.data_vector_dict_dbls)

        # test that shape of output is (num_lenses,num_fpd_samples,1) for dbls
        for i,s in enumerate([1,5,1]):
            self.assertEqual(np.shape(td_predicted_dbls)[i],s)

        # check for last dim. (num_lenses,num_fpd_samples,3) for quads
        td_predicted_quads = quad_lklhd.td_pred_from_fpd_pred(proposed_cosmo,
            data_vector_dict=self.data_vector_dict_quads)
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

            # lens 1 (the double)
            lens1_computed_ll = dbl_lklhd.full_log_likelihood(hyperparameters,
                data_vector_dict=self.data_vector_dict_dbls)

            proposed_cosmo = dbl_lklhd.construct_proposed_cosmo(hyperparameters)
            proposed_gamma_model = norm(loc=hyperparameters[-2],scale=hyperparameters[-1])
            td_pred_samples = dbl_lklhd.td_pred_from_fpd_pred(proposed_cosmo,
                data_vector_dict=self.data_vector_dict_dbls)

            # lens 1 (the double)
            lens1_likelihood = 0
            # loop thru each importance sample
            for f in range(0,5):
                my_pred = td_pred_samples[0][f][0]
                dbl_td_measured = self.data_vector_dict_dbls['td_measured'][0][0]
                exponent = (-0.5*(my_pred - dbl_td_measured)**2 * 
                    self.data_vector_dict_dbls['td_likelihood_prec'][0][0][0])
                log_prefactor = (np.log((1/(2*np.pi))**(0.5) / 
                    np.sqrt(10.)) )# NOTE: hardcoded
                gamma_samp = self.data_vector_dict_dbls['lens_param_samples'][0,f,3]
                # default assumption = uninformative interim prior 
                #   (rw factor just comes from proposed pop model)
                rw_factor = proposed_gamma_model.logpdf(gamma_samp)
                lens1_log_likelihood = log_prefactor + exponent + rw_factor
                lens1_likelihood += np.exp(lens1_log_likelihood)

            lens1_likelihood /= 5

            self.assertAlmostEqual(lens1_computed_ll,np.log(lens1_likelihood))
                
            # lens 2 (the quad)
            lens2_computed_ll = quad_lklhd.full_log_likelihood(hyperparameters,
                data_vector_dict=self.data_vector_dict_quads)

            # proposed_cosmo is still the same here...
            td_pred_samples = quad_lklhd.td_pred_from_fpd_pred(proposed_cosmo,
                data_vector_dict=self.data_vector_dict_quads)

            lens2_likelihood = 0
            for f in range(0,5):
                my_pred = np.asarray(td_pred_samples[0][f])
                x_minus_mu = (my_pred -
                    self.data_vector_dict_quads['td_measured'][0])
                prec_mat = self.data_vector_dict_quads['td_likelihood_prec'][0]
                exponent = -0.5 * np.matmul(x_minus_mu,np.matmul(prec_mat,x_minus_mu))
                log_prefactor = (np.log((1/(2*np.pi))**(1.5) / 
                    np.sqrt(20.*22.*24.))) # NOTE: hardcoded
                gamma_samp = self.data_vector_dict_quads['lens_param_samples'][0,f,3]
                # default assumption = uninformative interim prior 
                #   (rw factor just comes from proposed pop model)
                rw_factor = proposed_gamma_model.logpdf(gamma_samp)
                lens2_log_likelihood = log_prefactor + exponent + rw_factor
                lens2_likelihood += np.exp(lens2_log_likelihood)

            lens2_likelihood /= 5

            self.assertAlmostEqual(lens2_computed_ll,np.log(lens2_likelihood))

        # LCDM case
        likelihood_test_case(dbl_lklhd,quad_lklhd,hyperparameters = [70.,0.3,2.0,0.2])
        # w0waCDM case
        likelihood_test_case(dbl_lklhd_w0wa,quad_lklhd_w0wa,hyperparameters=[70,0.3,-1.,0.,2.0,0.2])

"""
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
        
    def _check_chain_moves(self,mcmc_chain):
        # loop over params, check that chain is moving
        for param_idx in range(0,mcmc_chain.shape[2]):
            # isolate a single walker
            single_chain = mcmc_chain[0,:,param_idx]
            # check if chain has moved away from starting point
            diff_from_initial = single_chain[1:] - single_chain[0]
            self.assertNotAlmostEqual(0.,np.sum(diff_from_initial) )    

    def test_fast_tdc(self):

        # make TDCLikelihood object
        z_lens = [0.5,0.6]
        z_src = [1.2,1.3]
        my_tdc = tdc_sampler.TDCLikelihood(
            self.td_measured_dbls,self.td_prec_dbls,
            self.fpd_pred_samples_dbls,self.gamma_pred_samples_dbls,
            z_lens=[0.5],z_src=[1.2])


        # check if it works, test_chain dims are: (walkers,samples,params)
        test_chain = tdc_sampler.fast_TDC([my_tdc],num_emcee_samps=5,
            n_walkers=20)
        self._check_chain_moves(test_chain)


        # Combine doubles and quads, check it works...
        quads_tdc_lhood = tdc_sampler.TDCLikelihood(
            self.td_measured_quads,self.td_prec_quads,
            self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
            z_lens=[0.6],z_src=[1.3])
        
        # check if it works, test_chain dims are: (walkers,samples,params)
        test_chain = tdc_sampler.fast_TDC([my_tdc,quads_tdc_lhood],
            num_emcee_samps=5,n_walkers=20)
        self._check_chain_moves(test_chain)


        # Check that inclusion of lambda_int works
        quad_kin_lklhd = tdc_sampler.TDCKinLikelihood(
            self.td_measured_quads,self.td_prec_quads,
            self.sigma_v_measured,self.sigma_v_likelihood_prec,
            self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
            self.kin_pred_samples,
            kappa_ext_samples=self.kappa_ext_samples,
            z_lens=[0.6],z_src=[1.3],cosmo_model='LCDM_lambda_int')
        
        # check if it works, test_chain dims are: (walkers,samples,params)
        test_chain = tdc_sampler.fast_TDC([quad_kin_lklhd],num_emcee_samps=5,
            n_walkers=20)
        self._check_chain_moves(test_chain)

    def test_fast_tdc_ifu(self):

        # construct a beta_ani modeling prior
        beta_ani_nu_int = norm(loc=0.,scale=0.2).logpdf

        # Check that inclusion of kappa_ext and beta_ani works
        ifu_quad_lklhd = tdc_sampler.TDCKinLikelihood(
            self.td_measured_quads,self.td_prec_quads,
            self.ifu_sigma_v_measured,self.ifu_sigma_v_likelihood_prec,
            self.fpd_pred_samples_quads,self.gamma_pred_samples_quads,
            self.ifu_sigma_v_pred_samples,
            beta_ani_samples=self.beta_ani_samples,
            log_prob_beta_ani_nu_int=beta_ani_nu_int,
            z_lens=[0.6],z_src=[1.3],cosmo_model='LCDM_lambda_int_beta_ani')

        test_chain = tdc_sampler.fast_TDC([ifu_quad_lklhd],num_emcee_samps=5,
            n_walkers=20)
        self._check_chain_moves(test_chain)        

    def test_ddt_posteriors_from_fpd_td(self):

        # set up something where we know the ground truth
        z_lens = 0.598
        z_src = 1.7982546
        fpd_truth = [ -0.03878689524637091, -0.11086611144147474, -0.12145087695149426]
        td_truth = [ -4.355989563838178, -12.450896658648382, -13.639626197438508]
        # Ground Truth Cosmology
        gt_cosmo = jax_cosmo.Cosmology(h=jnp.float32(70./100),
                        Omega_c=jnp.float32(0.3-0.05), # "cold dark matter fraction", OmegaM = 0.3
                        Omega_b=jnp.float32(0.05), # "baryonic fraction"
                        Omega_k=jnp.float32(0.),
                        w0=jnp.float32(-1.),
                        wa=jnp.float32(0.),
                        sigma8 = jnp.float32(0.8), n_s=jnp.float32(0.96))
        Ddt_truth = tdc_utils.jax_ddt_from_redshifts(gt_cosmo,z_lens,z_src)

        fpd_samps = multivariate_normal.rvs(mean=fpd_truth,
            cov=(0.02**2)*np.eye(3),size=5000) # 0.02 measurement error
        

        ddt_chain = tdc_sampler.TDCLikelihood.ddt_posterior_from_td_fpd(
            td_measured=td_truth,
            td_likelihood_prec=np.eye(3)*(1/4.), # 2-day measurement error
            fpd_samples=fpd_samps,
            num_emcee_samps=5000
        )

        # Stack and condense the walker dimension of ddt_chain
        ddt_chain_stacked = np.reshape(ddt_chain[1000:], (-1, ddt_chain.shape[-1]))
        
        # TODO
        ddt_chain_mean = np.mean(ddt_chain_stacked)
        ddt_chain_sigma = np.std(ddt_chain_stacked,ddof=1)
        print('Predicted ddt: ', ddt_chain_mean, ' +/- ', ddt_chain_sigma)
        print("True ddt: ", Ddt_truth)
        
"""
