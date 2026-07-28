import time
import sys
from functools import partial
import emcee
import numpy as np
from astropy.cosmology import w0waCDM
from scipy.stats import norm, truncnorm, uniform, multivariate_normal
import Utils.tdc_utils as tdc_utils
import math
from priors import *
from completo_tdc_extensions import *

USE_JAX = False

if USE_JAX:
    import jax
    import jax.numpy as jnp
    import jax_cosmo
    import tdc_jax_utils as jax_utils
"""
cosmo_models available: 
    'LCDM': [H0,OmegaM,mu(gamma_lens),sigma(gamma_lens)]
    'LCDM_completo_cPDF': [H0,OmegaM,mu(params),chol_elements(params)]
    'w0waCDM': [H0,OmegaM,w0,wa,mu(gamma_lens),sigma(gamma_lens)]
    'LCDM_lambda_int': [H0,OmegaM,mu(lambda_int),sigma(lambda_int),
        mu(gamma_lens),sigma(gamma_lens)]
    'LCDM_lambda_int_beta_ani'
    'w0waCDM_lambda_int_beta_ani'
    'w0waCDM_fullcPDF'
"""

###########################
# TDC Likelihood Functions
###########################

# I think we want this to be a class, so we can keep track of quantities
# internally 


class TDCLikelihood():

    def __init__(self, fpd_sample_shape, cosmo_model='LCDM',
                 use_gamma_info=True, use_astropy=False):
        """
        Keep track of quantities that remain constant throughout the inference

        Args:
            fpd_sample_shape ()
            cosmo_model (string): 'LCDM' or 'w0waCDM'
            use_gamma_info (bool): If False, removes reweighting from likelihood
                evaluation (any population level gamma params should just
                return the prior then...)

        Note:
            likelihood evaluation requires an accompanying data_vector_dict 
            with key/value pairs:
                'td_measured' shape=(n_lenses,n_td)
                'td_likelihood_prec' shape=(n_lenses,n_td,n_td)
                'td_likelihood_prefactors' shape=(n_lenses)
                'fpd_samples' shape=(n_lenses,n_imp_samples,n_td)
                'lens_param_samples shape=(n_lenses,n_imp_samples,n_lens_params)
                'z_lens' shape=(n_lenses)
                'z_src' shape=(n_lenses)
            Optional: 
                'log_prob_lens_param_samps_nu_int' shape=(n_lenses,n_imp_samples)
        """

        if cosmo_model not in ['LCDM', 'LCDM_lambda_int',
                               'LCDM_lambda_int_beta_ani', 
                               'LCDM_completo_cPDF',
                               'w0waCDM', 
                               'w0waCDM_lambda_int_beta_ani',
                               'w0waCDM_fullcPDF','w0waCDM_fullcPDF_noKIN']:
            raise ValueError("choose from available cosmo_models: " +
                             "LCDM, LCDM_lambda_int, LCDM_lambda_int_beta_ani, LCDM_completo_cPDF, w0waCDM, " +
                             "w0waCDM_lambda_int_beta_ani, w0waCDM_fullcPDF")
        self.cosmo_model = cosmo_model
        self.use_gamma_info = use_gamma_info
        self.use_astropy = use_astropy
        # make sure the dims are right
        self.num_lenses, self.num_fpd_samples, self.dim_fpd = fpd_sample_shape

    # compute predicted time delays from predicted fermat potential differences
    # requires an assumed cosmology (from hyperparameters) and redshifts


    def td_pred_from_fpd_pred(self, proposed_cosmo, data_vector_dict=None, 
            global_data_vector_idx=None, lambda_int_samples=None):
        """
        Args:
            proposed_cosmo (default: jax_cosmo.Cosmology): built by
                construct_proposed_cosmo() (see below)
            data_vector_dict ({'key': np.array()}, default=None): contains:
                'z_lens', 'z_src', 'fpd_samples'
            global_data_vector_idx (int, default=None)
            lambda_int_samples (): shape=(num_lenses,num_fpd_samples)

        Notes: pass data_vector_dict directly OR index into a globally set list 
            with global_data_vector_idx. Cannot do both.
            

        Returns:
            td_pred_samples (size:(n_lenses,n_samples,3))
        """

        # check whether using global data vectors or passing directly...
        if global_data_vector_idx is not None:
            # make sure we're not trying to do two things at once
            if data_vector_dict is not None:
                raise ValueError('pass data vector OR index into global data vector, not both')
            # retrieve globally stored data vectors
            data_vector_dict = data_vector_global[global_data_vector_idx]

        if self.use_astropy:
            Ddt_computed = tdc_utils.ddt_from_redshifts(proposed_cosmo,
                                                        data_vector_dict['z_lens'],
                                                        data_vector_dict['z_src'])
        else:
            Ddt_computed = tdc_utils.jax_ddt_from_redshifts(proposed_cosmo,
                                                            data_vector_dict['z_lens'],
                                                            data_vector_dict['z_src'])

        Ddt_computed = np.array(Ddt_computed)
        # add batch dimensions for Ddt computed...
        Ddt_repeated = np.repeat(Ddt_computed[:, np.newaxis],
                                 self.num_fpd_samples, axis=1)
        Ddt_repeated = np.repeat(Ddt_repeated[:, :, np.newaxis],
                                 self.dim_fpd, axis=2)
        # compute predicted time delays (this function should work w/ arrays)
        td_pred = tdc_utils.td_from_ddt_fpd(Ddt_repeated, data_vector_dict['fpd_samples'])

        # Account for mass sheets:
        #   td = lambda * td
        #   lambda = (1-kappa_ext)*lambda_int

        # Linear scaling if lambda_int is present
        if lambda_int_samples is not None:
            lambda_int_repeated = np.repeat(lambda_int_samples[:, :, np.newaxis],
                                            self.dim_fpd, axis=2)
            td_pred *= lambda_int_repeated
        # Scaling if kappa_ext is present...
        if 'kappa_ext_samples' in data_vector_dict.keys():
            kappa_ext_repeated = np.repeat(data_vector_dict['kappa_ext_samples'][:, :, np.newaxis],
                                           self.dim_fpd, axis=2)
            td_pred *= (1 - kappa_ext_repeated)

        return td_pred

    def td_log_likelihood_per_samp(self, td_pred_samples, data_vector_dict=None, 
            global_data_vector_idx=None,):
        """
        Args:
            td_pred_samples (n_lenses,n_fpd_samps,3)

        Returns:
            td_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        # check whether using global data vectors or passing directly...
        if global_data_vector_idx is not None:
            # make sure we're not trying to do two things at once
            if data_vector_dict is not None:
                raise ValueError('pass data vector OR index into global data vector, not both')
            # retrieve globally stored data vectors
            data_vector_dict = data_vector_global[global_data_vector_idx]

        x_minus_mu = (td_pred_samples - data_vector_dict['td_measured'])
        # add dimension s.t. x_minus_mu is 2D
        x_minus_mu = np.expand_dims(x_minus_mu, axis=-1)
        # matmul should condense the (# of time delays) dim.
        exponent = -0.5 * np.matmul(np.transpose(x_minus_mu, axes=(0, 1, 3, 2)),
                                    np.matmul(data_vector_dict['td_likelihood_prec'], x_minus_mu))

        # reduce to two dimensions: (n_lenses,n_fpd_samples)
        # reduce only the last two dims to avoid edge cases (i.e. what if only one lens...)
        exponent = np.squeeze(exponent,axis=-1)
        exponent = np.squeeze(exponent,axis=-1)

        # log-likelihood
        return data_vector_dict['td_likelihood_prefactors'] + exponent


    def construct_proposed_cosmo(self, hyperparameters):
        """
        Args:
            hyperparameters ():
                - LCDM order: [H0,Omega_M,mu_gamma,sigma_gamma]
                - LCDM_lambda_int order: [H0,Omega_M,mu_lambda_int,
                    sigma_lambda_int,mu_gamma,sigma_gamma]
                - w0waCDM order: [H0,Omega_M,w0,wa,mu_gamma,sigma_gamma]
        """
        # construct cosmology object from hyperparameters
        h0_input = hyperparameters[0]
        # NOTE: baryonic fraction hardcoded to 0.05
        omega_m_input = hyperparameters[1]
        omega_c_input = hyperparameters[1] - 0.05  # CDM fraction
        omega_de_input = 1. - omega_m_input
        if self.cosmo_model in ['LCDM', 'LCDM_lambda_int',
                                'LCDM_completo_cPDF',
                                'LCDM_lambda_int_beta_ani']:
            w0_input = -1.
            wa_input = 0.
        elif self.cosmo_model in ['w0waCDM', 'w0waCDM_lambda_int_beta_ani',
                                  'w0waCDM_fullcPDF','w0waCDM_fullcPDF_noKIN']:
            w0_input = hyperparameters[2]
            wa_input = hyperparameters[3]

        if self.use_astropy:
            # instantiate astropy cosmology object
            astropy_cosmo = w0waCDM(H0=h0_input,
                                    Om0=omega_m_input, Ode0=omega_de_input,
                                    w0=w0_input, wa=wa_input)

            return astropy_cosmo

        else:
            # NOTE: baryonic fraction hardcoded to 0.05
            my_jax_cosmo = jax_cosmo.Cosmology(h=jnp.float32(h0_input / 100),
                                               Omega_c=jnp.float32(omega_c_input),  # "cold dark matter fraction"
                                               Omega_b=jnp.float32(0.05),  # "baryonic fraction"
                                               Omega_k=jnp.float32(0.),
                                               w0=jnp.float32(w0_input),
                                               wa=jnp.float32(wa_input), sigma8=jnp.float32(0.8), n_s=jnp.float32(0.96))

            return my_jax_cosmo

    def process_hyperparam_proposal(self, hyperparameters):
        """
        Args:
            hyperparameters ():
                - LCDM order: [H0,Omega_M,mu_gamma,sigma_gamma]
                - LCDM_lambda_int order: [H0,Omega_M,mu_lambda_int,
                    sigma_lambda_int,mu_gamma,sigma_gamma]
                - w0waCDM order: [H0,Omega_M,w0,wa,mu_gamma,sigma_gamma]
        Returns:
            proposed_cosmo (default=jax_cosmo.Cosmology)
            lambda_int_samples (): Set to None if no lambda_int in hypermodel.
                If in hypermodel, shape=(num_lenses,num_fpd_samples)
        """

        # importance sampling over lambda_int based on proposal distribution
        lambda_int_samples = None
        mu_lint = None
        if self.cosmo_model == 'LCDM_lambda_int':
            # NOTE: hardcoding of hyperparameter order!! (-4 is mu, -3 is sigma)
            mu_lint = hyperparameters[-4]
            sigma_lint = hyperparameters[-3]
        elif self.cosmo_model in ['LCDM_lambda_int_beta_ani',
                                  'w0waCDM_lambda_int_beta_ani']:
            # NOTE: hardcoding of hyperparameter order!! (-6 is mu, -5 is sigma)
            mu_lint = hyperparameters[-6]
            sigma_lint = hyperparameters[-5]

        elif self.cosmo_model == 'w0waCDM_fullcPDF':
            # NOTE: hardcoding of hyperparameter order!! (4 is mu, 5 is sigma)
            mu_lint = hyperparameters[4]
            sigma_lint = hyperparameters[5]

        if mu_lint is not None:
            lambda_int_samples = truncnorm.rvs(-mu_lint / sigma_lint, np.inf,
                                               loc=mu_lint, scale=sigma_lint,
                                               size=(self.num_lenses, self.num_fpd_samples))

        return self.construct_proposed_cosmo(hyperparameters), lambda_int_samples

    def full_log_likelihood(self, hyperparameters, data_vector_dict=None, 
            global_data_vector_idx=None):
        """
        Args:
            hyperparameters ([H0,mu_gamma,sigma_gamma] or [H0,w0,wa,mu_gamma,sigma_gamma])
            fpd_pred_samples (size:(n_lenses,n_samples,3)): Note, it is assumed
                that doubles are padded w/ zeros
        """

        # construct cosmology + lint samps (if required) from hyperparameters
        proposed_cosmo, lambda_int_samples = self.process_hyperparam_proposal(
            hyperparameters)

        # td_pred_samples from fpd_pred_samples
        td_pred_samples = self.td_pred_from_fpd_pred(proposed_cosmo, data_vector_dict=data_vector_dict, 
            global_data_vector_idx=global_data_vector_idx,lambda_int_samples=lambda_int_samples)
        # td likelihood for every sample from every lens
        td_log_likelihoods = self.td_log_likelihood_per_samp(
            td_pred_samples, data_vector_dict=data_vector_dict, 
            global_data_vector_idx=global_data_vector_idx
        )

        # reweighting factor
        # TODO: fix this for new framework
        if self.use_gamma_info:
            rw_factor = self.compute_rw_factor(hyperparameters,data_vector_dict=data_vector_dict, 
                global_data_vector_idx=global_data_vector_idx)
        else:
            rw_factor = 0.

        # sum across fpd samples
        individ_likelihood = np.mean(np.exp(td_log_likelihoods + rw_factor), axis=1)

        # sum over all lenses
        if np.sum(individ_likelihood == 0) > 0:
            return -np.inf

        log_likelihood = np.sum(np.log(individ_likelihood))

        return log_likelihood
    
    def compute_rw_factor(self,hyperparameters,data_vector_dict=None, 
                global_data_vector_idx=None):
        """ Re-weighting for pop model (nu) vs. interim prior (nu_int)
        Args: 
            hyperparameters ():
            data_vector_dict ():
            global_data_vector_idx ():

        Returns:
            rw_factor ():
        """

        # check whether using global data vectors or passing directly...
        if global_data_vector_idx is not None:
            # make sure we're not trying to do two things at once
            if data_vector_dict is not None:
                raise ValueError('pass data vector OR index into global data vector, not both')
            # retrieve globally stored data vectors
            data_vector_dict = data_vector_global[global_data_vector_idx]


        # modify into proposed hypermodel
        if self.cosmo_model == 'w0waCDM_fullcPDF':
            nu_means = np.empty(6)
            nu_stddevs = np.empty(6)

            # theta_E
            nu_means[0] = hyperparameters[10]
            nu_stddevs[0] = hyperparameters[11]
            # external shear (gamma1,gamma2)
            nu_means[1] = 0.
            nu_means[2] = 0.
            nu_stddevs[1] = hyperparameters[12]
            nu_stddevs[2] = hyperparameters[12]
            # gamma_lens
            nu_means[3] = hyperparameters[8]
            nu_stddevs[3] = hyperparameters[9]
            # ellipticity (e1,e2) 
            nu_means[4] = 0.
            nu_means[5] = 0.
            nu_stddevs[4] = hyperparameters[13]
            nu_stddevs[5] = hyperparameters[13]

            # evaluate over 6 lens params
            eval_at_proposed_nu = multivariate_normal.logpdf(
                data_vector_dict['lens_param_samples'],
                mean=nu_means,
                cov=np.diag(nu_stddevs**2))

        elif self.cosmo_model == 'w0waCDM_fullcPDF_noKIN': 
            nu_means = np.empty(6)
            nu_stddevs = np.empty(6)

            # theta_E
            nu_means[0] = hyperparameters[6]
            nu_stddevs[0] = hyperparameters[7]
            # external shear (gamma1,gamma2)
            nu_means[1] = 0.
            nu_means[2] = 0.
            nu_stddevs[1] = hyperparameters[8]
            nu_stddevs[2] = hyperparameters[8]
            # gamma_lens
            nu_means[3] = hyperparameters[4]
            nu_stddevs[3] = hyperparameters[5]
            # ellipticity (e1,e2) 
            nu_means[4] = 0.
            nu_means[5] = 0.
            nu_stddevs[4] = hyperparameters[9]
            nu_stddevs[5] = hyperparameters[9]

            # evaluate over 6 lens params
            eval_at_proposed_nu = multivariate_normal.logpdf(
                data_vector_dict['lens_param_samples'],
                mean=nu_means,
                cov=np.diag(nu_stddevs**2))

        else: # all other models assume a population over gamma_lens by default
            gamma_mean = hyperparameters[-2]
            gamma_stddev = hyperparameters[-1]

            # just evaluate over one param (gamma_lens is at index 3)
            eval_at_proposed_nu = norm.logpdf(
                data_vector_dict['lens_param_samples'][:,:,3],
                loc=gamma_mean,scale=gamma_stddev)

        # compute the rw factor by comparing to interim prior 
        if 'log_prob_lens_param_samps_nu_int' in data_vector_dict.keys():
            # informative interim prior specified 
            rw_factor = (eval_at_proposed_nu - 
                data_vector_dict['log_prob_lens_param_samps_nu_int'])
        else:
            # uniform / uninformative interim prior
            rw_factor = eval_at_proposed_nu
            
        return rw_factor

    @staticmethod
    def ddt_posterior_from_td_fpd(td_measured, td_likelihood_prec, fpd_samples,
                                  num_emcee_samps=10000):
        """Computes ddt posterior from measured time delay(s) and
            samples from fermat potential difference posterior(s)
            for a SINGLE lens

        The inference:
            p(Ddt | delta_t, d_img) /propto p(Ddt) /integral [
                p(delta_t | delta_phi, Ddt) p( delta_phi | d_img, nu_int)
                p(delta_phi) / p(delta_phi | nu_int) d delta_phi   ]

        Args:
            td_measured ([n_td])
            td_likelihood_prec ([n_td,n_td])
            fpd_samples ([n_importance_samples,n_td])

        Returns:
            emcee.EnsembleSampler.get_chain()
        """

        # set up variables here
        n_td = len(td_measured)
        n_walkers = 10
        td_likelihood_prefactor = np.log((1 / (2 * np.pi) ** (n_td / 2)) /
                                         np.sqrt(np.linalg.det(np.linalg.inv(td_likelihood_prec))))

        def td_log_likelihood(Ddt_proposed):

            # TODO: check dimensions heres
            td_predicted = tdc_utils.td_from_ddt_fpd(Ddt_proposed, fpd_samples)

            x_minus_mu = (td_predicted - td_measured)
            # add dimension s.t. x_minus_mu is 2D
            x_minus_mu = np.expand_dims(x_minus_mu, axis=-1)
            # matmul should condense the (# of time delays) dim.
            # TODO: probably only 3 dimensions here? check...
            exponent = -0.5 * np.matmul(np.transpose(x_minus_mu, axes=(0, 2, 1)),
                                        np.matmul(td_likelihood_prec, x_minus_mu))

            # reduce to one dimension: (n_fpd_samples)
            # reduce only the last two dims to avoid edge cases (i.e. what if only one sample...)
            exponent = np.squeeze(exponent,axis=-1)
            exponent = np.squeeze(exponent,axis=-1)

            imp_samp_likelihood = np.mean(np.exp(td_likelihood_prefactor + exponent))

            return np.log(imp_samp_likelihood)

        def td_log_posterior(Ddt_proposed):

            # what's a good prior for Ddt?
            if Ddt_proposed < 0. or Ddt_proposed > 15000:
                return -np.inf
            else:
                return td_log_likelihood(Ddt_proposed)

        # set-up emcee sampler
        cur_state = np.empty((n_walkers, 1))
        cur_state[:, 0] = uniform.rvs(loc=0., scale=15000., size=n_walkers)
        sampler = emcee.EnsembleSampler(n_walkers,
                                        cur_state.shape[1], td_log_posterior)

        # run mcmc
        _ = sampler.run_mcmc(cur_state, nsteps=num_emcee_samps, progress=True)

        # return chain
        return sampler.get_chain()
    


class TDCKinLikelihood(TDCLikelihood):

    def __init__(self, fpd_sample_shape, kin_pred_samples_shape,
                 cosmo_model='LCDM' ,use_gamma_info=True,
                 use_astropy=False):
        """
        Keep track of quantities that remain constant throughout the inference

        Args:
            fpd_sample_shape: shape of fpd samples (n_lenses,n_fpd_samples,dim_fpd)
            kin_pred_samples_shape: shape of kinematic samples (n_lenses,n_fpd_samples,num_kin_bins)
            log_prob_gamma_nu_int: TODO
            cosmo_model (string): 'LCDM', 'w0waCDM', 'LCDM_lambda_int', or
                'LCDM_lambda_int_beta_ani'
            use_gamma_info (bool): If False, removes reweighting from likelihood
                evaluation (any population level gamma params should just
                return the prior then...)

        Note:
            likelihood evaluation requires an accompanying data_vector_dict 
            with key/value pairs:
                'td_measured' (n_lenses,n_td)
                'td_likelihood_prec' (n_lenses,n_td,n_td)
                'td_likelihood_prefactors' (n_lenses)
                'fpd_samples' (n_lenses,n_imp_samples,n_td)
                'lens_param_samples (n_lenses,n_imp_samples,n_lens_params)
                'z_lens' (n_lenses)
                'z_src' (n_lenses)
        """

        super().__init__(fpd_sample_shape, cosmo_model ,use_gamma_info,
                         use_astropy)

        self.num_kin_bins = kin_pred_samples_shape[2]


    def sigma_v_pred_from_kin_pred(self ,proposed_cosmo, data_vector_dict=None, 
            global_data_vector_idx=None, lambda_int_samples=None):
        """
        Args:
            proposed_cosmo (default: jax_cosmo.Cosmology): built by
                construct_proposed_cosmo() (see below)
            lambda_int_samples (): shape=(num_lenses,num_fpd_samples)
        """

        # check whether using global data vectors or passing directly...
        if global_data_vector_idx is not None:
            # make sure we're not trying to do two things at once
            if data_vector_dict is not None:
                raise ValueError('pass data vector OR index into global data vector, not both')
            # retrieve globally stored data vectors
            data_vector_dict = data_vector_global[global_data_vector_idx]

        if self.use_astropy:
            Ds_div_Dds_computed = tdc_utils.kin_distance_ratio(
                proposed_cosmo , data_vector_dict['z_lens'],
                data_vector_dict['z_src'])

            # raise ValueError("astropy option not implemented for TDC+Kin")
        else:
            Ds_div_Dds_computed = tdc_utils.jax_kin_distance_ratio(
                proposed_cosmo, data_vector_dict['z_lens'],
                data_vector_dict['z_src'])

        Ds_div_Dds_computed = np.array(Ds_div_Dds_computed)
        # add batch dimensions for fpd_samples
        Ds_div_Dds_repeated = np.repeat(Ds_div_Dds_computed[:, np.newaxis],
                                        self.num_fpd_samples, axis=1)
        # add batch dimension for # kinematic bins
        Ds_div_Dds_repeated = np.repeat(Ds_div_Dds_repeated[:, :, np.newaxis],
                                        self.num_kin_bins, axis=2)
        # scale the kin_pred with cosmology term: sigma_v = sqrt(Ds/Dds)*c*sqrt(mathcal{J})
        sigma_v_pred = np.sqrt(Ds_div_Dds_repeated ) *data_vector_dict['kin_pred_samples']

        # Account for mass sheets:
        #   sigma_v = sqrt(lambda) * sigma_v
        #   lambda = (1-kappa_ext)*lambda_int

        # sqrt(lambda) scaling if lambda_int is present
        if lambda_int_samples is not None:
            lambda_int_repeated = np.repeat(lambda_int_samples[: ,: ,np.newaxis],
                                            self.num_kin_bins, axis=2)
            sigma_v_pred *= np.sqrt(lambda_int_repeated)
        # sqrt(1-kappa_ext) scaling
        if 'kappa_ext_samples' in data_vector_dict.keys():
            kappa_ext_repeated = np.repeat(data_vector_dict['kappa_ext_samples'][: ,: ,np.newaxis],
                                           self.num_kin_bins, axis=2)
            sigma_v_pred *= np.sqrt(1 - kappa_ext_repeated)

        return sigma_v_pred


    def sigma_v_log_likelihood_per_samp(self,sigma_v_pred_samples, data_vector_dict=None, 
            global_data_vector_idx=None):
        """
        Args:
            sigma_v_pred_samples (n_lenses,n_fpd_samps,num_kin_bins)

        Returns:
            sigma_v_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        # check whether using global data vectors or passing directly...
        if global_data_vector_idx is not None:
            # make sure we're not trying to do two things at once
            if data_vector_dict is not None:
                raise ValueError('pass data vector OR index into global data vector, not both')
            # retrieve globally stored data vectors
            data_vector_dict = data_vector_global[global_data_vector_idx]

        x_minus_mu = (sigma_v_pred_samples - data_vector_dict['sigma_v_measured'])
        # add dimension s.t. x_minus_mu is 2D
        x_minus_mu = np.expand_dims(x_minus_mu ,axis=-1)
        # matmul should condense the (# of time delays) dim.
        exponent = -0.5 *np.matmul(np.transpose(x_minus_mu ,axes=(0 ,1 ,3 ,2)),
                                  np.matmul(data_vector_dict['sigma_v_likelihood_prec'],x_minus_mu))

        # reduce to two dimensions: (n_lenses,n_fpd_samples)
        # reduce only the last two dims to avoid edge cases (i.e. what if only one lens...)
        exponent = np.squeeze(exponent,axis=-1)
        exponent = np.squeeze(exponent,axis=-1)

        # log-likelihood
        return data_vector_dict['sigma_v_likelihood_prefactors'] + exponent
    

    def full_log_likelihood(self, hyperparameters, data_vector_dict=None, 
            global_data_vector_idx=None,print_debug=False):
        """Evaluate full log likelihood (for td and kin) and sum across all lenses
        Args:
            hyperparameters ():
            data_vector_dict ():
            global_data_vector_idx ():

        Returns:
        """

        # construct cosmology from hyperparameters
        proposed_cosmo, lambda_int_samples = self.process_hyperparam_proposal(
            hyperparameters)

        # td log likelihood per sample
        td_pred_samples = self.td_pred_from_fpd_pred(
            proposed_cosmo, data_vector_dict=data_vector_dict, 
                global_data_vector_idx=global_data_vector_idx, 
                lambda_int_samples=lambda_int_samples)
        td_log_likelihoods = self.td_log_likelihood_per_samp(
            td_pred_samples, data_vector_dict=data_vector_dict, 
            global_data_vector_idx=global_data_vector_idx)
        td_log_likelihoods = np.asarray(td_log_likelihoods)

        # kin log likelihood per sample
        sigma_v_pred_samples = self.sigma_v_pred_from_kin_pred(
            proposed_cosmo, data_vector_dict=data_vector_dict, 
            global_data_vector_idx=global_data_vector_idx, 
            lambda_int_samples=lambda_int_samples)
        sigma_v_log_likelihoods = self.sigma_v_log_likelihood_per_samp(
            sigma_v_pred_samples, data_vector_dict=data_vector_dict, 
                global_data_vector_idx=global_data_vector_idx
        )

        # reweighting factor
        # NOTE: hardcoding of hyperparameter order!! (-2 is mu, -1 is sigma)
        if self.use_gamma_info:
            rw_factor = self.compute_rw_factor(hyperparameters,data_vector_dict=data_vector_dict, 
                global_data_vector_idx=global_data_vector_idx)
        else:
            rw_factor = 0.


        if self.cosmo_model in ['LCDM_lambda_int_beta_ani',
            'w0waCDM_lambda_int_beta_ani','w0waCDM_fullcPDF']:

            # extract proposed mean/stddev of beta_ani 
            if self.cosmo_model == 'w0waCDM_fullcPDF':
                proposed_loc = hyperparameters[6]
                proposed_scale = hyperparameters[7]
            else:
                proposed_loc = hyperparameters[-4]
                proposed_scale = hyperparameters[-3]

            # retrieve beta_ani samples from either global data vector or provided data vector
            if data_vector_dict is not None:
                beta_ani_samples = data_vector_dict['beta_ani_samples']
                beta_interim_log_prob = data_vector_dict['log_prob_beta_ani_samps_nu_int']
            elif global_data_vector_idx is not None:
                beta_ani_samples = data_vector_global[global_data_vector_idx]['beta_ani_samples']
                beta_interim_log_prob = data_vector_global[global_data_vector_idx]['log_prob_beta_ani_samps_nu_int']

            # compare proposed prob to modeling prior prob
            eval_at_proposed_beta_pop = norm.logpdf(beta_ani_samples,
                loc=proposed_loc,scale=proposed_scale)
            beta_rw_factor = (eval_at_proposed_beta_pop - 
                beta_interim_log_prob)
            # additive in log space
            rw_factor += beta_rw_factor

        individ_likelihood = np.mean(
            np.exp(td_log_likelihoods +sigma_v_log_likelihoods +rw_factor),
            axis=1)
        
        if print_debug:
            print('individ. lens likelihood (NOT log): ', individ_likelihood)
        
        # sum over all lenses
        # TODO: there is a way to do this in jax
        if np.sum(individ_likelihood == 0) > 0:
            log_likelihood = -jnp.inf

        else:
            log_likelihood = np.sum(np.log(individ_likelihood))


        return log_likelihood
    

# overwrite TDCLikelihood (changing the compute_rw_factor() function)
class TDCLikelihoodCompleto(TDCLikelihood):
    """
        Keep track of quantities that remain constant throughout the inference

        Args:
            fpd_sample_shape ()
            cosmo_model (string): 'LCDM' or 'w0waCDM'
            use_gamma_info (bool): If False, removes reweighting from likelihood
                evaluation (any population level gamma params should just
                return the prior then...)

        Note:
            likelihood evaluation requires an accompanying data_vector_dict 
            with key/value pairs:
                'td_measured' shape=(n_lenses,n_td)
                'td_likelihood_prec' shape=(n_lenses,n_td,n_td)
                'td_likelihood_prefactors' shape=(n_lenses)
                'fpd_samples' shape=(n_lenses,n_imp_samples,n_td)
                'z_lens' shape=(n_lenses)
                'z_src' shape=(n_lenses)
            NEW REQUIRED PARAMS FOR COMPLETO SETTING: 
                'cPDF_param_samples': shape=(n_lenses,n_imp_samples,n_cPDF_params)
                    all parameters are normalized s.t. mu=0., std.dev.=1.
                'log_prob_cPDF_params_nu_int': shape=(n_lenses,n_imp_samples)
                    log_prob of each lens_param samp evaluated against the interim prior (nu_int)
    """

    def compute_rw_factor(self,hyperparameters,data_vector_dict=None, 
                global_data_vector_idx=None):
        """ Re-weighting for pop model (nu) vs. interim prior (nu_int)
        Args: 
            hyperparameters ():
            data_vector_dict ():
            global_data_vector_idx ():

        Returns:
            rw_factor ():
        """

        # check whether using global data vectors or passing directly...
        if global_data_vector_idx is not None:
            # make sure we're not trying to do two things at once
            if data_vector_dict is not None:
                raise ValueError('pass data vector OR index into global data vector, not both')
            # retrieve globally stored data vectors
            data_vector_dict = data_vector_global[global_data_vector_idx]

        if self.cosmo_model == 'LCDM_completo_cPDF':
            num_cp = 2 # number of cosmological params at front of DV
        else:
            raise ValueError('cosmology not compatible with completo likelihood (yet!)')

        # here comes the lens params cPDF
        num_h = len(hyperparameters[num_cp:]) # num_h = N + N(N+1)/2 (num. hyperparameters)
        # solv quad. eqn. to determine # of lens params (0 = n^2 + 3n -2(num_h))
        num_lp = int((-3 + np.sqrt(9+8*num_h))/2) # we only need the positive solution...

        means = hyperparameters[num_cp:(num_lp+num_cp)]
        chol_elements = hyperparameters[(num_lp+num_cp):]

        # chol_elements -> cov. matrix
        L = np.zeros((num_lp, num_lp))
        L[np.tril_indices(num_lp)] = chol_elements
        # Reconstruct covariance matrix: Cov = L @ L.T
        cov_matrix = L @ L.T

        # retrieve params to evaluate on
        param_samps = data_vector_dict['cPDF_param_samples']
        n_lenses, n_fpd_samps, _ = param_samps.shape
        param_samps = param_samps.reshape(-1,num_lp) # flatten first two dims
        # eval at proposed hypermodel
        eval_at_proposed_nu = multivariate_normal.logpdf(param_samps,mean=means,
            cov=cov_matrix)
        eval_at_proposed_nu = eval_at_proposed_nu.reshape(n_lenses, n_fpd_samps) # unravel dims

        # compute the rw factor by comparing to interim prior 
        if 'log_prob_cPDF_params_nu_int' in data_vector_dict.keys():
            # informative interim prior specified 
            rw_factor = (eval_at_proposed_nu - 
                data_vector_dict['log_prob_cPDF_params_nu_int'])
        else:
            # uniform / uninformative interim prior
            rw_factor = eval_at_proposed_nu
            
        #print('nan in rw_factor?: ', np.sum(np.isnan(rw_factor)))
        return rw_factor


#########################
# Sampling Implementation
#########################

def generate_initial_state(n_walkers,cosmo_model,use_tdcosmo25=False,
        random_seed=None,num_cpdf_params=None):
    """
    Args:
        n_walkers (int): number of emcee walkers
        cosmo_model (string): 'LCDM' or 'w0waCDM'
        num_cdpf_params (int): needed when using 'LCDM_completo_cPDF'
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    if cosmo_model == 'LCDM':
        # order: [H0,Omega_M,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,4))
        cur_state[:,0] = uniform.rvs(loc=65,scale=10,size=n_walkers) #h0
        cur_state[:,1] = uniform.rvs(loc=0.25,scale=0.1,size=n_walkers) #Omega_M
        cur_state[:,2] = uniform.rvs(loc=1.5,scale=1.,size=n_walkers)
        cur_state[:,3] = uniform.rvs(loc=0.001,scale=0.199,size=n_walkers)

        return cur_state
    
    if cosmo_model == 'LCDM_lambda_int':
        # order: [H0,Omega_M,mu_lambda_int,sigma_lambda_int,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,6))
        cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers) #h0
        cur_state[:,1] = uniform.rvs(loc=0.1,scale=0.35,size=n_walkers) #Omega_M
        cur_state[:,2] = uniform.rvs(loc=0.9,scale=0.2,size=n_walkers)
        cur_state[:,3] = uniform.rvs(loc=0.001,scale=0.499,size=n_walkers)
        cur_state[:,4] = uniform.rvs(loc=1.5,scale=1.,size=n_walkers)
        cur_state[:,5] = uniform.rvs(loc=0.001,scale=0.199,size=n_walkers)

        return cur_state
    
    if cosmo_model == 'LCDM_lambda_int_beta_ani':
        # order: [H0,Omega_M,mu_lambda_int,sigma_lambda_int,
        #   mu_beta_ani,sigma_beta_ani,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,8))
        cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers) #h0
        cur_state[:,1] = uniform.rvs(loc=0.1,scale=0.35,size=n_walkers) #Omega_M
        cur_state[:,2] = uniform.rvs(loc=0.9,scale=0.2,size=n_walkers)
        cur_state[:,3] = uniform.rvs(loc=0.001,scale=0.499,size=n_walkers)
        cur_state[:,4] = uniform.rvs(loc=-0.1,scale=0.2,size=n_walkers)
        cur_state[:,5] = uniform.rvs(loc=0.001,scale=0.199,size=n_walkers)
        cur_state[:,6] = uniform.rvs(loc=1.5,scale=1.,size=n_walkers)
        cur_state[:,7] = uniform.rvs(loc=0.001,scale=0.199,size=n_walkers)

        return cur_state
    
    # needs # of lens cPDF params...
    elif cosmo_model == 'LCDM_completo_cPDF':
        return LCDM_completo_cPDF_log_prior_generate_initialstate(n_walkers,
            cosmo_model,num_lp=num_cpdf_params,random_seed=None)
    
    elif cosmo_model == 'w0waCDM':
        # order: [H0,Omega_M,w0,wa,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,6))
        cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers) #h0
        cur_state[:,1] = uniform.rvs(loc=0.1,scale=0.35,size=n_walkers) #Omega_M
        cur_state[:,2] = uniform.rvs(loc=-1.5,scale=1.,size=n_walkers)
        cur_state[:,3] = uniform.rvs(loc=-1,scale=2,size=n_walkers)
        cur_state[:,4] = uniform.rvs(loc=1.5,scale=1.,size=n_walkers)
        cur_state[:,5] = uniform.rvs(loc=0.001,scale=0.19,size=n_walkers)

        return cur_state
    
    if cosmo_model == 'w0waCDM_lambda_int_beta_ani':
        # TODO: try this one with intializing with a compact ball!
        # order: [H0,Omega_M,w0,wa,mu_lambda_int,sigma_lambda_int,
        #   mu_beta_ani,sigma_beta_ani,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,10))
        cur_state[:,0] = norm.rvs(loc=70.,scale=5.,size=n_walkers) #h0
        cur_state[:,1] = truncnorm.rvs(-.3/.1,.2/0.1,loc=0.3,scale=0.1,size=n_walkers) #Omega_M
        # NOTE: out of bounds for TDCOSMO25 prior! 
        if use_tdcosmo25:
            cur_state[:,2] = truncnorm.rvs(-0.5/0.2,1.5/0.2,loc=-1.,scale=0.2,size=n_walkers) #w0
        else:
            cur_state[:,2] = truncnorm.rvs(-1/.2,1/.2,loc=-1.,scale=0.2,size=n_walkers) #w0
        cur_state[:,3] = truncnorm.rvs(-1/.2,1/.2,loc=0.,scale=0.2,size=n_walkers) #wa
        cur_state[:,4] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=1.,scale=0.1,size=n_walkers) # mu(lambda_int)
        cur_state[:,5] = uniform.rvs(loc=0.01,scale=0.49,size=n_walkers)
        cur_state[:,6] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=0.,scale=0.1,size=n_walkers) # mu(beta_ani)
        cur_state[:,7] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)
        cur_state[:,8] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=2.,scale=0.1,size=n_walkers) # mu(gamma_lens)
        cur_state[:,9] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)

        return cur_state
    
    if cosmo_model == 'w0waCDM_fullcPDF':
        # TODO: try this one with intializing with a compact ball!
        # order: [H0,Omega_M,w0,wa,mu_lambda_int,sigma_lambda_int,
        #   mu_beta_ani,sigma_beta_ani,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,14))
        cur_state[:,0] = norm.rvs(loc=70.,scale=5.,size=n_walkers) #h0
        cur_state[:,1] = truncnorm.rvs(-.3/.1,.2/0.1,loc=0.3,scale=0.1,size=n_walkers) #Omega_M
        cur_state[:,2] = truncnorm.rvs(-1/.2,1/.2,loc=-1.,scale=0.2,size=n_walkers) #w0
        cur_state[:,3] = truncnorm.rvs(-1/.2,1/.2,loc=0.,scale=0.2,size=n_walkers) #wa
        cur_state[:,4] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=1.,scale=0.1,size=n_walkers) # mu(lambda_int)
        cur_state[:,5] = uniform.rvs(loc=0.01,scale=0.49,size=n_walkers)
        cur_state[:,6] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=0.,scale=0.1,size=n_walkers) # mu(beta_ani)
        cur_state[:,7] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)
        cur_state[:,8] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=2.,scale=0.1,size=n_walkers) # mu(gamma_lens)
        cur_state[:,9] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)
        cur_state[:,10] = truncnorm.rvs(-3.,3.,loc=0.8,scale=0.2,size=n_walkers) # mu(theta_E)
        cur_state[:,11] = uniform.rvs(loc=0.01,scale=0.49,size=n_walkers)
        cur_state[:,12] = uniform.rvs(loc=0.01,scale=0.09,size=n_walkers) # sigma(gamma1/2)
        cur_state[:,13] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers) # sigma(e1/2)

        return cur_state
    
    if cosmo_model == 'w0waCDM_fullcPDF_noKIN':
        # TODO: try this one with intializing with a compact ball!
        # order: [H0,Omega_M,w0,wa,mu_lambda_int,sigma_lambda_int,
        #   mu_beta_ani,sigma_beta_ani,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,10))
        cur_state[:,0] = norm.rvs(loc=70.,scale=5.,size=n_walkers) #h0
        cur_state[:,1] = truncnorm.rvs(-.3/.1,.2/0.1,loc=0.3,scale=0.1,size=n_walkers) #Omega_M
        cur_state[:,2] = truncnorm.rvs(-1/.2,1/.2,loc=-1.,scale=0.2,size=n_walkers) #w0
        cur_state[:,3] = truncnorm.rvs(-1/.2,1/.2,loc=0.,scale=0.2,size=n_walkers) #wa
        cur_state[:,4] = truncnorm.rvs(-0.5/0.1,0.5/0.1,loc=2.,scale=0.1,size=n_walkers) # mu(gamma_lens)
        cur_state[:,5] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)
        cur_state[:,6] = truncnorm.rvs(-3.,3.,loc=0.8,scale=0.2,size=n_walkers) # mu(theta_E)
        cur_state[:,7] = uniform.rvs(loc=0.01,scale=0.49,size=n_walkers)
        cur_state[:,8] = uniform.rvs(loc=0.01,scale=0.09,size=n_walkers) # sigma(gamma1/2)
        cur_state[:,9] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers) # sigma(e1/2)

        return cur_state

def log_likelihood(hyperparameters,tdc_likelihood_list):
    """Iterate through sub-samples and add together log-likelihood
    """
    fll = 0
    for i, tdc_likelihood in enumerate(tdc_likelihood_list):
        fll += tdc_likelihood.full_log_likelihood(hyperparameters, global_data_vector_idx = i)
    return fll

def log_posterior(hyperparameters, cosmo_model, tdc_likelihood_list,
    use_informative=False,use_inf_pop=False,
    use_OmegaM=False,use_tdcosmo25=False):
    """
    Args:
        hyperparameters ([float]): 
            - LCDM: [H0,Omega_M,mu_gamma,sigma_gamma] 
            - LCDM_lambda_int_beta_ani: [H0,Omega_M,
                mu_lint,sigma_lint,mu_bani,sigma_bani,mu_gamma,sigma_gamma] 
            - w0waCDM: [H0,Omega_M,w0,wa,mu_gamma,sigma_gamma]
    """
    #rank = MPI.COMM_WORLD.Get_rank()
    #pid = os.getpid()
    #print(f"[Rank {rank} | PID {pid}] Evaluating log-posterior at {hyperparameters}")

    # Prior
    if cosmo_model == 'LCDM':
        lp = LCDM_log_prior(hyperparameters)
    elif cosmo_model == 'LCDM_lambda_int':
        lp = LCDM_lambda_int_log_prior(hyperparameters)
    elif cosmo_model == 'LCDM_lambda_int_beta_ani':
        if use_inf_pop:
            lp = InformedPop_InformedOmegaM_LCDM_lambda_int_beta_ani_log_prior(hyperparameters)
        elif use_OmegaM:
            lp = OmegaM_LCDM_lambda_int_beta_ani_log_prior(hyperparameters)
        else:
            lp = LCDM_lambda_int_beta_ani_log_prior(hyperparameters)
    elif cosmo_model =='LCDM_completo_cPDF':
        lp = LCDM_completo_cPDF_log_prior(hyperparameters)
        # TODO: switch to this format (can multiply the OmegaM prior in at this step w/out rewriting every function)
        if use_OmegaM and np.isfinite(lp):
            lp += norm.logpdf(hyperparameters[1],loc=0.3,scale=0.018)
    elif cosmo_model == 'w0waCDM':
        lp = w0waCDM_log_prior(hyperparameters)
    elif cosmo_model == 'w0waCDM_lambda_int_beta_ani':
        if use_informative: # for redshift configuration test...
            lp = INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
            if use_OmegaM:
                lp = OmegaM_INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
        elif use_inf_pop:
            lp = InformedPop_InformedOmegaM_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
        elif use_OmegaM:
            lp = OmegaM_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
        elif use_tdcosmo25:
            lp = tdcosmo25_lambda_int_beta_ani_log_prior(hyperparameters)
        else:
            lp = w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
    elif cosmo_model == 'w0waCDM_fullcPDF':
        lp = w0waCDM_fullcPDF_log_prior(hyperparameters)
    elif cosmo_model == 'w0waCDM_fullcPDF_noKIN':
        lp = w0waCDM_fullcPDF_noKIN_log_prior(hyperparameters)
    # Likelihood
    if not math.isinf(lp):
        lp += log_likelihood(hyperparameters,tdc_likelihood_list)

    return lp


def fast_TDC(tdc_likelihood_list, data_vector_list, num_emcee_samps=1000,
    n_walkers=20, use_mpi=False, use_multiprocess=False, backend_path=None, 
    reset_backend=True,sampler_type='emcee',use_informative=False,
    use_inf_pop=False,use_OmegaM=False,use_tdcosmo25=False,init_seed=None,
    num_cpdf_params=None):
    """
    Args:
        tdc_likelihood_list ([TDCLikelihood]): list of likelihood objects 
            (will add log likelihoods together)
        data_vector_list ([{'key': np.array()}]): list of dictionaries
            containing key/value pairs that store data vectors as np.arrays()
        num_emcee_samps (int): Number of iterations for MCMC inference
        n_walkers (int): Number of emcee walkers
        use_mpi (bool): If True, uses MPI for parallelization
        backend_path (string): If not None, saves a backend .h5 file. 
            Otherwise, returns the chain.
        sampler_type (string): 'emcee' or 'dynesty'
        use_informative, use_OmegaM: Boolean flags, control the use of informative priors...
        init_seed (int or None): if specified, seeds the random 
            initialization of walkers
        num_cpdf_params (int or None): needed when using LCDM_completo_cPDF cosmo.
        
    Returns: 
        mcmc chain (emcee.EnsemblerSampler.chain or dynesty.NestedSampler.)
    """

    # Retrieve cosmo_model from likelihood object?
    cosmo_model = tdc_likelihood_list[0].cosmo_model
    for i in range(1,len(tdc_likelihood_list)):
        if tdc_likelihood_list[i].cosmo_model != cosmo_model:
            raise ValueError("")

    # make the variable global to speed up multiprocessing access during the sampling
    global data_vector_global
    data_vector_global = data_vector_list

    log_posterior_fn = partial(log_posterior, cosmo_model=cosmo_model,
        tdc_likelihood_list=tdc_likelihood_list,use_informative=use_informative,
        use_inf_pop=use_inf_pop,use_OmegaM=use_OmegaM,use_tdcosmo25=use_tdcosmo25)
    
    # TODO testing likelihood evaluation
    #hyperparameters = [70.,0.3,-1.,0.,1.,0.1,0.,0.1,2.,0.2]
    #print('log likelihood 1', log_likelihood(hyperparameters,tdc_likelihood_list))
    #log_likelihood(hyperparameters,tdc_likelihood_list)

    # generate initial state
    cur_state = generate_initial_state(n_walkers,cosmo_model,
        use_tdcosmo25=use_tdcosmo25,
        random_seed=init_seed,
        num_cpdf_params=num_cpdf_params)

    # emcee stuff here
    if not use_mpi:
        backend = None
        if backend_path is not None and sampler_type == 'emcee':
            backend = emcee.backends.HDFBackend(backend_path)
            # if False, will pick-up where chain left off
            if reset_backend:
                backend.reset(n_walkers,cur_state.shape[1])
        
        # Single node multiprocessing
        if use_multiprocess:
            from multiprocess import Pool, cpu_count
            cpu_count = cpu_count()
            print("Using multiprocessing for parallelization...")
            print("Number of CPUs: %d" % cpu_count)
            with Pool() as pool:

                if sampler_type == 'emcee':
                    sampler = emcee.EnsembleSampler(n_walkers,cur_state.shape[1],
                        log_posterior_fn, pool=pool, backend=backend)
                    # run mcmc
                    tik_mcmc = time.time()
                    if not reset_backend and backend is not None:
                        # init_state=None will have it pick-up where it left off?
                        _ = sampler.run_mcmc(None,nsteps=num_emcee_samps,progress=False)
                    else:
                        _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps,progress=False)
                    tok_mcmc = time.time()
                    print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))
                elif sampler_type == 'dynesty':
                    raise ValueError("dynesty implementation removed")

        # No multiprocessing
        else:

            if sampler_type == 'emcee':
                sampler = emcee.EnsembleSampler(n_walkers,cur_state.shape[1],
                    log_posterior_fn, backend=backend)
                # run mcmc
                tik_mcmc = time.time()
                if not reset_backend and backend is not None:
                    # init_state=None will have it pick-up where it left off?
                    _ = sampler.run_mcmc(None,nsteps=num_emcee_samps,progress=True)
                else:
                    _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps,progress=True)
                tok_mcmc = time.time()
                print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))

            elif sampler_type == 'dynesty':
                raise ValueError("dynesty sampling removed")

    # MPI
    else: 
        print("Using MPI for parallelization...")
        from schwimmbad import MPIPool
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            # should be safe to put backend here? since only master is running this line?
            backend = None
            if backend_path is not None:
                backend = emcee.backends.HDFBackend(backend_path)
                # if False, will pick-up where chain left off
                if reset_backend:
                    backend.reset(n_walkers,cur_state.shape[1])
                #else:
                #    last_pos = backend.get_last_sample()#.coords
                #    cur_state = last_pos

            if sampler_type == 'emcee':
                sampler = emcee.EnsembleSampler(n_walkers,cur_state.shape[1],
                    log_posterior_fn, pool=pool, backend=backend)
                # run mcmc
                tik_mcmc = time.time()
                if not reset_backend and backend is not None:
                    # init_state=None will have it pick-up where it left off?
                    _ = sampler.run_mcmc(None,nsteps=num_emcee_samps,progress=False)
                else:
                    _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps,progress=False)
                tok_mcmc = time.time()
                print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))
     
            elif sampler_type == 'dynesty':
                raise ValueError("dynesty sampling removed")

    if backend_path is None:
        return sampler.get_chain()