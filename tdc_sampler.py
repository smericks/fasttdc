import time
import sys
from functools import partial
import emcee
import dynesty
import jax
import jax.numpy as jnp
import jax_cosmo
import numpy as np
from astropy.cosmology import w0waCDM
from scipy.stats import norm, truncnorm, uniform, multivariate_normal
import Utils.tdc_utils as tdc_utils
import math

USE_JAX = False

if USE_JAX:
    import tdc_jax_utils as jax_utils
"""
cosmo_models available: 
    'LCDM': [H0,OmegaM,mu(gamma_lens),sigma(gamma_lens)]
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
        """

        # no processing needed (np.squeeze ensures any dimensions of size 1
        #    are removed)
        if cosmo_model not in ['LCDM', 'LCDM_lambda_int',
                               'LCDM_lambda_int_beta_ani', 'w0waCDM', 
                               'w0waCDM_lambda_int_beta_ani',
                               'w0waCDM_fullcPDF','w0waCDM_fullcPDF_noKIN']:
            raise ValueError("choose from available cosmo_models: " +
                             "LCDM, LCDM_lambda_int, LCDM_lambda_int_beta_ani, w0waCDM, " +
                             "w0waCDM_lambda_int_beta_ani, w0waCDM_fullcPDF")
        self.cosmo_model = cosmo_model
        self.use_gamma_info = use_gamma_info
        self.use_astropy = use_astropy
        # make sure the dims are right
        self.num_lenses, self.num_fpd_samples, self.dim_fpd = fpd_sample_shape

    # compute predicted time delays from predicted fermat potential differences
    # requires an assumed cosmology (from hyperparameters) and redshifts


    def td_pred_from_fpd_pred(self, proposed_cosmo, index_likelihood_list, lambda_int_samples=None, ):
        """
        Args:
            proposed_cosmo (default: jax_cosmo.Cosmology): built by
                construct_proposed_cosmo() (see below)
            lambda_int_samples (): shape=(num_lenses,num_fpd_samples)

        Returns:
            td_pred_samples (size:(n_lenses,n_samples,3))
        """

        if self.use_astropy:
            Ddt_computed = tdc_utils.ddt_from_redshifts(proposed_cosmo,
                                                        data_vector_global[index_likelihood_list]['z_lens'],
                                                        data_vector_global[index_likelihood_list]['z_src'])
        else:
            Ddt_computed = tdc_utils.jax_ddt_from_redshifts(proposed_cosmo,
                                                            data_vector_global[index_likelihood_list]['z_lens'],
                                                            data_vector_global[index_likelihood_list]['z_src'])

        Ddt_computed = np.array(Ddt_computed)
        # add batch dimensions for Ddt computed...
        Ddt_repeated = np.repeat(Ddt_computed[:, np.newaxis],
                                 self.num_fpd_samples, axis=1)
        Ddt_repeated = np.repeat(Ddt_repeated[:, :, np.newaxis],
                                 self.dim_fpd, axis=2)
        # compute predicted time delays (this function should work w/ arrays)
        td_pred = tdc_utils.td_from_ddt_fpd(Ddt_repeated, data_vector_global[index_likelihood_list]['fpd_samples'])

        # Account for mass sheets:
        #   td = lambda * td
        #   lambda = (1-kappa_ext)*lambda_int

        # Linear scaling if lambda_int is present
        if lambda_int_samples is not None:
            lambda_int_repeated = np.repeat(lambda_int_samples[:, :, np.newaxis],
                                            self.dim_fpd, axis=2)
            td_pred *= lambda_int_repeated
        # Scaling if kappa_ext is present...
        if data_vector_global[index_likelihood_list]['kappa_ext_samples'] is not None:
            kappa_ext_repeated = np.repeat(data_vector_global[index_likelihood_list]['kappa_ext_samples'][:, :, np.newaxis],
                                           self.dim_fpd, axis=2)
            td_pred *= (1 - kappa_ext_repeated)

        return td_pred

    def td_log_likelihood_per_samp(self, td_pred_samples, index_likelihood_list):
        """
        Args:
            td_pred_samples (n_lenses,n_fpd_samps,3)

        Returns:
            td_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        x_minus_mu = (td_pred_samples - data_vector_global[index_likelihood_list]['td_measured'])
        # add dimension s.t. x_minus_mu is 2D
        x_minus_mu = np.expand_dims(x_minus_mu, axis=-1)
        # matmul should condense the (# of time delays) dim.
        exponent = -0.5 * np.matmul(np.transpose(x_minus_mu, axes=(0, 1, 3, 2)),
                                    np.matmul(data_vector_global[index_likelihood_list]['td_likelihood_prec'], x_minus_mu))

        # reduce to two dimensions: (n_lenses,n_fpd_samples)
        exponent = np.squeeze(exponent)

        # log-likelihood
        return data_vector_global[index_likelihood_list]['td_likelihood_prefactors'] + exponent


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

    def full_log_likelihood(self, hyperparameters, index_likelihood_list):
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
        td_pred_samples = self.td_pred_from_fpd_pred(proposed_cosmo, index_likelihood_list,
                                                     lambda_int_samples)
        td_log_likelihoods = self.td_log_likelihood_per_samp(
            td_pred_samples, index_likelihood_list
        )

        # reweighting factor
        # TODO: fix this for new framework
        if self.use_gamma_info:
            rw_factor = self.compute_rw_factor(hyperparameters,index_likelihood_list)
        else:
            rw_factor = 0.

        # sum across fpd samples
        individ_likelihood = np.mean(np.exp(td_log_likelihoods + rw_factor), axis=1)

        # sum over all lenses
        if np.sum(individ_likelihood == 0) > 0:
            return -np.inf

        log_likelihood = np.sum(np.log(individ_likelihood))

        return log_likelihood
    
    def compute_rw_factor(self,hyperparameters,index_likelihood_list):

        # retrieve interim hypermodel
        nu_means = data_vector_global[index_likelihood_list]['lens_params_nu_int_means']
        nu_stddevs = data_vector_global[index_likelihood_list]['lens_params_nu_int_stddevs']

        # modify into proposed hypermodel
        if self.cosmo_model == 'w0waCDM_fullcPDF':
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

        elif self.cosmo_model == 'w0waCDM_fullcPDF_noKIN': 
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

        else: # all other models
            # only change gamma_lens
            nu_means[3] = hyperparameters[-2]
            nu_stddevs[3] = hyperparameters[-1]

        # TODO: check if this returns the right dimensional thing
        eval_at_proposed_nu = multivariate_normal.logpdf(
            data_vector_global[index_likelihood_list]['lens_param_samples'],
            mean=nu_means,
            cov=np.diag(nu_stddevs**2))

        rw_factor = (eval_at_proposed_nu - 
            data_vector_global[index_likelihood_list]['log_prob_lens_param_samps_nu_int'])
        
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
            exponent = np.squeeze(exponent)

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
            beta_ani_samples (): None if beta_ani not in population model
                (n_lenses,n_fpd_samples)
        """

        super().__init__(fpd_sample_shape, cosmo_model ,use_gamma_info,
                         use_astropy)

        self.num_kin_bins = kin_pred_samples_shape[2]


    def sigma_v_pred_from_kin_pred(self ,proposed_cosmo,index_likelihood_list, lambda_int_samples=None):
        """
        Args:
            proposed_cosmo (default: jax_cosmo.Cosmology): built by
                construct_proposed_cosmo() (see below)
            lambda_int_samples (): shape=(num_lenses,num_fpd_samples)
        """

        if self.use_astropy:
            Ds_div_Dds_computed = tdc_utils.kin_distance_ratio(
                proposed_cosmo , data_vector_global[index_likelihood_list]['z_lens'],
                data_vector_global[index_likelihood_list]['z_src'])

            # raise ValueError("astropy option not implemented for TDC+Kin")
        else:
            Ds_div_Dds_computed = tdc_utils.jax_kin_distance_ratio(
                proposed_cosmo, data_vector_global[index_likelihood_list]['z_lens'],
                data_vector_global[index_likelihood_list]['z_src'])

        Ds_div_Dds_computed = np.array(Ds_div_Dds_computed)
        # add batch dimensions for fpd_samples
        Ds_div_Dds_repeated = np.repeat(Ds_div_Dds_computed[:, np.newaxis],
                                        self.num_fpd_samples, axis=1)
        # add batch dimension for # kinematic bins
        Ds_div_Dds_repeated = np.repeat(Ds_div_Dds_repeated[:, :, np.newaxis],
                                        self.num_kin_bins, axis=2)
        # scale the kin_pred with cosmology term: sigma_v = sqrt(Ds/Dds)*c*sqrt(mathcal{J})
        sigma_v_pred = np.sqrt(Ds_div_Dds_repeated ) *data_vector_global[index_likelihood_list]['kin_pred_samples']

        # Account for mass sheets:
        #   sigma_v = sqrt(lambda) * sigma_v
        #   lambda = (1-kappa_ext)*lambda_int

        # sqrt(lambda) scaling if lambda_int is present
        if lambda_int_samples is not None:
            lambda_int_repeated = np.repeat(lambda_int_samples[: ,: ,np.newaxis],
                                            self.num_kin_bins, axis=2)
            sigma_v_pred *= np.sqrt(lambda_int_repeated)
        # sqrt(1-kappa_ext) scaling
        if data_vector_global[index_likelihood_list]['kappa_ext_samples'] is not None:
            kappa_ext_repeated = np.repeat(data_vector_global[index_likelihood_list]['kappa_ext_samples'][: ,: ,np.newaxis],
                                           self.num_kin_bins, axis=2)
            sigma_v_pred *= np.sqrt(1 - kappa_ext_repeated)

        return sigma_v_pred

    # TODO: jaxify & jit
    def sigma_v_log_likelihood_per_samp(self,sigma_v_pred_samples, index_likelihood_list):
        """
        Args:
            sigma_v_pred_samples (n_lenses,n_fpd_samps,num_kin_bins)

        Returns:
            sigma_v_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        x_minus_mu = (sigma_v_pred_samples -data_vector_global[index_likelihood_list]['sigma_v_measured'])
        # add dimension s.t. x_minus_mu is 2D
        x_minus_mu = np.expand_dims(x_minus_mu ,axis=-1)
        # matmul should condense the (# of time delays) dim.
        exponent = -0.5 *np.matmul(np.transpose(x_minus_mu ,axes=(0 ,1 ,3 ,2)),
                                  np.matmul(data_vector_global[index_likelihood_list]['sigma_v_likelihood_prec'],x_minus_mu))

        # reduce to two dimensions: (n_lenses,n_fpd_samples)
        exponent = np.squeeze(exponent)

        # log-likelihood
        return data_vector_global[index_likelihood_list]['sigma_v_likelihood_prefactors'] + exponent
    

    def full_log_likelihood(self, hyperparameters, index_likelihood_list):

        # construct cosmology from hyperparameters
        proposed_cosmo, lambda_int_samples = self.process_hyperparam_proposal(
            hyperparameters)

        # td log likelihood per sample
        td_pred_samples = self.td_pred_from_fpd_pred(
            proposed_cosmo, index_likelihood_list, lambda_int_samples)
        td_log_likelihoods = self.td_log_likelihood_per_samp(
            td_pred_samples, index_likelihood_list
        )
        td_log_likelihoods = np.asarray(td_log_likelihoods)

        # kin log likelihood per sample
        sigma_v_pred_samples = self.sigma_v_pred_from_kin_pred(
            proposed_cosmo, index_likelihood_list, lambda_int_samples)
        sigma_v_log_likelihoods = self.sigma_v_log_likelihood_per_samp(
            sigma_v_pred_samples, index_likelihood_list
        )

        # reweighting factor
        # NOTE: hardcoding of hyperparameter order!! (-2 is mu, -1 is sigma)
        if self.use_gamma_info:
            rw_factor = self.compute_rw_factor(hyperparameters,index_likelihood_list)
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

            # compare proposed prob to modeling prior prob
            eval_at_proposed_beta_pop = norm.logpdf(data_vector_global[index_likelihood_list]['beta_ani_samples'],
                loc=proposed_loc,scale=proposed_scale)
            beta_rw_factor = (eval_at_proposed_beta_pop - 
                data_vector_global[index_likelihood_list]['log_prob_beta_ani_samps_nu_int'])
            # additive in log space
            rw_factor += beta_rw_factor

        individ_likelihood = np.mean(
            np.exp(td_log_likelihoods +sigma_v_log_likelihoods +rw_factor),
            axis=1)
        
        
        # sum over all lenses
        # TODO: there is a way to do this in jax
        if np.sum(individ_likelihood == 0) > 0:
            log_likelihood = -jnp.inf

        else:
            log_likelihood = np.sum(np.log(individ_likelihood))


        return log_likelihood
    

#########################
# Sampling Implementation
#########################

def LCDM_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    if hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    elif hyperparameters[2] < 1.5 or hyperparameters[2] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[3] < 0.001 or hyperparameters[3] > 0.2: #sigma(gamma_lens)
        return -np.inf
    
    return 0

def LCDM_lambda_int_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_lambda_int,sigma_lambda_int,
            mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    if hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    elif hyperparameters[2] < 0.5 or hyperparameters[2] > 1.5: #mu(lambda_int)
        return -np.inf
    elif hyperparameters[3] < 0.001 or hyperparameters[3] > 0.5: #sigma(lambda_int)
        return -np.inf
    elif hyperparameters[4] < 1.5 or hyperparameters[4] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[5] < 0.001 or hyperparameters[5] > 0.2: #sigma(gamma_lens)
        return -np.inf
    
    return 0

def LCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_lambda_int,sigma_lambda_int,
            mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    if hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    elif hyperparameters[2] < 0.5 or hyperparameters[2] > 1.5: #mu(lambda_int)
        return -np.inf
    elif hyperparameters[3] < 0.001 or hyperparameters[3] > 0.5: #sigma(lambda_int)
        return -np.inf
    elif hyperparameters[4] < -0.5 or hyperparameters[4] > 0.5: #mu(beta_ani)
        return -np.inf
    elif hyperparameters[5] < 0.001 or hyperparameters[5] > 0.2: #sigma(beta_ani)
        return -np.inf
    elif hyperparameters[6] < 1.5 or hyperparameters[4] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[7] < 0.001 or hyperparameters[5] > 0.2: #sigma(gamma_lens)
        return -np.inf
    
    return 0

def w0waCDM_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,Omega_M,w0,wa,mu_gamma,sigma_gamma])
    """

    # h0 [0,150]
    if hyperparameters[0] < 0 or hyperparameters[0] > 150: 
        return -np.inf
    # Omega_M [0.05,0.5]
    if hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: 
        return -np.inf
    #w0 [-2,0]
    elif hyperparameters[2] < -2 or hyperparameters[2] > 0:
        return -np.inf
    #wa [-2,2]
    elif hyperparameters[3] < -2 or hyperparameters[3] > 2:
        return -np.inf
    #mu(gamma)
    elif hyperparameters[4] < 1.5 or hyperparameters[4] > 2.5:
        return -np.inf
    #sigma(gamma)
    elif hyperparameters[5] < 0.001 or hyperparameters[5] > 0.2:
        return -np.inf
    
    return 0

def w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_lambda_int,sigma_lambda_int,
            mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    elif hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    #w0 [-2,0]
    elif hyperparameters[2] < -2 or hyperparameters[2] > 0:
        return -np.inf
    #wa [-2,2]
    elif hyperparameters[3] < -2 or hyperparameters[3] > 2:
        return -np.inf
    elif hyperparameters[4] < 0.5 or hyperparameters[4] > 1.5: #mu(lambda_int)
        return -np.inf
    elif hyperparameters[5] < 0.001 or hyperparameters[5] > 0.5: #sigma(lambda_int)
        return -np.inf
    elif hyperparameters[6] < -0.5 or hyperparameters[6] > 0.5: #mu(beta_ani)
        return -np.inf
    elif hyperparameters[7] < 0.001 or hyperparameters[7] > 0.2: #sigma(beta_ani)
        return -np.inf
    elif hyperparameters[8] < 1.5 or hyperparameters[8] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[9] < 0.001 or hyperparameters[9] > 0.2: #sigma(gamma_lens)
        return -np.inf
    
    return 0

def INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters):

    # returns 0 or -np.inf
    within_bounds = w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    if within_bounds == 0:
        """
        HARCODED_COV = [[ 3.44223963e+00, -4.11838162e-02, -2.41929684e-01,
            6.79514752e-01,  1.21649708e-02,  1.41206148e-03,
            -1.62017974e-03,  1.25331307e-04],
        [-4.11838162e-02,  7.86808826e-03, -9.04086446e-03,
            -3.23963110e-02, -1.12975063e-03,  4.37048533e-05,
            1.19877612e-04,  4.91279229e-05],
        [-2.41929684e-01, -9.04086446e-03,  4.77945452e-02,
            -6.15320898e-02,  1.89141384e-03, -1.62559966e-04,
            -4.46466403e-04, -2.48102582e-04],
        [ 6.79514752e-01, -3.23963110e-02, -6.15320898e-02,
            5.57771936e-01,  5.72993589e-04, -2.47347909e-05,
            -1.33506956e-05,  1.97205440e-04],
        [ 1.21649708e-02, -1.12975063e-03,  1.89141384e-03,
            5.72993589e-04,  3.81515870e-04, -2.60852078e-06,
            -5.07138565e-05, -2.90699248e-05],
        [ 1.41206148e-03,  4.37048533e-05, -1.62559966e-04,
            -2.47347909e-05, -2.60852078e-06,  5.86037706e-05,
            -1.60569404e-05,  2.57434106e-06],
        [-1.62017974e-03,  1.19877612e-04, -4.46466403e-04,
            -1.33506956e-05, -5.07138565e-05, -1.60569404e-05,
            3.49452453e-04,  1.17920808e-04],
        [ 1.25331307e-04,  4.91279229e-05, -2.48102582e-04,
            1.97205440e-04, -2.90699248e-05,  2.57434106e-06,
            1.17920808e-04,  2.08479045e-04]]
        """
        HARCODED_COV = np.asarray([[ 6.12374896e+00, -2.15141853e-02, -5.71087181e-01,
            6.32316635e-01,  6.94284277e-03,  2.95354324e-06,
            -1.26510967e-02,  1.61769785e-03],
        [-2.15141853e-02,  8.55982425e-03, -1.62265553e-02,
            -5.83908439e-02, -1.32491066e-03, -2.36135703e-05,
            -8.11721417e-05, -2.57776450e-04],
        [-5.71087181e-01, -1.62265553e-02,  1.19990785e-01,
            -3.55331429e-02,  4.35775938e-03,  1.29247499e-04,
            9.69690762e-04,  2.02107237e-04],
        [ 6.32316635e-01, -5.83908439e-02, -3.55331429e-02,
            1.29002622e+00,  3.60024828e-03, -4.88360131e-04,
            7.42754178e-04,  3.08261219e-03],
        [ 6.94284277e-03, -1.32491066e-03,  4.35775938e-03,
            3.60024828e-03,  5.62846938e-04,  9.40058479e-06,
            -7.31255777e-05,  2.98668773e-05],
        [ 2.95354324e-06, -2.36135703e-05,  1.29247499e-04,
            -4.88360131e-04,  9.40058479e-06,  1.13811882e-04,
            9.53635163e-06,  8.34327893e-06],
        [-1.26510967e-02, -8.11721417e-05,  9.69690762e-04,
            7.42754178e-04, -7.31255777e-05,  9.53635163e-06,
            6.78136734e-04,  4.25489473e-05],
        [ 1.61769785e-03, -2.57776450e-04,  2.02107237e-04,
            3.08261219e-03,  2.98668773e-05,  8.34327893e-06,
            4.25489473e-05,  7.58572080e-04]])
            
        HARDCODED_MEAN = np.asarray([ 70.,  0.3, -1.,  0.,
            1.,  0.1,  0.,  0.1])
        
        return multivariate_normal.logpdf(hyperparameters[4:8],
            mean=HARDCODED_MEAN[4:],cov=HARCODED_COV[4:,4:])

    else:
        return within_bounds
        


def w0waCDM_fullcPDF_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_lambda_int,sigma_lambda_int,
            mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    elif hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    #w0 [-2,0]
    elif hyperparameters[2] < -2 or hyperparameters[2] > 0:
        return -np.inf
    #wa [-2,2]
    elif hyperparameters[3] < -2 or hyperparameters[3] > 2:
        return -np.inf
    elif hyperparameters[4] < 0.5 or hyperparameters[4] > 1.5: #mu(lambda_int)
        return -np.inf
    elif hyperparameters[5] < 0.001 or hyperparameters[5] > 0.5: #sigma(lambda_int)
        return -np.inf
    elif hyperparameters[6] < -0.5 or hyperparameters[6] > 0.5: #mu(beta_ani)
        return -np.inf
    elif hyperparameters[7] < 0.001 or hyperparameters[7] > 0.2: #sigma(beta_ani)
        return -np.inf
    # LENS PARAMS
    elif hyperparameters[8] < 1.5 or hyperparameters[8] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[9] < 0.001 or hyperparameters[9] > 0.2: #sigma(gamma_lens)
        return -np.inf
    elif hyperparameters[10] < 0.2 or hyperparameters[10] > 2.0: #mu(theta_E)
        return -np.inf
    elif hyperparameters[11] < 0.001 or hyperparameters[11] > 0.7: #sigma(theta_E)
        return -np.inf
    elif hyperparameters[12] < 0.001 or hyperparameters[12] > 0.1: #sigma(gamma1/2)
        return -np.inf
    elif hyperparameters[13] < 0.001 or hyperparameters[13] > 0.2: #sigma(e1/2)
        return -np.inf
    
    return 0

def w0waCDM_fullcPDF_noKIN_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_lambda_int,sigma_lambda_int,
            mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    elif hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    #w0 [-2,0]
    elif hyperparameters[2] < -2 or hyperparameters[2] > 0:
        return -np.inf
    #wa [-2,2]
    elif hyperparameters[3] < -2 or hyperparameters[3] > 2:
        return -np.inf
    # LENS PARAMS
    elif hyperparameters[4] < 1.5 or hyperparameters[4] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[5] < 0.001 or hyperparameters[5] > 0.2: #sigma(gamma_lens)
        return -np.inf
    elif hyperparameters[6] < 0.2 or hyperparameters[6] > 2.0: #mu(theta_E)
        return -np.inf
    elif hyperparameters[7] < 0.001 or hyperparameters[7] > 0.7: #sigma(theta_E)
        return -np.inf
    elif hyperparameters[8] < 0.001 or hyperparameters[8] > 0.1: #sigma(gamma1/2)
        return -np.inf
    elif hyperparameters[9] < 0.001 or hyperparameters[9] > 0.2: #sigma(e1/2)
        return -np.inf
    
    return 0



def dynesty_prior_transform(uniform_draw):
    """Transforms the uniform random variable `u ~ Unif[0., 1.)`
    to the parameter of interest `x ~ Unif[-10., 10.)`."""

    x = uniform_draw
    # H0
    x[0] = 150*x[0] # scale to [0,150.]
    # OmegaM
    x[1] = 0.45*x[1] + 0.05 # scale to [0,.45], shift to [0.05,0.5]
    # w0
    x[2] = 2*x[2] - 2. # scale to [0,2.], shift to [-2,0.]
    # wa
    x[3] = 4*x[3] - 2. # scale to [0,4.], shift to [-2,2.]
    #mu(lambda_int)
    x[4] = x[4] + 0.5 # scale to [0,1.], shift to [0.5,1.5]
    # sigma(lambda_int)
    x[5] = 0.499*x[5] + 0.001 # scale to [0,0.499], shift to [0.001,0.5]
    # mu(beta_ani)
    x[6] = x[6] - 0.5 # scale to [0,1.], shift to [-0.5,0.5]
    # sigma(beta_ani)
    x[7] = 0.199*x[7] + 0.001 # scale to [0,0.199], shift to [0.001,0.2]
    # mu(gamma_lens)
    x[8] = x[8] + 1.5 # scale to [0,1.], shift to [1.5,2.5]
    # sigma(gamma_lens)
    x[9] = 0.199*x[9] + 0.001 # scale to [0,0.199], shift to [0.001,0.2]

    return x

def generate_initial_state(n_walkers,cosmo_model):
    """
    Args:
        n_walkers (int): number of emcee walkers
        cosmo_model (string): 'LCDM' or 'w0waCDM'
    """

    if cosmo_model == 'LCDM':
        # order: [H0,Omega_M,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,4))
        cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers) #h0
        cur_state[:,1] = uniform.rvs(loc=0.1,scale=0.35,size=n_walkers) #Omega_M
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
        fll += tdc_likelihood.full_log_likelihood(hyperparameters, index_likelihood_list = i)
    return fll

def log_posterior(hyperparameters, cosmo_model, tdc_likelihood_list,
    use_informative=False):
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
        lp = LCDM_lambda_int_beta_ani_log_prior(hyperparameters)
    elif cosmo_model == 'w0waCDM':
        lp = w0waCDM_log_prior(hyperparameters)
    elif cosmo_model == 'w0waCDM_lambda_int_beta_ani':
        if use_informative:
            lp = INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
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
    reset_backend=True,sampler_type='emcee',use_informative=False):
    """
    Args:
        tdc_likelihood_list ([TDCLikelihood]): list of likelihood objects 
            (will add log likelihoods together)
        num_emcee_samps (int): Number of iterations for MCMC inference
        n_walkers (int): Number of emcee walkers
        use_mpi (bool): If True, uses MPI for parallelization
        backend_path (string): If not None, saves a backend .h5 file. 
            Otherwise, returns the chain.
        sampler_type (string): 'emcee' or 'dynesty'
        
    Returns: 
        mcmc chain (emcee.EnsemblerSampler.chain or dynesty.NestedSampler.)
    """

    # Retrieve cosmo_model from likelihood object?
    cosmo_model = tdc_likelihood_list[0].cosmo_model
    for i in range(1,len(tdc_likelihood_list)):
        if tdc_likelihood_list[i].cosmo_model != cosmo_model:
            raise ValueError("")
        

    # dynesty only works with one model rn because of a hardcoded prior transform
    if sampler_type == 'dynesty' and cosmo_model != 'w0waCDM_lambda_int_beta_ani':
        raise ValueError('dynesty sampling not implemented for chosen cosmology')

    # make the variable global to speed up multiprocessing access during the sampling
    global data_vector_global
    data_vector_global = data_vector_list

    log_posterior_fn = partial(log_posterior, cosmo_model=cosmo_model,
        tdc_likelihood_list=tdc_likelihood_list,use_informative=use_informative)
    # need this fnc for dynesty
    log_likelihood_fn = partial(log_likelihood,
        tdc_likelihood_list=tdc_likelihood_list)
    
    # TODO testing likelihood evaluation
    #hyperparameters = [70.,0.3,-1.,0.,1.,0.1,0.,0.1,2.,0.2]
    #print('log likelihood 1', log_likelihood(hyperparameters,tdc_likelihood_list))
    #log_likelihood(hyperparameters,tdc_likelihood_list)

    # generate initial state
    cur_state = generate_initial_state(n_walkers,cosmo_model)

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
                    sampler = dynesty.NestedSampler(loglikelihood=log_likelihood_fn,
                        prior_transform=dynesty_prior_transform,
                        ndim=10) # TODO: fix hard-coding of ndim for dynesty!!!!!
                    sampler.run_nested(maxiter=num_emcee_samps,
                        checkpoint_file=backend_path,checkpoint_every=60)

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
                # TODO: test with dynamic nested sampler instead...
                # bound='single', sample='unif',rstate=rstate
                dsampler = dynesty.DynamicNestedSampler(loglikelihood=log_likelihood_fn,
                    prior_transform=dynesty_prior_transform,
                    ndim=10, bound='single', sample='unif')
                #sampler = dynesty.NestedSampler(loglikelihood=log_likelihood_fn,
                #    prior_transform=dynesty_prior_transform,
                #    ndim=10) # TODO: fix hard-coding of ndim for dynesty!!!!!
                print('checkpointing to: ', backend_path)
                dsampler.run_nested(maxiter=num_emcee_samps,
                    checkpoint_file=backend_path,checkpoint_every=10)
                #sampler.run_nested(maxiter=num_emcee_samps,
                #    checkpoint_file=backend_path,checkpoint_every=10)

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
                sampler = dynesty.NestedSampler(loglikelihood=log_likelihood_fn,
                    prior_transform=dynesty_prior_transform,
                    ndim=10) # TODO: fix hard-coding of ndim for dynesty!!!!!
                sampler.run_nested(maxiter=num_emcee_samps,
                    checkpoint_file=backend_path,checkpoint_every=60)
                
                results = sampler.results
                samples_equal = results.samples_equal()

    if backend_path is None:
        return sampler.get_chain()