import time
import sys
from functools import partial
import emcee
import jax
import jax.numpy as jnp
import jax_cosmo
import numpy as np
from astropy.cosmology import w0waCDM
from scipy.stats import norm, truncnorm, uniform
import Utils.tdc_utils as tdc_utils

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
            td_measured: array of td_measured
                doubles: (n_lenses,1)
                quads: (n_lenses,3)
            td_likelihood_prec: array of precision matrices:
                - doubles: ((1/sigma^2))
                - quads: ((1/sigma^2 0 0), (0 1/sigma^2 0), (0 0 1/sigma^2))
            fpd_samples: array of fermat potential difference posterior samples
                doubles: (n_lenses,n_fpd_samples,1)
                quads: (n_lenses,n_fpd_samples,3)
            gamma_pred_samples (np.array(float)):
                gamma samples associated with each set of fpd samples.
                (n_lenses,n_fpd_samples)
            z_lens (np.array(float), size:(n_lenses)): lens redshifts
            z_src (np.array(float), size:(n_lenses)): source redshifts
            kappa_ext_samples (np.array(float), size:(n_lenses,n_fpd_samples)):
                default is None (kappa_ext not considered)
            log_prob_gamma_nu_int (callable): function that produces logpdf(values)
                for the modeling prior on gamma_lens (also called nu_int)
            cosmo_model (string): 'LCDM' or 'w0waCDM'
            use_gamma_info (bool): If False, removes reweighting from likelihood
                evaluation (any population level gamma params should just
                return the prior then...)
        """

        # no processing needed (np.squeeze ensures any dimensions of size 1
        #    are removed)
        if cosmo_model not in ['LCDM', 'LCDM_lambda_int',
                               'LCDM_lambda_int_beta_ani', 'w0waCDM', 'w0waCDM_lambda_int_beta_ani']:
            raise ValueError("choose from available cosmo_models: " +
                             "LCDM, LCDM_lambda_int, LCDM_lambda_int_beta_ani, w0waCDM, " +
                             "w0waCDM_lambda_int_beta_ani")
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
        elif self.cosmo_model in ['w0waCDM', 'w0waCDM_lambda_int_beta_ani']:
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
        elif self.cosmo_model == 'LCDM_lambda_int_beta_ani':
            # NOTE: hardcoding of hyperparameter order!! (-6 is mu, -5 is sigma)
            mu_lint = hyperparameters[-6]
            sigma_lint = hyperparameters[-5]
            # truncating to avoid values below 0 (unphysical)
        elif self.cosmo_model == 'w0waCDM_lambda_int_beta_ani':
            # NOTE: hardcoding of hyperparameter order!! (-6 is mu, -5 is sigma)
            mu_lint = hyperparameters[-6]
            sigma_lint = hyperparameters[-5]
            # truncating to avoid values below 0 (unphysical)

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
        # TODO test jitting this
        if USE_JAX:
            td_log_likelihoods = jax_utils.jax_td_log_likelihood_per_samp(
                jnp.asarray(td_pred_samples), jnp.asarray(data_vector_global[index_likelihood_list]['td_measured']),
                jnp.asarray(data_vector_global[index_likelihood_list]['td_likelihood_prec']),
                jnp.asarray(data_vector_global[index_likelihood_list]['td_likelihood_prefactors']))
            td_log_likelihoods = np.asarray(td_log_likelihoods)
        else:
            td_log_likelihoods = self.td_log_likelihood_per_samp(
                td_pred_samples, index_likelihood_list
            )

        # reweighting factor
        # NOTE: hardcoding of hyperparameter order!! (-2 is mu, -1 is sigma)
        #TODO check this
        if self.use_gamma_info:
            eval_at_proposed_nu = norm.logpdf(data_vector_global[index_likelihood_list]['gamma_pred_samples'],
                                              loc=hyperparameters[-2], scale=hyperparameters[-1])
            rw_factor = eval_at_proposed_nu - data_vector_global[index_likelihood_list]['log_prob_gamma_samps_nu_int']
        else:
            rw_factor = 0.

        # sum across fpd samples
        individ_likelihood = np.mean(np.exp(td_log_likelihoods + rw_factor), axis=1)

        # sum over all lenses
        if np.sum(individ_likelihood == 0) > 0:
            return -np.inf

        log_likelihood = np.sum(np.log(individ_likelihood))

        return log_likelihood

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
        # TODO: test jitting this
        if USE_JAX:
            td_log_likelihoods = jax_utils.jax_td_log_likelihood_per_samp(
                jnp.asarray(td_pred_samples) ,jnp.asarray(data_vector_global[index_likelihood_list]['td_measured']),
                jnp.asarray(data_vector_global[index_likelihood_list]['td_likelihood_prec']),
                jnp.asarray(data_vector_global[index_likelihood_list]['td_likelihood_prefactors']))
        else:
            td_log_likelihoods = self.td_log_likelihood_per_samp(
                td_pred_samples, index_likelihood_list
            )
        td_log_likelihoods = np.asarray(td_log_likelihoods)

        # kin log likelihood per sample
        sigma_v_pred_samples = self.sigma_v_pred_from_kin_pred(
            proposed_cosmo, index_likelihood_list, lambda_int_samples)
        # TODO: test jitting this
        if USE_JAX:
            sigma_v_log_likelihoods = jax_utils.jax_sigma_v_log_likelihood_per_samp(
                jnp.asarray(sigma_v_pred_samples),
                jnp.asarray(data_vector_global[index_likelihood_list]['sigma_v_measured']),
                jnp.asarray(data_vector_global[index_likelihood_list]['sigma_v_likelihood_prec']),
                jnp.asarray(data_vector_global[index_likelihood_list]['sigma_v_likelihood_prefactors']))
            sigma_v_log_likelihoods = np.asarray(sigma_v_log_likelihoods)
        else:
            sigma_v_log_likelihoods = self.sigma_v_log_likelihood_per_samp(
                sigma_v_pred_samples, index_likelihood_list
            )

        # reweighting factor
        # NOTE: hardcoding of hyperparameter order!! (-2 is mu, -1 is sigma)
        if self.use_gamma_info:
            eval_at_proposed_nu = norm.logpdf(data_vector_global[index_likelihood_list]['gamma_pred_samples'],
                                              loc=hyperparameters[-2] ,scale=hyperparameters[-1])
            rw_factor = eval_at_proposed_nu - data_vector_global[index_likelihood_list]['log_prob_gamma_samps_nu_int']
        else:
            rw_factor = 0.

        if self.cosmo_model in ['LCDM_lambda_int_beta_ani' ,'w0waCDM_lambda_int_beta_ani']:
            eval_at_proposed_beta_pop = norm.logpdf(data_vector_global[index_likelihood_list]['beta_ani_samples'],
                                                    loc=hyperparameters[-4] ,scale=hyperparameters[-3])
            beta_rw_factor = eval_at_proposed_beta_pop - data_vector_global[index_likelihood_list]['log_prob_beta_ani_samps_nu_int']
            rw_factor += beta_rw_factor

        # TODO: testing some jitting here (jit doesn't like the ==0 statement)

        if USE_JAX:
            individ_likelihood = jax_utils.jax_fpd_samp_summation(jnp.asarray(
                td_log_likelihoods +sigma_v_log_likelihoods +rw_factor))
            individ_likelihood = np.asarray(individ_likelihood)

        else:
            # sum across fpd samples
            individ_likelihood = np.mean(
                np.exp(td_log_likelihoods +sigma_v_log_likelihoods +rw_factor),
                axis=1)

        # sum over all lenses
        # TODO: there is a way to do this in jax
        if jnp.sum(individ_likelihood == 0) > 0:
            log_likelihood = -jnp.inf

        if USE_JAX:
            log_likelihood = jax_utils.jax_lenses_summation(jnp.asarray(individ_likelihood))
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
        # order: [H0,Omega_M,w0,wa,mu_lambda_int,sigma_lambda_int,
        #   mu_beta_ani,sigma_beta_ani,mu_gamma,sigma_gamma]
        cur_state = np.empty((n_walkers,10))
        cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers) #h0
        cur_state[:,1] = uniform.rvs(loc=0.1,scale=0.35,size=n_walkers) #Omega_M
        cur_state[:,2] = uniform.rvs(loc=-1.5,scale=1.,size=n_walkers) #w0
        cur_state[:,3] = uniform.rvs(loc=-1,scale=2,size=n_walkers) #wa
        cur_state[:,4] = uniform.rvs(loc=0.9,scale=0.2,size=n_walkers)
        cur_state[:,5] = uniform.rvs(loc=0.001,scale=0.499,size=n_walkers)
        cur_state[:,6] = uniform.rvs(loc=-0.1,scale=0.2,size=n_walkers)
        cur_state[:,7] = uniform.rvs(loc=0.001,scale=0.199,size=n_walkers)
        cur_state[:,8] = uniform.rvs(loc=1.5,scale=1.,size=n_walkers)
        cur_state[:,9] = uniform.rvs(loc=0.001,scale=0.199,size=n_walkers)

        return cur_state



def log_posterior(hyperparameters, cosmo_model, tdc_likelihood_list):
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
        lp = w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)
    # Likelihood
    if lp == 0:
        for i, tdc_likelihood in enumerate(tdc_likelihood_list):
            fll = tdc_likelihood.full_log_likelihood(hyperparameters, index_likelihood_list = i)
            lp += fll

    return lp

def prepare_data_vector_list(data_vector_list, tdc_likelihood_list):
    """
    Args:
        data_vector_list ([DataVector]): list of data vector objects (dictionnary)
    """

    print("Preparing data vector list for sampling...")

    #expand the axis according to the number of samples in the data vector
    for i in range(len(data_vector_list)):
        #expand the axis according to the number of samples in the data vector
        num_lenses, num_fpd_samples, dim_fpd= data_vector_list[i]['fpd_samples'].shape
        data_vector_list[i]['td_measured'] = np.repeat(data_vector_list[i]['td_measured'][:, np.newaxis, :],
                num_fpd_samples, axis=1)
        data_vector_list[i]['td_likelihood_prec'] = np.repeat(data_vector_list[i]['td_likelihood_prec'][:, np.newaxis, :, :],
            num_fpd_samples, axis=1)

        data_vector_list[i]['td_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(dim_fpd/2)) /
                np.sqrt(np.linalg.det(np.linalg.inv(data_vector_list[i]['td_likelihood_prec']))) )

        # evaluate the modeling prior for gamma_lens from a provided distribution or on the provided samples
        if tdc_likelihood_list[i].log_prob_gamma_nu_int is None:
            #default prior
            log_prob_modeling_prior = uniform.logpdf(data_vector_list[i]['gamma_pred_samples'], loc=1., scale=2.)

        else:
            log_prob_modeling_prior = np.empty((data_vector_list[i]['gamma_pred_samples'].shape))
            for j in range(0, data_vector_list[i]['gamma_pred_samples'].shape[0]):
                log_prob_modeling_prior[i, :] = tdc_likelihood_list[i].log_prob_gamma_nu_int(data_vector_list[i]['gamma_pred_samples'][j])

        data_vector_list[i]['log_prob_modeling_prior'] = log_prob_modeling_prior

        if isinstance(tdc_likelihood_list[i], TDCKinLikelihood):
            #expand the axis according to the number of samples in the data vector
            if 'kin_pred_samples' in data_vector_list[i]:
                num_kin_bins = data_vector_list[i]['kin_pred_samples'].shape[2]
                # pad measurements with a 2nd batch dim for # of fpd samples
                data_vector_list[i]['sigma_v_measured'] = np.repeat(
                    data_vector_list[i]['sigma_v_measured'][:, np.newaxis, :],
                    num_fpd_samples, axis=1)
                data_vector_list[i]['sigma_v_likelihood_prec'] = np.repeat(
                    data_vector_list[i]['sigma_v_likelihood_prec'][:, np.newaxis, :, :],
                    num_fpd_samples, axis=1)

                data_vector_list[i]['sigma_v_likelihood_prefactors'] = np.log((1 / (2 * np.pi) ** (num_kin_bins / 2)) /
                                                                              np.sqrt(np.linalg.det(np.linalg.inv(
                                                                                  data_vector_list[i][
                                                                                      'sigma_v_likelihood_prec']))))
            else:
                raise ValueError('kin_pred_samples not found in data_vector_list whereas TDCKinLikelihood is used')


            if tdc_likelihood_list[i].cosmo_model in ['LCDM_lambda_int_beta_ani' ,'w0waCDM_lambda_int_beta_ani']:
                # check that beta_ani_samples are provided...
                if not 'beta_ani_samples' in data_vector_list[i] or data_vector_list[i]['beta_ani_samples'] is None:
                    raise ValueError('Must provide beta_ani_samples if using  '+
                                     'beta_ani in cosmo_model')

                # evaluate the modeling prior from a provided distribution or on the provided samples
                if tdc_likelihood_list[i].log_prob_beta_ani_nu_int is None:
                    # default: assume un-informative prior
                    data_vector_list[i]['log_prob_beta_ani_nu_int'] = uniform.logpdf(
                        data_vector_list[i]['beta_ani_samples'],loc=-0.5 ,scale=1.)
                else:
                    # user-provided modeling prior
                    data_vector_list[i]['log_prob_beta_ani_nu_int'] = np.empty((data_vector_list[i]['beta_ani_samples'].shape))
                    for j in range(0, data_vector_list[i]['beta_ani_samples'].shape[0]):
                        data_vector_list[i]['log_prob_beta_ani_nu_int'][j, :] = tdc_likelihood_list[i].log_prob_beta_ani_nu_int(
                            data_vector_list[i]['beta_ani_samples'][j])

    return data_vector_list

def fast_TDC(tdc_likelihood_list, data_vector_list, num_emcee_samps=1000,
    n_walkers=20, use_mpi=False, use_multiprocess=False, backend_path=None, reset_backend=True):
    """
    Args:
        tdc_likelihood_list ([TDCLikelihood]): list of likelihood objects 
            (will add log likelihoods together)
        num_emcee_samps (int): Number of iterations for MCMC inference
        n_walkers (int): Number of emcee walkers
        use_mpi (bool): If True, uses MPI for parallelization
        backend_path (string): If not None, saves a backend .h5 file
        
    Returns: 
        mcmc chain (emcee.EnsemblerSampler.chain)
    """

    # Retrieve cosmo_model from likelihood object?
    cosmo_model = tdc_likelihood_list[0].cosmo_model
    for i in range(1,len(tdc_likelihood_list)):
        if tdc_likelihood_list[i].cosmo_model != cosmo_model:
            raise ValueError("")

    # TODO: prepare the data vectors

    # make the variable global to speed up multiprocessing access during the sampling
    global data_vector_global
    data_vector_global = data_vector_list

    log_posterior_fn = partial(log_posterior, cosmo_model=cosmo_model,
        tdc_likelihood_list=tdc_likelihood_list)

    # generate initial state
    cur_state = generate_initial_state(n_walkers,cosmo_model)

    print('Initial likelihood call : ')
    hyperparameters_init = [6.47242793e+01, 2.33623746e-01, 9.14884078e-01, 1.34268817e-01,
                            -3.15238075e-02, 8.41097628e-02, 2.22904902e+00, 8.78549539e-03]
    print('hyperparameters_init', hyperparameters_init)
    log_post_val = log_posterior(hyperparameters_init, cosmo_model,
                  tdc_likelihood_list)
    print('log_posterior', log_post_val)
    print('expected : log_posterior -1309.0113851331898')

    # emcee stuff here
    if not use_mpi:
        backend = None
        if backend_path is not None:
            backend = emcee.backends.HDFBackend(backend_path)
            # if False, will pick-up where chain left off
            if reset_backend:
                backend.reset(n_walkers,cur_state.shape[1])

        if use_multiprocess:
            from multiprocess import Pool, cpu_count
            cpu_count = cpu_count()
            print("Using multiprocessing for parallelization...")
            print("Number of CPUs: %d" % cpu_count)
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(n_walkers,cur_state.shape[1],
                    log_posterior_fn,backend=backend, pool=pool)

                # run mcmc
                tik_mcmc = time.time()
                _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps,progress=False)
                tok_mcmc = time.time()
                print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))
        else:
            sampler = emcee.EnsembleSampler(n_walkers, cur_state.shape[1],
                                            log_posterior_fn, backend=backend)

            # run mcmc
            tik_mcmc = time.time()
            _ = sampler.run_mcmc(cur_state, nsteps=num_emcee_samps, progress=True)
            tok_mcmc = time.time()
            print("Avg. Time per MCMC Step: %.3f seconds" % ((tok_mcmc - tik_mcmc) / num_emcee_samps))
    else: 
        print("Using MPI for parallelization...")
        from schwimmbad import MPIPool
        with MPIPool() as pool:
            if not pool.is_master():
                pool.wait()
                sys.exit(0)
            sampler = emcee.EnsembleSampler(n_walkers,cur_state.shape[1],log_posterior_fn, pool=pool)
            # run mcmc
            tik_mcmc = time.time()
            _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps,progress=False)
            tok_mcmc = time.time()
            print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))
        
    return sampler.get_chain()