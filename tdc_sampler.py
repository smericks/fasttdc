import jax_cosmo
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm, uniform, gaussian_kde
import tdc_utils
import emcee
import time

#################################
# Preprocessing of TDC Quantities
#################################

def preprocess_td_measured(td_measured,td_cov,fpd_samples):
    """ Takes in output from LensSample.tdc_sampler_input() and pre-processes
        it for fast TDC inference

    Constructs prefactors, 3d precision matrices, and 3d td_measured such that
        double and quad likelihoods can be evaluated per lens at the same time.
        For lenses with only one time delay, prefactors and "precision matrices" 
        are constructed s.t. when they are used for 3D Gaussian evaluation, it 
        is equivalent to evaluating a 1D Gaussian.

    This means that empty values are padded with !!!zeros!!! 

    Args:
        td_measured (np.array(float), size:(n_lenses,3)): list of 1d arrays of 
            time delay measurements for each lens
        td_cov (np.array(float), size:(n_lenses,3,3)): list of 2d arrays of 
            time delay covariance matrices, padded with Nans
        fpd_samples (list of [float,float]): list of !!variable size!! 2d arrays
            of fpd samples. Some will have dim (1,n_samples), others will have dim
            (3,n_samples)

    Returns:
        array of td_measured_padded (n_lenses,3)
        array of fpd_samples_padded (n_lenses,n_fpd_samples,3)
        array of log-space additive prefactors: log( (1/(2pi)^k/2) * 1/sqrt(det(Sigma)) )
            - doubles: k=1,det(Sigma)=det([sigma^2])
            - quads: k=3, det(Sigma)=det(Sigma)
        array of precision matrices: 
            - doubles: ((1/sigma^2 0 0 ),(0 0 0),(0 0 0 )
            - quads: (1/sigma^2 0 0), (0 1/sigma^2 0), (0 0 1/sigma^2)
    """
    
    num_lenses = len(td_measured)
    num_fpd_samples = len(fpd_samples[0][0])
    # HARDCODED TO 3 (max we use are quads)
    td_measured_padded = np.empty((num_lenses,3))
    fpd_samples_padded = np.empty((num_lenses,num_fpd_samples,3))
    td_likelihood_prefactors = np.empty((num_lenses))
    td_likelihood_prec = np.empty((num_lenses,3,3))

    # I'm not sure if I can use indexing for conditioning b/c of the variable length
    for l in range(0,num_lenses):
        num_td = len(fpd_samples[l])
        # doubles
        if num_td ==1:
            td_measured_padded[l] = [td_measured[l][0],0,0]
            td_likelihood_prec[l] = np.asarray([[1/td_cov[l][0][0],0,0],
                [0,0,0],[0,0,0]])
            fpd_samples_padded[l] = np.asarray([fpd_samples[l][0],
                np.zeros((num_fpd_samples)),np.zeros((num_fpd_samples))]).T
            # 1 / (sigma*sqrt(2pi))
            td_likelihood_prefactors[l] = ((1/(2*np.pi))**(num_td/2) 
                / np.sqrt(td_cov[l][0][0]))
        # quads
        elif num_td == 3:
            td_measured_padded[l] = td_measured[l]
            td_likelihood_prec[l] = np.linalg.inv(np.asarray(td_cov[l]))
            fpd_samples_padded[l] = np.asarray(fpd_samples[l]).T
            td_likelihood_prefactors[l] = ((1/(2*np.pi))**(num_td/2) 
                / np.sqrt(np.linalg.det(td_cov[l])))
        else:
            print(("Number of time delays must be 1 or 3"+ 
                "lens %d has %d time delays"%(l,len(td_measured[l]))))
            raise ValueError
        
    return td_measured_padded, fpd_samples_padded, np.log(td_likelihood_prefactors), td_likelihood_prec


###########################
# TDC Likelihood Functions
###########################

# I think we want this to be a class, so we can keep track of quantities
# internally 
class TDCLikelihood():

    def __init__(self,td_measured_padded,td_likelihood_prec,td_likelihood_prefactors,
        fpd_samples_padded,gamma_pred_samples,z_lens,z_src,
        log_prob_modeling_prior=None,cosmo_model='LCDM'):
        """
        Keep track of quantities that remain constant throughout the inference

        Args: 
            td_measured_padded: array of td_measured, doubles padded w/ zeros 
                (n_lenses,3)
            td_likelihood_prec: array of precision matrices: 
                - doubles: ((1/sigma^2 0 0 ),(0 0 0),(0 0 0 )
                - quads: (1/sigma^2 0 0), (0 1/sigma^2 0), (0 0 1/sigma^2)
            td_likelihood_prec:  array of log-space additive prefactors: 
                log( (1/(2pi)^k/2) * 1/sqrt(det(Sigma)) )
                - doubles: k=1,det(Sigma)=det([sigma^2])
                - quads: k=3, det(Sigma)=det(Sigma)
            fpd_samples_padded: array of fpd_samples, doubles padded w/ zeros
                (n_lenses,n_fpd_samples,3)
            gamma_pred_samples (np.array(float)): 
                gamma samples associated with each set of fpd samples.
                (n_lenses,n_samples)
            z_lens (np.array(float), size:(n_lenses)): lens redshifts
            z_src (np.array(float), size:(n_lenses)): source redshifts
            log_prob_modeling_prior: TODO
            cosmo_model (string): 'LCDM' or 'w0waCDM'
        """

        # no processing needed (np.squeeze ensures any dimensions of size 1
        #    are removed)
        self.z_lens = np.squeeze(np.asarray(z_lens))
        self.z_src = np.squeeze(np.asarray(z_src))
        self.cosmo_model = cosmo_model
        # make sure the dims are right
        self.gamma_pred_samples = gamma_pred_samples
        self.num_fpd_samples = len(gamma_pred_samples[0])
        
        # keep track internally
        self.fpd_samples_padded = fpd_samples_padded
        # pad with a 2nd batch dim for # of fpd samples
        self.td_measured_padded = np.repeat(td_measured_padded[:, np.newaxis, :],
            self.num_fpd_samples, axis=1)
        self.td_likelihood_prefactors = np.repeat(td_likelihood_prefactors[:, np.newaxis],
            self.num_fpd_samples, axis=1)
        self.td_likelihood_prec = np.repeat(td_likelihood_prec[:, np.newaxis, :, :],
            self.num_fpd_samples, axis=1)


        # TODO: fix hardcoding of this
        if log_prob_modeling_prior is None:
            self.log_prob_modeling_prior = norm.logpdf(gamma_pred_samples,loc=2.,scale=0.2)
        else:
            hst_train0 = pd.read_csv('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/MassModels/hst_train0_metadata.csv')
            gamma_vals = hst_train0['main_deflector_parameters_gamma'].to_numpy().astype(float)
            gamma_kde = gaussian_kde(gamma_vals)
            self.log_prob_modeling_prior = np.empty((gamma_pred_samples.shape))
            for i in range(0,gamma_pred_samples.shape[0]):
                self.log_prob_modeling_prior[i,:] = gamma_kde.logpdf(gamma_pred_samples[i])

    # compute predicted time delays from predicted fermat potential differences
    # requires an assumed cosmology (from hyperparameters) and redshifts
    def td_pred_from_fpd_pred(self,hyperparameters):
        """
        Args:
            hyperparameters ():
            fpd_pred_samples (size:(n_lenses,n_samples,3)): Note: it is assumed
                doubles are padded with zeros

        Returns:
            td_pred_samples (size:(n_lenses,n_samples,3))
        """

        # construct cosmology object from hyperparameters
        h0_input = hyperparameters[0]
        if self.cosmo_model == 'LCDM':
             w0_input = -1.
             wa_input = 0.
        elif self.cosmo_model == 'w0waCDM':
             w0_input = hyperparameters[1]
             wa_input = hyperparameters[2]
        my_jax_cosmo = jax_cosmo.Cosmology(h=jnp.float32(h0_input/100),
                    Omega_c=jnp.float32(0.3),
                    Omega_k=jnp.float32(0.),Omega_b=jnp.float32(0.0), w0=jnp.float32(w0_input),
                    wa=jnp.float32(wa_input),sigma8 = jnp.float32(0.8), n_s=jnp.float32(0.96))
        
        # compute time delay distances from cosmology and redshifts
        Ddt_computed = tdc_utils.jax_ddt_from_redshifts(my_jax_cosmo,self.z_lens,self.z_src)
        # convert to numpy
        Ddt_computed = np.array(Ddt_computed)
        # add batch dimensions for Ddt computed...
        Ddt_repeated = np.repeat(Ddt_computed[:, np.newaxis],
            self.num_fpd_samples, axis=1)
        Ddt_repeated = np.repeat(Ddt_repeated[:,:, np.newaxis],
            3, axis=2)
        # compute predicted time delays (this function should work w/ arrays)
        return tdc_utils.td_from_ddt_fpd(Ddt_repeated,self.fpd_samples_padded)

    # TDC Likelihood per lens per fpd sample (only condense along num. images dim.)
    def td_log_likelihood_per_samp(self,td_pred_samples):
        """
        Args:
            td_pred_samples (n_lenses,n_fpd_samps,3)

        Returns:
            td_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        x_minus_mu = (td_pred_samples-self.td_measured_padded)
        # add batch dimension for # of time delays dim.
        x_minus_mu = np.expand_dims(x_minus_mu,axis=-1)
        # matmul should condense the (# of time delays) dim.
        exponent = -0.5*np.matmul(np.transpose(x_minus_mu,axes=(0,1,3,2)),
            np.matmul(self.td_likelihood_prec,x_minus_mu))

        # reduce to two dimensions: (n_lenses,n_fpd_samples)
        exponent = np.squeeze(exponent)

        # TODO: should I change this to log-likelihood?
        return self.td_likelihood_prefactors + exponent
        
        
    def full_log_likelihood(self,hyperparameters):
        """
        Args:
            hyperparameters ([H0,mu_gamma,sigma_gamma] or [H0,w0,wa,mu_gamma,sigma_gamma])
            fpd_pred_samples (size:(n_lenses,n_samples,3)): Note, it is assumed 
                that doubles are padded w/ zeros
        """

        # TODO: construct td_pred_samples from fpd_pred_samples
        td_pred_samples = self.td_pred_from_fpd_pred(hyperparameters)

        td_log_likelihoods = self.td_log_likelihood_per_samp(td_pred_samples)

        # reweighting factor
        eval_at_proposed_nu = norm.logpdf(self.gamma_pred_samples,
            loc=hyperparameters[-2],scale=hyperparameters[-1])
        rw_factor = eval_at_proposed_nu - self.log_prob_modeling_prior

        # sum across fpd samples 
        individ_likelihood = np.mean(np.exp(td_log_likelihoods+rw_factor),axis=1)

        # sum over all lenses
        # TODO: check for any log of zero!!
        if np.sum(individ_likelihood == 0) > 0:
            return -np.inf

        log_likelihood = np.sum(np.log(individ_likelihood))

        return log_likelihood


#########################
# Sampling Implementation
#########################

def fast_TDC(tdc_likelihood,cosmo_model='LCDM',num_emcee_samps=1000,
    n_walkers=20):
    """
    Args:
        tdc_likelihood (TDCLikelihood object)
        cosmo_model (string): 'LCDM' or 'w0waCDM'
        num_emcee_samps (int): Number of iterations for MCMC inference
        n_walkers (int): Number of emcee walkers
        
    """

    def LCDM_log_prior(hyperparameters):
        """
        Args:
            hyperparameters ([H0,mu_gamma,sigma_gamma])
        """

        if hyperparameters[0] < 0 or hyperparameters[0] > 150:
                return -np.inf
        elif hyperparameters[1] < 1.5 or hyperparameters[1] > 2.5:
            return -np.inf
        elif hyperparameters[2] < 0.001 or hyperparameters[2] > 0.2:
            return -np.inf
        
        return 0
    
    def w0waCDM_log_prior(hyperparameters):
        """
        Args:
            hyperparameters ([H0,w0,wa,mu_gamma,sigma_gamma])
        """

        # h0 [0,150]
        if hyperparameters[0] < 0 or hyperparameters[0] > 150: 
                return -np.inf
        #w0 [-2,0]
        elif hyperparameters[1] < -2 or hyperparameters[1] > 0:
                return -np.inf
        #wa [-2,2]
        elif hyperparameters[2] < -2 or hyperparameters[2] > 2:
                return -np.inf
        #mu(gamma)
        elif hyperparameters[3] < 1.5 or hyperparameters[3] > 2.5:
            return -np.inf
        #sigma(gamma)
        elif hyperparameters[4] < 0.001 or hyperparameters[4] > 0.2:
            return -np.inf
        
        return 0
    
    def generate_initial_state(n_walkers,cosmo_model):
        """
        Args:
            cosmo_model (string)
        """

        if cosmo_model == 'LCDM':
            cur_state = np.empty((n_walkers,3))
            # fill h0 initial state
            cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers)
            cur_state[:,1] = uniform.rvs(loc=1.8,scale=0.4,size=n_walkers)
            cur_state[:,2] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)

            return cur_state
        
        elif cosmo_model == 'w0waCDM':
            cur_state = np.empty((n_walkers,5))
            # fill h0 initial state
            cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers)
            cur_state[:,1] = uniform.rvs(loc=-1.5,scale=1.,size=n_walkers)
            cur_state[:,2] = uniform.rvs(loc=-1,scale=2,size=n_walkers)
            cur_state[:,3] = uniform.rvs(loc=1.8,scale=0.4,size=n_walkers)
            cur_state[:,4] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)

            return cur_state


    
    def log_posterior(hyperparameters):
        """
        Args:
            hyperparameters ([H0,mu_gamma,sigma_gamma] or [H0,w0,wa,mu_gamma,sigma_gamma])
        """
        # Prior
        if cosmo_model == 'LCDM':
            lp = LCDM_log_prior(hyperparameters)
        elif cosmo_model == 'w0waCDM':
            lp = w0waCDM_log_prior(hyperparameters)
        # Likelihood
        if lp == 0:
            fll = tdc_likelihood.full_log_likelihood(hyperparameters)
            lp += fll

        return lp
    
    # emcee stuff here
    cur_state = generate_initial_state(n_walkers,cosmo_model)
    sampler = emcee.EnsembleSampler(n_walkers,cur_state.shape[1],log_posterior)

    # run mcmc
    tik_mcmc = time.time()
    _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps,progress=True)
    tok_mcmc = time.time()
    print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))

    return sampler.chain