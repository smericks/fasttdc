import jax_cosmo
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm
from om10_lens_sample import OM10Sample
import tdc_utils

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
        array of prefactors: (1/(2pi)^k/2) * 1/sqrt(det(Sigma))
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
        
    return td_measured_padded, fpd_samples_padded, td_likelihood_prefactors, td_likelihood_prec


###########################
# TDC Likelihood Functions
###########################

# I think we want this to be a class, so we can keep track of quantities
# internally 
class TDCLikelihood():

    def __init__(self,td_measured,td_cov,z_lens,z_src,fpd_pred_samples,
        gamma_pred_samples):
        """
        Keep track of quantities that remain constant throughout the inference

        Args: 
            td_measured (np.array(float), size:(n_lenses,3)): list of 1d arrays of 
                time delay measurements for each lens
            td_cov (np.array(float), size:(n_lenses,3,3)): list of 2d arrays of 
                time delay covariance matrices, padded with Nans
            z_lens (np.array(float), size:(n_lenses)): lens redshifts
            z_src (np.array(float), size:(n_lenses)): source redshifts
            fpd_ped_samples (list of [float]): list of !!variable size!! 2d arrays
                of fpd samples. Some will have dim (1,n_samples), others will have dim
                (3,n_samples)
            gamma_pred_samples (np.array(float), size:(n_lenses,n_samples)): 
                list of 1d arrays of list of gamma samples associated with each 
                set of fpd samples.
        """

        # no processing needed
        self.z_lens = z_lens
        self.z_src = z_src
        self.gamma_pred_samples = gamma_pred_samples
        num_fpd_samples = len(fpd_pred_samples[0][0])

        # padding
        (td_measured_padded,fpd_samples_padded,td_likelihood_prefactors,
            td_likelihood_prec) = preprocess_td_measured(td_measured,td_cov,
            fpd_pred_samples)
        
        # keep track internally
        self.fpd_samples_padded = fpd_samples_padded
        # pad with a 2nd batch dim for # of fpd samples
        self.td_measured_padded = np.repeat(td_measured_padded[:, np.newaxis, :],
            num_fpd_samples, axis=1)
        self.td_likelihood_prefactors = np.repeat(td_likelihood_prefactors[:, np.newaxis],
            num_fpd_samples, axis=1)
        self.td_likelihood_prec = np.repeat(td_likelihood_prec[:, np.newaxis, :, :],
            num_fpd_samples, axis=1)


        # TODO: fix hardcoding of this
        self.log_prob_modeling_prior = norm.logpdf(gamma_pred_samples,loc=2.,scale=0.2)

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
        my_jax_cosmo = jax_cosmo.Cosmology(h=jnp.float32(hyperparameters[0]/100),
                    Omega_c=jnp.float32(0.3),
                    Omega_k=jnp.float32(0.),Omega_b=jnp.float32(0.0), w0=jnp.float32(-1.),
                    wa=jnp.float32(0.),sigma8 = jnp.float32(0.8), n_s=jnp.float32(0.96))
        # Q1: can jax-cosmo do arrays of redshifts?
        # compute time delay distance from cosmology and redshifts
        Ddt_computed = tdc_utils.jax_ddt_from_redshifts(my_jax_cosmo,self.z_lens,self.z_src)
        # compute predicted time delays (this function should work w/ arrays)
        # TODO: might have to add a batch dimension for Ddt computed...
        return tdc_utils.td_from_ddt_fpd(Ddt_computed,self.fpd_samples_padded)

    # TDC Likelihood per lens per fpd sample (only condense along num. images dim.)
    def td_likelihood_per_samp(self,td_pred_samples):
        """
        Args:
            td_pred_samples (n_lenses,n_fpd_samps,3)

        Returns:
            td_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        x_minus_mu = (td_pred_samples-self.td_measured_padded)
        x_minus_mu = np.expand_dims(x_minus_mu,axis=-1)
        print('x_minus_mu shape',x_minus_mu.shape)
        print('td_likelihood_prec shape',self.td_likelihood_prec.shape)
        print('testing')
        np.matmul(self.td_likelihood_prec,x_minus_mu)
        print('test passed')
        exponent = -0.5*np.matmul(np.transpose(x_minus_mu,axes=(0,1,3,2)),
            np.matmul(self.td_likelihood_prec,x_minus_mu))

        exponent = np.squeeze(exponent)

        print('exponent shape: ',exponent.shape)

        return self.td_likelihood_prefactors*np.exp(exponent)
        
        
    def full_log_likelihood(self,hyperparameters):
        """
        Args:
            hyperparameters ([H0,mu_gamma,sigma_gamma])
            fpd_pred_samples (size:(n_lenses,n_samples,3)): Note, it is assumed 
                that doubles are padded w/ zeros
        """

        # TODO: construct td_pred_samples from fpd_pred_samples
        td_pred_samples = self.td_pred_from_fpd_pred(hyperparameters)

        td_likelihoods = self.td_likelihood_per_samp(td_pred_samples)

        # reweighting factor
        eval_at_proposed_nu = norm.logpdf(self.gamma_pred_samples,
            loc=hyperparameters[1],scale=hyperparameters[2])
        rw_factor = eval_at_proposed_nu - self.log_prob_modeling_prior

        # sum across fpd samples 
        individ_likelihood = np.mean(np.exp(td_likelihoods+rw_factor),axis=1)

        # sum over all lenses
        # TODO: check for any log of zero!!
        if np.sum(individ_likelihood == 0) > 0:
            return -np.inf
        log_likelihood = np.sum(np.log(individ_likelihood))

        return log_likelihood


#########################
# Sampling Implementation
#########################

def fast_TDC(td_measured,td_cov,z_lens,z_src,
    fpd_pred_samples,gamma_pred_samples):
    """
    Args:
        td_measured ()
        td_cov ()
        z_lens (size:(n_lenses))
        z_src (size:(n_lenses))
        fpd_pred_samples ()
        gamma_pred_samples ( size:(n_lenses,num_fpd_samples))
        
    """

    num_fpd_samples = len(fpd_pred_samples[0][0])

    tdc_likelihood = TDCLikelihood(td_measured,td_cov,z_lens,z_src,
        fpd_pred_samples,gamma_pred_samples)
    
    # put the function def here so it can use pre-computed quantities

    def log_prior(hyperparameters):
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
    
    def log_posterior(hyperparameters):
        """
        Args:
            hyperparameters ([H0,mu_gamma,sigma_gamma])
        """

        lp = log_prior(hyperparameters)
        if lp == 0:
            lp += tdc_likelihood.full_log_likelihood(hyperparameters)

        return lp
    
    # TODO: emcee stuff here

om10_sample = OM10Sample('om10_metadata_venkatraman24.csv')
om10_sample.choose_gold_silver()
gold_modeling_error_dict={
    'theta_E':0.01,
	'gamma':0.06,
	'e1':0.05,
    'e2':0.05,
	'center_x':0.01,
	'center_y':0.01,
	'gamma1':0.05,
    'gamma2':0.05,
	'src_center_x':0.02,
	'src_center_y':0.02
}
silver_modeling_error_dict={
    'theta_E':0.05,
	'gamma':0.1,
	'e1':0.07,
    'e2':0.07,
	'center_x':0.03,
	'center_y':0.03,
	'gamma1':0.07,
    'gamma2':0.07,
	'src_center_x':0.04,
	'src_center_y':0.04
}
om10_sample.populate_modeling_preds(gold_modeling_error_dict,silver_modeling_error_dict)