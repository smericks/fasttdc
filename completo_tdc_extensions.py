# helper functions that are imported into tdc_sampler
import numpy as np
import pymc as pm
from scipy.stats import uniform

def LCDM_completo_cPDF_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_elements,chol_elements])
            Note: if there are N lens params... there are N mu_elements, and (N*N - N)/2 + N chol_elements
            Note: chol_elements are the elements of a lower triangular matrix (the cholesky decomposition)
                we have to parameterize this way, to ensure we get a symmetric, positive-definite cov. matrix
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    if hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    
    # here comes the lens params cPDF
    num_h = len(hyperparameters[2:]) # num_h = N + N(N+1)/2 (num. hyperparameters)
    # solv quad. eqn. to determine # of lens params (0 = n^2 + 3n -2(num_h))
    num_lp = (-3 + np.sqrt(9+8*num_h))/2 # we only need the positive solution...
    num_lp = int(num_lp)

    means = hyperparameters[2:(num_lp+2)]
    chol_elements = hyperparameters[(num_lp+2):]

    # prior over the means, NOTE assuming all params have been normalized to center at 0. with stddev = 1.
    for m in means:
        if m < -2 or m > 2: # weak, uniform prior!!
            return -np.inf

    # prior over the chol_elements
    lkj_dist = pm.LKJCholeskyCov.dist(
        n=num_lp, # number_params
        eta=2.0, # controls pref. for weak or strong correlations. 
        # eta=2. is a slight pref. for weak corrs. (eta=1. is uniform)
        sd_dist=pm.HalfNormal.dist(sigma=2.0), # 'moderate' (medium variances expected...whatever that means)
        compute_corr = False # we only want to evalute on 'packed' chol elements
    )
    log_prob_chol = pm.logp(lkj_dist, chol_elements).eval()

    if np.isnan(log_prob_chol): # I think this is how LKJCholeskyCov indicates violation of the prior?
        return -np.inf

    return log_prob_chol

def LCDM_completo_cPDF_log_prior_generate_initialstate(n_walkers,cosmo_model,
    num_lp,random_seed=None):
  
    # prior over the chol_elements
    lkj_dist = pm.LKJCholeskyCov.dist(
        n=num_lp, # number_params
        eta=2.0, # controls pref. for weak or strong correlations. 
        # eta=2. is a slight pref. for weak corrs. (eta=1. is uniform)
        sd_dist=pm.HalfNormal.dist(sigma=2.0), # 'moderate' (medium variances expected...whatever that means)
        compute_corr=False # we only want the 'packed' chol elements
    )
    initial_chol_states = pm.draw(lkj_dist, draws=n_walkers) # (n_chains, num_chol_elements)

    num_cp = 2 # num cosmo params at beginning of hyperparams
    num_hyperparam = num_cp + num_lp + np.shape(initial_chol_states)[1]
    cur_state = np.empty((n_walkers,num_hyperparam))
    cur_state[:,0] = uniform.rvs(loc=65,scale=10,size=n_walkers) # H0
    cur_state[:,1] = uniform.rvs(loc=0.25,scale=0.1,size=n_walkers) # OmegaM
    cur_state[:,num_cp:(num_lp+num_cp)] = uniform.rvs(loc=-1,scale=2,size=(n_walkers,num_lp)) # means
    cur_state[:,(num_lp+num_cp):] = initial_chol_states # chol_elements

    return cur_state



    


