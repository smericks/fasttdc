import numpy as np
from scipy.stats import norm, multivariate_normal

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
    elif hyperparameters[6] < 1.5 or hyperparameters[6] > 2.5: #mu(gamma_lens)
        return -np.inf
    elif hyperparameters[7] < 0.001 or hyperparameters[7] > 0.2: #sigma(gamma_lens)
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

def tdcosmo25_lambda_int_beta_ani_log_prior(hyperparameters):
    """
    Args:
        hyperparameters ([H0,omega_M,mu_lambda_int,sigma_lambda_int,
            mu_gamma,sigma_gamma])
    """

    if hyperparameters[0] < 0 or hyperparameters[0] > 150: #h0
        return -np.inf
    elif hyperparameters[1] < 0.05 or hyperparameters[1] > 0.5: #omega_M 
        return -np.inf
    #w0 [-1.5,0.5]
    elif hyperparameters[2] < -1.5 or hyperparameters[2] > 0.5:
        return -np.inf
    #wa [-10,10]
    elif hyperparameters[3] < -10 or hyperparameters[3] > 10:
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

def OmegaM_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """Include approximation of Pantheon+ Prior used in TDCOSMO 2025 (https://arxiv.org/pdf/2506.03023)
        Note page 17: "Pantheon+ effectively provided a prior on Ωm (i.e., Ωm = 0.334 ± 0.018)"
    """

    # returns 0 or -np.inf
    within_bounds = w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    if within_bounds == 0:   
        # note we center our ground truth at 0.3     
        return norm.logpdf(hyperparameters[1],loc=0.3,scale=0.018)

    else:
        return within_bounds

# TODO: finish incorporating this option for informative OmegaM + LCDM
def OmegaM_LCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """Include approximation of Pantheon+ Prior used in TDCOSMO 2025 (https://arxiv.org/pdf/2506.03023)
        Note page 17: "Pantheon+ effectively provided a prior on Ωm (i.e., Ωm = 0.334 ± 0.018)"
    """

    # returns 0 or -np.inf
    within_bounds = LCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    if within_bounds == 0:   
        # note we center our ground truth at 0.3     
        return norm.logpdf(hyperparameters[1],loc=0.3,scale=0.018)

    else:
        return within_bounds

def InformedPop_InformedOmegaM_LCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """Informative prior on BOTH OmegaM and populations of lambda_int, beta_ani
        assuming some external sample is constraining lambda_int, beta_ani properties 
    """

    # returns 0 or -np.inf
    within_bounds = LCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    if within_bounds == 0:   
        # note we center our ground truth at 0.3     
        om_prior = norm.logpdf(hyperparameters[1],loc=0.3,scale=0.018)
        # NOTE: modified to extreme amt. of precision
        lens_pop_prior = multivariate_normal.logpdf(hyperparameters[2:6],
            mean=[1.,0.05,0.,0.05],
            cov=np.diag(np.asarray([0.01,0.01,0.01,0.01])**2))
        #lint_mu_prior = norm.logpdf(hyperparameters[2],loc=1.,scale=0.05)
        #bani_prior = norm.logpdf(hyperparameters[4],loc=0.,scale=0.05)
        
        return (om_prior+lens_pop_prior)

    else:
        return within_bounds


def InformedPop_InformedOmegaM_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """Informative prior on BOTH OmegaM and populations of lambda_int, beta_ani
        assuming some external sample is constraining lambda_int, beta_ani properties 
    """

    # returns 0 or -np.inf
    within_bounds = w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    if within_bounds == 0:   
        # note we center our ground truth at 0.3     
        om_prior = norm.logpdf(hyperparameters[1],loc=0.3,scale=0.018)
        # NOTE: modified to extreme amt. of precision
        lens_pop_prior = multivariate_normal.logpdf(hyperparameters[4:8],
            mean=[1.,0.05,0.,0.05],
            cov=np.diag(np.asarray([0.01,0.01,0.01,0.01])**2))
        #lint_prior = norm.logpdf(hyperparameters[4],loc=1.,scale=0.05)
        #bani_prior = norm.logpdf(hyperparameters[6],loc=0.,scale=0.05)

        return (om_prior+lens_pop_prior)

    else:
        return within_bounds
        

def INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    """
    Used for redshift configuration test. Only evaluates on params 4-8 
        (mu_lambda_int,sigma_lambda_int,mu_beta_ani,sigma_beta_ani). 
    An informative prior on these params to simulate being within a larger
        population inference...
    """

    # returns 0 or -np.inf
    within_bounds = w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    if within_bounds == 0:
        # this cov matrix is hardcoded, taken from a gold-only chain
        # we only evaluate on the last 4 params, so this is only a prior
        # on lambda_int and beta_ani...
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
        

def OmegaM_INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters):
    
    log_prob = INFORMATIVE_w0waCDM_lambda_int_beta_ani_log_prior(hyperparameters)

    # multiply in p(Omega_M)
    if np.isfinite(log_prob):
        log_prob += norm.logpdf(hyperparameters[1],loc=0.3,scale=0.018)

    return log_prob


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