import h5py
import pandas as pd
import numpy as np
import corner
from scipy.stats import norm, multivariate_normal, gaussian_kde
from matplotlib.lines import Line2D
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler


# USER SETTINGS HERE (TODO: change filepaths)
data_vectors = {
    'gold_quads':{
        'h5_file':'DataVectors/gold/quad_posteriors_KIN.h5',
        'metadata_file':'DataVectors/gold/truth_metadata.csv',
        'td_measurement_error_days':5.,
        'sigma_v_measurement_error_kmpersec':5.
    }
}
NUM_EMCEE_SAMPS = 10
USE_ASTROPY = True #use this flag to avoid jax_cosmo DDts 

#############
# HARDCODINGS
#############
# Interim Modeling Priors
gamma_kde = norm(loc=2.0,scale=0.2).logpdf
beta_ani_modeling_prior = norm(loc=0.,scale=0.2).logpdf


######################
# LOAD IN DATA VECTORS
#####################

for lens_sample in data_vectors.keys():

    h5_file = data_vectors[lens_sample]['h5_file']
    h5 = h5py.File(h5_file, 'r')
    data_vectors[lens_sample]['fpd_samps'] = h5['fpd_samps'][:]
    data_vectors[lens_sample]['lens_param_samps'] = h5['lens_param_samps'][:]
    data_vectors[lens_sample]['beta_ani_samps']= h5['beta_ani_samps'][:]
    data_vectors[lens_sample]['c_sqrtJ_samps'] = h5['c_sqrtJ_samps'][:]
    data_vectors[lens_sample]['MUSE_c_sqrtJ_samps'] = h5['MUSE_c_sqrtJ_samps'][:]
    data_vectors[lens_sample]['NIRSPEC_c_sqrtJ_samps'] = h5['NIRSPEC_c_sqrtJ_samps'][:]
    data_vectors[lens_sample]['catalog_idxs'] = h5['catalog_idxs'][:]
    h5.close()

# load in truth information

for lens_sample in data_vectors.keys():

    metadata_df = pd.read_csv(data_vectors[lens_sample]['metadata_file'])
    # pick which lenses based on catalog_idx
    metadata_subset = metadata_df[metadata_df['catalog_idx'].isin(data_vectors[lens_sample]['catalog_idxs'])]
    z_lens = metadata_subset.loc[:,'main_deflector_parameters_z_lens'].to_numpy()
    z_src = metadata_subset.loc[:,'source_parameters_z_source'].to_numpy()
    num_td = np.shape(data_vectors[lens_sample]['fpd_samps'])[-1]
    if num_td == 1: 
        # NOTE: will this keep the last dim=(..,..,1) ? 
        td_truth = metadata_subset.loc[:,['td01']].to_numpy()
    elif num_td == 3:
        td_truth = metadata_subset.loc[:,['td01','td02','td03']].to_numpy()
    sigma_v_truth = metadata_subset.loc[:,['sigma_v_4MOST_kmpersec']].to_numpy()

    # store in data_vectors dict
    data_vectors[lens_sample]['z_lens'] = z_lens
    data_vectors[lens_sample]['z_src'] = z_src
    data_vectors[lens_sample]['td_truth'] = td_truth
    data_vectors[lens_sample]['sigma_v_truth'] = sigma_v_truth


# load in IFU truth information when requested
ifu_lens_samples = ['gold_quads']

for lens_sample in ifu_lens_samples:
    metadata_df = pd.read_csv(data_vectors[lens_sample]['metadata_file'])
    # pick which lenses based on catalog_idx
    metadata_subset = metadata_df[metadata_df['catalog_idx'].isin(data_vectors[lens_sample]['catalog_idxs'])]

    # track MUSE
    muse_keys = ['sigma_v_MUSE_bin'+str(j)+'_kmpersec' for j in range(0,3)]
    data_vectors[lens_sample]['MUSE_sigma_v_truth'] = metadata_subset.loc[:,muse_keys].to_numpy()

    # track NIRSPEC
    nirspec_keys = ['sigma_v_NIRSPEC_bin'+str(j)+'_kmpersec' for j in range(0,10)]
    data_vectors[lens_sample]['NIRSPEC_sigma_v_truth'] = metadata_subset.loc[:,nirspec_keys].to_numpy()

#######################
# EMULATE DATA VECTORS
#######################
for lens_sample in data_vectors.keys():
    # STEP 1: Emulate Time-Delay Measurements
    # grab number of time-delays, number lenses
    num_td = np.shape(data_vectors[lens_sample]['fpd_samps'])[-1]
    num_fpd_samps = np.shape(data_vectors[lens_sample]['fpd_samps'])[-2]
    num_lenses = np.shape(data_vectors[lens_sample]['td_truth'])[0]
    # construct covariance / precision matrices
    cov_td = np.eye(num_td)*(
        data_vectors[lens_sample]['td_measurement_error_days'])**2
    td_measurement_prec = np.linalg.inv(cov_td)[np.newaxis,:,:]
    td_measurement_prec = np.repeat(td_measurement_prec,repeats=num_lenses,axis=0)
    # emulate mean measurement off of the truth
    td_truth = data_vectors[lens_sample]['td_truth']
    td_measured = np.empty(
        np.shape(data_vectors[lens_sample]['td_truth']))
    for lens_idx in range(0,num_lenses):
        td_measured[lens_idx] = multivariate_normal.rvs(mean=td_truth[lens_idx],cov=cov_td)
    # save to data vectors dict
    data_vectors[lens_sample]['td_measured'] = td_measured
    data_vectors[lens_sample]['td_measurement_prec'] = td_measurement_prec


    # STEP 2: Emulate Kinematic Measurements
    # grab number of kin bins
    num_kin_bins = np.shape(data_vectors[lens_sample]['c_sqrtJ_samps'])[-1]
    # construct covariance / precision matrices
    cov_sigma_v = np.eye(num_kin_bins)*(
        data_vectors[lens_sample]['sigma_v_measurement_error_kmpersec'])**2
    sigma_v_measurement_prec = np.linalg.inv(cov_sigma_v)[np.newaxis,:,:]
    sigma_v_measurement_prec = np.repeat(sigma_v_measurement_prec,repeats=num_lenses,axis=0)
    # emulate mean measurement off of the truth
    sigma_v_truth = data_vectors[lens_sample]['sigma_v_truth']
    sigma_v_measured = np.empty(
        np.shape(data_vectors[lens_sample]['sigma_v_truth']))
    for lens_idx in range(0,num_lenses):
        sigma_v_measured[lens_idx] = multivariate_normal.rvs(mean=sigma_v_truth[lens_idx],cov=cov_sigma_v)
    # save to data vectors dict
    data_vectors[lens_sample]['sigma_v_measured'] = sigma_v_measured
    data_vectors[lens_sample]['sigma_v_measurement_prec'] = sigma_v_measurement_prec


    # STEP 3: Emulate Kinematic Measurements
    # TODO: turn this into a helper function (code duplication!!)
    # MUSE
    MUSE_csqrtJ_samps = data_vectors[lens_sample]['MUSE_c_sqrtJ_samps']
    num_kin_bins = MUSE_csqrtJ_samps.shape[-1]
    # construct covariance / precision matrices
    MUSE_cov_sigma_v = np.eye(num_kin_bins)*(
        data_vectors[lens_sample]['sigma_v_measurement_error_kmpersec'])**2
    MUSE_sigma_v_measurement_prec = np.linalg.inv(MUSE_cov_sigma_v)[np.newaxis,:,:]
    MUSE_sigma_v_measurement_prec = np.repeat(MUSE_sigma_v_measurement_prec,repeats=num_lenses,axis=0)
    # emulate mean measurement off of the truth
    MUSE_sigma_v_truth = data_vectors[lens_sample]['MUSE_sigma_v_truth']
    MUSE_sigma_v_measured = np.empty(
        np.shape(MUSE_sigma_v_truth))
    for lens_idx in range(0,num_lenses):
        MUSE_sigma_v_measured[lens_idx] = multivariate_normal.rvs(
            mean=MUSE_sigma_v_truth[lens_idx],cov=MUSE_cov_sigma_v)
    # save to data vectors dict
    data_vectors[lens_sample]['MUSE_sigma_v_measured'] = MUSE_sigma_v_measured
    data_vectors[lens_sample]['MUSE_sigma_v_measurement_prec'] = MUSE_sigma_v_measurement_prec

    # NIRSPEC
    NIRSPEC_csqrtJ_samps = data_vectors[lens_sample]['NIRSPEC_c_sqrtJ_samps']
    num_kin_bins = NIRSPEC_csqrtJ_samps.shape[-1]
    # construct covariance / precision matrices
    NIRSPEC_cov_sigma_v = np.eye(num_kin_bins)*(
        data_vectors[lens_sample]['sigma_v_measurement_error_kmpersec'])**2
    NIRSPEC_sigma_v_measurement_prec = np.linalg.inv(NIRSPEC_cov_sigma_v)[np.newaxis,:,:]
    NIRSPEC_sigma_v_measurement_prec = np.repeat(NIRSPEC_sigma_v_measurement_prec,repeats=num_lenses,axis=0)
    # emulate mean measurement off of the truth
    NIRSPEC_sigma_v_truth = data_vectors[lens_sample]['NIRSPEC_sigma_v_truth']
    NIRSPEC_sigma_v_measured = np.empty(
        np.shape(NIRSPEC_sigma_v_truth))
    for lens_idx in range(0,num_lenses):
        NIRSPEC_sigma_v_measured[lens_idx] = multivariate_normal.rvs(
            mean=NIRSPEC_sigma_v_truth[lens_idx],cov=NIRSPEC_cov_sigma_v)
    # save to data vectors dict
    data_vectors[lens_sample]['NIRSPEC_sigma_v_measured'] = NIRSPEC_sigma_v_measured
    data_vectors[lens_sample]['NIRSPEC_sigma_v_measurement_prec'] = NIRSPEC_sigma_v_measurement_prec


    # STEP 4: emulate kappa_ext
    kappa_ext_samps = norm.rvs(loc=0.,scale=0.05,size=(num_lenses,5000))
    data_vectors[lens_sample]['kappa_ext_samps'] = kappa_ext_samps

#########################
# Gaussianizing helper...
#########################
def gaussianize_samples(input_samps,num_gaussian_samps=5000):
    """takes in input samples, fits a Gaussian, and returns new samples
        from that Gaussian
    Args:
        input_samps (n_input_samps,n_params)
    Returns:
        output_samps (n_gaussian_samps,n_params)
    """

    Mu = np.mean(input_samps,axis=0)
    Cov = np.cov(input_samps,rowvar=False)

    gaussianized_samps = multivariate_normal.rvs(mean=Mu,cov=Cov,
        size=num_gaussian_samps)

    return gaussianized_samps

#########################
# Construct likelihood.
#########################
input_samps_array = np.stack((data_vectors['gold_quads']['fpd_samps'][:10,:,0],
    data_vectors['gold_quads']['fpd_samps'][:10,:,1],
    data_vectors['gold_quads']['fpd_samps'][:10,:,2],
    data_vectors['gold_quads']['lens_param_samps'][:10,:,3],
    data_vectors['gold_quads']['beta_ani_samps'][:10],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,0],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,1],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,2],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,3],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,4],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,5],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,6],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,7],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,8],
    data_vectors['gold_quads']['NIRSPEC_c_sqrtJ_samps'][:10,:,9]),axis=-1)

num_lenses = input_samps_array.shape[0]
num_params = input_samps_array.shape[-1]
output_samps_array = np.empty((num_lenses,5000,num_params))
for lens_idx in range(0,num_lenses):
    output_samps = gaussianize_samples(input_samps_array[lens_idx],num_gaussian_samps=5000)
    output_samps_array[lens_idx] = output_samps

# tdc_sampler likelihood object
quad_kin_lklhd_kappa_ext = tdc_sampler.TDCKinLikelihood(
    td_measured=data_vectors['gold_quads']['td_measured'][:10],
    td_likelihood_prec=data_vectors['gold_quads']['td_measurement_prec'][:10],
    sigma_v_measured=data_vectors['gold_quads']['NIRSPEC_sigma_v_measured'][:10],
    sigma_v_likelihood_prec=data_vectors['gold_quads']['NIRSPEC_sigma_v_measurement_prec'][:10],
    fpd_samples=output_samps_array[:,:,0:3],
    gamma_pred_samples=output_samps_array[:,:,3],
    beta_ani_samples=output_samps_array[:,:,4],
    kin_pred_samples=output_samps_array[:,:,5:],
    kappa_ext_samples=data_vectors['gold_quads']['kappa_ext_samps'][:10],
    z_lens=data_vectors['gold_quads']['z_lens'][:10],
    z_src=data_vectors['gold_quads']['z_src'][:10],
    cosmo_model='LCDM_lambda_int_beta_ani',
    log_prob_gamma_nu_int=gamma_kde,
    log_prob_beta_ani_nu_int=beta_ani_modeling_prior,
    use_astropy=USE_ASTROPY)


###################
# RUN MCMC HERE!!!
###################
tenIFU_chain = tdc_sampler.fast_TDC([quad_kin_lklhd_kappa_ext],num_emcee_samps=NUM_EMCEE_SAMPS,
    n_walkers=20)

