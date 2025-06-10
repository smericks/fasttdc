# experiment 0.1: Gold+Silver Baseline, Randomly Selected

import h5py
from scipy.stats import norm
import numpy as np

# file locations
static_dv_file = 'InferenceRuns/exp0_1/TEST_static_datavectors.json'
gold_quads_h5_file = 'DataVectors/gold/quad_posteriors_KIN.h5'
gold_dbls_h5_file = 'DataVectors/gold/dbl_posteriors_KIN.h5'
gold_metadata_file = 'DataVectors/gold/truth_metadata.csv'
silver_quads_h5_file = 'DataVectors/silver/quad_posteriors_KIN.h5'
silver_dbls_h5_file = 'DataVectors/silver/dbl_posteriors_KIN.h5'
silver_metadata_file = 'DataVectors/silver/truth_metadata.csv'

NUM_FPD_SAMPS = 5000
NUM_MCMC_EPOCHS = 2
NUM_MCMC_WALKERS = 48
RANDOM_SEED = 123
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'
GAMMA_LENS_PRIOR = norm(loc=2.,scale=0.2).logpdf
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
BACKEND_PATH = 'InferenceRuns/exp0_1/TEST_backend.h5'
RESET_BACKEND=True

# catalog indices available
with h5py.File(gold_quads_h5_file,'r') as h5:
    quad_catalog_idxs = h5['catalog_idxs'][:]
with h5py.File(gold_dbls_h5_file,'r') as h5:
    dbl_catalog_idxs = h5['catalog_idxs'][:]

# NOTE: remove bad indices (nans in doubles silver-quality kinematic samples)
bad_dbls =  [106, 134, 158 ,233 ,263 ,269 ,353, 446, 579 ,618 ,669, 877, 1052]
dbl_catalog_idxs = dbl_catalog_idxs[~np.isin(dbl_catalog_idxs, bad_dbls)]

# separate out gold-quality indices ()
# we need 200 (24 quads, 174 doubles)
np.random.seed(RANDOM_SEED)
gold_quad_idxs = np.random.choice(quad_catalog_idxs, size=24, replace=False)
gold_dbl_idxs = np.random.choice(dbl_catalog_idxs,size=174,replace=False)

# now, pick silver-quality indices

# make sure no overlap with gold lenses + remove wider than prior NPE posteriors
silver_wider_than_prior_idxs = [  49,  100,  222,  243,  263,  278,  316,  327,  544,  579,  815,
        942, 1280, 1322, 1545,  117,  132,  151,  257,  317,  582,  765,
        837,  938,  947,  960, 1092, 1110, 1209, 1428, 1431, 1499, 1503]
avail_quad_idxs = quad_catalog_idxs[~np.isin(quad_catalog_idxs,
    np.concatenate((gold_quad_idxs,silver_wider_than_prior_idxs)))]
avail_dbl_idxs = dbl_catalog_idxs[~np.isin(dbl_catalog_idxs,
    np.concatenate((gold_dbl_idxs,silver_wider_than_prior_idxs)))]
# now, choose randomly (72 quads, 528 doubles)
silver_quad_idxs = np.random.choice(avail_quad_idxs, size=72, replace=False)
silver_dbl_idxs = np.random.choice(avail_dbl_idxs,size=528,replace=False)


likelihood_configs = {

    # NIRSPEC likelihoods (10 lenses: 1 quad, 9 doubles)
    'nirspec_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':[gold_quad_idxs[0]],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'NIRSPEC',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'nirspec_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':gold_dbl_idxs[:9],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'NIRSPEC',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # MUSE likelihoods (40 lenses: 5 quads, 35 doubles)
    'muse_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':gold_quad_idxs[1:6],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'MUSE',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'muse_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':gold_dbl_idxs[9:44],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'MUSE',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # 4MOST likelihoods (150 lenses: 18 quads, 132 doubles)
    '4MOST_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':gold_quad_idxs[6:24],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    '4MOST_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':gold_dbl_idxs[44:176],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },
    ################
    # Silver Lenses
    ################

    # Silver 4MOST likelihoods (300 lenses: 36 quads, 264 doubles)
    'silver_4MOST_quads':{
        'posteriors_h5_file':silver_quads_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_quad_idxs[:36],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'silver_4MOST_dbls':{
        'posteriors_h5_file':silver_dbls_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_dbl_idxs[:264],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },


    # Silver no kinematics (300 lenses: 36 quads, 264 doubles)
    'silver_nokin_quads':{
        'posteriors_h5_file':silver_quads_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_quad_idxs[36:],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'silver_nokin_dbls':{
        'posteriors_h5_file':silver_dbls_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_dbl_idxs[264:],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },
}