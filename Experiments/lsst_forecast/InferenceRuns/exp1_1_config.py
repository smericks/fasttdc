# experiment 1.1: Gold-Only Baseline, Randomly Selected

import h5py
from scipy.stats import norm
import numpy as np

# file locations
static_dv_file = 'InferenceRuns/exp1_1/static_datavectors.json'
gold_quads_h5_file = 'DataVectors/gold/quad_posteriors_KIN.h5'
gold_dbls_h5_file = 'DataVectors/gold/dbl_posteriors_KIN.h5'
gold_metadata_file = 'DataVectors/gold/truth_metadata.csv'

NUM_FPD_SAMPS = 5000
NUM_MCMC_EPOCHS = 100
NUM_MCMC_WALKERS = 48
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'
GAMMA_LENS_PRIOR = norm(loc=2.,scale=0.2).logpdf
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
BACKEND_PATH = 'InferenceRuns/exp1_1/mpi_debug_backend.h5'
RESET_BACKEND=True

# catalog indices available
with h5py.File(gold_quads_h5_file,'r') as h5:
    quad_catalog_idxs = h5['catalog_idxs'][:]
with h5py.File(gold_dbls_h5_file,'r') as h5:
    dbl_catalog_idxs = h5['catalog_idxs'][:]

# NOTE: remove bad indices (nans in doubles silver-quality kinematic samples)
bad_dbls =  [106, 134, 158 ,233 ,263 ,269 ,353, 446, 579 ,618 ,669]
dbl_catalog_idxs = dbl_catalog_idxs[~np.isin(dbl_catalog_idxs, bad_dbls)]

likelihood_configs = {

    # NIRSPEC likelihoods (10 lenses: 1 quad, 9 doubles)
    'nirspec_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':[quad_catalog_idxs[0]],
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
        'catalog_idxs':dbl_catalog_idxs[:9],
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
        'catalog_idxs':quad_catalog_idxs[1:6],
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
        'catalog_idxs':dbl_catalog_idxs[9:44],
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
        'catalog_idxs':quad_catalog_idxs[6:24],
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
        'catalog_idxs':dbl_catalog_idxs[44:176],
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
}