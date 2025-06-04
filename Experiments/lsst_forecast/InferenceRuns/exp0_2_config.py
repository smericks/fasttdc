# experiment 1.2: Gold-Only Baseline, Human-Bias Selected

import h5py
import pandas as pd
import numpy as np
from scipy.stats import norm

# random seed
RANDOM_SEED = 1

# file locations
static_dv_file = 'InferenceRuns/exp0_2/TESTstatic_datavectors_seed'+str(RANDOM_SEED)+'.json'
gold_quads_h5_file = 'DataVectors/gold/quad_posteriors_KIN.h5'
gold_dbls_h5_file = 'DataVectors/gold/dbl_posteriors_KIN.h5'
gold_metadata_file = 'DataVectors/gold/truth_metadata.csv'
silver_quads_h5_file = 'DataVectors/silver/quad_posteriors_KIN.h5'
silver_dbls_h5_file = 'DataVectors/silver/dbl_posteriors_KIN.h5'
silver_metadata_file = 'DataVectors/silver/truth_metadata.csv'

NUM_FPD_SAMPS = 5000
NUM_MCMC_EPOCHS = 1
NUM_MCMC_WALKERS = 48
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'
GOLD_GAMMA_LENS_PRIOR = norm(loc=2.09,scale=0.16).logpdf # hst_norms.csv: 2.0882867897222503,0.16008433847145742
SILVER_GAMMA_LENS_PRIOR = norm(loc=2.03,scale=0.19).logpdf # norms2.csv: 2.033213914041585,0.1915982613222065
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
BACKEND_PATH = 'InferenceRuns/exp0_2/lcdm_seed'+str(RANDOM_SEED)+'_backend.h5'
RESET_BACKEND=True

# truth information for those indices
truth_df = pd.read_csv(gold_metadata_file)
# NOTE: subset to remove bad indices (nans in the doubles silver-quality kinematic samples)
# remove rows from dataframe that have 'catalog_idx' in bad_dbls
bad_dbls =  [106, 134, 158 ,233 ,263 ,269 ,353, 446, 579 ,618 ,669]
gold_df = truth_df[~truth_df['catalog_idx'].isin(bad_dbls)].reset_index(drop=True)
# track catalog_idxs
gold_df_catalog_idxs = gold_df.loc[:,'catalog_idx'].to_numpy()

#########################
# Human selection cuts!!
#########################

# GOLD NIRSPEC
num_quads = 10
nirspec_quads_avail = np.where(
    (gold_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(gold_df['td01'].to_numpy()) > 30.) | 
     (np.abs(gold_df['td02'].to_numpy()) > 30.) | 
     (np.abs(gold_df['td03'].to_numpy()) > 30.)) &
    #(truth_df['lens_light_parameters_mag_app'].to_numpy() > 22.) &
    (gold_df['lens_light_parameters_mag_app'].to_numpy() < 24.) &
    (gold_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]
# take the catalog idxs you want
catalog_idx_avail = gold_df.loc[nirspec_quads_avail,'catalog_idx'].to_numpy()
nirspec_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_quads,replace=False)
# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(nirspec_quads_catalog_idxs)].reset_index(drop=True)

# GOLD MUSE

# available for MUSE quads (most stringent cut)
num_total = 40
num_quads = 20
muse_quads_avail = np.where(
    (gold_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(gold_df['td01'].to_numpy()) > 30.) | 
     (np.abs(gold_df['td02'].to_numpy()) > 30.) | 
     (np.abs(gold_df['td03'].to_numpy()) > 30.)) &
    (gold_df['lens_light_parameters_mag_app'].to_numpy() < 22.) &
    (gold_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]
# if not enough quads, include more doubles
if len(muse_quads_avail)<num_quads:
    num_quads = len(muse_quads_avail)
num_dbls = num_total - num_quads
# pick the quad idxs...
# take the catalog idxs you want
catalog_idx_avail = gold_df.loc[muse_quads_avail,'catalog_idx'].to_numpy()
muse_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_quads,replace=False)
# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(muse_quads_catalog_idxs)].reset_index(drop=True)


muse_dbls_avail = np.where(
    (gold_df['point_source_parameters_num_images'].to_numpy() == 2) &
    ((np.abs(gold_df['td01'].to_numpy()) > 30.) | 
     (np.abs(gold_df['td02'].to_numpy()) > 30.) | 
     (np.abs(gold_df['td03'].to_numpy()) > 30.)) &
    (gold_df['lens_light_parameters_mag_app'].to_numpy() < 22.) &
    (gold_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]

# take the catalog idxs you want
catalog_idx_avail = gold_df.loc[muse_dbls_avail,'catalog_idx'].to_numpy()
muse_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_dbls,replace=False)
# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(muse_dbls_catalog_idxs)].reset_index(drop=True)


# GOLD 4MOST
num_quads = 75
num_total = 150
fourmost_quads_avail = np.where(
    (gold_df['point_source_parameters_num_images'].to_numpy() == 4))[0]
# if not enough quads, include more doubles
if len(fourmost_quads_avail)<num_quads:
    num_quads = len(fourmost_quads_avail)
num_dbls = num_total - num_quads
# take the catalog idxs you want
catalog_idx_avail = gold_df.loc[fourmost_quads_avail,'catalog_idx'].to_numpy()
fourmost_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_quads,replace=False)
# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(fourmost_quads_catalog_idxs)].reset_index(drop=True)

# doubles
fourmost_dbls_avail = np.where(
    (gold_df['point_source_parameters_num_images'].to_numpy() == 2))[0]
# take the catalog idxs you want
catalog_idx_avail = gold_df.loc[fourmost_dbls_avail,'catalog_idx'].to_numpy()
fourmost_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_dbls,replace=False)
# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(fourmost_dbls_catalog_idxs)].reset_index(drop=True)

# TODO: remove wide posterior silver...
silver_wider_than_prior_idxs = [  49,  100,  222,  243,  263,  278,  316,  327,  544,  579,  815,
        942, 1280, 1322, 1545,  117,  132,  151,  257,  317,  582,  765,
        837,  938,  947,  960, 1092, 1110, 1209, 1428, 1431, 1499, 1503]
silver_df = gold_df[~gold_df['catalog_idx'].isin(silver_wider_than_prior_idxs)].reset_index(drop=True)

# SILVER 4MOST
num_quads = 36
num_total = 300
silver_withkin_quads_avail = np.where(
    (silver_df['point_source_parameters_num_images'].to_numpy() == 4))[0]
# if not enough quads, include more doubles
if len(silver_withkin_quads_avail)<num_quads:
    num_quads = len(fourmost_quads_avail)
num_dbls = num_total - num_quads
# take the catalog idxs you want
catalog_idx_avail = silver_df.loc[silver_withkin_quads_avail,'catalog_idx'].to_numpy()
silver_withkin_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_quads,replace=False)
# then remove them from the dataframe
silver_df = silver_df[~silver_df['catalog_idx'].isin(silver_withkin_quads_catalog_idxs)].reset_index(drop=True)
# doubles
silver_withkin_dbls_avail = np.where(
     (silver_df['point_source_parameters_num_images'].to_numpy() == 2))[0]
# take the catalog idxs you want
catalog_idx_avail = silver_df.loc[silver_withkin_dbls_avail,'catalog_idx'].to_numpy()
silver_withkin_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_dbls,replace=False)
# then remove them from the dataframe
silver_df = silver_df[~silver_df['catalog_idx'].isin(silver_withkin_dbls_catalog_idxs)].reset_index(drop=True)

# SILVER NO KIN
num_quads = 36
num_total = 300
silver_quads_avail = np.where(
    (silver_df['point_source_parameters_num_images'].to_numpy() == 4))[0]
# if not enough quads, include more doubles
if len(silver_quads_avail)<num_quads:
    num_quads = len(silver_quads_avail)
num_dbls = num_total - num_quads
# take the catalog idxs you want
silver_quads_catalog_idxs = None # edge case where no quads left...
if num_quads > 0:
    catalog_idx_avail = silver_df.loc[silver_quads_avail,'catalog_idx'].to_numpy()
    silver_quads_catalog_idxs = np.random.choice(catalog_idx_avail,
        size=num_quads,replace=False)
    # then remove them from the dataframe
    silver_df = silver_df[~silver_df['catalog_idx'].isin(silver_quads_catalog_idxs)].reset_index(drop=True)

silver_dbls_avail = np.where(
    (silver_df['point_source_parameters_num_images'].to_numpy() == 2))[0]
# take the catalog idxs you want
catalog_idx_avail = silver_df.loc[silver_dbls_avail,'catalog_idx'].to_numpy()
silver_dbls_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_dbls,replace=False)
# then remove them from the dataframe
silver_df = silver_df[~silver_df['catalog_idx'].isin(silver_dbls_catalog_idxs)].reset_index(drop=True)



##############################
# Set-up inference configs
##############################
likelihood_configs = {

    # NIRSPEC likelihoods (10 lenses)
    'nirspec_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':nirspec_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'NIRSPEC',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GOLD_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # MUSE likelihoods (40 lenses)
    'muse_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':muse_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'MUSE',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GOLD_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'muse_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':muse_dbls_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'MUSE',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GOLD_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # 4MOST likelihoods (150 lenses)
    '4MOST_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':fourmost_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GOLD_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    '4MOST_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':fourmost_dbls_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':GOLD_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    ################
    # Silver Lenses
    ################

    # Silver 4MOST likelihoods (300 lenses)
    'silver_4MOST_quads':{
        'posteriors_h5_file':silver_quads_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_withkin_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':SILVER_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    'silver_4MOST_dbls':{
        'posteriors_h5_file':silver_dbls_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_withkin_dbls_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':SILVER_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },


    # Silver no kinematics (300 lenses)
    'silver_nokin_dbls':{
        'posteriors_h5_file':silver_dbls_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_dbls_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':SILVER_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },
}

# handle edge case where no silver quads w/out kin
if silver_quads_catalog_idxs is not None:
    likelihood_configs['silver_nokin_quads'] = {
        'posteriors_h5_file':silver_quads_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':silver_quads_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'log_prob_gamma_nu_int':SILVER_GAMMA_LENS_PRIOR,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    }