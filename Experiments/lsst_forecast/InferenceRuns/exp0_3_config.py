# experiment 1.3: HYPOTHESIS: high z_src + long td = most useful for (w0,wa)

import h5py
import pandas as pd
import numpy as np
from scipy.stats import norm, multivariate_normal

# random seed
RANDOM_SEED = 1

# file locations
static_dv_file = 'InferenceRuns/exp0_3/static_datavectors_seed'+str(RANDOM_SEED)+'.json'
gold_quads_h5_file = 'DataVectors/gold/quad_posteriors_DEBIASED.h5'
gold_dbls_h5_file = 'DataVectors/gold/dbl_posteriors_DEBIASED.h5'
gold_metadata_file = 'DataVectors/gold/truth_metadata.csv'
silver_quads_h5_file = 'DataVectors/silver/quad_posteriors_DEBIASED.h5'
silver_dbls_h5_file = 'DataVectors/silver/dbl_posteriors_DEBIASED.h5'
silver_metadata_file = 'DataVectors/silver/truth_metadata.csv'

NUM_FPD_SAMPS = 5000
NUM_MCMC_EPOCHS = 10
NUM_MCMC_WALKERS = 50
COSMO_MODEL = 'w0waCDM_lambda_int_beta_ani'
HI_REWEIGHTING = False
mu_lp_gold = np.asarray([0.85,0.,0.,2.09,0.,0.,0.,0.,0.,0.]) # hst_norms.csv
stddev_lp_gold = np.asarray([0.28,0.06,0.06,0.16,0.20,0.20,0.06,0.06,0.34,0.34])
mu_lp_silver = np.asarray([1.42,0.,0.,2.03,0.,0.,0.,0.,0.,0.])# norms2.csv
stddev_lp_silver = np.asarray([0.70,0.1,0.1,0.20,0.20,0.20,0.06,0.06,0.37,0.37])
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
BACKEND_PATH = 'InferenceRuns/exp0_3/w0wa_seed'+str(RANDOM_SEED)+'_backend.h5'
RESET_BACKEND=True

# truth information for those indices
truth_df = pd.read_csv(gold_metadata_file)

# NOTE: when evaluating kinematics at each sample, some samples return nan, we exclude those lenses
gold_nan_kin_vals =  []
gold_df = truth_df[~truth_df['catalog_idx'].isin(gold_nan_kin_vals)].reset_index(drop=True)
# track catalog_idxs
gold_df_catalog_idxs = gold_df.loc[:,'catalog_idx'].to_numpy()

# use the random seed
np.random.seed(RANDOM_SEED)

#########################
# Human selection cuts!!
#########################
num_muse=40
# TODO: MUSE first, more stringent cut...
longest_td = np.abs(gold_df['td03'].to_numpy()) # 3rd td for quads
longest_td[np.isnan(longest_td)] = np.abs(gold_df['td01'].to_numpy())[np.isnan(longest_td)] # 1st td for doubles
muse_lenses_avail = np.where(
    (longest_td > 200.) & 
    (gold_df['source_parameters_z_source'].to_numpy() > 2.2) & 
    (gold_df['lens_light_parameters_mag_app'].to_numpy() < 22.) 
)
catalog_idx_avail = gold_df.loc[muse_lenses_avail,'catalog_idx'].to_numpy()
muse_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_muse,replace=False)

muse_df = gold_df[gold_df['catalog_idx'].isin(muse_catalog_idxs)].reset_index(drop=True)

muse_quads_catalog_idxs = muse_df[~np.isnan(muse_df['td03'].to_numpy())]['catalog_idx'].to_numpy()
muse_dbls_catalog_idxs = muse_df[np.isnan(muse_df['td03'].to_numpy())]['catalog_idx'].to_numpy()

# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(muse_catalog_idxs)].reset_index(drop=True)

# GOLD NIRSPEC
num_nirspec=10
# select on time-delay and source z
longest_td = np.abs(gold_df['td03'].to_numpy()) # 3rd td for quads
longest_td[np.isnan(longest_td)] = np.abs(gold_df['td01'].to_numpy())[np.isnan(longest_td)] # 1st td for doubles
nirspec_lenses_avail = np.where(
    (longest_td > 200.) & 
    (gold_df['source_parameters_z_source'].to_numpy() > 2.2) & 
    (gold_df['lens_light_parameters_mag_app'].to_numpy() < 24.) 
)
catalog_idx_avail = gold_df.loc[nirspec_lenses_avail,'catalog_idx'].to_numpy()
nirspec_catalog_idxs = np.random.choice(catalog_idx_avail,
    size=num_nirspec,replace=False)

nirspec_df = gold_df[gold_df['catalog_idx'].isin(nirspec_catalog_idxs)].reset_index(drop=True)

nirspec_quads_catalog_idxs = nirspec_df[~np.isnan(nirspec_df['td03'].to_numpy())]['catalog_idx'].to_numpy()
nirspec_dbls_catalog_idxs = nirspec_df[np.isnan(nirspec_df['td03'].to_numpy())]['catalog_idx'].to_numpy()
# then remove them from the dataframe
gold_df = gold_df[~gold_df['catalog_idx'].isin(nirspec_catalog_idxs)].reset_index(drop=True)


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

# NOTE: when evaluating kinematics at each sample, some samples return nan, we exclude those lenses
silver_nan_kinematic_vals = [ 26,   41,   56,  104,  106,  134,  198,  263, 
    269,  353,  544,  616,  618,  643, 661,  669,  727,  842,  848, 862,  877,
    1300, 1322, 1411, 938,  947,  960, 1431 ] 
silver_df = gold_df[~gold_df['catalog_idx'].isin(silver_nan_kinematic_vals)].reset_index(drop=True)

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
    'nirspec_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':nirspec_dbls_catalog_idxs,
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'NIRSPEC',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS,
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },

    # MUSE likelihoods (40 lenses)
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
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
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
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
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
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
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
        'lens_params_nu_int_means':mu_lp_silver,
        'lens_params_nu_int_stddevs':stddev_lp_silver,
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
        'lens_params_nu_int_means':mu_lp_silver,
        'lens_params_nu_int_stddevs':stddev_lp_silver,
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
        'lens_params_nu_int_means':mu_lp_silver,
        'lens_params_nu_int_stddevs':stddev_lp_silver,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    },
}

# handle edge case for platinum quads
if len(nirspec_quads_catalog_idxs)>0:
    likelihood_configs['nirspec_quads'] = {
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
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    }

if len(muse_quads_catalog_idxs)>0:
    likelihood_configs['muse_quads'] = {
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
        'lens_params_nu_int_means':mu_lp_gold,
        'lens_params_nu_int_stddevs':stddev_lp_gold,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
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
        'lens_params_nu_int_means':mu_lp_silver,
        'lens_params_nu_int_stddevs':stddev_lp_silver,
        'log_prob_beta_ani_nu_int':BETA_ANI_PRIOR
    }