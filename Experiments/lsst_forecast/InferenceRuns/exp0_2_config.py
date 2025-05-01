# experiment 1.2: Gold-Only Baseline, Human-Bias Selected

import h5py
import pandas as pd
import numpy as np

# file locations
gold_quads_h5_file = 'DataVectors/gold/quad_posteriors_KIN.h5'
gold_dbls_h5_file = 'DataVectors/gold/dbl_posteriors_KIN.h5'
gold_metadata_file = 'DataVectors/gold/truth_metadata.csv'
silver_quads_h5_file = 'DataVectors/silver/quad_posteriors_KIN.h5'
silver_dbls_h5_file = 'DataVectors/silver/dbl_posteriors_KIN.h5'
silver_metadata_file = 'DataVectors/silver/truth_metadata.csv'


NUM_FPD_SAMPS = 5000
NUM_MCMC_EPOCHS = 5
NUM_MCMC_WALKERS = 20
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'
BACKEND_PATH = 'InferenceRuns/exp0_2/lcdm_backend.h5'

# catalog indices available
with h5py.File(gold_quads_h5_file,'r') as h5:
    quad_catalog_idxs = h5['catalog_idxs'][:]
with h5py.File(gold_dbls_h5_file,'r') as h5:
    dbl_catalog_idxs = h5['catalog_idxs'][:]

# truth information for those indices
truth_df = pd.read_csv(gold_metadata_file)
truth_df_catalog_idxs = truth_df.loc[:,'catalog_idx'].to_numpy()

##############################
# Human selection cuts!!
##############################
nirspec_quads_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(truth_df['td01'].to_numpy()) > 30.) | 
     (np.abs(truth_df['td02'].to_numpy()) > 30.) | 
     (np.abs(truth_df['td03'].to_numpy()) > 30.)) &
    #(truth_df['lens_light_parameters_mag_app'].to_numpy() > 22.) &
    (truth_df['lens_light_parameters_mag_app'].to_numpy() < 24.) &
    (truth_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]

print('%d NIRSPEC lenses available'%(len(nirspec_quads_avail)))
print('Choosing 10...')
nirspec_quads_idxs = nirspec_quads_avail[:10]
quads_taken = nirspec_quads_idxs

# available for MUSE (most stringent cut)
muse_quads_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 4) &
    ((np.abs(truth_df['td01'].to_numpy()) > 30.) | 
     (np.abs(truth_df['td02'].to_numpy()) > 30.) | 
     (np.abs(truth_df['td03'].to_numpy()) > 30.)) &
    (truth_df['lens_light_parameters_mag_app'].to_numpy() < 22.) &
    (truth_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]
# remove the ones already chosen for nirspec
muse_quads_avail = np.setdiff1d(muse_quads_avail, quads_taken)

print('%d MUSE quad lenses available'%(len(muse_quads_avail)))
print('Choosing 19...')
muse_quads_idxs = muse_quads_avail
quads_taken = np.concatenate((quads_taken,muse_quads_idxs))

muse_dbls_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 2) &
    ((np.abs(truth_df['td01'].to_numpy()) > 30.) | 
     (np.abs(truth_df['td02'].to_numpy()) > 30.) | 
     (np.abs(truth_df['td03'].to_numpy()) > 30.)) &
    (truth_df['lens_light_parameters_mag_app'].to_numpy() < 22.) &
    (truth_df['source_parameters_mag_app'].to_numpy() < 24.)
)[0]

print('%d MUSE double lenses available'%(len(muse_dbls_avail)))
print('Choosing 21...')
muse_dbls_idxs = muse_dbls_avail[:21]
dbls_taken = muse_dbls_idxs

fourmost_quads_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 4))[0]
fourmost_quads_avail = np.setdiff1d(fourmost_quads_avail,quads_taken)
print('%d 4MOST quad lenses available'%(len(fourmost_quads_avail)))
print('Choosing 75')
fourmost_quads_idxs = fourmost_quads_avail[:75]
quads_taken = np.concatenate((quads_taken,fourmost_quads_idxs))

fourmost_dbls_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 2))[0]
fourmost_dbls_avail = np.setdiff1d(fourmost_dbls_avail,dbls_taken)
print('%d 4MOST double lenses available'%(len(fourmost_dbls_avail)))
print('Choosing 75')
fourmost_dbls_idxs = fourmost_dbls_avail[:75]
dbls_taken = np.concatenate((dbls_taken,fourmost_dbls_idxs))


# now we add in the silver lenses with kin
silver_withkin_quads_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 4))[0]
silver_withkin_quads_avail = np.setdiff1d(silver_withkin_quads_avail,quads_taken)
print('%d Silver 4MOST quad lenses available'%(len(silver_withkin_quads_avail)))
print('Choosing 36')
silver_withkin_quads_idxs = silver_withkin_quads_avail[:36]
quads_taken = np.concatenate((quads_taken,silver_withkin_quads_idxs))

silver_withkin_dbls_avail = np.where(
     (truth_df['point_source_parameters_num_images'].to_numpy() == 2))[0]
silver_withkin_dbls_avail = np.setdiff1d(silver_withkin_dbls_avail,dbls_taken)
print('%d Silver 4MOST double lenses available'%(len(silver_withkin_dbls_avail)))
print('Choosing 264')
silver_withkin_dbls_idxs = silver_withkin_dbls_avail[:264]
dbls_taken = np.concatenate((dbls_taken,silver_withkin_dbls_idxs))


# now we add in the silver with no kin lenses
silver_quads_avail = np.where(
    (truth_df['point_source_parameters_num_images'].to_numpy() == 4))[0]
silver_quads_avail = np.setdiff1d(silver_quads_avail,quads_taken)
print('%d Silver quad lenses available'%(len(silver_quads_avail)))
print('Choosing 12')
silver_quads_idxs = silver_quads_avail
quads_taken = np.concatenate((quads_taken,silver_quads_idxs))

silver_dbls_avail = np.where(
     (truth_df['point_source_parameters_num_images'].to_numpy() == 2))[0]
silver_dbls_avail = np.setdiff1d(silver_dbls_avail,dbls_taken)
print('%d Silver double lenses available'%(len(silver_dbls_avail)))
print('Choosing 288')
silver_dbls_idxs = silver_dbls_avail[:288]
dbls_taken = np.concatenate((dbls_taken,silver_dbls_idxs))


##############################
# Set-up inference configs
##############################
likelihood_configs = {

    # NIRSPEC likelihoods (10 lenses)
    'nirspec_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[nirspec_quads_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'NIRSPEC',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    # MUSE likelihoods (40 lenses)
    'muse_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[muse_quads_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'MUSE',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    'muse_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[muse_dbls_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':0.03,
        'td_meas_error_days':None,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'MUSE',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    # 4MOST likelihoods (150 lenses)
    '4MOST_quads':{
        'posteriors_h5_file':gold_quads_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[fourmost_quads_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    '4MOST_dbls':{
        'posteriors_h5_file':gold_dbls_h5_file,
        'metadata_file':gold_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[fourmost_dbls_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    ################
    # Silver Lenses
    ################

    # Silver 4MOST likelihoods (300 lenses)
    'silver_4MOST_quads':{
        'posteriors_h5_file':silver_quads_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[silver_withkin_quads_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    'silver_4MOST_dbls':{
        'posteriors_h5_file':silver_dbls_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[silver_withkin_dbls_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':'4MOST',
        'kin_meas_error_percent':0.05,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },


    # Silver no kinematics (300 lenses)
    'silver_nokin_quads':{
        'posteriors_h5_file':silver_quads_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[silver_quads_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },

    'silver_4MOST_dbls':{
        'posteriors_h5_file':silver_dbls_h5_file,
        'metadata_file':silver_metadata_file,
        'catalog_idxs':truth_df_catalog_idxs[silver_dbls_idxs],
        'cosmo_model':COSMO_MODEL,
        'td_meas_error_percent':None,
        'td_meas_error_days':5.,
        'kappa_ext_meas_error_value':0.05,
        'kinematic_type':None,
        'kin_meas_error_percent':None,
        'kin_meas_error_kmpersec':None,
        'num_gaussianized_samps':NUM_FPD_SAMPS
    },
}