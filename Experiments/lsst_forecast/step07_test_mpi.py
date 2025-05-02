import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, '../..'))
from Experiments.lsst_forecast.DataVectors.prep_data_vectors import create_static_data_vectors
import tdc_sampler
import time
import argparse

parser = argparse.ArgumentParser(description="Run model with specific configurations.")
parser.add_argument("--use-MPI", action="store_true", help="Use MPI for parallel processing.")
parser.add_argument("--use-multiprocess", action="store_true", help="Use MPI for parallel processing.")
args = parser.parse_args()
use_MPI = args.use_MPI
use_multiprocess = args.use_multiprocess

# USER SETTINGS HERE (TODO: change filepaths)
static_dv_filepath = 'DataVectors/gold/mpi_test.json'
np.random.seed(123)
print(np.random.uniform(0,1))

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

# retrieve usable catalog idxs
with h5py.File('DataVectors/gold/quad_posteriors_KIN.h5','r') as h5:
    quad_catalog_idxs = h5['catalog_idxs'][:]

# create static data vectors 
data_vector_dict = create_static_data_vectors(
    posteriors_h5_file='DataVectors/gold/quad_posteriors_KIN.h5',
    metadata_file='DataVectors/gold/truth_metadata.csv',
    catalog_idxs=quad_catalog_idxs[:10],
    cosmo_model='LCDM_lambda_int_beta_ani',
    td_meas_error_days=5.,
    kappa_ext_meas_error_value=0.05,
    kinematic_type='NIRSPEC',
    kin_meas_error_kmpersec=5.,
    num_gaussianized_samps=5000,
    log_prob_gamma_nu_int=norm(loc=2.,scale=0.2).logpdf,
    log_prob_beta_ani_nu_int=norm(loc=0.,scale=0.2).logpdf
)

# TODO: try switching it to a list
for key in data_vector_dict.keys():
    data_vector_dict[key] = data_vector_dict[key].tolist()

# Check if the file already exists
if os.path.exists(static_dv_filepath):
    print(f"File {static_dv_filepath} already exists.")
else:
    # Write list of static data vectors to a JSON file
    with open(static_dv_filepath, 'w') as file:
        json.dump([data_vector_dict], file, indent=4)

###################
# RUN MCMC HERE!!!
###################

# load in static data vectors
with open(static_dv_filepath, 'r') as file:
    data_vector_dict_list = json.load(file)

# return it to np.array
for dv_dict in data_vector_dict_list:
    for key in dv_dict.keys():
        dv_dict[key] = np.asarray(dv_dict[key])


# tdc_sampler likelihood object
fpd_sample_shape = data_vector_dict_list[0]['fpd_samples'].shape
kin_pred_samples_shape = data_vector_dict_list[0]['kin_pred_samples'].shape

quad_kin_lklhd_kappa_ext = tdc_sampler.TDCKinLikelihood(
    fpd_sample_shape, kin_pred_samples_shape,
    cosmo_model='LCDM_lambda_int_beta_ani',
    use_astropy=USE_ASTROPY)


start = time.time()
tenIFU_chain = tdc_sampler.fast_TDC([quad_kin_lklhd_kappa_ext],data_vector_dict_list, NUM_EMCEE_SAMPS,
    n_walkers=20, use_mpi=use_MPI, use_multiprocess=use_multiprocess)
end = time.time()
print('Time to run MCMC:',end-start)
