# NOTES: 
# - hardcoded to use lens 720 (see notebook for explanation)

import numpy as np
import pandas as pd
import h5py
import json
from scipy.stats import norm, multivariate_normal, truncnorm
import sys
import os
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, '../..'))
import tdc_sampler
import time 
import copy
import argparse

# feed in lens redshift mean and source redshift mean as args
parser = argparse.ArgumentParser(description="Run model with specific configurations.")
# TODO: feed in config name
parser.add_argument('--z_lens',type=float,help="Mean lens redshift") # ex: exp0_1_config
parser.add_argument('--z_src',type=float,help="Mean lens redshift") # ex: exp0_1_config
parser.add_argument("--use-MPI", action="store_true", help="Use MPI for parallel processing.")
parser.add_argument("--use-multiprocess", action="store_true", help="Use multiprocess for parallel processing.")
args = parser.parse_args()
z_lens_mean = args.z_lens
print(type(z_lens_mean))
z_src_mean = args.z_src
use_MPI = args.use_MPI
use_multiprocess = args.use_multiprocess
OMEGA_M_PRIOR = True

# backend path
backend_path = 'InferenceRuns/FOM_vs_z/OmegaM_redshift_zlens=%.2f_zsrc=%.2f_backend.h5'%(z_lens_mean,z_src_mean)


# RANDOM SEED FOR REPRODUCIBILITY
np.random.seed(6)

# load in pre-computed portion of the data vectors
static_dv_filepath = 'InferenceRuns/FOM_vs_z/dv_dict_lens720_fp_prec_0.02.json'
with open(static_dv_filepath, 'r') as file:
    data_vector_dict_lens720 = json.load(file)

# constants
num_td = 1
num_fpd_samps = 5000
num_kin_bins = 10

# ground truth part
truth_metadata_file = 'DataVectors/gold/truth_metadata.csv'
truth_df = pd.read_csv(truth_metadata_file)

# return it to np.array
for key in data_vector_dict_lens720.keys():
    if isinstance(data_vector_dict_lens720[key], list):
        data_vector_dict_lens720[key] = np.asarray(data_vector_dict_lens720[key])


def modify_data_vector_dict720(z_lens,z_src,kin_meas_error_percent=0.01,
        kin_meas_error_kmpersec=None):


    data_vector_dict_lens720['z_lens'] = np.asarray([z_lens])
    data_vector_dict_lens720['z_src'] = np.asarray([z_src])

    # constants
    from astropy.cosmology import FlatLambdaCDM, w0waCDM
    gt_cosmo_astropy = FlatLambdaCDM(H0=70.,Om0=0.3)
    # DESI + CMB in Table V of DESI DRII COSMO PAPER
    #gt_cosmo_astropy = w0waCDM(H0=63.6, Om0=0.35, Ode0=0.65, w0=-0.42, wa=-1.75)

    td_meas_error_percent = 0.01
    td_meas_error_days = None

    lens720_row = truth_df[truth_df['catalog_idx']==720]
    lens720_row = lens720_row.reset_index(drop=True)

    # change redshift
    lens720_row['main_deflector_parameters_z_lens'] = data_vector_dict_lens720['z_lens'][0]
    lens720_row['source_parameters_z_source'] = data_vector_dict_lens720['z_src'][0]

    from Experiments.lsst_forecast.DataVectors.prep_data_vectors import emulate_measurements, retrieve_truth_td, retrieve_truth_kin
    from Utils.ground_truth_utils import populate_truth_Ddt_timedelays, populate_truth_sigma_v_IFU
    # re-compute td ground truth

    populate_truth_Ddt_timedelays(lens720_row,gt_cosmo_astropy)

    # emulate time-delay
    td_truth = retrieve_truth_td(lens720_row, num_td)
    td_meas, td_meas_prec = emulate_measurements(td_truth, 
        td_meas_error_percent,td_meas_error_days)
    # TODO: set mean to be the truth every time (avoid random-ness)
    td_meas = td_truth

    # populate data vector with time delays
    data_vector_dict_lens720['td_measured'] = np.repeat(td_meas[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    data_vector_dict_lens720['td_likelihood_prec'] = np.repeat(td_meas_prec[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    data_vector_dict_lens720['td_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(data_vector_dict_lens720['td_likelihood_prec']))) )

    # TODO: emulate kin. obs
    # re-compute kinematics truth
    populate_truth_sigma_v_IFU(lens720_row,gt_cosmo_astropy)

    # emulate kin. obs.
    kin_truth = retrieve_truth_kin(lens720_row,kinematic_type='NIRSPEC')
    sigma_v_meas,sigma_v_meas_prec = emulate_measurements(kin_truth,
        kin_meas_error_percent,kin_meas_error_kmpersec)
    sigma_v_meas = kin_truth

    # populate data vector with kinematics
    data_vector_dict_lens720['sigma_v_measured'] = np.repeat(sigma_v_meas[:, np.newaxis, :],
                    num_fpd_samps, axis=1)
    data_vector_dict_lens720['sigma_v_likelihood_prec'] = np.repeat(
        sigma_v_meas_prec[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)

    data_vector_dict_lens720['sigma_v_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(num_kin_bins/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(data_vector_dict_lens720['sigma_v_likelihood_prec']))))

###################################################################
# STEP 2: Loop through and run inference toggling % precision
###################################################################

# constants
COSMO_MODEL = 'w0waCDM_lambda_int_beta_ani'
USE_ASTROPY = True
HI_REWEIGHTING = False

data_vector_dict_list = []
# let's use 10 lenses, pulling redshifts from narrow-ish Gaussian
for j in range(0,10):
    z_d = norm.rvs(loc=z_lens_mean,scale=0.1)
    z_s = truncnorm.rvs(-(z_src_mean-z_d)/0.1,np.inf,loc=z_src_mean,scale=0.1)
    modify_data_vector_dict720(z_d,z_s)
    data_vector_dict_list.append(copy.deepcopy(data_vector_dict_lens720))

likelihood_obj_list = []
for dv_dict in data_vector_dict_list:
    #print('Processing ', subsamp)
    fpd_sample_shape = dv_dict['fpd_samples'].shape
    if 'kin_pred_samples' in dv_dict.keys():
        kin_pred_samples_shape = dv_dict['kin_pred_samples'].shape
        lklhd_obj = tdc_sampler.TDCKinLikelihood(
            fpd_sample_shape, kin_pred_samples_shape,
            cosmo_model=COSMO_MODEL,
            use_astropy=USE_ASTROPY,
            use_gamma_info=HI_REWEIGHTING)
    else:
        lklhd_obj = tdc_sampler.TDCLikelihood(fpd_sample_shape,
            cosmo_model=COSMO_MODEL,
            use_astropy=USE_ASTROPY,
            use_gamma_info=HI_REWEIGHTING)

    likelihood_obj_list.append(lklhd_obj)

# tdc_sampler likelihood object

NUM_MCMC_EPOCHS = 10000 # don't need as many b/c few lenses + informative prior
NUM_MCMC_WALKERS = 50

start = time.time()
add_one_lens_chain = tdc_sampler.fast_TDC(likelihood_obj_list,data_vector_dict_list,
    num_emcee_samps=NUM_MCMC_EPOCHS,
    n_walkers=NUM_MCMC_WALKERS,
    use_mpi=use_MPI, use_multiprocess=use_multiprocess,
    use_informative=True,
    backend_path=backend_path,
    use_OmegaM=OMEGA_M_PRIOR)
end = time.time()
print('Time to run MCMC:',end-start)

# save the chain!!!


# TODO: this might only be fair without the informative prior on beta_ani, lambda_int ...
# ordering is: fpd, kin, time-delay
# baseline (3%, 3%, 3%)

# fpd test ([1%,3%,5%,10%], 3%, 3%)

# kin test (3%, [1%,3%,5%,10%], 3%)

# time-delay test (3%, 3%, [1%,3%,5%,10%])