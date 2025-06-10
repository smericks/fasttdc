import os
import sys
import json
import numpy as np
from importlib import import_module

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, 'InferenceRuns'))
sys.path.insert(0, os.path.join(dirname, '../..'))
import tdc_sampler
import time
import argparse

parser = argparse.ArgumentParser(description="Run model with specific configurations.")
# TODO: feed in config name
parser.add_argument('--config',help="Name of config, stored in InferenceRuns/") # ex: exp0_1_config
parser.add_argument("--use-MPI", action="store_true", help="Use MPI for parallel processing.")
parser.add_argument("--use-multiprocess", action="store_true", help="Use multiprocess for parallel processing.")
args = parser.parse_args()
config_name = args.config
use_MPI = args.use_MPI
use_multiprocess = args.use_multiprocess

# TODO: switch to feeding in a config file
config_module = import_module(config_name)


# USER SETTINGS HERE
USE_ASTROPY = True
np.random.seed(config_module.RANDOM_SEED)

static_dv_filepath = config_module.static_dv_file 

# Check if static data vectors already exist
if not os.path.exists(static_dv_filepath):
    print(f"File {static_dv_filepath} does not exist. Please use step08_datavectors.py to create it.")

###################
# RUN MCMC HERE!!!
###################

# load in static data vectors
with open(static_dv_filepath, 'r') as file:
    data_vector_dict_list = json.load(file)

# return it to np.array
for dv_dict in data_vector_dict_list:
    for key in dv_dict.keys():
        if isinstance(dv_dict[key], list):
            dv_dict[key] = np.asarray(dv_dict[key])

likelihood_obj_list = []
# TODO: need to handle edge case when re-sampling, and might be missing silver quads...
for dv_dict in data_vector_dict_list:
    #print('Processing ', subsamp)
    fpd_sample_shape = dv_dict['fpd_samples'].shape
    if 'kin_pred_samples' in dv_dict.keys():
        kin_pred_samples_shape = dv_dict['kin_pred_samples'].shape
        lklhd_obj = tdc_sampler.TDCKinLikelihood(
            fpd_sample_shape, kin_pred_samples_shape,
            cosmo_model=config_module.COSMO_MODEL,
            use_astropy=USE_ASTROPY)
    else:
        lklhd_obj = tdc_sampler.TDCLikelihood(fpd_sample_shape,
            cosmo_model=config_module.COSMO_MODEL,
            use_astropy=USE_ASTROPY)

    likelihood_obj_list.append(lklhd_obj)

# tdc_sampler likelihood object

start = time.time()
tenIFU_chain = tdc_sampler.fast_TDC(likelihood_obj_list,data_vector_dict_list,
    num_emcee_samps=config_module.NUM_MCMC_EPOCHS,
    n_walkers=config_module.NUM_MCMC_WALKERS,
    use_mpi=use_MPI, use_multiprocess=use_multiprocess,
    backend_path=config_module.BACKEND_PATH,
    reset_backend=config_module.RESET_BACKEND)
end = time.time()
print('Time to run MCMC:',end-start)

