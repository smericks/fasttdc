import os
import sys
import json
import numpy as np
from importlib import import_module

dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, 'InferenceRuns'))
sys.path.insert(0, os.path.join(dirname, '../..'))
from Experiments.lsst_forecast.DataVectors.prep_data_vectors import create_static_data_vectors
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
likelihood_configs = config_module.likelihood_configs

# Check if static data vectors already exist
if os.path.exists(static_dv_filepath):
    print(f"File {static_dv_filepath} already exists. Using existing file.")

else:
    print(f"Writing new static data vectors file: {static_dv_filepath}")

    # loop thru each subsample and create static data vectors
    data_vector_dict_list = []
    for subsamp in likelihood_configs.keys():
        #print('Processing ', subsamp)
        input_dict = likelihood_configs[subsamp]

        # TODO: replace args with **input_dict (check all params are there)
        data_vector_dict = create_static_data_vectors(**input_dict)

        # switch numpy arrays to lists for writing to .json
        for key in data_vector_dict.keys():
            data_vector_dict[key] = data_vector_dict[key].tolist()

        # append to list (one for each likelihood object)
        data_vector_dict_list.append(data_vector_dict)

    # Write list of static data vectors to a JSON file
    with open(static_dv_filepath, 'w') as file:
        json.dump(data_vector_dict_list, file, indent=4)

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

# TODO: change how likelihood objects are created...
likelihood_obj_list = []
for sidx,subsamp in enumerate(likelihood_configs.keys()):
    #print('Processing ', subsamp)
    input_dict = likelihood_configs[subsamp]
    fpd_sample_shape = data_vector_dict_list[sidx]['fpd_samples'].shape
    if input_dict['kinematic_type'] is not None:
        kin_pred_samples_shape = data_vector_dict_list[sidx]['kin_pred_samples'].shape
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

