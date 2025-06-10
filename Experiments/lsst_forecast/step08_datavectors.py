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
args = parser.parse_args()
config_name = args.config
config_module = import_module(config_name)

np.random.seed(config_module.RANDOM_SEED)

static_dv_filepath = config_module.static_dv_file 
likelihood_configs = config_module.likelihood_configs

# Check if static data vectors already exist
if os.path.exists(static_dv_filepath):
    print(f"File {static_dv_filepath} already exists, exiting.")

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
            if isinstance(data_vector_dict[key], np.ndarray):
                data_vector_dict[key] = data_vector_dict[key].tolist()

        # append to list (one for each likelihood object)
        data_vector_dict_list.append(data_vector_dict)

    # Write list of static data vectors to a JSON file
    with open(static_dv_filepath, 'w') as file:
        json.dump(data_vector_dict_list, file, indent=4)

