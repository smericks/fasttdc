import h5py
import sys
from importlib import import_module
import argparse
import os
dirname = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(dirname, 'InferenceRuns')) 
sys.path.insert(0, os.path.join(dirname, '../..')) 
#sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler

import Experiments.lsst_forecast.DataVectors.prep_data_vectors as prep_data_vectors

""""
ex usage: 
    python3 step08_inference.py --config_path=InferenceRuns/exp1_1/exp1_1_config.py
"""


def main(args):

    print('args.config_path', args.config_path )
    config_module = import_module(args.config_path)

    likelihood_configs = config_module.likelihood_configs

    likelihood_objs = []
    for subsamp in likelihood_configs.keys():
        input_dict = likelihood_configs[subsamp]
        likelihood_objs.append( 
            prep_data_vectors.construct_likelihood_obj(**input_dict)
        )

    # saves to a backend file for us!

    _ = tdc_sampler.fast_TDC(
        likelihood_objs,
        num_emcee_samps=config_module.NUM_MCMC_EPOCHS,
        n_walkers=config_module.NUM_MCMC_WALKERS,
        backend_path=config_module.BACKEND_PATH)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Runs hierarchical inference"
    )
    parser.add_argument(
        "--config_path",
        type=str)

    args = parser.parse_args()
    main(args)

