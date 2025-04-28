import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler
import Experiments.lsst_forecast.DataVectors.prep_data_vectors as prep_data_vectors

# chain save_path
chain_save_path = 'DataVectors/gold/baseline_chain.h5'

# MCMC settings
NUM_EPOCHS = 5

# file locations
gold_quads_h5_file = 'DataVectors/gold/quad_posteriors_KIN.h5'
gold_dbls_h5_file = 'DataVectors/gold/dbl_posteriors_KIN.h5'
gold_metadata_file = 'DataVectors/gold/truth_metadata.csv'

# catalog indices available
with h5py.File(gold_quads_h5_file,'r') as h5:
    quad_catalog_idxs = h5['catalog_idxs'][:]
with h5py.File(gold_dbls_h5_file,'r') as h5:
    dbl_catalog_idxs = h5['catalog_idxs'][:]

# NIRSPEC likelihoods (10 lenses: 1 quad, 9 doubles)
nirspec_quads_idxs = [quad_catalog_idxs[0]]
nirspec_dbls_idxs = dbl_catalog_idxs[:9]

nirspec_quads_lklhd = prep_data_vectors.construct_likelihood_obj(
    gold_quads_h5_file,gold_metadata_file,nirspec_quads_idxs,
    td_meas_error_percent=0.03,
    kinematic_type='NIRSPEC',kin_meas_error_percent=0.05,
    num_gaussianized_samps=5000)

nirspec_dbls_lklhd = prep_data_vectors.construct_likelihood_obj(
    gold_dbls_h5_file,gold_metadata_file,nirspec_dbls_idxs,
    td_meas_error_percent=0.03,
    kinematic_type='NIRSPEC',kin_meas_error_percent=0.05,
    num_gaussianized_samps=5000)

# MUSE likelihoods (40 lenses: 5 quads, 35 doubles)
muse_quads_idxs = quad_catalog_idxs[1:6]
muse_dbls_idxs = dbl_catalog_idxs[9:44]

muse_quads_lklhd = prep_data_vectors.construct_likelihood_obj(
    gold_quads_h5_file,gold_metadata_file,muse_quads_idxs,
    td_meas_error_percent=0.03,
    kinematic_type='MUSE',kin_meas_error_percent=0.05,
    num_gaussianized_samps=5000)


muse_dbls_lklhd = prep_data_vectors.construct_likelihood_obj(
    gold_dbls_h5_file,gold_metadata_file,muse_dbls_idxs,
    td_meas_error_percent=0.03,
    kinematic_type='MUSE',kin_meas_error_percent=0.05,
    num_gaussianized_samps=5000)

# 4MOST likelihoods (150 lenses: 18 quads, 132 doubles)
fourmost_quads_idxs = quad_catalog_idxs[6:24]
fourmost_dbls_idxs = dbl_catalog_idxs[44:176]

fourmost_quads_lklhd = prep_data_vectors.construct_likelihood_obj(
    gold_quads_h5_file,gold_metadata_file,fourmost_quads_idxs,
    td_meas_error_days=5.,
    kinematic_type='4MOST',kin_meas_error_percent=0.05,
    num_gaussianized_samps=5000)


fourmost_dbls_lklhd = prep_data_vectors.construct_likelihood_obj(
    gold_dbls_h5_file,gold_metadata_file,fourmost_dbls_idxs,
    td_meas_error_days=5.,
    kinematic_type='4MOST',kin_meas_error_percent=0.05,
    num_gaussianized_samps=5000)

my_chain = tdc_sampler.fast_TDC(
    [nirspec_quads_lklhd,nirspec_dbls_lklhd,
     muse_quads_lklhd,muse_dbls_lklhd,
     fourmost_quads_lklhd,fourmost_dbls_lklhd],
    num_emcee_samps=NUM_EPOCHS,
    n_walkers=20)


# TODO: write chain to .h5 file
with h5py.File(chain_save_path,'w') as h5:

    h5.create_dataset('mcmc_chain',data=my_chain)

