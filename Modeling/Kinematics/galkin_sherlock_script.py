import h5py
import argparse
import pandas as pd
import numpy as np
from filelock import FileLock
import galkin_utils # requires lenstronomy

"""
python3 galkin_sherlock_script.py --lens_idx=1 
    --h5_posteriors_path=/scratch/users/sydney3/forecast/darkenergy-from-LAGN/Experiments/lsst_forecast/DataVectors/gold/quad_posteriors.h5
"""
    
def main(args):
    """ Main function for script. """

    # get args
    LENS_IDX = args.lens_idx
    h5_posteriors_file = args.h5_posteriors_path
    
    #('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/'+
    #    'Experiments/lsst_forecast/DataVectors/gold/quad_posteriors.h5')
    # NOTE: hardcoded for gold right now...
    truth_df = pd.read_csv('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/'+
        'Experiments/lsst_forecast/DataVectors/gold/truth_metadata.csv')

    # read in fpd samps, lens_param_samps, beta_ani_samps
    with FileLock(h5_posteriors_file + ".lock"):

        h5 = h5py.File(h5_posteriors_file, 'r')
        fpd_samps = h5['fpd_samps'][:]
        lens_param_samps = h5['lens_param_samps'][:]
        beta_ani_samps = h5['beta_ani_samps'][:]
        catalog_idxs = h5['catalog_idxs'][:]
        h5.close()

    # compute kinematics
    n_fpd_samps = np.shape(fpd_samps)[1]
    c_sqrtJ_samps = np.empty((n_fpd_samps,1))
    catalog_idx = catalog_idxs[LENS_IDX]
    R_sersic_truth = truth_df.loc[truth_df['catalog_idx']==catalog_idx,
        'lens_light_parameters_R_sersic'].item()
    n_sersic_truth = truth_df.loc[truth_df['catalog_idx']==catalog_idx,
        'lens_light_parameters_n_sersic'].item()

    # compute samples one by one
    for fp_idx in range(0,n_fpd_samps):
        csqrtJ = galkin_utils.ground_truth_c_sqrtJ(
            theta_E=lens_param_samps[LENS_IDX,fp_idx,0],
            gamma_lens=lens_param_samps[LENS_IDX,fp_idx,3],
            R_sersic=R_sersic_truth,n_sersic=n_sersic_truth,
            beta_ani=beta_ani_samps[LENS_IDX,fp_idx])
        c_sqrtJ_samps[fp_idx,0] = csqrtJ

    # filelock & write kinematics...
    with FileLock(h5_posteriors_file + ".lock"):
        h5 = h5py.File(h5_posteriors_file, 'r+')
        h5['c_sqrtJ_samps'][LENS_IDX, ...] = c_sqrtJ_samps
        h5.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Computes c_sqrtJ samples for one lens"
    )
    parser.add_argument(
        "--lens_idx",
        type=int,
        default=0,
        help=("Lens index in posteriors.h5 file (not catalog_idx!)")
    )
    parser.add_argument(
        "--h5_posteriors_path",
        type=str)

    args = parser.parse_args()
    main(args)