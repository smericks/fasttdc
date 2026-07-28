# imports
import h5py
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from mpi4py import MPI

# ad-hoc fix for imports 
import os
import sys
dirname = os.getcwd()
sys.path.insert(0, os.path.join(dirname, '../..'))
import tdc_sampler
# moved helper functions to script
from completo_h0test_utils import dv_dict_from_completo, dv_dict_from_completo_extendedCPDF

# ---- MPI ----
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# HARDCODED GROUND TRUTH COSMOLOGY
GROUNDTRUTH_COSMO = FlatLambdaCDM(H0=70.,Om0=0.3)

# HARDCODED SAVE PATH FOR CHAIN
TEST_PATH = '/scratch/users/sydney3/project3/test1/'
model_type = 'nsf/'
MODEL_PATH = TEST_PATH + model_type
CHAIN_PATH = '/scratch/users/sydney3/project3/test1/InferenceRuns/nsf/with_lenscenter_200lenses_backend.h5'

# HARDCODED LENS INDICES FOR USE WITH: 
narrow_gamma_tenplus_td = [ # gamma centered ~ 2.03, time-delay > 10 days 
    3, 7, 12, 18, 22, 34, 35, 37, 41, 80, 82, 84, 91, 93, 109, 113, 114, 122,
    128, 129, 139, 148, 154, 157, 164, 166, 167, 170, 173, 179, 191, 193, 196,
    209, 212, 220, 246, 251, 254, 255, 260, 264, 269, 271, 272, 274, 275, 278,
    285, 288, 290, 292, 293, 297, 299, 300, 301, 304, 312, 314, 317, 320, 321,
    328, 329, 332, 333, 334, 339, 340, 341, 342, 346, 352, 357, 359, 360, 373,
    375, 393, 397, 408, 409, 414, 416, 423, 426, 433, 436, 437, 439, 440, 445,
    453, 462, 465, 472, 477, 479, 481, 487, 501, 507, 509, 510, 511, 515, 520,
    522, 523, 529, 535, 536, 539, 540, 543, 550, 568, 570, 572, 576, 579, 583,
    606, 610, 619, 620, 628, 629, 633, 638, 642, 656, 658, 661, 673, 676, 683,
    689, 693, 695, 699, 713, 719, 723, 728, 731, 737, 738, 739, 748, 750, 753,
    756, 766, 780, 784, 789, 793, 797, 799, 811, 815, 816, 821, 824, 829, 843,
    855, 857, 863, 865, 867, 872, 886, 889, 900, 903, 908, 913, 916, 921, 927,
    938, 940, 950, 955, 956, 959, 964, 966, 970, 972, 974, 976, 977, 978, 985,
    988, 991
]


# ==========================================================
# Only rank 0 builds the (expensive) data vectors
# ==========================================================
if rank == 0:

    dv_dict_dbls_nsf_CPDF, dv_dict_quads_nsf_CPDF = dv_dict_from_completo_extendedCPDF(
        (MODEL_PATH + '1ktest_seed7_5ksamps.npy'),
        (TEST_PATH + 'test10k_seed7.h5'),
        (TEST_PATH+'train10k_seed1.h5'),
        chosen_idxs=narrow_gamma_tenplus_td, GROUNDTRUTH_COSMO=GROUNDTRUTH_COSMO)
else:
    dv_dict_dbls_nsf_CPDF = None
    dv_dict_quads_nsf_CPDF = None

# ==========================================================
# Broadcast to all ranks (workers need it for the pickled closure)
# ==========================================================
dv_dict_dbls_nsf_CPDF  = comm.bcast(dv_dict_dbls_nsf_CPDF,  root=0)
dv_dict_quads_nsf_CPDF = comm.bcast(dv_dict_quads_nsf_CPDF, root=0)

# sanity check
print(f"[rank {rank}] quads fpd shape: "
      f"{np.shape(dv_dict_quads_nsf_CPDF['fpd_samples'])}", flush=True)

# ==========================================================
# All ranks build likelihood objects (cheap - just needs shapes)
# ==========================================================
lklhd_obj_dbls = tdc_sampler.TDCLikelihoodCompleto(
    fpd_sample_shape=np.shape(dv_dict_dbls_nsf_CPDF['fpd_samples']),
    cosmo_model='LCDM_completo_cPDF',
    use_astropy=True, use_gamma_info=True)

lklhd_obj_quads = tdc_sampler.TDCLikelihoodCompleto(
    fpd_sample_shape=np.shape(dv_dict_quads_nsf_CPDF['fpd_samples']),
    cosmo_model='LCDM_completo_cPDF',
    use_astropy=True, use_gamma_info=True)

# ==========================================================
# Run chain — ALL ranks call this.
# Inside, workers hit pool.wait(); master drives the sampler.
# ==========================================================

# run chain & save to backend_path
my_chain_nsf_cPDF = tdc_sampler.fast_TDC([lklhd_obj_quads,lklhd_obj_dbls], 
    [dv_dict_quads_nsf_CPDF,dv_dict_dbls_nsf_CPDF], num_emcee_samps=500, # NOTE when debugging turn this down!
    n_walkers=35, use_mpi=False, use_multiprocess=True, backend_path=CHAIN_PATH, # <-- TODO save path
    reset_backend=True,sampler_type='emcee',num_cpdf_params=2) # NOTE: we have to specify this now!!