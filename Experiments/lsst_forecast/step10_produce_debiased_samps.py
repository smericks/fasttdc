# NOTES: 
# - hardcoded to use lens 720 (see notebook for explanation)
# - this script just produces the kin / fermat potentials so they are static...

import numpy as np
import pandas as pd
import h5py
from scipy.stats import norm, multivariate_normal
import sys
import json
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
from Experiments.lsst_forecast.DataVectors.prep_data_vectors import gaussianize_samples
import Modeling.Kinematics.galkin_utils as galkin_utils
from scipy.stats import truncnorm

# save prefix
save_location = 'InferenceRuns/FOM_vs_z/'

# chosen lens for this test
chosen_cidx = 720
h5_posteriors_file = 'DataVectors/gold/dbl_posteriors_KIN.h5'
truth_metadata_file = 'DataVectors/gold/truth_metadata.csv'
truth_df = pd.read_csv(truth_metadata_file)

# constants
kappa_ext_measurement_error = 0.05
num_fpd_samps = 5000
num_td = 1
lens_params_nu_int_means =  np.asarray([0.85,0.,0.,2.09,0.,0.,0.,0.,0.,0.]) # hst_norms.csv
lens_params_nu_int_stddevs = np.asarray([0.28,0.06,0.06,0.16,0.20,0.20,0.06,0.06,0.34,0.34])
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2)


#######################
# De-Biasing Posteriors
#######################

# TODO: optimize for one lens at a time!
def debiased_samples(h5_posteriors_file,truth_metadata_file,chosen_cidx,fermat_pot_prec):
    """
        fermat_pot_prec (if 2%, fermat_pot_prec=0.02)
    """

    truth_df = pd.read_csv(truth_metadata_file)

    with h5py.File(h5_posteriors_file, 'r') as h5:
        fpd_samps = h5['fpd_samps'][:]
        lens_param_samps = h5['lens_param_samps'][:]
        catalog_idxs = h5['catalog_idxs'][:]

    chosen_fpd_samps = fpd_samps[catalog_idxs == chosen_cidx]
    chosen_lens_param_samps = lens_param_samps[catalog_idxs == chosen_cidx]


    my_samps = np.concatenate((chosen_fpd_samps[0],chosen_lens_param_samps[0]),axis=1)
    truth_df_row = truth_df[truth_df['catalog_idx'] == chosen_cidx]
    if np.shape(fpd_samps)[-1] == 3:
        gt_for_debiasing = truth_df_row[['fpd01','fpd02','fpd03',
            'main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
            'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
            'main_deflector_parameters_e1','main_deflector_parameters_e2',
            'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
            'source_parameters_center_x','source_parameters_center_y']].to_numpy()[0]
    elif np.shape(fpd_samps)[-1] == 1:
        gt_for_debiasing = truth_df_row[['fpd01',
            'main_deflector_parameters_theta_E','main_deflector_parameters_gamma1',
            'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
            'main_deflector_parameters_e1','main_deflector_parameters_e2',
            'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
            'source_parameters_center_x','source_parameters_center_y']].to_numpy()[0]
        
    # starting point
    Cov = np.cov(my_samps,rowvar=False)
    # set Mu to be perfect!
    Mu = gt_for_debiasing

    # change precision of the Cov s.t. sigma(fpd01) is 2% (Williams)
    desired_fpd_std_dev = fermat_pot_prec * gt_for_debiasing[0]
    change_precision_factor = desired_fpd_std_dev / np.sqrt(Cov[0,0]) 
    Cov *= (change_precision_factor**2)

    print('theta_E standard dev: ', np.sqrt(Cov[-10,-10]))
    print('gamma_lens standard dev: ', np.sqrt(Cov[-7,-7]))

    # produce the de-biased samples
    num_gaussian_samps = np.shape(fpd_samps)[1]
    gaussianized_samps = multivariate_normal.rvs(mean=Mu,cov=Cov,
        size=num_gaussian_samps)
    
    # separate into fpds and lens params
    num_fpd = np.shape(fpd_samps)[-1]
    debiased_fpd_samps = gaussianized_samps[:,:num_fpd]
    debiased_lens_param_samps = gaussianized_samps[:,num_fpd:]

    return debiased_fpd_samps, debiased_lens_param_samps
        
################################################
# STEP 1: Produce fermat potentials + kinematics
################################################
with h5py.File(h5_posteriors_file, 'r') as h5:
    fpd_samps = h5['fpd_samps'][:]
    lens_param_samps = h5['lens_param_samps'][:]
    catalog_idxs = h5['catalog_idxs'][:]

R_sersic_truth = truth_df.loc[truth_df['catalog_idx']==720,
        'lens_light_parameters_R_sersic'].item()
n_sersic_truth = truth_df.loc[truth_df['catalog_idx']==720,
        'lens_light_parameters_n_sersic'].item()



for fp_prec in [0.01,0.02,0.03,0.05,0.10]:

    debiased_fpd, debiased_lp = debiased_samples(h5_posteriors_file,
        truth_metadata_file,chosen_cidx=720,fermat_pot_prec=fp_prec)


    ########################
    # Re-Compute Kinematics
    ########################
    print(debiased_fpd.shape)# debiased_lp
    beta_ani_samps = truncnorm.rvs(-0.5/0.2,0.5/0.2,loc=0.,scale=0.2,size=len(debiased_fpd))

    NIRSPEC_c_sqrtJ_samps = np.empty((len(debiased_fpd),10))

    # compute samples one by one
    for fp_idx in range(0,len(debiased_fpd)):

        # NIRSPEC
        nirspec_csqrtJ = galkin_utils.ground_truth_ifu_c_sqrtJ(
            galkin_utils.kinematicsAPI_NIRSPEC,
            theta_E=debiased_lp[fp_idx,0],
            gamma_lens=debiased_lp[fp_idx,3],
            R_sersic=R_sersic_truth,n_sersic=n_sersic_truth,
            beta_ani=beta_ani_samps[fp_idx])
        NIRSPEC_c_sqrtJ_samps[fp_idx] = nirspec_csqrtJ


    #######################
    # Gaussianizing Samples
    #######################

    # constants
    num_td = 1
    num_kin_bins = 10
    num_fpd_samps = 5000

    # gaussianize lens model posterior quantities
    to_gaussianize_input = []
    # fpds
    for i in range(0,num_td):
        to_gaussianize_input.append(debiased_fpd[:,i])
    # all lens params
    num_lp = np.shape(debiased_lp)[-1]
    for lp in range(0,num_lp):
        to_gaussianize_input.append(debiased_lp[:,lp])

    # kinematics
    # beta_ani
    to_gaussianize_input.append(beta_ani_samps)
    beta_idx = num_td+num_lp
    # sigma_v bins
    for j in range(0,num_kin_bins):
        to_gaussianize_input.append(NIRSPEC_c_sqrtJ_samps[:,j])

    to_gaussianize_input = np.asarray(to_gaussianize_input)
    # switch 1st dim to last dim (parameters dim)
    input_samps = np.transpose(to_gaussianize_input,axes=(1,0))
    # now gaussianize
    gaussian_samps = gaussianize_samples(input_samps,num_fpd_samps)

    # emulate kappa_ext
    kappa_ext_samps = norm.rvs(loc=0.,scale=kappa_ext_measurement_error,
            size=(1,num_fpd_samps))


    data_vector_dict_lens720 = {
        'catalog_idxs':np.asarray([720]),
        'fpd_samples':np.asarray([gaussian_samps[:,0:num_td]]),
        'lens_param_samples':np.asarray([gaussian_samps[:,num_td:(num_td+num_lp)]]),
        'kappa_ext_samples':kappa_ext_samps,
        'beta_ani_samples':np.asarray([gaussian_samps[:,beta_idx]]),
        'kin_pred_samples':np.asarray([gaussian_samps[:,-num_kin_bins:]])
    }

    # keep track of lens param modeling prior
    # save info to data_vector_dict for later use
    data_vector_dict_lens720['lens_params_nu_int_means'] = lens_params_nu_int_means
    data_vector_dict_lens720['lens_params_nu_int_stddevs'] = lens_params_nu_int_stddevs
    # construct multivariate normal
    log_prob_lens_params_nu_int = multivariate_normal(
        mean=lens_params_nu_int_means,
        cov=np.diag(lens_params_nu_int_stddevs**2)).logpdf
    # log prob condenses over lens params dimension
    data_vector_dict_lens720['log_prob_lens_param_samps_nu_int'] = np.empty(
        (data_vector_dict_lens720['lens_param_samples'].shape[0:2]))
    for i in range(0,1):
        data_vector_dict_lens720['log_prob_lens_param_samps_nu_int'][i] = log_prob_lens_params_nu_int(
            data_vector_dict_lens720['lens_param_samples'][i])

    # keep track of beta_ani prior
    # user-provided modeling prior
    data_vector_dict_lens720['log_prob_beta_ani_samps_nu_int'] = np.empty(
        (data_vector_dict_lens720['beta_ani_samples'].shape))
    for i in range(0,1):
        data_vector_dict_lens720['log_prob_beta_ani_samps_nu_int'][i] = BETA_ANI_PRIOR.logpdf(
            data_vector_dict_lens720['beta_ani_samples'][i])


    # TODO: save data_vector_dict to a .json (same as other way)
    # switch numpy arrays to lists for writing to .json
    for key in data_vector_dict_lens720.keys():
        if isinstance(data_vector_dict_lens720[key], np.ndarray):
            data_vector_dict_lens720[key] = data_vector_dict_lens720[key].tolist()

    # Write list of static data vectors to a JSON file
    static_dv_filepath = save_location+'dv_dict_lens720_fp_prec_%.2f.json'%(fp_prec)
    with open(static_dv_filepath, 'w') as file:
        json.dump(data_vector_dict_lens720, file, indent=4)

###################################################################
# STEP 2: Loop through and run inference toggling % precision
###################################################################

# ordering is: fpd, kin, time-delay
# baseline (3%, 3%, 3%)

# fpd test ([1%,3%,5%,10%], 3%, 3%)

# kin test (3%, [1%,3%,5%,10%], 3%)

# time-delay test (3%, 3%, [1%,3%,5%,10%])