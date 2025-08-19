# This helper code finishes the emulation and prepares data vectors for input to tdc_sampler
import numpy as np
import pandas as pd
import h5py
from scipy.stats import norm, multivariate_normal, uniform
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import tdc_sampler


# NOTE: hard-coded modeling priors (can change later)
GAMMA_LENS_PRIOR = norm(loc=2.,scale=0.2).logpdf
BETA_ANI_PRIOR = norm(loc=0.,scale=0.2).logpdf
COSMO_MODEL = 'LCDM_lambda_int_beta_ani'


def retrieve_truth_fpd(metadata_df,num_td):
    if num_td == 1: 
        fpd_truth = metadata_df.loc[:,['fpd01']].to_numpy()
    elif num_td == 3:
        fpd_truth = metadata_df.loc[:,['fpd01','fpd02','fpd03']].to_numpy()

    return fpd_truth

def retrieve_truth_lp(metadata_df):

    return metadata_df.loc[:,['main_deflector_parameters_theta_E',
    'main_deflector_parameters_gamma1',
    'main_deflector_parameters_gamma2','main_deflector_parameters_gamma',
    'main_deflector_parameters_e1','main_deflector_parameters_e2',
    'main_deflector_parameters_center_x','main_deflector_parameters_center_y',
    'source_parameters_center_x','source_parameters_center_y']].to_numpy()


def retrieve_truth_td(metadata_df,num_td):
    if num_td == 1: 
        # NOTE: will this keep the last dim=(..,..,1) ? 
        td_truth = metadata_df.loc[:,['td01']].to_numpy()
    elif num_td == 3:
        td_truth = metadata_df.loc[:,['td01','td02','td03']].to_numpy()

    return td_truth

def retrieve_truth_kin(metadata_df,kinematic_type):
    """
    Args:
        metadata_df
        kinematic_type (string): Options are: '4MOST', 'MUSE', or 'NIRSPEC'
    """

    # 4MOST
    if kinematic_type == '4MOST':
        return metadata_df.loc[:,['sigma_v_4MOST_kmpersec']].to_numpy()

    # MUSE
    if kinematic_type == 'MUSE':
        muse_keys = ['sigma_v_MUSE_bin'+str(j)+'_kmpersec' for j in range(0,3)]
        return metadata_df.loc[:,muse_keys].to_numpy()

    # NIRSPEC
    if kinematic_type == 'NIRSPEC':
        nirspec_keys = ['sigma_v_NIRSPEC_bin'+str(j)+'_kmpersec' for j in range(0,10)]
        return metadata_df.loc[:,nirspec_keys].to_numpy()


    raise ValueError("kinematic_type not supported")


def gaussianize(input_samps,gt_for_debiasing=None):
    """takes in input samples, fits a Gaussian, and returns Mu,Cov of that 
        Gaussian
    Args:
        input_samps (n_input_samps,n_params)
        gt_for_debiasing (n_params): Default is None, no de-biasing. We must 
            have access to ground truth to de-bias.
    Returns:
        Mu,Cov
    """
    Mu = np.mean(input_samps,axis=0)
    Cov = np.cov(input_samps,rowvar=False)

    # de-bias if requested
    if gt_for_debiasing is not None:
        Mu = multivariate_normal.rvs(mean=gt_for_debiasing,cov=Cov)

    return Mu,Cov


def gaussianize_samples(input_samps,num_gaussian_samps=5000,
    gt_for_debiasing=None):
    """takes in input samples, fits a Gaussian, and returns new samples
        from that Gaussian
    Args:
        input_samps (n_input_samps,n_params)
        gt_for_debiasing (n_params): Default is None, no de-biasing. We must 
            have access to ground truth to de-bias.
    Returns:
        output_samps (n_gaussian_samps,n_params)
    """

    Mu,Cov = gaussianize(input_samps,gt_for_debiasing)

    gaussianized_samps = multivariate_normal.rvs(mean=Mu,cov=Cov,
        size=num_gaussian_samps)

    return gaussianized_samps

def gaussianize_scale_and_debias(input_samps,
    desired_param_prec,desired_param_idx,gt_for_debiasing,
    num_gaussian_samps=5000):
    """
    Args:
        input_samps (n_input_samps,n_params)
        desired_param_prec: fractional percent error on a chosen parameter
        desired_param_idx: index of this parameter in input_samps and 
            gt_for_debiasing
        gt_for_debiasing: must have access to ground truth to de-bias
        num_gaussian_samps (int)
    """


    Mu = np.mean(input_samps,axis=0)
    Cov = np.cov(input_samps,rowvar=False)

    # re-scale based on desired precision of a chosen parameter
    current_std = np.sqrt(Cov[desired_param_idx,desired_param_idx])
    desired_std_gamma = gt_for_debiasing[desired_param_idx]*desired_param_prec
    Cov *= (desired_std_gamma/current_std)**2

    # debiasing is required to avoid over/under confident posteriors
    Mu = multivariate_normal.rvs(mean=gt_for_debiasing,cov=Cov)

    # now, take new samples!
    gaussianized_samps = multivariate_normal.rvs(mean=Mu,cov=Cov,
        size=num_gaussian_samps)

    return gaussianized_samps


#############################
# Construct likelihood object
#############################

def create_static_data_vectors(
    posteriors_h5_file,metadata_file,catalog_idxs,
    cosmo_model,
    td_meas_error_percent=None,td_meas_error_days=None,
    kappa_ext_meas_error_value=0.05,
    kinematic_type=None,
    kin_meas_error_percent=None,kin_meas_error_kmpersec=None,
    num_gaussianized_samps=None,
    lens_params_nu_int_means=None,
    lens_params_nu_int_stddevs=None,
    log_prob_beta_ani_nu_int=None,
    debias_models=False):
    """
    Args:
        posteriors_h5_file ()
        metadata_file ()
        catalog_idxs (np.array[int]): catalog indices of the subset of lenses 
            used from these files
        cosmo_model (str): Options are 'LCDM', etc...
        kinematic_type (string): Default=None. Options are: '4MOST', 'MUSE', or 'NIRSPEC'
        num_gaussianized_samps (int): Default=None (use samples as is). If specified,
            a Gaussian will be fit to the provided samples, and a new batch of 
            |num_gaussianized_samps| samples will be drawn from that distribution 
        lens_params_nu_int_means ([float]): Means of the training prior, default=None.
            Must have dim (n_lens_params)
        lens_params_nu_int_stddevs ([float]): Std. Devs. of the training prior,default=None
            Must have dim (n_lens_params)
        log_prob_beta_ani_nu_int (callable): default=None
    
    Returns: 
        (dict) data_vector_dict = 
            {   'td_measured':,
                'td_likelihood_prec':,
                'sigma_v_measured':,
                'sigma_v_likelihood_prec':,
                'fpd_samples':,
                'gamma_pred_samples':,
                'beta_ani_samples':,
                'kin_pred_samples':,
                'kappa_ext_samples':,
                'z_lens':,
                'z_src':,
            }
        
    """

    if debias_models and kinematic_type is not None:
        raise ValueError('de-biasing only works in TD-only case (for now)')

    # load in from posteriors file
    with h5py.File(posteriors_h5_file, "r") as h5:

        # set-up indexing
        h5_catalog_idxs = h5['catalog_idxs'][:]
        my_idxs = np.isin(h5_catalog_idxs,catalog_idxs)

        fpd_samps = h5['fpd_samps'][my_idxs]
        lens_param_samps = h5['lens_param_samps'][my_idxs]
        beta_ani_samps = h5['beta_ani_samps'][my_idxs]
        h5_catalog_idxs = h5['catalog_idxs'][my_idxs]

        # pull c_sqrtJ_samps based on kinematic type
        if kinematic_type is not None:
            if kinematic_type == '4MOST':
                c_sqrtJ_samps = h5['c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'MUSE':
                c_sqrtJ_samps = h5['MUSE_c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'NIRSPEC':
                c_sqrtJ_samps = h5['NIRSPEC_c_sqrtJ_samps'][my_idxs]
            else:
                raise ValueError("kinematic_type not supported")
            
            num_kin_bins = np.shape(c_sqrtJ_samps)[-1]
            
    # set up some sizes
    num_lenses = np.shape(fpd_samps)[0]
    num_td = np.shape(fpd_samps)[-1]
    num_lp = np.shape(lens_param_samps)[-1]
            
    # load in from metadata file
    all_metadata_df = pd.read_csv(metadata_file)
    # set up indexing
    metadata_catalog_idxs = all_metadata_df.loc[:,'catalog_idx']
    metadata_idx = np.isin(metadata_catalog_idxs,catalog_idxs)
    metadata_df = all_metadata_df.loc[metadata_idx]

    # emulate time-delay measurement
    td_truth = retrieve_truth_td(metadata_df, num_td)
    td_meas, td_meas_prec = emulate_measurements(td_truth, 
        td_meas_error_percent,td_meas_error_days)
    
    # emulate kappa_ext
    if num_gaussianized_samps is not None:
        num_fpd_samps = num_gaussianized_samps
    else:
        num_fpd_samps = np.shape[fpd_samps][1]

    kappa_ext_samps = norm.rvs(loc=0.,
        scale=kappa_ext_meas_error_value,
        size=(num_lenses,num_fpd_samps))
    
    # emulate kinematics
    if kinematic_type is not None:
        kin_truth = retrieve_truth_kin(metadata_df,kinematic_type)
        sigma_v_meas,sigma_v_meas_prec = emulate_measurements(kin_truth,
            kin_meas_error_percent,kin_meas_error_kmpersec)

    # gaussianize samples if requested
    if num_gaussianized_samps is not None:
        to_gaussianize_input = []
        # fpds
        fpd_truth = retrieve_truth_fpd(metadata_df,num_td)
        for i in range(0,num_td):
            to_gaussianize_input.append(fpd_samps[:,:,i])
        # TODO: all lens params
        lp_truth = retrieve_truth_lp(metadata_df)
        for lp in range(0,num_lp):
            to_gaussianize_input.append(lens_param_samps[:,:,lp])

        # kinematics
        if kinematic_type is not None:
            # beta_ani
            to_gaussianize_input.append(beta_ani_samps)
            beta_idx = num_td+num_lp
            # sigma_v bins
            for j in range(0,num_kin_bins):
                to_gaussianize_input.append(c_sqrtJ_samps[:,:,j])

        to_gaussianize_input = np.asarray(to_gaussianize_input)
        # switch 1st dim to last dim (parameters dim)
        input_samps = np.transpose(to_gaussianize_input,axes=(1,2,0))
        # now gaussianize
        gaussian_samps = np.empty((num_lenses,
            num_gaussianized_samps,np.shape(input_samps)[-1]))
        for l_idx in range(0,num_lenses):
            tdonly_truth = None
            if debias_models:
                tdonly_truth = np.concatenate((fpd_truth[l_idx],lp_truth[l_idx]))
            gaussian_samps[l_idx] = gaussianize_samples(
                input_samps[l_idx],num_gaussianized_samps,
                gt_for_debiasing=tdonly_truth)


    # TODO: get this into format for likelihood ... (add repeated axes, etc.)
    data_vector_dict ={}

    # catalog idxs
    data_vector_dict['catalog_idxs'] = np.asarray(catalog_idxs)

    # redshifts
    data_vector_dict['z_lens'] = metadata_df.loc[:,'main_deflector_parameters_z_lens'].to_numpy()
    data_vector_dict['z_src'] = metadata_df.loc[:,'source_parameters_z_source'].to_numpy()

    # samples already with fpd dimension
    if num_gaussianized_samps is not None:
        data_vector_dict['fpd_samples'] = gaussian_samps[:,:,0:num_td]
        data_vector_dict['lens_param_samples'] = gaussian_samps[:,:,num_td:(num_td+num_lp)]
        data_vector_dict['kappa_ext_samples'] = kappa_ext_samps

        if kinematic_type is not None:
            data_vector_dict['beta_ani_samples'] = gaussian_samps[:,:,beta_idx]
            data_vector_dict['kin_pred_samples'] = gaussian_samps[:,:,-num_kin_bins:]
    else:
        raise ValueError("not implemented")
    
    # time-delays
    # pad with a 2nd batch dim for # of fpd samples
    data_vector_dict['td_measured'] = np.repeat(td_meas[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    data_vector_dict['td_likelihood_prec'] = np.repeat(td_meas_prec[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    data_vector_dict['td_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(data_vector_dict['td_likelihood_prec']))) )

    # save info to data_vector_dict for later use
    data_vector_dict['lens_params_nu_int_means'] = lens_params_nu_int_means
    data_vector_dict['lens_params_nu_int_stddevs'] = lens_params_nu_int_stddevs

    # setting these to None = a uniform modeling prior
    if lens_params_nu_int_means is None and lens_params_nu_int_stddevs is None:
        # set log prob. to zero, equivalent to a uniform prior...
        data_vector_dict['log_prob_lens_param_samps_nu_int'] = np.zeros(
            (data_vector_dict['lens_param_samples'].shape[0:2]))
        
    # diagonal Gaussian modeling prior...
    else:
        # construct multivariate normal
        log_prob_lens_params_nu_int = multivariate_normal(
            mean=lens_params_nu_int_means,
            cov=np.diag(lens_params_nu_int_stddevs**2)).logpdf
        # log prob condenses over lens params dimension
        data_vector_dict['log_prob_lens_param_samps_nu_int'] = np.empty(
            (data_vector_dict['lens_param_samples'].shape[0:2]))
        for i in range(0,num_lenses):
            # TODO: will this work with a multivariate log_prob function? this still doesn't have 
            # a condensed sample dimension?
            data_vector_dict['log_prob_lens_param_samps_nu_int'][i] = log_prob_lens_params_nu_int(
                data_vector_dict['lens_param_samples'][i])
            

    # kinematics if requested
    if kinematic_type is not None:

        # measured sigma_v
        # pad with a 2nd batch dim for # of fpd samples
        data_vector_dict['sigma_v_measured'] = np.repeat(sigma_v_meas[:, np.newaxis, :],
                num_fpd_samps, axis=1)
        data_vector_dict['sigma_v_likelihood_prec'] = np.repeat(
            sigma_v_meas_prec[:, np.newaxis, :, :],
            num_fpd_samps, axis=1)
        
        data_vector_dict['sigma_v_likelihood_prefactors'] = np.log( (1/(2*np.pi)**(num_kin_bins/2)) / 
            np.sqrt(np.linalg.det(np.linalg.inv(data_vector_dict['sigma_v_likelihood_prec']))))
        
        # beta_ani
        if log_prob_beta_ani_nu_int is None:
            # default: assume un-informative prior
            data_vector_dict['log_prob_beta_ani_samps_nu_int'] = uniform.logpdf(
                data_vector_dict['beta_ani_samples'],loc=-0.5,scale=1.)
        else:
            # user-provided modeling prior
            data_vector_dict['log_prob_beta_ani_samps_nu_int'] = np.empty(
                (data_vector_dict['beta_ani_samples'].shape))
            for i in range(0,num_lenses):
                data_vector_dict['log_prob_beta_ani_samps_nu_int'][i] = log_prob_beta_ani_nu_int(
                    data_vector_dict['beta_ani_samples'][i])


    return data_vector_dict


def FOM_vs_z_static_DV(posteriors_h5_file,
    metadata_file,chosen_catalog_idx,
    cosmo_model,
    td_meas_error_percent=None,td_meas_error_days=None,
    kappa_ext_meas_error_value=0.05,
    kinematic_type=None,
    kin_meas_error_percent=None,kin_meas_error_kmpersec=None,
    num_gaussianized_samps=None,
    lens_params_nu_int_means=None,
    lens_params_nu_int_stddevs=None,
    log_prob_beta_ani_nu_int=None,
    debias_models=False):
    """
    Can keep fpd, c*sqrt(J) the same, must change measured time-delay, 
        measured velocity dispersion, etc.
    
    """

    if debias_models and kinematic_type is not None:
        raise ValueError('de-biasing only works in TD-only case (for now)')

    # pull relevant quantities for a SINGLE lens
    # load in from posteriors file
    with h5py.File(posteriors_h5_file, "r") as h5:

        # find the right index
        h5_catalog_idxs = h5['catalog_idxs'][:]
        my_idx = np.isin(h5_catalog_idxs,chosen_catalog_idx)

        fpd_samps = h5['fpd_samps'][my_idx]
        lens_param_samps = h5['lens_param_samps'][my_idx]
        beta_ani_samps = h5['beta_ani_samps'][my_idx]
        h5_catalog_idxs = h5['catalog_idxs'][my_idx]

        # pull c_sqrtJ_samps based on kinematic type
        if kinematic_type is not None:
            if kinematic_type == '4MOST':
                c_sqrtJ_samps = h5['c_sqrtJ_samps'][my_idx]
            elif kinematic_type == 'MUSE':
                c_sqrtJ_samps = h5['MUSE_c_sqrtJ_samps'][my_idx]
            elif kinematic_type == 'NIRSPEC':
                c_sqrtJ_samps = h5['NIRSPEC_c_sqrtJ_samps'][my_idx]
            else:
                raise ValueError("kinematic_type not supported")
            
            num_kin_bins = np.shape(c_sqrtJ_samps)[-1]

    # pull relevant metadata
    all_metadata_df = pd.read_csv(metadata_file)
    # set up indexing
    metadata_catalog_idxs = all_metadata_df.loc[:,'catalog_idx']
    metadata_idx = np.isin(metadata_catalog_idxs,chosen_catalog_idx)
    metadata_row = all_metadata_df.loc[metadata_idx]

    # loop through z_lens, z_src (make sure z_src > z_lens)
    for z_lens in np.arange(0.,1.5,0.1):
        for z_src in np.arange(0.5,3.0,0.1):
            if z_src < z_lens: 
                skip

            # re-compute TD truth
            # ALWAYS center on truth, keep 3% constraint

            # re-compute KIN truth (sigma_v_NIRSPEC_kmpersec)
            # ALWAYS center on truth, keep 5% constraint




#########################################
# Construct analytical static DVs
#########################################

def create_analytical_data_vectors(
    posteriors_h5_file,metadata_file,catalog_idxs,
    td_meas_error_percent=None,td_meas_error_days=None,
    kappa_ext_meas_error_value=0.05,
    num_kappa_ext_samps=None,
    kinematic_type=None,
    kin_meas_error_percent=None,kin_meas_error_kmpersec=None,
    lens_params_nu_int_means=None,
    lens_params_nu_int_stddevs=None,
    beta_ani_nu_int_mean=None,
    beta_ani_nu_int_stddev=None,
    fermat_pot_nu_int_mean=None,
    fermat_pot_nu_int_stddev=None,
    csqrtJ_nu_int_mean=None,
    csqrtJ_nu_int_stddev=None,
    debias_models=False):
    """ For analytical version of likelihood, need to save 2 Gaussians 
        for each lens (Mean,Prec) in order (fpd,csqrtJ,lens_params,beta_ani). 
        The image-based model (_model) and the interim prior (_int)
    Args:
        
    
    Returns: 
        (dict) data_vector_dict = 
            {   'catalog_idxs':,
                'z_src':,
                'z_lens':,
                'td_measured':,
                'td_likelihood_prec':,
                'sigma_v_measured':,
                'sigma_v_likelihood_prec':,
                'kappa_ext_samples',
                'mu_model':,
                'prec_model':,
                'mu_int':,
                'prec_int':
            }
        
    """

    if debias_models and kinematic_type is not None:
        raise ValueError('de-biasing only works in TD-only case (for now)')

    # load in from posteriors file
    with h5py.File(posteriors_h5_file, "r") as h5:

        # set-up indexing
        h5_catalog_idxs = h5['catalog_idxs'][:]
        my_idxs = np.isin(h5_catalog_idxs,catalog_idxs)

        fpd_samps = h5['fpd_samps'][my_idxs]
        lens_param_samps = h5['lens_param_samps'][my_idxs]
        beta_ani_samps = h5['beta_ani_samps'][my_idxs]
        h5_catalog_idxs = h5['catalog_idxs'][my_idxs]

        # pull c_sqrtJ_samps based on kinematic type
        if kinematic_type is not None:
            if kinematic_type == '4MOST':
                c_sqrtJ_samps = h5['c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'MUSE':
                c_sqrtJ_samps = h5['MUSE_c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'NIRSPEC':
                c_sqrtJ_samps = h5['NIRSPEC_c_sqrtJ_samps'][my_idxs]
            else:
                raise ValueError("kinematic_type not supported")
            
            num_kin_bins = np.shape(c_sqrtJ_samps)[-1]
            
    # set up some sizes
    num_lenses = np.shape(fpd_samps)[0]
    num_td = np.shape(fpd_samps)[-1]
    num_lp = np.shape(lens_param_samps)[-1]
            
    # load in from metadata file
    all_metadata_df = pd.read_csv(metadata_file)
    # set up indexing
    metadata_catalog_idxs = all_metadata_df.loc[:,'catalog_idx']
    metadata_idx = np.isin(metadata_catalog_idxs,catalog_idxs)
    metadata_df = all_metadata_df.loc[metadata_idx]

    # emulate time-delay measurement
    td_truth = retrieve_truth_td(metadata_df, num_td)
    td_meas, td_meas_prec = emulate_measurements(td_truth, 
        td_meas_error_percent,td_meas_error_days)
    
    # emulate kappa_ext
    kappa_ext_samps = norm.rvs(loc=0.,
        scale=kappa_ext_meas_error_value,
        size=(num_lenses,num_kappa_ext_samps))
    
    # emulate kinematics
    if kinematic_type is not None:
        kin_truth = retrieve_truth_kin(metadata_df,kinematic_type)
        sigma_v_meas,sigma_v_meas_prec = emulate_measurements(kin_truth,
            kin_meas_error_percent,kin_meas_error_kmpersec)

    # Gaussianize the image-based model
    # FOR ANALYTICAL: NEEDS TO BE IN ORDER:
    #   with kin: (fpd,csqrtJ,lens_params,beta_ani)
    #   without kin: (fpd,lens_params)
    to_gaussianize_input = []

    # Component 1: fpds
    fpd_truth = retrieve_truth_fpd(metadata_df,num_td)
    for i in range(0,num_td):
        to_gaussianize_input.append(fpd_samps[:,:,i])

    # Component 2: csqrtJ (if kin)
    if kinematic_type is not None:
        # sigma_v bins
        for j in range(0,num_kin_bins):
            to_gaussianize_input.append(c_sqrtJ_samps[:,:,j])

    # Component 3: lens_params
    lp_truth = retrieve_truth_lp(metadata_df)
    for lp in range(0,num_lp):
        to_gaussianize_input.append(lens_param_samps[:,:,lp])

    # Component 4: beta_ani (if kin)
    if kinematic_type is not None:
        to_gaussianize_input.append(beta_ani_samps)
    # kinematics

    # Now we actually do the Gaussian-izing here...
    to_gaussianize_input = np.asarray(to_gaussianize_input)
    # switch 1st dim to last dim (parameters dim)
    input_samps = np.transpose(to_gaussianize_input,axes=(1,2,0))
    num_params = np.shape(to_gaussianize_input)[-1]
    mu_model_list = np.empty((num_lenses,num_params))
    cov_model_list = np.empty((num_lenses,num_params,num_params))
    # now gaussianize
    for l_idx in range(0,num_lenses):
        mu_model,cov_model = gaussianize(input_samps[l_idx])
        mu_model_list[l_idx] = mu_model
        cov_model_list[l_idx] = cov_model

    # Now, fill in the dict of data vectors
    data_vector_dict ={}

    # catalog idxs
    data_vector_dict['catalog_idxs'] = np.asarray(catalog_idxs)

    # redshifts
    data_vector_dict['z_lens'] = metadata_df.loc[:,'main_deflector_parameters_z_lens'].to_numpy()
    data_vector_dict['z_src'] = metadata_df.loc[:,'source_parameters_z_source'].to_numpy()

    # save emulated measurements
    # time-delays
    data_vector_dict['td_measured'] = td_meas
    data_vector_dict['td_likelihood_prec'] = td_meas_prec
    # kappa_ext samples
    data_vector_dict['kappa_ext_samples'] = kappa_ext_samps
    # kinematics
    if kinematic_type is not None:
        data_vector_dict['sigma_v_measured'] = sigma_v_meas
        data_vector_dict['sigma_v_likelihood_prec'] = sigma_v_meas_prec

    # save mu_model, prec_model
    data_vector_dict['mu_model'] = mu_model_list
    # NOTE: might need to double check symmetry after this (was having issues 
    # with numpy numerical limitations before)
    data_vector_dict['prec_model'] = np.linalg.inv(cov_model_list)

    # construct & save mu_int,prec_int
    # in order (fpd,csqrtJ,lens_params,beta_ani)
    mu_int = np.asarray(fermat_pot_nu_int_mean)
    if kinematic_type is not None:
        mu_int = np.concatenate((mu_int,csqrtJ_nu_int_mean))
    mu_int = np.concatenate((mu_int,lens_params_nu_int_means))
    if kinematic_type is not None:
        mu_int = np.concatenate((mu_int,beta_ani_nu_int_mean))

    stddev_int = np.asarray(fermat_pot_nu_int_stddev)
    if kinematic_type is not None:
        stddev_int = np.concatenate((stddev_int,csqrtJ_nu_int_stddev))
    stddev_int = np.concatenate((stddev_int,lens_params_nu_int_stddevs))
    if kinematic_type is not None:
        stddev_int = np.concatenate((stddev_int,beta_ani_nu_int_stddev))
    # put 1/sigma^2 on the diagonal
    prec_int = np.diag(1/(stddev_int**2))

    # repeat mu_int, prec_int once for each lens
    mu_int = np.repeat(mu_int[np.newaxis,:],num_lenses,axis=0)
    prec_int = np.repeat(prec_int[np.newaxis,:,:],num_lenses,axis=0)

    data_vector_dict['mu_int'] = mu_int
    data_vector_dict['prec_int'] = prec_int

    return data_vector_dict


#############################
# Construct likelihood object
#############################

def construct_likelihood_obj(
    posteriors_h5_file,metadata_file,catalog_idxs,
    cosmo_model,
    td_meas_error_percent=None,td_meas_error_days=None,
    kappa_ext_meas_error_value=0.05,
    kinematic_type=None,
    kin_meas_error_percent=None,kin_meas_error_kmpersec=None,
    num_gaussianized_samps=None,
    gaussianize_scale_debias_kwargs=None):
    """
    Args:
        posteriors_h5_file ()
        metadata_file ()
        catalog_idxs (np.array[int]): catalog indices of the subset of lenses 
            used from these files
        cosmo_model (str): Options are 'LCDM', etc...
        kinematic_type (string): Default=None. Options are: '4MOST', 'MUSE', or 'NIRSPEC'
        num_gaussianized_samps (int): Default=None (use samples as is). If specified,
            a Gaussian will be fit to the provided samples, and a new batch of 
            |num_gaussianized_samps| samples will be drawn from that distribution 
        gaussianize_scale_debias_kwargs (dict): Default = None (not used). If specified,
            will call gaussianize_scale_and_debias() to:
                1) Gaussianize
                2) Re-scale covariance matrix s.t. a parameter has a desired precision
                3) De-bias the mean of the prediction to avoid over/under confidence
            This allows us to make "higher fidelity" models from the NPE models
        
    """

    # load in from posteriors file
    with h5py.File(posteriors_h5_file, "r") as h5:

        # set-up indexing
        h5_catalog_idxs = h5['catalog_idxs'][:]
        my_idxs = np.isin(h5_catalog_idxs,catalog_idxs)

        fpd_samps = h5['fpd_samps'][my_idxs]
        lens_param_samps = h5['lens_param_samps'][my_idxs]
        beta_ani_samps = h5['beta_ani_samps'][my_idxs]
        h5_catalog_idxs = h5['catalog_idxs'][my_idxs]

        # pull c_sqrtJ_samps based on kinematic type
        if kinematic_type is not None:
            if kinematic_type == '4MOST':
                c_sqrtJ_samps = h5['c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'MUSE':
                c_sqrtJ_samps = h5['MUSE_c_sqrtJ_samps'][my_idxs]
            elif kinematic_type == 'NIRSPEC':
                c_sqrtJ_samps = h5['NIRSPEC_c_sqrtJ_samps'][my_idxs]
            else:
                raise ValueError("kinematic_type not supported")
            
            num_kin_bins = np.shape(c_sqrtJ_samps)[-1]
            
    # set up some sizes
    num_lenses = np.shape(fpd_samps)[0]
    num_td = np.shape(fpd_samps)[-1]
            
    # load in from metadata file
    all_metadata_df = pd.read_csv(metadata_file)
    # set up indexing
    metadata_catalog_idxs = all_metadata_df.loc[:,'catalog_idx']
    metadata_idx = np.isin(metadata_catalog_idxs,catalog_idxs)
    metadata_df = all_metadata_df.loc[metadata_idx]

    # emulate time-delay measurement
    td_truth = retrieve_truth_td(metadata_df, num_td)
    td_meas, td_meas_prec = emulate_measurements(td_truth, 
        td_meas_error_percent,td_meas_error_days)
    
    # TODO: emulate kappa_ext
    if num_gaussianized_samps is not None:
        kappa_ext_samps = norm.rvs(loc=0.,
            scale=kappa_ext_meas_error_value,
            size=(num_lenses,num_gaussianized_samps))
    else:
        kappa_ext_samps = norm.rvs(loc=0.,
            scale=kappa_ext_meas_error_value,
            size=(num_lenses,np.shape[fpd_samps][1]))
    
    # emulate kinematics
    if kinematic_type is not None:
        kin_truth = retrieve_truth_kin(metadata_df,kinematic_type)
        sigma_v_meas,sigma_v_meas_prec = emulate_measurements(kin_truth,
            kin_meas_error_percent,kin_meas_error_kmpersec)

    # gaussianize samples if requested
    if num_gaussianized_samps is not None:
        to_gaussianize_input = []
        gt_params = [] # (n_lenses,n_params)
        # fpds
        fpd_truth = retrieve_truth_fpd(metadata_df,num_td)
        for i in range(0,num_td):
            to_gaussianize_input.append(fpd_samps[:,:,i])
            gt_params.append(fpd_truth[:,i])
        # gamma_lens
        to_gaussianize_input.append(lens_param_samps[:,:,3])
        gamma_idx = num_td
        gamma_truth = metadata_df.loc[:,'main_deflector_parameters_gamma'].to_numpy()
        gt_params.append(gamma_truth)

        # kinematics
        if kinematic_type is not None:
            # beta_ani
            to_gaussianize_input.append(beta_ani_samps)
            beta_idx = num_td+1
            # NOTE: beta_truth has to be centered at zero (keep modeling prior)
            beta_truth = np.zeros(np.shape(gamma_truth))
            gt_params.append(beta_truth)
            # sigma_v bins
            for j in range(0,num_kin_bins):
                to_gaussianize_input.append(c_sqrtJ_samps[:,:,j])

        to_gaussianize_input = np.asarray(to_gaussianize_input)
        # switch 1st dim to last dim (parameters dim)
        input_samps = np.transpose(to_gaussianize_input,axes=(1,2,0))
        # now gaussianize
        gaussian_samps = np.empty((num_lenses,
            num_gaussianized_samps,np.shape(input_samps)[-1]))
        for l_idx in range(0,num_lenses):
            gaussian_samps[l_idx] = gaussianize_samples(
                input_samps[l_idx],num_gaussianized_samps)

    # deal with edge cases of 1 td, 1 kinematic bin
    # 1 td
    gaussian_fpd_samps = gaussian_samps[:,:,0:num_td]
    #if num_td == 1:
    #    gaussian_fpd_samps = gaussian_fpd_samps[:,:,np.newaxis]
    # 1 kin bin
    if kinematic_type is not None:
        gaussian_kin_samps = gaussian_samps[:,:,-num_kin_bins:]
    #    if num_kin_bins == 1:
    #        gaussian_kin_samps = gaussian_kin_samps[:,:,np.newaxis]

    
    if kinematic_type is not None:

        # TODO: kappa_ext!!
        likelihood_obj = tdc_sampler.TDCKinLikelihood(
            td_measured=td_meas,
            td_likelihood_prec=td_meas_prec,
            sigma_v_measured=sigma_v_meas,
            sigma_v_likelihood_prec=sigma_v_meas_prec,
            fpd_samples=gaussian_fpd_samps,
            gamma_pred_samples=gaussian_samps[:,:,gamma_idx],
            beta_ani_samples=gaussian_samps[:,:,beta_idx],
            kin_pred_samples=gaussian_kin_samps,
            kappa_ext_samples=kappa_ext_samps,
            z_lens=metadata_df.loc[:,'main_deflector_parameters_z_lens'].to_numpy(),
            z_src=metadata_df.loc[:,'source_parameters_z_source'].to_numpy(),
            cosmo_model=cosmo_model,
            log_prob_gamma_nu_int=GAMMA_LENS_PRIOR,
            log_prob_beta_ani_nu_int=BETA_ANI_PRIOR)

    else:

        likelihood_obj = tdc_sampler.TDCLikelihood(
            td_measured=td_meas,
            td_likelihood_prec=td_meas_prec,
            fpd_samples=gaussian_fpd_samps,
            gamma_pred_samples=gaussian_samps[:,:,gamma_idx],
            kappa_ext_samples=kappa_ext_samps,
            z_lens=metadata_df.loc[:,'main_deflector_parameters_z_lens'].to_numpy(),
            z_src=metadata_df.loc[:,'source_parameters_z_source'].to_numpy(),
            cosmo_model=cosmo_model,
            log_prob_gamma_nu_int=GAMMA_LENS_PRIOR)


    return likelihood_obj

def emulate_measurements(sigma_v_truth, 
        measurement_error_percent=None,measurement_error_kmpersec=None):
    """NOTE: this is written in terms of sigma_v, but this can be used for
        time-delays as well :) 

    Args:
        sigma_v_truth (n_lenses,n_kin_bins)
        measurement_error_percent (float): a fractional value
            (i.e. use 0.01 to represent 1% error)
        measurement_error_kmpersec
    Returns: 
        sigma_v_measured (n_lenses,n_kin_bins), 
        sigma_v_measurement_prec (n_lenses,n_kin_bins, n_kin_bins)
    """

    # must define either measurement_error_percent or measurement_error_kmpersec
    if measurement_error_percent is not None and measurement_error_kmpersec is not None:
        raise ValueError('Must specify kin. meas. error in either percent OR kmpersec (not both)')
    elif measurement_error_percent is None and measurement_error_kmpersec is None:
        raise ValueError('Must specify kin. meas. error in either percent OR kmpersec')
    
    # grab number of kin bins
    num_kin_bins = np.shape(sigma_v_truth)[-1]
    num_lenses = np.shape(sigma_v_truth)[0]

    # construct array of measurement errors
    # meas_sigma has shape (n_lenses,n_kin_bins)
    if measurement_error_percent is not None:
        meas_sigma = measurement_error_percent*np.abs(sigma_v_truth) # must be a positive number!!
    elif measurement_error_kmpersec is not None:
        meas_sigma = measurement_error_kmpersec*np.ones(np.shape(sigma_v_truth))  


    # construct covariance / precision matrices (NOTE: diagonal for now!)
    cov_sigma_v = np.repeat(np.eye(num_kin_bins)[np.newaxis,:,:],repeats=num_lenses,axis=0)
    for bin in range(0,num_kin_bins):
        cov_sigma_v[:,bin,bin] = meas_sigma[:,bin]**2
    prec_sigma_v = np.linalg.inv(cov_sigma_v)

    # emulate mean measurement off of the truth
    sigma_v_measured = np.empty(np.shape(sigma_v_truth))
    for lens_idx in range(0,num_lenses):
        sigma_v_measured[lens_idx] = multivariate_normal.rvs(
            mean=sigma_v_truth[lens_idx],cov=cov_sigma_v[lens_idx])

    # save to data vectors dict
    return sigma_v_measured, prec_sigma_v
