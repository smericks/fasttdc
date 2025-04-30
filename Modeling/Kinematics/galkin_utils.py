import argparse
import numpy as np
import h5py
from scipy.stats import truncnorm
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import sys
sys.path.insert(0, '/scratch/users/sydney3/forecast/darkenergy-from-LAGN/')
import Utils.tdc_utils as tdc_utils

##################
# 4MOST Kinematics
##################
R_APERTURE = 0.725
PSF_FWHM = 0.5

def setup_4MOST_kinematics():

    kwargs_aperture = {
        'aperture_type': 'shell', 
        'r_in': 0., 
        'r_out': R_APERTURE,
        'center_ra': 0, 'center_dec': 0}

    kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': PSF_FWHM}

    kwargs_numerics_galkin = { 
        'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
        'log_integration': True,  # log or linear interpolation of surface brightness and mass models
        'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals

    kwargs_model = {
        'lens_model_list':['SPP'],
        'lens_light_model_list':['SERSIC']
    }

    anisotropy_model = 'const'

    kinematicsAPI = KinematicsAPI(0.5, 2., kwargs_model, 
        kwargs_aperture, kwargs_seeing, anisotropy_model, 
        kwargs_numerics_galkin=kwargs_numerics_galkin, 
        lens_model_kinematics_bool=[True, False],
        sampling_number=5000,MGE_light=True)
    
    return kinematicsAPI

kinematicsAPI_4MOST = setup_4MOST_kinematics()

##################
# MUSE Kinematics
##################
MUSE_R_BINS = [0.,0.5,1.,1.5] #arcsec
MUSE_PSF_FWHM = 0.5

def setup_MUSE_kinematics():

    kwargs_aperture = {
        'aperture_type': 'IFU_shells', 
        'r_bins': MUSE_R_BINS,
        'center_ra': 0, 'center_dec': 0}

    kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': MUSE_PSF_FWHM}

    kwargs_numerics_galkin = { 
        'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
        'log_integration': True,  # log or linear interpolation of surface brightness and mass models
        'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals

    kwargs_model = {
        'lens_model_list':['SPP'],
        'lens_light_model_list':['SERSIC']
    }

    anisotropy_model = 'const'

    kinematicsAPI = KinematicsAPI(0.5, 2., kwargs_model, 
        kwargs_aperture, kwargs_seeing, anisotropy_model, 
        kwargs_numerics_galkin=kwargs_numerics_galkin, 
        lens_model_kinematics_bool=[True, False],
        sampling_number=5000,MGE_light=True)
    
    return kinematicsAPI

kinematicsAPI_MUSE = setup_MUSE_kinematics()


#########################
# JWST NIRSPEC Kinematics
#########################
JWSTNIRSPEC_R_BINS = [0.,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0] #arcsec
JWSTNIRSPEC_PSF_FWHM = 0.05

# calcium III at redshift 0.5, take wavelength in angstrom, compute diffraction limit with D=6.5M
# 0.05 arcsec
# also see this paper that says 0.08 arcsec https://www.aanda.org/articles/aa/pdf/2022/05/aa42663-21.pdf
def setup_NIRSPEC_kinematics():

    kwargs_aperture = {
        'aperture_type': 'IFU_shells', 
        'r_bins': JWSTNIRSPEC_R_BINS,
        'center_ra': 0, 'center_dec': 0}

    kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': JWSTNIRSPEC_PSF_FWHM}

    kwargs_numerics_galkin = { 
        'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
        'log_integration': True,  # log or linear interpolation of surface brightness and mass models
        'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals

    kwargs_model = {
        'lens_model_list':['SPP'],
        'lens_light_model_list':['SERSIC']
    }

    anisotropy_model = 'const'

    kinematicsAPI = KinematicsAPI(0.5, 2., kwargs_model, 
        kwargs_aperture, kwargs_seeing, anisotropy_model, 
        kwargs_numerics_galkin=kwargs_numerics_galkin, 
        lens_model_kinematics_bool=[True, False],
        sampling_number=5000,MGE_light=True)
    
    return kinematicsAPI

kinematicsAPI_NIRSPEC = setup_NIRSPEC_kinematics()


#####################
# 4MOST APERTURE KIN
#####################

def ground_truth_veldisp(theta_E,gamma_lens,R_sersic,n_sersic,
    z_lens,z_src,gt_cosmo_astropy,beta_ani=0.):

    default_distance_ratio = np.sqrt(
        kinematicsAPI_4MOST._kwargs_cosmo['d_s'] / 
        kinematicsAPI_4MOST._kwargs_cosmo['d_ds'])

    kwargs_anisotropy = {'beta': beta_ani}

    kwargs_lens = [{
        'theta_E':theta_E, 
        'gamma':gamma_lens, 
        "center_x":0., 
        "center_y":0.
    }]

    kwargs_lens_light = [{
        'amp': 10.,
        'R_sersic': R_sersic,
        'n_sersic': n_sersic,
        'center_x': 0.,
        'center_y': 0.,
    }]

    vel_disp_numerical = kinematicsAPI_4MOST.velocity_dispersion(kwargs_lens, 
        kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)
    
    corrected_distance_ratio = np.sqrt(
        tdc_utils.kin_distance_ratio(gt_cosmo_astropy,z_lens,z_src))

    vel_disp_numerical *= (corrected_distance_ratio/default_distance_ratio)

    return vel_disp_numerical


def ground_truth_c_sqrtJ(theta_E,gamma_lens,R_sersic,n_sersic,beta_ani):

    default_distance_ratio = np.sqrt(
        kinematicsAPI_4MOST._kwargs_cosmo['d_s'] / 
        kinematicsAPI_4MOST._kwargs_cosmo['d_ds'])

    kwargs_anisotropy = {'beta': beta_ani}

    kwargs_lens = [{
        'theta_E':theta_E, 
        'gamma':gamma_lens, 
        "center_x":0., 
        "center_y":0.
    }]

    kwargs_lens_light = [{
        'amp': 10.,
        'R_sersic': R_sersic,
        'n_sersic': n_sersic,
        'center_x': 0.,
        'center_y': 0.,
    }]

    vel_disp_numerical = kinematicsAPI_4MOST.velocity_dispersion(kwargs_lens, 
        kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)

    c_sqrtJ = vel_disp_numerical / default_distance_ratio

    return c_sqrtJ



#######################
# IFU KIN (muse/nirspec)
#######################

def ground_truth_ifu_vdisp(ifu_api,theta_E,gamma_lens,R_sersic,n_sersic,
    z_lens,z_src,gt_cosmo_astropy,beta_ani=0.):
    """
    Args:
        ifu_api (kinematicsAPI_MUSE or kinematicsAPI_NIRSPEC)
    Returns:
        [v_disp] size=(3,) for muse
        [v_disp] size=(10,) for nirspec
    """

    default_distance_ratio = np.sqrt(
            ifu_api._kwargs_cosmo['d_s'] / 
            ifu_api._kwargs_cosmo['d_ds'])

    kwargs_anisotropy = {'beta': beta_ani}

    kwargs_lens = [{
        'theta_E':theta_E, 
        'gamma':gamma_lens, 
        "center_x":0., 
        "center_y":0.
    }]

    kwargs_lens_light = [{
        'amp': 10.,
        'R_sersic': R_sersic,
        'n_sersic': n_sersic,
        'center_x': 0.,
        'center_y': 0.,
    }]

    vel_disp_numerical = ifu_api.velocity_dispersion_map(kwargs_lens, 
        kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)

    corrected_distance_ratio = np.sqrt(
        tdc_utils.kin_distance_ratio(gt_cosmo_astropy,z_lens,z_src))

    vel_disp_numerical *= (corrected_distance_ratio/default_distance_ratio)

    return vel_disp_numerical


def ground_truth_ifu_c_sqrtJ(ifu_api,theta_E,gamma_lens,R_sersic,n_sersic,
    beta_ani):
    """
    Args:
        ifu_api (kinematicsAPI_MUSE or kinematicsAPI_NIRSPEC)
    Returns:
        [c_sqrtJ] size=(3,) for muse
        [c_sqrtJ] size=(10,) for nirspec
    """

    default_distance_ratio = np.sqrt(
        ifu_api._kwargs_cosmo['d_s'] / 
        ifu_api._kwargs_cosmo['d_ds'])

    kwargs_anisotropy = {'beta': beta_ani}

    kwargs_lens = [{
        'theta_E':theta_E, 
        'gamma':gamma_lens, 
        "center_x":0., 
        "center_y":0.
    }]

    kwargs_lens_light = [{
        'amp': 10.,
        'R_sersic': R_sersic,
        'n_sersic': n_sersic,
        'center_x': 0.,
        'center_y': 0.,
    }]

    vel_disp_numerical = ifu_api.velocity_dispersion_map(kwargs_lens, 
        kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)

    c_sqrtJ = vel_disp_numerical / default_distance_ratio

    return c_sqrtJ