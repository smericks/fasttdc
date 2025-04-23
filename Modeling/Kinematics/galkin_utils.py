import argparse
import numpy as np
import h5py
from scipy.stats import truncnorm
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import Utils.tdc_utils as tdc_utils

# hard-coded assumptions for 4MOST
R_APERTURE = 0.725
PSF_FWHM = 0.5

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


def ground_truth_veldisp(theta_E,gamma_lens,R_sersic,n_sersic,
    z_lens,z_src,gt_cosmo_astropy):

    default_distance_ratio = np.sqrt(
        kinematicsAPI._kwargs_cosmo['d_s'] / 
        kinematicsAPI._kwargs_cosmo['d_ds'])

    kwargs_anisotropy = {'beta': 0.}

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

    vel_disp_numerical = kinematicsAPI.velocity_dispersion(kwargs_lens, 
        kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)
    
    corrected_distance_ratio = np.sqrt(
        tdc_utils.kin_distance_ratio(gt_cosmo_astropy,z_lens,z_src))

    vel_disp_numerical *= (corrected_distance_ratio/default_distance_ratio)

    return vel_disp_numerical


def ground_truth_c_sqrtJ(theta_E,gamma_lens,R_sersic,n_sersic,beta_ani):

    default_distance_ratio = np.sqrt(
        kinematicsAPI._kwargs_cosmo['d_s'] / 
        kinematicsAPI._kwargs_cosmo['d_ds'])

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

    vel_disp_numerical = kinematicsAPI.velocity_dispersion(kwargs_lens, 
        kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)

    c_sqrtJ = vel_disp_numerical / default_distance_ratio

    return c_sqrtJ