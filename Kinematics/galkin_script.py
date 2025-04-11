import argparse
import numpy as np
import h5py
from scipy.stats import truncnorm
from lenstronomy.Analysis.kinematics_api import KinematicsAPI

R_APERTURE = 0.725 #arcseconds
PSF_FWHM = 0.5 #arcseconds
#H5_PATH = ('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/'+
#            'DataVectors/src_mag_cut_silver_debiased/gold_quads.h5')
H5_PATH = ('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/'+
            'DataVectors/gold_quads.h5')

BETA_PRIOR_MU = 0.
BETA_PRIOR_SIGMA = 0.1

# circular aperture
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
    sampling_number=5000,MGE_light=True)  # numerical ray-shooting, should converge -> infinity)


def v_disp_galkin(beta_ani,theta_E,gamma_lens,R_sersic,n_sersic):
    """
    Args:
        beta_ani (float): constant anisotropy value
        inputs_dict_row (pandas df row): 
    """

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

    return vel_disp_numerical # in km/s

def main(args):

    lens_idx = args.lens_idx
    debug = args.debug

    if lens_idx is None:
        raise ValueError("Must specify --lens_idx")

    input_keys = ['lens_param_samps','z_lens_truth','z_src_truth',
                'lens_light_parameters_R_sersic_truth', 
                'lens_light_parameters_n_sersic_truth']
    samples_dict = {}

    h5f = h5py.File(H5_PATH, 'r')
    for key in input_keys:
        # Some of these are single values, others are arrays, be careful!
        samples_dict[key] = h5f[key][lens_idx]
    h5f.close()

    beta_ani_samps = truncnorm.rvs(-np.inf,1/.1,loc=0.,scale=0.1,size=5000)

    distance_scaling_factor = (
        kinematicsAPI._kwargs_cosmo['d_s'] / kinematicsAPI._kwargs_cosmo['d_ds'])
    
    vel_disp_samples = np.empty(5000)
    for i,lens_params in enumerate(samples_dict['lens_param_samps']):
        vel_disp_samples[i] = v_disp_galkin(beta_ani=beta_ani_samps[i],
            theta_E=lens_params[0],gamma_lens=lens_params[3],
            R_sersic=samples_dict['lens_light_parameters_R_sersic_truth'],
            n_sersic=samples_dict['lens_light_parameters_n_sersic_truth'])
        
        if debug and i == 10:
            print(vel_disp_samples[:10])
            break

    # NOTE: dividing out distance ratio, need to multiply back in later...
    vel_disp_samples = vel_disp_samples / distance_scaling_factor
    np.save('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/DataVectors/lens%03d_vdisp.npy'%(lens_idx),
        vel_disp_samples)
    np.save('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/DataVectors/lens%03d_beta_ani.npy'%(lens_idx),
        beta_ani_samps)    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lens_idx",
        type=int,
        default=None,
        help=("Lens index in gold_quads.h5 (0->49)")
    )

    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help=("If true, only compute 10 samples")
    )

    args = parser.parse_args()
    main(args)
