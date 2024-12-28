# Configuration for silver sample training set

import numpy as np
from scipy.stats import norm, truncnorm
import paltas.Sampling.distributions as dist
from paltas.MainDeflector.simple_deflectors import PEMDShear
from paltas.Sources.sersic import SingleSersicSource
from paltas.PointSource.single_point_source import SinglePointSource

# calculated using .fits header
output_ab_zeropoint = 28.17
num_years = 5

kwargs_numerics = {'supersampling_factor':1}

# size of cutout
numpix = 33

# quads_only
doubles_quads_only = True
# point source magnification cut
#ps_magnification_cut = 2

psf_kernels = np.load('/home/users/sydney3/forecast/lsst_psf_library.npy')
# psf_kernels = np.load('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/MassModels/lsst_psf_library.npy')
def draw_psf_kernel():
	random_psf_index = np.random.randint(psf_kernels.shape[0])
	chosen_psf = psf_kernels[random_psf_index, :, :]
	chosen_psf[chosen_psf<0]=0
	return chosen_psf

config_dict = {
	'main_deflector':{
		'class': PEMDShear,
		'parameters':{
			'z_lens': truncnorm(-2.5,np.inf,loc=0.5,scale=0.2).rvs,
			'gamma': truncnorm(-(2./.2),np.inf,loc=2.0,scale=0.2).rvs,
            'theta_E': truncnorm(-2./4.,np.inf,loc=0.6,scale=0.4).rvs,
            'e1':norm(loc=0,scale=0.2).rvs,
            'e2':norm(loc=0,scale=0.2).rvs,
            # see cross_object below
			'center_x':None,
			'center_y':None,
			'gamma1':norm(loc=0,scale=0.06).rvs,
            'gamma2':norm(loc=0,scale=0.06).rvs,
			'ra_0':0.0,
			'dec_0':0.0,
		}
	},
    'source':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':truncnorm(-5,np.inf,loc=2.,scale=0.4).rvs,
            # range: 20 to 27, centered at 23.5
            'mag_app':truncnorm(-3./2.,3./2.,loc=23.5,scale=7./3.).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-(.5/.5),np.inf,loc=0.5,scale=0.5).rvs,
			'n_sersic':norm(loc=4.,scale=0.1).rvs,
			'e1':truncnorm(-3,3,loc=0,scale=0.12).rvs,
            'e2':truncnorm(-3,3,loc=0,scale=0.12).rvs,
            # see cross_object below
			'center_x':None,
			'center_y':None}

	},
    'lens_light':{
		'class': SingleSersicSource,
		'parameters':{
			'z_source':None,
            # range: 
            'mag_app':truncnorm(-5./2.,2./2.,loc=21.,scale=2.).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'R_sersic':truncnorm(-(.5/.6),np.inf,loc=0.5,scale=0.6).rvs,
			'n_sersic':norm(loc=4.,scale=0.1).rvs,
			'e1':truncnorm(-3,3,loc=0,scale=0.24).rvs,
            'e2':truncnorm(-3,3,loc=0,scale=0.24).rvs,
            # see cross_object below
			'center_x':None,
			'center_y':None}
	},
    'point_source':{
		'class': SinglePointSource,
		'parameters':{
            # see cross_object below for z,x,y
            'z_point_source':None,
			'x_point_source':None,
			'y_point_source':None,
            # range: 19 to 25
            'mag_app':truncnorm(-3./2.,3./2.,loc=22.,scale=2.).rvs,
			'output_ab_zeropoint':output_ab_zeropoint,
			'mag_pert': dist.MultipleValues(dist=truncnorm(-1/0.3,np.inf,1,0.3).rvs,num=10),
            'compute_time_delays':False
		}
	},
    'cosmology':{
		'parameters':{
			'cosmology_name': 'planck18'
		}
	},
    'psf':{
		'parameters':{
			'psf_type':'PIXEL',
			'kernel_point_source':draw_psf_kernel,
			'point_source_supersampling_factor':1
		}
	},
	'detector':{
		'parameters':{
			'pixel_scale':0.2,'ccd_gain':2.3,'read_noise':9,
			'magnitude_zero_point':output_ab_zeropoint,
			'exposure_time':15,'sky_brightness':20.46,
			'num_exposures':30*num_years,'background_noise':None
		}
	},
    'cross_object':{
		'parameters':{
            ('main_deflector:center_x,lens_light:center_x'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.06).rvs,scatter=0.005),
            ('main_deflector:center_y,lens_light:center_y'):dist.DuplicateScatter(
                dist=norm(loc=0,scale=0.06).rvs,scatter=0.005),
            ('source:center_x,source:center_y,point_source:x_point_source,'+
                'point_source:y_point_source'):dist.DuplicateXY(
                x_dist=norm(loc=0.0,scale=0.4).rvs,
                y_dist=norm(loc=0.0,scale=0.4).rvs),
			('main_deflector:z_lens,lens_light:z_source,source:z_source,'+
                 'point_source:z_point_source'):dist.RedshiftsPointSource(
				z_lens_min=0,z_lens_mean=0.5,z_lens_std=0.2,
				z_source_min=0,z_source_mean=2,z_source_std=0.4)
		}
	}
}
    
