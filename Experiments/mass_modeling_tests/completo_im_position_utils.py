from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from astropy.cosmology import FlatLambdaCDM
from slsim.Deflectors import Deflector
from slsim.LOS.los_individual import LOSIndividual
from slsim.Sources.source import Source
from slsim.Lenses.lens import Lens
# NOTE: all hard-coded for EPL!!

def completo_param_samps_to_lenst_kwargs(lens_param_sample):
    """
    Args:
        lens_param_sample: [theta_E,gamma1,gamma2,gamma_lens,e1,e2,x_lens,y_lens,x_src,y_src]

    Returns:
        lens_model_list, ps_model_list, lens_model_kwargs, ps_model_kwargs
    """

    # Unpack parameters
    theta_E, gamma1, gamma2, gamma_lens, e1, e2, x_lens, y_lens, x_src, y_src = lens_param_sample
    
    # Define model lists
    lens_model_list = ['EPL', 'SHEAR']
    ps_model_list = ['SOURCE_POSITION']
    
    # EPL (Elliptical Power Law) lens model kwargs
    epl_kwargs = {
        'theta_E': theta_E,
        'gamma': gamma_lens,
        'e1': -e1, # NOTE: different convention!!!!!
        'e2': e2,
        'center_x': x_lens,
        'center_y': y_lens
    }
    
    # SHEAR model kwargs
    shear_kwargs = {
        'ra_0': 0.,
        'dec_0': 0.,
        'gamma1': -gamma1, # NOTE: different convention!!!!!
        'gamma2': gamma2
    }
    
    # Combine lens model kwargs
    lens_model_kwargs = [epl_kwargs, shear_kwargs]
    
    # Point source kwargs (source position)
    ps_model_kwargs = [{
        'ra_source': x_src,
        'dec_source': y_src,
        'point_amp':10. # NOTE: shouldn't affect image positions...
    }]
    
    return lens_model_list, ps_model_list, lens_model_kwargs, ps_model_kwargs

def complete_param_samps_to_slsim(lens_param_sample):
    """
    Args:
        lens_param_sample: [theta_E,gamma1,gamma2,gamma_lens,e1,e2,x_lens,y_lens,x_src,y_src]

    Returns:
        slsim.Lens object
    """

    
    # make a deflector object
    deflector_dict = {
        'theta_E':lens_param_sample[0],
        'gamma_pl':lens_param_sample[3],
        'e1_mass':lens_param_sample[4],
        'e2_mass':lens_param_sample[5],
        'center_x':lens_param_sample[6],
        'center_y':lens_param_sample[7],
        'e1_light':0., # NOTE: these don't matter for PS positions...
        'e2_light':0.,
        'mag_i':22.,
        'mag_F158':22.,
        'angular_size':1.,
        'n_sersic':4.
    }
    deflector_obj = Deflector(deflector_type='EPL_SERSIC',
        z=0.5,**deflector_dict) # NOTE: redshift doesn't matter, hardcode

    # make a LOS object for ext. shear
    gamma = [lens_param_sample[1],lens_param_sample[2]]
    los_obj = LOSIndividual(gamma=gamma)

    # make a source object
    source_dict = {
        'ps_mag_i':22., # NOTE: none matter for PS positions except for center_x,center_y
        'ps_mag_F158':22.,
        'e1':0.,
        'e2':0.,
        'center_x':lens_param_sample[8],
        'center_y':lens_param_sample[9],
        'mag_i':23.,
        'mag_F158':23.,
        'angular_size':1.,
        'n_sersic':1.
    }
    source_obj = Source(extended_source_type='single_sersic',
        point_source_type='quasar',
        z=1., # NOTE doesn't matter for PS positions
        **source_dict)

    # combine into a lens object
    # TODO: need to add microlensing kwargs here?
    groundtruth_cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # NOTE hardcoded
    slsim_lens_obj = Lens(source_class=source_obj,deflector_class=deflector_obj,
        los_class=los_obj,cosmo=groundtruth_cosmo)
    
    return slsim_lens_obj


def image_positions(lenst_lens_model_list,lenst_ps_model_list,
        lenst_lens_kwargs, lenst_ps_kwargs):
    """
    Returns:
        ([x_coords],[y_coords]): Two arrays containing ra/dec positions of
            point source images
    """

    lens_model = LensModel(lenst_lens_model_list)

    point_source_model = PointSource(
        lenst_ps_model_list,lens_model=lens_model,
        save_cache=True,fixed_magnification_list=[True]) # NOTE: decide if these flags are OK

    image_positions_ps = point_source_model.image_position(
        kwargs_ps=lenst_ps_kwargs,
        kwargs_lens=lenst_lens_kwargs)

    return [image_positions_ps[0][0],image_positions_ps[1][0]] 


def completo_image_positions(lens_param_sample):
    """
    Args:
        lens_param_sample: [theta_E,gamma1,gamma2,gamma_lens,e1,e2,x_lens,y_lens,x_src,y_src]

    Returns: 
        [x_im],[y_im]
    """

    slsim_lens = complete_param_samps_to_slsim(lens_param_sample)
    im_pos_list = slsim_lens.point_source_image_positions()
    
    return im_pos_list[0][0],im_pos_list[0][1]
