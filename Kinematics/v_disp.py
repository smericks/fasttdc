import SKiNN
import torch
import pickle
import numpy as np
import time
import h5py
assert(torch.cuda.is_available()==True) #must be True to function
from SKiNN.batched_generator import BatchedGenerator

from scipy.ndimage import gaussian_filter
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.pixel_grid import PixelGrid
from lenstronomy.Data.psf import PSF
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util.param_util import phi_q2_ellipticity, ellipticity2phi_q
from lenstronomy.Analysis.kinematics_api import KinematicsAPI


######################################
# Create SKiNN input from Data Vectors
######################################

def skinn_input(inputs_dict):

    # 5k samples of lens parameters for each lens shape=(n_lenses,5e3,10)
    lens_param_samps = inputs_dict['gold_quads']['lens_param_samps']
    num_imp_samps = np.shape(lens_param_samps)[1]
    #[q_mass, q_light, theta_E, n_sersic_light, R_sersic_light, r_core , gamma, b_ani, inclination]

    mass_e1 = lens_param_samps[:,:,1]
    mass_e2 = lens_param_samps[:,:,2]
    phi_mass,q_mass = ellipticity2phi_q(mass_e1,mass_e2)

    light_e1 = inputs_dict['gold_quads']['lens_light_parameters_e1_truth']
    light_e2 = inputs_dict['gold_quads']['lens_light_parameters_e2_truth']
    # repeat to add imp. samples dimension
    light_e1 = np.repeat(light_e1[:,None],axis=-1,repeats=num_imp_samps)
    light_e2 = np.repeat(light_e2[:,None],axis=-1,repeats=num_imp_samps)
    phi_light,q_light = ellipticity2phi_q(light_e1,light_e2)

    theta_E_lenst = lens_param_samps[:,:,0]
    gamma_lenst = lens_param_samps[:,:,3] 

    # switch to glee convention for theta_E and gamma
    gamma = (gamma_lenst - 1) / 2
    r_e_scale = (2 / (1 + q_mass)) ** (1 / (2 * gamma)) * np.sqrt(q_mass)
    theta_E = theta_E_lenst / r_e_scale

    n_sersic_light = inputs_dict['gold_quads']['lens_light_parameters_n_sersic_truth']
    R_sersic_light = inputs_dict['gold_quads']['lens_light_parameters_R_sersic_truth']
    # repeat to add imp. samples dimension
    n_sersic_light = np.repeat(n_sersic_light[:,None],axis=-1,repeats=num_imp_samps)
    R_sersic_light = np.repeat(R_sersic_light[:,None],axis=-1,repeats=num_imp_samps)

    r_core = np.ones(n_sersic_light.shape)*0.08

    # TODO: how to initialize these for now?
    b_ani = np.ones(np.shape(r_core))
    inclination = np.ones(np.shape(r_core))*85


    my_input = np.stack([q_mass,q_light,theta_E,
        n_sersic_light,R_sersic_light,r_core,
        gamma,b_ani,inclination],axis=-1)
    
    return my_input

# TODO: this needs lenstronomy, so we need to check the numpy versions, etc.
class VelocityDisp:

    def __init__(self,psf_fwhm=0.5,R_aperture=0.725,R_inner_mask=0.5):
        """
        Args: 
            psf_fwhm (float): in arcseconds
            R_aperture (float): in arcseconds. Default = 4MOST aperture
            R_inner_mask (float): in arcseconds. Default = 0.5"
        """

        self.NUMPIX = 551
        self.PIX_SIZE = 0.02

        # set to match v_rms map from SKiNN! (551x551 at 0.02" resolution)
        self.pixel_grid = PixelGrid(
                nx=self.NUMPIX,ny=self.NUMPIX,
                transform_pix2angle=np.asarray([[self.PIX_SIZE,0.],[0.,self.PIX_SIZE]]),
                ra_at_xy_0=-self.PIX_SIZE*275.,dec_at_xy_0=-self.PIX_SIZE*275.
        )

        # PSF is simple, fwhm=0.5"
        psf_obj = PSF(
            psf_type='GAUSSIAN',fwhm=psf_fwhm
        )
        # set up PSF convolution params
        self.psf_fwhm_arcsec = psf_fwhm
        psf_fwhm_pix = psf_fwhm/self.PIX_SIZE # in pixels
        self.psf_sigma_pix = psf_fwhm_pix / (2*np.sqrt(2*np.log(2))) # FWHM = 2sqrt(2*ln(2)) * sigma

        # Set up grid of Radius values for masking
        # sum within a circular aperture, with center region masked
        self.R_ap_arcsec = R_aperture
        self.R_ap = R_aperture/self.PIX_SIZE # 2" aperture
        self.R_inner_arcsec = R_inner_mask
        self.R_inner = R_inner_mask/self.PIX_SIZE # mask center 0.5" pixels
        # create grid and evaluate whether Radius is inside or outside R_aperture
        center_pix_vals = np.arange(-self.NUMPIX/2 +0.5,self.NUMPIX/2 +0.5, 1.)
        x_grid, y_grid = np.meshgrid(center_pix_vals, center_pix_vals)
        self.R_grid = np.sqrt(x_grid**2 + y_grid**2)

        # Lens Light Model
        light_model = LightModel(['SERSIC_ELLIPSE'])
        self.lens_galaxy_model = ImageModel(
            data_class=self.pixel_grid,
            psf_class=psf_obj,
            lens_light_model_class=light_model  
        )
        self.kwargs_lens_light = [{
            'amp':10.,
            'n_sersic':None,
            'R_sersic':None,
            'e1':0.,
            'e2':0.,
            'center_x':0.,
            'center_y':0.
        }]

        self._setup_galkin()

    def _setup_galkin(self):

        # circular aperture
        self.kwargs_aperture = {
            'aperture_type': 'shell', 
            'r_in': self.R_inner_arcsec , 
            'r_out': self.R_ap_arcsec,
            'center_ra': 0, 'center_dec': 0}
        self.kwargs_seeing = {'psf_type': 'GAUSSIAN', 'fwhm': self.psf_fwhm_arcsec}

        self.kwargs_numerics_galkin = { 
            'interpol_grid_num': 1000,  # numerical interpolation, should converge -> infinity
            'log_integration': True,  # log or linear interpolation of surface brightness and mass models
            'max_integrate': 100, 'min_integrate': 0.001}  # lower/upper bound of numerical integrals


        self.kwargs_model = {
            'lens_model_list':['SPP'],
            'lens_light_model_list':['SERSIC']
        }

    def v_disp_galkin(self,beta_ani,theta_E,gamma_lens,R_sersic,n_sersic):
        """
        Args:
            beta_ani (float): constant anisotropy value
            inputs_dict_row (pandas df row): 
        """

        anisotropy_model = 'const'
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


        tik = time.time()
        kinematicsAPI = KinematicsAPI(0.5, 2., self.kwargs_model, 
            self.kwargs_aperture, self.kwargs_seeing, anisotropy_model, 
            kwargs_numerics_galkin=self.kwargs_numerics_galkin, 
            lens_model_kinematics_bool=[True, False],
            sampling_number=5000,MGE_light=True)  # numerical ray-shooting, should converge -> infinity)

        vel_disp_numerical = kinematicsAPI.velocity_dispersion(kwargs_lens, 
            kwargs_lens_light, kwargs_anisotropy, r_eff=R_sersic, theta_E=theta_E)
        tok = time.time()

        print('galkin eval time: ', tok-tik)

        return vel_disp_numerical # in km/s


    def v_disp_from_v_rms(self,vrms_map,n_sersic,R_sersic):
        """
        Args: 
            vrms_map (array(float)): 551x551 v_rms map from SKiNN
            n_sersic (float): 
            R_sersic (float):
        """

        # reset lens light kwargs
        self.kwargs_lens_light[0]['n_sersic'] = n_sersic
        self.kwargs_lens_light[0]['R_sersic'] = R_sersic

        # Sigma(x,y) (surface brightness of the lens)
        tik = time.time()
        lens_sb = self.lens_galaxy_model.lens_surface_brightness(
            self.kwargs_lens_light,unconvolved=True)
        tok = time.time()
        print('sb calculation time: ', tok-tik)
        # v_rms(x,y)
        v_rms = vrms_map

        # convolve with PSF
        tik = time.time()
        numerator = gaussian_filter((lens_sb * (v_rms**2)),sigma=self.psf_sigma_pix,
            mode='nearest')
        denominator = gaussian_filter(lens_sb,sigma=self.psf_sigma_pix,
            mode='nearest')
        tok = time.time()
        print('convolution time: ', tok-tik)

        # zeroing out these #s is equivalent to ignoring them in the averaging
        numerator[self.R_grid>self.R_ap] = 0.
        denominator[self.R_grid>self.R_ap] = 0.
        numerator[self.R_grid<self.R_inner] = 0.
        denominator[self.R_grid<self.R_inner] = 0.
        # luminosity weighting
        v_rms_weighted = np.sum(numerator) / np.sum(denominator)

        #[sqrt{<$v_{rms}^2$>}]$_{A}$ 
        v_rms_pred = np.sqrt(v_rms_weighted)

        return v_rms_pred

########
# Main
########

if __name__ == '__main__':


    print('reading data!')

    #exp_folder = ('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/'+
    #        'DataVectors/src_mag_cut_silver_debiased')
    exp_folder = ('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/'+
            'DataVectors/')

    lens_types = ['gold_quads']

    inputs_dict = {
        'gold_quads':{},
    }

    input_keys = ['lens_param_samps','z_lens_truth','z_src_truth',
                'lens_light_parameters_R_sersic_truth', 
                'lens_light_parameters_n_sersic_truth',
                'lens_light_parameters_e1_truth', 
                'lens_light_parameters_e2_truth']

    for l in lens_types:
        my_filepath = (exp_folder+'/'+l+'.h5')
        h5f = h5py.File(my_filepath, 'r')
        for key in input_keys:
            inputs_dict[l][key] = h5f.get(key)[:]
        h5f.close()

    # generate skinn-formatted input from data vectors
    my_input = skinn_input(inputs_dict)

    print('initializing generator')
    generator=BatchedGenerator()

    # initialize vdisp_calculator
    vdisp_calculator = VelocityDisp(psf_fwhm=0.5,R_aperture=0.725,
        R_inner_mask=0.2)

    # test on 0th lens! start with 10 samples
    print('generating v_disps')
    start_time = time.time()
    first5_maps = generator.generate_map(my_input[0,:5])
    
    v_disp_skinn = []
    for i in range(0,5):
        v_disp_skinn.append(vdisp_calculator.v_disp_from_v_rms(
            first5_maps[i],n_sersic=my_input[0,i,3],R_sersic=my_input[0,i,4]
        ))
    end_time = time.time()
    print(f"Time taken for 5 SKiNN vdisp evaluations: {end_time - start_time} seconds")

    lens_param_samps = inputs_dict['gold_quads']['lens_param_samps']
    R_sersic0 = inputs_dict['gold_quads']['lens_light_parameters_R_sersic_truth'][0]
    n_sersic0 = inputs_dict['gold_quads']['lens_light_parameters_n_sersic_truth'][0]
    # let's try 5 lenses with galkin
    v_disp_galkin = []
    for i in range(0,5):
        v_disp_galkin.append(v_disp_galkin(
            beta_ani=1.,theta_E=lens_param_samps[0,i,0],
            gamma_lens=lens_param_samps[0,i,3],
            R_sersic=R_sersic0,n_sersic=n_sersic0))

    print("SKiNN v_disp output, lens 0: ", v_disp_skinn[0])
    print(" ")
    print("Galkin v_disp output, lens 0: ", v_disp_galkin[0])
