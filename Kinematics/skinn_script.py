import SKiNN
import torch
import pickle
import numpy as np
import pandas as pd
assert(torch.cuda.is_available()==True) #must be True to function
from SKiNN.generator import Generator

# FROM LENSTRONOMY: https://github.com/lenstronomy/lenstronomy/blob/6de71de7f6cd8c5512764bb511e3bee5ac6d1d41/lenstronomy/Util/param_util.py#L92
def ellipticity2phi_q(e1, e2):
    """Transforms complex ellipticity moduli in orientation angle and axis ratio.

    :param e1: eccentricity in x-direction
    :param e2: eccentricity in xy-direction
    :return: angle in radian, axis ratio (minor/major)
    """
    phi = np.arctan2(e2, e1) / 2
    c = np.sqrt(e1**2 + e2**2)
    c = np.minimum(c, 0.9999)
    q = (1 - c) / (1 + c)
    return phi, q

############
# MAIN BODY
############

gt_df = pd.read_csv('om10_venkatraman_erickson24.csv')

#[q_mass, q_light, theta_E, n_sersic_light, R_sersic_light, r_core , gamma, b_ani, inclination]

mass_e1 = gt_df['main_deflector_parameters_e1'].to_numpy().astype(float)
mass_e2 = gt_df['main_deflector_parameters_e2'].to_numpy().astype(float)
phi_mass,q_mass = ellipticity2phi_q(mass_e1,mass_e2)

light_e1 = gt_df['lens_light_parameters_e1'].to_numpy().astype(float)
light_e2 = gt_df['lens_light_parameters_e2'].to_numpy().astype(float)
phi_light,q_light = ellipticity2phi_q(light_e1,light_e2)

theta_E_lenst = gt_df['main_deflector_parameters_theta_E'].to_numpy().astype(float)
gamma_lenst = gt_df['main_deflector_parameters_gamma'].to_numpy().astype(float)

gamma = (gamma_lenst - 1) / 2
r_e_scale = (2 / (1 + q_mass)) ** (1 / (2 * gamma)) * np.sqrt(q_mass)
theta_E = theta_E_lenst / r_e_scale

n_sersic_light = gt_df['lens_light_parameters_n_sersic'].to_numpy().astype(float)
R_sersic_light = gt_df['lens_light_parameters_R_sersic'].to_numpy().astype(float)

r_core = np.ones(n_sersic_light.shape)*0.08

# TODO: how to initialize?
b_ani = np.ones(np.shape(r_core))
inclination = np.ones(np.shape(r_core))*85


my_input = np.vstack((q_mass,q_light,theta_E,n_sersic_light,R_sersic_light,r_core,gamma,
                      b_ani,inclination)).T


generator=Generator()
vrms_map=np.array(generator.generate_map(my_input[0]))
maps_first10 = np.empty((10,np.shape(vrms_map)[0],np.shape(vrms_map)[1]))
maps_first10[0] = vrms_map
for i in range(1,10):
	vrms_map=np.array(generator.generate_map(my_input[i]))
	maps_first10[i] = vrms_map
np.save('test_vrms.npy',maps_first10)
