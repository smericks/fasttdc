#import SKiNN
#import torch
import pickle
import numpy as np
import pandas as pd
import time
import h5py
#assert(torch.cuda.is_available()==True) #must be True to function
#from SKiNN.generator import Generator

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

# TODO: read from DataVector
exp_folder = ('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/'+
        'DataVectors/src_mag_cut_silver_debiased')
#exp_folder = ('/scratch/users/sydney3/forecast/darkenergy-from-LAGN/'+
#        'DataVectors/')

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


print(q_mass.shape)

my_input = np.stack([q_mass,q_light,theta_E,
    n_sersic_light,R_sersic_light,r_core,
    gamma,b_ani,inclination],axis=-1)

print(my_input.shape)



generator=Generator()
# test with a batch dim of 100...
start_time = time.time()
first100_maps = generator.generate_map(my_input[0,:100])
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

"""
# TODO: make this operate over a batch
# TODO: add timing

vrms_map=np.array(generator.generate_map(my_input[0]))
maps_first10 = np.empty((10,np.shape(vrms_map)[0],np.shape(vrms_map)[1]))
maps_first10[0] = vrms_map
for i in range(1,10):
	vrms_map=np.array(generator.generate_map(my_input[i]))
	maps_first10[i] = vrms_map
np.save('test_vrms.npy',maps_first10)
"""