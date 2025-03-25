import time
import h5py
from scipy.stats import norm
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
from tdc_sampler import TDCLikelihood,fast_TDC

# inference configurations
num_emcee_samps = 5000
# where data vectors are stored
exp_folder = ('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/'+
        'DataVectors/fpd_eval_DV_silverALLDEBIASED')

lens_types = ['silver_quads','silver_dbls']

inputs_dict = {
    'silver_quads':{},
    'silver_dbls':{}
}

input_keys = ['measured_td','measured_prec','prefactor','fpd_samps',
    'gamma_samps','z_lens_truth','z_src_truth']

for l in lens_types:
    my_filepath = (exp_folder+'/'+l+'.h5')
    h5f = h5py.File(my_filepath, 'r')
    for key in input_keys:
        inputs_dict[l][key] = h5f.get(key)[:]
    h5f.close()


# let's run silver doubles ONLY
# let's run gold+silver
silver_measured_td = np.vstack((
    inputs_dict['silver_quads']['measured_td'],
    inputs_dict['silver_dbls']['measured_td']))

silver_measured_prec = np.vstack((
    inputs_dict['silver_quads']['measured_prec'],
    inputs_dict['silver_dbls']['measured_prec']
))

silver_prefactor = np.append(inputs_dict['silver_quads']['prefactor'],
    inputs_dict['silver_dbls']['prefactor'])

silver_fpd_samps = np.vstack((
    inputs_dict['silver_quads']['fpd_samps'],
    inputs_dict['silver_dbls']['fpd_samps']
))

silver_gamma_samps = np.vstack((
    inputs_dict['silver_quads']['gamma_samps'],
    inputs_dict['silver_dbls']['gamma_samps']
))

silver_z_lens = np.append(inputs_dict['silver_quads']['z_lens_truth'],
    inputs_dict['silver_dbls']['z_lens_truth'])

silver_z_src = np.append(inputs_dict['silver_quads']['z_src_truth'],inputs_dict['silver_dbls']['z_src_truth'])


###################
# All silver lenses
###################
my_tdc_w0waCDM = TDCLikelihood(
    silver_measured_td,
    silver_measured_prec,
    silver_prefactor,
    silver_fpd_samps,
    silver_gamma_samps,
    silver_z_lens,
    silver_z_src,
    cosmo_model='w0waCDM',
    use_gamma_info=True,
    log_prob_modeling_prior='hardcoded',
    use_astropy=False)

mcmc_chain_w0waCDM = fast_TDC(my_tdc_w0waCDM,num_emcee_samps=num_emcee_samps,
    cosmo_model='w0waCDM',n_walkers=20)

h5f = h5py.File((exp_folder+'/silver_chain_5e3_w0waCDM.h5'), 'w')
h5f.create_dataset('mcmc_chain', data=mcmc_chain_w0waCDM)
h5f.close()