import time
import h5py
from scipy.stats import norm
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
from tdc_sampler import TDCLikelihood,fast_TDC

# inference configurations
num_emcee_samps = 100
# where data vectors are stored
exp_folder = ('/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/'+
        'DataVectors/src_mag_cut_silver_debiased')

lens_types = ['gold_quads','gold_dbls']

inputs_dict = {
    'gold_quads':{},
    'gold_dbls':{},
}

input_keys = ['measured_td','measured_prec','prefactor','fpd_samps',
    'lens_param_samps','z_lens_truth','z_src_truth']

for l in lens_types:
    my_filepath = (exp_folder+'/'+l+'.h5')
    h5f = h5py.File(my_filepath, 'r')
    for key in input_keys:
        inputs_dict[l][key] = h5f.get(key)[:]
    h5f.close()


# let's run gold-only and gold+silver
gold_measured_td = np.vstack((
    inputs_dict['gold_quads']['measured_td'],
    inputs_dict['gold_dbls']['measured_td']))
gold_measured_prec = np.vstack((
    inputs_dict['gold_quads']['measured_prec'],
    inputs_dict['gold_dbls']['measured_prec']
))
gold_prefactor = np.append(
    inputs_dict['gold_quads']['prefactor'],
    inputs_dict['gold_dbls']['prefactor']
)
gold_fpd_samps = np.vstack((
    inputs_dict['gold_quads']['fpd_samps'],
    inputs_dict['gold_dbls']['fpd_samps']
))

print(np.shape(inputs_dict['gold_quads']['lens_param_samps']))

gold_gamma_samps = np.vstack((
    inputs_dict['gold_quads']['lens_param_samps'][:,:,3],
    inputs_dict['gold_dbls']['lens_param_samps'][:,:,3]
))
gold_z_lens = np.append(
    inputs_dict['gold_quads']['z_lens_truth'],
    inputs_dict['gold_dbls']['z_lens_truth']
)
gold_z_src = np.append(
    inputs_dict['gold_quads']['z_src_truth'],
    inputs_dict['gold_dbls']['z_src_truth']
)


my_tdc_w0waCDM = TDCLikelihood(
    gold_measured_td,
    gold_measured_prec,
    gold_prefactor,
    gold_fpd_samps,
    gold_gamma_samps,
    gold_z_lens,
    gold_z_src,
    cosmo_model='w0waCDM',
    use_gamma_info=True,
    log_prob_modeling_prior='hardcoded',
    use_astropy=False)

mcmc_chain_gold_w0wa = fast_TDC(my_tdc_w0waCDM,num_emcee_samps=num_emcee_samps,
    cosmo_model='w0waCDM',n_walkers=20)


h5f = h5py.File((exp_folder+'/gold_chain_5e3_w0waCDM.h5'), 'w')
h5f.create_dataset('mcmc_chain', data=mcmc_chain_gold_w0wa)
h5f.close()
