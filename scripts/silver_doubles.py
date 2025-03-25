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

lens_types = ['silver_dbls']

inputs_dict = {
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


###################
# Silver doubles
###################
my_tdc_w0waCDM = TDCLikelihood(
    inputs_dict['silver_dbls']['measured_td'],
    inputs_dict['silver_dbls']['measured_prec'],
    inputs_dict['silver_dbls']['prefactor'],
    inputs_dict['silver_dbls']['fpd_samps'],
    inputs_dict['silver_dbls']['gamma_samps'],
    inputs_dict['silver_dbls']['z_lens_truth'],
    inputs_dict['silver_dbls']['z_src_truth'],
    cosmo_model='w0waCDM',
    use_gamma_info=True,
    log_prob_modeling_prior='hardcoded',
    use_astropy=False)

mcmc_chain_w0waCDM = fast_TDC(my_tdc_w0waCDM,num_emcee_samps=num_emcee_samps,
    cosmo_model='w0waCDM',n_walkers=20)

h5f = h5py.File((exp_folder+'/silver_doubles_ALL_chain_5e3_w0waCDM.h5'), 'w')
h5f.create_dataset('mcmc_chain', data=mcmc_chain_w0waCDM)
h5f.close()

bad_idx = [4,  54,  87,  90, 197, 298, 324] # these idxs have >100% error on Ddt!

good_idx = np.arange(0,len(inputs_dict['silver_dbls']['z_lens_truth']))
good_idx = np.setdiff1d(good_idx, bad_idx)

#############################
# Silver doubles remove "bad"
#############################
my_tdc_w0waCDM = TDCLikelihood(
    inputs_dict['silver_dbls']['measured_td'][good_idx],
    inputs_dict['silver_dbls']['measured_prec'][good_idx],
    inputs_dict['silver_dbls']['prefactor'][good_idx],
    inputs_dict['silver_dbls']['fpd_samps'][good_idx],
    inputs_dict['silver_dbls']['gamma_samps'][good_idx],
    inputs_dict['silver_dbls']['z_lens_truth'][good_idx],
    inputs_dict['silver_dbls']['z_src_truth'][good_idx],
    cosmo_model='w0waCDM',
    use_gamma_info=True,
    log_prob_modeling_prior='hardcoded',
    use_astropy=False)

mcmc_chain_w0waCDM = fast_TDC(my_tdc_w0waCDM,num_emcee_samps=num_emcee_samps,
    cosmo_model='w0waCDM',n_walkers=20)

h5f = h5py.File((exp_folder+'/silver_doubles_REMOVEBAD_chain_5e3_w0waCDM.h5'), 'w')
h5f.create_dataset('mcmc_chain', data=mcmc_chain_w0waCDM)
h5f.close()