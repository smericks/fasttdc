from scipy.stats import multivariate_normal, norm
import h5py 
import numpy as np
import batched_fermatpot

def make_data_vectors(lens_catalog,lens_indices,num_images,td_meas_error,
        npe_mu,npe_cov,h5_save_path,num_fpd_samps=1000,emulated=False):
    """
    Args:
        lens_catalog (lens_catalog.LensCatalog object): contains ground truth
            information about lens sample
        lens_indices ([int]): subsample of lens_catalog to compute data vectors 
            for
        num_images (int: 2 or 4): specify whether lens subsample is quads or doubles
            NOTE: we must separate doubles & quads when making data vectors
        td_meas_error (float): time delay measurement error (in days) applied to
            every lens in the sub sample (assumed Gaussian error)
        NOTE: we assume full-cov. Gaussian posteriors from NPE
        npe_mu (np.array, size=(n_lenses,n_params)): List of Gaussian mean vector 
            of NPE posterior for each lens
        npe_cov (np.array, size=(n_lenses,n_params,n_params)): List of Gaussian 
            covariance matrix of NPE posterior for each lens
        h5_save_path (string): where to save data vectors 
        num_fpd_samps (int, default=1000): posteriors for fermat potential 
            differences are stored as a collection of samples, which are later 
            used for importance sampling. This parameter sets the number of 
            samples stored. NOTE: For final TDC inference, 5000 samples is 
            recommended.

    Returns:
        saves data vectors to a .h5 file. Saves objects with names: 
            ['measured_td','measured_prec','prefactor','fpd_samps',
            'gamma_samps','z_lens','z_src']
    """

    if num_images not in [2,4]:
        print('in make_data_vectors(), num_images must be 2 or 4')
        raise ValueError
    
    size_subsamp = len(lens_indices)
    
    # STEP 1: emulate time-delay measurements

    measured_td = np.zeros((size_subsamp,3)) # dbls padded w/ zeros
    sigma = np.ones(size_subsamp)*td_meas_error

    if num_images == 2:
        mus = lens_catalog.lens_df.loc[lens_indices,'td01']
        measured_td[:,0] = norm.rvs(loc=mus,scale=sigma)
        measured_prec = np.zeros((size_subsamp,3,3))
        measured_prec[:,0,0] += 1/(td_meas_error**2)
        # 1d Gaussian multiplicative factor out front
        prefactor = np.log(np.ones(size_subsamp)* (1/(2*np.pi))**(1/2) / 
            td_meas_error)
    elif num_images == 4:
        mus = lens_catalog.lens_df.loc[lens_indices,['td01','td02','td03']].to_numpy()
        for i in range(0,3):
            measured_td[:,i] = norm.rvs(loc=mus[:,i],scale=sigma)
        measured_prec = np.eye(3,3)/(td_meas_error**2)
        measured_prec = np.repeat(measured_prec[np.newaxis,:],size_subsamp, 
            axis=0)
        # 3d Gaussian multiplicative factor out front
        prefactor = np.log(np.ones(size_subsamp)* (1/(2*np.pi))**(3/2) / (
            np.sqrt(np.linalg.det(np.eye(3,3)*(td_meas_error**2)))))

    # STEP 2: compute fpd samps, track gamma samps from npe posteriors
    if emulated:
        fpd_samps,gamma_samps = emulated_fpd_gamma_samples(size_subsamp,
            num_fpd_samps,num_images,lens_indices,lens_catalog,npe_mu,npe_cov)
    else:
        fpd_samps, gamma_samps = fpd_gamma_samples(size_subsamp,num_fpd_samps,
            num_images,lens_indices,lens_catalog,npe_mu,npe_cov)

    # let's dump everything into a .h5 file

    h5f = h5py.File(h5_save_path, 'w')
    h5f.create_dataset('measured_td', data=measured_td)
    h5f.create_dataset('measured_prec',data=measured_prec)
    h5f.create_dataset('prefactor',data=prefactor)
    h5f.create_dataset('fpd_samps',data=fpd_samps)
    h5f.create_dataset('gamma_samps',data=gamma_samps)

    # add in ground truth info from lens_catalog (including redshifts)
    for key in lens_catalog.lens_df.keys():
        h5f.create_dataset(key+'_truth',data=lens_catalog.lens_df.loc[lens_indices,
            key].to_numpy().astype(float))
    h5f.close()



def fpd_gamma_samples(size_subsamp,num_fpd_samps,num_images,lens_indices,lens_catalog,
    npe_mu,npe_cov):
    """
    Args:
    """
    fpd_samps = np.zeros((size_subsamp,num_fpd_samps,3)) # dbls padded w/ zeros
    gamma_samps = np.empty((size_subsamp,num_fpd_samps))

    if num_images==2:
        for i in range(0,size_subsamp):
            idx = lens_indices[i]
            lens_param_samps = multivariate_normal.rvs(mean=npe_mu[idx],
                cov=npe_cov[idx],size=num_fpd_samps)
            x_im = lens_catalog.lens_df.loc[idx,
                ['x_im0','x_im1']].to_numpy().astype(float)
            y_im = lens_catalog.lens_df.loc[idx,
                ['y_im0','y_im1']].to_numpy().astype(float)
            fermatpot_samps = batched_fermatpot.eplshear_fp_samples(x_im,y_im,
                lens_param_samps[:,:8],lens_param_samps[:,8],
                lens_param_samps[:,9])
            fpd_samps[i,:,0] = fermatpot_samps[:,0] - fermatpot_samps[:,1]
            gamma_samps[i,:] = lens_param_samps[:,3]

    elif num_images==4:
        for i in range(0,size_subsamp):
            idx = lens_indices[i]
            lens_param_samps = multivariate_normal.rvs(mean=npe_mu[idx],
                cov=npe_cov[idx],size=num_fpd_samps)
            x_im = lens_catalog.lens_df.loc[idx,
                ['x_im0','x_im1','x_im2','x_im3']].to_numpy().astype(float)
            y_im = lens_catalog.lens_df.loc[idx,
                ['y_im0','y_im1','y_im2','y_im3']].to_numpy().astype(float)
            fermatpot_samps = batched_fermatpot.eplshear_fp_samples(x_im,y_im,
                lens_param_samps[:,:8],lens_param_samps[:,8],
                lens_param_samps[:,9])
            fpd_samps[i,:,0] = fermatpot_samps[:,0] - fermatpot_samps[:,1]
            fpd_samps[i,:,1] = fermatpot_samps[:,0] - fermatpot_samps[:,2]
            fpd_samps[i,:,2] = fermatpot_samps[:,0] - fermatpot_samps[:,3]
            gamma_samps[i,:] = lens_param_samps[:,3]

    return fpd_samps, gamma_samps 


def emulated_fpd_gamma_samples(size_subsamp,num_fpd_samps,num_images,lens_indices,lens_catalog,
    npe_mu,npe_cov):
    """
    Args:
    """
    fpd_samps = np.zeros((size_subsamp,num_fpd_samps,3)) # dbls padded w/ zeros
    gamma_samps = np.empty((size_subsamp,num_fpd_samps))

    if num_images==2:
        for i in range(0,size_subsamp):
            idx = lens_indices[i]
            lens_param_samps = multivariate_normal.rvs(mean=npe_mu[idx],
                cov=npe_cov[idx],size=num_fpd_samps)
            gamma_samps[i,:] = lens_param_samps[:,3]

            fpd01_truth = lens_catalog.lens_df.loc[idx,
                ['fpd01']]
            # random 5% error
            fermatpot_samps = norm.rvs(loc=fpd01_truth,
                scale=0.05*(np.abs(fpd01_truth)),size=num_fpd_samps)
            fpd_samps[i,:,0] = fermatpot_samps

    elif num_images==4:
        for i in range(0,size_subsamp):
            idx = lens_indices[i]
            lens_param_samps = multivariate_normal.rvs(mean=npe_mu[idx],
                cov=npe_cov[idx],size=num_fpd_samps)
            gamma_samps[i,:] = lens_param_samps[:,3]

            for j in range(0,3):
                fpd0j_truth = lens_catalog.lens_df.loc[idx,
                ['fpd0%d'%(j+1)]]
                # random 5% error
                fermatpot_samps = norm.rvs(loc=fpd0j_truth,
                scale=0.05*(np.abs(fpd0j_truth)),size=num_fpd_samps)
                fpd_samps[i,:,j] = fermatpot_samps


    return fpd_samps, gamma_samps 