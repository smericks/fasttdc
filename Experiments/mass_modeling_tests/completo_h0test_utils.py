# FUNCTION to convert alpaca samples/truths into a fasttdc data_vector_dict
import numpy as np
import h5py 
from sklearn.neighbors import KernelDensity

from Utils.data_vector_utils import emulate_measurements
from Utils.tdc_utils import ddt_from_redshifts, td_from_ddt_fpd
from completo_im_position_utils import completo_image_positions, completo_param_samps_to_lenst_kwargs, image_positions
from batched_fermatpot import eplshear_fp_samples
import os
import sys
import time
dirname = os.getcwd()
sys.path.insert(0, os.path.join(dirname, '../../../sbi-stronglensing'))
from src.training.data_loader import load_hdf5_labels


def load_truth_vals_dict(completo_truth_h5, chosen_idxs, GROUNDTRUTH_COSMO): 

    # load in ground truths
    truth_labels = [
        'deflector_center_x','deflector_center_y', # lens center (TODO will be modified in future)
        'deflector_LOG_z','source_LOG_zS_minus_Zd', # redshifts
        'deflector_LOG_theta_E','los_gamma1','los_gamma2', # NEED lens params for g.t. im positions
        'deflector_LOG_gamma_pl','deflector_e1_mass','deflector_e2_mass',
        'source_center_x','source_center_y'
    ]
    truth_vals_dict = {}
    with h5py.File(completo_truth_h5, 'r') as f:
        for label in truth_labels:
            truth_vals_dict[label] = f[label][chosen_idxs]

    # convert redshifts...
    truth_vals_dict['z_lens'] = np.exp(truth_vals_dict['deflector_LOG_z'])
    truth_vals_dict['z_src'] = np.exp(truth_vals_dict['source_LOG_zS_minus_Zd']) + truth_vals_dict['z_lens']

    # convert lens params
    truth_vals_dict['deflector_theta_E'] = np.exp(truth_vals_dict['deflector_LOG_theta_E'])
    truth_vals_dict['deflector_gamma_pl'] = np.exp(truth_vals_dict['deflector_LOG_gamma_pl'])

    # initialize storage of image positions & number images
    x_empty_keys = ['point_source_x_im_%d'%(xd) for xd in range(0,6)]
    y_empty_keys = ['point_source_y_im_%d'%(yd) for yd in range(0,6)]
    fpd_empty_keys = ['fpd_0%d'%(imidx) for imidx in range(1,6)]
    td_empty_keys = ['td_0%d'%(imidx) for imidx in range(1,6)]
    for xkey in x_empty_keys:
        truth_vals_dict[xkey] = np.nan*np.ones(np.shape(truth_vals_dict['z_lens']))
    for ykey in y_empty_keys:
        truth_vals_dict[ykey] = np.nan*np.ones(np.shape(truth_vals_dict['z_lens']))
    for fkey in fpd_empty_keys:
        truth_vals_dict[fkey] = np.nan*np.ones(np.shape(truth_vals_dict['z_lens']))
    for tkey in td_empty_keys:
        truth_vals_dict[tkey] = np.nan*np.ones(np.shape(truth_vals_dict['z_lens']))
    
    # num image positions
    truth_vals_dict['point_source_num_images'] = np.nan*np.ones(np.shape(truth_vals_dict['z_lens']))
    truth_vals_dict['Ddt_Mpc'] = np.nan*np.ones(np.shape(truth_vals_dict['z_lens']))

    # loop through every lens & compute ground truths
    for lidx in range(0,len(truth_vals_dict['z_lens'])):
        truth_params_slsim = [
            truth_vals_dict['deflector_theta_E'][lidx],
            truth_vals_dict['los_gamma1'][lidx],
            truth_vals_dict['los_gamma2'][lidx],
            truth_vals_dict['deflector_gamma_pl'][lidx],
            truth_vals_dict['deflector_e1_mass'][lidx],
            truth_vals_dict['deflector_e2_mass'][lidx],
            truth_vals_dict['deflector_center_x'][lidx],
            truth_vals_dict['deflector_center_y'][lidx],
            truth_vals_dict['source_center_x'][lidx],
            truth_vals_dict['source_center_y'][lidx],
        ]
        truth_params_lenst = [
            truth_vals_dict['deflector_theta_E'][lidx],
            -truth_vals_dict['los_gamma1'][lidx],
            truth_vals_dict['los_gamma2'][lidx],
            truth_vals_dict['deflector_gamma_pl'][lidx],
            -truth_vals_dict['deflector_e1_mass'][lidx],
            truth_vals_dict['deflector_e2_mass'][lidx],
            truth_vals_dict['deflector_center_x'][lidx],
            truth_vals_dict['deflector_center_y'][lidx],
            truth_vals_dict['source_center_x'][lidx],
            truth_vals_dict['source_center_y'][lidx],
        ]

        # (1) COMPUTE ground truths

        # image positions
        x_im_list, y_im_list = completo_image_positions(truth_params_slsim)

        # fermat potentials
        fp_at_ims = eplshear_fp_samples(x_im_list,y_im_list,
            np.asarray([truth_params_lenst[:-2]]),
           np.asarray([truth_params_lenst[-2]]),np.asarray([truth_params_lenst[-1]])) #outputs [n_samps,n_ims]

        fpd_list = []
        for i in range(1,len(fp_at_ims[0])):
            fpd_list.append(fp_at_ims[0][0] - fp_at_ims[0][i])
        fpd_list = np.asarray(fpd_list)

        # ddt from redshifts
        ddt_truth = ddt_from_redshifts(GROUNDTRUTH_COSMO,
            truth_vals_dict['z_lens'][lidx],truth_vals_dict['z_src'][lidx]).value
        
        # td from ddt + fpds
        td_truth = td_from_ddt_fpd(ddt_truth,fpd_list)

        # (2) STORE ground truths

        # one per lens
        truth_vals_dict['point_source_num_images'][lidx] = len(x_im_list)
        truth_vals_dict['Ddt_Mpc'][lidx] = ddt_truth

        # one per p.s. image
        for im_idx in range(0,len(x_im_list)):
            # im positions
            truth_vals_dict['point_source_x_im_%d'%(im_idx)][lidx] = x_im_list[im_idx]
            truth_vals_dict['point_source_y_im_%d'%(im_idx)][lidx] = y_im_list[im_idx]
            if im_idx > 0:
                # fermat potential
                truth_vals_dict['fpd_0%d'%(im_idx)][lidx] = fpd_list[im_idx-1]
                # time-delays
                truth_vals_dict['td_0%d'%(im_idx)][lidx] = td_truth[im_idx-1]

    return truth_vals_dict


def fasttdc_samps_from_posterior_samps(posterior_samps,truth_vals_dict):

    num_fpd_samps = np.shape(posterior_samps)[1] 
    fasttdc_lens_samps = np.asarray([
        np.exp(posterior_samps[:,:,0]), # LOG theta_E -> theta_E
        -posterior_samps[:,:,1], # DIFF convention gamma1 -> -gamma1
        posterior_samps[:,:,2],
        np.exp(posterior_samps[:,:,3]), # LOG gamma_lens -> gamma_lens
        -posterior_samps[:,:,4], # DIFF convention e1 -> -e1
        posterior_samps[:,:,5],
        posterior_samps[:,:,6], # x_lens NOTE this is changed!!
        posterior_samps[:,:,7], # y_lens
        #np.repeat(np.expand_dims(truth_vals_dict['deflector_center_x'],axis=1),num_fpd_samps,axis=1),# TODO: THIS IS WRONG NOW
        #np.repeat(np.expand_dims(truth_vals_dict['deflector_center_y'],axis=1),num_fpd_samps,axis=1),
        posterior_samps[:,:,-3], # x_src
        posterior_samps[:,:,-2]  # y_src
    ])
    fasttdc_lens_samps = np.transpose(fasttdc_lens_samps,axes=(1,2,0))


    # TODO: separate into doubles and quads
    dbls_idxs = np.where(truth_vals_dict['point_source_num_images'] == 2)[0]
    quads_idxs = np.where(truth_vals_dict['point_source_num_images'] == 4)[0]

    fasttdc_lens_samps_dbls = fasttdc_lens_samps[dbls_idxs]
    num_dbls = np.shape(fasttdc_lens_samps_dbls)[0]
    fasttdc_lens_samps_quads = fasttdc_lens_samps[quads_idxs]
    num_quads = np.shape(fasttdc_lens_samps_quads)[0]                                   

    # fill in fpd samps for the doubles
    fasttdc_fpd_samps_dbls = np.empty((num_dbls,num_fpd_samps,1))
    for dbl_idx in range(0,num_dbls):
        lidx = dbls_idxs[dbl_idx] # global index into all lenses dict
        fp_at_ims = eplshear_fp_samples(
            x_im = np.asarray([truth_vals_dict['point_source_x_im_0'][lidx],
                    truth_vals_dict['point_source_x_im_1'][lidx]]),
            y_im = np.asarray([truth_vals_dict['point_source_y_im_0'][lidx],
                    truth_vals_dict['point_source_y_im_1'][lidx]]),
            lens_model_samps = fasttdc_lens_samps_dbls[dbl_idx,:,:8],
            x_src_samps = fasttdc_lens_samps_dbls[dbl_idx,:,-2],
            y_src_samps = fasttdc_lens_samps_dbls[dbl_idx,:,-1]
        )
        fasttdc_fpd_samps_dbls[dbl_idx,:,0] = fp_at_ims[:,0] - fp_at_ims[:,1] # fpd01

    # fill in the fpd samps for the quads
    fasttdc_fpd_samps_quads = np.empty((num_quads,num_fpd_samps,3))
    for q_idx in range(0,num_quads):
        lidx = quads_idxs[q_idx] # global index into all lenses dict
        fp_at_ims = eplshear_fp_samples(
            x_im = np.asarray([truth_vals_dict['point_source_x_im_0'][lidx],
                    truth_vals_dict['point_source_x_im_1'][lidx],
                    truth_vals_dict['point_source_x_im_2'][lidx],
                    truth_vals_dict['point_source_x_im_3'][lidx]]),
            y_im = np.asarray([truth_vals_dict['point_source_y_im_0'][lidx],
                    truth_vals_dict['point_source_y_im_1'][lidx],
                    truth_vals_dict['point_source_y_im_2'][lidx],
                    truth_vals_dict['point_source_y_im_3'][lidx]]),
            lens_model_samps = fasttdc_lens_samps_quads[q_idx,:,:8],
            x_src_samps = fasttdc_lens_samps_quads[q_idx,:,-2],
            y_src_samps = fasttdc_lens_samps_quads[q_idx,:,-1]
        )
        fasttdc_fpd_samps_quads[q_idx,:,0] = fp_at_ims[:,0] - fp_at_ims[:,1] # fpd01
        fasttdc_fpd_samps_quads[q_idx,:,1] = fp_at_ims[:,0] - fp_at_ims[:,2] # fpd02
        fasttdc_fpd_samps_quads[q_idx,:,2] = fp_at_ims[:,0] - fp_at_ims[:,3] # fpd03


    return (dbls_idxs, quads_idxs, fasttdc_fpd_samps_dbls, 
                fasttdc_fpd_samps_quads, fasttdc_lens_samps_dbls, 
                fasttdc_lens_samps_quads)

# TODO: switch from num_truth to indexing particular lenses...
def dv_dict_from_completo(completo_samples_npy, completo_truth_h5, #training_samps_h5,
    chosen_idxs, GROUNDTRUTH_COSMO, td_meas_err_percent=0.05):
    """
    Args: 
        completo_samples_npy (string): 
        completo_truth_h5: i.e. '/Users/smericks/Desktop/StrongLensing/project3/slsim/slsim/TrainingSets/10ktest_seed7.h5'
        training_samps_h5 (string): training examples needed to build understanding of the interim prior
        num_truth (int): number of lenses we're actually using from the truth catalog...
        
    Returns: 
        (dict) data_vector_dict with keys: 
            [ 'td_measured':,
              'td_likelihood_prec':,
              'td_likelihood_prefactors':,
              'fpd_samples':,
              'gamma_pred_samples':,
              'z_lens':,
              'z_src': ]
    """

    # read in from samples file
    # in order: ['deflector_LOG_theta_E','los_gamma1','los_gamma2',
    #   'deflector_LOG_gamma_pl','deflector_e1_mass','deflector_e2_mass',
    #    'deflector_e1_light','deflector_e2_light','deflector_LOG_angular_size',
    #    'deflector_n_sersic',
    #    'source_center_x','source_center_y','source_LOG_angular_size']

    # load in posterior samples
    posterior_samps = np.load(completo_samples_npy) # shape: n_samps, n_lenses n_params
    posterior_samps = np.transpose(posterior_samps,axes=(1,0,2)) # shape: n_lenses, n_samps, n_params
    posterior_samps = posterior_samps[chosen_idxs] # only take certain # of lenses
    print('posterior samps shape: ', posterior_samps.shape)

    truth_vals_dict = load_truth_vals_dict(completo_truth_h5, chosen_idxs, GROUNDTRUTH_COSMO)

    (dbls_idxs, quads_idxs, fasttdc_fpd_samps_dbls, 
        fasttdc_fpd_samps_quads, fasttdc_lens_samps_dbls, 
        fasttdc_lens_samps_quads) = fasttdc_samps_from_posterior_samps(posterior_samps,truth_vals_dict)

    # dbls time-delays
    td_meas_dbls, td_meas_prec_dbls = emulate_measurements(
        np.expand_dims(truth_vals_dict['td_01'][dbls_idxs],axis=1),measurement_error_kmpersec=2. # note this is actually days
    )

    # quads time-delays
    td_meas_quads, td_meas_prec_quads = emulate_measurements(
        np.vstack((truth_vals_dict['td_01'][quads_idxs],
                   truth_vals_dict['td_02'][quads_idxs],
                   truth_vals_dict['td_03'][quads_idxs])).T,
        measurement_error_kmpersec=2. # note this is actually days
    )

    num_fpd_samps = np.shape(fasttdc_fpd_samps_quads)[1]

    # add repeats on 2nd axis for compatibility with importance samples
    td_meas_dbls = np.repeat(td_meas_dbls[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    td_meas_prec_dbls = np.repeat(td_meas_prec_dbls[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    td_meas_quads = np.repeat(td_meas_quads[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    td_meas_prec_quads = np.repeat(td_meas_prec_quads[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    
    # add prefactors to track wheter its a 1D vs 3D Gaussian evaluation
    num_td = 1
    td_gaussian_prefactor_dbls = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(td_meas_prec_dbls))) )
    
    num_td = 3
    td_gaussian_prefactor_quads = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(td_meas_prec_quads))) )
    

    # create the data vector dictionary
    data_vector_dict_dbls = {   
        'td_measured':td_meas_dbls,
        'td_likelihood_prec':td_meas_prec_dbls,
        'td_likelihood_prefactors':td_gaussian_prefactor_dbls,
        'fpd_samples':fasttdc_fpd_samps_dbls,
        'lens_param_samples':fasttdc_lens_samps_dbls,
        'z_lens':truth_vals_dict['z_lens'][dbls_idxs],
        'z_src':truth_vals_dict['z_src'][dbls_idxs],
    }

    data_vector_dict_quads = {   
        'td_measured':td_meas_quads,
        'td_likelihood_prec':td_meas_prec_quads,
        'td_likelihood_prefactors':td_gaussian_prefactor_quads,
        'fpd_samples':fasttdc_fpd_samps_quads,
        'lens_param_samples':fasttdc_lens_samps_quads,
        'z_lens':truth_vals_dict['z_lens'][quads_idxs],
        'z_src':truth_vals_dict['z_src'][quads_idxs],
    }

    return data_vector_dict_dbls, data_vector_dict_quads


def dv_dict_from_completo_extendedCPDF(completo_samples_npy, completo_truth_h5, training_samps_h5,
    chosen_idxs, GROUNDTRUTH_COSMO, td_meas_err_percent=0.05):
    """
    Args: 
        completo_samples_npy (string): 
        completo_truth_h5: i.e. '/Users/smericks/Desktop/StrongLensing/project3/slsim/slsim/TrainingSets/10ktest_seed7.h5'
        training_samps_h5 (string): training examples needed to build understanding of the interim prior
        num_truth (int): number of lenses we're actually using from the truth catalog...
        
    Returns: 
        (dict) data_vector_dict with keys: 
            [ 'td_measured':,
              'td_likelihood_prec':,
              'td_likelihood_prefactors':,
              'fpd_samples':,
              'gamma_pred_samples':,
              'z_lens':,
              'z_src': ]
    """

    # read in from samples file
    # in order: ['deflector_LOG_theta_E','los_gamma1','los_gamma2',
    #   'deflector_LOG_gamma_pl','deflector_e1_mass','deflector_e2_mass',
    #    'deflector_e1_light','deflector_e2_light','deflector_LOG_angular_size',
    #    'deflector_n_sersic',
    #    'source_center_x','source_center_y','source_LOG_angular_size']

    # load in posterior samples
    posterior_samps = np.load(completo_samples_npy) # shape: n_samps, n_lenses n_params
    posterior_samps = np.transpose(posterior_samps,axes=(1,0,2)) # shape: n_lenses, n_samps, n_params
    posterior_samps = posterior_samps[chosen_idxs] # only take certain # of lenses

    truth_vals_dict = load_truth_vals_dict(completo_truth_h5, chosen_idxs, GROUNDTRUTH_COSMO)

    (dbls_idxs, quads_idxs, fasttdc_fpd_samps_dbls, 
        fasttdc_fpd_samps_quads, fasttdc_lens_samps_dbls, 
        fasttdc_lens_samps_quads) = fasttdc_samps_from_posterior_samps(posterior_samps,truth_vals_dict)

    # dbls time-delays
    td_meas_dbls, td_meas_prec_dbls = emulate_measurements(
        np.expand_dims(truth_vals_dict['td_01'][dbls_idxs],axis=1),measurement_error_kmpersec=2. # note this is actually days
    )

    # quads time-delays
    td_meas_quads, td_meas_prec_quads = emulate_measurements(
        np.vstack((truth_vals_dict['td_01'][quads_idxs],
                   truth_vals_dict['td_02'][quads_idxs],
                   truth_vals_dict['td_03'][quads_idxs])).T,
        measurement_error_kmpersec=2. # note this is actually days
    )

    num_fpd_samps = np.shape(fasttdc_fpd_samps_quads)[1]

    # add repeats on 2nd axis for compatibility with importance samples
    td_meas_dbls = np.repeat(td_meas_dbls[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    td_meas_prec_dbls = np.repeat(td_meas_prec_dbls[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    td_meas_quads = np.repeat(td_meas_quads[:, np.newaxis, :],
        num_fpd_samps, axis=1)
    td_meas_prec_quads = np.repeat(td_meas_prec_quads[:, np.newaxis, :, :],
        num_fpd_samps, axis=1)
    
    # add prefactors to track wheter its a 1D vs 3D Gaussian evaluation
    num_td = 1
    td_gaussian_prefactor_dbls = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(td_meas_prec_dbls))) )
    
    num_td = 3
    td_gaussian_prefactor_quads = np.log( (1/(2*np.pi)**(num_td/2)) / 
        np.sqrt(np.linalg.det(np.linalg.inv(td_meas_prec_quads))) )
    
    # TODO: finish implementing training prior stuff...
    # add in log_prob_lens_params_nu_int
    cPDF_param_labels = [
        'deflector_LOG_theta_E','deflector_LOG_gamma_pl'
    ]
    theta_training = np.asarray(load_hdf5_labels(training_samps_h5,cPDF_param_labels))
    nu_int_mus = np.mean(theta_training,axis=0) # condense the 
    nu_int_stddevs = np.std(theta_training,axis=0)
    # normalize to mu=0,stddev=1.
    theta_training -= nu_int_mus
    theta_training /= nu_int_stddevs
    # DOWNSAMPLE FOR FITTING THE KDE
    nu_int_kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    nu_int_kde.fit(theta_training[:5000])  # Note: sklearn uses (n_samples, n_features)


    # isolate the samples of interest [n_lenses,n_fpd_samps,2]
    # TODO: fix hardcoding here...
    log_thetaE_samps = np.log(fasttdc_lens_samps_dbls[:,:,0])# LOG_theta_E
    log_gamma_pl_samps = np.log(fasttdc_lens_samps_dbls[:,:,3]) # LOG_gamma_pl
    # stack them
    log_params_dbls = np.stack([log_thetaE_samps, log_gamma_pl_samps], axis=-1)
    # normalize
    log_params_dbls -= nu_int_mus
    log_params_dbls /= nu_int_stddevs

    # evaluate the log prob. & store it [n_lenses,n_fpd_samps]
    n_lenses, n_fpd_samps, _ = log_params_dbls.shape
    log_params_reshaped = log_params_dbls.reshape(-1, 2)  # reshape to (n_lenses*n_fpd_samps,2)
    print('KDE Score on Dbls')
    tik = time.time()
    log_prob_nu_int_dbls  = nu_int_kde.score_samples(log_params_reshaped)
    tok = time.time()
    print('time to KDE-score dbls: ', tok-tik)
    log_prob_nu_int_dbls = log_prob_nu_int_dbls.reshape(n_lenses, n_fpd_samps) # unravel


    # repeat for quads
    log_thetaE_samps = np.log(fasttdc_lens_samps_quads[:,:,0])# LOG_theta_E
    log_gamma_pl_samps = np.log(fasttdc_lens_samps_quads[:,:,3]) # LOG_gamma_pl
    # stack them
    log_params_quads = np.stack([log_thetaE_samps, log_gamma_pl_samps], axis=-1)
    # normalize
    log_params_quads -= nu_int_mus
    log_params_quads /= nu_int_stddevs

    # evaluate the log prob. & store it [n_lenses,n_fpd_samps]
    n_lenses, n_fpd_samps, _ = log_params_quads.shape
    log_params_reshaped = log_params_quads.reshape(-1, 2)  # reshape to (n_lenses*n_fpd_samps,2)
    print('KDE Score on Quads')
    log_prob_nu_int_quads  = nu_int_kde.score_samples(log_params_reshaped)
    log_prob_nu_int_quads = log_prob_nu_int_quads.reshape(n_lenses, n_fpd_samps) # unravel


    # create the data vector dictionary
    data_vector_dict_dbls = {   
        'td_measured':td_meas_dbls,
        'td_likelihood_prec':td_meas_prec_dbls,
        'td_likelihood_prefactors':td_gaussian_prefactor_dbls,
        'fpd_samples':fasttdc_fpd_samps_dbls,
        'cPDF_param_samples':log_params_dbls,
        'log_prob_cPDF_params_nu_int':log_prob_nu_int_dbls,
        'mu_norm_cPDF_params':nu_int_mus,
        'stddev_norm_cPDF_params':nu_int_stddevs,
        'z_lens':truth_vals_dict['z_lens'][dbls_idxs],
        'z_src':truth_vals_dict['z_src'][dbls_idxs],
    }

    data_vector_dict_quads = {   
        'td_measured':td_meas_quads,
        'td_likelihood_prec':td_meas_prec_quads,
        'td_likelihood_prefactors':td_gaussian_prefactor_quads,
        'fpd_samples':fasttdc_fpd_samps_quads,
        'cPDF_param_samples':log_params_quads,
        'log_prob_cPDF_params_nu_int':log_prob_nu_int_quads,
        'mu_norm_cPDF_params':nu_int_mus,
        'stddev_norm_cPDF_params':nu_int_stddevs,
        'z_lens':truth_vals_dict['z_lens'][quads_idxs],
        'z_src':truth_vals_dict['z_src'][quads_idxs],
    }

    return data_vector_dict_dbls, data_vector_dict_quads


def truth_and_posteriors(completo_samples_npy, completo_truth_h5, chosen_idxs, 
        GROUNDTRUTH_COSMO, td_meas_err_percent=0.05):
    """
    Args: 
        completo_samples_npy (string): 
        completo_truth_h5: i.e. '/Users/smericks/Desktop/StrongLensing/project3/slsim/slsim/TrainingSets/10ktest_seed7.h5'
        chosen_idxs [int]: which lenses we're actually using from the truth catalog...
        
    Returns: 
        (dict) data_vector_dict with keys: 
            [ 'td_measured':,
              'td_likelihood_prec':,
              'td_likelihood_prefactors':,
              'fpd_samples':,
              'gamma_pred_samples':,
              'z_lens':,
              'z_src': ]
    """

    # load in posterior samples
    posterior_samps = np.load(completo_samples_npy) # shape: n_lenses, n_samps, n_params
    posterior_samps = np.transpose(posterior_samps,axes=(1,0,2))
    posterior_samps = posterior_samps[chosen_idxs]

    truth_vals_dict = load_truth_vals_dict(completo_truth_h5, chosen_idxs, GROUNDTRUTH_COSMO)

    (dbls_idxs, quads_idxs, fasttdc_fpd_samps_dbls, 
        fasttdc_fpd_samps_quads, fasttdc_lens_samps_dbls, 
        fasttdc_lens_samps_quads) = fasttdc_samps_from_posterior_samps(posterior_samps,truth_vals_dict)

    return truth_vals_dict, dbls_idxs, quads_idxs, fasttdc_fpd_samps_dbls, fasttdc_fpd_samps_quads