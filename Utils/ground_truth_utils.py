import numpy as np
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
from astropy.cosmology import FlatLambdaCDM
# my own utils
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import Utils.tdc_utils as tdc_utils
import Modeling.Kinematics.galkin_utils as galkin_utils
import batched_fermatpot

"""
Modifies a paltas-formatted metadata_df (pandas.DataFrame)
- Assumes image positions are already pre-computed.
"""

def populate_fermat_differences(metadata_df):
    """Populates ground truth fermat potential differences at image positions
    Args:
    Returns:
        modifies metadata_df in place to add fpd_01 (& fpd02,fpd03 for quads)
    """

    num_images = metadata_df.loc[:,'point_source_parameters_num_images'].to_numpy()
    dbls_idxs = np.where(num_images == 2.)[0]
    quads_idxs = np.where(num_images == 4.)[0]

    #fill in fpd01 for doubles
    x_im_dbls = metadata_df.loc[dbls_idxs,
        ['point_source_parameters_x_image_0',
         'point_source_parameters_x_image_1']].to_numpy().astype(float)
    y_im_dbls = metadata_df.loc[dbls_idxs,
        ['point_source_parameters_y_image_0',
         'point_source_parameters_y_image_1']].to_numpy().astype(float)
    lens_params_dbls = metadata_df.loc[dbls_idxs,
        ['main_deflector_parameters_theta_E',
        'main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2', 
        'main_deflector_parameters_gamma', 
        'main_deflector_parameters_e1', 
        'main_deflector_parameters_e2',
        'main_deflector_parameters_center_x', 
        'main_deflector_parameters_center_y']].to_numpy().astype(float)
    x_src_dbls = metadata_df.loc[dbls_idxs,
        ['source_parameters_center_x']].to_numpy().astype(float)
    y_src_dbls = metadata_df.loc[dbls_idxs,
        ['source_parameters_center_y']].to_numpy().astype(float)
    for i in range(0,len(dbls_idxs)):
        dbls_fermatpot = batched_fermatpot.eplshear_fp_samples(x_im_dbls[i],
            y_im_dbls[i],[lens_params_dbls[i]],x_src_dbls[i],y_src_dbls[i])
        # write in new values!
        metadata_df.loc[dbls_idxs[i],'fpd01'] = dbls_fermatpot[0][0] - dbls_fermatpot[0][1]

    #fill in fpd01,fpd02,fpd03 for quads
    x_im_quads = metadata_df.loc[quads_idxs,
        ['point_source_parameters_x_image_0',
         'point_source_parameters_x_image_1',
         'point_source_parameters_x_image_2',
         'point_source_parameters_x_image_3']].to_numpy().astype(float)
    y_im_quads = metadata_df.loc[quads_idxs,
        ['point_source_parameters_y_image_0',
         'point_source_parameters_y_image_1',
         'point_source_parameters_y_image_2',
         'point_source_parameters_y_image_3']].to_numpy().astype(float)
    lens_params_quads = metadata_df.loc[quads_idxs,
        ['main_deflector_parameters_theta_E',
        'main_deflector_parameters_gamma1',
        'main_deflector_parameters_gamma2', 
        'main_deflector_parameters_gamma', 
        'main_deflector_parameters_e1', 
        'main_deflector_parameters_e2',
        'main_deflector_parameters_center_x', 
        'main_deflector_parameters_center_y']].to_numpy().astype(float)
    x_src_quads = metadata_df.loc[quads_idxs,
        ['source_parameters_center_x']].to_numpy().astype(float)
    y_src_quads = metadata_df.loc[quads_idxs,
        ['source_parameters_center_y']].to_numpy().astype(float)
    
    for i in range(0,len(quads_idxs)):
        quads_fermatpot = batched_fermatpot.eplshear_fp_samples(x_im_quads[i],
            y_im_quads[i],[lens_params_quads[i]],x_src_quads[i],y_src_quads[i])
        # write in new values here!!
        for j in range(1,4):
            column_name = 'fpd0'+str(j)
            metadata_df.loc[quads_idxs[i], column_name] = (
                quads_fermatpot[0][0] - quads_fermatpot[0][j])
            

def populate_truth_Ddt_timedelays(metadata_df,gt_cosmo_astropy):
    """Populate truth time delay distances (Ddt) using ground truth 
        redshifts & cosmology. Then, use fpds and Ddts to fill in time-delays.

    Returns: 
        modifies metadata_df in place!
    """

    truth_Ddt = np.empty((len(metadata_df)))
    for r in range(0,len(metadata_df)):
        Ddt = tdc_utils.ddt_from_redshifts(gt_cosmo_astropy,
            metadata_df.loc[r,'main_deflector_parameters_z_lens'],
            metadata_df.loc[r,'source_parameters_z_source'])
        truth_Ddt[r] = Ddt.value
        
    metadata_df['Ddt_Mpc'] = truth_Ddt

    # for doubles, will just write in a nan...
    for j in range(0,3):
        metadata_df['td0'+str(j+1)] = tdc_utils.td_from_ddt_fpd(
            metadata_df['Ddt_Mpc'],
            metadata_df['fpd0'+str(j+1)])
        
def populate_truth_sigma_v_4MOST(metadata_df,gt_cosmo_astropy):
    """Using ground truth lens properties + a ground truth cosmology, computes
        the velocity dispersion in the 4MOST R=0.725" aperture

    Returns:
        modifies metadata_df in place
    """

    for r in range(0,len(metadata_df)):
        sigma_v = galkin_utils.ground_truth_veldisp(
            metadata_df.loc[r,'main_deflector_parameters_theta_E'],
            metadata_df.loc[r,'main_deflector_parameters_gamma'],
            metadata_df.loc[r,'lens_light_parameters_R_sersic'],
            metadata_df.loc[r,'lens_light_parameters_n_sersic'],
            metadata_df.loc[r,'main_deflector_parameters_z_lens'],
            metadata_df.loc[r,'source_parameters_z_source'],
            gt_cosmo_astropy)
        # write in the value!
        metadata_df.loc[r,'sigma_v_4MOST_kmpersec'] = sigma_v