import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# lenstronomy stuff
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.Plots import lens_plot
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

LENS_PARAMS = {
    'PEMD':['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y', 
            'gamma1', 'gamma2',
            'src_center_x','src_center_y']
}

class LensSample:
    """
        self.lens_params: array of strings storing lens model parameter names
        self.lens_df: pandas dataframe storing all relevant info. Each row is 
            a lensing system
    """

    def __init__(self,y_truth,y_pred,std_pred,lens_type='PEMD',
        param_indices=None):
        """Assumes there is a Gaussian prediction for each lens param

            y_truth ([n_lenses,n_params]): ground truth lens params
            y_pred ([n_lenses,n_params]): predicted mean, lens params
            std_pred ([n_lenses,n_params]): predicted std. dev., lens params
            lens_type (string): 
            param_indices ([n_params]): If ordering of params provided does not 
                match assumed ordering, use this to translate between your 
                ordering and the ordering assumed here. (see LENS_PARAMS above
                for assumed ordering for each lens_type.)
        """
        if lens_type not in ['PEMD']:
            print('Supported lens_type values: \'PEMD\'')
            return ValueError
        self.lens_type = lens_type
        self.lens_params = LENS_PARAMS[lens_type]
        if lens_type == 'PEMD':
            self.lenstronomy_lens_model = LensModel(['PEMD', 'SHEAR'])

        # Let's create the column names
        # first, lens mass properties
        columns = []
        for suffix in ['_truth','_pred','_stddev']:
            columns.extend([param + suffix for param in self.lens_params])
        # second, image positions
        im_positions = ['x_im0','x_im1','x_im2','x_im3','y_im0','y_im1','y_im2','y_im3']
        columns.extend(im_positions)

        # now we fill the dataframe!
        self.lens_df = pd.DataFrame(columns=columns)
        for i,param in enumerate(self.lens_params):
            idx = i
            if param_indices is not None:
                idx = param_indices[i]
            self.lens_df[param+'_truth'] = y_truth[:,idx]
            self.lens_df[param+'_pred'] = y_pred[:,idx]
            self.lens_df[param+'_stddev'] = std_pred[:,idx]


    def construct_lenstronomy_kwargs(self,row_idx,model_type='truth'):
        """
        Args: 
            row_idx (int): lens index for self.dataframe
            type (string): 'truth' or 'pred'.

        Returns: 
            kwargs_lens
        """
        # lenstronomy lens model object
        lens_row = self.lens_df.iloc[row_idx]
        if self.lens_type == 'PEMD':
            # list of dicts, one for 'PEMD', one for 'SHEAR'
            kwargs_lens = [
                {
                    'theta_E':lens_row['theta_E_'+model_type],
                    'gamma':lens_row['gamma_'+model_type],
                    'e1':lens_row['e1_'+model_type],
                    'e2':lens_row['e2_'+model_type],
                    'center_x':lens_row['center_x_'+model_type],
                    'center_y':lens_row['center_y_'+model_type]
                },
                {
                    'gamma1':lens_row['gamma1_'+model_type],
                    'gamma2':lens_row['gamma2_'+model_type],
                    'ra_0':0.,
                    'dec_0':0.
                }]
            
        return kwargs_lens
    
    def compute_image_positions(self):
        """Populates image positions in lens_df based on ground truth lens model
        Args:
           
        Returns:
            modifies lens_df in place (changes x_im0,...y_im3)

        """
        model_type = 'truth'
        # TODO: should we instantiate this only once?
        solver = LensEquationSolver(self.lenstronomy_lens_model)
        for r in range(0,len(self.lens_df)):
            theta_x, theta_y = solver.image_position_from_source(
                self.lens_df.iloc[r]['src_center_x_'+model_type],
                self.lens_df.iloc[r]['src_center_y_'+model_type],
                self.construct_lenstronomy_kwargs(r,model_type=model_type)
            )
            for i in range(0,len(theta_x)):
                self.lens_df.at[r, 'x_im'+str(i)] = theta_x[i]
                self.lens_df.at[r, 'y_im'+str(i)] = theta_y[i]



    def compute_fermat_differences(self):
        """
        
        Returns:
            modifies lens_df in place (adds fpd_01,fpd02,fpd03)
        """

        return None
