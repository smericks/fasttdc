import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, truncnorm, uniform
import emcee
import time
import jax_cosmo
import jax.numpy as jnp

# lenstronomy stuff
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology.cosmology import fromAstropy
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LensModel.Profiles.epl_numba import EPL_numba

#hierArc stuff
from hierarc.Likelihood.LensLikelihood.ddt_hist_likelihood import DdtHistLikelihood

import tdc_utils

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

    def __init__(self,lens_type='PEMD',truth_cosmology=None):
        """Assumes there is a Gaussian prediction for each lens param

            lens_type (string): 
            truth_cosmology (astropy cosmology object): if None, defaults to
                FlatLambdaCDM(H0=70.,Om0=0.3).
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
        # Now, let's create an empty dataframe to be filled
        self.lens_df = pd.DataFrame(columns=columns)

        # instantiate the cosmology object
        if truth_cosmology is None:
            self.my_cosmology = FlatLambdaCDM(H0=70.,Om0=0.3)
        else:
            self.my_cosmology = truth_cosmology


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
    
    def sample_lenstronomy_kwargs(self,row_idx):
        """uses _pred and _stddev to sample an instance of lenstronomy kwargs
        from the diagonal Gaussian posterior

        Returns:
            kwargs_lens
        """

        lens_row = self.lens_df.iloc[row_idx]
        model_type = 'pred'
        if self.lens_type == 'PEMD':
            # list of dicts, one for 'PEMD', one for 'SHEAR'
            # TODO: if precision of these is too high, sampling will take forever!!
            kwargs_lens = [
                {
                    'theta_E':norm.rvs(loc=lens_row['theta_E_'+model_type],
                                       scale=lens_row['theta_E_stddev']),
                    'gamma':norm.rvs(loc=lens_row['gamma_'+model_type],
                                       scale=lens_row['gamma_stddev']),
                    'e1':norm.rvs(loc=lens_row['e1_'+model_type],
                                       scale=lens_row['e1_stddev']),
                    'e2':norm.rvs(loc=lens_row['e2_'+model_type],
                                       scale=lens_row['e2_stddev']),
                    'center_x':norm.rvs(loc=lens_row['center_x_'+model_type],
                                       scale=lens_row['center_x_stddev']),
                    'center_y':norm.rvs(loc=lens_row['center_y_'+model_type],
                                       scale=lens_row['center_y_stddev'])
                },
                {
                    'gamma1':norm.rvs(loc=lens_row['gamma1_'+model_type],
                                       scale=lens_row['gamma1_stddev']),
                    'gamma2':norm.rvs(loc=lens_row['gamma2_'+model_type],
                                       scale=lens_row['gamma2_stddev']),
                    'ra_0':0.,
                    'dec_0':0.
                }]
            
        return kwargs_lens

    def populate_redshifts(self):
        """Populates lens and source redshifts. FOR NOW, assumes lens redshift
        centered at 0.5, source redshift centered at 2.
        
        Returns:
            modifies lens_df in place, introduces (z_lens,z_src)
        """

        num_lenses = len(self.lens_df)

        z_lens = truncnorm.rvs(-0.5/0.2,np.inf,loc=0.5,scale=0.2,size=num_lenses)
        z_src = np.empty((num_lenses))
        src_mean = 2.
        src_stddev = 0.4
        for l in range(0,num_lenses):
            # TODO: assumes z_lens < 2.(maybe put in the edge case?)
            min_in_std = (src_mean - z_lens[l])/src_stddev
            z_src[l] = truncnorm.rvs(-min_in_std,np.inf,loc=src_mean,scale=src_stddev)
            
        self.lens_df['z_lens'] = z_lens
        self.lens_df['z_src'] = z_src

    def single_row_image_positions(self,r,model_type='truth',solver=None):
        """
        Args:
            r (int): index of lens to compute for
            model_type (string): 'truth' or 'pred'
            solver (lenstronomy.LensEquationSolver): if lens model stays the 
                same, we can pass this instead of re-instantiating many times
        Returns:
            modifies lens_df in place at row r (changes x_im0,...y_im3)
        """
        if solver is None:
            solver = LensEquationSolver(self.lenstronomy_lens_model)
        theta_x, theta_y = solver.image_position_from_source(
                self.lens_df.iloc[r]['src_center_x_'+model_type],
                self.lens_df.iloc[r]['src_center_y_'+model_type],
                self.construct_lenstronomy_kwargs(r,model_type=model_type)
            )
        for i in range(0,len(theta_x)):
            self.lens_df.at[r, 'x_im'+str(i)] = theta_x[i]
            self.lens_df.at[r, 'y_im'+str(i)] = theta_y[i]

    def populate_image_positions(self):
        """Populates image positions in lens_df based on ground truth lens model

        Returns:
            modifies lens_df in place (changes x_im0,...y_im3)

        """
        model_type = 'truth'
        solver = LensEquationSolver(self.lenstronomy_lens_model)
        for r in range(0,len(self.lens_df)):
            self.single_row_image_positions(r,model_type=model_type,
                solver=solver)

    def fermat_differences(self,lens_idx,lens_kwargs,x_src,y_src):
        """Computes fermat potential differences at image positions

        Args:
            lens_idx (int): lens index (i.e. which row in the dataframe)
            lens_kwargs (dict): lenstronomy LensModel kwargs
            x_src (float): x(ra) coordinate of source position
            y_src (float): y(dec) coordinate of source position
        
        Returns:
            fpd_list ([float]): list of fermat potential differences 
                fpd01(,fpd02,fpd03)
        """

        # check if x_im0 exists, if not, compute im positions for this row
        if np.isnan(self.lens_df.loc[lens_idx,'x_im0']):
            self.single_row_image_positions(lens_idx,
                model_type='truth',solver=None)
            

        fpd_list = []
        zeroth_fp = self.lenstronomy_lens_model.fermat_potential(
                self.lens_df.iloc[lens_idx]['x_im0'],
                self.lens_df.iloc[lens_idx]['y_im0'],
                lens_kwargs,
                x_source=x_src,
                y_source=y_src
            )
        for j in range(1,4):
            if np.isnan(self.lens_df.iloc[lens_idx]['x_im'+str(j)]):
                break
            jth_fp = self.lenstronomy_lens_model.fermat_potential(
                self.lens_df.iloc[lens_idx]['x_im'+str(j)],
                self.lens_df.iloc[lens_idx]['y_im'+str(j)],
                lens_kwargs,
                x_source=x_src,
                y_source=y_src
            )
            fpd_list.append(zeroth_fp - jth_fp)

        return fpd_list
        

    def populate_fermat_differences(self):
        """Populates ground truth fermat potential differences at image positions
        Args:
            model_type (string): 'truth' or 'pred'
        Returns:
            modifies lens_df in place to add fpd_01 (& fpd02,fpd03 for quads)
        """
        model_type='truth'
        for r in range(0,len(self.lens_df)):

            fpd_list = self.fermat_differences(r,
                self.construct_lenstronomy_kwargs(r,model_type),
                self.lens_df.iloc[r]['src_center_x_'+model_type],
                self.lens_df.iloc[r]['src_center_y_'+model_type])
            
            for j in range(0,len(fpd_list)):
                column_name = 'fpd0'+str(j+1)
                self.lens_df.at[r, column_name] = fpd_list[j]
      

    def pred_fpd_samples(self,lens_idx,n_samps=int(1e3),gamma_lens=False):
        """samples lens model params & computes fpd for each sample at the 
            ground truth image positions
        Args:
            lens_idx (int): index in dataframe
            n_samps (int): # of samples
            gamma_lens (bool): Default=False. If True, returns 
                fpd_samps and gamma_lens_samps
        Returns:
            a list of fpd samples size (num_fpd,n_samps)
            (If gamma_lens = True, returns fpd_samps,gamma_lens_samps)
        """

        lens_row = self.lens_df.iloc[lens_idx]

        # check if 2 images or 4 images
        if np.isnan(lens_row['x_im2']):
            num_fpd = 1
        # edge case for triples, but we don't expect this?
        elif np.isnan(lens_row['x_im3']):
            num_fpd = 2
        else:
            num_fpd = 3

        # TODO: consistent naming!
        fpd_samps = np.empty((num_fpd,n_samps))
        if gamma_lens:
            gamma_lens_samps = np.empty((n_samps))
        for s in range(0,n_samps):
            samp_kwargs = self.sample_lenstronomy_kwargs(lens_idx)
            if gamma_lens:
                gamma_lens_samps[s] = samp_kwargs[0]['gamma']
            # sample x_src,y_src
            x_src = norm.rvs(loc=lens_row['src_center_x_pred'],
                scale=lens_row['src_center_x_stddev'])
            y_src = norm.rvs(loc=lens_row['src_center_y_pred'],
                scale=lens_row['src_center_y_stddev'])
            
            fpd_samps[:,s] = self.fermat_differences(lens_idx,samp_kwargs,
                x_src,y_src)

        if gamma_lens:
            return fpd_samps,gamma_lens_samps
        return fpd_samps
    

    def populate_truth_Ddt(self):
        """Populate truth time delay distances (Ddt) using ground truth 
            redshifts & cosmology
        """
    
        truth_Ddt = np.empty((len(self.lens_df)))
        for r in range(0,len(self.lens_df)):
            Ddt = tdc_utils.ddt_from_redshifts(self.my_cosmology,
                self.lens_df.loc[r,'z_lens'],
                self.lens_df.loc[r,'z_src'])
            truth_Ddt[r] = Ddt.value
            
        self.lens_df['Ddt_Mpc_truth'] = truth_Ddt
        
    def populate_truth_time_delays(self):
        """Populate truth time delays using ground truth Ddt 
            (from populate_truth_Ddt()) and ground truth fermat potential 
            differences

        Returns:
            modifies self.lens_df in place to include 'td01' (td02,td03) which
            is the ground truth time delay in days
        """

        # make sure we have Ddt and fpd already populated
        if 'Ddt_Mpc_truth' not in self.lens_df.columns:
            self.populate_truth_Ddt()
        if 'fpd01' not in self.lens_df.columns:
            self.populate_fermat_differences()

        for j in range(0,3):
            self.lens_df['td0'+str(j+1)] = tdc_utils.td_from_ddt_fpd(
                self.lens_df['Ddt_Mpc_truth'],
                self.lens_df['fpd0'+str(j+1)])
            
    def populate_measured_time_delays(self,measurement_error=2):
        """
        Args:
            measurement_error (float): measurement error in days (interpreted as 
                Gaussian std. dev.)
        """
        
        # TODO: switch this to an indexing way?
        for r in range(0,len(self.lens_df)):
            for j in range(0,3):
                truth_td = self.lens_df.loc[r,'td0'+str(j+1)]
                if np.isnan(truth_td):
                    self.lens_df.loc[r,'td0'+str(j+1)+'_measured'] = np.nan
                    self.lens_df.loc[r,'td0'+str(j+1)+'_stddev'] = np.nan

                else:
                    measured_td = norm.rvs(loc=truth_td,scale=measurement_error)
                    self.lens_df.loc[r,'td0'+str(j+1)+'_measured'] = measured_td
                    self.lens_df.loc[r,'td0'+str(j+1)+'_stddev'] = measurement_error

    def tdc_sampler_input(self,lens_idxs,num_fpd_samps=int(1e3)):
        """
        Args:
            lens_idxs ([int]): which lenses to return
            num_fpd_samps (int): default=1e3
        Returns:
            td_measured (n_lenses,3): doubles are padded with nans
            td_cov (n_lenses,3,3): currently diagonal, doubles are padded with Nans on the 
                diagonal 
            fpd_pred_samples (n_lenses,n_images-1,n_fpd_samples): list of 
                !!variable length!! arrays of fpd samples
            gamma_pred_samples (n_lenses,n_samples): list of arrays of power-law slope (gamma) samples

        """
        # just make what's most convenient here and then go back and re-write the preprocess function

        # if not populated yet, fill it in!
        if 'td01_measured' not in self.lens_df.keys():
            self.populate_measured_time_delays()

        td_measured = self.lens_df.loc[lens_idxs, ['td01_measured', 'td02_measured', 'td03_measured']].to_numpy()
        td_cov = np.zeros((len(lens_idxs),3,3))

        for i in range(0,3):
            td_cov[:,i,i] = np.squeeze(self.lens_df.loc[lens_idxs,['td0'+str(i+1)+'_stddev']].to_numpy())**2

        fpd_pred_samples_list = []
        gamma_samples_list = []
        for l in lens_idxs:
            fpd_samps,gamma_lens_samps = self.pred_fpd_samples(l,
                n_samps=num_fpd_samps,gamma_lens=True)
        
            fpd_pred_samples_list.append(fpd_samps)
            gamma_samples_list.append(gamma_lens_samps)

        z_lens = self.lens_df.loc[lens_idxs, ['z_lens']].to_numpy()
        z_src = self.lens_df.loc[lens_idxs, ['z_src']].to_numpy()

        return td_measured,td_cov,z_lens,z_src,fpd_pred_samples_list,gamma_samples_list

    
    def save_lens_df(self,save_path):
        """Saves a copy of the current self.lens_df dataframe to a .csv
        
        Args:
            save_path (string): path of where to write the .csv
        """

        self.lens_df.to_csv(save_path)
    

# make this one inherited too so we don't have to rewrite stuff
class ModeledLensSample(LensSample):

    def __init__(self,y_truth,y_pred,std_pred,lens_type='PEMD',
        param_indices=None,truth_cosmology=None):
        """
        Args:
            y_truth ([n_lenses,n_params]): ground truth lens params
            y_pred ([n_lenses,n_params]): predicted mean, lens params
            std_pred ([n_lenses,n_params]): predicted std. dev., lens params
            lens_type (string): 
            param_indices ([n_params]): If ordering of params provided does not 
                match assumed ordering, use this to translate between your 
                ordering and the ordering assumed here. (see LENS_PARAMS above
                for assumed ordering for each lens_type.)
            truth_cosmology (astropy cosmology object): if None, defaults to
                FlatLambdaCDM(H0=70.,Om0=0.3).
        """

        # super init
        super().__init__(lens_type,truth_cosmology)

        # now we fill the dataframe!
        for i,param in enumerate(self.lens_params):
            idx = i
            if param_indices is not None:
                idx = param_indices[i]
            self.lens_df[param+'_truth'] = y_truth[:,idx]
            self.lens_df[param+'_pred'] = y_pred[:,idx]
            self.lens_df[param+'_stddev'] = std_pred[:,idx]

# inherit from above class, just change how params are populated.
class EmulatedLensSample(LensSample):

    def __init__(self,param_dict,num_lenses,lens_type='PEMD',
        truth_cosmology=None):
        """
        Args:
            param_dict (dict): keys are parameter names, values are scipy.stats 
                callable distributions
            num_lenses (int): # lenses to populate 
            lens_type (string): 
            truth_cosmology (astropy cosmology object): if None, defaults to
                FlatLambdaCDM(H0=70.,Om0=0.3).
        """

        # super init
        super().__init__(lens_type,truth_cosmology)
        # set a random seed for reproducible results
        np.random.seed(seed=233423)
        # now, fill in the lenses
        # now we fill the dataframe!

        # gonna have to do rejection sampling for length of time delay
        num_successes = 0
        while num_successes < num_lenses:
            # construct row
            for i,param in enumerate(self.lens_params):
                self.lens_df.loc[num_successes,param+'_truth'] = param_dict[param].rvs()

            # construct lens kwargs & pull x_src,y_src
            lens_kwargs = self.construct_lenstronomy_kwargs(num_successes)
            # sample x_src,y_src
            x_src = self.lens_df.loc[num_successes,'src_center_x_truth']
            y_src = self.lens_df.loc[num_successes,'src_center_y_truth']
            # compute image positions
            self.single_row_image_positions(num_successes,model_type='truth',
                solver=None)
            # check that actually a lens (i.e. check for single image systems)
            if np.isnan(self.lens_df.loc[num_successes,'x_im1']):
                continue
            # need fpd, redshifts, Ddt
            fpd_list = self.fermat_differences(num_successes,
                lens_kwargs,x_src,y_src)
            # pull redshifts
            z_lens = truncnorm.rvs(-0.5/0.2,np.inf,loc=0.5,scale=0.2)
            src_mean = 2.
            src_stddev = 0.4
            min_in_std = (src_mean - z_lens)/src_stddev
            z_src = truncnorm.rvs(-min_in_std,np.inf,loc=src_mean,scale=src_stddev)
            
            self.lens_df.loc[num_successes,'z_lens'] = z_lens
            self.lens_df.loc[num_successes,'z_src'] = z_src
            Ddt = tdc_utils.ddt_from_redshifts(self.my_cosmology,
                self.lens_df.loc[num_successes,'z_lens'],
                self.lens_df.loc[num_successes,'z_src']).value
            self.lens_df.loc[num_successes,'Ddt_Mpc_truth'] = Ddt
            
            td = tdc_utils.td_from_ddt_fpd(Ddt,np.asarray(fpd_list))

            for j in range(0,len(td)):
                self.lens_df.loc[num_successes,'td0'+str(j+1)] = td[j]

            # if too short, overwrites the row in the next pass of the loop
            # for now, don't use any of the 3s
            if np.max(np.abs(td)) > 2. and len(td)!=2:
                num_successes += 1

    def populate_modeling_preds(self,modeling_error_dict):
        """Populates a _pred and _stddev for each lens param

        Args:
            modeling_error_dict (dict): assumed amt. of modeling error for each
                param in lens_params
        """

        for i,param in enumerate(self.lens_params):

            stddev = modeling_error_dict[param]

            num_lenses = len(self.lens_df)
            std_devs = np.ones((num_lenses))*stddev
            means = self.lens_df[param+'_truth'].to_numpy()
            self.lens_df[param+'_pred'] = norm.rvs(loc=means,scale=std_devs)
            self.lens_df[param+'_stddev'] = std_devs
