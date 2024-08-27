from lens_sample import LensSample
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

import pandas as pd
import numpy as np
from scipy.stats import norm

class OM10Sample(LensSample):

    def __init__(self,metadata_path,lens_type='PEMD',truth_cosmology=None):
        """
        Args:
            metadata_path (string): 
            lens_type (string): default is 'PEMD'
            truth_cosmology (astropy.Cosmology object or None):
        """

        # super init
        super().__init__(lens_type,truth_cosmology)

        metadata_params = [
            'main_deflector_parameters_theta_E',
            'main_deflector_parameters_gamma',
            'main_deflector_parameters_e1',
            'main_deflector_parameters_e2',
            'main_deflector_parameters_center_x',
            'main_deflector_parameters_center_y',
            'main_deflector_parameters_gamma1',
            'main_deflector_parameters_gamma2',
            'source_parameters_center_x',
            'source_parameters_center_y'
        ]
        df = pd.read_csv(metadata_path)
        for i in range(0,len(metadata_params)):
            self.lens_df[self.lens_params[i]+'_truth'] = df[metadata_params[i]]

        # redshifts!
        self.lens_df['z_lens'] = df['main_deflector_parameters_z_lens']
        self.lens_df['z_src'] = df['source_parameters_z_source']

        # drop not doubles/quads, compute ground truth time delays
        self._fill_im_positions_time_delays()


    def _fill_im_positions_time_delays(self):

        # single lenstronomy solver object
        solver = LensEquationSolver(self.lenstronomy_lens_model)
        model_type='truth'
        # remove not 2s, 4s
        to_remove = []

        for r in range(0,len(self.lens_df)):
            theta_x, theta_y = solver.image_position_from_source(
                    self.lens_df.iloc[r]['src_center_x_'+model_type],
                    self.lens_df.iloc[r]['src_center_y_'+model_type],
                    self.construct_lenstronomy_kwargs(r,model_type=model_type)
                )
            
            if len(theta_x) not in {2,4}:
                to_remove.append(r)
            else:
                for i in range(0,len(theta_x)):
                    self.lens_df.at[r, 'x_im'+str(i)] = theta_x[i]
                    self.lens_df.at[r, 'y_im'+str(i)] = theta_y[i]

        # remove not 2s, 4s
        self.lens_df.drop(to_remove, inplace=True)
        # TODO: but now the indices are messed up :(
        self.lens_df.reset_index(drop=True, inplace=True)
        self.populate_truth_time_delays()


    def choose_gold_silver(self,min_time_delay=10.,num_gold=20.,num_silver=500.):
        """
        Args:
            min_time_delay (int): cutoff for when to consider feasible to 
                measure light curve
            num_gold (int): # of 'gold' quality lenses

        Returns:
            changes in place self.gold_indices,silver_indices
        """

        # time delay cut
        # TODO: Fix to look @ all tds if a quad
        gold_candidates = self.lens_df[
            (np.abs(self.lens_df['td01']) > 10.) 
            & (self.lens_df['theta_E_truth'] > 0.7)].index
        # make a random choice of 20 out of the candidates
        self.gold_indices = np.random.choice(gold_candidates, 
            size=int(num_gold), replace=False)
        # every one over the time_delay cut NOT in gold_indices
        self.silver_indices = self.lens_df[
            np.abs(self.lens_df['td01']) > 10.].index
        self.silver_indices = np.setdiff1d(self.silver_indices, self.gold_indices)
        if len(self.silver_indices > num_silver):
            self.silver_indices = np.random.choice(self.silver_indices, 
                size=int(num_silver),replace=False)
            
    def populate_modeling_preds(self,gold_modeling_error_dict,
        silver_modeling_error_dict):
        """Populates a _pred and _stddev for each lens param

        Args:
            gold_modeling_error_dict (dict): assumed amt. of modeling error for 
                each param in lens_params, 'gold' quality lenses
            silver_modeling_error_dict (dict): assumed amt. of modeling error 
                for each param in lens_params, 'silver' quality lenses
        """
        sample_list = [self.gold_indices,self.silver_indices]

        for i,param in enumerate(self.lens_params):

            for j,modeling_error_dict in enumerate([gold_modeling_error_dict,
                silver_modeling_error_dict]):

                my_idxs = sample_list[j]
                num_lenses = len(my_idxs)

                stddev = modeling_error_dict[param]
                std_devs = np.ones((num_lenses))*stddev
                means = self.lens_df[param+'_truth'].to_numpy()[my_idxs]

                self.lens_df.loc[my_idxs,param+'_pred'] = norm.rvs(
                    loc=means,scale=std_devs)
                self.lens_df.loc[my_idxs,param+'_stddev'] = std_devs


    def populate_measured_time_delays(self,gold_measurement_error=2,
            silver_measurement_error=5):
        """
        Args:
            gold_measurement_error (float): measurement error in days 
                (interpreted as Gaussian std. dev.)
            silver_measurement_error (float): measurement error in days 
                (interpreted as Gaussian std. dev.)
        """

        sample_list = [self.gold_indices,self.silver_indices]

        for i,measurement_error in enumerate([gold_measurement_error,
            silver_measurement_error]):

            # TODO: switch this to an indexing way?
            for r in sample_list[i]:

                for j in range(0,3):
                    truth_td = self.lens_df.loc[r,'td0'+str(j+1)]
                    if np.isnan(truth_td):
                        self.lens_df.loc[r,'td0'+str(j+1)+'_measured'] = np.nan

                    measured_td = norm.rvs(loc=truth_td,scale=measurement_error)
                    self.lens_df.loc[r,'td0'+str(j+1)+'_measured'] = measured_td
                    self.lens_df.loc[r,'td0'+str(j+1)+'_stddev'] = measurement_error


    def h0_gamma_gold_only(self,nu_int,num_emcee_samps=6000):
        """
        TODO
        """

        return self.H0_gamma_lens_joint_inference(nu_int,
            lens_idxs=self.gold_indices,num_emcee_samps=num_emcee_samps)

        
    def h0_gamma_gold_silver(self,nu_int,num_emcee_samps=6000):
        """
        TODO
        """

        lens_idxs = np.concatenate((self.gold_indices, self.silver_indices))

        return self.H0_gamma_lens_joint_inference(nu_int,lens_idxs=lens_idxs,
            num_emcee_samps=num_emcee_samps)


        
