import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, truncnorm, uniform

# lenstronomy stuff
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

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

    def __init__(self,y_truth,y_pred,std_pred,lens_type='PEMD',
        param_indices=None,truth_cosmology=None):
        """Assumes there is a Gaussian prediction for each lens param

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

    def populate_image_positions(self):
        """Populates image positions in lens_df based on ground truth lens model

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

    def pred_fpd_samples(self,lens_idx,n_samps=int(1e3)):
        """samples lens model params & computes fpd for each sample at the 
            ground truth image positions
        Returns:
            a list of fpd samples size (num_fpd,n_samps)
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

        for s in range(0,n_samps):
            samp_kwargs = self.sample_lenstronomy_kwargs(lens_idx)
            # sample x_src,y_src
            x_src = norm.rvs(loc=lens_row['src_center_x_pred'],
                scale=lens_row['src_center_x_stddev'])
            y_src = norm.rvs(loc=lens_row['src_center_y_pred'],
                scale=lens_row['src_center_y_stddev'])
            
            fpd_samps[:,s] = self.fermat_differences(lens_idx,samp_kwargs,
                x_src,y_src)

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

                measured_td = norm.rvs(loc=truth_td,scale=measurement_error)
                self.lens_df.loc[r,'td0'+str(j+1)+'_measured'] = measured_td
            


    def Ddt_posterior(self,lens_idx,td_uncertainty=2):
        """Infers Ddt posterior for a single lens using _pred, _stddev 
            lens model params
        Args:
            lens_idx (int): row of the lens_df
            td_uncertainty (float): in days, Gaussian

        Returns:
            samples & weights
        """

        # compute fpd samples
        n_samps = int(1e3)
        fpd_samples = self.pred_fpd_samples(lens_idx,n_samps)

        # construct td_cov based on # of fpd_samps
        td_cov = np.eye(fpd_samples.shape[0])*(td_uncertainty**2)

        td_measured = np.asarray(self.lens_df.loc[lens_idx,['td01_measured','td02_measured','td03_measured']])

        # sample Ddt from prior
        # uniform prior, 0 -> 15,000 Mpc
        Ddt_samps = uniform.rvs(loc=0,scale=15000,size=5000)

        #def Ddt_likelihood():

        # compute weight for each Ddt_samp

        Ddt_likelihoods = np.asarray([])

        for D in Ddt_samps:

            # convert fpd_samples to predicted time delays
            td_pred = tdc_utils.td_from_ddt_fpd(D,fpd_samples)

            # compute log likelihood
            loglikelihood = (-0.5*np.sum(np.matmul(td_measured-td_pred.T,np.linalg.inv(td_cov))*((td_measured-td_pred.T)),axis=1) 
                - 0.5*np.log(np.linalg.det(td_cov)) 
                - ((np.shape(td_pred)[0])/2.)*np.log(2*np.pi))
        
            Ddt_likelihoods = np.append(Ddt_likelihoods, np.mean(np.exp(loglikelihood.astype(np.float32))))

        return Ddt_samps, Ddt_likelihoods
        
        #self.lik_a_cov = np.exp(self.loglik_a_cov)


    def H0_from_lens(self,lens_idx):
        """Infers H0 from time delays and lens model params ASSUMING
            FlatLambdaCDM w/ Om0=0.3
        Args:
            lens_idx (int)

        Returns:
            samps, weights for H0
        
        """

        # get the redshifts
        z_lens = self.lens_df.loc[lens_idx,'z_lens']
        z_src = self.lens_df.loc[lens_idx,'z_src']

        # compute Ddt samples & weights
        ddt_samps, ddt_weights = self.Ddt_posterior(lens_idx,td_uncertainty=2)

        # hierArc likelihood object
        my_likelihood = DdtHistLikelihood(z_lens,z_src,ddt_samps,ddt_weights)

        # propose a bunch of H0s
        H_0_samps = uniform.rvs(loc=40,scale=60,size=5000)

        likelihoods = np.asarray([])
        for h0 in H_0_samps:

            # compute ddt 
            ddt_proposed = tdc_utils.ddt_from_redshifts(FlatLambdaCDM(H0=h0,Om0=0.3),z_lens,z_src)
            # evaluate likelihood
            log_likelihood = my_likelihood.log_likelihood(ddt_proposed)
            likelihoods = np.append(likelihoods,np.exp(log_likelihood))

        return H_0_samps, likelihoods
