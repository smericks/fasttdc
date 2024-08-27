import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm, truncnorm, uniform
import emcee
import time

# lenstronomy stuff
from astropy.cosmology import FlatLambdaCDM
from colossus.cosmology.cosmology import fromAstropy
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
            lens_idx (int)
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

                measured_td = norm.rvs(loc=truth_td,scale=measurement_error)
                self.lens_df.loc[r,'td0'+str(j+1)+'_measured'] = measured_td
                self.lens_df.loc[r,'td0'+str(j+1)+'_stddev'] = measurement_error

    def Ddt_posterior(self,lens_idx):
        """Infers Ddt posterior for a single lens using _pred, _stddev 
            lens model params
        Args:
            lens_idx (int): row of the lens_df

        Returns:
            samples & weights
        """

        # TODO: Fix this function to handle multiple images!

        # compute fpd samples
        n_samps = int(1e3)
        fpd_samples = self.pred_fpd_samples(lens_idx,n_samps)

        # construct td_measured & td_cov
        td_measured = np.asarray(self.lens_df.loc[lens_idx,['td01_measured',
            'td02_measured','td03_measured']]).astype(np.float32)
        to_idx = 3 - np.sum(np.isnan(td_measured)) # number of tds not nan
        td_measured = td_measured[0:to_idx]

        # construct td_cov based on # of images
        td_uncertainty = self.lens_df.loc[lens_idx,['td01_stddev',
            'td02_stddev','td03_stddev']].astype(np.float32)
        td_uncertainty = td_uncertainty[0:to_idx]
        td_cov = np.diag(td_uncertainty**2)

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
            loglikelihood = tdc_utils.td_log_likelihood(td_measured,td_cov,td_pred)
        
            Ddt_likelihoods = np.append(Ddt_likelihoods, np.mean(np.exp(loglikelihood.astype(np.float32))))

        return Ddt_samps, Ddt_likelihoods
        
        #self.lik_a_cov = np.exp(self.loglik_a_cov)


    def H0_log_likelihood_lens(self,h0_samps,lens_idx):
        """Computes log_likelihood for each h0 samp for one lens

        Args:
            h0_samps ([float]): list of proposed h0 values
            lens_idx (int): row in lens_df

        Returns:
            list of log_likelihoods (one for each h0 samp)
        """

        # get the redshifts
        z_lens = self.lens_df.loc[lens_idx,'z_lens']
        z_src = self.lens_df.loc[lens_idx,'z_src']

        # compute Ddt samples & weights
        ddt_samps, ddt_weights = self.Ddt_posterior(lens_idx)

        # hierArc likelihood object
        my_likelihood = DdtHistLikelihood(z_lens,z_src,ddt_samps,ddt_weights)

        log_likelihoods = np.asarray([])
        for h0 in h0_samps:

            # compute ddt 
            ddt_proposed = tdc_utils.ddt_from_redshifts(FlatLambdaCDM(H0=h0,Om0=0.3),z_lens,z_src)
            # evaluate likelihood
            ll = my_likelihood.log_likelihood(ddt_proposed)
            log_likelihoods = np.append(log_likelihoods,ll)

        return log_likelihoods



    def H0_individual_lens(self,lens_idx):
        """Infers H0 from time delays and lens model params ASSUMING
            FlatLambdaCDM w/ Om0=0.3
        Args:
            lens_idx (int)

        Returns:
            samps, weights for H0
        
        """

        # propose a bunch of H0s
        H_0_samps = uniform.rvs(loc=0,scale=150,size=5000)

        log_likelihood_list = self.H0_log_likelihood_lens(H_0_samps,lens_idx)

        return H_0_samps, np.exp(log_likelihood_list)
    

    def H0_joint_inference(self):
        """
        Uses all the lenses in the sample to infer H0 by simply multiplying 
            likelihoods

        Returns:
            h0 samples, weights for each sample (i.e. likelihoods)
        """

        # propose a bunch of H0s
        # TODO: change to more samples (just debugging)
        num_samps = 5000
        H_0_samps = uniform.rvs(loc=0,scale=200,size=num_samps)

        all_lenses_log_likelihoods = np.empty((len(self.lens_df),num_samps))
        # we already have a function that can compute the likelihood for each h0 samp for each lens!
        for r in range(0,len(self.lens_df)):

            all_lenses_log_likelihoods[r] = self.H0_log_likelihood_lens(H_0_samps,r)
            
        # sum log likelihoods from each lens
        joint_log_likelihood = np.sum(all_lenses_log_likelihoods,axis=0)

        return H_0_samps, np.exp(joint_log_likelihood)


    def H0_gamma_lens_joint_inference(self,nu_int,lens_idxs=None,
            num_emcee_samps=6000):
        """ Joint inference for mu(gamma_lens),sigma(gamma_lens), and H0
            which requires many evaluations of the predicted time delay
        
        Args:
            nu_int (scipy.stats object): stores probability distribution for
                interim lens modeling prior
            lens_idxs ([int]): which lenses to include in the inference. 
                Default=None means use all
            num_emcee_samps (int): # samples during MCMC

        Returns:    
            emcee sampler.chain (n_walkers,n_samples)
        """
        if lens_idxs is None:
            lens_idxs = range(0,len(self.lens_df))
        n_lenses = len(lens_idxs)

        # TODO: hardcoded for quads!
        all_lenses_fpd_samps = np.empty((n_lenses,3,int(1e3)))
        all_lenses_gamma_samps = np.empty((n_lenses,int(1e3)))
        all_lenses_gamma_log_nu_int = np.empty((n_lenses,int(1e3)))

        # loop thru each lens
        tik_setup = time.time()
        for i,r in enumerate(lens_idxs):
            # samples of fermat potential diffs. & corresponding gamma_lens
            fpd_samps,gamma_lens_samps = self.pred_fpd_samples(r,
                n_samps=int(1e3),gamma_lens=True)
            
            gamma_samps_nu_int = nu_int.logpdf(gamma_lens_samps)

            # TODO: fix here to account for not quads!
            if len(fpd_samps) == 3:
                all_lenses_fpd_samps[i] = fpd_samps
            elif len(fpd_samps) == 2:
                all_lenses_fpd_samps[i,0:2] = fpd_samps
                all_lenses_fpd_samps[i,2] = np.nan*np.ones(len(gamma_lens_samps))
            elif len(fpd_samps) ==1:
                all_lenses_fpd_samps[i,0] = fpd_samps
                all_lenses_fpd_samps[i,1:] = np.nan*np.ones((2,len(gamma_lens_samps)))
            all_lenses_gamma_samps[i] = gamma_lens_samps
            all_lenses_gamma_log_nu_int[i] = gamma_samps_nu_int

        # now that we've constructed our list of samples, use those to construct
        # a log posterior function
        tok_setup = time.time()

        print("Time to compute fpd samples: %.3f seconds"%(tok_setup-tik_setup))

        # Timing stuff
        likelihood_times = []

        def h0_gamma_log_prior(hyperparams):
            """
            Assumes ordering: h0, mu(gamma_lens), sigma(gamma_lens)
            """

            if hyperparams[0] < 0 or hyperparams[0] > 150:
                return -np.inf
            elif hyperparams[1] < 1.5 or hyperparams[1] > 2.5:
                return -np.inf
            elif hyperparams[2] < 0.001 or hyperparams[2] > 0.2:
                return -np.inf
            
            return 0

        def h0_gamma_log_likelihood(hyperparams):
            """
            Assumes ordering: h0, mu(gamma_lens), sigma(gamma_lens)
            """
            tik = time.time()
            log_likelihood = 0

            my_cosmo = FlatLambdaCDM(H0=hyperparams[0],Om0=0.3,Ob0=0.049)
            # defaults: 'Ob0': 0.049, 'sigma8': 0.81, 'ns': 0.95            
            #colossus_cosmo = fromAstropy(my_cosmo,sigma8=0.81,ns=0.95,
            #    cosmo_name='my_cosmo',interpolation=True)

            for i,r in enumerate(lens_idxs):
                # compute predicted td
                z_lens = self.lens_df.loc[r,'z_lens']
                z_src = self.lens_df.loc[r,'z_src']
                # remember this returns a quantity not a number
                ddt_proposed = tdc_utils.ddt_from_redshifts(
                    my_cosmo,z_lens,z_src).value
                
                #ddt_proposed = tdc_utils.ddt_from_redshifts_colossus(colossus_cosmo,z_lens,z_src)

                # fpd samps has shape (num_images,num_samples)
                fpd_samps = all_lenses_fpd_samps[i]
                # doubles
                if np.isnan(fpd_samps[1,0]):
                    fpd_samps = fpd_samps[0,:]
                    num_images = 2
                # triples
                elif np.isnan(fpd_samps[1,0]):
                    fpd_samps = fpd_samps[0:2,:]
                    num_images = 3
                else:
                    num_images = 4

                td_pred = tdc_utils.td_from_ddt_fpd(ddt_proposed,
                    fpd_samps)

                # get measured td w/ covariance
                # need to account for # of images
                td_measured = np.asarray(self.lens_df.loc[r,['td01_measured',
                    'td02_measured','td03_measured']])
                to_idx = num_images-1
                td_measured = td_measured[0:to_idx]

                # construct td_cov based on # of images
                td_uncertainty = self.lens_df.loc[r,['td01_stddev',
                    'td02_stddev','td03_stddev']].astype(np.float32)
                td_uncertainty = td_uncertainty[0:to_idx]
                td_cov = np.diag(td_uncertainty**2)

                # TODO: needs to handle quads & doubles @ the same time!
                td_log_likelihood = tdc_utils.td_log_likelihood(td_measured,
                    td_cov,td_pred).astype(np.float64)

                # reweighting factor
                eval_at_nu = norm.logpdf(all_lenses_gamma_samps[i],
                    loc=hyperparams[1],scale=hyperparams[2])
                rw_factor = eval_at_nu - all_lenses_gamma_log_nu_int[i]

                # sum across xi_k samples
                individ_likelihood = np.mean(np.exp(td_log_likelihood+rw_factor))

                # TODO need to deal with this overflow issue another way...
                if individ_likelihood == 0:
                    return -np.inf
                log_individ_likelihood = np.log(individ_likelihood)
                if np.isnan(log_individ_likelihood):
                    return -np.inf

                log_likelihood += log_individ_likelihood

            tok = time.time()
            likelihood_times.append(tok-tik)

            return log_likelihood
        
        def h0_gamma_log_posterior(hyperparams):

            # prior is either 0 or -np.inf
            prior = h0_gamma_log_prior(hyperparams)

            # only evaluate likelihood if within prior range
            if prior == 0:
                return prior+h0_gamma_log_likelihood(hyperparams)
            
            return prior
        

        # ok great, MCMC time!

        # 10 walkers, 3 dimensions
        n_walkers = 10
        sampler = emcee.EnsembleSampler(n_walkers,3,h0_gamma_log_posterior)
        # create initial state
        cur_state = np.empty((10,3))
        # fill h0 initial state
        cur_state[:,0] = uniform.rvs(loc=40,scale=60,size=n_walkers)
        #cur_state[:,0] = norm.rvs(loc=70,scale=5,size=n_walkers)
        # fill mu(gamma_lens) initial state
        cur_state[:,1] = uniform.rvs(loc=1.8,scale=0.4,size=n_walkers)
        #cur_state[:,1] = norm.rvs(loc=2.05,scale=0.02,size=n_walkers)
        # fill sigma(gamma_lens) initial state
        cur_state[:,2] = uniform.rvs(loc=0.01,scale=0.19,size=n_walkers)
        #cur_state[:,2] = norm.rvs(loc=0.1,scale=0.02,size=n_walkers)

        # run mcmc
        tik_mcmc = time.time()
        _ = sampler.run_mcmc(cur_state,nsteps=num_emcee_samps)
        tok_mcmc = time.time()
        print("Avg. Time per MCMC Step: %.3f seconds"%((tok_mcmc-tik_mcmc)/num_emcee_samps))

        print("Avg. Time to Evaluate Likelihood: %.3f seconds"%(np.mean(likelihood_times)))

        return sampler.chain
    

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