import numpy as np
import pandas as pd
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from astropy.cosmology import FlatLambdaCDM
# my own utils
import tdc_utils
import batched_fermatpot

"""
LensCatalog stores ground truth values for a given lens sample
- This class does NOT store any emulated posteriors

Functional Form assumptions:
    lens mass = PEMD (EPL) + Shear
    lens light = Elliptical Sersic

Suggested order of operations: 
- initialize LensCatalog
- populate truth time delays, which will call truth_DDt and truth_fpd in the process
"""


class LensCatalog:
    """
    Stores a ground truth catalog of lensed quasars in a pandas dataframe
    """

    def __init__(self,truth_cosmology=None):
        """
        Args:
            truth_cosmology (astropy cosmology object): if None, defaults to
                FlatLambdaCDM(H0=70.,Om0=0.3).
        """
        # NOTE: HARDCODED PEMD+SHEAR ASSUMPTION
        self.lenstronomy_lens_model = LensModel(['PEMD', 'SHEAR'])
        self.lens_params = ['theta_E', 'gamma1', 'gamma2', 'gamma', 'e1', 'e2', 
            'center_x', 'center_y', 'src_center_x','src_center_y']
        
        # initialize lens_df object
        columns = self.lens_params
        columns.extend(['x_im0','x_im1','x_im2','x_im3',
            'y_im0','y_im1','y_im2','y_im3'])
        self.lens_df = pd.DataFrame(columns=columns)

        # instantiate the cosmology object
        if truth_cosmology is None:
            self.my_cosmology = FlatLambdaCDM(H0=70.,Om0=0.3)
        else:
            self.my_cosmology = truth_cosmology

        # will be overwritten when image positions are computed!
        self.dbls_idxs = None
        self.quads_idxs = None

    def _construct_lenstronomy_kwargs(self,row_idx):
        """
        Args: 
            row_idx (int): lens index for self.dataframe
            type (string): 'truth' or 'pred'.

        Returns: 
            kwargs_lens
        """
        # lenstronomy lens model object
        # TODO: check that the entry for row_idx exists and is not nan!
        lens_row = self.lens_df.iloc[row_idx]
        # list of dicts, one for 'PEMD', one for 'SHEAR'
        kwargs_lens = [
            {
                'theta_E':lens_row['theta_E'],
                'gamma':lens_row['gamma'],
                'e1':lens_row['e1'],
                'e2':lens_row['e2'],
                'center_x':lens_row['center_x'],
                'center_y':lens_row['center_y']
            },
            {
                'gamma1':lens_row['gamma1'],
                'gamma2':lens_row['gamma2'],
                'ra_0':0.,
                'dec_0':0.
            }]
        
        return kwargs_lens
    
    def _single_row_image_positions(self,r,solver=None):
        """
        Args:
            r (int): index of lens to compute for
            solver (lenstronomy.LensEquationSolver): if lens model stays the 
                same, we can pass this instead of re-instantiating many times
        Returns:
            modifies lens_df in place at row r (changes x_im0,...y_im3)
        """
        if solver is None:
            solver = LensEquationSolver(self.lenstronomy_lens_model)
        theta_x, theta_y = solver.image_position_from_source(
                self.lens_df.iloc[r]['src_center_x'],
                self.lens_df.iloc[r]['src_center_y'],
                self._construct_lenstronomy_kwargs(r)
            )
        for i in range(0,len(theta_x)):
            self.lens_df.at[r, 'x_im'+str(i)] = theta_x[i]
            self.lens_df.at[r, 'y_im'+str(i)] = theta_y[i]

    def populate_image_positions(self):
        """Populates image positions in lens_df based on ground truth lens model

        Returns:
            modifies lens_df in place (changes x_im0,...y_im3)

        """
        solver = LensEquationSolver(self.lenstronomy_lens_model)
        for r in range(0,len(self.lens_df)):
            self._single_row_image_positions(r,solver=solver)

        # find the doubles, find the quads, run eplshear_fp_samples twice
        im3 = self.lens_df.loc[:,'x_im3'].to_numpy().astype(float)
        im2 = self.lens_df.loc[:,'x_im2'].to_numpy().astype(float)
        self.dbls_idxs = np.where(np.isnan(im3) & np.isnan(im2))[0]
        self.triples_idxs = np.where(np.isnan(im3) & ~np.isnan(im2))[0]
        self.quads_idxs = np.where(~np.isnan(im3))[0]

    def doubles_indices(self):
        return self.dbls_idxs
    
    def triples_indices(self):
        return self.triples_idxs

    def quads_indices(self):
        return self.quads_idxs

    def populate_fermat_differences(self):
        """Populates ground truth fermat potential differences at image positions
        Args:
        Returns:
            modifies lens_df in place to add fpd_01 (& fpd02,fpd03 for quads)
        """
        # check that image positions exist
        if np.isnan(self.lens_df.loc[0,'x_im0']):
            print('populating images')
            self.populate_image_positions()

        dbls_idxs = self.dbls_idxs
        quads_idxs = self.quads_idxs

        #fill in fpd01 for doubles
        x_im_dbls = self.lens_df.loc[dbls_idxs,['x_im0','x_im1']].to_numpy().astype(float)
        y_im_dbls = self.lens_df.loc[dbls_idxs,['y_im0','y_im1']].to_numpy().astype(float)
        lens_params_dbls = self.lens_df.loc[dbls_idxs,['theta_E', 'gamma1',
            'gamma2', 'gamma', 'e1', 'e2','center_x', 'center_y']].to_numpy().astype(float)
        x_src_dbls = self.lens_df.loc[dbls_idxs,['src_center_x']].to_numpy().astype(float)
        y_src_dbls = self.lens_df.loc[dbls_idxs,['src_center_y']].to_numpy().astype(float)
        for i in range(0,len(dbls_idxs)):
            dbls_fermatpot = batched_fermatpot.eplshear_fp_samples(x_im_dbls[i],
                y_im_dbls[i],[lens_params_dbls[i]],x_src_dbls[i],y_src_dbls[i])
            self.lens_df.loc[dbls_idxs[i],'fpd01'] = dbls_fermatpot[0][0] - dbls_fermatpot[0][1]

        #fill in fpd01,fpd02,fpd03 for quads
        x_im_quads = self.lens_df.loc[quads_idxs,['x_im0','x_im1','x_im2','x_im3']].to_numpy().astype(float)
        y_im_quads = self.lens_df.loc[quads_idxs,['y_im0','y_im1','y_im2','y_im3']].to_numpy().astype(float)
        lens_params_quads = self.lens_df.loc[quads_idxs,['theta_E', 'gamma1',
            'gamma2', 'gamma', 'e1', 'e2','center_x', 'center_y']].to_numpy().astype(float)
        x_src_quads = self.lens_df.loc[quads_idxs,['src_center_x']].to_numpy().astype(float)
        y_src_quads = self.lens_df.loc[quads_idxs,['src_center_y']].to_numpy().astype(float)
        
        for i in range(0,len(quads_idxs)):
            quads_fermatpot = batched_fermatpot.eplshear_fp_samples(x_im_quads[i],
                y_im_quads[i],[lens_params_quads[i]],x_src_quads[i],y_src_quads[i])
            for j in range(1,4):
                column_name = 'fpd0'+str(j)
                self.lens_df.loc[quads_idxs[i], column_name] = (
                    quads_fermatpot[0][0] - quads_fermatpot[0][j])

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


class OM10LensCatalog(LensCatalog):

    def __init__(self,metadata_path,truth_cosmology=None):
        """
        Args:
            metadata_path (string)
        """

         # super init
        super().__init__(truth_cosmology)

        # HARDCODED FOR COMPATABILITY WITH PALTAS metadata.csv FORMAT
        metadata_params = [
            'main_deflector_parameters_theta_E',
            'main_deflector_parameters_gamma1',
            'main_deflector_parameters_gamma2',
            'main_deflector_parameters_gamma',
            'main_deflector_parameters_e1',
            'main_deflector_parameters_e2',
            'main_deflector_parameters_center_x',
            'main_deflector_parameters_center_y',
            'source_parameters_center_x',
            'source_parameters_center_y'
        ]

        df = pd.read_csv(metadata_path)
        for i in range(0,len(metadata_params)):
            self.lens_df[self.lens_params[i]] = df[metadata_params[i]]

        # track other params as well
        self.lens_df['src_mag_app'] = df['source_parameters_mag_app']
        # save lens light info for kinematics
        ll_param_list = [
            'lens_light_parameters_R_sersic',
            'lens_light_parameters_n_sersic',
            'lens_light_parameters_e1','lens_light_parameters_e2',
        ]
        for key in ll_param_list:
            self.lens_df[key] = df[key]
            
        # redshifts!
        self.lens_df['z_lens'] = df['main_deflector_parameters_z_lens']
        self.lens_df['z_src'] = df['source_parameters_z_source']

        # drop not doubles/quads, compute ground truth time delays
        self.populate_truth_time_delays()