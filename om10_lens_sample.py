from lens_sample import LensSample

import pandas as pd

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

