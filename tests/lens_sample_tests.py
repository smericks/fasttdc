import unittest
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
import lens_sample

class LensSampleTests(unittest.TestCase):

    def test_im_positions(self):

        # pick a random lensing configuration
        kwargs_lens_params = [{
                'theta_E':1.,
                'gamma':2.,
                'e1':0.01,
                'e2':0.01,
                'center_x':0.,
                'center_y':-0.01
            },
            {
                'ra_0':0.,
                'dec_0':0.,
                'gamma1':-0.03,
                'gamma2':0.
            }
        ]
        src_pos = {
            'center_x':-0.01,
            'center_y':0.
        }

        # compute image positions to compare against
        lens_model = LensModel(['PEMD', 'SHEAR'])
        solver = LensEquationSolver(lens_model)
        theta_x, theta_y = solver.image_position_from_source(
                src_pos['center_x'],
                src_pos['center_y'],
                kwargs_lens_params
        )

        # construct LensSample object with one row
        params = np.asarray([[
            kwargs_lens_params[0]['theta_E'],
            kwargs_lens_params[0]['gamma'],
            kwargs_lens_params[0]['e1'],
            kwargs_lens_params[0]['e2'],
            kwargs_lens_params[0]['center_x'],
            kwargs_lens_params[0]['center_y'],
            kwargs_lens_params[1]['gamma1'],
            kwargs_lens_params[1]['gamma2'],
            src_pos['center_x'],
            src_pos['center_y']
        ]])
        my_lens_sample = lens_sample.LensSample(params,params,params,
            lens_type='PEMD',param_indices=None)

        # populate image positions
        my_lens_sample.compute_image_positions()

        # check that they match what you expect!
        lens_row = my_lens_sample.lens_df.iloc[0]
        for i in range(0,len(theta_x)):
            # assert equality here
            self.assertAlmostEqual(theta_x[i],lens_row['x_im'+str(i)])
            self.assertAlmostEqual(theta_y[i],lens_row['y_im'+str(i)])