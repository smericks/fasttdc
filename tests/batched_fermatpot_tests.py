import unittest
import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
import sys
sys.path.insert(0, '/Users/smericks/Desktop/StrongLensing/darkenergy-from-LAGN/')
from batched_fermatpot import eplshear_fp_samples

class BatchFermatPotTests(unittest.TestCase):

    def setUp(self):

        self.lenstronomy_lm = LensModel(['PEMD', 'SHEAR'])

    def test_eplshear_fp_samples(self):

        # test lens configuration
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

        def test_fp(kwargs_lens_params,src_pos):

            # compute image positions
            solver = LensEquationSolver(self.lenstronomy_lm)
            theta_x, theta_y = solver.image_position_from_source(
                    src_pos['center_x'],
                    src_pos['center_y'],
                    kwargs_lens_params
            )

            while len(theta_x) < 4:
                theta_x = np.append(theta_x,0.)
                theta_y = np.append(theta_y,0.)

            print('im pos: ', theta_x)

            # compute fermat potentials from lenstronomy
            lenst_fp = self.lenstronomy_lm.fermat_potential(
                    theta_x,
                    theta_y,
                    kwargs_lens_params,
                    x_source=src_pos['center_x'],
                    y_source=src_pos['center_y']
                )

            lm_params = [kwargs_lens_params[0]['theta_E'],
                        kwargs_lens_params[1]['gamma1'],
                        kwargs_lens_params[1]['gamma2'],
                        kwargs_lens_params[0]['gamma'],
                        kwargs_lens_params[0]['e1'],
                        kwargs_lens_params[0]['e2'],
                        kwargs_lens_params[0]['center_x'],
                        kwargs_lens_params[0]['center_y']]
            lm_params = np.expand_dims(lm_params,axis=0)
            lm_params = np.repeat(lm_params,2,axis=0)

            print(lm_params.shape)

            x_src_samps = np.asarray([
                src_pos['center_x'], src_pos['center_x']
            ])
            y_src_samps = np.asarray([
                src_pos['center_y'], src_pos['center_y']
            ])

            # compute batched fermat potentials
            batched_fp = eplshear_fp_samples(theta_x,theta_y,
                lm_params,x_src_samps,y_src_samps)

            print('lenstronomy fp: ', lenst_fp)
            print('batched fp: ', batched_fp)

        test_fp(kwargs_lens_params,src_pos)

        # try to make this a double
        kwargs_lens_params[0]['e1'] = 0.02
        kwargs_lens_params[0]['e2'] = 0.
        kwargs_lens_params[1]['gamma1'] = 0.
        kwargs_lens_params[1]['gamma2'] = 0.
        test_fp(kwargs_lens_params,src_pos)