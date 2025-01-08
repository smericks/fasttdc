# Helper functions to compute batched fermat potential!
import numpy as np
from lenstronomy.LensModel.Profiles.epl_numba import EPL_numba
import lenstronomy.Util.param_util as param_util


def eplshear_fp_samples(x_im,y_im,lens_model_samps,x_src_samps,
    y_src_samps):
    """Computes fermat potential at image positions (x_im,y_im) for samples from 
        an EPL+Shear lens mass model

    Note: 
        LENS MODEL SAMPLES MUST BE PROVIDED IN A SPECIFIC ORDER!!
            (theta_E, gamma1, gamma2, gamma, e1, e2, center_x, center_y)
    Args:
        x_im ([float]), shape=(n_images,): ra image positions
        y_im ([float]), shape=(n_images,): dec image positions
        lens_model_samps ([float,float]), shape=(n_samps,8): samples from lens 
            model posterior in the order: (theta_E, gamma1, gamma2, gamma, e1, e2, center_x, center_y)
        x_src_samps ([float]), shape=(n_samps,): samples for x_src (ra)
        y_src_samps ([float]), shape=(n_samps,): samples for y_src (dec)


    Returns:
        a list of fp samples size (n_samps,n_images)
    
    """

    n_samps = np.shape(lens_model_samps)[0]
    n_images = np.shape(x_im)[0]

    # TODO: check if x_im,y_im contain nans, nans will cause an infinite loop
    if np.sum(np.isnan(x_im)) != 0:
        print('x_im contains nans')
        raise ValueError
    elif np.sum(np.isnan(y_im)) != 0:
        print('y_im contains nans')
        raise ValueError

    # add batch dim to x_im,y_im
    x_im = np.repeat(x_im[np.newaxis,:],n_samps, axis=0)
    y_im = np.repeat(y_im[np.newaxis,:],n_samps, axis=0)
    
    # add im_pos dimension to lens_model_samps (not repeated yet)
    lens_model_samps = np.expand_dims(lens_model_samps,axis=-1)

    # batched epl_numba calculation
    # lp = lensing potential
    #epl_lp_samps = epl_lp(x=x_im, y=y_im,
    #    theta_E=lens_model_samps[:,0],gamma=lens_model_samps[:,3],
    #    e1=lens_model_samps[:,n_images],e2=lens_model_samps[:,5],
    #    center_x=lens_model_samps[:,6],center_y=lens_model_samps[:,7])

    # repeat gamma1, gamma2 for the image dimension
    gamma1_samps = np.repeat(lens_model_samps[:,1],n_images,axis=-1)
    gamma2_samps = np.repeat(lens_model_samps[:,2],n_images,axis=-1)

    # batched shear calculation
    # citation: https://github.com/lenstronomy/lenstronomy/blob/5144659b9b09e8e6937c845442fea52bd78181c3/lenstronomy/LensModel/Profiles/shear.py#L31
    #shear_lp_samps = (1 / 2.0 * (gamma1_samps * x_im * x_im + 2 * 
    #    gamma2_samps * x_im * y_im - gamma1_samps * y_im * y_im))

    # add & return!
    #lp_samps = epl_lp_samps + shear_lp_samps
    
    # need to add extra dim to x_src/y_src
    x_src = np.expand_dims(x_src_samps,axis=-1)
    x_src = np.repeat(x_src,n_images,axis=-1)
    y_src = np.expand_dims(y_src_samps,axis=-1)
    y_src = np.repeat(y_src,n_images,axis=-1)
    #geometry_samps = ((x_im - x_src) ** 2 + (y_im - y_src) ** 2) / 2.0

    # needs to have final shape: (n_samps,n_images)
    return eplshear_fp(x_im,y_im,theta_E=lens_model_samps[:,0],
        gamma1=gamma1_samps,gamma2=gamma2_samps,gamma=lens_model_samps[:,3],
        e1=lens_model_samps[:,4],e2=lens_model_samps[:,5],
        center_x=lens_model_samps[:,6],center_y=lens_model_samps[:,7],
        src_x=x_src,src_y=y_src)

def eplshear_fp(x_im, y_im, theta_E, gamma1, gamma2, gamma, e1, e2, 
    center_x, center_y, src_x, src_y):
    """
    Computes PEMD+shear fermat potential at x_im,y_im with no batching overhead
    """

    # pemd lensing potential
    lp = epl_lp(x_im, y_im, theta_E, gamma, e1, e2, center_x, center_y)

    # add shear lensing potential
    lp += (1 / 2.0 * (gamma1 * x_im * x_im + 2 * 
        gamma2 * x_im * y_im - gamma1 * y_im * y_im))

    # geometry term
    geometry = ((x_im - src_x) ** 2 + (y_im - src_y) ** 2) / 2.0

    # combine
    return geometry - lp

# HERE DOWN IS COPIED FROM LENSTRONOMY (B/C I NEED TO MAKE A CHANGE IDK HOW TO PUSH YET TO LENSTRONOMY)
# SOURCE: https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/LensModel/Profiles/epl_numba.py
__author__ = "ewoudwempe"

def epl_lp(x, y, theta_E, gamma, e1, e2, center_x=0.0, center_y=0.0):
    """computes EPL lensing potential

    :param x: x-coordinate (angle)
    :param y: y-coordinate (angle)
    :param theta_E: Einstein radius (angle), pay attention to specific definition!
    :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
    :param e1: eccentricity component
    :param e2: eccentricity component
    :param center_x: x-position of lens center
    :param center_y: y-position of lens center
    :return: lensing potential
    """
    z, b, t, q, ang = param_transform(
        x, y, theta_E, gamma, e1, e2, center_x, center_y
    )
    alph = alpha(z.real, z.imag, b, q, t)
    return 1 / (2 - t) * (z.real * alph.real + z.imag * alph.imag)

def epl_derivatives(x, y, theta_E, gamma, e1, e2, center_x=0.0, center_y=0.0):
    """

    :param x: x-coordinate (angle)
    :param y: y-coordinate (angle)
    :param theta_E: Einstein radius (angle), pay attention to specific definition!
    :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
    :param e1: eccentricity component
    :param e2: eccentricity component
    :param center_x: x-position of lens center
    :param center_y: y-position of lens center
    :return: deflection angles alpha_x, alpha_y
    """
    z, b, t, q, ang = param_transform(
        x, y, theta_E, gamma, e1, e2, center_x, center_y
    )
    alph = alpha(z.real, z.imag, b, q, t) * np.exp(1j * ang)
    return alph.real, alph.imag

def epl_hessian(x, y, theta_E, gamma, e1, e2, center_x=0.0, center_y=0.0):
    """

    :param x: x-coordinate (angle)
    :param y: y-coordinate (angle)
    :param theta_E: Einstein radius (angle), pay attention to specific definition!
    :param gamma: logarithmic slope of the power-law profile. gamma=2 corresponds to isothermal
    :param e1: eccentricity component
    :param e2: eccentricity component
    :param center_x: x-position of lens center
    :param center_y: y-position of lens center
    :return: Hessian components f_xx, f_yy, f_xy
    """
    z, b, t, q, ang_ell = param_transform(
        x, y, theta_E, gamma, e1, e2, center_x, center_y
    )
    ang = np.angle(z)
    # r = np.abs(z)
    zz_ell = z.real * q + 1j * z.imag
    R = np.abs(zz_ell)
    phi = np.angle(zz_ell)

    # u = np.minimum(nan_to_num((b/R)**t),1e100)
    u = np.fmin(
        (b / R) ** t, 1e10
    )  # I remove all factors of (b/R)**t to only have to remove nans once.
    # The np.fmin is a regularisation near R=0, to avoid overflows
    # in the magnification calculations
    kappa = (2 - t) / 2
    Roverr = np.sqrt(np.cos(ang) ** 2 * q**2 + np.sin(ang) ** 2)

    Omega = omega(phi, t, q)
    alph = (2 * b) / (1 + q) / b * Omega
    gamma_shear = (
        -np.exp(2j * (ang + ang_ell)) * kappa
        + (1 - t) * np.exp(1j * (ang + 2 * ang_ell)) * alph * Roverr
    )

    f_xx = (kappa + gamma_shear.real) * u
    f_yy = (kappa - gamma_shear.real) * u
    f_xy = gamma_shear.imag * u
    # Fix the nans if x=y=0 is filled in

    return f_xx, f_xy, f_xy, f_yy


def param_transform(x, y, theta_E, gamma, e1, e2, center_x=0.0, center_y=0.0):
    """Converts the parameters from lenstronomy definitions (as defined in PEMD) to the
    definitions of Tessore+ (2015)"""
    t = gamma - 1
    phi_G, q = param_util.ellipticity2phi_q(e1, e2)
    x_shift = x - center_x
    y_shift = y - center_y
    ang = phi_G
    z = np.exp(-1j * phi_G) * (x_shift + y_shift * 1j)
    return z, theta_E * np.sqrt(q), t, q, ang

def alpha(x, y, b, q, t, Omega=None):
    """Calculates the complex deflection.

    :param x: x-coordinate (angle)
    :param y: y-coordinate (angle)
    :param b: Einstein radius (angle), pay attention to specific definition!
    :param q: axis ratio
    :param t: logarithmic power-law slope. Is t=gamma-1
    :param Omega: If given, use this Omega (to avoid recalculations)
    :return: complex deflection angle
    """
    zz = x * q + 1j * y
    R = np.abs(zz)
    phi = np.angle(zz)
    if Omega is None:
        Omega = omega(phi, t, q)
    # Omega = omega(phi, t, q)
    # TODO: check whether numba is active with np.nan_to_num instead of numba_util.nan_to_num
    alph = (2 * b) / (1 + q) * np.nan_to_num((b / R) ** t * R / b) * Omega
    return alph

def omega(phi, t, q, niter_max=200, tol=1e-16):
    f = (1 - q) / (1 + q)
    omegas = np.zeros_like(phi, dtype=np.complex128)
    # NOTE: THIS IS WHERE THE CHANGE TO LENSTRONOMY OCCURS
    # NOTE: if q=1., this breaks!!!
    if hasattr(f, "__len__"):
        niter = min(
            niter_max, int(np.max(np.log(tol) / np.log(f)))+2
        )
    else:
        niter = min(
            niter_max, int(np.log(tol) / np.log(f)) + 2
        )# The absolute value of each summand is always less than f, hence this limit for the number of iterations.
    Omega = 1 * np.exp(1j * phi)
    fact = -f * np.exp(2j * phi)
    for n in range(1, niter):
        omegas += Omega
        Omega *= (2 * n - (2 - t)) / (2 * n + (2 - t)) * fact
    omegas += Omega
    return omegas
