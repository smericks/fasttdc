import numpy as np
#import jax
#jax.config.update("jax_enable_x64", True)
#import jax_cosmo.background as jc_background
#import jax_cosmo.utils as jc_utils

C_kmpersec = 299792
Mpc_in_km = 3.086e+19 #("Mpc in units of km")
arcsec_in_rad = 4.84814e-6 #("arcsec in units of rad")

"""
@jax.jit
def jax_ddt_from_redshifts(my_cosmology,z_lens,z_src):

    Ddt = (1+z_lens) (D_d*D_s)/(D_ds)

    Args:
        my_cosmology (jax-cosmo Cosmology): jax-cosmo cosmology object
        z_lens ([float]): lens redshifts
        z_src ([float]): source redshifts

    Returns:
        ddt (jax array): time delay distances in Mpc 


    # translate redshift to scale factor
    a_lens = jc_utils.z2a(z_lens)
    a_src = jc_utils.z2a(z_src)

    D_d = jc_background.angular_diameter_distance(my_cosmology,a_lens)/my_cosmology.h
    D_s = jc_background.angular_diameter_distance(my_cosmology,a_src)/my_cosmology.h
    # must do angular_diameter_distance_z1z2 by hand
    comoving_ds = (jc_background.radial_comoving_distance(my_cosmology,a_src) -
        jc_background.radial_comoving_distance(my_cosmology,a_lens))
    D_ds = (comoving_ds / (z_src+1.))/my_cosmology.h

    Ddt = (1+z_lens)*D_d*D_s/D_ds

    return Ddt
"""

def ddt_from_redshifts_colossus(my_cosmology,z_lens,z_src):
    """
    Ddt = (1+z_lens) (D_d*D_s)/(D_ds)

    Args:
        my_cosmology (colossus Cosmology): colossus cosmology object
        z_lens (float): lens redshift
        z_src (float): source redshift

    Returns:
        ddt (float): time delay distance in Mpc
    """

    D_d = my_cosmology.angularDiameterDistance(z_lens) / my_cosmology.h
    D_s = my_cosmology.angularDiameterDistance(z_src) / my_cosmology.h
    D_ds = (my_cosmology.comovingDistance(z_min=z_lens, z_max=z_src, transverse=True) / (z_src + 1.0)) / my_cosmology.h

    Ddt = (1+z_lens)*D_d*D_s/D_ds

    return Ddt

def ddt_from_redshifts(my_cosmology,z_lens,z_src):
    """

    Ddt = (1+z_lens) (D_d*D_s)/(D_ds)

    Args:
        my_cosmology (astropy.cosmology.Cosmology): astropy cosmology object
        z_lens (float): lens redshift
        z_src (float): source redshift

    Returns:
        ddt (Quantity): time delay distance in Mpc
    """

    D_d = my_cosmology.angular_diameter_distance(z_lens)
    D_s = my_cosmology.angular_diameter_distance(z_src)
    D_ds = my_cosmology.angular_diameter_distance_z1z2(z_lens, z_src)

    Ddt = (1+z_lens)*D_d*D_s/D_ds

    return Ddt

def td_from_ddt_fpd(Ddt,delta_phi):
    """
    Args:
        Ddt (float): time delay distance in Mpc
        delta_phi (float): fermat potential difference in arcsec^2

    Returns:
        time delay (float): in days
    """

    # convert Ddt to km
    Ddt_km = Ddt*Mpc_in_km
    # convert delta_phi from arcsec^2 to radian^2
    delta_phi_rad = delta_phi*arcsec_in_rad**2
    delta_t_sec = Ddt_km*delta_phi_rad/C_kmpersec

    return delta_t_sec/(24*60*60) #24hrs, 60mins, 60sec

def ddt_from_td_fpd(td,delta_phi):
    """
    Args:
        time delay (float): in days 
        delta_phi (float): fermat potential difference in arcsec^2

    Returns:
        Ddt (float): time delay distance in Mpc
    """

    delta_phi_rad = delta_phi*arcsec_in_rad**2 # arcsec^2 to radian^2
    delta_t_sec = td*(24*60*60) # days to seconds

    Ddt_km = delta_t_sec*C_kmpersec/delta_phi_rad
 
    Ddt = Ddt_km/Mpc_in_km # km to Mpc

    return Ddt


"""
@jax.jit
def jax_kin_distance_ratio(my_cosmology,z_lens,z_src):

    Computes: D_s / D_ds

    Args:
        my_cosmology (jax-cosmo Cosmology): jax-cosmo cosmology object
        z_lens ([float]): lens redshifts
        z_src ([float]): source redshifts

    Returns:
        kin_dist_ratio (jax array): (D_s / D_ds) (unitless)


    # translate redshift to scale factor
    a_lens = jc_utils.z2a(z_lens)
    a_src = jc_utils.z2a(z_src)

    D_s = jc_background.angular_diameter_distance(my_cosmology,a_src)/my_cosmology.h
    # must do angular_diameter_distance_z1z2 by hand
    comoving_ds = (jc_background.radial_comoving_distance(my_cosmology,a_src) -
        jc_background.radial_comoving_distance(my_cosmology,a_lens))
    D_ds = (comoving_ds / (z_src+1.))/my_cosmology.h

    return D_s/D_ds
"""
    
def kin_distance_ratio(my_cosmology,z_lens,z_src):
    """
    Computes: D_s / D_ds

    Args:
        my_cosmology (astropy.cosmology.Cosmology): jax-cosmo cosmology object
        z_lens ([float]): lens redshifts
        z_src ([float]): source redshifts

    Returns:
        kin_dist_ratio (jax array): (D_s / D_ds) (unitless)
    """

    D_s = my_cosmology.angular_diameter_distance(z_src)
    D_ds = my_cosmology.angular_diameter_distance_z1z2(z_lens, z_src)

    return D_s/D_ds