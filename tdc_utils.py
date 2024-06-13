
C_kmpersec = 299792
Mpc_in_km = 3.086e+19 #("Mpc in units of km")
arcsec_in_rad = 4.84814e-6 #("arcsec in units of rad")

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


# this is probably wrong!! 
#def ddt_from_td_fpd(delta_t,delta_phi):
#    """
#    Args:
#        delta_t (float): time delay difference in days
#        delta_phi (float): fermat potential difference

#    Returns:
#        ddt (float): time delay distance in Mpc
#    """

#    delta_t_sec = delta_t*24*60*60 #24hrs, 60mins, 60sec

#    return C_kmpersec*delta_t_sec/delta_phi 

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

