import numpy as np


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


def td_log_likelihood(td_measured,td_cov,td_pred):
    """Helper function for time delay cosmography

    Args:
        td_measured ([float]): (n_time_delays)
        td_cov ([float]): (n_time_delays,n_time_delays)
        td_pred ([float]): (n_time_delays,n_samps)
    Returns:
        log_likelihoods (n_samps)

    """
    # TODO: handle edge case of 1d time delay
    # just get rid of the sum?
    if len(td_measured) == 1:

        return (-0.5*((td_measured[0]-td_pred)**2)/(td_cov[0][0])
            - 0.5*np.log(np.linalg.det(td_cov)) 
            - (1/2.)*np.log(2*np.pi))
    
    return (-0.5*np.sum(np.matmul(td_measured-td_pred.T,np.linalg.inv(td_cov))*((td_measured-td_pred.T)),axis=1) 
            - 0.5*np.log(np.linalg.det(td_cov)) 
            - ((np.shape(td_pred)[0])/2.)*np.log(2*np.pi))
