import jax_cosmo
import jax.numpy as jnp
import numpy as np


def preprocess_td_measured(td_measured,td_cov,fpd_samples):
    """
    Constructs prefactors, 3d precision matrices, and 3d td_measured such that
        double and quad likelihoods can be evaluated per lens at the same time

    Args:
        td_measured (list of [float]): list of !!variable length!! 1d arrays of 
            time delay measurements for each lens  
        td_cov (list of [float,float]): list of !!variable size!! 2d arrays of 
            time delay covariance matrices (from measuremet error)
        fpd_samples (list of [float,float]): list of !!variable size!! 2d arrays
            of fpd samples. Some will have dim (1,n_samples), others will have dim
            (3,n_samples)

    Returns:
        array of td_measured_padded (n_lenses,3)
        array of fpd_samples_padded (n_lenses,n_fpd_samples,3)
        array of prefactors: (1/(2pi)^k/2) * 1/sqrt(det(Sigma))
            - doubles: k=1,det(Sigma)=det([sigma^2])
            - quads: k=3, det(Sigma)=det(Sigma)
        array of precision matrices: 
            - doubles: ((1/sigma^2 0 0 ),(0 0 0),(0 0 0 )
            - quads: (1/sigma^2 0 0), (0 1/sigma^2 0), (0 0 0)
    """
    
    num_lenses = len(td_measured)
    num_fpd_samples = len(fpd_samples[0][0])
    # HARDCODED TO 3 (max we use are quads)
    td_measured_padded = np.empty((num_lenses,3))
    fpd_samples_padded = np.empty((num_lenses,num_fpd_samples,3))
    td_likelihood_prefactors = np.empty((num_lenses))
    td_likelihood_prec = np.empty((num_lenses,3,3))

    # I'm not sure if I can use indexing for conditioning b/c of the variable length
    for l in range(0,num_lenses):
        num_td = len(td_measured[l])
        # doubles
        if num_td ==1:
            td_measured_padded[l] = [td_measured[l][0],0,0]
            td_likelihood_prec[l] = np.asarray([[1/td_cov[l][0][0],0,0],
                [0,0,0],[0,0,0]])
            fpd_samples_padded[l] = np.asarray([fpd_samples[l][0],
                np.zeros((num_fpd_samples)),np.zeros((num_fpd_samples))]).T
        # quads
        elif num_td == 3:
            td_measured_padded[l] = td_measured[l]
            td_likelihood_prec[l] = np.linalg.inv(np.asarray(td_cov[l]))
            fpd_samples_padded[l] = np.asarray(fpd_samples[l]).T
        else:
            print(("# time delays must be 1 or 3"+ 
                "lens %d has %d time delays"%(l,len(td_measured[l]))))
            raise ValueError
        
        td_likelihood_prefactors[l] = ((1/(2*np.pi))**(num_lenses/2) 
            / np.sqrt(np.linalg.det(td_cov[l])))
        
    return td_measured_padded, fpd_samples_padded, td_likelihood_prefactors, td_likelihood_prec


def fast_TDC(td_measured,td_cov,fpd_pred_samples,gamma_pred_samples):
    """
        td_measured ()

    """

    num_fpd_samples = len(fpd_pred_samples[0][0])

    (td_measured_padded,fpd_samples_padded,td_likelihood_prefactors,
        td_likelihood_prec) = preprocess_td_measured(td_measured,td_cov,fpd_pred_samples)
    
    # pad with a 2nd batch dim for # of fpd samples
    td_measured_padded = np.repeat(td_measured_padded[:, np.newaxis, :],
        num_fpd_samples, axis=1)
    td_likelihood_prefactors = np.repeat(td_likelihood_prefactors[:, np.newaxis],
        num_fpd_samples, axis=1)
    td_likelihood_prec = np.repeat(td_likelihood_prec[:, np.newaxis, :, :],
        num_fpd_samples, axis=1)
    
    def td_likelihood_per_samp(td_pred_samples):
        """
        Args:
            td_pred_samples (n_lenses,n_fpd_samps,3)

        Returns:
            td_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
        """

        x_minus_mu = (td_pred_samples-td_measured_padded)
        x_minus_mu = np.expand_dims(x_minus_mu,axis=-1)
        print('x_minus_mu shape',x_minus_mu.shape)
        print('td_likelihood_prec shape',td_likelihood_prec.shape)
        print('testing')
        np.matmul(td_likelihood_prec,x_minus_mu)
        print('test passed')
        exponent = -0.5*np.matmul(np.transpose(x_minus_mu,axes=(0,1,3,2)),
            np.matmul(td_likelihood_prec,x_minus_mu))

        exponent = np.squeeze(exponent)

        print('exponent shape: ',exponent.shape)


        return td_likelihood_prefactors*np.exp(exponent)
    

    td_pred_samples = fpd_samples_padded

    my_result = td_likelihood_per_samp(td_pred_samples)

    print('desired shape: (3,5)')
    print('my shape: ',my_result.shape)



td_measured = [
    [1],
    [2,3,5],
    [4]
]

td_cov = [
    [[0.2]],
    [[0.2,0.,0.],[0.,0.3,0.],[0.,0.,0.4]],
    [[0.2]]
]

# 5 samples!
fpd_pred_samples = [
    [[1.2,1.1,1.3,0.9,1.3]],
    [
        [2.2,1.8,1.9,2.1,1.8],
        [2.9,2.7,3.3,3.4,3.1],
        [6.,5.1,4.,4.5,5.5]
    ],
    [[4.1,4.2,4.3,4.4,4.5]]
]

fast_TDC(td_measured,td_cov,fpd_pred_samples,None)