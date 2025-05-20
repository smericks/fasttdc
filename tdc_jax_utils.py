import jax
import jax.numpy as jnp


@jax.jit
def jax_fpd_samp_summation(log_likelihoood_per_samp):
    """
        log_likelihood_per_samp (n_lenses,n_fpd_samps)
    """
    return jnp.mean(jnp.exp(log_likelihoood_per_samp),axis=1)

@jax.jit
def jax_lenses_summation(individ_likelihood):
    """
        individ_likelihood (n_lenses)
    """
    return jnp.sum(jnp.log(individ_likelihood))

@jax.jit
def jax_td_log_likelihood_per_samp(td_pred_samples, td_measured,
                                    td_likelihood_prec, td_likelihood_prefactors):
    """
    Args:
        td_pred_samples (n_lenses,n_fpd_samps,n_td)
        td_measured (n_lenses,n_fpd_samps,n_td)
        td_likelihood_prec (n_lenses,n_fpd_samps,n_td,n_td)
        td_likelihood_prefactor (n_lenses,n_fpd_samps)

    Returns:
        td_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
    """

    x_minus_mu = (td_pred_samples - td_measured)
    # add dimension s.t. x_minus_mu is 2D
    x_minus_mu = jnp.expand_dims(x_minus_mu, axis=-1)
    # matmul should condense the (# of time delays) dim.
    exponent = -0.5 * jnp.matmul(jnp.transpose(x_minus_mu, axes=(0, 1, 3, 2)),
                                    jnp.matmul(td_likelihood_prec, x_minus_mu))

    # reduce to two dimensions: (n_lenses,n_fpd_samples)
    exponent = jnp.squeeze(exponent)

    # log-likelihood
    return td_likelihood_prefactors + exponent

@jax.jit
def jax_sigma_v_log_likelihood_per_samp(sigma_v_pred_samples ,sigma_v_measured,
                                        sigma_v_likelihood_prec ,sigma_v_likelihood_prefactors):
    """
    Args:
        sigma_v_pred_samples (n_lenses,n_fpd_samps,num_kinbins)
        sigma_v_measured (n_lenses,n_fpd_samps,num_kinbins)
        sigma_v_likelihood_prec (n_lenses,n_fpd_samps,n_kinbins,n_kinbins)
        sigma_v_likelihood_prefactor (n_lenses,n_fpd_samps)

    Returns:
        sigma_v_log_likelihood_per_fpd_samp (n_lenses,n_fpd_samps)
    """

    x_minus_mu = (sigma_v_pred_samples -sigma_v_measured)
    # add dimension s.t. x_minus_mu is 2D
    x_minus_mu = jnp.expand_dims(x_minus_mu ,axis=-1)
    # matmul should condense the (# of time delays) dim.
    exponent = -0.5 *jnp.matmul(jnp.transpose(x_minus_mu ,axes=(0 ,1 ,3 ,2)),
                                jnp.matmul(sigma_v_likelihood_prec ,x_minus_mu))

    # reduce to two dimensions: (n_lenses,n_fpd_samples)
    exponent = jnp.squeeze(exponent)

    # log-likelihood
    return sigma_v_likelihood_prefactors + exponent