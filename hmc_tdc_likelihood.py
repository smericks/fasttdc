import jax
import jax.numpy as jnp
from Utils.tdc_utils import jax_ddt_from_redshifts, jax_kin_distance_ratio, arcsec_in_rad

# NOTE: Lines 8-133 written with support from GPT-5 / Stanford AI playground 
# queried 01/12/2026

def safe_cholesky(S, jitter=1e-9):
    """Cholesky with jitter for numerical stability."""
    S = (S + S.T) / 2.0
    return jnp.linalg.cholesky(S + jitter * jnp.eye(S.shape[-1]))

def logdet_from_chol(L):
    """log|S| from its Cholesky factor."""
    return 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

def solve_psd(S, b):
    """Solve S x = b for SPD S via Cholesky (supports matrix b)."""
    L = safe_cholesky(S)
    # forward and backward substitution
    y = jax.scipy.linalg.solve_triangular(L, b, lower=True)
    x = jax.scipy.linalg.solve_triangular(L.T, y, lower=False)
    return x

def mvn_logpdf(x, mean, cov):
    """log N(x | mean, cov) for SPD cov."""
    d = x.shape[-1]
    L = safe_cholesky(cov)
    diff = x - mean
    sol = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
    quad = jnp.sum(sol**2)  # (x-μ)^T Σ^{-1} (x-μ)
    logdet = logdet_from_chol(L)
    return -0.5 * (quad + logdet + d * jnp.log(2.0 * jnp.pi))

def obs_gaussian_params(delta_t_obs, sigma_t, delta_v_obs, sigma_v, A, beta):
    """
    Build μ_obs and Σ_obs for x = (Δφ, c√J).
    From Δφ = Δt / A, c√J = δv / β.
    """
    mu_obs = jnp.array([delta_t_obs / A, delta_v_obs / beta])
    Sigma_obs = jnp.diag(jnp.array([(sigma_t / A)**2, (sigma_v / beta)**2]))
    return mu_obs, Sigma_obs

def collapse_ratio_of_gaussians(mu0, S0, mu1, S1, mu2, S2, idx_x=(0,1)):
    """
    Collapse N(y|μ0,S0) * N(y|μ1,S1) / N(y|μ2,S2) into:
    front_factor * N(x | β_x, Σ*_xx),
    where Σ* = (S0^{-1} + S1^{-1} - S2^{-1})^{-1} and
          β = Σ* (S0^{-1} μ0 + S1^{-1} μ1 - S2^{-1} μ2).

    Returns:
      beta_x      : mean on x dims (shape [len(idx_x)])
      Sigma_xx    : covariance on x dims (shape [len(idx_x), len(idx_x)])
      log_front   : log front factor independent of x (depends on μ,S).
    """
    d = mu0.shape[0]

    # Inverse times vector via solves
    S0_inv_mu0 = solve_psd(S0, mu0)
    S1_inv_mu1 = solve_psd(S1, mu1)
    S2_inv_mu2 = solve_psd(S2, mu2)

    # Build "precision" A_inv = S0^{-1} + S1^{-1} - S2^{-1}
    # For stability, compute via solves on standard basis
    I = jnp.eye(d)
    S0_inv = solve_psd(S0, I)
    S1_inv = solve_psd(S1, I)
    S2_inv = solve_psd(S2, I)
    A_inv = S0_inv + S1_inv - S2_inv

    # Σ* = A_inv^{-1}
    Sigma_star = jnp.linalg.solve(A_inv, I)

    # β = Σ* (S0^{-1} μ0 + S1^{-1} μ1 - S2^{-1} μ2)
    rhs = S0_inv_mu0 + S1_inv_mu1 - S2_inv_mu2
    beta = Sigma_star @ rhs

    # log determinants via Cholesky
    L0 = safe_cholesky(S0); logdet_S0 = logdet_from_chol(L0)
    L1 = safe_cholesky(S1); logdet_S1 = logdet_from_chol(L1)
    L2 = safe_cholesky(S2); logdet_S2 = logdet_from_chol(L2)
    Lstar = safe_cholesky(Sigma_star); logdet_Star = logdet_from_chol(Lstar)

    # quadratic terms
    q0 = mu0 @ S0_inv_mu0
    q1 = mu1 @ S1_inv_mu1
    q2 = mu2 @ S2_inv_mu2
    # β^T Σ*^{-1} β = β^T A_inv β
    qstar = beta @ (A_inv @ beta)

    # front factor (2π cancels when keeping normalized Gaussians)
    # log_front = 0.5*(log|Σ*| - log|S0| - log|S1| + log|S2|) - 0.5*(q0 + q1 - q2 - qstar)
    log_front = 0.5 * (logdet_Star - logdet_S0 - logdet_S1 + logdet_S2) \
                - 0.5 * (q0 + q1 - q2 - qstar)

    # Extract x-blocks
    idx_x = jnp.array(idx_x)
    Sigma_xx = Sigma_star[jnp.ix_(idx_x, idx_x)]
    beta_x = beta[idx_x]

    return beta_x, Sigma_xx, log_front

def analytic_inner_loglik(
    delta_t_obs, sigma_t, delta_v_obs, sigma_v,
    A, beta_scale,             # A and β as in the derivation
    mu_model, S_model,         # shape [d], [d,d]
    mu_pop,   S_pop,
    mu_int,   S_int,
    idx_x=(0,1),
    jitter=1e-9,
):
    """
    Computes log I, the analytic inner integral over (Δφ, c√J, s_k, β_ani),
    leaving only dependence on observables, A, β, and the Gaussian modeling terms.

    Returns: scalar log-likelihood contribution (log I).
    """
    # Build the observation Gaussian in x
    mu_obs, S_obs = obs_gaussian_params(
        delta_t_obs, sigma_t, delta_v_obs, sigma_v, A, beta_scale
    )

    # Collapse the ratio/product of modeling Gaussians onto x
    beta_x, Sxx_star, log_front = collapse_ratio_of_gaussians(
        mu_model, S_model, mu_pop, S_pop, mu_int, S_int, idx_x=idx_x
    )

    # Final integral over x: ∫ N(x|μ_obs,S_obs) N(x|β_x,Sxx_star) dx
    # equals N(μ_obs | β_x, S_obs + Sxx_star)
    S_sum = S_obs + Sxx_star
    log_last = mvn_logpdf(mu_obs, beta_x, S_sum)

    return log_front + log_last


def analytic_likelihood_eval(mu_td_meas,cov_td_meas,z_lens_meas,z_src_meas,
        mu_mass_model, cov_mass_model, mu_pop_model, cov_pop_model, 
        mu_in_prior, cov_int_prior, proposed_jax_cosmo,
        lambda_int=1.,kappa_ext=0.,
        mu_kin_meas=None,cov_kin_meas=None):
    """Analytic likelihood evaluation for a single lens

    Args:
        mu_td_meas: Time-delay measurement central value(s) in days
        cov_td_meas: Time-delay measurement covariance matrix in days
        z_lens_meas: Lens redshift
        z_src_meas: Source redshift
        NOTE: these 6 mu/cov are over params: 
            no kin: (delta_phi, lens_params, beta_ani)
            with kin: (delta_phi, c*sqrt(J), lens_params, beta_ani)
        mu_mass_model: Mean of Gaussian posterior from lens mass model
        cov_mass_model: Covariance matrix of Gaussian posterior from lens mass model
        mu_pop_model: Mean of Gaussian population model 
        cov_pop_model: Covariance of Gaussian population model
        mu_int_prior: Mean of Gaussian interim modeling prior
        cov_int_prior: Covariance matrix of Gaussian interim modeling prior
        proposed_jax_cosmo (jax-cosmo Cosmology): Proposed cosmology for which to evaluate the likelihood
        lambda_int (optional): Internal mass sheet parameter, default=1.
        kappa_ext (optional): External convergence value, default=0.
        mu_kin: (optional) Kinematic velocity dispersion measurement central value(s)
        cov_kin: (optional) Kinematic velocity dispersion covariance matrix
    """

    # compute Ddt, Dds/Ds from proposed cosmology using jax_cosmo
    Ddt = jax_ddt_from_redshifts(proposed_jax_cosmo,z_lens_meas,z_src_meas)
    if mu_kin_meas is not None:
        Dkin_ratio = jax_kin_distance_ratio(proposed_jax_cosmo,z_lens_meas,
            z_src_meas)

    # construct re-scaled covariance matrix so we can evaluate on delta_phi
    # this relies on a constant factor: A_td
    # delta_t = A_td * delta_phi, where A = lambda_int (1-kappa_ext) * Ddt / c
    # p(d_td|...) = p(delta_phi | mu = mu_td/A_td, cov = cov_td / A_td**2)
    c_Mpc_per_day = 8.39429e-10
    A_td = lambda_int * (1-kappa_ext) * Ddt / c_Mpc_per_day 
    # this is in radian^2, and needs to be converted to arcsec^2 for compatability with fermat pot. diff.
    A_td = A_td / (arcsec_in_rad**2)

    mu_obs_td = mu_td_meas / A_td
    cov_obs_td = cov_td_meas / (A_td**2)
    
    # if kin also provided, stack together the mu_obs, cov_obs matrix accordingly
    if mu_kin_meas is not None:

        B_kin_squared = Dkin_ratio * lambda_int * (1 - kappa_ext)
        mu_obs_kin = mu_kin_meas / jnp.sqrt(B_kin_squared)
        cov_obs_kin = cov_kin_meas / B_kin_squared
        
        n_td = jnp.size(mu_obs_td)[0]
        n_kin = jnp.size(mu_obs_kin)[0]
        n_total = n_td + n_kin
        mu_obs = jnp.zeros(n_total)
        # fill in time-delays, then kin.
        mu_obs[:n_td] = mu_obs_td
        mu_obs[-n_kin:] = mu_obs_kin
        cov_obs = jnp.zeros((n_total,n_total))
        # fill in time-delay cov, then kin. cov, keep zeros elsewhere
        cov_obs[:n_td,:n_td] = cov_obs_td
        cov_obs[-n_kin:,-n_kin:] = cov_obs_kin

    # if no kin, just use time-delays
    else:
        mu_obs = mu_obs_td
        cov_obs = cov_obs_td

    # ok, now let's do the inner integration over lens_params and beta_ani
    # this only changes depending on the proposed population model
    beta_x, Sigma_xx, log_front = collapse_ratio_of_gaussians(mu0, S0, mu1, S1, mu2, S2, idx_x=(0,1))

    # then the outer integration over delta_phi, c*sqrt(J)
    # TODO: a collapse ratio of gaussians where its only mu0,mu1 (no mu2)






