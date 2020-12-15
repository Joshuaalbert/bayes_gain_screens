from jax import numpy as jnp, vmap
from jax._src.lax.control_flow import while_loop
from jax._src.scipy.special import erf

from bayes_gain_screens.utils import inverse_update, windowed_mean, chunked_pmap


def leave_one_out_predictive(K, Cinv, Y_obs):
    """
    Compute the leave-one-out conditional predictive GP,
        P(y* | Y\y*, K, uncert) for all y*
        which are given by N(mu_star, sigma_star**2)
    Args:
        K: [M, M] covariance matrix
        Cinv: [M, M] inv(K + uncert**2 * I)
        Y_obs: [M] observables, zero centred

    Returns:
        [M] mu_star
        [M] sigma_star
        [M, M-1, M-1] reduced Cinv per observable.
        [M, M] drop matrix
    """

    def single_log_likelihood(m):
        Cinv_reduced, drop = inverse_update(Cinv, m, return_drop=True)
        kstar_reduced = jnp.take(K[m, :], drop, axis=0)
        JT = Cinv_reduced @ kstar_reduced[:, None]
        sigma2_star = K[m, m] - jnp.sum(kstar_reduced * JT[:, 0])
        Y_obs_reduced = jnp.take(Y_obs, drop, axis=0)
        mu_star = jnp.sum(JT[:, 0] * Y_obs_reduced)
        return mu_star, jnp.sqrt(sigma2_star)

    return vmap(single_log_likelihood)(jnp.arange(K.shape[0]))


def decide_outlier(y_star, mu_star, sigma_star, kappa=5., mode='clip'):
    z = jnp.abs(y_star - mu_star) / sigma_star
    if mode == 'clip':
        return z, z > kappa
    elif mode == 'full':
        z = jnp.maximum(z, kappa)
        ek = erf(-kappa / jnp.sqrt(2))
        ez = erf(-z / jnp.sqrt(2))
        prob = (ek - ez) / (1. + ek)
        return prob, prob > 0.97
    else:
        raise ValueError("Mode {} invalid.".format(mode))


def leave_one_out_outlier_detection(K, Y_obs, uncert, kappa=6.):
    C = K + jnp.diag(uncert ** 2)
    Cinv = jnp.linalg.pinv(C)

    def body(state):
        (done, Y_obs, outliers) = state
        mu_star, sigma_star = leave_one_out_predictive(K, Cinv, Y_obs)
        metric, potential_outliers = decide_outlier(Y_obs, mu_star, sigma_star, kappa=kappa, mode='full')
        done = ~jnp.any(potential_outliers)
        outlier = jnp.argmax(metric)
        chosen_outlier = (jnp.arange(outliers.shape[0]) == outlier) & potential_outliers[outlier]
        outliers = chosen_outlier | outliers
        Y_obs = jnp.where(chosen_outlier, mu_star, Y_obs)
        return (done, Y_obs, outliers)

    init_outliers = jnp.zeros(Y_obs.shape[0], dtype=jnp.bool_)
    (done, _, outliers) = while_loop(lambda state: ~state[0],
                                     body,
                                     (jnp.array(False), Y_obs, init_outliers))
    return outliers


def predict_f(Y_obs, K, uncert):
    """
    Predictive mu and sigma with outliers removed.

    Args:
        Y_obs: [N]
        K: [N,N]
        uncert: [N] outliers encoded with inf

    Returns:
        mu [N]
        sigma [N]
    """
    # (K + sigma.sigma)^-1 = sigma^-1.(sigma^-1.K.sigma^-1 + I)^-1.sigma^-1
    C = K / (uncert[:, None] * uncert[None, :]) + jnp.eye(K.shape[0])
    JT = jnp.linalg.solve(C, K / uncert[:, None])
    mu_star = JT.T @ (Y_obs / uncert)
    sigma2_star = jnp.diag(K - JT.T @ (K / uncert[:, None]))
    return mu_star, sigma2_star


def single_detect_outliers(uncert, Y_obs, times):
    """
    Detect outlier in `Y_obs` using leave-one-out Gaussian processes.

    Args:
        uncert: [M] obs. uncertainty of Y_obs
        Y_obs: [M] observables
        K: [M,M] GP kernel

    Returns:
        mu_star [M] predictive mean excluding outliers
        sigma_star [M] predictive std excluding outliers
        outliers [M] bool, outliers
    """
    M = Y_obs.shape[0]
    _Y_obs = Y_obs
    for _ in range(3):
        Y_smooth = windowed_mean(_Y_obs, 15)
        outliers = (jnp.abs(Y_obs - Y_smooth) > 20.)
        _Y_obs = jnp.where(outliers, Y_smooth, Y_obs)
    return _Y_obs, jnp.where(outliers, 20., uncert), outliers

    # est_uncert = jnp.sqrt(jnp.sum(jnp.where(outliers, 0., (Y_obs - Y_smooth) ** 2)) / jnp.sum(~outliers))
    # kernel = RBF()
    # dt = jnp.mean(jnp.diff(times))
    # l = dt * 10.
    # moving_sigma = windowed_mean(jnp.diff(Y_smooth) ** 2, 250)
    # sigma2 = 0.5 * moving_sigma / (1. - jnp.exp(-0.5*dt/l))
    # sigma = jnp.sqrt(sigma2)
    # sigma = jnp.concatenate([sigma[:1], sigma])
    # K = kernel(times[:, None], times[:, None], l, 1.)
    # K = (sigma[:, None] * sigma) * K
    #
    # mu_star, sigma2_star = predict_f(Y_obs, K, jnp.where(outliers, jnp.inf, est_uncert))
    # sigma2_star = sigma2_star + est_uncert ** 2
    # sigma_star = jnp.sqrt(sigma2_star)
    # return mu_star, sigma_star, outliers


def detect_outliers(tec_mean, tec_std, times):
    """
    Detect outliers in tec (in batch)
    Args:
        tec_mean: [N, Nt] tec means
        tec_std: [N, Nt] tec uncert
        times: [Nt]
    Returns:
        mu_star mean tec after outlier selection
        sigma_star uncert in tec after outlier selection
        outliers outliers
    """
    # kernel = RBF()
    # l = jnp.mean(jnp.diff(times)) * 5.
    # moving_sigma = windowed_mean(jnp.diff(tec_mean) ** 2, 40)
    # sigma2 = 0.5 * moving_sigma / (1. - jnp.exp(-0.5))
    # sigma = jnp.sqrt(sigma2)
    # sigma = jnp.concatenate([sigma[:1], sigma])
    # print(l, sigma)
    # K = kernel(times[:, None], times[:, None], l, 1.)
    # K = (sigma[:,None]*sigma) * K

    Nd, Na, Nt = tec_mean.shape
    tec_mean = tec_mean.reshape((Nd * Na, Nt))
    tec_std = tec_std.reshape((Nd * Na, Nt))
    mu_star, sigma_star, outliers = chunked_pmap(
        lambda tec_mean, tec_std: single_detect_outliers(tec_std, tec_mean, times), tec_mean, tec_std,
        chunksize=None)
    # print(jnp.isnan(sigma_star).any())
    tec_mean = jnp.where(outliers, mu_star, tec_mean).reshape((Nd, Na, Nt))
    tec_std = jnp.where(outliers, sigma_star, tec_std).reshape((Nd, Na, Nt))
    outliers = outliers.reshape((Nd, Na, Nt))
    return tec_mean, tec_std, outliers