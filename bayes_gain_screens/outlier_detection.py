from jax import numpy as jnp, vmap, jit, tree_map
from jax._src.lax.control_flow import while_loop
from jax._src.scipy.special import erf

import numpy as np

from bayes_gain_screens.utils import inverse_update, windowed_mean, chunked_pmap, windowed_nanmean, polyfit


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


def single_detect_outliers(sequence, window, init_outliers=None):
    """
    Detect outlier in by looking for differences between a smoothed signal and the sequence.
    The difference threshold is determined by 3*sigma

    Args:
        sequence: [M] sequence to look for outliers in
        window: int, window size

    Returns:
        sequence of outliers [M]
    """
    if init_outliers is None:
        _sequence = sequence
    else:
        _sequence = jnp.where(init_outliers, jnp.nan, sequence)
    for _ in range(3):
        sequence_smooth = windowed_nanmean(_sequence, window)
        dseq = jnp.abs(sequence - sequence_smooth)
        outliers = (dseq > 3. * jnp.std(dseq))
        _sequence = jnp.where(outliers, jnp.nan, sequence)
    if init_outliers is not None:
        outliers = outliers | init_outliers
    return outliers


def detect_dphase_outliers(dphase):
    """
    Detect outliers in dphase (in batch)
    Args:
        dphase: [Nd, Na, Nf, Nt] tec uncert
        times: [Nt]
    Returns:
        outliers [Nd, Na, Nf, Nt]
    """

    Nd, Na, Nf, Nt = dphase.shape
    dphase = dphase.reshape((Nd * Na * Nf, Nt))
    outliers = jnp.abs(dphase) > 1.
    outliers = outliers | (jnp.abs(dphase) > 5. * jnp.sqrt(jnp.mean(dphase ** 2)))
    print(outliers.sum())
    outliers = chunked_pmap(
        lambda dphase, outliers: single_detect_outliers(dphase, window=15, init_outliers=outliers), dphase, outliers,
        chunksize=None)
    outliers = outliers.reshape((Nd, Na, Nf, Nt))
    print(outliers.sum())
    return outliers


def detect_tec_outliers(times, tec_mean, tec_std):
    """
    Detect outliers in dphase (in batch)
    Args:
        tec: [Nd, Na, Nt] tec uncert
        times: [Nt]
    Returns:
        outliers [Nd, Na,  Nt]
    """

    times, tec_mean, tec_std = jnp.asarray(times), jnp.asarray(tec_mean), jnp.asarray(tec_std)
    Nd, Na, Nt = tec_mean.shape
    tec_mean = tec_mean.reshape((Nd * Na, Nt))
    tec_std = tec_std.reshape((Nd * Na, Nt))
    res = chunked_pmap(
        lambda tec_mean, tec_std: single_detect_tec_outliers(times, tec_mean, tec_std), tec_mean, tec_std,
        chunksize=None)
    res = tree_map(lambda x:x.reshape((Nd, Na, Nt)), res)
    return res


def predict(t, f, t2):
    deg = t.size - 1
    coeffs = polyfit(t, f, deg)
    # print(t, t2, f, coeffs, deg)
    return sum([p * t2 ** (deg - i) for i, p in enumerate(coeffs)])

def filter(times, tec_mean, window=1):
    def _slice(y, i):
        res = []
        for j in range(0, window):
            # [i-window, ..., i-window +window-1]
            # [i-window,..., i-1]
            offset = j - window
            res.append(y[i + offset])
        return jnp.stack(res, axis=0)

    def filter_body(state):
        (i, mod_tec_mean) = state
        _times = _slice(times, i)
        _tec_mean = _slice(mod_tec_mean, i)
        y = predict(_times-_times[0], _tec_mean, times[i]-_times[0])
        y = jnp.where(jnp.abs(y - tec_mean[i]) > 50., y, tec_mean[i])
        y = jnp.where(jnp.abs(y) < jnp.abs(tec_mean[i]), y, tec_mean[i])
        mod_tec_mean = mod_tec_mean.at[i].set(y)# - tec_mean[i]
        return (i + 1, mod_tec_mean)

    _, tec_mean = while_loop(lambda s: s[0] < times.size,
                             filter_body,
                             (jnp.asarray(window), tec_mean))
    return tec_mean


def single_detect_tec_outliers(times, tec_mean, tec_std, window=1):
    y = filter(times, tec_mean, window=window)
    y = filter(times[::-1], y[::-1], window=window)[::-1]
    outliers = (jnp.abs(tec_mean - y) > 1e-5) | (tec_std > 30.)
    return y, outliers
