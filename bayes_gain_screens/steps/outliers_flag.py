import os
import sys
import numpy as np
import pylab as plt
import argparse
from jax import numpy as jnp, random, vmap, jit
from jax.lax import while_loop
import logging
from functools import partial

from bayes_gain_screens.utils import chunked_pmap, inverse_update, windowed_mean

logger = logging.getLogger(__name__)

# from bayes_gain_screens.plotting import animate_datapack, add_colorbar_to_axes

from h5parm import DataPack
from jax.scipy.special import erf
from h5parm.utils import make_soltab

from jaxns.gaussian_process.kernels import RBF


def get_data(solution_file):
    logger.info("Getting DDS5 data.")
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1), ant=slice(54,None), time=slice(0,600), dir=16)
        h.select(**select)
        tec, axes = h.tec
        tec_std, axes = h.weights_tec
        tec = tec[0, ...]
        tec_std = tec_std[0, ...]
        _, times = h.get_times(axes['time'])
        times = times.mjd * 86400.
    return tec, tec_std, times


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


def leave_one_out_outlier_detection(K, Y_obs, uncert,kappa=6.):
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


def single_detect_outliers(uncert, Y_obs, times, kappa=2.5):
    """
    Detect outlier in `Y_obs` using leave-one-out Gaussian processes.

    Args:
        uncert: [M] obs. uncertainty of Y_obs
        Y_obs: [M] observables
        K: [M,M] GP kernel
        kappa: z-score limit of outlier definition.

    Returns:

    """
    M = Y_obs.shape[0]
    Y_smooth = windowed_mean(Y_obs, 10)
    dY = Y_obs-Y_smooth
    outliers = jnp.abs(dY) > 20.

    kernel = RBF()
    l = jnp.mean(jnp.diff(times)) * 10.
    moving_sigma = windowed_mean(jnp.diff(Y_smooth) ** 2, 300)
    sigma2 = 0.5 * moving_sigma / (1. - jnp.exp(-0.5))
    sigma = jnp.sqrt(sigma2)
    sigma = jnp.concatenate([sigma[:1], sigma])
    K = kernel(times[:, None], times[:, None], l, 1.)
    K = (sigma[:, None] * sigma) * K

    mu_star, sigma2_star = predict_f(Y_obs, K, jnp.where(outliers, jnp.inf, uncert))
    sigma2_star = sigma2_star + uncert ** 2
    sigma_star = sigma2_star
    return mu_star, sigma_star, outliers


def detect_outliers(tec_mean, tec_std, times, kappa=6.):
    """
    Detect outliers in tec (in batch)
    Args:
        tec_mean: [N, Nt] tec means
        tec_std: [N, Nt] tec uncert
        times: [Nt]
        kappa: float, the z-score limit defining an outlier

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
        lambda tec_mean, tec_std: single_detect_outliers(tec_std, tec_mean, times, kappa=kappa), tec_mean, tec_std,
    chunksize=1)
    return mu_star.reshape((Nd, Na, Nt)), sigma_star.reshape((Nd, Na, Nt)), outliers.reshape((Nd, Na, Nt))


def debug_outlier_prob():
    from jaxns.gaussian_process.kernels import RBF
    N = 600
    times = jnp.arange(N)*30.
    kernel = RBF()
    K = kernel(times[:,None], times[:, None], 300., 1.) + 1e-6*jnp.eye(N)
    Y = jnp.linalg.cholesky(K) @ random.normal(random.PRNGKey(1234), shape=(N, 1))
    Y = jnp.where((times > 500)&(times<600), Y[:,0]+5., Y[:,0])
    Y = jnp.where((times > 4000)&(times<4100), Y+5., Y)
    Y = Y[None,None,:]
    Y_star, sigma_star, outliers = detect_outliers(Y, 0.1*jnp.ones_like(Y), times, kappa=3.)
    Y = Y[0,0,:]
    Y_star = Y_star[0,0,:]
    sigma_star = sigma_star[0,0,:]
    outliers = outliers[0,0,:]
    plt.plot(times, Y)
    plt.scatter(times[outliers], Y_star[outliers])
    plt.show()
    plt.scatter(times, outliers)
    plt.show()
    plt.plot(times, sigma_star)
    plt.show()


def main(data_dir, working_dir, obs_num, ncpu):
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
    logger.info("Performing outlier flagging with Gaussian processes.")
    dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    tec, tec_std, times = get_data(solution_file=dds5_h5parm)
    times = times - times[0]
    tec_mean, tec_std, outliers = detect_outliers(tec, tec_std, times, kappa=6.)
    # tec_std = jnp.where(outliers, jnp.inf, tec_std)

    # logger.info("Storing outliers")
    # with DataPack(dds5_h5parm, readonly=False) as h:
    #     h.current_solset = 'sol000'
    #     h.select(pol=slice(0, 1, 1))
    #     h.weights_tec = np.asarray(tec_std)[None, ...]
    #     axes = h.axes_tec
    #     patch_names, _ = h.get_directions(axes['dir'])
    #     antenna_labels, _ = h.get_antennas(axes['ant'])

    logger.info("Plotting results.")
    data_plot_dir = os.path.join(working_dir, 'data_plots')
    os.makedirs(data_plot_dir, exist_ok=True)
    Nd, Na, Nt = tec.shape

    for ia in range(Na):
        for id in range(Nd):
            fig, axs = plt.subplots(4, 1, sharex=True)

            # axs[0].plot(times, outlier_prob[id, ia, :], c='black', label='outlier_prob')

            axs[1].scatter(times, outliers[id, ia, :], c='black', label='outliers')

            axs[2].plot(times, tec[id, ia, :], c='black', label='tec')
            axs[2].plot(times, tec_mean[id, ia, :], c='green', ls='dotted', label='tec*')

            axs[3].plot(times, tec_std[id, ia, :], c='black', label='sigma*')

            axs[0].legend()
            axs[1].legend()
            axs[2].legend()
            axs[3].legend()

            axs[0].set_ylabel("Prob")
            axs[1].set_ylabel("Flag")
            axs[2].set_ylabel("DTEC [mTECU]")
            axs[3].set_ylabel("DTEC uncert [mTECU]")
            axs[3].set_xlabel("time [s]")

            fig.savefig(os.path.join(data_plot_dir, 'outliers_ant{:02d}_dir{:02d}.png'.format(ia, id)))
            plt.close("all")

    # animate_datapack(dds5_h5parm, os.path.join(working_dir, 'tec_uncert_plots'), num_processes=ncpu,
    #                  solset='sol000',
    #                  observable='weights_tec', vmin=0., vmax=10., plot_facet_idx=True,
    #                  labels_in_radec=True, plot_crosses=False, phase_wrap=False,
    #                  flag_outliers=False)


def debug_main():
    os.chdir('/home/albert/data/gains_screen/working_dir/')

    main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 342938, 8)
    # main('/home/albert/data/gains_screen/data','/home/albert/data/gains_screen/working_dir/',100000, 8)


def add_args(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument('--obs_num', help='Obs number L*',
                        default=None, type=int, required=True)
    parser.add_argument('--data_dir', help='Where are the ms files are stored.',
                        default=None, type=str, required=True)
    parser.add_argument('--working_dir', help='Where to perform the imaging.',
                        default=None, type=str, required=True)
    parser.add_argument('--ncpu', help='How many processors available.',
                        default=None, type=int, required=True)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
        # debug_outlier_prob()
        # test_inverse_update()
        exit(0)
    parser = argparse.ArgumentParser(
        description='Infer tec, const, clock and smooth the gains.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    flags, unparsed = parser.parse_known_args()
    logger.info("Running with:")
    for option, value in vars(flags).items():
        logger.info("    {} -> {}".format(option, value))
    main(**vars(flags))
