import os
import sys
import pylab as plt
import argparse
from jax import numpy as jnp, random
import logging

from bayes_gain_screens.outlier_detection import detect_outliers

logger = logging.getLogger(__name__)

from h5parm import DataPack


def get_data(solution_file):
    logger.info("Getting DDS5 data.")
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1), ant=slice(54, None), time=slice(0, 600), dir=16)
        h.select(**select)
        tec, axes = h.tec
        tec_std, axes = h.weights_tec
        tec = tec[0, ...]
        tec_std = tec_std[0, ...]
        _, times = h.get_times(axes['time'])
        times = times.mjd * 86400.
    return tec, tec_std, times


def debug_outlier_prob():
    from jaxns.gaussian_process.kernels import RBF
    N = 600
    times = jnp.arange(N) * 30.
    kernel = RBF()
    K = kernel(times[:, None], times[:, None], 300., 1.) + 1e-6 * jnp.eye(N)
    Y = jnp.linalg.cholesky(K) @ random.normal(random.PRNGKey(1234), shape=(N, 1))
    Y = jnp.where((times > 500) & (times < 600), Y[:, 0] + 5., Y[:, 0])
    Y = jnp.where((times > 4000) & (times < 4100), Y + 5., Y)
    Y = Y[None, None, :]
    Y_star, sigma_star, outliers = detect_outliers(Y, 0.1 * jnp.ones_like(Y), times, kappa=3.)
    Y = Y[0, 0, :]
    Y_star = Y_star[0, 0, :]
    sigma_star = sigma_star[0, 0, :]
    outliers = outliers[0, 0, :]
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
    print(jnp.any(jnp.isnan(tec_std)))
    times = times - times[0]
    tec_mean, tec_std, outliers = detect_outliers(tec, tec_std, times)

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

    # main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 342938, 8)
    main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 100000, 8)


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
