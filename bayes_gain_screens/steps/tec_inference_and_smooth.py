import matplotlib

matplotlib.use('Agg')

import os
import sys
import numpy as np
import pylab as plt
import argparse
from timeit import default_timer
from jax import numpy as jnp, jit, random, vmap
from jax.scipy.special import i0
from jax.lax import scan
import logging
import astropy.units as au

from bayes_gain_screens.utils import chunked_pmap, poly_smooth, wrap, link_overwrite, windowed_mean
from bayes_gain_screens.outlier_detection import detect_outliers

logger = logging.getLogger(__name__)

from bayes_gain_screens.plotting import add_colorbar_to_axes, DatapackPlotter, make_animation
from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, DeterministicTransformPrior, NormalPrior
from jaxns.nested_sampling import NestedSampler

TEC_CONV = -8.4479745e6  # mTECU/Hz

def log_wrapped_laplace(x, mean, uncert):
    dx = wrap(wrap(x) - wrap(mean)) / uncert
    return -jnp.log(2. * uncert) - jnp.abs(dx)

def log_von_mises(x, mu, kappa):
    return kappa * jnp.cos(x-mu) - jnp.log(2.*jnp.pi) - jnp.log(i0(kappa))


def unconstrained_solve(freqs, key, phase_obs):
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_likelihood(tec, const, clock, uncert, **kwargs):
        phase = tec * (TEC_CONV / freqs) + const + clock * (1e-9 * 2. * jnp.pi) * freqs
        return jnp.sum(log_wrapped_laplace(phase, phase_obs, uncert))

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300., 300.)) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(UniformPrior('clock', -1., 1.)) \
        .push(HalfLaplacePrior('uncert', 0.2))

    # prior_chain.test_prior(key, 10000, log_likelihood=log_likelihood)

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       const_mean=lambda const, **kwargs: jnp.concatenate([jnp.cos(const), jnp.sin(const)]),
                       clock_mean=lambda clock, **kwargs: clock,
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 only_marginalise=True,
                 collect_samples=False,
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=2, num_slices=3))

    const_mean = jnp.arctan2(results.marginalised['const_mean'][1], results.marginalised['const_mean'][0])
    clock_mean = results.marginalised['clock_mean']
    tec_mean = results.marginalised['tec_mean']
    return (tec_mean, const_mean, clock_mean)


def constrained_solve(freqs, key, phase_obs, const_mu, clock_mu):
    """
    Perform constrained solve with better priors, including outlier detection, for tec, const, and clock.

    Args:
        freqs: [Nf] frequencies
        key: PRNG key
        Y_obs: [2Nf] obs in real, imag order
        amp: [Nf] smoothed amplitudes
        tec_mean: prior tec mean
        tec_std: prior tec uncert
        const_mu: delta const prior
        clock_mu: delta clock prior

    Returns:
        tec_mean post. tec
        tec_std post. uncert
        phase_mean post. phase
    """

    def log_likelihood(phase, uncert, **kwargs):
        # return jnp.sum(log_von_mises(phase, phase_obs, kappa))
        return jnp.sum(log_wrapped_laplace(phase, phase_obs, uncert))

    def phase_transform(tec):
        return tec * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)

    tec = UniformPrior('tec', -300., 300.)
    uncert = HalfLaplacePrior('uncert', 0.1)
    prior_chain = PriorChain() \
        .push(uncert) \
        .push(DeterministicTransformPrior('phase', phase_transform, phase_obs.shape, tec, tracked=False))

    # prior_chain.test_prior(key, 1000, log_likelihood=log_likelihood)
    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 only_marginalise=True,
                 collect_samples=False,
                 termination_frac=0.001,
                 sampler_kwargs=dict(depth=3, num_slices=3))

    tec_mean = results.marginalised['tec_mean']
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'] - tec_mean ** 2)

    return (tec_mean, tec_std)


def solve_and_smooth(phase_obs, times, freqs):
    Nd, Na, Nf, Nt = phase_obs.shape
    phase_obs = phase_obs.transpose((0, 1, 3, 2)).reshape((Nd * Na * Nt, Nf))  # Nd*Na*Nt, 2*Nf
    # Stop problem with singular evidences
    phase_obs = phase_obs + 0.01 * random.normal(random.PRNGKey(45326), shape=phase_obs.shape)

    logger.info("Number of nan: {}".format(jnp.sum(jnp.isnan(phase_obs))))
    logger.info("Number of inf: {}".format(jnp.sum(jnp.isinf(phase_obs))))
    T = phase_obs.shape[0]
    logger.info("Performing solve for tec, const, clock.")
    keys = random.split(random.PRNGKey(int(1000*default_timer())), T)

    tec_est, const_est, clock_est = chunked_pmap(lambda *args: unconstrained_solve(freqs, *args),
                                                          keys,
                                                          phase_obs,
                                                          debug_mode=False)

    def smooth(y):
        y = y.reshape((Nd * Na, Nt))  # Nd*Na,Nt
        y = chunked_pmap(lambda y: poly_smooth(times, y, deg=3), y).reshape(
            (Nd * Na * Nt,))  # Nd*Na*Nt
        return y

    logger.info("Smoothing and outlier rejection of tec, const, and clock (a weak prior).")
    # Nd*Na*Nt
    clock_mean = smooth(clock_est)
    const_real_mean = smooth(jnp.cos(const_est))
    const_imag_mean = smooth(jnp.sin(const_est))
    const_mean = jnp.arctan2(const_imag_mean, const_real_mean)
    # outlier flagging on tec, to set better priors for constrained solve
    tec_mean = tec_est.reshape((Nd * Na * Nt))

    _tec_mean = tec_mean.reshape((Nd * Na, Nt))
    _const_mean = const_mean.reshape((Nd * Na, Nt))
    _clock_mean = clock_mean.reshape((Nd * Na, Nt))
    _phase_mean = _tec_mean[..., None] * (TEC_CONV / freqs) \
                  + _const_mean[..., None] \
                  + _clock_mean[..., None] * (1e-9 * (2. * jnp.pi * freqs))

    uncert = jnp.sqrt(windowed_mean(jnp.square(phase_obs.reshape((Nd*Na,Nt,Nf)) - _phase_mean), 50, axis=-1))
    #convert to mean-absolute-deviation
    uncert = jnp.sqrt(2./jnp.pi)*uncert.reshape((Nd*Na*Nt,Nf))

    # constrained_solve(freqs,random.split(random.PRNGKey(int(default_timer())), T)[0], phase_obs[0],
    #                  const_mean[0], clock_mean[0] )
    logger.info("Performing tec-only solve, with fixed const and clock.")
    (tec_mean, tec_std) = \
        chunked_pmap(lambda *args: constrained_solve(freqs, *args),
                     random.split(random.PRNGKey(int(default_timer())), T), phase_obs,
                     const_mean, clock_mean,
                     debug_mode=False)

    tec_mean = tec_mean.reshape((Nd, Na, Nt))
    tec_std = tec_std.reshape((Nd, Na, Nt))
    const_mean = const_mean.reshape((Nd, Na, Nt))
    clock_mean = clock_mean.reshape((Nd, Na, Nt))
    tec_est = tec_est.reshape((Nd, Na, Nt))
    const_est = const_est.reshape((Nd, Na, Nt))
    clock_est = clock_est.reshape((Nd, Na, Nt))
    # compute phase before outlier suppression as the smoothed solutions are important.
    phase_mean = tec_mean[..., None, :] * (TEC_CONV / freqs[:, None]) \
                 + const_mean[..., None, :] \
                 + clock_mean[..., None, :] * 1e-9 * (2. * jnp.pi * freqs[:, None])

    logger.info("Performing outlier detection.")
    _, _, outliers = detect_outliers(tec_mean, tec_std, times)

    return phase_mean, tec_mean, tec_std, const_mean, clock_mean, tec_est, const_est, clock_est, outliers


def get_data(solution_file):
    logger.info("Getting DDS4 data.")
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))#, ant=slice(54, None), dir=[44])#, time=slice(0, 600))
        h.select(**select)
        phase, axes = h.phase
        phase = phase[0, ...]
        amp, axes = h.amplitude
        amp = amp[0, ...]
        _, freqs = h.get_freqs(axes['freq'])
        freqs = freqs.to(au.Hz).value
        _, times = h.get_times(axes['time'])
        times = times.mjd * 86400.
        logger.info("Shape: {}".format(phase.shape))

        (Nd, Na, Nf, Nt) = phase.shape

        @jit
        def smooth(amp):
            '''
            Smooth amplitudes
            Args:
                amp: [Nt, Nf]
            '''
            log_amp = jnp.log(amp)
            log_amp = vmap(lambda log_amp: poly_smooth(times, log_amp, deg=3))(log_amp.T).T
            log_amp = vmap(lambda log_amp: poly_smooth(freqs, log_amp, deg=3))(log_amp)
            amp = jnp.exp(log_amp)
            return amp

        logger.info("Smoothing amplitudes")
        amp = chunked_pmap(smooth, amp.reshape((Nd * Na, Nf, Nt)).transpose((0, 2, 1)))  # Nd*Na,Nt,Nf
        amp = amp.transpose((0, 2, 1)).reshape((Nd, Na, Nf, Nt))
    return phase, amp, times, freqs


def prepare_soltabs(dds4_h5parm, dds5_h5parm):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000+clock000")
    make_soltab(dds4_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'clock000', 'tec_outliers000'],
                remake_solset=True,
                to_datapack=dds5_h5parm)


def main(data_dir, working_dir, obs_num, ncpu, plot_results):
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
    logger.info("Performing data smoothing via tec+const+clock inference.")
    dds4_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    dds5_h5parm = os.path.join(working_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    linked_dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    logger.info("Looking for {}".format(dds4_h5parm))
    # link_overwrite(dds5_h5parm, linked_dds5_h5parm)
    prepare_soltabs(dds4_h5parm, dds5_h5parm)
    phase_obs, amp, times, freqs = get_data(solution_file=dds4_h5parm)
    phase_mean, tec_mean, tec_std, const_mean, clock_mean, tec_est, const_est, clock_est, outliers = \
        solve_and_smooth(phase_obs, times, freqs)
    logger.info("Storing smoothed phase, amplitudes, tec, const, and clock")
    with DataPack(dds5_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        h.select(pol=slice(0, 1, 1))#, ant=slice(54, None), dir=[44])#, time=slice(0, 600))
        h.phase = np.asarray(phase_mean)[None, ...]
        h.amplitude = np.asarray(amp)[None, ...]
        h.tec = np.asarray(tec_mean)[None, ...]
        h.tec_outliers = np.asarray(outliers)[None, ...]
        h.weights_tec = np.asarray(tec_std)[None, ...]
        h.const = np.asarray(const_mean)[None, ...]
        h.clock = np.asarray(clock_mean)[None, ...]
        axes = h.axes_phase
        patch_names, _ = h.get_directions(axes['dir'])
        antenna_labels, _ = h.get_antennas(axes['ant'])

    Y_mean = jnp.concatenate([amp * jnp.cos(phase_mean), amp * jnp.sin(phase_mean)], axis=-2)
    Y_obs = jnp.concatenate([amp * jnp.cos(phase_obs), amp * jnp.sin(phase_obs)], axis=-2)

    if plot_results:
        logger.info("Plotting results.")
        data_plot_dir = os.path.join(working_dir, 'data_plots')
        os.makedirs(data_plot_dir, exist_ok=True)
        Nd, Na, Nf, Nt = phase_mean.shape

        for ia in range(Na):
            for id in range(Nd):
                fig, axs = plt.subplots(4, 1, sharex=True)
                # smooth_tec = windowed_mean(tec_mean[id, ia, :], 15)
                axs[0].plot(times, tec_mean[id, ia, :], c='black', label='tec')
                ylim = axs[0].get_ylim()
                axs[0].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                axs[0].plot(times, tec_est[id, ia, :], c='green', ls='dotted', label='tec*')
                axs[0].set_ylim(*ylim)

                axs[1].plot(times, const_mean[id, ia, :], c='black', label='const')
                ylim = axs[1].get_ylim()
                axs[1].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                # axs[1].plot(times, const_est[id, ia, :], c='green', ls='dotted', label='const*')
                axs[1].set_ylim(*ylim)

                axs[2].plot(times, clock_mean[id, ia, :], c='black', label='clock')
                ylim = axs[2].get_ylim()
                axs[2].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                # axs[2].plot(times, clock_est[id, ia, :], c='green', ls='dotted', label='clock*')
                axs[2].set_ylim(*ylim)

                axs[3].plot(times, tec_std[id, ia, :], c='black', label='tec_std')
                ylim = axs[3].get_ylim()
                axs[3].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
                axs[3].plot(times, jnp.abs(tec_mean[id, ia, :] - tec_est[id, ia, :]), c='green', ls='dotted',
                            label='|tec-tec*|')
                axs[3].set_ylim(*ylim)

                axs[0].legend()
                axs[1].legend()
                axs[2].legend()
                axs[3].legend()

                axs[0].set_ylabel("DTEC [mTECU]")
                axs[1].set_ylabel("phase [rad]")
                axs[2].set_ylabel("delay [ns]")
                axs[3].set_ylabel("DTEC uncert [mTECU]")
                axs[3].set_xlabel("time [s]")

                fig.savefig(os.path.join(data_plot_dir, 'sol_ant{:02d}_dir{:02d}.png'.format(ia, id)))
                plt.close("all")

                fig, axs = plt.subplots(3, 1)

                vmin = jnp.percentile(Y_mean[id, ia, :, :], 2)
                vmax = jnp.percentile(Y_mean[id, ia, :, :], 98)

                axs[0].imshow(Y_obs[id, ia, :, :], vmin=vmin, vmax=vmax, cmap='PuOr', aspect='auto',
                              interpolation='nearest')
                axs[0].set_title("Y_obs")
                add_colorbar_to_axes(axs[0], "PuOr", vmin=vmin, vmax=vmax)

                axs[1].imshow(Y_mean[id, ia, :, :], vmin=vmin, vmax=vmax, cmap='PuOr', aspect='auto',
                              interpolation='nearest')
                axs[1].set_title("Y mean")
                add_colorbar_to_axes(axs[1], "PuOr", vmin=vmin, vmax=vmax)

                phase_obs = jnp.arctan2(Y_obs[id, ia, Nf:, :], Y_obs[id, ia, :Nf, :])
                phase = jnp.arctan2(Y_mean[id, ia, Nf:, :], Y_mean[id, ia, :Nf, :])
                dphase = wrap(phase - phase_obs)

                vmin = -0.3
                vmax = 0.3

                axs[2].imshow(dphase, vmin=vmin, vmax=vmax, cmap='coolwarm', aspect='auto',
                              interpolation='nearest')
                axs[2].set_title("diff phase")
                add_colorbar_to_axes(axs[2], "coolwarm", vmin=vmin, vmax=vmax)

                fig.savefig(os.path.join(data_plot_dir, 'gains_ant{:02d}_dir{:02d}.png'.format(ia, id)))
                plt.close("all")
        return
        d = os.path.join(working_dir, 'tec_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds5_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            vmin=-60,
            vmax=60., observable='tec', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'const_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds5_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            vmin=-np.pi,
            vmax=np.pi, observable='const', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'clock_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds5_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            vmin=None,
            vmax=None,
            observable='clock', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)

        d = os.path.join(working_dir, 'amplitude_plots')
        os.makedirs(d, exist_ok=True)
        DatapackPlotter(dds5_h5parm).plot(
            fignames=[os.path.join(d, "fig-{:04d}.png".format(j)) for j in range(Nt)],
            log_scale=True, observable='amplitude', phase_wrap=False, plot_crosses=False,
            plot_facet_idx=True, labels_in_radec=True, per_timestep_scale=True,
            solset='sol000', cmap=plt.cm.PuOr)
        make_animation(d, prefix='fig', fps=4)


def debug_main():
    os.chdir('/home/albert/data/gains_screen/working_dir/')
    main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 342938, 8, True)
    # main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 100000, 1, True)


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
    parser.add_argument('--plot_results', help='Whether to plot results.',
                        default=True, type="bool", required=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        debug_main()
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
