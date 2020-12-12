import os
import sys
import numpy as np
import pylab as plt
import argparse
from timeit import default_timer
from jax import numpy as jnp, jit, random, vmap
from jax.lax import scan
import logging
import astropy.units as au

from bayes_gain_screens.utils import chunked_pmap, poly_smooth, wrap, windowed_mean
from bayes_gain_screens.outlier_detection import detect_outliers

logger = logging.getLogger(__name__)

from bayes_gain_screens.plotting import add_colorbar_to_axes, DatapackPlotter, make_animation
from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, DeterministicTransformPrior, NormalPrior
from jaxns.nested_sampling import NestedSampler

TEC_CONV = -8.4479745e6  # mTECU/Hz


def prepare_soltabs(dds4_h5parm, dds5_h5parm):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000+clock000")
    make_soltab(dds4_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'clock000', 'tec_outliers000'],
                remake_solset=True,
                to_datapack=dds5_h5parm)


def log_laplace(x, mean, uncert):
    dx = (x - mean) / uncert
    return - jnp.log(2. * uncert) - jnp.abs(dx)


def unconstrained_solve(freqs, key, Y_obs, amp):
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_likelihood(tec, const, clock, uncert, **kwargs):
        phase = tec * (TEC_CONV / freqs) + const + clock * (1e-9 * 2. * jnp.pi) * freqs
        return jnp.sum(log_laplace(amp * jnp.cos(phase), Y_obs[:freqs.size], uncert)
                       + log_laplace(amp * jnp.sin(phase), Y_obs[freqs.size:], uncert))

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300., 300.)) \
        .push(UniformPrior('const', -jnp.pi, jnp.pi)) \
        .push(UniformPrior('clock', -1., 1.)) \
        .push(HalfLaplacePrior('uncert', 0.2))

    # prior_chain.test_prior(key, 10000, log_likelihood=log_likelihood)

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2,
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
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'] - tec_mean ** 2)
    return (tec_mean, tec_std, const_mean, clock_mean)


def single_unconstrained_solve(freqs, key, Y_obs):
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    amp = jnp.sqrt(Y_obs[:24] ** 2 + Y_obs[24:] ** 2)

    def log_likelihood(tec, const, clock, uncert, **kwargs):
        phase = tec * (TEC_CONV / freqs) + const + clock * (1e-9 * 2. * jnp.pi) * freqs
        return jnp.sum(log_laplace(amp * jnp.cos(phase), Y_obs[:freqs.size], uncert)
                       + log_laplace(amp * jnp.sin(phase), Y_obs[freqs.size:], uncert))

    prior_chain = PriorChain() \
        .push(UniformPrior('tec', -300., 300.)) \
        .push(UniformPrior('const', -1.5 * jnp.pi, 1.5 * jnp.pi)) \
        .push(UniformPrior('clock', -30., 30.)) \
        .push(HalfLaplacePrior('uncert', 0.2))

    # prior_chain.test_prior(key, 10000, log_likelihood=log_likelihood)

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2,
                       const_mean=lambda const, **kwargs: jnp.concatenate([jnp.cos(const), jnp.sin(const)]),
                       clock_mean=lambda clock, **kwargs: clock,
                       )

    results = ns(key=key,
                 num_live_points=1000,
                 max_samples=1e5,
                 only_marginalise=False,
                 collect_samples=True,
                 termination_frac=0.001,
                 sampler_kwargs=dict(depth=4, num_slices=6))

    from jaxns.plotting import plot_diagnostics, plot_cornerplot

    plot_diagnostics(results)
    plot_cornerplot(results)


def debug_constrained_solve(freqs, key, Y_obs, amp, tec_mean, tec_std, const_mu, clock_mu):
    TEC_CONV = -8.4479745e6  # mTECU/Hz

    def log_likelihood(Y, uncert, **kwargs):
        return jnp.sum(log_laplace(Y, Y_obs, uncert))

    def Y_transform(tec):
        phase = tec * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)])

    tec = NormalPrior('tec', tec_mean, tec_std)
    prior_chain = PriorChain() \
        .push(HalfLaplacePrior('uncert', 0.2)) \
        .push(DeterministicTransformPrior('Y', Y_transform, (freqs.size * 2,), tec))

    def body(state, key):
        U = random.uniform(key, shape=(prior_chain.U_ndims,))
        Y = prior_chain(U)
        isnan = jnp.any(jnp.array([jnp.any(jnp.isnan(v)) for k, v in Y.items()]))
        ll = log_likelihood(**Y)
        return (), (isnan, jnp.isnan(ll))

    _, results = scan(body, (), random.split(key, 1000))
    return jnp.any(results[0]), jnp.any(results[1])


def constrained_solve(freqs, key, Y_obs, amp, tec_mean, tec_std, const_mu, clock_mu):
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

    def log_likelihood(Y, uncert, **kwargs):
        return jnp.sum(log_laplace(Y, Y_obs, uncert))

    def Y_transform(tec):
        phase = tec * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)])

    tec = NormalPrior('tec', tec_mean, tec_std)
    prior_chain = PriorChain() \
        .push(HalfLaplacePrior('uncert', 0.2)) \
        .push(DeterministicTransformPrior('Y', Y_transform, (freqs.size * 2,), tec, tracked=False))

    # prior_chain.test_prior(key, 10000, log_likelihood=log_likelihood)
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
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=2, num_slices=3))

    tec_mean = results.marginalised['tec_mean']
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'] - tec_mean ** 2)
    # tec_samples = resample(key, results.samples, results.log_p, S=1000)['tec']
    # tec_15, tec_50, tec_85 = jnp.percentile(tec_samples, [15, 50, 85])
    # tec_std = 0.5 * (tec_85 - tec_15)
    # tec_mean = tec_50

    return (tec_mean, tec_std)


def get_error_data(Y_obs, amp, freqs):
    indices = [219,
               1813690,
               220,
               2629756,
               1178950,
               1541653,
               181534,
               1904360,
               90848,
               1360313,
               2811110,
               544219,
               2720422,
               1088283,
               997598,
               272209,
               2539083,
               2357735,
               2176385]
    from jax import tree_map
    chunksize = 32
    args = [Y_obs, amp]
    N = args[0].shape[0]
    remainder = N % chunksize
    if (remainder != 0) and (N > chunksize):
        extra = chunksize - remainder
        args = tree_map(lambda arg: jnp.concatenate([arg, arg[:extra]], axis=0), args)
        N = args[0].shape[0]
    # args = tree_map(lambda arg: jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:]), args)
    [Y_obs, amp] = args

    Y_obs = Y_obs[indices, :]
    amp = amp[indices, :]
    for i, y, a in zip(indices, Y_obs, amp):
        print(y.shape)
        fig, axs = plt.subplots(2, 1)
        axs[0].scatter(y[:24], y[24:])
        axs[1].plot(freqs, a)
        axs[0].set_title(f"Index {i}")
        plt.show()


def solve_and_smooth(Y_obs, times, freqs):
    # single_unconstrained_solve(freqs,
    #                     random.PRNGKey(1245524),
    #                     Y_obs[0,57,:,250]
    #                     )
    # return
    #

    Nd, Na, twoNf, Nt = Y_obs.shape
    Nf = twoNf // 2
    Y_obs = Y_obs.transpose((0, 1, 3, 2)).reshape((Nd * Na * Nt, 2 * Nf))  # Nd*Na*Nt, 2*Nf
    amp = jnp.sqrt(Y_obs[:, :freqs.size] ** 2 + Y_obs[:, freqs.size:] ** 2)
    # Stop problem with singular evidences
    Y_obs = Y_obs + 0.01 * random.normal(random.PRNGKey(45326), shape=Y_obs.shape)

    logger.info("Min/max amp: {} {}".format(jnp.min(amp), jnp.max(amp)))
    logger.info("Number of nan: {}".format(jnp.sum(jnp.isnan(Y_obs))))
    logger.info("Number of inf: {}".format(jnp.sum(jnp.isinf(Y_obs))))
    T = Y_obs.shape[0]
    logger.info("Performing solve for tec, const, clock.")
    keys = random.split(random.PRNGKey(int(746583)), T)

    # print(Y_obs.dtype)
    # plt.scatter(Y_obs[220,:24], Y_obs[220,24:])
    # plt.show()
    # plt.plot(freqs, amp[220,:])
    # plt.show()
    # get_error_data(Y_obs, amp, freqs)

    # from jax import disable_jit
    # with disable_jit():
    #     unconstrained_solve(freqs,
    #                         keys[1813690],
    #                         Y_obs[1813690],
    #                         amp[1813690]
    #                         )

    tec_est, tec_std, const_est, clock_est = chunked_pmap(lambda *args: unconstrained_solve(freqs, *args),
                                                          keys,
                                                          Y_obs,
                                                          amp,
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
    tec_mean = tec_est.reshape((Nd, Na, Nt))
    tec_std = tec_std.reshape((Nd, Na, Nt))

    tec_mean, tec_std, outliers = detect_outliers(tec_mean, tec_std, times)
    tec_mean = tec_mean.reshape((Nd * Na * Nt,))
    tec_std = tec_std.reshape((Nd * Na * Nt,))

    # tec_mean = tec_mean.reshape((Nd, Na, Nt))
    # tec_std = tec_std.reshape((Nd, Na, Nt))
    # const_mean = const_mean.reshape((Nd, Na, Nt))
    # clock_mean = clock_mean.reshape((Nd, Na, Nt))
    # amp = amp.reshape((Nd, Na, Nt, Nf)).transpose((0, 1, 3, 2))
    # phase_mean = tec_mean[..., None, :] * (TEC_CONV / freqs[:, None]) \
    #              + const_mean[..., None, :] \
    #              + clock_mean[..., None, :] * 1e-9 * (2. * jnp.pi * freqs[:, None])
    # return phase_mean, amp, tec_mean, tec_std, const_mean, clock_mean

    tec_std = 20. * jnp.ones_like(tec_std)

    logger.info("Performing tec-only solve, with fixed const and clock.")
    # (nan_res0, nan_res1) = chunked_pmap(lambda *args: debug_constrained_solve(freqs, *args),
    #                                     random.split(random.PRNGKey(int(default_timer())), T), Y_obs,
    #                                     amp, tec_mean, tec_std, const_mean, clock_mean)
    # logger.info("Nan prior test {} {}".format(jnp.any(nan_res1), jnp.any(nan_res0)))
    (tec_mean, tec_std) = \
        chunked_pmap(lambda *args: constrained_solve(freqs, *args),
                     random.split(random.PRNGKey(int(default_timer())), T), Y_obs,
                     amp, tec_mean, tec_std, const_mean, clock_mean,
                     debug_mode=False)

    tec_mean = tec_mean.reshape((Nd, Na, Nt))
    tec_std = tec_std.reshape((Nd, Na, Nt))
    const_mean = const_mean.reshape((Nd, Na, Nt))
    clock_mean = clock_mean.reshape((Nd, Na, Nt))
    tec_est = tec_est.reshape((Nd, Na, Nt))
    const_est = const_est.reshape((Nd, Na, Nt))
    clock_est = clock_est.reshape((Nd, Na, Nt))
    # compute phase before outlier suppression as the smoothed solutions are important.
    amp = amp.reshape((Nd, Na, Nt, Nf)).transpose((0, 1, 3, 2))
    phase_mean = tec_mean[..., None, :] * (TEC_CONV / freqs[:, None]) \
                 + const_mean[..., None, :] \
                 + clock_mean[..., None, :] * 1e-9 * (2. * jnp.pi * freqs[:, None])

    logger.info("Performing outlier detection.")
    tec_mean, tec_std, outliers = detect_outliers(tec_mean, tec_std, times)

    return phase_mean, amp, tec_mean, tec_std, const_mean, clock_mean, tec_est, const_est, clock_est, outliers


def link_overwrite(src, dst):
    if os.path.islink(dst):
        logger.info("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
    logger.info("Linking {} -> {}".format(src, dst))
    os.symlink(src, dst)


def get_data(solution_file):
    logger.info("Getting DDS4 data.")
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))#, ant=slice(54, None), dir=44, time=slice(0, 600))
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
        Y_obs = jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)], axis=2)
    return Y_obs, times, freqs


def main(data_dir, working_dir, obs_num, ncpu):
    os.environ['XLA_FLAGS'] = f"--xla_force_host_platform_device_count={ncpu}"
    logger.info("Performing data smoothing via tec+const+clock inference.")
    dds4_h5parm = os.path.join(data_dir, 'L{}_DDS4_full_merged.h5'.format(obs_num))
    dds5_h5parm = os.path.join(working_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    linked_dds5_h5parm = os.path.join(data_dir, 'L{}_DDS5_full_merged.h5'.format(obs_num))
    logger.info("Looking for {}".format(dds4_h5parm))
    link_overwrite(dds5_h5parm, linked_dds5_h5parm)
    prepare_soltabs(dds4_h5parm, dds5_h5parm)
    Y_obs, times, freqs = get_data(solution_file=dds4_h5parm)
    phase_mean, amp_mean, tec_mean, tec_std, const_mean, clock_mean, tec_est, const_est, clock_est, outliers = \
        solve_and_smooth(Y_obs, times, freqs)
    logger.info("Storing smoothed phase, amplitudes, tec, const, and clock")
    with DataPack(dds5_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        h.select(pol=slice(0, 1, 1))#, ant=slice(54, None), dir=44, time=slice(0, 600))
        h.phase = np.asarray(phase_mean)[None, ...]
        h.amplitude = np.asarray(amp_mean)[None, ...]
        h.tec = np.asarray(tec_mean)[None, ...]
        h.tec_outliers = np.asarray(outliers)[None, ...]
        h.weights_tec = np.asarray(tec_std)[None, ...]
        h.const = np.asarray(const_mean)[None, ...]
        h.clock = np.asarray(clock_mean)[None, ...]
        axes = h.axes_phase
        patch_names, _ = h.get_directions(axes['dir'])
        antenna_labels, _ = h.get_antennas(axes['ant'])

    Y_mean = jnp.concatenate([amp_mean * jnp.cos(phase_mean), amp_mean * jnp.sin(phase_mean)], axis=-2)

    logger.info("Plotting results.")
    data_plot_dir = os.path.join(working_dir, 'data_plots')
    os.makedirs(data_plot_dir, exist_ok=True)
    Nd, Na, Nf, Nt = phase_mean.shape

    for ia in range(Na):
        for id in range(Nd):
            fig, axs = plt.subplots(4, 1, sharex=True)
            from bayes_gain_screens.utils import windowed_mean
            # smooth_tec = windowed_mean(tec_mean[id, ia, :], 15)
            axs[0].plot(times, tec_mean[id, ia, :], c='black', label='tec')
            ylim=axs[0].get_ylim()
            axs[0].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
            axs[0].plot(times, tec_est[id, ia, :], c='green', ls='dotted', label='tec*')
            axs[0].set_ylim(*ylim)

            axs[1].plot(times, const_mean[id, ia, :], c='black', label='const')
            ylim = axs[1].get_ylim()
            axs[1].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
            axs[1].plot(times, const_est[id, ia, :], c='green', ls='dotted', label='const*')
            axs[1].set_ylim(*ylim)

            axs[2].plot(times, clock_mean[id, ia, :], c='black', label='clock')
            ylim = axs[2].get_ylim()
            axs[2].vlines(times[outliers[id, ia, :]], *ylim, colors='red', label='outliers', alpha=0.5)
            axs[2].plot(times, clock_est[id, ia, :], c='green', ls='dotted', label='clock*')
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
    main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 342938, 8)
    # main('/home/albert/data/gains_screen/data', '/home/albert/data/gains_screen/working_dir/', 100000, 1)


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
