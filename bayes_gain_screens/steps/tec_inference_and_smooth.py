import os
import sys
import numpy as np
import pylab as plt
import argparse
from timeit import default_timer
from jax import numpy as jnp, jit, random, vmap
from jax.lax import while_loop, scan
from jax.scipy.special import erf
import logging
import astropy.units as au

from bayes_gain_screens.utils import chunked_pmap, inverse_update, poly_smooth, wrap, windowed_mean

logger = logging.getLogger(__name__)

from bayes_gain_screens.plotting import animate_datapack, add_colorbar_to_axes

from h5parm import DataPack
from h5parm.utils import make_soltab

from jaxns.prior_transforms import UniformPrior, PriorChain, HalfLaplacePrior, DeterministicTransformPrior, NormalPrior
from jaxns.nested_sampling import NestedSampler
from jaxns.gaussian_process.kernels import RBF


def prepare_soltabs(dds4_h5parm, dds5_h5parm):
    logger.info("Creating sol000/phase000+amplitude000+tec000+const000+clock000")
    make_soltab(dds4_h5parm, from_solset='sol000', to_solset='sol000', from_soltab='phase000',
                to_soltab=['phase000', 'amplitude000', 'tec000', 'const000', 'clock000'], remake_solset=True,
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

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       uncert_mean=lambda uncert, **kwargs: uncert,
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2,
                       const_mean=lambda const, **kwargs: jnp.concatenate([jnp.cos(const), jnp.sin(const)]),
                       clock_mean=lambda clock, **kwargs: clock,
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 collect_samples=False,
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=1, num_slices=3))
    const_mean = jnp.arctan2(results.marginalised['const_mean'][1], results.marginalised['const_mean'][0])
    clock_mean = results.marginalised['clock_mean']
    tec_mean = results.marginalised['tec_mean']
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'] - tec_mean ** 2)
    return (tec_mean, tec_std, const_mean, clock_mean)


def debug_constrained_solve(freqs, key, Y_obs, amp, tec_mean, tec_std, const_mu, clock_mu):
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    print(Y_obs, amp, tec_mean, tec_std, const_mu, clock_mu)

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
    TEC_CONV = -8.4479745e6  # mTECU/Hz
    print(Y_obs, amp, tec_mean, tec_std, const_mu, clock_mu)

    def log_likelihood(Y, uncert, **kwargs):
        return jnp.sum(log_laplace(Y, Y_obs, uncert))

    def Y_transform(tec):
        phase = tec * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)
        return jnp.concatenate([amp * jnp.cos(phase), amp * jnp.sin(phase)])

    tec = NormalPrior('tec', tec_mean, tec_std)
    prior_chain = PriorChain() \
        .push(HalfLaplacePrior('uncert', 0.2)) \
        .push(DeterministicTransformPrior('Y', Y_transform, (freqs.size * 2,), tec))

    ns = NestedSampler(log_likelihood, prior_chain,
                       sampler_name='slice',
                       tec_mean=lambda tec, **kwargs: tec,
                       tec2_mean=lambda tec, **kwargs: tec ** 2
                       )

    results = ns(key=key,
                 num_live_points=100,
                 max_samples=1e5,
                 collect_samples=False,
                 termination_frac=0.01,
                 sampler_kwargs=dict(depth=2, num_slices=3))

    tec_mean = results.marginalised['tec_mean']
    tec_std = jnp.sqrt(results.marginalised['tec2_mean'] - tec_mean ** 2)
    phase_mean = tec_mean * (TEC_CONV / freqs) + const_mu + clock_mu * 1e-9 * (2. * jnp.pi * freqs)
    return (tec_mean, tec_std, phase_mean)


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
        kappa: z-score limit of outlier definition.

    Returns:

    """
    Y_smooth = windowed_mean(Y_obs, 10)
    dY = Y_obs - Y_smooth
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


def detect_outliers(tec_mean, tec_std, times):
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
    Nd, Na, Nt = tec_mean.shape
    tec_mean = tec_mean.reshape((Nd * Na, Nt))
    tec_std = tec_std.reshape((Nd * Na, Nt))
    mu_star, sigma_star, outliers = chunked_pmap(
        lambda tec_mean, tec_std: single_detect_outliers(tec_std, tec_mean, times), tec_mean, tec_std,
        chunksize=1)
    return mu_star.reshape((Nd, Na, Nt)), sigma_star.reshape((Nd, Na, Nt)), outliers.reshape((Nd, Na, Nt))


def solve_and_smooth(Y_obs, times, freqs):
    Nd, Na, twoNf, Nt = Y_obs.shape
    Nf = twoNf // 2
    Y_obs = Y_obs.transpose((0, 1, 3, 2)).reshape((Nd * Na * Nt, 2 * Nf))  # Nd*Na*Nt, 2*Nf
    amp = jnp.sqrt(Y_obs[:, :freqs.size] ** 2 + Y_obs[:, freqs.size:] ** 2)
    logger.info("Min/max amp: {} {}".format(jnp.min(amp), jnp.max(amp)))
    logger.info("Number of nan: {}".format(jnp.sum(jnp.isnan(Y_obs))))
    logger.info("Number of inf: {}".format(jnp.sum(jnp.isinf(Y_obs))))
    T = Y_obs.shape[0]
    logger.info("Performing solve for tec, const, clock.")
    tec_mean, tec_std, const_mean, clock_mean = chunked_pmap(lambda *args: unconstrained_solve(freqs, *args),
                                                             random.split(random.PRNGKey(int(746583)), T),
                                                             Y_obs, amp)

    def smooth(y):
        y = y.reshape((Nd * Na, Nt))  # Nd*Na,Nt
        y = chunked_pmap(lambda y: poly_smooth(times, y, deg=3), y).reshape(
            (Nd * Na * Nt,))  # Nd*Na*Nt
        return y

    logger.info("Smoothing const and clock (a strong prior).")
    # Nd*Na*Nt
    clock_mean = smooth(clock_mean)
    const_mean = smooth(const_mean)
    # outlier flagging on tec, to set better priors for constrained solve
    tec_mean = tec_mean.reshape((Nd * Na, Nt))
    tec_std = tec_std.reshape((Nd * Na, Nt))
    logger.info("Performing outlier detection.")
    tec_mean, tec_std, outliers = detect_outliers(tec_mean.reshape((Nd, Na, Nt)), tec_std.reshape((Nd, Na, Nt)), times)
    tec_mean = tec_mean.reshape((Nd * Na * Nt,))
    tec_std = tec_std.reshape((Nd * Na * Nt,))
    outliers = outliers.reshape((Nd * Na * Nt,))
    tec_std = 10. * jnp.ones_like(tec_std)

    logger.info("Performing tec-only solve, with fixed const and clock.")
    (nan_res0, nan_res1) = chunked_pmap(lambda *args: debug_constrained_solve(freqs, *args),
                                        random.split(random.PRNGKey(int(default_timer())), T), Y_obs,
                                        amp, tec_mean, tec_std, const_mean, clock_mean)
    logger.info("Nan prior test {} {}".format(jnp.any(nan_res1), jnp.any(nan_res0)))
    (tec_mean, tec_std, phase_mean) = \
        chunked_pmap(lambda *args: constrained_solve(freqs, *args),
                     random.split(random.PRNGKey(int(default_timer())), T), Y_obs,
                     amp, tec_mean, tec_std, const_mean, clock_mean)
    phase_mean = phase_mean.reshape((Nd, Na, Nt, Nf)).transpose((0, 1, 3, 2))
    amp = amp.reshape((Nd, Na, Nt, Nf)).transpose((0, 1, 3, 2))
    tec_mean = tec_mean.reshape((Nd, Na, Nt))
    tec_std = tec_std.reshape((Nd, Na, Nt))
    const_mean = const_mean.reshape((Nd, Na, Nt))
    clock_mean = clock_mean.reshape((Nd, Na, Nt))
    return phase_mean, amp, tec_mean, tec_std, const_mean, clock_mean


def link_overwrite(src, dst):
    if os.path.islink(dst):
        print("Unlinking pre-existing sym link {}".format(dst))
        os.unlink(dst)
    print("Linking {} -> {}".format(src, dst))
    os.symlink(src, dst)


def get_data(solution_file):
    logger.info("Getting DDS4 data.")
    with DataPack(solution_file, readonly=True) as h:
        select = dict(pol=slice(0, 1, 1))  # , ant=slice(54, None), dir=16, time=slice(0, 600))
        h.select(**select)
        phase, axes = h.phase
        phase = phase[0, ...]
        amp, axes = h.amplitude
        amp = amp[0, ...]
        _, freqs = h.get_freqs(axes['freq'])
        freqs = freqs.to(au.Hz).value
        _, times = h.get_times(axes['time'])
        times = times.mjd / 86400.
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
    phase_mean, amp_mean, tec_mean, tec_std, const_mean, clock_mean = solve_and_smooth(Y_obs, times, freqs)
    logger.info("Storing smoothed phase, amplitudes, tec, const, and clock")
    with DataPack(dds5_h5parm, readonly=False) as h:
        h.current_solset = 'sol000'
        h.select(pol=slice(0, 1, 1))  # , ant=slice(54, None), dir=16, time=slice(0, 600))
        h.phase = np.asarray(phase_mean)[None, ...]
        h.amplitude = np.asarray(amp_mean)[None, ...]
        h.tec = np.asarray(tec_mean)[None, ...]
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
            fig, axs = plt.subplots(3, 1, sharex=True)

            axs[0].plot(times, tec_mean[id, ia, :], c='black', label='tec')
            axs[0].plot(times, tec_mean[id, ia, :] + tec_std[id, ia, :], ls='dotted', c='black')
            axs[0].plot(times, tec_mean[id, ia, :] - tec_std[id, ia, :], ls='dotted', c='black')

            axs[1].plot(times, const_mean[id, ia, :], c='black', label='const')
            axs[2].plot(times, clock_mean[id, ia, :], c='black', label='clock')

            axs[0].legend()
            axs[1].legend()
            axs[2].legend()

            axs[0].set_ylabel("DTEC [mTECU]")
            axs[1].set_ylabel("phase [rad]")
            axs[2].set_ylabel("delay [ns]")
            axs[2].set_xlabel("time [s]")

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

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'tec_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='tec', vmin=-60., vmax=60., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'const_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='const', vmin=-np.pi, vmax=np.pi, plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=True,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'clock_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='clock', vmin=-1., vmax=1., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'tec_uncert_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='weights_tec', vmin=0., vmax=10., plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False,
                     flag_outliers=False)

    animate_datapack(dds5_h5parm, os.path.join(working_dir, 'smoothed_amp_plots'), num_processes=ncpu,
                     solset='sol000',
                     observable='amplitude', plot_facet_idx=True,
                     labels_in_radec=True, plot_crosses=False, phase_wrap=False)


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
